"""LeRobot v3 dataset adapter for mimic-video.

Wraps a LeRobot v3 dataset (Parquet + MP4) and produces the batch format
expected by mimic-video's training pipeline:

    obs/workspace_rgb      : float32 (3, T_obs,    H, W)  in [-1, 1]
    obs/lowdim_concat      : float32 (1,  D)
    obs/language_embedding : float32 (1,  512, 1024)
    action/workspace_rgb   : float32 (3, T_act_rgb, H, W) in [-1, 1]
    action/lowdim_concat   : float32 (T_act_ld, D)
"""

import hashlib
import json
import pathlib
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from cosmos_predict2.module.normalizer import array_to_stats


# ---------------------------------------------------------------------------
# Lazy lerobot import (supports vendored copy at ../../../../../../lerobot/src)
# ---------------------------------------------------------------------------

def _import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except ImportError:
        lerobot_src = pathlib.Path(__file__).parents[5] / "lerobot" / "src"
        if lerobot_src.exists() and str(lerobot_src) not in sys.path:
            sys.path.insert(0, str(lerobot_src))
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class LeRobotDataset(torch.utils.data.Dataset):
    """Adapter that wraps a LeRobot v3 dataset for mimic-video training.

    Parameters
    ----------
    repo_id:
        HuggingFace repo id, e.g. ``"myorg/myrobot-task"``.
    root:
        Local directory where the dataset is stored. Pass ``None`` to let
        LeRobot download / cache from the Hub.
    image_key:
        Feature key for the camera images, e.g. ``"observation.images.front"``.
    state_key:
        Feature key for proprioception, e.g. ``"observation.state"``.
    action_key:
        Feature key for actions, e.g. ``"action"``.
    obs_image_horizon:
        Number of observation image frames (looking backwards).
    action_image_horizon:
        Number of action video frames (looking forwards).
    action_lowdim_horizon:
        Number of action lowdim steps (looking forwards).
    target_fps:
        Sampling frequency (Hz) for **images**. Always kept at 5 Hz to match
        the video diffusion model's fixed 56-frame output spec.
    lowdim_target_fps:
        Sampling frequency (Hz) for **lowdim** state and action. Set to the
        robot's native control rate (e.g. 30 Hz for SO-101) for full-resolution
        action prediction. ``action_lowdim_horizon`` must cover the desired time
        window: e.g. 90 steps x (1/30 s) = 3 s.
    action_shift_s:
        How far (seconds) the action window starts after the current timestep.
    image_resize:
        ``(H, W)`` to resize images to.
    language_embeddings_path:
        Path to a ``.npz`` file mapping task strings → ``(1, 512, 1024)``
        float16 arrays. Computed and cached automatically on first run if
        ``t5_encoder_ckpt`` is provided instead.
    t5_encoder_ckpt:
        Path to the Cosmos T5-XXL checkpoint directory. Used only when
        embeddings are not already cached.
    num_val_episodes:
        Number of episodes held out for validation.
    seed:
        RNG seed for train/val episode split.
    train:
        If ``True``, use the training split; otherwise the validation split.
    cache_dir:
        Directory for normalization statistics cache. Defaults to
        ``~/.cache/mimic_lerobot_stats``.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        root: Optional[str] = None,
        image_key: str = "observation.images.front",
        state_key: str = "observation.state",
        action_key: str = "action",
        obs_image_horizon: int = 5,
        action_image_horizon: int = 56,
        action_lowdim_horizon: int = 90,
        target_fps: float = 5.0,
        lowdim_target_fps: float = 30.0,
        action_shift_s: float = 0.2,
        image_resize: tuple[int, int] = (480, 640),
        language_embeddings_path: Optional[str] = None,
        t5_encoder_ckpt: Optional[str] = None,
        num_val_episodes: int = 1,
        seed: int = 42,
        train: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        _LeRobotDataset = _import_lerobot_dataset()

        self._repo_id = repo_id
        self._root = pathlib.Path(root) if root else None
        self._image_key = image_key
        self._state_key = state_key
        self._action_key = action_key
        self._image_resize = image_resize
        self._n_obs_img = obs_image_horizon
        self._n_act_img = action_image_horizon

        # --- Episode train/val split ---
        # Load metadata only (lightweight) to discover episode count and fps.
        _meta = _LeRobotDataset(repo_id, root=root)
        fps = _meta.fps
        all_episodes = list(range(_meta.num_episodes))
        val_set = self._sample_val_episodes(all_episodes, num_val_episodes, seed)
        self._episodes: Optional[list[int]] = (
            [e for e in all_episodes if e not in val_set] if train else sorted(val_set)
        )
        # None means "all" for the lerobot API — keep explicit list either way
        # so get_statistics() uses the same split.

        # --- Delta timestamps (snapped to valid multiples of 1/fps) ---
        # Images run at target_fps (5 Hz); lowdim runs at lowdim_target_fps (30 Hz).
        dt_img_frames = max(1, round(fps / target_fps))
        dt_img_s = dt_img_frames / fps

        dt_ld_frames = max(1, round(fps / lowdim_target_fps))
        dt_ld_s = dt_ld_frames / fps

        shift_frames = max(1, round(action_shift_s * fps))
        shift_s = shift_frames / fps

        # obs images: T frames looking backwards, ending at t=0
        obs_img_deltas = [-(obs_image_horizon - 1 - i) * dt_img_s for i in range(obs_image_horizon)]
        # action images: T frames looking forwards, starting at t=shift_s
        act_img_deltas = [shift_s + i * dt_img_s for i in range(action_image_horizon)]
        # action lowdim: same shift, but at native robot frequency
        act_ld_deltas = [shift_s + i * dt_ld_s for i in range(action_lowdim_horizon)]
        self._act_ld_deltas = act_ld_deltas

        delta_timestamps = {
            image_key: obs_img_deltas + act_img_deltas,  # concatenated, split later in __getitem__
            state_key: [0.0],
            action_key: act_ld_deltas,
        }

        self._inner = _LeRobotDataset(
            repo_id,
            root=root,
            episodes=self._episodes,
            delta_timestamps=delta_timestamps,
        )

        # --- Language embeddings ---
        t5_cache_dir = self._root or (pathlib.Path.home() / ".cache" / "mimic_lerobot_t5")
        self._task_embeddings = self._load_or_compute_embeddings(
            tasks=_meta.meta.tasks,
            language_embeddings_path=language_embeddings_path,
            t5_encoder_ckpt=t5_encoder_ckpt,
            cache_dir=t5_cache_dir,
            repo_id=repo_id,
        )

        # --- Stats cache ---
        self._stats_cache_dir = pathlib.Path(cache_dir) if cache_dir else (
            pathlib.Path.home() / ".cache" / "mimic_lerobot_stats"
        )
        self._stats_id_val = hashlib.sha256(
            str((repo_id, root, obs_image_horizon, action_image_horizon,
                 action_lowdim_horizon, target_fps, lowdim_target_fps,
                 action_shift_s, num_val_episodes, seed, train)).encode()
        ).hexdigest()

    # ------------------------------------------------------------------
    # Core dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict:
        sample = self._inner[idx]

        # Images: (T_obs+T_act_rgb, C, H, W) float32 [0,1]
        imgs_all = sample[self._image_key]
        obs_imgs = self._process_images(imgs_all[: self._n_obs_img])   # (3, T_obs, H, W)
        act_imgs = self._process_images(imgs_all[self._n_obs_img :])   # (3, T_act_rgb, H, W)

        # Lowdim: (1, D) and (T_act_ld, D)
        state = self._to_numpy(sample[self._state_key])    # (1, D)
        action = self._to_numpy(sample[self._action_key])  # (T_act_ld, D)

        # Language embedding: (1, 512, 1024) float32
        lang_emb = self._task_embeddings[sample["task"]].astype(np.float32)

        return {
            "obs/workspace_rgb": obs_imgs,
            "obs/lowdim_concat": state,
            "obs/language_embedding": lang_emb,
            "action/workspace_rgb": act_imgs,
            "action/lowdim_concat": action,
        }

    # ------------------------------------------------------------------
    # Statistics (used by the trainer to build the normalizer)
    # ------------------------------------------------------------------

    @property
    def stats_id(self) -> str:
        return self._stats_id_val

    def get_statistics(self) -> dict:
        cache_path = self._stats_cache_dir / self._stats_id_val
        self._stats_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with cache_path.open() as f:
                raw = json.load(f)
            return {
                k: {sk: np.array(sv, dtype=np.float32) for sk, sv in v.items()}
                for k, v in raw.items()
            }
        except FileNotFoundError:
            pass

        stats = self._compute_statistics()

        with cache_path.open("w") as f:
            json.dump(
                {k: {sk: sv.tolist() for sk, sv in v.items()} for k, v in stats.items()},
                f,
            )
        return stats

    def _compute_statistics(self) -> dict:
        # Use a lightweight dataset (no video) for fast iteration.
        _LeRobotDataset = _import_lerobot_dataset()
        stats_ds = _LeRobotDataset(
            self._repo_id,
            root=str(self._root) if self._root else None,
            episodes=self._episodes,
            delta_timestamps={
                self._state_key: [0.0],
                self._action_key: self._act_ld_deltas,
            },
        )
        states, actions = [], []
        for i in tqdm.trange(len(stats_ds), desc="Computing normalization statistics"):
            sample = stats_ds[i]
            states.append(self._to_numpy(sample[self._state_key]))
            actions.append(self._to_numpy(sample[self._action_key]))

        return {
            "obs/lowdim_concat": array_to_stats(np.stack(states, axis=0).astype(np.float32)),
            "action/lowdim_concat": array_to_stats(np.stack(actions, axis=0).astype(np.float32)),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _process_images(self, imgs: torch.Tensor) -> np.ndarray:
        """(T, C, H, W) float32 [0,1] → (C, T, H_out, W_out) float32 [-1,1]."""
        H_out, W_out = self._image_resize
        if imgs.shape[-2:] != (H_out, W_out):
            imgs = F.interpolate(imgs, size=(H_out, W_out), mode="bilinear", align_corners=False)
        imgs = 2.0 * imgs - 1.0          # [0,1] → [-1,1]
        imgs = imgs.permute(1, 0, 2, 3)  # (T, C, H, W) → (C, T, H, W)
        return imgs.numpy()

    @staticmethod
    def _to_numpy(t) -> np.ndarray:
        if isinstance(t, torch.Tensor):
            return t.numpy().astype(np.float32)
        return np.asarray(t, dtype=np.float32)

    @staticmethod
    def _sample_val_episodes(
        all_episodes: list[int], num_val_episodes: int, seed: int
    ) -> set[int]:
        if num_val_episodes <= 0:
            return set()
        rng = np.random.default_rng(seed=seed)
        n = min(num_val_episodes, len(all_episodes))
        idxs = rng.choice(len(all_episodes), size=n, replace=False)
        return {all_episodes[int(i)] for i in idxs}

    @staticmethod
    def _load_or_compute_embeddings(
        tasks,  # pandas DataFrame: task_str → task_index
        language_embeddings_path: Optional[str],
        t5_encoder_ckpt: Optional[str],
        cache_dir: pathlib.Path,
        repo_id: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        task_strings = list(tasks.index)

        # 1. Explicit path provided by the user
        if language_embeddings_path is not None:
            data = np.load(language_embeddings_path, allow_pickle=False)
            return {t: data[t] for t in task_strings}

        # 2. Auto-cache keyed by task strings
        cache_key = hashlib.sha256(str(sorted(task_strings)).encode()).hexdigest()
        auto_cache = cache_dir / ".t5_embeddings" / f"{cache_key}.npz"
        if auto_cache.exists():
            data = np.load(auto_cache, allow_pickle=False)
            return {t: data[t] for t in task_strings}

        # 3. Try to fetch from the HuggingFace dataset repo
        if repo_id is not None:
            try:
                from huggingface_hub import hf_hub_download
                remote_path = f".t5_embeddings/{cache_key}.npz"
                print(f"Downloading T5 embeddings from {repo_id} ({remote_path}) ...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                    local_dir=str(cache_dir),
                )
                data = np.load(auto_cache, allow_pickle=False)
                return {t: data[t] for t in task_strings}
            except Exception as e:
                print(f"Could not fetch T5 embeddings from HuggingFace ({e}); falling back.")

        # 4. Compute with T5 encoder
        if t5_encoder_ckpt is None:
            raise ValueError(
                "T5 language embeddings are required but not found.\n"
                "Options:\n"
                "  (a) Pass language_embeddings_path=<path.npz> with pre-computed embeddings.\n"
                "  (b) Pass t5_encoder_ckpt=<ckpt_dir> to compute them now (requires GPU).\n"
                "  (c) Run data_preprocessing/action/precompute_t5_lerobot.py first.\n"
                "  (d) Push embeddings to the HuggingFace dataset repo so they are fetched automatically."
            )

        from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig

        encoder = CosmosT5TextEncoder(CosmosT5TextEncoderConfig(ckpt_path=t5_encoder_ckpt))
        embeddings = {}
        for task_str in tqdm.tqdm(task_strings, desc="Computing T5 embeddings"):
            emb = encoder.encode_prompts(task_str, max_length=512, return_mask=False)
            embeddings[task_str] = emb.cpu().numpy().astype(np.float16)  # (1, 512, 1024)

        auto_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(auto_cache, **embeddings)
        return embeddings
