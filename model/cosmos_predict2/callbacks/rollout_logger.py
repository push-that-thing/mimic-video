# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RolloutLogger callback.

Every ``rollout_iter`` training steps, generate a predicted future video for
each configured test clip and log predicted + ground-truth side-by-side to
wandb. All inference work is wrapped in try/except so any failure
(file-missing, OOM, sampler crash) logs a warning and never interrupts
training. Conditioning frames are deliberately taken from the START of each
test episode (sampled at 5 Hz to match training distribution), not the END,
so successive rollouts at increasing training steps are directly comparable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from decord import VideoReader, cpu

from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import distributed, log


def _load_clip(
    dataset_dir: str,
    ep_idx: int,
    num_video_frames: int,
    target_fps: float = 5.0,
) -> tuple[torch.Tensor, np.ndarray, str]:
    """Read a clip from a mimic-video formatted directory.

    Returns:
        vid_input: (1, 3, num_video_frames, H, W) float32 in [0, 1]. First 5
            entries are the conditioning frames (source frames 0, step, 2*step,
            3*step, 4*step at ``target_fps``-Hz stride); remaining entries are
            the 5th conditioning frame repeated as filler (the model overwrites
            them anyway).
        gt_uint8: (num_video_frames, H, W, 3) uint8 — first ``num_video_frames``
            frames at 5 Hz, for side-by-side ground-truth display.
        prompt: text from the matching metas/ep_*.txt.
    """
    base = Path(dataset_dir)
    video_path = base / "video" / f"ep_{ep_idx:03d}.mp4"
    meta_path = base / "metas" / f"ep_{ep_idx:03d}.txt"
    prompt = meta_path.read_text().strip()

    vr = VideoReader(str(video_path), ctx=cpu(0))
    source_fps = vr.get_avg_fps()
    step = max(1, int(round(source_fps / target_fps)))
    n = len(vr)

    full_indices = [min(i * step, n - 1) for i in range(num_video_frames)]
    full_frames = vr.get_batch(full_indices).asnumpy()  # (T, H, W, 3) uint8

    cond = full_frames[:5]
    pad = np.repeat(cond[-1:], num_video_frames - 5, axis=0)
    vid_np = np.concatenate([cond, pad], axis=0)

    vid_input = torch.from_numpy(vid_np).float().div_(255.0)
    vid_input = vid_input.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # (1, 3, T, H, W)
    return vid_input, full_frames, prompt


def _unwrap(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


class RolloutLogger(EveryN):
    def __init__(
        self,
        test_clips: list[dict] | None = None,
        rollout_iter: int = 500,
        num_sampling_step: int = 25,
        guidance: float = 5.0,
        seed: int = 0,
        num_latent_conditional_frames: int = 2,
        num_video_frames: int = 61,
    ) -> None:
        super().__init__(every_n=rollout_iter, run_at_start=False, barrier_after_run=True)
        self.test_clips = list(test_clips or [])
        self.num_sampling_step = num_sampling_step
        self.guidance = guidance
        self.seed = seed
        self.num_latent_conditional_frames = num_latent_conditional_frames
        self.num_video_frames = num_video_frames
        self._clips: list[tuple[str, torch.Tensor, np.ndarray, str]] | None = None

    def _ensure_clips_loaded(self, device: torch.device) -> None:
        if self._clips is not None:
            return
        loaded: list[tuple[str, torch.Tensor, np.ndarray, str]] = []
        for entry in self.test_clips:
            name = entry.get("name", "clip")
            try:
                vid_input, gt_uint8, prompt = _load_clip(
                    entry["dataset_dir"],
                    int(entry["ep_idx"]),
                    num_video_frames=self.num_video_frames,
                )
                vid_input = vid_input.to(device=device)
                loaded.append((name, vid_input, gt_uint8, prompt))
                log.info(f"RolloutLogger: cached clip '{name}' ({vid_input.shape})")
            except Exception:
                log.warning(f"RolloutLogger: failed to load test clip '{name}'", exc_info=True)
        self._clips = loaded

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        del trainer, data_batch, output_batch, loss
        if not distributed.is_rank0() or wandb.run is None or not self.test_clips:
            return

        try:
            inner = _unwrap(model)
            device = next(inner.parameters()).device
            self._ensure_clips_loaded(device)
            if not self._clips:
                return

            dit = inner.pipe.dit
            was_training = dit.training
            try:
                dit.eval()
                for name, vid_input, gt_uint8, prompt in self._clips:
                    self._run_one_rollout(inner, name, vid_input, gt_uint8, prompt, iteration)
            finally:
                if was_training:
                    dit.train()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.warning(f"RolloutLogger: CUDA OOM at iter {iteration}; rollout skipped")
        except Exception:
            log.warning(f"RolloutLogger: rollout block failed at iter {iteration}", exc_info=True)

    def _run_one_rollout(
        self,
        inner_model: Any,
        name: str,
        vid_input: torch.Tensor,
        gt_uint8: np.ndarray,
        prompt: str,
        iteration: int,
    ) -> None:
        try:
            with torch.no_grad():
                output = inner_model.pipe.generate_video(
                    vid_input=vid_input,
                    num_latent_conditional_frames=self.num_latent_conditional_frames,
                    prompt=prompt,
                    guidance=self.guidance,
                    num_sampling_step=self.num_sampling_step,
                    seed=self.seed,
                )
            pred = output[0].clamp(-1.0, 1.0).add_(1.0).mul_(127.5).clamp_(0, 255).to(torch.uint8)
            pred = pred.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, 3)
            wandb.log(
                {
                    f"rollout/{name}/pred": wandb.Video(pred, fps=5, format="mp4"),
                    f"rollout/{name}/gt": wandb.Video(gt_uint8, fps=5, format="mp4"),
                },
                step=iteration,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.warning(f"RolloutLogger: OOM during clip '{name}' at iter {iteration}; skipped")
        except Exception:
            log.warning(f"RolloutLogger: clip '{name}' failed at iter {iteration}", exc_info=True)
