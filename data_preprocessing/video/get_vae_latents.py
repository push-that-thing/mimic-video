# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-compute VAE latents for all valid clips in a mimic-video dataset.

The VAE encoder is frozen during fine-tuning, so re-encoding the same video clips
on every training step is pure wasted compute. This script encodes every valid clip
once, saves the result to {dataset_path}/vae_latents/, and the Dataset will load
those files directly instead of running the encoder at training time.

Clip enumeration mirrors the dataset's _get_frames logic: clips are sampled at
5 Hz (step = data_fps / 5) and we enumerate all starting positions where no
boundary clamping occurs.  For episodes that are too short for any such clip we
fall back to a single centered clip.

Storage: each latent is saved as bfloat16 [16, T_lat, H_lat, W_lat].
For the default 480x640 / 61-frame setup that is ~2.3 MB per clip.

Usage (run from mimic-video/model/ with the project venv active):
    python -m data_preprocessing.video.get_vae_latents \\
        --dataset_path /path/to/push-mimic \\
        --vae_pth     /path/to/checkpoints/video_backbone/tokenizer/tokenizer.pth
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from decord import VideoReader, cpu

from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-compute frozen VAE latents for a mimic-video dataset")
    p.add_argument("--dataset_path", required=True, type=pathlib.Path, help="Root of the mimic-video dataset")
    p.add_argument("--vae_pth", required=True, type=str, help="Path to tokenizer.pth checkpoint")
    p.add_argument("--num_frames", type=int, default=61, help="Clip length in frames (must be in {1,5,61})")
    p.add_argument("--video_size", nargs=2, type=int, default=[480, 640], metavar=("H", "W"))
    p.add_argument("--data_fps", type=float, default=30.0, help="Source video FPS")
    p.add_argument("--obs_history", type=int, default=5, help="Number of observation history frames")
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Raw-frame stride between successive clips (default: step = data_fps/5). "
        "Increase to reduce disk usage at the cost of clip diversity.",
    )
    return p.parse_args()


@torch.no_grad()
def encode_clip(
    tokenizer: TokenizerInterface,
    frames_uint8: np.ndarray,
    out_h: int,
    out_w: int,
    device: str,
) -> torch.Tensor:
    """Encode one clip.

    Args:
        frames_uint8: uint8 numpy array [C, T, H_src, W_src]

    Returns:
        Latent tensor [16, T_lat, H_lat, W_lat] in bfloat16 on CPU.
    """
    C, T, H, W = frames_uint8.shape
    x = torch.from_numpy(frames_uint8).float()  # [C, T, H, W]

    if H != out_h or W != out_w:
        x = F.interpolate(
            x.view(C * T, 1, H, W),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).view(C, T, out_h, out_w)
        x = x.round().clamp_(0, 255)

    # Normalize to [-1, 1] and move to device as bfloat16
    x = (x / 127.5 - 1.0).to(device=device, dtype=torch.bfloat16)
    x = x.unsqueeze(0)  # [1, C, T, H, W]

    latent = tokenizer.encode(x)  # [1, 16, T_lat, H_lat, W_lat]
    return latent[0].cpu()  # [16, T_lat, H_lat, W_lat]


def valid_start_positions(n: int, step: int, num_frames: int, obs_history: int, stride: int) -> list[int]:
    """Return all clip start indices (raw frame space) where no boundary clamping occurs."""
    i_min = (obs_history - 1) * step
    i_max = n - (num_frames - obs_history) * step  # exclusive

    if i_min >= i_max:
        # Episode too short for interior clips; fall back to a single centred clip.
        return [n // 2]

    return list(range(i_min, i_max, stride))


def sample_frames(vr: VideoReader, i: int, step: int, num_frames: int, obs_history: int) -> np.ndarray:
    """Sample one clip from a VideoReader, returning uint8 [C, T, H, W]."""
    n = len(vr)
    k = np.arange(num_frames, dtype=np.float64) - (obs_history - 1)
    idx = np.rint(i + k * step).astype(np.int64)
    idx = np.clip(idx, 0, n - 1)

    uniq, inverse = np.unique(idx, return_inverse=True)
    frames = vr.get_batch(uniq).asnumpy()  # [Tv, H, W, C]
    frames = frames[inverse]               # [T, H, W, C]

    # [C, T, H, W]
    return frames.transpose(3, 0, 1, 2)


def main() -> None:
    args = parse_args()

    step = int(round(args.data_fps / 5.0))
    stride = args.stride if args.stride is not None else step
    out_h, out_w = args.video_size

    latent_dir = args.dataset_path / "vae_latents"
    latent_dir.mkdir(exist_ok=True)

    tokenizer = TokenizerInterface(
        chunk_duration=81,
        temporal_window=16,
        vae_pth=args.vae_pth,
    )
    tokenizer.to(device="cuda")

    video_paths = sorted((args.dataset_path / "video").glob("*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No .mp4 files found under {args.dataset_path / 'video'}")

    for video_path in tqdm.tqdm(video_paths, desc="episodes"):
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=0)
        n = len(vr)

        starts = valid_start_positions(n, step, args.num_frames, args.obs_history, stride)
        for i in tqdm.tqdm(starts, desc=video_path.stem, leave=False):
            out_path = latent_dir / f"{video_path.stem}_start{i:06d}.pt"
            if out_path.exists():
                continue

            frames = sample_frames(vr, i, step, args.num_frames, args.obs_history)
            latent = encode_clip(tokenizer, frames, out_h, out_w, device="cuda")
            torch.save(latent, out_path)

        del vr

    print(f"Done. Latents saved to {latent_dir}")


if __name__ == "__main__":
    main()
