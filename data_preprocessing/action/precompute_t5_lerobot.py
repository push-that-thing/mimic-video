"""Pre-compute T5-XXL language embeddings for a LeRobot dataset.

Run this once before training so the T5 model does not need to be loaded at
dataset init time.  The output is a .npz file saved in the location that
LeRobotDataset (mimic-video's adapter) discovers automatically via its cache
lookup, so no extra configuration is required after running this script.

About --repo-id
---------------
In LeRobot, every dataset has a repo_id of the form "username/dataset-name".
For a locally recorded dataset this is whatever name you chose when recording,
e.g. "myorg/myrobot-task".  It does NOT need to exist on HuggingFace.
When combined with --root, lerobot looks for the data at:
  {root}/{username}/{dataset-name}/

Usage — local dataset (no internet required)
---------------------------------------------
python precompute_t5_lerobot.py \
    --repo-id username/dataset-name \
    --root /path/to/datasets \
    --t5-ckpt /path/to/cosmos_t5_model \
    --local-only

Usage — dataset on HuggingFace Hub
------------------------------------
python precompute_t5_lerobot.py \
    --repo-id username/dataset-name \
    --t5-ckpt /path/to/cosmos_t5_model

Output location
---------------
  If --root is given : {root}/.t5_embeddings/{hash}.npz
  Otherwise          : ~/.cache/mimic_lerobot_t5/.t5_embeddings/{hash}.npz

The hash is derived from the sorted list of unique task strings in the dataset,
so the same cache file is shared across train and val splits.
"""

import argparse
import hashlib
import os
import pathlib
import sys

import numpy as np
import tqdm


def _import_lerobot():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except ImportError:
        lerobot_src = pathlib.Path(__file__).parents[3] / "lerobot" / "src"
        if lerobot_src.exists() and str(lerobot_src) not in sys.path:
            sys.path.insert(0, str(lerobot_src))
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset


def main():
    p = argparse.ArgumentParser(
        description="Pre-compute Cosmos T5-XXL embeddings for a LeRobot dataset."
    )
    p.add_argument(
        "--repo-id", required=True,
        help="HuggingFace repo id, e.g. 'myorg/myrobot-task'",
    )
    p.add_argument(
        "--root", default=None,
        help="Local dataset root directory (optional; uses HF cache if omitted)",
    )
    p.add_argument(
        "--t5-ckpt", required=True,
        help="Path to the Cosmos T5-XXL checkpoint directory",
    )
    p.add_argument(
        "--local-only", action="store_true",
        help="Block all HuggingFace network calls — use when the dataset is fully local",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite the cache file if it already exists",
    )
    args = p.parse_args()

    if args.local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    LeRobotDataset = _import_lerobot()

    # Load dataset metadata to discover unique task strings.
    print(f"Loading dataset metadata for '{args.repo_id}' ...")
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    task_strings = list(dataset.meta.tasks.index)

    print(f"\nFound {len(task_strings)} unique task(s):")
    for t in task_strings:
        print(f"  • {t!r}")

    # Cache path — must match the logic in dataset_lerobot.LeRobotDataset.__init__
    cache_dir = (
        pathlib.Path(args.root) if args.root
        else pathlib.Path.home() / ".cache" / "mimic_lerobot_t5"
    )
    cache_key = hashlib.sha256(str(sorted(task_strings)).encode()).hexdigest()
    out_path = cache_dir / ".t5_embeddings" / f"{cache_key}.npz"

    if out_path.exists() and not args.force:
        print(f"\nCache already exists at:\n  {out_path}")
        print("Pass --force to overwrite.")
        return

    # Load T5 encoder.
    print("\nLoading T5 encoder (this may take a moment) ...")
    from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig

    encoder = CosmosT5TextEncoder(CosmosT5TextEncoderConfig(ckpt_path=args.t5_ckpt))

    # Encode every task string.
    embeddings = {}
    for task_str in tqdm.tqdm(task_strings, desc="Encoding tasks"):
        emb = encoder.encode_prompts(task_str, max_length=512, return_mask=False)
        embeddings[task_str] = emb.cpu().numpy().astype(np.float16)  # (1, 512, 1024)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **embeddings)

    print(f"\nSaved to:\n  {out_path}")
    print("\nThe LeRobotDataset adapter will find this cache automatically.")
    print("Or pass it explicitly via:\n  language_embeddings_path='" + str(out_path) + "'")


if __name__ == "__main__":
    main()