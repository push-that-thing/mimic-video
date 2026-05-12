"""Offline held-out evaluation for an SO-101 action decoder checkpoint.

This script evaluates a saved world-to-action checkpoint against held-out
LeRobot samples. It uses the ground-truth future video to obtain the same
video-backbone hidden states used during training, then compares predicted
action chunks to recorded actions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override


DEFAULT_EXPERIMENT = "w2a_so101_lerobot_v2w_push_that_thing_lr1.000e-04_layer20_bsz1"
DEFAULT_JOB_NAME = "w2a_so101_a100_normfix_bsz2_accum8"


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    parser.add_argument("--iteration", type=int, default=250)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--stop-steps", default="20")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--video-sampling-steps", type=int, default=35)
    parser.add_argument("--output-dir", type=Path, default=Path("eval/action_decoder"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vam"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def model_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint.expanduser().resolve()
    ckpt = (
        model_dir()
        / "checkpoints"
        / "vam"
        / "so101"
        / args.job_name
        / "checkpoints"
        / "model"
        / f"iter_{args.iteration:09d}.pt"
    )
    return ckpt.resolve()


def to_cuda_batch(sample: dict[str, np.ndarray | torch.Tensor]) -> dict[str, torch.Tensor]:
    batch = {}
    for key, value in sample.items():
        batch[key] = torch.as_tensor(value).unsqueeze(0).cuda(non_blocking=True)
    batch["obs/language_embedding"] = batch["obs/language_embedding"].squeeze(1)
    return batch


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    count = min(count, length)
    if count <= 0:
        return []
    return sorted(set(np.linspace(0, length - 1, num=count, dtype=np.int64).tolist()))


@torch.inference_mode()
def predict_from_gt_video(
    video_pipe: Video2WorldPipeline,
    action_pipe: World2ActionPipeline,
    batch: dict[str, torch.Tensor],
    stop_step: int,
    action_seed: int,
    video_seed: int,
) -> torch.Tensor:
    _, video_B_C_T_H_W, condition = video_pipe.get_mimic_data_and_condition(batch)

    torch.manual_seed(video_seed)
    video_epsilon_B_C_T_H_W = torch.randn(video_B_C_T_H_W.size(), dtype=torch.bfloat16, device="cuda")
    B = video_B_C_T_H_W.shape[0]
    video_sigma_B_1 = video_pipe.scheduler.sigmas[stop_step].repeat(B).unsqueeze(1)

    world_pred = video_pipe.denoise(
        video_B_C_T_H_W + video_epsilon_B_C_T_H_W * rearrange(video_sigma_B_1, "b t -> b 1 t 1 1"),
        video_sigma_B_1,
        condition,
        use_cuda_graphs=False,
        return_only_hidden_states_up_to=action_pipe.config.xattn_layer_idx,
        return_decoded_video=False,
    )
    crossattn_emb = world_pred.hidden_states[action_pipe.config.xattn_layer_idx]
    crossattn_emb = crossattn_emb.reshape(crossattn_emb.shape[0], -1, crossattn_emb.shape[-1])

    return action_pipe(
        state_B_HO_O=batch["obs/lowdim_concat"],
        crossattn_emb=crossattn_emb,
        context_timesteps_B_1=video_sigma_B_1,
        seed=action_seed,
        use_cuda_graphs=False,
    )


def flatten_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    out = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row]
        out[f"{key}/mean"] = float(np.mean(vals))
        out[f"{key}/std"] = float(np.std(vals))
    return out


def save_plot(path: Path, pred: np.ndarray, target: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    x = np.arange(target.shape[0])
    for dim, ax in enumerate(axes.flat):
        ax.plot(x, target[:, dim], label="target", linewidth=2.0)
        ax.plot(x, pred[:, dim], label="pred", linewidth=1.5)
        ax.set_title(f"dim {dim}")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def maybe_log_wandb(args: argparse.Namespace, metrics: dict[str, float], output_dir: Path, plots: list[Path]) -> None:
    if not args.wandb:
        return
    import wandb

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        job_type="checkpoint_eval",
        name=f"{args.job_name}_iter{args.iteration:09d}_eval",
        config=vars(args),
    )
    log_payload = dict(metrics)
    for plot_path in plots[:8]:
        log_payload[f"plot/{plot_path.stem}"] = wandb.Image(str(plot_path))
    run.log(log_payload)
    run.save(str(output_dir / "metrics.json"))
    run.finish()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    stop_steps = parse_csv_ints(args.stop_steps)
    seeds = parse_csv_ints(args.seeds)
    if not stop_steps:
        raise ValueError("--stop-steps must contain at least one integer.")
    if not seeds:
        raise ValueError("--seeds must contain at least one integer.")

    output_dir = (args.output_dir / args.job_name / f"iter_{args.iteration:09d}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_config()
    config = override(config, ["--", f"experiment={args.experiment}"])
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    train_dataset = instantiate(config.dataloader_train.dataset)
    val_dataset = instantiate(config.dataloader_val.dataset)
    data_config = instantiate(config.data_config)

    print(f"Loading video backbone: {config.model.config.video_dit_path}")
    video_pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path=config.model.config.video_dit_path,
        use_text_encoder=False,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    video_pipe.scheduler.set_timesteps(args.video_sampling_steps, device="cuda")

    print(f"Loading action checkpoint: {checkpoint}")
    action_pipe = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path=str(checkpoint),
        device="cuda",
        dtype=torch.bfloat16,
    )

    stats = train_dataset.get_statistics()
    action_pipe.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
        concat_groups=data_config.policy_io.concat_groups,
        device="cuda",
        dtype=torch.bfloat16,
    )

    mean_action = torch.as_tensor(stats["action/lowdim_concat"]["mean_clamped"], dtype=torch.float32)
    indices = evenly_spaced_indices(len(val_dataset), args.num_samples)
    print(f"Evaluating {len(indices)} held-out samples from val split of length {len(val_dataset)}")

    rows: list[dict[str, float]] = []
    saved_preds: dict[str, np.ndarray] = {}
    plots: list[Path] = []

    for sample_i, dataset_idx in enumerate(indices):
        sample = val_dataset[dataset_idx]
        batch = to_cuda_batch(sample)
        target = batch["action/lowdim_concat"].float().cpu()
        baseline = mean_action.unsqueeze(0)

        for stop_step in stop_steps:
            if stop_step < 0 or stop_step >= len(video_pipe.scheduler.sigmas):
                raise ValueError(
                    f"stop_step={stop_step} is outside scheduler range 0..{len(video_pipe.scheduler.sigmas) - 1}"
                )
            for seed in seeds:
                pred = predict_from_gt_video(
                    video_pipe,
                    action_pipe,
                    batch,
                    stop_step=stop_step,
                    action_seed=seed,
                    video_seed=seed + 10_000 * (sample_i + 1) + stop_step,
                ).float().cpu()

                mse = F.mse_loss(pred, target).item()
                mae = F.l1_loss(pred, target).item()
                first_action_mse = F.mse_loss(pred[:, :1], target[:, :1]).item()
                first_10_mse = F.mse_loss(pred[:, :10], target[:, :10]).item()
                baseline_mse = F.mse_loss(baseline, target).item()
                pred_std = pred.std(dim=(0, 1)).mean().item()
                target_std = target.std(dim=(0, 1)).mean().item()

                row = {
                    "sample_idx": float(dataset_idx),
                    "stop_step": float(stop_step),
                    "seed": float(seed),
                    "mse": mse,
                    "mae": mae,
                    "first_action_mse": first_action_mse,
                    "first_10_mse": first_10_mse,
                    "train_mean_baseline_mse": baseline_mse,
                    "mse_vs_train_mean_ratio": mse / baseline_mse if baseline_mse > 0 else float("nan"),
                    "pred_dim_std_mean": pred_std,
                    "target_dim_std_mean": target_std,
                }
                for dim in range(target.shape[-1]):
                    row[f"dim_{dim}_mse"] = F.mse_loss(pred[..., dim], target[..., dim]).item()
                rows.append(row)

                stem = f"sample{sample_i:03d}_idx{dataset_idx}_step{stop_step:02d}_seed{seed}"
                saved_preds[f"{stem}_pred"] = pred.numpy()
                saved_preds[f"{stem}_target"] = target.numpy()

                if not args.no_plots and len(plots) < 12:
                    plot_path = output_dir / "plots" / f"{stem}.png"
                    save_plot(
                        plot_path,
                        pred[0].numpy(),
                        target[0].numpy(),
                        title=f"val idx {dataset_idx}, stop {stop_step}, seed {seed}",
                    )
                    plots.append(plot_path)

                print(
                    f"idx={dataset_idx:6d} stop={stop_step:02d} seed={seed:02d} "
                    f"mse={mse:.6f} mae={mae:.6f} baseline={baseline_mse:.6f} "
                    f"ratio={row['mse_vs_train_mean_ratio']:.3f}"
                )

        del batch
        torch.cuda.empty_cache()

    aggregate = flatten_metrics(rows)
    result = {
        "checkpoint": str(checkpoint),
        "experiment": args.experiment,
        "job_name": args.job_name,
        "iteration": args.iteration,
        "num_samples": len(indices),
        "stop_steps": stop_steps,
        "seeds": seeds,
        "aggregate": aggregate,
        "rows": rows,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(output_dir / "predictions.npz", **saved_preds)

    maybe_log_wandb(args, aggregate, output_dir, plots)

    print("\nAggregate metrics:")
    for key, value in aggregate.items():
        if key.endswith("/mean"):
            print(f"  {key}: {value:.6f}")
    print(f"\nSaved metrics: {metrics_path}")
    if plots:
        print(f"Saved plots: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
