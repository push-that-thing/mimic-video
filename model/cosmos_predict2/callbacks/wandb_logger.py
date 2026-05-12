import os
import numbers
from collections.abc import Mapping

import torch
import wandb

from imaginaire.utils import distributed
from imaginaire.utils.callback import Callback


class WandBLogger(Callback):
    def __init__(self, mode: str | None = None):
        self.mode = mode

    @distributed.rank0_only
    def on_train_start(self, model, iteration: int = 0) -> None:
        del model
        if wandb.run is not None:
            return

        wandb.init(
            project=self.config.job.project,
            group=self.config.job.group,
            name=self.config.job.name,
            config=self._run_config(),
            mode=self.mode,
            resume="allow",
        )

    @distributed.rank0_only
    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        del model, data_batch
        metrics = {"train/loss": loss.detach().float().item()}
        metrics.update(self._flatten_scalars(output_batch, prefix="train"))
        wandb.log(metrics, step=iteration)

    @distributed.rank0_only
    def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        del model, data_batch
        metrics = {"val/loss": loss.detach().float().item()}
        metrics.update(self._flatten_scalars(output_batch, prefix="val"))
        wandb.log(metrics, step=iteration)

    @distributed.rank0_only
    def on_app_end(self) -> None:
        if wandb.run is not None:
            wandb.finish()

    def _flatten_scalars(self, values: Mapping, prefix: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[f"{prefix}/{key}"] = value.detach().float().item()
            elif isinstance(value, numbers.Number):
                metrics[f"{prefix}/{key}"] = float(value)
        return metrics

    def _run_config(self) -> dict[str, object]:
        train_bsz = self._get("dataloader_train", "batch_size")
        grad_accum = self._get("trainer", "grad_accum_iter", default=1)
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        effective_bsz = None
        if isinstance(train_bsz, numbers.Integral) and isinstance(grad_accum, numbers.Integral):
            effective_bsz = int(train_bsz) * int(grad_accum) * world_size

        return self._jsonable(
            {
                "job": {
                    "project": self._get("job", "project"),
                    "group": self._get("job", "group"),
                    "name": self._get("job", "name"),
                },
                "data": {
                    "repo_id": self._get("dataloader_train", "dataset", "repo_id"),
                    "image_key": self._get("dataloader_train", "dataset", "image_key"),
                    "state_key": self._get("dataloader_train", "dataset", "state_key"),
                    "action_key": self._get("dataloader_train", "dataset", "action_key"),
                    "obs_image_horizon": self._get("dataloader_train", "dataset", "obs_image_horizon"),
                    "action_image_horizon": self._get("dataloader_train", "dataset", "action_image_horizon"),
                    "action_lowdim_horizon": self._get("dataloader_train", "dataset", "action_lowdim_horizon"),
                    "target_fps": self._get("dataloader_train", "dataset", "target_fps"),
                    "lowdim_target_fps": self._get("dataloader_train", "dataset", "lowdim_target_fps"),
                    "action_shift_s": self._get("dataloader_train", "dataset", "action_shift_s"),
                    "image_resize": self._get("dataloader_train", "dataset", "image_resize"),
                    "video_backend": self._get("dataloader_train", "dataset", "video_backend"),
                },
                "training": {
                    "micro_bsz": train_bsz,
                    "grad_accum_iter": grad_accum,
                    "world_size": world_size,
                    "effective_bsz": effective_bsz,
                    "max_iter": self._get("trainer", "max_iter"),
                    "logging_iter": self._get("trainer", "logging_iter"),
                    "validation_iter": self._get("trainer", "validation_iter"),
                    "run_validation": self._get("trainer", "run_validation"),
                    "checkpoint_save_iter": self._get("checkpoint", "save_iter"),
                    "lr": self._get("optimizer", "lr"),
                    "loss_scale": self._get("model", "config", "loss_scale"),
                    "loss_reduce": self._get("model", "config", "loss_reduce"),
                },
                "video_backbone": {
                    "checkpoint": self._get("model", "config", "video_dit_path"),
                    "xattn_layer_idx": self._get("model", "config", "pipe_config", "xattn_layer_idx"),
                },
                "action_decoder": {
                    "max_horizon": self._get("world2action_pipe", "net", "max_horizon"),
                    "in_channels": self._get("world2action_pipe", "net", "in_channels"),
                    "out_channels": self._get("world2action_pipe", "net", "out_channels"),
                    "model_channels": self._get("world2action_pipe", "net", "model_channels"),
                    "num_blocks": self._get("world2action_pipe", "net", "num_blocks"),
                    "num_heads": self._get("world2action_pipe", "net", "num_heads"),
                    "mlp_ratio": self._get("world2action_pipe", "net", "mlp_ratio"),
                    "atten_backend": self._get("world2action_pipe", "net", "atten_backend"),
                    "crossattn_emb_channels": self._get("world2action_pipe", "net", "crossattn_emb_channels"),
                    "use_adaln_lora": self._get("world2action_pipe", "net", "use_adaln_lora"),
                    "adaln_lora_dim": self._get("world2action_pipe", "net", "adaln_lora_dim"),
                    "pair_timestep_feature_rank": self._get(
                        "world2action_pipe", "net", "pair_timestep_feature_rank"
                    ),
                    "sac_config": self._get("world2action_pipe", "net", "sac_config", "mode"),
                },
            }
        )

    def _get(self, *path: str, default=None):
        value = self.config
        for key in path:
            try:
                if isinstance(value, Mapping):
                    value = value[key]
                else:
                    value = getattr(value, key)
            except (AttributeError, KeyError, TypeError):
                return default
        return value

    def _jsonable(self, value):
        if isinstance(value, Mapping):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v) for v in value]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, numbers.Number) or isinstance(value, str) or value is None or isinstance(value, bool):
            return value
        return str(value)
