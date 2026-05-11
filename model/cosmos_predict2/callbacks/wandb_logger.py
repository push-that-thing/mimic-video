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
