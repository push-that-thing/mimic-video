from __future__ import annotations

import math

import torch
import wandb

from imaginaire.utils import distributed, log
from imaginaire.utils.callback import Callback


class EarlyStopping(Callback):
    """Stop training when mean validation loss stops improving."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        start_iter: int = 0,
        mode: str = "min",
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}.")
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")
        self.patience = patience
        self.min_delta = min_delta
        self.start_iter = start_iter
        self.mode = mode
        self.best: float | None = None
        self.bad_validations = 0
        self._losses: list[float] = []

    def on_validation_start(self, model, dataloader_val, iteration: int = 0) -> None:
        del model, dataloader_val, iteration
        self._losses = []

    def on_validation_step_end(self, model, data_batch, output_batch, loss: torch.Tensor, iteration: int = 0) -> None:
        del model, data_batch, output_batch, iteration
        if distributed.is_rank0():
            self._losses.append(loss.detach().float().item())

    def on_validation_end(self, model, iteration: int = 0) -> None:
        del model
        if not distributed.is_rank0() or not self._losses:
            return

        current = float(sum(self._losses) / len(self._losses))
        if not math.isfinite(current):
            log.warning(f"EarlyStopping observed non-finite val/loss={current}; counting as no improvement.")
            improved = False
        elif self.best is None:
            improved = True
        elif self.mode == "min":
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta

        if improved:
            self.best = current
            self.bad_validations = 0
            log.info(f"EarlyStopping: val/loss improved to {current:.6f} at iteration {iteration}.")
            self._log_wandb(current, iteration)
            return

        self.bad_validations += 1
        best_display = "None" if self.best is None else f"{self.best:.6f}"
        log.info(
            "EarlyStopping: val/loss did not improve "
            f"({current:.6f}; best={best_display}; "
            f"{self.bad_validations}/{self.patience}) at iteration {iteration}."
        )
        if iteration >= self.start_iter and self.bad_validations >= self.patience:
            log.warning(
                f"EarlyStopping: stopping at iteration {iteration}; "
                f"best val/loss={best_display}."
            )
            self.trainer.should_stop = True
        self._log_wandb(current, iteration)

    def _log_wandb(self, current: float, iteration: int) -> None:
        if wandb.run is None:
            return
        wandb.log(
            {
                "val/loss_mean": current,
                "early_stopping/best": self.best if self.best is not None else float("nan"),
                "early_stopping/bad_validations": self.bad_validations,
            },
            step=iteration,
        )
