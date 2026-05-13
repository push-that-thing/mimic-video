# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WandbLogger callback: initialize a wandb run at training start, push
loss / lr / iter_time every ``every_n`` iterations, and call ``wandb.finish()``
at the end. Standard ``WANDB_*`` environment variables (``WANDB_PROJECT``,
``WANDB_ENTITY``, ``WANDB_NAME``) override the constructor arguments via
wandb's native env-var precedence.
"""
from __future__ import annotations

import os
import time

import attrs
import torch
import wandb

from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import distributed, log
from imaginaire.utils.distributed import rank0_only


class WandbLogger(EveryN):
    def __init__(
        self,
        project: str,
        entity: str | None = None,
        every_n: int = 100,
        group: str | None = None,
    ) -> None:
        super().__init__(every_n=every_n, run_at_start=False, barrier_after_run=False)
        self.project = project
        self.entity = entity
        self.group = group
        self._last_log_time: float | None = None
        self._last_lr: float = 0.0
        self._initialized = False

    @rank0_only
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model
        cfg_dict: dict = {}
        try:
            if attrs.has(type(self.trainer.config)):
                cfg_dict = attrs.asdict(self.trainer.config, recurse=True)
        except Exception:
            log.opt(exception=True).warning("WandbLogger: failed to serialize config; continuing with empty config")
        run_name = os.environ.get("WANDB_NAME") or self.trainer.config.job.name or None
        wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            group=self.group,
            config=cfg_dict,
            resume="allow",
        )
        self._initialized = True
        self._last_log_time = time.time()
        log.info(f"wandb initialized: project={self.project} entity={self.entity} run={run_name}")

    def on_before_optimizer_step(
        self,
        model_ddp,
        optimizer: torch.optim.Optimizer,
        scheduler,
        grad_scaler,
        iteration: int = 0,
    ) -> None:
        del model_ddp, grad_scaler, iteration
        try:
            self._last_lr = float(scheduler.get_last_lr()[0])
        except Exception:
            self._last_lr = float(optimizer.param_groups[0]["lr"])

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        del trainer, model, data_batch, output_batch
        if not distributed.is_rank0() or not self._initialized:
            return

        now = time.time()
        elapsed = now - (self._last_log_time or now)
        iter_time = elapsed / max(1, self.every_n)
        self._last_log_time = now

        wandb.log(
            {
                "train/loss": loss.item(),
                "train/lr": self._last_lr,
                "train/iter_time_s": iter_time,
            },
            step=iteration,
        )

    @rank0_only
    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model, iteration
        if self._initialized:
            wandb.finish()
