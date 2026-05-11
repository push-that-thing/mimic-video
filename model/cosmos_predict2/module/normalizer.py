import numpy as np
import torch
from torch import nn

from cosmos_predict2.data.action.types import NormalizationType

PERCENTILE_STEP: float = 0.1  # in percent
PERCENTILE_GRID: np.ndarray = np.arange(0.0, 100.0 + 1e-9, PERCENTILE_STEP, dtype=float)


def grid_index_for_prob(prob_percent: float) -> int:
    """Return index in PERCENTILE_GRID closest to `prob_percent` (0-100)."""
    return round(prob_percent / PERCENTILE_STEP)


_P2_IDX: int = grid_index_for_prob(2)
_P98_IDX: int = grid_index_for_prob(98)


def array_to_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute per-dataset statistics for an array of samples.

    Returns:
    -------
    dict
        {
            "min":          (*feat_shape,)
            "max":          (*feat_shape,)
            "mean":         (*feat_shape,)
            "std":          (*feat_shape,)
            "percentiles":  (P, *feat_shape) values at PERCENTILE_GRID
            "clamp_min":    (*feat_shape,)  # heuristic local clamp lower
            "clamp_max":    (*feat_shape,)  # heuristic local clamp upper
            "mean_clamped": (*feat_shape,)  # exact local clamped mean
            "std_clamped":  (*feat_shape,)  # exact local clamped std
        }
    """
    arr = np.asarray(arr)
    percentiles = np.percentile(arr, PERCENTILE_GRID, axis=0)

    mean = np.mean(arr, axis=0)
    p2 = np.take(percentiles, _P2_IDX, axis=0)
    p98 = np.take(percentiles, _P98_IDX, axis=0)

    clamp_min = mean + 1.5 * (p2 - mean)
    clamp_max = mean + 1.5 * (p98 - mean)

    clamped = np.clip(arr, clamp_min, clamp_max)
    mean_clamped = np.mean(clamped, axis=0)
    std_clamped = np.std(clamped, axis=0)

    return {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": mean,
        "std": np.std(arr, axis=0),
        "percentiles": percentiles,
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
        "mean_clamped": mean_clamped,
        "std_clamped": std_clamped,
    }


class _StaticLinearNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.tensor(0.0), persistent=True)
        self.register_buffer("scale", torch.tensor(1.0), persistent=True)
        self.register_buffer("clamp_min", torch.tensor(float("-inf")), persistent=True)
        self.register_buffer("clamp_max", torch.tensor(float("inf")), persistent=True)
        self._register_load_state_dict_pre_hook(self._auto_resize, with_module=True)

    @staticmethod
    def _auto_resize(module: nn.Module, state_dict: dict, prefix: str, *_) -> None:
        for name in ("shift", "scale", "clamp_min", "clamp_max"):
            key = prefix + name
            if key in state_dict and state_dict[key].shape != module._buffers[name].shape:  # ty:ignore[unresolved-attribute]
                module._buffers[name] = torch.empty_like(
                    state_dict[key],
                    device=module._buffers[name].device,  # ty:ignore[unresolved-attribute]
                )

    def set_parameters(
        self,
        *,
        shift: np.ndarray | torch.Tensor,
        scale: np.ndarray | torch.Tensor,
        clamp_min: np.ndarray | torch.Tensor | None = None,
        clamp_max: np.ndarray | torch.Tensor | None = None,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        shift = torch.as_tensor(shift, dtype=dtype, device=device)
        scale = torch.as_tensor(scale, dtype=dtype, device=device)
        if shift.shape != scale.shape:
            msg = "shift and scale must have the same shape."
            raise ValueError(msg)
        self.shift = shift.clone().to(device=device)
        self.scale = scale.clone().to(device=device)

        if clamp_min is not None:
            self.clamp_min = torch.as_tensor(clamp_min, dtype=dtype, device=device)
        if clamp_max is not None:
            self.clamp_max = torch.as_tensor(clamp_max, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x - self.shift) * self.scale
        if not torch.isinf(self.clamp_min).all():
            y = torch.maximum(y, self.clamp_min)
        if not torch.isinf(self.clamp_max).all():
            y = torch.minimum(y, self.clamp_max)
        return y

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale + self.shift


class StaticBatchNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norms: nn.ModuleDict = nn.ModuleDict()
        self.register_buffer("_device_tracker", torch.tensor(0.0), persistent=False)
        self._register_load_state_dict_pre_hook(self._attach, with_module=True)

    @staticmethod
    def _attach(module: nn.Module, state_dict: dict, prefix: str, *_) -> None:
        base = prefix + "norms."
        for full_key in state_dict:
            if full_key.startswith(base):
                key = full_key.removeprefix(base).split(".", 1)[0]
                if key not in module.norms:  # ty:ignore[unsupported-operator]
                    dev = module._buffers["_device_tracker"].device  # ty:ignore[unresolved-attribute]
                    module.norms[key] = _StaticLinearNormalizer().to(dev)  # ty:ignore[invalid-assignment]

    def set_parameters(
        self,
        parameters: dict[str, dict[str, np.ndarray | torch.Tensor]],
        *,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        for key, key_params in parameters.items():
            if key not in self.norms:
                self.norms[key] = _StaticLinearNormalizer()
            self.norms[key].set_parameters(**key_params, dtype=dtype, device=device)

    def build_from_stats(
        self,
        stats: dict[str, dict[str, np.ndarray | torch.Tensor]],
        *,
        normalization_types: dict[str, NormalizationType],
        concat_groups: dict[str, list[str]] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> "StaticBatchNormalizer":
        """Fit self from per field dataset statistics."""

        def _to_np(x):
            return np.asarray(x, dtype=np.float32)

        params: dict[str, dict[str, np.ndarray]] = {}

        for key, mode in normalization_types.items():
            if mode is NormalizationType.NONE:
                continue

            s = stats[key]
            if mode is NormalizationType.SQUASH:
                p2, p98 = (
                    _to_np(s["percentiles"][grid_index_for_prob(2)]),
                    _to_np(s["percentiles"][grid_index_for_prob(98)]),
                )
                half_range = 0.5 * (p98 - p2)
                with np.errstate(divide="ignore"):
                    scale = np.where(abs(half_range) < 1e-5, 1.0, 1.0 / half_range).astype(np.float32)
                params[key] = {
                    "shift": 0.5 * (p2 + p98),
                    "scale": scale,
                    "clamp_min": np.full_like(p2, -1.5, dtype=np.float32),
                    "clamp_max": np.full_like(p2, 1.5, dtype=np.float32),
                }

            elif mode is NormalizationType.SQUASH_HARD:
                p2, p98 = (
                    _to_np(s["percentiles"][grid_index_for_prob(2)]),
                    _to_np(s["percentiles"][grid_index_for_prob(98)]),
                )
                unit_mean = 0.5 * (p2 + p98)
                unit_half_range = 0.5 * (p98 - p2)

                start = np.maximum(unit_mean - 1.5 * unit_half_range, s["min"])
                end = np.minimum(unit_mean + 1.5 * unit_half_range, s["max"])

                half_range = 0.5 * (end - start)
                with np.errstate(divide="ignore"):
                    scale = np.where(abs(half_range) < 1e-5, 1.0, 1.0 / half_range).astype(np.float32)
                params[key] = {
                    "shift": 0.5 * (start + end),
                    "scale": scale,
                    "clamp_min": np.full_like(p2, -1.0, dtype=np.float32),
                    "clamp_max": np.full_like(p2, 1.0, dtype=np.float32),
                }

            elif mode is NormalizationType.VARIANCE:
                mean, mean_clamped, std_clamped = (
                    _to_np(s["mean"]),
                    _to_np(s["mean_clamped"]),
                    _to_np(s["std_clamped"]),
                )
                with np.errstate(divide="ignore"):
                    scale = np.where(std_clamped < 1e-5, 1.0, 1.0 / std_clamped).astype(np.float32)

                p2 = _to_np(s["percentiles"][grid_index_for_prob(2)])
                p98 = _to_np(s["percentiles"][grid_index_for_prob(98)])
                # pre norm
                clamp_min_raw = mean + 1.5 * (p2 - mean)
                clamp_max_raw = mean + 1.5 * (p98 - mean)
                # post norm
                clamp_min = (clamp_min_raw - mean_clamped) * scale
                clamp_max = (clamp_max_raw - mean_clamped) * scale

                params[key] = {"shift": mean_clamped, "scale": scale, "clamp_min": clamp_min, "clamp_max": clamp_max}

            elif mode is NormalizationType.IDENTITY:
                mean = _to_np(s["mean"])
                params[key] = {
                    "shift": np.zeros_like(mean),
                    "scale": np.ones_like(mean),
                    "clamp_min": np.full_like(mean, -np.inf, dtype=np.float32),
                    "clamp_max": np.full_like(mean, np.inf, dtype=np.float32),
                }

            else:
                raise NotImplementedError(mode)

        if concat_groups:
            for new_key, group in concat_groups.items():
                params[new_key] = {
                    "shift": np.concatenate([params[k]["shift"] for k in group], axis=-1),
                    "scale": np.concatenate([params[k]["scale"] for k in group], axis=-1),
                    "clamp_min": np.concatenate([params[k]["clamp_min"] for k in group], axis=-1),
                    "clamp_max": np.concatenate([params[k]["clamp_max"] for k in group], axis=-1),
                }
                for k in group:
                    if k != new_key:
                        params.pop(k, None)

        self.set_parameters(params, dtype=dtype, device=device)  # ty:ignore[invalid-argument-type]
        return self

    def forward(self, batch: dict[str, torch.Tensor], *, strict: bool = False) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, tensor in batch.items():
            if key in self.norms:
                out[key] = self.norms[key](tensor)
            elif strict:
                raise KeyError(key)
            else:
                out[key] = tensor
        return out

    def unnormalize(self, batch: dict[str, torch.Tensor], *, strict: bool = False) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, tensor in batch.items():
            if key in self.norms:
                out[key] = self.norms[key].unnormalize(tensor)
            elif strict:
                raise KeyError(key)
            else:
                out[key] = tensor
        return out
