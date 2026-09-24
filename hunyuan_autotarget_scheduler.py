"""Solver-boundary cache targets for Hunyuan Restricted MeanFlow inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


TARGET_KINDS = ("native", "solver_update", "next_latent")


def collapse_flow_target_class(target_kind: str) -> str:
    """Return the rollout-equivalence class for deterministic flow Euler."""

    if target_kind in {"native", "velocity", "x0", "noise_endpoint"}:
        return "velocity"
    if target_kind in {"solver_update", "next_latent"}:
        return target_kind
    raise ValueError(f"Unknown flow target: {target_kind}")


def flow_endpoints(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct clean and noise endpoints under x_sigma=(1-sigma)x0+sigma*eps."""

    sigma_tensor = torch.as_tensor(sigma, device=sample.device, dtype=torch.float32)
    sample_f = sample.float()
    velocity_f = velocity.float()
    x0 = sample_f - sigma_tensor * velocity_f
    epsilon = sample_f + (1.0 - sigma_tensor) * velocity_f
    return x0, epsilon


def velocity_from_clean_endpoint(
    sample: torch.Tensor,
    x0: torch.Tensor,
    sigma: float | torch.Tensor,
) -> torch.Tensor:
    sigma_tensor = torch.as_tensor(sigma, device=sample.device, dtype=torch.float32)
    return (sample.float() - x0.float()) / sigma_tensor


def velocity_from_noise_endpoint(
    sample: torch.Tensor,
    epsilon: torch.Tensor,
    sigma: float | torch.Tensor,
) -> torch.Tensor:
    sigma_tensor = torch.as_tensor(sigma, device=sample.device, dtype=torch.float32)
    return (epsilon.float() - sample.float()) / (1.0 - sigma_tensor)


def encode_target(
    target_kind: str,
    sample: torch.Tensor,
    model_output: torch.Tensor,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    """Encode one equal-shape cache target in the model output dtype."""

    if target_kind not in TARGET_KINDS:
        raise ValueError(f"Unsupported target kind: {target_kind}")
    dt_tensor = torch.as_tensor(dt, device=sample.device, dtype=torch.float32)
    output_f = model_output.float()
    if target_kind == "native":
        encoded = output_f
    elif target_kind == "solver_update":
        encoded = dt_tensor * output_f
    else:
        encoded = sample.float() + dt_tensor * output_f
    return encoded.to(dtype=model_output.dtype).detach()


def decode_target(
    target_kind: str,
    cache: torch.Tensor,
    sample: torch.Tensor,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    """Decode a cached target into the model output expected by Euler.step."""

    if target_kind not in TARGET_KINDS:
        raise ValueError(f"Unsupported target kind: {target_kind}")
    dt_tensor = torch.as_tensor(dt, device=sample.device, dtype=torch.float32)
    if bool((dt_tensor == 0).any()):
        raise ZeroDivisionError("Cannot decode a solver target for dt=0")
    if target_kind == "native":
        return cache.float()
    if target_kind == "solver_update":
        return cache.float() / dt_tensor
    return (cache.float() - sample.float()) / dt_tensor


@dataclass(frozen=True)
class CacheEvent:
    step: int
    anchor: bool
    dt: float
    target_kind: str
    cache_bytes: int
    cache_dtype: str
    cache_shape: tuple[int, ...]


def make_autotarget_scheduler(
    base_scheduler: type,
    target_kind: str,
    anchor_steps: Iterable[int],
) -> type:
    """Create a configured scheduler class for apply_disca's constructor hook."""

    if target_kind not in TARGET_KINDS:
        raise ValueError(f"Unsupported target kind: {target_kind}")
    anchors = frozenset(int(step) for step in anchor_steps)
    if 0 not in anchors:
        raise ValueError("The fixed schedule must include step 0")

    class AutoTargetScheduler(base_scheduler):
        autotarget_kind = target_kind
        autotarget_anchor_steps = anchors

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.autotarget_cache = None
            self.autotarget_events: list[CacheEvent] = []

        def step(self, model_output, timestep, sample, return_dict=True):
            if self.step_index is None:
                self._init_step_index(timestep)
            step_index = int(self.step_index)
            dt = self.sigmas[step_index + 1] - self.sigmas[step_index]
            is_anchor = step_index in self.autotarget_anchor_steps

            if is_anchor:
                self.autotarget_cache = encode_target(
                    self.autotarget_kind,
                    sample,
                    model_output,
                    dt,
                )
            else:
                if self.autotarget_cache is None:
                    raise RuntimeError("AutoTarget cache is empty at a non-anchor step")
                model_output = decode_target(
                    self.autotarget_kind,
                    self.autotarget_cache,
                    sample,
                    dt,
                )

            cache = self.autotarget_cache
            self.autotarget_events.append(
                CacheEvent(
                    step=step_index,
                    anchor=is_anchor,
                    dt=float(dt),
                    target_kind=self.autotarget_kind,
                    cache_bytes=cache.numel() * cache.element_size(),
                    cache_dtype=str(cache.dtype),
                    cache_shape=tuple(cache.shape),
                )
            )
            return super().step(
                model_output,
                timestep,
                sample,
                return_dict=return_dict,
            )

    AutoTargetScheduler.__name__ = (
        f"AutoTarget{target_kind.title().replace('_', '')}Scheduler"
    )
    return AutoTargetScheduler


def make_calibration_scheduler(
    base_scheduler: type,
    target_kinds: Iterable[str],
    anchor_steps: Iterable[int],
    collect_transition_probes: bool = False,
) -> type:
    """Score cache targets on one exact trace without candidate DiT rollouts."""

    kinds = tuple(target_kinds)
    invalid = [kind for kind in kinds if kind not in TARGET_KINDS]
    if invalid:
        raise ValueError(f"Unsupported target kinds: {invalid}")
    anchors = frozenset(int(step) for step in anchor_steps)
    if 0 not in anchors:
        raise ValueError("The fixed schedule must include step 0")

    class CalibrationScheduler(base_scheduler):
        autotarget_kinds = kinds
        autotarget_anchor_steps = anchors

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.autotarget_caches = {}
            self.autotarget_calibration = {kind: [] for kind in kinds}
            self.autotarget_transition_probes = []

        def step(self, model_output, timestep, sample, return_dict=True):
            if self.step_index is None:
                self._init_step_index(timestep)
            step_index = int(self.step_index)
            dt = self.sigmas[step_index + 1] - self.sigmas[step_index]

            if step_index in self.autotarget_anchor_steps:
                self.autotarget_caches = {
                    kind: encode_target(kind, sample, model_output, dt)
                    for kind in self.autotarget_kinds
                }
            else:
                exact_update = dt.float() * model_output.float()
                exact_next = sample.float() + exact_update
                denominator = exact_update.square().mean().clamp_min(1e-12)
                predicted_next = {}
                for kind in self.autotarget_kinds:
                    predicted_output = decode_target(
                        kind,
                        self.autotarget_caches[kind],
                        sample,
                        dt,
                    )
                    predicted_update = dt.float() * predicted_output
                    if collect_transition_probes:
                        predicted_next[kind] = (
                            sample.float() + predicted_update
                        ).detach().cpu()
                    delta = predicted_update - exact_update
                    self.autotarget_calibration[kind].append(
                        {
                            "step": step_index,
                            "update_mse": float(delta.square().mean()),
                            "normalized_update_mse": float(
                                delta.square().mean() / denominator
                            ),
                        }
                    )
                if collect_transition_probes:
                    self.autotarget_transition_probes.append(
                        {
                            "step": step_index,
                            "exact_next": exact_next.detach().cpu(),
                            "predicted_next": predicted_next,
                        }
                    )

            return super().step(
                model_output,
                timestep,
                sample,
                return_dict=return_dict,
            )

    CalibrationScheduler.__name__ = "AutoTargetCalibrationScheduler"
    return CalibrationScheduler
