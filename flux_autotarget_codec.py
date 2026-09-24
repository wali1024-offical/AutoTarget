"""Equal-byte solver-observable cache targets for FLUX Flow Euler inference."""

from __future__ import annotations

from typing import Any

import torch


SIGMAS = (1.0, 0.75, 0.5, 0.25, 0.0)
TARGETS = ("velocity", "step_update", "next_latent")
EQUIVALENCE_CLASSES = (
    ("velocity", "clean_endpoint", "noise_endpoint"),
    ("step_update",),
    ("next_latent",),
)


def encode_target(
    target: str,
    velocity: torch.Tensor,
    sample: torch.Tensor,
    sigma: float,
    dt: float,
) -> torch.Tensor:
    velocity32 = velocity.float()
    sample32 = sample.float()
    if target == "velocity":
        value = velocity32
    elif target == "clean_endpoint":
        value = sample32 - sigma * velocity32
    elif target == "noise_endpoint":
        value = sample32 + (1.0 - sigma) * velocity32
    elif target == "step_update":
        value = dt * velocity32
    elif target == "next_latent":
        value = sample32 + dt * velocity32
    else:
        raise ValueError(target)
    return value.to(velocity.dtype)


def decode_velocity(
    target: str,
    cached: torch.Tensor,
    sample: torch.Tensor,
    sigma: float,
    dt: float,
) -> torch.Tensor:
    if dt == 0.0:
        raise ZeroDivisionError("Cannot decode a cache target at zero step size")
    cached32 = cached.float()
    sample32 = sample.float()
    if target == "velocity":
        value = cached32
    elif target == "clean_endpoint":
        value = (sample32 - cached32) / sigma
    elif target == "noise_endpoint":
        value = (cached32 - sample32) / (1.0 - sigma)
    elif target == "step_update":
        value = cached32 / dt
    elif target == "next_latent":
        value = (cached32 - sample32) / dt
    else:
        raise ValueError(target)
    return value.to(cached.dtype)


def sigma_context(timestep: torch.Tensor, scheduler) -> tuple[float, float, int]:
    normalized_timestep = timestep.detach().reshape(-1)[0].cpu()
    observed = float(normalized_timestep.float())
    sigmas = [float(value) for value in scheduler.sigmas.detach().float().cpu()]
    # FluxPipeline casts the discrete value sigma * 1000 to the latent dtype
    # before dividing by 1000. Reproduce that BF16 quantization when matching.
    expected = [
        float(
            (
                torch.tensor(sigma * 1000, dtype=normalized_timestep.dtype)
                / 1000
            ).float()
        )
        for sigma in sigmas[:-1]
    ]
    index = min(range(len(expected)), key=lambda item: abs(expected[item] - observed))
    if abs(expected[index] - observed) > 5e-4:
        raise ValueError(
            f"Unexpected FLUX normalized timestep {observed}; expected one of {expected}"
        )
    return sigmas[index], sigmas[index + 1] - sigmas[index], index


def make_cache_callbacks(pipe, target: str) -> dict[str, Any]:
    if target not in TARGETS:
        raise ValueError(target)

    def context(args, kwargs):
        sample = kwargs.get("hidden_states", args[0] if args else None)
        if sample is None or "timestep" not in kwargs:
            raise RuntimeError("Missing FLUX sample or timestep context")
        sigma, dt, _ = sigma_context(kwargs["timestep"], pipe.scheduler)
        return sample, sigma, dt

    def select(raw_output, args, kwargs):
        sample, sigma, dt = context(args, kwargs)
        return encode_target(target, raw_output, sample, sigma, dt)

    def rebuild(cached, args, kwargs):
        sample, sigma, dt = context(args, kwargs)
        velocity = decode_velocity(target, cached, sample, sigma, dt)
        return (velocity.to(sample.dtype),)

    return {
        "contextual_velocity_selector": select,
        "contextual_cached_output_builder": rebuild,
    }


def one_skip_terminal(
    target: str,
    anchor_velocity: torch.Tensor,
    anchor_sample: torch.Tensor,
    current_sample: torch.Tensor,
    anchor_sigma: float,
    anchor_dt: float,
    current_sigma: float,
    current_dt: float,
    storage_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    cached = encode_target(
        target,
        anchor_velocity.to(storage_dtype),
        anchor_sample.to(storage_dtype),
        anchor_sigma,
        anchor_dt,
    ).to(storage_dtype)
    estimate = decode_velocity(
        target,
        cached,
        current_sample.to(storage_dtype),
        current_sigma,
        current_dt,
    ).to(storage_dtype)
    return (current_sample.to(storage_dtype).float() + current_dt * estimate.float()).to(
        storage_dtype
    )
