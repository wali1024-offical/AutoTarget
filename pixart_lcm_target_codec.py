"""Cache-target conversion at the PixArt-LCM solver boundary.

This is the implementation of the cache representation and its inverse. It
does not include checkpoint loading, prompts, calibration runs, or evaluation.
"""

from __future__ import annotations

from typing import Any

import torch


TARGETS = ("epsilon", "x0", "v", "denoised", "denoised_residual")


def coefficients(scheduler, timestep: int, sample: torch.Tensor):
    alpha = scheduler.alphas_cumprod[timestep].to(device=sample.device, dtype=sample.dtype)
    beta = 1 - alpha
    c_skip, c_out = scheduler.get_scalings_for_boundary_condition_discrete(timestep)
    c_skip = torch.as_tensor(c_skip, dtype=sample.dtype, device=sample.device)
    c_out = torch.as_tensor(c_out, dtype=sample.dtype, device=sample.device)
    return alpha, beta, c_skip, c_out


def epsilon_to_target(
    target: str,
    epsilon: torch.Tensor,
    sample: torch.Tensor,
    timestep: int,
    scheduler,
) -> torch.Tensor:
    """Store one equal-shape tensor after an exact DiT call."""
    alpha, beta, c_skip, c_out = coefficients(scheduler, timestep, sample)
    x0 = (sample - beta.sqrt() * epsilon) / alpha.sqrt()
    denoised = c_out * x0 + c_skip * sample
    if target == "epsilon":
        return epsilon
    if target == "x0":
        return x0
    if target == "v":
        return alpha.sqrt() * epsilon - beta.sqrt() * x0
    if target == "denoised":
        return denoised
    if target == "denoised_residual":
        return denoised - sample
    raise ValueError(target)


def target_to_epsilon(
    target: str,
    cached: torch.Tensor,
    sample: torch.Tensor,
    timestep: int,
    scheduler,
) -> torch.Tensor:
    """Interpret the cached tensor at the current reuse step."""
    alpha, beta, c_skip, c_out = coefficients(scheduler, timestep, sample)
    if target == "epsilon":
        return cached
    if target == "x0":
        x0 = cached
    elif target == "v":
        return alpha.sqrt() * cached + beta.sqrt() * sample
    elif target == "denoised":
        x0 = (cached - c_skip * sample) / c_out
    elif target == "denoised_residual":
        x0 = (cached + sample - c_skip * sample) / c_out
    else:
        raise ValueError(target)
    return (sample - alpha.sqrt() * x0) / beta.sqrt()


def guided_epsilon(raw: torch.Tensor, guidance_scale: float, latent_channels: int) -> torch.Tensor:
    """Convert the DiT's classifier-free-guidance output to epsilon."""
    uncond, text = raw.chunk(2)
    output = uncond + guidance_scale * (text - uncond)
    if output.shape[1] == 2 * latent_channels:
        output = output.chunk(2, dim=1)[0]
    return output


def target_adapters(pipe, target: str, guidance_scale: float):
    """Return encoder/decoder callbacks for ``VelocityController``."""
    latent_channels = int(pipe.transformer.config.in_channels)
    out_channels = int(pipe.transformer.config.out_channels)

    def select(raw: torch.Tensor, args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
        model_input = kwargs.get("hidden_states", args[0] if args else None)
        if model_input is None:
            raise RuntimeError("Missing PixArt hidden states")
        sample = model_input.chunk(2)[0]
        timestep = int(kwargs["timestep"].flatten()[0].detach().cpu())
        epsilon = guided_epsilon(raw, guidance_scale, latent_channels)
        return epsilon_to_target(target, epsilon, sample, timestep, pipe.scheduler)

    def rebuild(cached: torch.Tensor, args: tuple[Any, ...], kwargs: dict[str, Any]):
        model_input = kwargs.get("hidden_states", args[0] if args else None)
        if model_input is None:
            raise RuntimeError("Missing PixArt hidden states")
        sample = model_input.chunk(2)[0]
        timestep = int(kwargs["timestep"].flatten()[0].detach().cpu())
        epsilon = target_to_epsilon(target, cached, sample, timestep, pipe.scheduler)
        raw = epsilon
        if out_channels == 2 * latent_channels:
            raw = torch.cat([raw, torch.zeros_like(raw)], dim=1)
        raw = torch.cat([raw, raw], dim=0)
        return (raw.to(dtype=model_input.dtype),)

    return select, rebuild
