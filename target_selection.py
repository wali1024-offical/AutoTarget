"""The small, model-independent part of AutoTarget's decision rule.

The caller supplies exact and cached decoded outputs from the same inputs.
This module deliberately omits checkpoint loading and experiment runners.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TargetDecision:
    selected: str | None
    best: str
    runner_up: str
    runner_up_over_best: float
    raw_risks: dict[str, float]
    normalized_risks: dict[str, float]


def decoded_rgb_risks(
    exact_images: Sequence[torch.Tensor],
    cached_images: Mapping[str, Sequence[torch.Tensor]],
) -> dict[str, float]:
    """Average per-image RGB MSE against matched uncached outputs."""
    if not exact_images:
        raise ValueError("At least one calibration image is required")
    if len(cached_images) < 2:
        raise ValueError("At least two cache targets are required")
    risks: dict[str, float] = {}
    for target, images in cached_images.items():
        if len(images) != len(exact_images):
            raise ValueError(f"Unmatched image count for {target}")
        errors = []
        for exact, cached in zip(exact_images, images):
            if exact.shape != cached.shape:
                raise ValueError(f"Unmatched image shape for {target}")
            errors.append(float(torch.mean((cached.float() - exact.float()).square())))
        risks[target] = sum(errors) / len(errors)
    return risks


def choose_target(
    raw_risks: Mapping[str, float], *, min_ratio: float = 1.10
) -> TargetDecision:
    """Select the lowest-risk target when the runner-up gap is large enough."""
    if len(raw_risks) < 2:
        raise ValueError("At least two candidate risks are required")
    if any(value < 0 or not torch.isfinite(torch.tensor(value)) for value in raw_risks.values()):
        raise ValueError("Risks must be finite and nonnegative")
    ranking = sorted(raw_risks, key=lambda target: (raw_risks[target], target))
    best, runner_up = ranking[:2]
    best_risk = float(raw_risks[best])
    ratio = float("inf") if best_risk == 0 else float(raw_risks[runner_up]) / best_risk
    scale = best_risk if best_risk > 0 else 1.0
    return TargetDecision(
        selected=best if ratio >= min_ratio else None,
        best=best,
        runner_up=runner_up,
        runner_up_over_best=ratio,
        raw_risks={target: float(value) for target, value in raw_risks.items()},
        normalized_risks={target: float(value) / scale for target, value in raw_risks.items()},
    )
