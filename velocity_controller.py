"""Record or directly reuse a diffusion transformer's final flow output."""

from __future__ import annotations

from contextlib import AbstractContextManager
import time
from typing import Any, Callable

import torch


def tensor_pair_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left = left.float().flatten()
    right = right.float().flatten()
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    delta_norm = torch.linalg.vector_norm(right - left)
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
        "relative_l2": float(delta_norm / left_norm.clamp_min(1e-12)),
        "norm_ratio": float(right_norm / left_norm.clamp_min(1e-12)),
    }


class VelocityController(AbstractContextManager):
    """Wrap one-call-per-step DiT inference with zero-order output reuse."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        exact_steps: list[int] | None = None,
        record_velocities: bool = True,
        record_model_inputs: bool = False,
        velocity_selector: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cached_output_builder: Callable[[torch.Tensor], Any] | None = None,
        contextual_velocity_selector: Callable[
            [torch.Tensor, tuple[Any, ...], dict[str, Any]], torch.Tensor
        ]
        | None = None,
        contextual_cached_output_builder: Callable[
            [torch.Tensor, tuple[Any, ...], dict[str, Any]], Any
        ]
        | None = None,
    ) -> None:
        self.transformer = transformer
        self.exact_steps = None if exact_steps is None else set(exact_steps)
        self.record_velocities = record_velocities
        self.record_model_inputs = record_model_inputs
        self.velocity_selector = velocity_selector or (lambda output: output)
        self.cached_output_builder = cached_output_builder or (lambda velocity: (velocity,))
        self.contextual_velocity_selector = contextual_velocity_selector
        self.contextual_cached_output_builder = contextual_cached_output_builder
        self.original_forward = transformer.forward
        self.step = 0
        self.cached_velocity: torch.Tensor | None = None
        self.velocities: list[torch.Tensor] = []
        self.model_inputs: list[torch.Tensor] = []
        self.records: list[dict[str, Any]] = []

    @staticmethod
    def _sync(tensor: torch.Tensor | None = None) -> None:
        if torch.cuda.is_available() and (tensor is None or tensor.is_cuda):
            torch.cuda.synchronize(tensor.device if tensor is not None else None)

    def _forward(self, *args, **kwargs):
        step = self.step
        self.step += 1
        computed = self.exact_steps is None or step in self.exact_steps
        model_input = kwargs.get("hidden_states", args[0] if args else None)

        self._sync(model_input)
        started = time.perf_counter()
        if computed:
            output = self.original_forward(*args, **kwargs)
            raw_output = output[0] if isinstance(output, tuple) else output.sample
            if self.contextual_velocity_selector is None:
                velocity = self.velocity_selector(raw_output)
            else:
                velocity = self.contextual_velocity_selector(raw_output, args, kwargs)
            self.cached_velocity = velocity.detach()
        else:
            if self.cached_velocity is None:
                raise RuntimeError("The first denoising step must be computed")
            velocity = self.cached_velocity
            if self.contextual_cached_output_builder is None:
                output = self.cached_output_builder(velocity)
            else:
                output = self.contextual_cached_output_builder(velocity, args, kwargs)
        self._sync(velocity)
        elapsed = time.perf_counter() - started

        if self.record_velocities:
            self.velocities.append(velocity.detach().float().cpu())
        if self.record_model_inputs:
            if model_input is None:
                raise RuntimeError("Cannot record a missing transformer model input")
            self.model_inputs.append(model_input.detach().float().cpu())
        self.records.append(
            {
                "step": step,
                "computed": computed,
                "elapsed_seconds": elapsed,
                "velocity_shape": list(velocity.shape),
                "velocity_dtype": str(velocity.dtype),
                "velocity_numel": velocity.numel(),
                "velocity_bytes": velocity.numel() * velocity.element_size(),
                "velocity_l2_norm": float(torch.linalg.vector_norm(velocity.float())),
            }
        )
        return output

    def __enter__(self) -> "VelocityController":
        self.transformer.forward = self._forward
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.transformer.forward = self.original_forward
