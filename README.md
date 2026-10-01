<div align="center">

# AutoTarget

**Rethinking What to Cache in Few-Step Diffusion Transformers: Solver-Aware Target Selection**

Shuo Yang<sup>*</sup>, Lihao Fang<sup>*</sup>, Yi Zhang, Haixiang Wang, Xingcheng Ye, Shufan Chen, Jipeng Guo, Youqing Wang<sup>†</sup>

<sup>*</sup> Equal contribution &nbsp;·&nbsp; <sup>†</sup> Corresponding author

*Submitted to ICLR 2027*

</div>

AutoTarget selects **which tensor to cache** for a fixed diffusion model,
sampling solver, timestep sequence, and schedule of exact DiT calls. It
compares candidate targets on matched uncached and cached calibration runs,
then reuses the selected target through the original solver.

> **Release status.** This is the official repository for the method
> implementation. It currently contains the target-selection rule and
> model-specific cache adapters, not the full experiment reproduction package.

## How it works

```mermaid
flowchart LR
    A[Exact DiT call and target encoding] --> B[Decoder-aware calibration and risk comparison]
    B --> C[Target selection and solver reuse]
```

Calibration scores the mean RGB reconstruction error after image decoding.
The target with the lowest risk is selected only when its risk is sufficiently
below the runner-up. At a reuse step, the cached tensor is interpreted using
the current latent and timestep before the solver update. The model, solver,
timestep sequence, exact-call schedule, and cache budget are held fixed when
comparing targets.

## Repository contents

| File | Role |
| --- | --- |
| [`target_selection.py`](target_selection.py) | Decoder-aware RGB risk, risk normalization, and the best-versus-runner-up decision. |
| [`pixart_lcm_target_codec.py`](pixart_lcm_target_codec.py) | PixArt-LCM target encoding, inverse conversion, and cache callbacks. |
| [`flux_autotarget_codec.py`](flux_autotarget_codec.py) | FLUX Euler target encoding, decoding, and cache callbacks. |
| [`hunyuan_autotarget_scheduler.py`](hunyuan_autotarget_scheduler.py) | Hunyuan target encoding, decoding, and scheduler integration. |
| [`velocity_controller.py`](velocity_controller.py) | Exact-call and reuse hook around the DiT forward pass. |
| [`joint_cache_planner.py`](joint_cache_planner.py) | Optional joint schedule-and-target planner, separate from the paper's fixed-schedule comparisons. |

Start with `target_selection.py` for the decision rule, then read the adapter
for the model of interest. The adapters expose the cache representation and
its conversion back to the quantity consumed by the solver.

## Using this code

These modules are implementation components rather than a standalone sampling
script. They use PyTorch and expect compatible upstream model and scheduler
classes. In particular, `target_selection.py` takes matched decoded images
from a caller; it does not load checkpoints or generate images.

The full reproduction package—checkpoints, prompt and seed lists, inference
launchers, evaluation scripts, raw outputs, and third-party baselines—is
planned for release after the review period. No model weights or data are
included here.

## Paper

The paper has been submitted to ICLR 2027. An arXiv link and citation will be
added when the preprint is available.
