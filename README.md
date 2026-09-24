# AutoTarget — implementation excerpt for anonymous review

This repository shows **how the cache-target method is implemented**. It is
not the complete reproduction package. The model checkpoints, prompt and seed
lists, training/inference launchers, evaluation scripts, raw outputs, and
third-party baselines are intentionally absent during review.

## Included code

| File | Implementation shown |
| --- | --- |
| `target_selection.py` | Decoder-aware RGB risk and the best-versus-runner-up decision rule. |
| `pixart_lcm_target_codec.py` | PixArt-LCM target encoding, conversion back to the solver input, and cache callbacks. |
| `flux_autotarget_codec.py` | FLUX Euler target encoding/decoding and cache callbacks. |
| `hunyuan_autotarget_scheduler.py` | Hunyuan target encoding/decoding and scheduler integration. |
| `velocity_controller.py` | Exact-call/reuse hook around the DiT forward pass. |
| `joint_cache_planner.py` | Optional dynamic-programming planner for joint schedules and targets. |

For a fixed exact-call schedule, an exact model output is converted to a
candidate cache tensor at an anchor. At a reuse step, the cached tensor is
interpreted using the *current* latent and timestep, then passed to the
original solver. During calibration, candidate cached outputs are compared
with matched uncached outputs after image decoding. The lowest-risk target is
chosen only if its risk is sufficiently below the runner-up. The optional
joint planner is separate from the paper's fixed-schedule comparisons.

The code is a focused implementation excerpt, not a turnkey reproduction
command. In particular, `target_selection.py` expects decoded images from a
caller and does not generate images itself. The full reproducibility package
can be released after the double-blind review period.

## Dependencies

The snippets use PyTorch. The model-specific adapters also require compatible
upstream model and scheduler classes. No model weights or data are included.
