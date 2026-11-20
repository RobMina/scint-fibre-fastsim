# Hardware-agnostic JAX

Photon transport uses plain JAX/XLA. There is **no** NVIDIA-only (CUDA photon-loop
or tallymap) hard dependency.

| Backend | Path |
|---|---|
| NVIDIA | JAX CUDA plugin / cuda12 wheels |
| AMD | JAX ROCm builds where available |
| Intel | JAX CPU or XPU/PJRT plugins |
| CI | JAX CPU only (`.github/workflows/ci.yml`) |

Open: exact jax pin matrix for CUDA + ROCm + CPU/XPU in CI (tracked in the wiki
plan). Prefer dtype-explicit kernels (`float32` / `float64` arguments) over
global JAX defaults.
