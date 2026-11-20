# Hardware and JAX backends

Photon transport uses plain JAX/XLA. There is **no** NVIDIA-only (CUDA photon-loop
or tallymap) hard dependency in this repository.

| Backend | Status in this repo |
|---|---|
| CPU (JAX) | **Tested** — GitHub Actions CI |
| NVIDIA (CUDA) | Intended via JAX CUDA wheels; not recorded in CI here |
| AMD (ROCm) | Intended where JAX ROCm builds exist; not recorded in CI here |
| Intel (XPU / CPU) | Intended via JAX CPU or XPU plugins; not recorded in CI here |

Prefer dtype-explicit kernels (`float32` / `float64` arguments) over global JAX
defaults. One CI leg pins `jax`/`numpy` to the versions used by mu2e-fastsim so
transport goldens do not drift silently.
