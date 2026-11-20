# Development

- Keep `scintfibrefastsim.core` free of experiment imports (tests enforce it).
- Do not vendor lumen's tallymap/CUDA-only stack.
- Prefer copy-with-NOTICE or shim when moving code out of mu2e-fastsim; never
  break flag-off parity in the same PR that deletes the duplicate.
- Docs: Diátaxis under `docs/`; examples must run as written.
