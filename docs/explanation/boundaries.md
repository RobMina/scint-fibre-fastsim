# Package boundaries

| Owns | Does not own |
|---|---|
| Strip/cell + WLS fibre optical MC → tables; toy geometries; G4 compare drivers | Offline / NOvASoft digi; Mu2e-internal v6 binaries; private full GDML worlds |
| Common photosensor **interface** and SiPM bind helpers | Production SiPM pixel model source (lives in `crvfastsim.response` until extracted) |
| Public NOvA TDR cell caricature | Private NOvA GDML |

`mu2e-fastsim` re-exports `crvfastsim.lookupgen.core` from this package
([PR #326](https://github.com/RobMina/mu2e-fastsim/pull/326)): the module objects are
shared. Experiment adapters keep offline table formats and Mu2e-tuned optical models.

## Public hygiene

This repository is public. Do **not** put private wiki paths, home-directory paths, or
site-specific scratch trees in docs, examples, commits, or PR text. Configure sites
through environment variables (`$SCINTFIBRE_*`, `$CRVFS_*`, kit `site.env` / endpoint
env) only.
