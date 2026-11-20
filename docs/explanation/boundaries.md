# Package boundaries

| Owns | Does not own |
|---|---|
| Strip/cell + WLS fibre optical MC → tables; toy geometries; G4 compare drivers | Offline / NOvASoft digi; Mu2e-internal v6 binaries; private full GDML worlds |
| Common photosensor **interface** | Production SiPM pixel model (still `crvfastsim.response` until extract) |
| Public NOvA TDR cell caricature | Private NOvA GDML |

`mu2e-fastsim` today still vendors `crvfastsim.lookupgen.core`. This package holds
a **copy-with-NOTICE** of that core so the public API can grow without breaking
flag-off parity. Next extract: shim or re-export from mu2e-fastsim onto this
package, then delete the duplicate.
