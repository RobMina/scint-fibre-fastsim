"""Experiment-neutral core of the strip/cell optical-photon table generator.

Modules: counter, binning, optics, materials, transport, run, limits.

Imports only the standard library, NumPy, JAX and modules of this package (tests enforce it).
An experiment backend supplies photosensor response, counters/cells, bin edges and numbering,
any further optical models (optics.register) and the offline table format.

Copied from ``crvfastsim.lookupgen.core`` in RobMina/mu2e-fastsim (see NOTICE). Until mu2e-fastsim
depends on this package, that tree keeps its own copy; a later extract PR will re-export from here.
"""
