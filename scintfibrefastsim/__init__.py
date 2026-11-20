"""scint-fibre-fastsim: scintillator + WLS-fibre optical MC and table generation.

Share fibre/WLS transport and the table contract across experiments. Photosensor
response and offline table formats live in backends (Mu2e SiPM, NOvA APD).
"""

__version__ = "0.1.2"

from . import backends, core, geometry, validation

__all__ = ["backends", "core", "geometry", "validation", "__version__"]
