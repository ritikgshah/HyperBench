# Author: Ritik Shah

"""Input/output utilities for HyperBench.

Currently supported formats:
- MATLAB .mat hyperspectral scenes
"""

from .loaders import load_hsi
from .matlab import load_mat_hsi

__all__ = [
    "load_hsi",
    "load_mat_hsi",
]