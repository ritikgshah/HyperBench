# Author: Ritik Shah

"""Top-level scene loader for HyperBench."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .matlab import load_mat_hsi


Array = np.ndarray


def validate_hsi_array(array: Array) -> Array:
    """Validate a loaded HSI array.

    Ensures the array is:
    - float32
    - finite
    - shape (H, W, B)

    Returns
    -------
    np.ndarray
        Validated HSI array.
    """
    array = np.asarray(array, dtype=np.float32)

    if array.ndim != 3:
        raise ValueError(
            f"HSI must have shape (H, W, B). Got array with shape {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError("HSI contains non-finite values")

    if min(array.shape) < 1:
        raise ValueError(f"Invalid HSI shape {array.shape}")

    return array


def load_hsi(path: str | Path, *, key: str | None = None) -> Array:
    """Load a hyperspectral scene.

    Parameters
    ----------
    path : str or Path
        Path to the `.mat` file.
    key : str, optional
        Variable name containing the HSI cube.

    Returns
    -------
    np.ndarray
        Hyperspectral cube with shape (H, W, B).
    """
    path = Path(path)

    if path.suffix.lower() != ".mat":
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. "
            "Currently only MATLAB `.mat` files are supported."
        )

    array = load_mat_hsi(path, key=key)

    return validate_hsi_array(array)