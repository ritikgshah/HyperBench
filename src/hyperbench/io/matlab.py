# Author: Ritik Shah

"""MATLAB file loading utilities for HyperBench."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.io as sio


Array = np.ndarray

_MATLAB_METADATA_KEYS = {"__header__", "__version__", "__globals__"}


def _is_valid_hsi_candidate(value: object) -> bool:
    """Check if a MATLAB variable could be an HSI cube."""
    if not isinstance(value, np.ndarray):
        return False
    if value.ndim != 3:
        return False
    if value.size == 0:
        return False
    if not np.issubdtype(value.dtype, np.number):
        return False
    return True


def infer_mat_hsi_key(path: str | Path) -> str:
    """Infer the most likely HSI variable in a MATLAB file."""
    path = Path(path)
    mat = sio.loadmat(path)

    candidates = []
    visible_keys = []

    for key, value in mat.items():
        if key in _MATLAB_METADATA_KEYS:
            continue

        visible_keys.append(key)

        if _is_valid_hsi_candidate(value):
            candidates.append(key)

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise ValueError(
            f"No 3D numeric arrays found in MATLAB file '{path}'. "
            f"Available variables: {visible_keys}"
        )

    raise ValueError(
        f"Multiple possible HSI arrays found in '{path}': {candidates}. "
        "Please specify the variable using `key=`."
    )


def load_mat_hsi(path: str | Path, *, key: str | None = None) -> Array:
    """Load an HSI cube from a MATLAB `.mat` file.

    Parameters
    ----------
    path : str or Path
        MATLAB file path.
    key : str, optional
        Variable name containing the HSI cube.

    Returns
    -------
    np.ndarray
        Float32 hyperspectral cube.
    """
    path = Path(path)
    mat = sio.loadmat(path)

    selected_key = infer_mat_hsi_key(path) if key is None else key

    if selected_key not in mat:
        available = [k for k in mat.keys() if k not in _MATLAB_METADATA_KEYS]
        raise KeyError(
            f"Key '{selected_key}' not found in MATLAB file '{path}'. "
            f"Available variables: {available}"
        )

    value = mat[selected_key]

    if not _is_valid_hsi_candidate(value):
        raise ValueError(
            f"Variable '{selected_key}' in '{path}' is not a valid HSI cube. "
            f"Shape={getattr(value, 'shape', None)}, dtype={getattr(value, 'dtype', None)}"
        )

    return np.asarray(value, dtype=np.float32)