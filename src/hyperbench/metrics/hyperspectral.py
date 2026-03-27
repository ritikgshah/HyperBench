# Author: Ritik Shah

"""Hyperspectral-specific metrics for HyperBench."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def _validate_pair(reference: Array, prediction: Array) -> tuple[Array, Array]:
    """Validate a reference/prediction HSI pair."""
    reference = np.asarray(reference, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)

    if reference.ndim != 3 or prediction.ndim != 3:
        raise ValueError(
            f"Both inputs must have shape (H, W, B); "
            f"got {reference.shape} and {prediction.shape}"
        )

    if reference.shape != prediction.shape:
        raise ValueError(
            f"Reference and prediction must have the same shape; "
            f"got {reference.shape} and {prediction.shape}"
        )

    if not np.all(np.isfinite(reference)):
        raise ValueError("Reference image contains non-finite values")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Prediction image contains non-finite values")

    return reference, prediction


def compute_sam(reference: Array, prediction: Array, eps: float = 1e-8) -> float:
    """Compute the Spectral Angle Mapper (SAM) in degrees.

    Parameters
    ----------
    reference:
        Ground-truth HSI of shape `(H, W, B)`.
    prediction:
        Reconstructed HSI of shape `(H, W, B)`.
    eps:
        Small constant for numerical stability.

    Returns
    -------
    float
        Mean spectral angle in degrees across all pixels.
    """
    reference, prediction = _validate_pair(reference, prediction)

    ref_norm = np.sqrt(np.sum(reference**2, axis=-1))
    pred_norm = np.sqrt(np.sum(prediction**2, axis=-1))
    denom = ref_norm * pred_norm

    dot = np.sum(reference * prediction, axis=-1)
    cosine = dot / (denom + eps)
    cosine = np.clip(cosine, -1.0 + 1e-9, 1.0 - 1e-9)

    angle_rad = np.arccos(cosine)
    angle_deg = np.degrees(np.mean(angle_rad))
    return float(angle_deg)