# Author: Ritik Shah

"""Framework-level validation helpers for HyperBench."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def ensure_channel_last_hsi(array: Array, *, name: str = "array") -> Array:
    """Validate that an array is a finite HSI cube with shape (H, W, B).

    Parameters
    ----------
    array:
        Input array.
    name:
        Human-readable variable name used in error messages.

    Returns
    -------
    np.ndarray
        Float32 validated array.
    """
    array = np.asarray(array, dtype=np.float32)

    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (H, W, B), got {array.shape}")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")

    if min(array.shape) < 1:
        raise ValueError(f"{name} has invalid shape {array.shape}")

    return array


def validate_prediction_shape(
    prediction: Array,
    reference: Array,
    *,
    prediction_name: str = "prediction",
    reference_name: str = "reference",
) -> Array:
    """Validate that a model prediction matches the reference HSI shape.

    Parameters
    ----------
    prediction:
        Predicted HSI.
    reference:
        Ground-truth HSI.

    Returns
    -------
    np.ndarray
        Validated prediction array as float32.
    """
    prediction = ensure_channel_last_hsi(prediction, name=prediction_name)
    reference = ensure_channel_last_hsi(reference, name=reference_name)

    if prediction.shape != reference.shape:
        raise ValueError(
            f"{prediction_name} shape {prediction.shape} does not match "
            f"{reference_name} shape {reference.shape}"
        )

    return prediction


def validate_downsample_ratio(ratio: int) -> int:
    """Validate a spatial downsampling ratio."""
    if not isinstance(ratio, int) or ratio <= 0:
        raise ValueError(f"downsample ratio must be a positive integer, got {ratio!r}")
    return ratio


def validate_scene_size_for_ratio(scene: Array, ratio: int) -> None:
    """Validate that a scene can be spatially downsampled by a ratio.

    Parameters
    ----------
    scene:
        Input HSI of shape `(H, W, B)`.
    ratio:
        Spatial downsampling factor.

    Raises
    ------
    ValueError
        If the scene is too small for the requested ratio.
    """
    scene = ensure_channel_last_hsi(scene, name="scene")
    ratio = validate_downsample_ratio(ratio)

    height, width, _ = scene.shape
    out_h = height // ratio
    out_w = width // ratio

    if out_h < 1 or out_w < 1:
        raise ValueError(
            f"Scene shape {scene.shape} is too small for downsample ratio {ratio}"
        )


def validate_band_count(
    num_bands: int,
    *,
    allowed: tuple[int, ...] = (3, 4, 8, 16),
    allow_panchromatic: bool = False,
) -> int:
    """Validate an MSI band count for HyperBench experiments.

    Parameters
    ----------
    num_bands:
        Requested MSI band count.
    allowed:
        Allowed built-in band counts for the benchmark.
    allow_panchromatic:
        Whether to allow `1` band.

    Returns
    -------
    int
        Validated band count.
    """
    if not isinstance(num_bands, int) or num_bands <= 0:
        raise ValueError(f"num_bands must be a positive integer, got {num_bands!r}")

    effective_allowed = allowed + ((1,) if allow_panchromatic else ())
    effective_allowed = tuple(sorted(set(effective_allowed)))

    if num_bands not in effective_allowed:
        raise ValueError(
            f"Unsupported MSI band count {num_bands}. "
            f"Allowed values: {effective_allowed}"
        )

    if num_bands == 1 and not allow_panchromatic:
        raise ValueError(
            "HyperBench benchmark protocol currently excludes c=1 "
            "(panchromatic MSI inputs)"
        )

    return num_bands


def validate_custom_srf(user_srf: Array, *, hsi_bands: int | None = None) -> Array:
    """Validate a user-provided SRF matrix.

    Parameters
    ----------
    user_srf:
        SRF matrix expected to have shape `(M, L)`.
    hsi_bands:
        Optional expected number of HSI bands `L`.

    Returns
    -------
    np.ndarray
        Float32 validated SRF matrix.
    """
    user_srf = np.asarray(user_srf, dtype=np.float32)

    if user_srf.ndim != 2:
        raise ValueError(f"user_srf must have shape (M, L), got {user_srf.shape}")

    if not np.all(np.isfinite(user_srf)):
        raise ValueError("user_srf contains non-finite values")

    if np.any(user_srf < 0):
        raise ValueError("user_srf must be non-negative")

    if np.any(user_srf.sum(axis=1) <= 1e-12):
        raise ValueError("user_srf contains one or more near-zero rows")

    if hsi_bands is not None and user_srf.shape[1] != hsi_bands:
        raise ValueError(
            f"user_srf expects {user_srf.shape[1]} HSI bands, but scene has {hsi_bands}"
        )

    return user_srf


def validate_experiment_inputs(
    scene: Array,
    *,
    downsample_ratio: int,
    num_msi_bands: int | None = None,
    allow_panchromatic: bool = False,
    user_srf: Array | None = None,
) -> None:
    """Validate a set of benchmark experiment inputs.

    This is a convenience function for runner/generator code that wants one
    place to validate the most common experiment inputs.
    """
    scene = ensure_channel_last_hsi(scene, name="scene")
    validate_scene_size_for_ratio(scene, downsample_ratio)

    if user_srf is not None:
        validate_custom_srf(user_srf, hsi_bands=scene.shape[2])

    if num_msi_bands is not None:
        validate_band_count(
            num_msi_bands,
            allow_panchromatic=allow_panchromatic,
        )