# Author: Ritik Shah

"""Core full-reference metrics for HyperBench."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from .hyperspectral import compute_sam


Array = np.ndarray


def _validate_pair(reference: Array, prediction: Array) -> tuple[Array, Array]:
    """Validate a reference/prediction image pair."""
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


def _to_uint8(image: Array) -> Array:
    """Convert a float image in approximately [0, 1] to uint8."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _require_sewar() -> dict[str, Any]:
    """Import sewar lazily and provide a clear error if unavailable."""
    try:
        from sewar.full_ref import ergas, psnr, rmse, ssim, uqi
    except ImportError as exc:
        raise ImportError(
            "HyperBench metrics require the optional dependency 'sewar'. "
            "Install it with: pip install sewar"
        ) from exc

    return {
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim,
        "uqi": uqi,
        "ergas": ergas,
    }


def compute_rmse(reference: Array, prediction: Array) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    reference, prediction = _validate_pair(reference, prediction)
    sewar = _require_sewar()
    value = sewar["rmse"](reference, prediction)
    return float(value)


def compute_psnr(reference: Array, prediction: Array) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    reference, prediction = _validate_pair(reference, prediction)
    sewar = _require_sewar()

    ref_u8 = _to_uint8(reference)
    pred_u8 = _to_uint8(prediction)

    value = sewar["psnr"](ref_u8, pred_u8)
    return float(value)


def compute_ssim(reference: Array, prediction: Array) -> float:
    """Compute Structural Similarity Index (SSIM)."""
    reference, prediction = _validate_pair(reference, prediction)
    sewar = _require_sewar()

    ref_u8 = _to_uint8(reference)
    pred_u8 = _to_uint8(prediction)

    value, _ = sewar["ssim"](ref_u8, pred_u8)
    return float(value)


def compute_uiqi(reference: Array, prediction: Array) -> float:
    """Compute Universal Image Quality Index (UIQI)."""
    reference, prediction = _validate_pair(reference, prediction)
    sewar = _require_sewar()

    value = sewar["uqi"](reference, prediction)
    return float(value)


def compute_ergas(reference: Array, prediction: Array) -> float:
    """Compute ERGAS."""
    reference, prediction = _validate_pair(reference, prediction)
    sewar = _require_sewar()

    value = sewar["ergas"](reference, prediction)
    return float(value)


DEFAULT_METRICS: tuple[str, ...] = (
    "rmse",
    "psnr",
    "ssim",
    "uiqi",
    "ergas",
    "sam",
)

AVAILABLE_METRICS: dict[str, Callable[[Array, Array], float]] = {
    "rmse": compute_rmse,
    "psnr": compute_psnr,
    "ssim": compute_ssim,
    "uiqi": compute_uiqi,
    "ergas": compute_ergas,
    "sam": compute_sam,
}


def evaluate_metrics(
    reference: Array,
    prediction: Array,
    metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    """Evaluate a set of full-reference metrics.

    Parameters
    ----------
    reference:
        Ground-truth HSI of shape `(H, W, B)`.
    prediction:
        Predicted HSI of shape `(H, W, B)`.
    metrics:
        Iterable of metric names. If None, uses `DEFAULT_METRICS`.

    Returns
    -------
    dict[str, float]
        Mapping from upper-case metric name to metric value.
    """
    reference, prediction = _validate_pair(reference, prediction)

    metric_names = tuple(DEFAULT_METRICS if metrics is None else metrics)
    if not metric_names:
        raise ValueError("At least one metric must be requested")

    results: dict[str, float] = {}
    unknown = [name for name in metric_names if name.lower() not in AVAILABLE_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {unknown}. "
            f"Available metrics: {sorted(AVAILABLE_METRICS)}"
        )

    for name in metric_names:
        key = name.lower()
        value = AVAILABLE_METRICS[key](reference, prediction)
        results[key.upper()] = float(value)

    return results