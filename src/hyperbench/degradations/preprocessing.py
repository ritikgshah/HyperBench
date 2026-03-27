# Author: Ritik Shah

"""Shared preprocessing utilities for HyperBench degradations."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def validate_hsi(image: Array) -> Array:
    """Validate and standardize an HSI array.

    Parameters
    ----------
    image:
        Input array expected to have shape `(H, W, B)`.

    Returns
    -------
    np.ndarray
        Float32 image array.

    Raises
    ------
    ValueError
        If the input is not a finite 3D array.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (H, W, B), got {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("Image contains non-finite values")
    return image


def normalize_image(
    image: Array,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Array:
    """Normalize an image to [0, 1] using percentile clipping.

    Parameters
    ----------
    image:
        Input HSI of shape `(H, W, B)`.
    lower_percentile:
        Lower clipping percentile.
    upper_percentile:
        Upper clipping percentile.

    Returns
    -------
    np.ndarray
        Normalized float32 image in `[0, 1]`.
    """
    image = validate_hsi(image)

    if not (0 <= lower_percentile < upper_percentile <= 100):
        raise ValueError(
            f"Invalid percentile range: lower={lower_percentile}, upper={upper_percentile}"
        )

    min_val, max_val = np.percentile(image, [lower_percentile, upper_percentile])
    denom = max(float(max_val - min_val), 1e-12)
    clipped = np.clip(image, min_val, max_val)
    normalized = (clipped - min_val) / denom
    return normalized.astype(np.float32)


def add_awgn_by_snr(
    image: Array,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> Array:
    """Add additive white Gaussian noise using a target SNR in dB.

    Noise power is computed independently per spectral band.

    Parameters
    ----------
    image:
        Input image of shape `(H, W, B)`.
    snr_db:
        Desired signal-to-noise ratio in decibels.
    rng:
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy image of the same shape as input.
    """
    image = validate_hsi(image)

    if not np.isfinite(snr_db):
        raise ValueError(f"snr_db must be finite, got {snr_db}")

    rng = rng or np.random.default_rng()

    signal_power = np.mean(image**2, axis=(0, 1))  # shape: (B,)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-12)
    noise_std = np.sqrt(noise_power).reshape(1, 1, -1)

    noise = rng.normal(loc=0.0, scale=1.0, size=image.shape).astype(np.float32)
    noisy = image + noise * noise_std
    return noisy.astype(np.float32)