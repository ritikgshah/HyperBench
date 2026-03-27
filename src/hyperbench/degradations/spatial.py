# Author: Ritik Shah

"""Spatial degradation utilities for HyperBench."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import convolve2d

from .preprocessing import add_awgn_by_snr, normalize_image, validate_hsi


Array = np.ndarray


def _validate_psf(psf: Array) -> Array:
    psf = np.asarray(psf, dtype=np.float32)
    if psf.ndim != 2:
        raise ValueError(f"Expected PSF with shape (K, K), got {psf.shape}")
    if not np.all(np.isfinite(psf)):
        raise ValueError("PSF contains non-finite values")
    if abs(float(psf.sum())) < 1e-12:
        raise ValueError("PSF sum is too close to zero")
    return psf


def apply_psf(image: Array, psf: Array) -> Array:
    """Convolve each spectral band with a 2D PSF using same-sized output."""
    image = validate_hsi(image)
    psf = _validate_psf(psf)

    height, width, bands = image.shape
    blurred = np.zeros((height, width, bands), dtype=np.float32)

    for b in range(bands):
        blurred[:, :, b] = convolve2d(
            image[:, :, b],
            psf,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

    return blurred


def downsample_image(image: Array, factor: int, interpolation: int = cv2.INTER_AREA) -> Array:
    """Downsample an HSI spatially by resizing each band independently."""
    image = validate_hsi(image)

    if not isinstance(factor, int) or factor <= 0:
        raise ValueError(f"factor must be a positive integer, got {factor!r}")

    height, width, bands = image.shape
    out_h = height // factor
    out_w = width // factor

    if out_h < 1 or out_w < 1:
        raise ValueError(
            f"Downsampling factor {factor} is too large for input shape {image.shape}"
        )

    downsampled_bands = [
        cv2.resize(image[:, :, b], (out_w, out_h), interpolation=interpolation)
        for b in range(bands)
    ]
    out = np.stack(downsampled_bands, axis=-1).astype(np.float32)
    return out


def spatial_degradation(
    image: Array,
    psf: Array,
    downsample_ratio: int,
    snr_db: float,
    *,
    normalize: bool = True,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    rng: np.random.Generator | None = None,
) -> Array:
    """Apply the full spatial degradation pipeline.

    Pipeline:
        normalize -> blur with PSF -> downsample -> add AWGN
    """
    work = validate_hsi(image)

    if normalize:
        work = normalize_image(
            work,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )

    work = apply_psf(work, psf)
    work = downsample_image(work, downsample_ratio)
    work = add_awgn_by_snr(work, snr_db, rng=rng)
    return work.astype(np.float32)