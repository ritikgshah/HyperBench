# Author: Ritik Shah

"""Spectral response function (SRF) utilities for HyperBench."""

from __future__ import annotations

import numpy as np


Array = np.ndarray

SUPPORTED_SRF_BAND_COUNTS: tuple[int, ...] = (3, 4, 8, 16)


def get_srf_band_specs(num_bands: int) -> Array:
    """Return band specifications as [center_nm, low_edge_nm, high_edge_nm].

    Supported built-in sensor configurations:
    - 3 bands: IKONOS RGB
    - 4 bands: IKONOS B/G/R/NIR
    - 8 bands: WorldView-2
    - 16 bands: WorldView-3
    """
    
    if num_bands == 3:
        band_specs = np.array([
            [480.0, 421.0, 539.0],
            [552.0, 480.0, 624.0],
            [666.0, 602.0, 729.0],
        ], dtype=np.float32)

    elif num_bands == 4:
        band_specs = np.array([
            [480.0, 421.0, 539.0],
            [552.0, 480.0, 624.0],
            [666.0, 602.0, 729.0],
            [803.0, 713.0, 893.0],
        ], dtype=np.float32)

    elif num_bands == 8:
        band_specs = np.array([
            [427.0, 396.0, 458.0],
            [478.0, 442.0, 515.0],
            [546.0, 506.0, 586.0],
            [608.0, 584.0, 632.0],
            [659.0, 624.0, 694.0],
            [724.0, 699.0, 749.0],
            [833.0, 765.0, 901.0],
            [949.0, 856.0, 1043.0],
        ], dtype=np.float32)

    elif num_bands == 16:
        band_specs = np.array([
            [426.0, 397.0, 454.0],
            [481.0, 445.0, 517.0],
            [547.0, 507.0, 586.0],
            [605.0, 580.0, 629.0],
            [661.0, 626.0, 696.0],
            [724.0, 698.0, 749.0],
            [832.0, 765.0, 899.0],
            [948.0, 857.0, 1039.0],
            [1210.0, 1184.0, 1235.0],
            [1572.0, 1546.0, 1598.0],
            [1661.0, 1636.0, 1686.0],
            [1730.0, 1702.0, 1759.0],
            [2164.0, 2137.0, 2191.0],
            [2203.0, 2174.0, 2232.0],
            [2260.0, 2228.0, 2292.0],
            [2329.0, 2285.0, 2373.0],
        ], dtype=np.float32)

    else:
        raise ValueError(
            f"Unsupported num_bands={num_bands}. "
            f"Supported values: {SUPPORTED_SRF_BAND_COUNTS}"
        )

    return band_specs


def normalize_srf_row(row: Array) -> Array:
    """Normalize one SRF row so its weights sum to 1."""
    row = np.asarray(row, dtype=np.float32)
    total = float(row.sum())
    if total <= 1e-12:
        raise ValueError("SRF row sum is too small to normalize safely")
    return (row / total).astype(np.float32)


def build_gaussian_srf(
    num_hsi_bands: int,
    band_specs: Array,
    fwhm_factor: float = 4.2,
) -> tuple[Array, Array]:
    """Build a Gaussian-approximated SRF matrix from band specifications.

    Parameters
    ----------
    num_hsi_bands:
        Number of hyperspectral bands in the input HSI.
    band_specs:
        Array of shape `(M, 3)` containing `[center_nm, low_nm, high_nm]`.
    fwhm_factor:
        Factor used to approximate standard deviation from band width.

    Returns
    -------
    srf:
        SRF matrix of shape `(M, L)`, where `L = num_hsi_bands`.
    wavelengths:
        Auto-generated wavelength grid of shape `(L,)`.
    """
    if not isinstance(num_hsi_bands, int) or num_hsi_bands <= 0:
        raise ValueError(f"num_hsi_bands must be a positive integer, got {num_hsi_bands!r}")
    if fwhm_factor <= 0:
        raise ValueError(f"fwhm_factor must be > 0, got {fwhm_factor}")

    band_specs = np.asarray(band_specs, dtype=np.float32)
    if band_specs.ndim != 2 or band_specs.shape[1] != 3:
        raise ValueError(
            f"band_specs must have shape (M, 3), got {band_specs.shape}"
        )

    min_edge = float(band_specs[:, 1].min())
    max_edge = float(band_specs[:, 2].max())
    wavelengths = np.linspace(min_edge, max_edge, num_hsi_bands, dtype=np.float32)

    num_msi_bands = band_specs.shape[0]
    srf = np.zeros((num_msi_bands, num_hsi_bands), dtype=np.float32)

    for i in range(num_msi_bands):
        center_nm, low_nm, high_nm = band_specs[i]
        std = (high_nm - low_nm) / fwhm_factor
        std = max(float(std), 1e-12)

        row = np.exp(-0.5 * ((wavelengths - center_nm) / std) ** 2)
        srf[i] = normalize_srf_row(row)

    return srf.astype(np.float32), wavelengths.astype(np.float32)


def validate_srf_matrix(srf: Array, num_hsi_bands: int | None = None) -> Array:
    """Validate an SRF matrix of shape `(M, L)`."""
    srf = np.asarray(srf, dtype=np.float32)

    if srf.ndim != 2:
        raise ValueError(f"SRF must be 2D with shape (M, L), got {srf.shape}")
    if not np.all(np.isfinite(srf)):
        raise ValueError("SRF contains non-finite values")
    if np.any(srf < 0):
        raise ValueError("SRF must be non-negative")

    row_sums = srf.sum(axis=1)
    if np.any(row_sums <= 1e-12):
        raise ValueError("One or more SRF rows have near-zero sum")

    if num_hsi_bands is not None and srf.shape[1] != num_hsi_bands:
        raise ValueError(
            f"SRF expects {srf.shape[1]} HSI bands but num_hsi_bands={num_hsi_bands}"
        )

    srf = srf / row_sums[:, None]
    return srf.astype(np.float32)


def apply_srf_matrix(hyper_image: Array, srf: Array) -> Array:
    """Apply an SRF matrix to an HSI.

    Parameters
    ----------
    hyper_image:
        Input HSI with shape `(H, W, L)`.
    srf:
        SRF matrix with shape `(M, L)`.

    Returns
    -------
    np.ndarray
        Output MSI with shape `(H, W, M)`.
    """
    hyper_image = np.asarray(hyper_image, dtype=np.float32)
    if hyper_image.ndim != 3:
        raise ValueError(f"Expected HSI shape (H, W, L), got {hyper_image.shape}")

    _, _, num_hsi_bands = hyper_image.shape
    srf = validate_srf_matrix(srf, num_hsi_bands=num_hsi_bands)

    msi = np.tensordot(hyper_image, srf.T, axes=([2], [0])).astype(np.float32)
    return msi