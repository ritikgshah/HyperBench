# Author: Ritik Shah

"""Spectral degradation pipeline for HyperBench."""

from __future__ import annotations

import numpy as np

from .preprocessing import add_awgn_by_snr, normalize_image, validate_hsi
from .srf import apply_srf_matrix, build_gaussian_srf, get_srf_band_specs, validate_srf_matrix


Array = np.ndarray


def spectral_degradation(
    image: Array,
    snr_db: float,
    *,
    num_bands: int = 4,
    fwhm_factor: float = 4.2,
    user_srf: Array | None = None,
    normalize: bool = True,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    rng: np.random.Generator | None = None,
) -> tuple[Array, Array, Array | None, Array | None]:
    """Apply the full spectral degradation pipeline.

    Pipeline:
        normalize -> apply SRF -> add AWGN

    Parameters
    ----------
    image:
        Input HSI of shape `(H, W, L)`.
    snr_db:
        Desired SNR in dB for additive Gaussian noise.
    num_bands:
        Built-in MSI band count to use when `user_srf is None`.
    fwhm_factor:
        Width factor for Gaussian SRF approximation.
    user_srf:
        Optional custom SRF matrix of shape `(M, L)`.
        If provided, `num_bands` and built-in band specs are bypassed.
    normalize:
        Whether to normalize the HSI before degradation.
    lower_percentile, upper_percentile:
        Percentile clipping values used when `normalize=True`.
    rng:
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    degraded_msi:
        Spectrally degraded MSI of shape `(H, W, M)`.
    srf:
        SRF matrix of shape `(M, L)`.
    band_specs:
        Built-in band specifications `(M, 3)` if using a built-in SRF,
        else `None`.
    wavelengths:
        Auto-generated wavelength grid of shape `(L,)` if using a built-in SRF,
        else `None`.
    """
    work = validate_hsi(image)

    if normalize:
        work = normalize_image(
            work,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )

    _, _, num_hsi_bands = work.shape

    if user_srf is not None:
        srf = validate_srf_matrix(user_srf, num_hsi_bands=num_hsi_bands)
        band_specs = None
        wavelengths = None
    else:
        band_specs = get_srf_band_specs(num_bands)
        srf, wavelengths = build_gaussian_srf(
            num_hsi_bands=num_hsi_bands,
            band_specs=band_specs,
            fwhm_factor=fwhm_factor,
        )

    degraded_msi = apply_srf_matrix(work, srf)
    degraded_msi = add_awgn_by_snr(degraded_msi, snr_db, rng=rng)

    return (
        degraded_msi.astype(np.float32),
        srf.astype(np.float32),
        None if band_specs is None else band_specs.astype(np.float32),
        None if wavelengths is None else wavelengths.astype(np.float32),
    )