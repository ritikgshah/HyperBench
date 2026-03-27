# Author: Ritik Shah

"""Degradation primitives for HyperBench.

This package contains the building blocks used to generate synthetic
hyperspectral super-resolution benchmark inputs:

- Point spread functions (PSFs)
- Spectral response functions (SRFs)
- Shared preprocessing utilities
- Spatial degradation pipeline
- Spectral degradation pipeline
"""

from .preprocessing import add_awgn_by_snr, normalize_image, validate_hsi
from .psf import (
    AVAILABLE_PSFS,
    airy_psf,
    delta_function_psf,
    gabor_psf,
    gaussian_psf,
    hermite_psf,
    kolmogorov_psf,
    lorentzian_squared_psf,
    make_psf,
    moffat_psf,
    parabolic_psf,
    sinc_psf,
)
from .spatial import apply_psf, downsample_image, spatial_degradation
from .spectral import spectral_degradation
from .srf import (
    SUPPORTED_SRF_BAND_COUNTS,
    apply_srf_matrix,
    build_gaussian_srf,
    get_srf_band_specs,
    validate_srf_matrix,
)

__all__ = [
    "AVAILABLE_PSFS",
    "SUPPORTED_SRF_BAND_COUNTS",
    "validate_hsi",
    "normalize_image",
    "add_awgn_by_snr",
    "make_psf",
    "gaussian_psf",
    "kolmogorov_psf",
    "airy_psf",
    "moffat_psf",
    "sinc_psf",
    "lorentzian_squared_psf",
    "hermite_psf",
    "parabolic_psf",
    "gabor_psf",
    "delta_function_psf",
    "get_srf_band_specs",
    "build_gaussian_srf",
    "validate_srf_matrix",
    "apply_srf_matrix",
    "apply_psf",
    "downsample_image",
    "spatial_degradation",
    "spectral_degradation",
]