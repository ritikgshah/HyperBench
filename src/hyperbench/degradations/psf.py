# Author: Ritik Shah

"""Point spread function (PSF) utilities for HyperBench."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.special import kv


Array = np.ndarray


def _validate_kernel_radius(kernel_radius: int) -> None:
    if not isinstance(kernel_radius, int) or kernel_radius < 0:
        raise ValueError(f"kernel_radius must be a non-negative integer, got {kernel_radius!r}")


def _validate_sigma(sigma: float) -> None:
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")


def _make_grid(kernel_radius: int) -> tuple[Array, Array, Array]:
    """Create centered meshgrid and radial distance map."""
    _validate_kernel_radius(kernel_radius)
    x = np.linspace(-kernel_radius, kernel_radius, 2 * kernel_radius + 1, dtype=np.float32)
    y = np.linspace(-kernel_radius, kernel_radius, 2 * kernel_radius + 1, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)
    r = np.sqrt(x_grid**2 + y_grid**2)
    return x_grid, y_grid, r


def _normalize_kernel(psf: Array, *, allow_signed: bool = True) -> Array:
    """Normalize a PSF kernel to sum to 1.

    Parameters
    ----------
    psf:
        Input kernel.
    allow_signed:
        Whether to allow negative-valued kernels. Some analytical PSFs
        such as Gabor-like forms can contain negative values.

    Returns
    -------
    np.ndarray
        Normalized float32 kernel.

    Raises
    ------
    ValueError
        If the kernel is invalid or has near-zero sum.
    """
    psf = np.asarray(psf, dtype=np.float32)

    if psf.ndim != 2:
        raise ValueError(f"PSF must be 2D, got shape {psf.shape}")

    if not np.all(np.isfinite(psf)):
        raise ValueError("PSF contains non-finite values")

    if not allow_signed and np.any(psf < 0):
        raise ValueError("PSF contains negative values but allow_signed=False")

    total = float(psf.sum())
    if abs(total) < 1e-12:
        raise ValueError("PSF sum is too close to zero to normalize safely")

    return (psf / total).astype(np.float32)


def gaussian_psf(sigma: float, kernel_radius: int) -> Array:
    """Generate a Gaussian PSF."""
    _validate_sigma(sigma)
    x_grid, y_grid, _ = _make_grid(kernel_radius)
    psf = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    return _normalize_kernel(psf, allow_signed=False)


def kolmogorov_psf(sigma: float, kernel_radius: int) -> Array:
    """Generate a Kolmogorov-style PSF."""
    _validate_sigma(sigma)
    _, _, r = _make_grid(kernel_radius)
    psf = np.exp(-((r / sigma) ** (5.0 / 3.0)))
    return _normalize_kernel(psf, allow_signed=False)


def airy_psf(sigma: float, kernel_radius: int) -> Array:
    """Generate a Airy-style PSF."""
    _validate_sigma(sigma)
    _, _, r = _make_grid(kernel_radius)

    z = (2.33811 * r) / sigma
    psf = np.empty_like(z, dtype=np.float32)

    nonzero = r != 0
    psf[nonzero] = ((2 * kv(1, z[nonzero]) / z[nonzero]) ** 2).astype(np.float32)
    psf[~nonzero] = 1.0

    return _normalize_kernel(psf, allow_signed=False)


def moffat_psf(sigma: float, kernel_radius: int, beta: float = 3.5) -> Array:
    """Generate a Moffat PSF."""
    _validate_sigma(sigma)
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")

    _, _, r = _make_grid(kernel_radius)
    psf = (1.0 + (r / sigma) ** 2) ** (-beta)
    return _normalize_kernel(psf, allow_signed=False)


def sinc_psf(sigma: float, kernel_radius: int) -> Array:
    """Generate a radial sinc PSF."""
    _validate_sigma(sigma)
    _, _, r = _make_grid(kernel_radius)
    psf = np.sinc(r / sigma)
    return _normalize_kernel(psf, allow_signed=True)


def lorentzian_squared_psf(sigma: float, kernel_radius: int) -> Array:
    """Generate a Lorentzian-squared PSF."""
    _validate_sigma(sigma)
    _, _, r = _make_grid(kernel_radius)
    psf = 1.0 / (1.0 + (r / sigma) ** 4)
    return _normalize_kernel(psf, allow_signed=False)


def hermite_psf(sigma: float, kernel_radius: int, order: int = 1) -> Array:
    """Generate a Hermite-modulated PSF."""
    _validate_sigma(sigma)
    if order < 0 or not isinstance(order, int):
        raise ValueError(f"order must be a non-negative integer, got {order!r}")

    _, _, r = _make_grid(kernel_radius)
    psf = np.exp(-(r**2) / (2 * sigma**2)) * hermval(r / sigma, [0] * order + [1])
    return _normalize_kernel(psf, allow_signed=True)


def parabolic_psf(sigma: float, kernel_radius: int) -> Array:
    """Generate a compact parabolic PSF."""
    _validate_sigma(sigma)
    _, _, r = _make_grid(kernel_radius)
    psf = np.maximum(0.0, 1.0 - (r / sigma) ** 2)
    return _normalize_kernel(psf, allow_signed=False)


def gabor_psf(
    sigma: float,
    kernel_radius: int,
    theta: float = 0.0,
    lambd: float = 1.0,
    gamma: float = 0.5,
    psi: float = 0.0,
) -> Array:
    """Generate a Gabor-like PSF."""
    _validate_sigma(sigma)
    if lambd <= 0:
        raise ValueError(f"lambd must be > 0, got {lambd}")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    x_grid, y_grid, _ = _make_grid(kernel_radius)
    x_theta = x_grid * np.cos(theta) + y_grid * np.sin(theta)
    y_theta = -x_grid * np.sin(theta) + y_grid * np.cos(theta)

    psf = np.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / (sigma**2))
    psf *= np.cos(2 * np.pi * x_theta / lambd + psi)

    return _normalize_kernel(psf, allow_signed=True)


def delta_function_psf(sigma: float, kernel_radius: int) -> Array:
    """Generate a delta-function PSF.

    Notes
    -----
    The `sigma` argument is accepted for API consistency and ignored.
    """
    _ = sigma
    _validate_kernel_radius(kernel_radius)

    size = 2 * kernel_radius + 1
    psf = np.zeros((size, size), dtype=np.float32)
    psf[kernel_radius, kernel_radius] = 1.0
    return psf


AVAILABLE_PSFS: dict[str, Callable[..., Array]] = {
    "gaussian": gaussian_psf,
    "kolmogorov": kolmogorov_psf,
    "airy": airy_psf,
    "moffat": moffat_psf,
    "sinc": sinc_psf,
    "lorentzian2": lorentzian_squared_psf,
    "hermite": hermite_psf,
    "parabolic": parabolic_psf,
    "gabor": gabor_psf,
    "delta": delta_function_psf,
}


def make_psf(name: str, sigma: float, kernel_radius: int, **kwargs) -> Array:
    """Construct a PSF by name.

    Parameters
    ----------
    name:
        PSF name. Must be one of `AVAILABLE_PSFS`.
    sigma:
        Spread parameter for the PSF.
    kernel_radius:
        Radius of the PSF support. Output shape will be
        `(2 * kernel_radius + 1, 2 * kernel_radius + 1)`.
    **kwargs:
        Additional PSF-specific keyword arguments.

    Returns
    -------
    np.ndarray
        PSF kernel of shape `(K, K)` with float32 dtype.
    """
    key = name.strip().lower()
    if key not in AVAILABLE_PSFS:
        raise ValueError(f"Unknown PSF '{name}'. Available PSFs: {sorted(AVAILABLE_PSFS)}")

    psf = AVAILABLE_PSFS[key](sigma=sigma, kernel_radius=kernel_radius, **kwargs)
    return np.asarray(psf, dtype=np.float32)