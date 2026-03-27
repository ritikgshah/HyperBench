# Author: Ritik Shah

"""Visualization and inspection helpers for HyperBench."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from spectral import get_rgb
except ImportError:  # pragma: no cover
    get_rgb = None

from hyperbench.degradations import AVAILABLE_PSFS, make_psf


Array = np.ndarray


def _validate_hwc(image: Array, name: str = "image") -> Array:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError("{} must have shape (H, W, B), got {}".format(name, image.shape))
    if not np.all(np.isfinite(image)):
        raise ValueError("{} contains non-finite values".format(name))
    return image


def print_data_stats(image: Array, name: str = "Data") -> None:
    image = _validate_hwc(image, name=name)

    print("{} type: {}".format(name, type(image)))
    print("{} dtype: {}".format(name, image.dtype))
    print("{} shape (H, W, B): {}".format(name, image.shape))
    print("{} min: {}".format(name, float(image.min())))
    print("{} max: {}".format(name, float(image.max())))
    print("{} mean: {}".format(name, float(image.mean())))
    print("{} std: {}".format(name, float(image.std())))


def _require_spectral() -> None:
    if get_rgb is None:
        raise ImportError(
            "These visualization helpers require the optional dependency "
            "'spectral'. Install it with: pip install spectral"
        )


def visualize_hsi(image: Array, bands: Tuple[int, int, int], title: str = "HSI") -> None:
    _require_spectral()
    image = _validate_hwc(image, name="image")

    img_rgb = get_rgb(image, bands)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def visualize_band(
    image: Array,
    band_idx: int,
    title: Optional[str] = None,
    cmap: str = "viridis",
) -> None:
    image = _validate_hwc(image, name="image")
    if not (0 <= band_idx < image.shape[2]):
        raise ValueError(
            "band_idx {} out of range for image with {} bands".format(
                band_idx, image.shape[2]
            )
        )

    plt.figure(figsize=(6, 5))
    plt.imshow(image[:, :, band_idx], cmap=cmap)
    plt.title(title or "Band {}".format(band_idx))
    plt.colorbar()
    plt.axis("off")
    plt.show()


def visualize_multispectral_with_srf(
    ms_image: Array,
    bands: Optional[Tuple[int, int, int]],
    title: str,
    srf: Array,
    wavelengths: Array,
    band_specs: Array,
) -> None:
    image = _validate_hwc(ms_image, name="ms_image")
    srf = np.asarray(srf, dtype=np.float32)
    wavelengths = np.asarray(wavelengths, dtype=np.float32)
    band_specs = np.asarray(band_specs, dtype=np.float32)

    num_bands = image.shape[2]
    if num_bands not in {3, 4, 8, 16}:
        raise ValueError("HyperBench MSI visualization only supports 3, 4, 8, or 16 bands.")

    if bands is not None:
        _require_spectral()

    if num_bands == 3:
        band_labels = ["Blue", "Green", "Red"]
        band_colors = ["b", "g", "r"]
    elif num_bands == 4:
        band_labels = ["Blue", "Green", "Red", "NIR"]
        band_colors = ["b", "g", "r", "k"]
    elif num_bands == 8:
        band_labels = ["Coastal", "Blue", "Green", "Yellow", "Red", "Red Edge", "NIR1", "NIR2"]
        band_colors = ["b", "g", "r", "c", "m", "y", "k", "orange"]
    else:
        band_labels = [
            "Coastal Blue", "Blue", "Green", "Yellow",
            "Red", "Red Edge", "NIR1", "NIR2",
            "SWIR1", "SWIR2", "SWIR3", "SWIR4",
            "SWIR5", "SWIR6", "SWIR7", "SWIR8",
        ]
        band_colors = [
            "b", "g", "r", "c", "m", "y", "k", "orange",
            "purple", "brown", "pink", "gray",
            "olive", "navy", "teal", "maroon",
        ]

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    if bands is None:
        axs[0].imshow(image[:, :, 0], cmap="gray")
        axs[0].set_title(title)
    else:
        img_rgb = get_rgb(image, bands)
        axs[0].imshow(img_rgb)
        axs[0].set_title(title)
    axs[0].axis("off")

    for i in range(num_bands):
        center_nm = band_specs[i, 0]
        axs[1].plot(
            wavelengths,
            srf[i, :],
            label="{} (Center: {:.1f} nm)".format(band_labels[i], center_nm),
            color=band_colors[i],
        )

    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].set_ylabel("Spectral Response")
    axs[1].set_title("Spectral Response Functions (SRF)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_spectra(
    ground_truth_hsi: Array,
    predicted_hsi: Array,
    x: int,
    y: int,
    gt_label: str = "Ground Truth",
    pred_label: str = "Prediction",
) -> None:
    gt = _validate_hwc(ground_truth_hsi, name="ground_truth_hsi")
    pred = _validate_hwc(predicted_hsi, name="predicted_hsi")

    if gt.shape != pred.shape:
        raise ValueError("Shape mismatch: {} vs {}".format(gt.shape, pred.shape))
    if not (0 <= x < gt.shape[1] and 0 <= y < gt.shape[0]):
        raise ValueError("Pixel (x={}, y={}) is out of bounds for shape {}".format(x, y, gt.shape))

    gt_spectrum = gt[y, x, :]
    pred_spectrum = pred[y, x, :]

    plt.figure(figsize=(10, 5))
    plt.plot(range(gt.shape[2]), gt_spectrum, label=gt_label)
    plt.plot(range(pred.shape[2]), pred_spectrum, label=pred_label)
    plt.xlabel("Spectral band")
    plt.ylabel("Reflectance / normalized intensity")
    plt.title("Spectra at pixel (x={}, y={})".format(x, y))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_psfs(
    psf_names,
    sigma: float = 3.4,
    kernel_radius: int = 7,
    cmap: str = "viridis",
    figsize: Optional[Tuple[int, int]] = None,
    ncols: int = 5,
    title: str = "Point Spread Functions (PSFs)",
) -> None:
    if not psf_names:
        raise ValueError("psf_names must contain at least one PSF name")

    normalized_names = [name.strip().lower() for name in psf_names]
    invalid = [name for name in normalized_names if name not in AVAILABLE_PSFS]
    if invalid:
        raise ValueError(
            "Unsupported PSF name(s): {}. Supported PSFs: {}".format(
                invalid, sorted(AVAILABLE_PSFS.keys())
            )
        )

    psf_display_names = {
        "gaussian": "Gaussian PSF",
        "kolmogorov": "Kolmogorov PSF",
        "airy": "Airy PSF",
        "moffat": "Moffat PSF",
        "sinc": "Sinc PSF",
        "lorentzian2": "Lorentzian Squared PSF",
        "hermite": "Hermite PSF",
        "parabolic": "Parabolic PSF",
        "gabor": "Gabor PSF",
        "delta": "Delta Function PSF",
    }

    psfs = [
        (make_psf(name, sigma=sigma, kernel_radius=kernel_radius), psf_display_names[name])
        for name in normalized_names
    ]

    n = len(psfs)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(float(n) / float(ncols)))

    if figsize is None:
        figsize = (4 * ncols, int(3.5 * nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if isinstance(axes, np.ndarray):
        axes = axes.reshape(-1)
    else:
        axes = np.array([axes])

    fig.suptitle(title, fontsize=16)

    for i, (psf, label) in enumerate(psfs):
        ax = axes[i]
        im = ax.imshow(psf, cmap=cmap)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(len(psfs), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()