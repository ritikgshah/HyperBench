# Visualization

HyperBench provides built-in visualization utilities to inspect hyperspectral
data, degradations, and intermediate representations.

These tools are designed for:

- understanding data quality
- debugging pipelines
- interpreting results
- exploring spectral behavior

---

## Overview

Visualization functions operate directly on NumPy arrays and are intended for
use in notebooks or interactive environments.

---

## Data Statistics

### print_data_stats

Prints summary statistics of a hyperspectral image.

```python
from hyperbench import print_data_stats

print_data_stats(gt_hsi)
```

Displays:
- shape
- dtype
- min, max, mean

---

## Hyperspectral Visualization

### visualize_hsi

Displays an RGB composite of a hyperspectral image.

```python
from hyperbench import visualize_hsi

visualize_hsi(gt_hsi)
```

This selects representative bands to form an RGB image.

---

## Band Visualization

### visualize_band

Displays a single spectral band.

```python
from hyperbench import visualize_band

visualize_band(gt_hsi, band_index=10)
```

Useful for inspecting spatial detail at a specific wavelength.

---

## Multispectral Visualization

### visualize_multispectral_with_srf

Displays an MSI image alongside its spectral response.

```python
from hyperbench import visualize_multispectral_with_srf

visualize_multispectral_with_srf(hr_msi, srf)
```

Helps interpret how spectral bands are combined.

---

## PSF Visualization

### visualize_psfs

Displays multiple PSFs.

```python
from hyperbench import visualize_psfs

visualize_psfs(
    psf_names=["gaussian", "moffat", "airy"],
    sigma=3.4,
    kernel_radius=7,
)
```

Each PSF is plotted as an image with a color scale.

---

## Spectral Plots

### plot_spectra

Plots spectral signatures for selected pixels.

```python
from hyperbench import plot_spectra

plot_spectra(gt_hsi, coordinates=[(50, 50), (100, 100)])
```

Useful for:
- analyzing spectral variation
- comparing reconstructed vs ground truth spectra

---

## Typical Workflow

```python
from hyperbench import (
    print_data_stats,
    visualize_hsi,
    visualize_band,
    visualize_psfs,
)

print_data_stats(gt_hsi)

visualize_hsi(gt_hsi)

visualize_band(gt_hsi, band_index=20)

visualize_psfs(psf_names=["gaussian", "moffat"])
```

---

## Notes

- Visualization functions require matplotlib
- Inputs should be normalized for best results
- These utilities are intended for interactive use (e.g., Jupyter notebooks)

---

## Summary

The visualization module provides:

- quick inspection of hyperspectral data
- visual understanding of degradations
- tools for debugging and analysis

These functions complement the benchmarking pipeline by making results easier to interpret.
