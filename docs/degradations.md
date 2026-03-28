# Degradations

This document describes the synthetic degradation pipeline in HyperBench.

Degradations are used to generate the two inputs required for fusion-based
hyperspectral super-resolution (HSR):

- Low-Resolution Hyperspectral Image (LR-HSI)
- High-Resolution Multispectral Image (HR-MSI)

Both are derived from a ground truth hyperspectral image (GT HSI).

---

## Overview

HyperBench implements two types of degradation:

1. Spatial degradation → produces LR-HSI
2. Spectral degradation → produces HR-MSI

These degradations simulate real imaging conditions.

---

## Spatial Degradation (LR-HSI)

Spatial degradation reduces spatial resolution while preserving spectral information.

It consists of:

1. Convolution with a Point Spread Function (PSF)
2. Downsampling by a factor r
3. Optional noise injection

---

### Function

```python
from hyperbench import spatial_degradation

lr_hsi = spatial_degradation(
    gt_hsi,
    psf,
    downsample_ratio=8,
    snr_db=30.0,
)
```

---

### Inputs

- gt_hsi: (H, W, B)
- psf: (k, k)
- downsample_ratio: integer (e.g., 4, 8, 16, 32)
- snr_db: noise level in decibels

---

### Output

- lr_hsi: (H / r, W / r, B)

---

## Spectral Degradation (HR-MSI)

Spectral degradation reduces spectral resolution while preserving spatial resolution.

It consists of:

1. Applying a Spectral Response Function (SRF)
2. Optional noise injection

---

### Function

```python
from hyperbench import spectral_degradation

hr_msi, srf = spectral_degradation(
    gt_hsi,
    msi_band_count=4,
    snr_db=40.0,
)
```

---

### Inputs

- gt_hsi: (H, W, B)
- msi_band_count: number of output spectral bands
- snr_db: noise level

---

### Outputs

- hr_msi: (H, W, C)
- srf: (C, B)

---

## Point Spread Functions (PSFs)

PSFs simulate spatial blur introduced by imaging systems.

---

### Creating a PSF

```python
from hyperbench import make_psf

psf = make_psf("gaussian", sigma=3.4, kernel_radius=7)
```

---

### Supported PSFs

- gaussian
- kolmogorov
- airy
- moffat
- sinc
- lorentzian2
- hermite
- parabolic
- gabor
- delta

---

### Parameters

- sigma: controls blur strength
- kernel_radius: determines kernel size

---

## Spectral Response Functions (SRFs)

SRFs define how hyperspectral bands are combined into multispectral bands.

---

### Built-in SRF Generation

```python
from hyperbench.degradations.srf import build_gaussian_srf, get_srf_band_specs

band_specs = get_srf_band_specs(4)

srf, wavelengths = build_gaussian_srf(
    num_hsi_bands=gt_hsi.shape[2],
    band_specs=band_specs,
)
```

---

### Supported MSI Band Counts

- 3
- 4
- 8
- 16

---

## Noise Model

Noise can be added to both degradations.

---

### SNR (Signal-to-Noise Ratio)

- Higher SNR → less noise
- Lower SNR → more noise

Typical values:
- 40 dB → low noise
- 30 dB → moderate noise
- 20 dB → high noise

---

## Normalization

It is recommended to normalize input data before applying degradations:

```python
from hyperbench import normalize_image

gt_hsi = normalize_image(gt_hsi)
```

---

## Typical Workflow

```python
psf = make_psf("gaussian", sigma=3.4, kernel_radius=7)

lr_hsi = spatial_degradation(gt_hsi, psf, downsample_ratio=8, snr_db=30.0)

hr_msi, srf = spectral_degradation(gt_hsi, msi_band_count=4, snr_db=40.0)
```

---

## Custom Degradation Cases

HyperBench allows specifying exact degradation combinations via configuration files.

Example:

- (r, c) = (4, 4)
- (r, c) = (8, 4)
- (r, c) = (8, 8)

This avoids unnecessary grid expansion and enables targeted experiments.

---

## Summary

The degradation module is responsible for:

- generating realistic LR-HSI inputs
- generating realistic HR-MSI inputs
- simulating sensor behavior
- enabling controlled benchmarking scenarios

This is the core of HyperBench’s synthetic evaluation pipeline.
