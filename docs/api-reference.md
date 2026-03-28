# API Reference

This document provides a reference for the public API of HyperBench.

It summarizes the primary functions, classes, and modules available to users.
For detailed usage examples, refer to the corresponding documentation pages.

---

## IO

### load_hsi

Load a hyperspectral image from a `.mat` file.

**Usage**
```python
from hyperbench import load_hsi

scene = load_hsi(path, key=None)
```

**Parameters**
- path: str  
- key: str (optional)

**Returns**
- NumPy array of shape (H, W, B)

---

## Preprocessing

### normalize_image

Normalize a hyperspectral image.

**Usage**
```python
from hyperbench import normalize_image

normalized = normalize_image(image)
```

---

## Degradations

### make_psf

Create a point spread function.

**Usage**
```python
psf = make_psf(name, sigma, kernel_radius)
```

---

### spatial_degradation

Generate LR-HSI.

**Usage**
```python
lr_hsi = spatial_degradation(gt_hsi, psf, downsample_ratio, snr_db)
```

---

### spectral_degradation

Generate HR-MSI and SRF.

**Usage**
```python
hr_msi, srf = spectral_degradation(gt_hsi, msi_band_count, snr_db)
```

---

## Metrics

### evaluate_metrics

Compute evaluation metrics.

**Usage**
```python
metrics = evaluate_metrics(gt_hsi, prediction)
```

---

## Visualization

### print_data_stats

Print summary statistics.

---

### visualize_hsi

Display RGB composite.

---

### visualize_band

Display a single spectral band.

---

### visualize_multispectral_with_srf

Display MSI and SRF relationship.

---

### visualize_psfs

Display PSFs.

---

### plot_spectra

Plot spectral signatures.

---

## Adapters

### PipelineAdapter

Adapter for `run_pipeline(...)`-based models.

---

### CallableAdapter

Adapter for direct Python callables.

---

## Benchmarking

### BenchmarkConfig

Configuration object for experiments.

---

### run_benchmark

Execute benchmark using an adapter and config.

---

## CLI

### hyperbench run

Run experiments via command line.

---

## Types and Utilities

Includes shared types, exceptions, and helper utilities for:

- device handling
- framework conversion
- shape management

---

## Notes

- All inputs are expected to be NumPy arrays unless specified
- Shapes follow (H, W, B) convention
- Outputs must match ground truth shape
- Clipping may be applied before metric computation

---

## Summary

The HyperBench API is organized into:

- IO  
- Preprocessing  
- Degradations  
- Metrics  
- Visualization  
- Adapters  
- Benchmarking  

Each component plays a role in the full evaluation pipeline.
