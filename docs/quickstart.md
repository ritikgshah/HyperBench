# Quickstart

This guide walks through a minimal end-to-end workflow using HyperBench.

You will:
1. Load a hyperspectral scene
2. Normalize it
3. Generate one synthetic degradation
4. Run a simple pipeline
5. Compute metrics

---

## 1. Installation

```bash
pip install hyperbench
```

---

## 2. Load a hyperspectral scene

```python
from hyperbench import load_hsi

scene = load_hsi("data/scene.mat", key="dc")
print(scene.shape)  # (H, W, B)
```

---

## 3. Normalize the scene

```python
from hyperbench import normalize_image

gt_hsi = normalize_image(scene)
```

---

## 4. Create degradations

```python
from hyperbench import make_psf, spatial_degradation, spectral_degradation

psf = make_psf("gaussian", sigma=3.4, kernel_radius=7)

lr_hsi = spatial_degradation(gt_hsi, psf, downsample_ratio=8, snr_db=30.0)

hr_msi, srf = spectral_degradation(gt_hsi, msi_band_count=4, snr_db=40.0)
```

---

## 5. Define a simple pipeline

```python
def run_pipeline(HR_MSI, LR_HSI, srf, psf=None, metadata=None):
    # Dummy example: return LR upsampled to HR size
    import numpy as np
    return np.repeat(np.repeat(LR_HSI, 8, axis=0), 8, axis=1)
```

---

## 6. Run the pipeline

```python
prediction = run_pipeline(hr_msi, lr_hsi, srf)
```

---

## 7. Evaluate metrics

```python
from hyperbench import evaluate_metrics

metrics = evaluate_metrics(gt_hsi, prediction)
print(metrics)
```

---

## Summary

You have now:
- loaded a scene
- generated degradations
- run a pipeline
- computed metrics

Next steps:
- use a real model
- run full benchmarks via CLI
- explore configuration files
