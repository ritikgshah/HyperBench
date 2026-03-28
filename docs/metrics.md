# Metrics

This document describes the evaluation metrics used in HyperBench to assess
the quality of reconstructed hyperspectral images.

All metrics are computed between:

- Ground Truth HSI (GT)
- Predicted HSI (model output)

Both inputs are expected to be:

- normalized (typically to [0, 1])
- of the same shape: (H, W, B)

---

## Overview

HyperBench provides a unified interface:

```python
from hyperbench import evaluate_metrics

metrics = evaluate_metrics(gt_hsi, prediction)
```

This returns a dictionary containing all supported metrics.

---

## Supported Metrics

### RMSE (Root Mean Squared Error)

Measures the average reconstruction error.

- Lower is better
- Sensitive to large errors

---

### PSNR (Peak Signal-to-Noise Ratio)

Measures signal quality relative to noise.

- Higher is better
- Common in image reconstruction tasks

---

### SSIM (Structural Similarity Index)

Measures perceived structural similarity.

- Range: [-1, 1]
- Higher is better

---

### UIQI (Universal Image Quality Index)

Measures structural distortion and correlation.

- Range: [-1, 1]
- Higher is better

---

### ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse)

Measures relative global error across bands.

- Lower is better
- Common in remote sensing literature

---

### SAM (Spectral Angle Mapper)

Measures spectral similarity between vectors.

- Lower is better
- Invariant to intensity scaling

---

## Input Requirements

Both inputs must:

- have identical shape (H, W, B)
- be NumPy arrays (or convertible)
- be normalized consistently

If shapes do not match, an error is raised.

---

## Example

```python
from hyperbench import evaluate_metrics

metrics = evaluate_metrics(gt_hsi, prediction)

for name, value in metrics.items():
    print(f"{name}: {value}")
```

---

## Output Format

Example output:

```python
{
    "RMSE": 0.012,
    "PSNR": 38.5,
    "SSIM": 0.97,
    "UIQI": 0.95,
    "ERGAS": 2.1,
    "SAM": 0.03
}
```

---

## Interpretation

| Metric | Better Value |
|------|-------------|
| RMSE | Lower |
| PSNR | Higher |
| SSIM | Higher |
| UIQI | Higher |
| ERGAS | Lower |
| SAM | Lower |

---

## Notes

- Metrics assume normalized data
- Clipping may be applied before evaluation
- Different metrics emphasize different aspects:
  - spatial quality (PSNR, SSIM)
  - spectral fidelity (SAM, ERGAS)

---

## Summary

The metrics module provides:

- standardized evaluation across experiments
- consistent metric computation
- easy integration into benchmarking pipelines

These metrics enable fair comparison between different HSR methods.
