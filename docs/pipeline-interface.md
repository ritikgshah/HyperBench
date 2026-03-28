# Pipeline Interface

This document defines how to integrate a model with HyperBench.

HyperBench is model-agnostic. Any method can be used as long as it follows the
expected pipeline interface.

---

## Required Interface

A model must expose a function with the following signature:

```python
def run_pipeline(HR_MSI, LR_HSI, srf, psf=None, metadata=None):
    return prediction
```

Optionally, the function may return:

```python
return prediction, stats
```

---

## Inputs

### HR_MSI
High-resolution multispectral image.

- Shape: (H, W, C_msi)
- Represents spatially high-resolution, spectrally degraded data

---

### LR_HSI
Low-resolution hyperspectral image.

- Shape: (H/r, W/r, C_hsi)
- Represents spectrally rich, spatially degraded data

---

### srf
Spectral Response Function.

- Shape: (C_msi, C_hsi)
- Maps hyperspectral bands to multispectral bands

---

### psf (optional)
Point Spread Function used during spatial degradation.

- Shape: (k, k)
- May be used by some models

---

### metadata (optional)
Dictionary containing additional information about the current benchmark case.

Examples:
- downsampling ratio
- noise levels
- wavelengths

---

## Outputs

### prediction (required)

The reconstructed high-resolution hyperspectral image.

- Expected shape: (H, W, C_hsi)

HyperBench will:
- validate shape
- convert type (NumPy / TensorFlow / PyTorch)
- apply clipping if enabled
- compute metrics

---

### stats (optional)

A dictionary containing model statistics.

Examples:
```python
stats = {
    "num_parameters": 12_000_000,
    "flops": 3.5e9,
    "gpu_memory_mb": 1024,
    "inference_time_sec": 0.12
}
```

If provided, these will be included in the CSV output.

---

## Supported Data Formats

HyperBench accepts:

- NumPy arrays
- TensorFlow tensors
- PyTorch tensors

Internally, all data is converted to a consistent representation.

---

## What HyperBench Handles

HyperBench automatically manages:

- input generation (LR-HSI, HR-MSI)
- normalization
- framework conversion
- output validation
- clipping policy
- metric computation
- CSV logging

---

## What the User Must Handle

The model pipeline is responsible for:

- implementing the reconstruction logic
- handling framework-specific operations
- ensuring correct output shape

---

## Example

```python
def run_pipeline(HR_MSI, LR_HSI, srf, psf=None, metadata=None):
    import numpy as np

    scale = metadata.get("downsample_ratio", 4)

    # Simple baseline: nearest neighbor upsampling
    upsampled = np.repeat(np.repeat(LR_HSI, scale, axis=0), scale, axis=1)

    return upsampled
```

---

## Best Practices

- Always return output with shape (H, W, C_hsi)
- Avoid modifying inputs in-place
- Use metadata when needed for scale-dependent logic
- Return stats when benchmarking model efficiency

---

## Summary

To integrate a model with HyperBench:

1. Implement `run_pipeline(...)`
2. Accept (HR_MSI, LR_HSI, srf)
3. Return a reconstructed HR-HSI
4. Optionally return model statistics

No additional framework-specific integration is required.
