# Model Integration

This guide explains how to integrate real-world hyperspectral super-resolution
models into HyperBench.

HyperBench is designed to be model-agnostic. Any method can be used as long as
it exposes a compatible pipeline interface.

---

## Integration Overview

To integrate a model with HyperBench, you need to:

1. Wrap your model inside a `run_pipeline(...)` function  
2. Ensure it accepts HyperBench inputs  
3. Return a prediction (and optionally stats)  
4. Use a PipelineAdapter or CLI to run experiments  

---

## Step 1: Define `run_pipeline(...)`

Your model must expose:

```python
def run_pipeline(HR_MSI, LR_HSI, srf, psf=None, metadata=None):
    return prediction
```

or:

```python
return prediction, stats
```

---

## Step 2: Understand Inputs

Your pipeline will receive:

- HR_MSI → (H, W, C_msi)
- LR_HSI → (H/r, W/r, C_hsi)
- srf → (C_msi, C_hsi)

Optional:
- psf
- metadata (dictionary with experiment info)

---

## Step 3: Handle Framework Conversion

HyperBench can pass inputs as:

- NumPy arrays
- TensorFlow tensors
- PyTorch tensors

You should write your model assuming the selected backend.

Example (TensorFlow):

```python
import tensorflow as tf

def run_pipeline(HR_MSI, LR_HSI, srf, psf=None, metadata=None):
    HR_MSI = tf.convert_to_tensor(HR_MSI)
    LR_HSI = tf.convert_to_tensor(LR_HSI)

    # model logic here

    return output
```

---

## Step 4: Ensure Output Format

Your model must return:

- shape: (H, W, C_hsi)
- any supported backend (NumPy / TF / Torch)

HyperBench will:
- convert to NumPy
- validate shape
- apply clipping if enabled

---

## Step 5: Optional Model Statistics

You may return additional stats:

```python
return prediction, {
    "num_parameters": 12_000_000,
    "flops": 3.5e9,
    "inference_time_sec": 0.1
}
```

These will be written to the CSV output.

---

## Step 6: Run with PipelineAdapter

```python
from hyperbench.adapters import PipelineAdapter

adapter = PipelineAdapter(
    pipeline_module="models/my_model.py",
    method_name="MyModel",
    input_backend="tensorflow",
    output_backend="tensorflow",
    device="auto",
)
```

---

## Step 7: Run Benchmark

Use either:

- Python API (`run_benchmark`)
- CLI (`hyperbench run`)

---

## Example: Minimal Pipeline

```python
def run_pipeline(HR_MSI, LR_HSI, srf, psf=None, metadata=None):
    import numpy as np

    scale = metadata.get("downsample_ratio", 4)

    # naive upsampling baseline
    return np.repeat(np.repeat(LR_HSI, scale, axis=0), scale, axis=1)
```

---

## Best Practices

- Keep all model logic inside `run_pipeline`
- Do not modify inputs in-place
- Ensure output shape matches ground truth
- Use metadata for dynamic behavior
- Return stats for benchmarking performance

---

## Common Pitfalls

- returning incorrect shape
- forgetting to handle batch dimension
- mixing frameworks unintentionally
- not matching backend selection

---

## Summary

To integrate a model into HyperBench:

1. Implement `run_pipeline(...)`
2. Accept HyperBench inputs
3. Return reconstructed HR-HSI
4. Optionally return stats
5. Run using adapter or CLI

This design ensures maximum flexibility while maintaining a consistent
benchmarking workflow.
