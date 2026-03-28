# Adapters

HyperBench uses adapters to standardize how models are executed within the
benchmarking pipeline.

Adapters act as an interface between:

- HyperBench (data generation, evaluation, logging)
- user-defined model pipelines

They ensure consistent handling of:

- input formats
- output formats
- device selection
- shape constraints

---

## Overview

An adapter wraps a model or function so that HyperBench can call it in a
uniform way across all experiments.

In most cases, users will interact with:

- PipelineAdapter (recommended)
- CallableAdapter (simple use cases)

---

## PipelineAdapter (Recommended)

The PipelineAdapter is designed for modern workflows where the model exposes
a `run_pipeline(...)` function.

### Usage

```python
from hyperbench.adapters import PipelineAdapter

adapter = PipelineAdapter(
    pipeline_module="models/spectralift_pipeline.py",
    method_name="SpectraLift",
    input_backend="tensorflow",
    output_backend="tensorflow",
    device="auto",
)
```

---

### What it does

The adapter:

1. Loads the pipeline module
2. Locates `run_pipeline(...)`
3. Converts inputs to the selected backend
4. Executes the pipeline
5. Converts outputs back to NumPy
6. Returns prediction (and optional stats)

---

## CallableAdapter

The CallableAdapter wraps a Python callable directly.

### Usage

```python
from hyperbench.adapters import CallableAdapter

def my_model(inputs):
    return prediction

adapter = CallableAdapter(my_model)
```

This is useful for quick testing or simple models.

---

## Input and Output Backends

Adapters support multiple frameworks:

- numpy
- tensorflow
- torch

### Input Backend

Defines how inputs are passed to the model.

### Output Backend

Defines expected output type.

If not specified, output backend defaults to input backend.

---

## Device Handling

Adapters support flexible device selection.

Options include:

- auto (default)
- cpu
- cuda (PyTorch)
- /GPU:0 (TensorFlow)

The adapter ensures that inputs are placed on the correct device.

---

## Shape Policy

Models often require inputs with specific spatial constraints.

HyperBench supports three policies:

### strict

- No modification
- Raises error if shapes are incompatible

---

### crop

- Crops inputs to satisfy divisibility constraints

---

### pad

- Pads inputs to satisfy constraints

---

## Shape Constraints

Adapters allow specifying:

- hr_multiple (divisibility for HR inputs)
- lr_multiple (divisibility for LR inputs)

These ensure compatibility with model architectures.

---

## Batch Dimension

Some models require batched inputs.

Use:

```python
add_batch_dim=True
```

This converts:

(H, W, C) → (1, H, W, C)

---

## Output Handling

After model execution, the adapter:

- converts output to NumPy
- removes batch dimension (if added)
- validates shape
- passes result to HyperBench for evaluation

---

## Stats Integration

If the pipeline returns:

```python
return prediction, stats
```

the adapter forwards `stats` to HyperBench, which adds them to the CSV output.

---

## Best Practices

- Use PipelineAdapter for real models
- Keep model logic inside `run_pipeline(...)`
- Let HyperBench handle conversion and validation
- Use shape policies only when necessary
- Return stats for performance benchmarking

---

## Summary

Adapters provide:

- a unified interface for all models
- framework-agnostic execution
- automatic data conversion
- device and shape handling

They are the bridge between user models and the HyperBench benchmarking system.
