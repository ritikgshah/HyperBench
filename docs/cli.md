# Command-Line Interface (CLI)

HyperBench provides a command-line interface for running benchmarks using
configuration files and external model pipelines.

The CLI enables fully reproducible and scriptable experiment workflows without
writing additional Python code.

---

## Basic Usage

The main entry point is:

```bash
hyperbench run --config config.yaml --pipeline-module path/to/model.py
```

---

## Command Overview

### Top-Level Command

```bash
hyperbench --help
```

Displays available commands.

---

### Run Command

```bash
hyperbench run --help
```

Runs a benchmark defined by a configuration file.

---

## Required Arguments

### --config

Path to a configuration file.

Supported formats:
- YAML (.yaml, .yml)
- JSON (.json)

Example:
```bash
--config configs/config.yaml
```

---

### --pipeline-module

Path to a Python module that defines the model pipeline.

The module must expose either:
- a `run_pipeline(...)` function, or
- an object containing this function

Example:
```bash
--pipeline-module models/spectralift_pipeline.py
```

---

## Optional Arguments

### --pipeline-name

Name of the pipeline object inside the module.

Use this if the module does not expose `run_pipeline` at the top level.

---

### --method-name

Name used to identify the method in output results.

Example:
```bash
--method-name SpectraLift
```

---

### --input-backend

Specifies how inputs are passed to the model.

Options:
- numpy
- tensorflow
- torch

Example:
```bash
--input-backend tensorflow
```

---

### --output-backend

Specifies the expected output type from the model.

Defaults to the same as input backend.

---

### --device

Controls where the model runs.

Examples:
- auto
- cpu
- cuda
- /GPU:0 (TensorFlow)

Example:
```bash
--device auto
```

---

### --shape-policy

Defines how input shapes are adjusted before being passed to the model.

Options:
- strict: no modification, raises error if incompatible
- crop: crops input to required dimensions
- pad: pads input to required dimensions

Example:
```bash
--shape-policy crop
```

---

### --hr-multiple

Required divisibility constraint for high-resolution inputs.

---

### --lr-multiple

Required divisibility constraint for low-resolution inputs.

---

### --add-batch-dim

Adds a batch dimension to inputs before passing them to the model.

Useful for frameworks that expect batched inputs.

---

### --no-device-summary

Disables printing of device and backend information at runtime.

---

## Example Command

```bash
hyperbench run   --config configs/config.yaml   --pipeline-module models/spectralift_pipeline.py   --method-name SpectraLift   --input-backend tensorflow   --output-backend tensorflow   --device auto   --shape-policy strict   --hr-multiple 1   --lr-multiple 1   --add-batch-dim
```

---

## Execution Flow

When the CLI is executed:

1. The configuration file is loaded
2. The model pipeline is imported
3. Degradation cases are generated
4. For each case:
   - inputs are created
   - inputs are converted to the selected backend
   - the pipeline is executed
   - outputs are converted back to NumPy
   - optional clipping is applied
   - metrics are computed
   - results are written to CSV

---

## Error Handling

- If `fail_fast` is enabled in the config, execution stops on first error
- Otherwise, failed cases are recorded and execution continues

---

## Best Practices

- Use YAML configs for readability
- Use `flush_csv_after_each_case` for long experiments
- Always specify `method-name` for clarity in results
- Start with a small config before scaling up

---

## Summary

The CLI provides a complete interface for:

- loading configurations
- running experiments
- integrating external models
- generating reproducible results

It is the recommended way to run large-scale benchmarks with HyperBench.
