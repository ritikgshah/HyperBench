# CLI: Running HyperBench from Python Scripts

The `cli/` directory provides a **script-based entry point** for running HyperBench end-to-end without relying on notebooks. It mirrors the workflow shown in **Notebook 07 (End-to-End Benchmark Pipeline)**, but in a form that is easier to execute, reproduce, and scale.

This interface is intentionally minimal:

> **one Python file + one config file = a complete benchmark run**

---

## Overview

The CLI design cleanly separates responsibilities:

### Config files (`configs/`)
Define **everything about the experiment**:

- dataset path and key  
- PSFs and degradation settings  
- explicit benchmark cases *or* sweep parameters  
- output paths  
- CSV behavior (`flush`, `overwrite`)  
- execution control (`fail_fast`)  

### Runner script (`run_benchmark.py`)
Defines:

- the reconstruction method (`run_pipeline(...)`)
- the `PipelineAdapter`
- how configs are loaded and executed

This allows users to:
- modify experiments without touching Python code  
- reuse the same runner across different experiments  

---

## Execution Flow

```
Config File (YAML / JSON)
        │
        ▼
Load + Normalize Config
        │
        ▼
Generate Degradations
(LR HSI + HR MSI)
        │
        ▼
run_pipeline(...)
(user-defined model)
        │
        ▼
Evaluation Metrics
(RMSE, PSNR, SSIM, SAM)
        │
        ▼
Structured Results
        │
        ▼
CSV Output
```

---

## Folder Structure

```
cli/
├── run_benchmark.py
└── configs/
    ├── config_explicit_degradation_specs.yaml
    ├── config_explicit_degradation_specs.json
    ├── config_sweep.yaml
```

---

## Two Benchmark Modes (Auto-detected)

### Explicit Mode

Triggered when the config contains:

```
degradation_specs:
```

You explicitly define each benchmark case:

- fixed `(downsample_ratio, msi_band_count)` pairs  
- controlled SNR values  
- deterministic experiment set  

---

### Sweep Mode

Triggered when `degradation_specs` is **not present**, and instead fields like:

```
msi_band_counts:
downsample_ratio_to_spatial_snr_db:
```

are provided.

HyperBench automatically generates all parameter combinations.

---

## How to Run

Run all commands from the **repository root**.

### Explicit benchmark (YAML)

```
python cli/run_benchmark.py cli/configs/config_explicit_degradation_specs.yaml
```

### Explicit benchmark (JSON)

```
python cli/run_benchmark.py cli/configs/config_explicit_degradation_specs.json
```

### Sweep benchmark

```
python cli/run_benchmark.py cli/configs/config_sweep.yaml
```

For sweep benchmark, it is strongly recommended that users execute yaml files, since specifying downsampling ratios along with their respective SNR for noise addition usually causes issues with the json parser and the type of input the HyperBench framework requires. Using a json file for sweep benchmark config files will lead to an error.

---

## What Happens During Execution

For each benchmark case:

1. The hyperspectral scene is loaded  
2. Degraded inputs are generated  
3. `run_pipeline(...)` is executed  
4. Metrics are computed  
5. Results are written to CSV  

---

## Important: This is a Demonstration Pipeline

The provided `run_pipeline(...)` is intentionally simple:

- nearest-neighbor upsampling  
- ignores HR MSI  
- NumPy-based  

Its purpose is to demonstrate:

- how the framework is executed  
- how configs control experiments  
- how models plug into HyperBench  

It is **not meant for performance benchmarking**.

---

## Using Real Models

To run real hyperspectral fusion methods:

- replace `run_pipeline(...)` with your model  
- or wrap your model using `PipelineAdapter`  

See [examples](../examples) for implementations of real research models running on the HyperBench framework.

---

## Recommended Workflow

1. Start with Notebook 07  
2. Run CLI version here  
3. Modify config files  
4. Replace pipeline with your model  
5. Scale experiments  

---

## Summary

The CLI provides a simple and reproducible execution path:

**config → pipeline → benchmark → CSV results**

It is designed to be:

- easy to understand  
- easy to modify  
- suitable for real experiments  
