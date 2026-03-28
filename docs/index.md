# HyperBench Documentation

This documentation provides a complete guide to using HyperBench for benchmarking
fusion-based hyperspectral super-resolution (HSR) methods.

HyperBench is designed for scenarios where a high-resolution hyperspectral image
(HR-HSI) is reconstructed by fusing:

- a low-resolution hyperspectral image (LR-HSI), and
- a high-resolution multispectral image (HR-MSI)

The framework handles synthetic degradation, evaluation, and experiment logging,
while users supply the reconstruction model.

---

## Getting Started

If you are new to HyperBench, begin here:

- [Quickstart](docs/quickstart.md): `quickstart.md`  
  A minimal end-to-end workflow demonstrating how to load data, generate degradations,
  run a pipeline, and compute metrics.

- [Core concepts](docs/core-concepts.md): `core-concepts.md`  
  Explanation of LR-HSI, HR-MSI, PSF, SRF, and the fusion-based HSR setting.

---

## Running Experiments

- [Configuration files](docs/config-files.md): `config-files.md`  
  Defines how to specify experiments using YAML or JSON.

- [Command Line Interface (CLI) usage](docs/cli.md): `cli.md`  
  Run benchmarks directly from the command line.

---

## Model Integration

- [Pipeline interface](docs/pipeline-interface.md): `pipeline-interface.md`  
  Defines the required `run_pipeline(...)` interface for all models.

- [Model integration guide](docs/model-integration.md): `model-integration.md`  
  Practical guidance for integrating real-world models.


---

## Framework Components

- [IO](docs/io.md): `io.md`  
  Loading and validating hyperspectral scenes.

- [Degradations](docs/degradations.md): `degradations.md`  
  PSFs, SRFs, and synthetic input generation.

- [Metrics](docs/metrics.md): `metrics.md`  
  Evaluation metrics and their interpretation.

- [Visualization](docs/visualization.md): `visualization.md`  
  Built-in tools for inspecting data and degradations.

- [Adapters](docs/adapters.md): `adapters.md`  
  Model wrappers and backend handling.

---

## Reference

- [API refence](docs/api-reference.md): `api-reference.md`  
  Detailed descriptions of all public functions, classes, and modules.

---

## Recommended Reading Order

For most users, the following order is recommended:

1. Quickstart  
2. Core Concepts  
3. Pipeline Interface  
4. Configuration Files  
5. CLI Usage  

After that, refer to module-specific documentation as needed.

---

## Summary

HyperBench is structured to support:

- reproducible benchmarking  
- consistent synthetic degradations  
- standardized evaluation metrics  
- flexible model integration  

The documentation is organized to guide users from basic usage to advanced
integration and full benchmark execution.
