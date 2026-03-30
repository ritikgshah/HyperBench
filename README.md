# HyperBench

**Authors:** Ritik Shah ([rgshah@umass.edu](mailto:rgshah@umass.edu)), Marco Duarte ([mduarte@ecs.umass.edu](mailto:mduarte@ecs.umass.edu)) 

HyperBench is a benchmarking framework for **fusion-based hyperspectral super-resolution (HSR)** methods.

It is designed for the setting in which a high-resolution hyperspectral image (HR-HSI) is reconstructed by fusing:

- a low-resolution hyperspectral image (LR-HSI), and  
- a high-resolution multispectral image (HR-MSI)  

HyperBench provides a standardized and reproducible pipeline for generating synthetic degradations, evaluating reconstruction quality, and logging structured results.

---

## Overview

Hyperspectral super-resolution methods are often evaluated under inconsistent experimental setups, making direct comparison difficult. HyperBench addresses this by providing:

- a unified degradation pipeline  
- a consistent evaluation protocol  
- a model-agnostic integration interface  
- structured experiment outputs  

The framework is intended strictly for evaluation, not training.

---

## Core Functionality

HyperBench implements the full evaluation workflow:

1. Load a hyperspectral scene from a .mat file  
2. Generate synthetic inputs:
   - LR-HSI via spatial degradation  
   - HR-MSI via spectral degradation  
3. Pass degraded inputs into a model pipeline  
4. Normalize and validate model outputs  
5. Optionally apply output clipping  
6. Compute evaluation metrics  
7. Record results to a structured CSV file  

---

## Key Features

### Synthetic Degradation Pipeline

- Built-in PSFs for spatial degradation  
- Built-in SRFs for MSI simulation  
- Configurable downsampling ratios  
- Configurable MSI band counts  
- Configurable noise levels  

---

### Evaluation Metrics

- RMSE  
- PSNR  
- SSIM  
- UIQI  
- ERGAS  
- SAM  

---

### Model Integration

Models must expose:

def run_pipeline(HR_MSI, LR_HSI, srf, psf=None, metadata=None):
    return prediction

or optionally:

return prediction, stats

Supported backends include NumPy, TensorFlow, and PyTorch.

---

### CLI and Configuration

Experiments can be defined using YAML or JSON configuration files and executed via:

hyperbench run --config config.yaml --pipeline-module path/to/model.py

---

## Installation

To install and seamlessly run hyperbench, create a fresh conda environment:

```
conda create --name <env-name> python=3.9.21
conda activate <env-name>
```

Once created and activated, proceed to install hyperbench:

```
pip install hyperbench
```

This package installs all the hyperbench dependencies, however, any additional libraries that your models require such as tensorflow, pytorch, tqdm, etc. will have to be installed seperately.

---

## Repository Structure
```
hyperbench/
├── src/hyperbench/
├── docs/
├── notebooks/
├── configs/
├── examples/
├── README.md
├── LICENSE
└── pyproject.toml
```
---

## Package Structure
```
src/hyperbench/
├── adapters/
├── benchmark/
├── degradations/
├── io/
├── metrics/
├── utils/
├── cli.py
├── config.py
├── exceptions.py
├── types.py
└── __init__.py
```
---

## Documentation

We provide in depth documentation, allowing a user to seamlessly start generating experiments with hyperbench. It is recommended that as a new user, you first go through the documentation in detail. [Documentation index](docs/index.md) is the first file recommended as it lays out the structure and sequence that the documentation files should be read in.

### Documentation Structure
```
docs/
├── index.md
├── quickstart.md
├── core-concepts.md
├── pipeline-interface.md
├── config-files.md
├── cli.md
├── io.md
├── degradations.md
├── metrics.md
├── visualization.md
├── adapters.md
├── model-integration.md
└── api-reference.md
```

---

## Notebooks

The `notebooks/` directory provides a structured, tutorial-based walkthrough of HyperBench.  

These notebooks are designed to guide users from basic usage to full benchmark orchestration in a clear and practical way. Each notebook focuses on a specific part of the framework, while collectively demonstrating how all components work together in a complete hyperspectral fusion benchmarking pipeline.

The sequence in which you should explore these notebooks is clearly explained in the [notebooks summary](notebooks/summary.md). It is recommended to read through this file before exploring individual notebooks to have a seamless experience in understanding the tutorial.

---

## Scope

HyperBench provides:
- synthetic benchmarking  
- degradation generation  
- evaluation metrics  
- CSV logging  
- model integration  

HyperBench does not provide:
- model training  
- dataset hosting  
- pretrained models  

---

## License

MIT License
