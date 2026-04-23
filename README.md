# HyperBench

**Authors:** Ritik Shah ([ritik.shah@aritheon.com](mailto:ritik.shah@aritheon.com)), Marco Duarte ([mduarte@ecs.umass.edu](mailto:mduarte@ecs.umass.edu)) 

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

```
def run_pipeline(HR_MSI, LR_HSI, srf, psf, metadata=None):
   # Your model logic
   return prediction
```
or optionally if you profile your model's FLOPs, number of parameters, GPU memory usage, etc. put the model statistics with their label into a dictionary called statistics for HyperBench to include these in the CSV files produced in addition to the quality metrics which will be included in this output CSV by default:
```
def run_pipeline(HR_MSI, LR_HSI, srf, psf, metadata=None):
   # Your model logic
   return prediction, stats
```
Supported backends include NumPy, TensorFlow, and PyTorch. This run_pipline function must be wrapped in a class:

```
class ExamplePipeline:
    def run_pipeline(self, HR_MSI, LR_HSI, srf, psf, metadata=None):
        return run_pipeline(HR_MSI, LR_HSI, srf, psf, metadata)
```

This ExamplePipeline class can then be given to HyperBench:

```
    adapter = PipelineAdapter(
        pipeline=ExamplePipeline(),
        name=method_name,
        input_backend="numpy",
        output_backend="numpy",
        add_batch_dim=False,
        device="auto",
    )
```

This setup allows HyperBench to directly give the LR HSI, HR MSI, SRF, and PSF as inputs to your model. If your model does not need any of these inputs (for example some models do not require the SRF or the PSF as inputs), you can simply define your ExamplePipeline class to take them as None:
```
class ExamplePipeline:
    def run_pipeline(self, HR_MSI, LR_HSI, srf=None, psf=None, metadata=None):
        return run_pipeline(HR_MSI, LR_HSI, srf, psf, metadata)
```

If your method requires scene specific or degradation specific hyperparameter tuning, you must take care of this within your run_pipeline function itself. HyperBench does not support any specific hyperparameter settings, since it expects the user to directly supply a model contract through the run_pipeline function defined by the user, allowing you to explicitly define the exact hyperparameters that your model needs when running HyperBench on a specific dataset and with specific degradations. 

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

## Data

This repository does not host any datasets. Popular hyperspectral datasets in the .mat format can be easily downloaded online. Users can also visit this link: https://github.com/ritikgshah/SpectraLift/tree/main/Datasets where the Washington DC Mall, Kennedy Space Center, Botswana, Pavia University, and Pavia Center datasets in .mat format are hosted.

---

## Repository Structure
```
hyperbench/
├── src/hyperbench/
├── docs/
├── notebooks/
├── cli/
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

## CLI: Running HyperBench from Python Scripts

The `cli/` directory provides a clean, script-based pathway to run HyperBench end-to-end—without notebooks. It mirrors the full workflow demonstrated in **Notebook 07 (End-to-End Benchmark Pipeline)** and is intended for repeatable, production-style experiments. [Command line interface help](cli/cli_README.md) provides further details about how to structure python scripts, how to use yaml and json configuration templates to run your own experiments, and how to integrate a simple pipeline (the same one used in the notebooks tutorial) to get started with the HyperBench framework.

---

## Examples: Integrating Real-World Models with HyperBench

The `examples/` directory demonstrates how **real-world hyperspectral super-resolution models** can be seamlessly integrated into the HyperBench framework.

Unlike the tutorial notebooks, which focus on understanding the framework using simplified pipelines, this section shows how HyperBench operates in **practical research settings** with full model implementations. [Examples readme](examples/examples_README.md) provides exact details about which methods are included, how to go through the examples directory, and enables users to seamless understand how to structure their models for easy integration with HyperBench.

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
