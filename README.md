# HyperBench

**HyperBench** is a synthetic benchmarking framework for **fusion-based
hyperspectral super-resolution (HSR)** methods.

It is designed specifically for the setting where a model reconstructs a
high-resolution hyperspectral image (HR-HSI) by **fusing**: - a
**low-resolution hyperspectral image (LR-HSI)**, and a **high-resolution multispectral image (HR-MSI)**

HyperBench provides a standardized, reproducible pipeline for generating
these inputs, evaluating models, and logging results.

------------------------------------------------------------------------

## What HyperBench Does

HyperBench handles the full evaluation pipeline:

1.  Load hyperspectral scenes from `.mat` files\
2.  Generate synthetic degradations:
    -   LR-HSI via spatial degradation (PSFs)
    -   HR-MSI via spectral degradation (SRFs)
3.  Pass inputs into a model pipeline (`run_pipeline`)\
4.  Convert outputs to a consistent internal format\
5.  Apply optional clipping policy\
6.  Compute metrics\
7.  Log structured results to CSV

------------------------------------------------------------------------

## Key Features

### Fusion-based Benchmarking

HyperBench is explicitly built for **HSR fusion methods** that take
`(LR-HSI, HR-MSI)` as inputs.

------------------------------------------------------------------------

### Synthetic Degradations

-   Built-in PSFs (Gaussian, Airy, Moffat, etc.)
-   Built-in SRFs for multiple MSI band configurations
-   Configurable downsampling ratios
-   Configurable SNR levels

------------------------------------------------------------------------

### Metrics

-   RMSE\
-   PSNR\
-   SSIM\
-   UIQI\
-   ERGAS\
-   SAM

------------------------------------------------------------------------

### Model-Agnostic Integration

Works with: - NumPy - TensorFlow - PyTorch

Models must expose:

def run_pipeline(HR_MSI, LR_HSI, srf, psf=None, metadata=None): return
prediction

or:

return prediction, stats

------------------------------------------------------------------------

### CLI + Config System

-   YAML and JSON configs\
-   Fully reproducible experiments\
-   Batch execution via CLI

------------------------------------------------------------------------

### Visualization

Built-in utilities for: - HSI visualization - band inspection - PSF
visualization - SRF visualization - spectral plots

------------------------------------------------------------------------

## Installation

pip install hyperbench

------------------------------------------------------------------------

## Repository Structure

hyperbench/ ├── src/hyperbench/ \# Core package ├── docs/ \# Full
documentation ├── notebooks/ \# Tutorial notebooks ├── configs/ \#
Example configs ├── examples/ \# Example scripts / pipelines ├──
README.md ├── LICENSE └── pyproject.toml

------------------------------------------------------------------------

## Package Structure

src/hyperbench/ ├── adapters/ \# Model integration layer ├── benchmark/
\# Benchmark execution engine ├── degradations/ \# PSF, SRF, and
degradation pipeline ├── io/ \# Scene loading ├── metrics/ \# Evaluation
metrics ├── utils/ \# Visualization, conversion, helpers ├── cli.py \#
CLI entry point ├── config.py \# Config loading (YAML / JSON) ├──
exceptions.py \# Custom exceptions ├── types.py \# Shared types └──
**init**.py

------------------------------------------------------------------------

## Where to Start

### Quickstart

➡️ docs/quickstart.md

### Model Integration

➡️ docs/model-integration.md

### CLI Usage

➡️ docs/cli.md

### Config Files

➡️ docs/config-files.md

------------------------------------------------------------------------

## Tutorials (Notebooks)

➡️ notebooks/

Includes: - end-to-end workflow - visualization - degradations -
TensorFlow / PyTorch conversion - SpectraLift integration - SpectraMorph
integration

------------------------------------------------------------------------

## Example Configs

➡️ configs/

Ready-to-use YAML and JSON configurations for CLI experiments.

------------------------------------------------------------------------

## Example Pipelines

➡️ examples/

Includes: - minimal pipeline example - SpectraLift wrapper -
SpectraMorph wrapper

------------------------------------------------------------------------

## Design Philosophy

HyperBench is built around:

-   Reproducibility\
-   Standardization\
-   Flexibility

------------------------------------------------------------------------

## Scope

### HyperBench does:

-   synthetic benchmarking
-   degradations
-   evaluation
-   CSV logging
-   model integration

### HyperBench does NOT do:

-   model training
-   dataset hosting
-   pretrained model distribution

------------------------------------------------------------------------

## License

MIT License
