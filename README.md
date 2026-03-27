# HyperBench

**HyperBench** is a synthetic benchmarking framework for hyperspectral
super-resolution (HSR) methods. It provides a standardized, reproducible
pipeline for evaluating models using controlled degradations, consistent
metrics, and structured experiment outputs.

HyperBench is designed as an **inference-time evaluation tool**. It does
not perform model training. Instead, it enables users to test
already-trained models under a wide range of synthetic scenarios.

------------------------------------------------------------------------

## Key Features

### Synthetic Degradation Pipeline

HyperBench generates realistic low-resolution inputs from
high-resolution hyperspectral scenes:

-   Spatial degradation via configurable Point Spread Functions (PSFs)
-   Spectral degradation via configurable Spectral Response Functions
    (SRFs)
-   Controlled noise injection using SNR (dB)

------------------------------------------------------------------------

### Flexible Experiment Design

Users can define experiments using:

-   Explicit degradation specifications
-   PSF configurations (type, sigma, kernel size)
-   Noise levels (spatial and spectral)
-   MSI band counts

------------------------------------------------------------------------

### Model-Agnostic Adapter Interface

HyperBench works with any model that follows a simple contract:

def run_pipeline(HR_MSI, LR_HSI, srf, psf=None, metadata=None): return
prediction

or

return prediction, stats

Supported formats: - NumPy - TensorFlow - PyTorch

------------------------------------------------------------------------

### Built-in Metrics

-   RMSE
-   PSNR
-   SSIM
-   UIQI
-   ERGAS
-   SAM

------------------------------------------------------------------------

### Output Clipping Policy

Optional clipping to \[0,1\] before metric evaluation.

------------------------------------------------------------------------

### Structured CSV Logging

Each experiment produces structured CSV output with: - parameters -
metrics - optional model stats

------------------------------------------------------------------------

### Visualization Utilities

-   RGB visualization
-   band inspection
-   PSF and SRF visualization
-   spectral plots

------------------------------------------------------------------------

### Framework Support

-   NumPy
-   TensorFlow
-   PyTorch

Automatic device handling included.

------------------------------------------------------------------------

## Installation

pip install hyperbench

------------------------------------------------------------------------

## Command Line Interface

hyperbench run\
--config config.yaml\
--pipeline-module path/to/model.py\
--method-name MyModel\
--input-backend tensorflow

------------------------------------------------------------------------

## Configuration

Supports JSON and YAML configuration files.

------------------------------------------------------------------------

## Design Philosophy

-   Reproducibility
-   Standardization
-   Flexibility

------------------------------------------------------------------------

## Scope

HyperBench is strictly an evaluation framework.

------------------------------------------------------------------------

## License

MIT License
