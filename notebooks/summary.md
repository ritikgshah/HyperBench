
### Notebook Overview

**[01_hyperbench_quickstart.ipynb](./notebooks/01_hyperbench_quickstart.ipynb)**  
Introduces the core workflow of HyperBench.  
You will learn how to:
- load a hyperspectral scene
- generate degraded inputs (LR HSI and HR MSI)
- run a simple reconstruction method
- compute evaluation metrics  

This notebook provides the fastest path to running a complete example.

---

**[02_loading_and_visualization.ipynb](./notebooks/02_loading_and_visualization.ipynb)**  
Focuses on data handling and visualization.  
You will learn how to:
- load `.mat` hyperspectral scenes
- normalize data consistently
- visualize spectral bands and RGB projections  

This establishes a clear understanding of the data representation used throughout the framework.

---

**[03_degradations_tutorial.ipynb](./notebooks/03_degradations_tutorial.ipynb)**  
Explores the synthetic degradation pipeline in detail.  
You will learn how to:
- generate and visualize different PSFs
- generate LR HSI using spatial degradation
- generate HR MSI using SRFs
- inspect how `(r, c)` configurations affect inputs  

This notebook makes the degradation process fully transparent.

---

**[04_metrics_and_evaluation.ipynb](./notebooks/04_metrics_and_evaluation.ipynb)**  
Covers evaluation of reconstruction quality.  
You will learn how to:
- compute standard hyperspectral metrics (RMSE, PSNR, SSIM, SAM)
- interpret metric values
- compare predictions against ground truth  

This notebook explains how HyperBench quantifies performance.

---

**[05_model_interface_and_adapters.ipynb](./notebooks/05_model_interface_and_adapters.ipynb)**  
Describes how models integrate into HyperBench.  
You will learn how to:
- implement the `run_pipeline(...)` interface
- understand required inputs and outputs
- return optional model statistics
- use `PipelineAdapter` to connect your model  

This notebook is essential for integrating custom methods.

---

**[06_framework_conversion_tutorial.ipynb](./notebooks/06_framework_conversion_tutorial.ipynb)**  
Explains framework interoperability.  
You will learn how to:
- convert data between NumPy, TensorFlow, and PyTorch
- handle device placement
- manage channel layouts and batch dimensions  

This ensures compatibility with real-world deep learning models.

---

**[07_end_to_end_benchmark_pipeline.ipynb](./notebooks/07_end_to_end_benchmark_pipeline.ipynb)**  
Demonstrates full benchmark orchestration.  
You will learn how to:
- define explicit benchmark cases using `DegradationSpec`
- construct parameter sweeps
- run `run_benchmark(...)`
- save results to CSV
- control execution behavior (`flush`, `overwrite`, `fail_fast`)
- inspect structured outputs  

This notebook ties together the entire framework into a complete experimental workflow.

---

### What You Can Achieve

By working through the notebooks, you will be able to:

- understand how hyperspectral fusion benchmarks are constructed
- generate realistic synthetic inputs for model evaluation
- integrate your own reconstruction methods into HyperBench
- run controlled experiments and parameter sweeps
- produce structured, reproducible benchmark results

The notebooks are designed to function both as a learning resource and as reusable templates for building your own experiments.
