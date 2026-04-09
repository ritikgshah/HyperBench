
## Purpose of This Section

The goal of the examples directory is to:

- demonstrate **end-to-end integration of real models**
- show how HyperBench handles **complex pipelines beyond toy methods**
- provide **reference implementations** that users can adapt for their own work
- illustrate both **interactive (notebook)** and **reproducible (CLI/script)** workflows

These examples confirm that HyperBench is not limited to synthetic demonstrations, but is fully capable of supporting **research-grade models and experiments**.

---

## Included Examples

This repository currently includes:

```
examples/
├── spectralift/
└── spectramorph/
```

These methods were selected because:

- they represent **modern approaches in fusion-based HSR**
- they require **non-trivial degradation handling**
- they highlight the importance of **systematic experimentation**

We intentionally limit the examples to these methods to ensure:
- correctness of implementation
- clarity of integration
- maintainability of the repository

---

## Structure of Each Example

Each example follows the same structure:

```
<method_name>/
├── <method_name>_hyperbench_integration.ipynb
├── <method_name>_hyperbench_pipeline.py
├── <method_name>_config_explicit_degradations.yaml
└── <method_name>_sweep_config.yaml
└── outputs/
```

### 1. Jupyter Notebook

The notebook provides an **interactive walkthrough** of:

- loading data
- defining degradation settings
- running the model
- visualizing outputs

It includes:
- explicit degradation examples
- small sweep demonstrations

This is the recommended entry point for understanding how the model integrates with HyperBench.

---

### 2. Python Script

The script provides a **reproducible execution path**:

- defines the model-specific `run_pipeline(...)`
- wraps it using `PipelineAdapter`
- loads configuration files
- runs full benchmarks

This mirrors real experimental usage and allows users to scale experiments easily.

---

### 3. Configuration Files

Each example includes two configuration styles:

#### Explicit Config
Defines specific degradation cases manually.

#### Sweep Config
Defines parameter ranges to automatically generate multiple experiment combinations.

These configs demonstrate how HyperBench enables:
- **systematic evaluation across conditions**
- **scalable experimentation without modifying code**

---

### 4. Outputs folder

This outputs folder containts the CSV files produced by HyperBench after running the experiments chosen by the user for the given method. Users can easily visualize the output that they can expect by using the HyperBench framework.

---

## What These Examples Show

These examples highlight several key capabilities of HyperBench:

- integration of **TensorFlow models**
- support for **complex degradation pipelines**
- separation of:
  - model logic
  - experiment configuration
  - evaluation
- ability to run both:
  - controlled experiments
  - large-scale sweeps

Most importantly, they demonstrate that HyperBench works with:

> **real research models, not just simplified baselines**

---

## When to Use This Section

Use the `examples/` directory if you want to:

- integrate your own model into HyperBench
- understand how real pipelines are structured
- run full experiments using configuration files
- extend HyperBench for new research

---

## Recommended Workflow

1. Start with the **tutorial notebooks** (`notebooks/`)
2. Move to the **examples directory**
3. Run a notebook to understand a real integration
4. Use the corresponding CLI script for reproducible experiments
5. Adapt the structure for your own model

---

## Summary

The examples directory bridges the gap between:

- learning how HyperBench works  
and  
- applying it to real research problems  

It demonstrates that HyperBench is:

- flexible  
- scalable  
- and ready for real-world use
