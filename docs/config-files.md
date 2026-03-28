# Configuration Files

HyperBench uses configuration files to define benchmark experiments.

Configurations can be written in:
- YAML (`.yaml`, `.yml`)
- JSON (`.json`)

These files specify:
- the input scene
- degradation settings
- evaluation options
- output behavior

---

## Basic Structure

A configuration file typically includes:

- scene information
- degradation specifications
- evaluation settings
- output settings

---

## Required Fields

### scene_path

Path to the hyperspectral scene file.

Example:
scene_path: data/scene.mat

---

### scene_key

Key inside the `.mat` file corresponding to the hyperspectral cube.

Example:
scene_key: dc

---

### degradation_specs

Defines the exact experiments to run.

Each entry specifies a combination of:

- downsample_ratio
- msi_band_count
- spatial_snr_db
- spectral_snr_db

Example:

degradation_specs:
  - downsample_ratio: 8
    msi_band_count: 4
    spatial_snr_db: 30.0
    spectral_snr_db: 40.0

---

## PSF Configuration

### psf_names

List of PSFs to use.

Example:
psf_names:
  - gaussian
  - moffat

---

### psf_sigmas

Controls blur strength.

Example:
psf_sigmas: [3.4]

---

### psf_kernel_radii

Controls kernel size.

Example:
psf_kernel_radii: [7]

---

## SRF / MSI Settings

Defined implicitly through:

- msi_band_count in degradation_specs

Typical values:
- 3
- 4
- 8
- 16

---

## Noise Settings

### spatial_snr_db

Noise level applied to LR-HSI.

### spectral_snr_db

Noise level applied to HR-MSI.

Lower values indicate higher noise.

---

## Normalization

### normalize_inputs

Whether to normalize input scenes.

Example:
normalize_inputs: true

---

### lower_percentile / upper_percentile

Used for percentile-based normalization.

Example:
lower_percentile: 1.0
upper_percentile: 99.0

---

## Clipping Policy

### clip_prediction_to_unit_interval

Whether to clip predictions before metric computation.

Example:
clip_prediction_to_unit_interval: true

---

### prediction_clip_min / prediction_clip_max

Defines clipping range.

Example:
prediction_clip_min: 0.0
prediction_clip_max: 1.0

---

## Output Settings

### save_csv

Whether to write results to CSV.

---

### output_csv_path

Path to output file.

Example:
output_csv_path: outputs/results.csv

---

### flush_csv_after_each_case

Writes each result immediately to disk.

Recommended for long runs.

---

### overwrite_csv_on_start

Whether to overwrite existing CSV file.

---

## Execution Settings

### seed

Random seed for reproducibility.

---

### fail_fast

If true, stops execution on first error.

---

## Complete YAML Example

scene_path: data/scene.mat
scene_key: dc

psf_names:
  - gaussian

psf_sigmas: [3.4]
psf_kernel_radii: [7]

degradation_specs:
  - downsample_ratio: 8
    msi_band_count: 4
    spatial_snr_db: 30.0
    spectral_snr_db: 40.0

normalize_inputs: true
lower_percentile: 1.0
upper_percentile: 99.0

clip_prediction_to_unit_interval: true
prediction_clip_min: 0.0
prediction_clip_max: 1.0

save_csv: true
output_csv_path: outputs/results.csv
flush_csv_after_each_case: true
overwrite_csv_on_start: true

seed: 42
fail_fast: true

---

## Complete JSON Example

{
  "scene_path": "data/scene.mat",
  "scene_key": "dc",
  "psf_names": ["gaussian"],
  "psf_sigmas": [3.4],
  "psf_kernel_radii": [7],
  "degradation_specs": [
    {
      "downsample_ratio": 8,
      "msi_band_count": 4,
      "spatial_snr_db": 30.0,
      "spectral_snr_db": 40.0
    }
  ],
  "normalize_inputs": true,
  "lower_percentile": 1.0,
  "upper_percentile": 99.0,
  "clip_prediction_to_unit_interval": true,
  "prediction_clip_min": 0.0,
  "prediction_clip_max": 1.0,
  "save_csv": true,
  "output_csv_path": "outputs/results.csv",
  "flush_csv_after_each_case": true,
  "overwrite_csv_on_start": true,
  "seed": 42,
  "fail_fast": true
}

---

## Summary

Configuration files define:

- what data to use
- what degradations to apply
- how to evaluate results
- how to store outputs

They are the primary way to run large-scale, reproducible experiments in HyperBench.
