# IO: Loading Hyperspectral Scenes

This document describes how to load hyperspectral images (HSI) into HyperBench
and how the framework interprets `.mat` files.

---

## Supported Format

HyperBench currently supports loading hyperspectral scenes from:

- MATLAB `.mat` files

A valid `.mat` file must contain a 3D array representing the hyperspectral cube.

---

## Expected Data Shape

All scenes are expected to follow:

(H, W, B)

where:
- H: height (spatial)
- W: width (spatial)
- B: number of spectral bands

---

## Loading a Scene

Use:

```python
from hyperbench import load_hsi

scene = load_hsi("data/scene.mat", key="dc")
```

---

## Parameters

### path

Path to the `.mat` file.

Example:
```
data/scene.mat
```

---

### key (optional but recommended)

The key inside the `.mat` file corresponding to the hyperspectral cube.

Example:
```
key="dc"
```

---

## Key Inference Behavior

If `key` is not provided:

- If exactly one valid 3D array exists → it is used automatically
- If multiple candidates exist → an error is raised
- If no valid arrays exist → an error is raised

For reliability, always specify the key explicitly.

---

## Validation

`load_hsi` performs validation checks:

- ensures the data is 3D
- ensures numeric type
- ensures non-empty dimensions

If validation fails, an exception is raised.

---

## Data Type

The returned array is:

- a NumPy array
- typically converted to `float32` or `float64`

---

## Normalization

Loading does not automatically normalize the data.

You should explicitly normalize using:

```python
from hyperbench import normalize_image

gt_hsi = normalize_image(scene)
```

---

## Common Issues

### Wrong Key

Error occurs if:
- key does not exist
- key does not map to a valid 3D array

---

### Multiple Arrays in File

If multiple 3D arrays exist, you must specify which one to use.

---

### Unexpected Shape

Ensure your data follows:

(H, W, B)

and not:
- (B, H, W)
- flattened arrays

---

## Example

```python
from hyperbench import load_hsi, normalize_image

scene = load_hsi("data/scene.mat", key="dc")
gt_hsi = normalize_image(scene)

print(gt_hsi.shape)
```

---

## Summary

The IO module is responsible for:

- loading hyperspectral scenes from `.mat` files
- validating structure and content
- returning a clean NumPy representation

Correct loading is essential, as all degradations and evaluations depend on this step.
