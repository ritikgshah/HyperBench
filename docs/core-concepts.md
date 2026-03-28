# Core Concepts

This document explains the fundamental concepts behind HyperBench and the
fusion-based hyperspectral super-resolution (HSR) setting it is designed for.

Understanding these concepts will make it much easier to use the framework
correctly and interpret results.

---

## Fusion-Based Hyperspectral Super-Resolution

HyperBench focuses on **fusion-based HSR**, where a high-resolution hyperspectral
image (HR-HSI) is reconstructed by combining two complementary inputs:

- a **low-resolution hyperspectral image (LR-HSI)**
- a **high-resolution multispectral image (HR-MSI)**

Each input provides different information:

- LR-HSI:
  - high spectral resolution
  - low spatial resolution

- HR-MSI:
  - high spatial resolution
  - low spectral resolution

The goal is to fuse these to recover a full-resolution hyperspectral image.

---

## Ground Truth (GT HSI)

The starting point in HyperBench is a **ground truth hyperspectral image**:

- Shape: (H, W, B)
- H, W: spatial dimensions
- B: number of spectral bands

This image is assumed to be:

- high spatial resolution
- high spectral resolution

All degradations are generated from this ground truth.

---

## Low-Resolution Hyperspectral Image (LR-HSI)

The LR-HSI is created by applying **spatial degradation** to the ground truth.

This involves:

1. Blurring using a **Point Spread Function (PSF)**
2. Downsampling by a factor `r`

Resulting shape:

(H / r, W / r, B)

The LR-HSI retains full spectral information but loses spatial detail.

---

## High-Resolution Multispectral Image (HR-MSI)

The HR-MSI is created by applying **spectral degradation** to the ground truth.

This involves:

1. Applying a **Spectral Response Function (SRF)**
2. Reducing spectral dimensionality

Resulting shape:

(H, W, C)

where `C` is the number of multispectral bands (e.g., 3, 4, 8, 16).

The HR-MSI retains spatial resolution but loses spectral detail.

---

## Point Spread Function (PSF)

A PSF models the spatial blur introduced by an imaging system.

In HyperBench, PSFs are used to simulate spatial degradation when generating LR-HSI.

Examples of supported PSFs include:

- Gaussian
- Airy
- Moffat
- Sinc
- and others

Key parameters:

- sigma (controls blur strength)
- kernel size

---

## Spectral Response Function (SRF)

An SRF defines how hyperspectral bands are combined to produce multispectral bands.

It is represented as a matrix:

(C_msi, B_hsi)

Each row corresponds to one MSI band and defines how it is formed from HSI bands.

SRFs simulate the spectral characteristics of real sensors.

---

## Downsampling Ratio

The downsampling ratio `r` controls the spatial resolution reduction:

- LR-HSI shape becomes (H / r, W / r, B)

Typical values:

- 4
- 8
- 16
- 32

Higher values correspond to more challenging reconstruction tasks.

---

## Noise and SNR

HyperBench allows adding noise during degradation:

- spatial noise (for LR-HSI)
- spectral noise (for HR-MSI)

Noise is controlled using **Signal-to-Noise Ratio (SNR)** in decibels (dB).

Lower SNR means more noise.

---

## Degradation Pipeline

The complete synthetic pipeline is:

1. Start with ground truth HSI
2. Generate LR-HSI:
   - apply PSF
   - downsample
   - add spatial noise
3. Generate HR-MSI:
   - apply SRF
   - add spectral noise

These two inputs are then passed to the model.

---

## Model Reconstruction

The model receives:

- HR-MSI
- LR-HSI
- SRF (and optionally PSF)

and produces:

- predicted HR-HSI

The output should match the ground truth shape:

(H, W, B)

---

## Normalization

HyperBench typically normalizes inputs to a consistent range, usually:

[0, 1]

This ensures:

- stable metric computation
- fair comparisons across models

---

## Clipping Policy

Model outputs may not always lie in the expected range.

HyperBench supports an optional clipping step:

- clip predictions to [0, 1] before metric evaluation

This is important because the ground truth is also normalized.

---

## Evaluation Metrics

HyperBench evaluates reconstruction quality using:

- RMSE
- PSNR
- SSIM
- UIQI
- ERGAS
- SAM

Each metric captures different aspects of reconstruction quality:

- spatial fidelity
- spectral fidelity
- structural similarity

---

## Summary

HyperBench simulates a realistic fusion-based HSR scenario by:

- starting from a high-quality ground truth
- generating complementary degraded inputs (LR-HSI and HR-MSI)
- evaluating how well a model reconstructs the original image

Understanding these components is essential for:

- designing experiments
- interpreting results
- integrating models correctly
