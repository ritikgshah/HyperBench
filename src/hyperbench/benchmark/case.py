# Author: Ritik Shah

"""Benchmark configuration and case dataclasses for HyperBench."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


Array = np.ndarray


@dataclass
class DegradationSpec:
    """One explicit synthetic degradation condition."""

    downsample_ratio: int
    msi_band_count: int
    spatial_snr_db: float
    spectral_snr_db: float = 40.0


@dataclass
class BenchmarkConfig:
    """Configuration for a HyperBench benchmark run."""

    scene_path: Union[str, Path]
    scene_key: Optional[str] = None

    psf_names: List[str] = field(default_factory=lambda: ["gaussian"])
    psf_sigmas: List[float] = field(default_factory=lambda: [3.4])
    psf_kernel_radii: List[int] = field(default_factory=lambda: [7])

    # Explicit degradation mode
    degradation_specs: Optional[List[DegradationSpec]] = None

    # Sweep mode fields
    msi_band_counts: List[int] = field(default_factory=lambda: [4])
    downsample_ratio_to_spatial_snr_db: Dict[int, float] = field(
        default_factory=lambda: {4: 35.0}
    )
    spectral_snr_dbs: List[float] = field(default_factory=lambda: [40.0])

    fwhm_factor: float = 4.2
    seed: int = 42

    metrics: List[str] = field(
        default_factory=lambda: ["rmse", "psnr", "ssim", "uiqi", "ergas", "sam"]
    )

    normalize_inputs: bool = True
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0

    # --------------------------------------------------------------
    # Prediction clipping policy
    # --------------------------------------------------------------
    clip_prediction_to_unit_interval: bool = True
    prediction_clip_min: float = 0.0
    prediction_clip_max: float = 1.0

    save_csv: bool = True
    output_csv_path: Union[str, Path] = "hyperbench_results.csv"
    flush_csv_after_each_case: bool = True
    overwrite_csv_on_start: bool = True

    fail_fast: bool = False

    user_srf: Optional[Array] = None
    user_psf: Optional[Array] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkCase:
    """One fully expanded benchmark case."""

    case_id: str
    scene_path: Union[str, Path]
    scene_key: Optional[str]

    psf_name: str
    psf_sigma: float
    psf_kernel_radius: int

    downsample_ratio: int
    msi_band_count: int

    spatial_snr_db: float
    spectral_snr_db: float

    fwhm_factor: float
    seed: int

    normalize_inputs: bool
    lower_percentile: float
    upper_percentile: float

    user_srf: Optional[Array] = None
    user_psf: Optional[Array] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticCase:
    """Fully materialized synthetic benchmark case for one run."""

    case: BenchmarkCase
    gt_hsi: Array
    lr_hsi: Array
    hr_msi: Array
    srf: Optional[Array]
    psf: Optional[Array]
    wavelengths: Optional[Array] = None
    band_specs: Optional[Array] = None