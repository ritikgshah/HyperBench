# Author: Ritik Shah

"""Benchmark runner for HyperBench."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from hyperbench.adapters import BaseAdapter, ReconstructionInputs
from hyperbench.benchmark.case import BenchmarkCase, BenchmarkConfig, SyntheticCase
from hyperbench.benchmark.generator import generate_cases
from hyperbench.benchmark.results import BenchmarkResults
from hyperbench.degradations import (
    make_psf,
    normalize_image,
    spatial_degradation,
    spectral_degradation,
)
from hyperbench.io import load_hsi
from hyperbench.metrics import evaluate_metrics
from hyperbench.utils import (
    configure_logger,
    ensure_channel_last_hsi,
    get_logger,
    log_benchmark_start,
    log_case_failure,
    log_case_start,
    log_case_success,
    make_rng,
    set_random_seed,
    validate_experiment_inputs,
    validate_prediction_shape,
)


Array = np.ndarray


def _build_synthetic_case(case: BenchmarkCase, scene: Array) -> SyntheticCase:
    scene = ensure_channel_last_hsi(scene, name="scene")
    validate_experiment_inputs(
        scene,
        downsample_ratio=case.downsample_ratio,
        num_msi_bands=case.msi_band_count,
        user_srf=case.user_srf,
    )

    rng_spatial = make_rng(case.seed)
    rng_spectral = make_rng(case.seed + 1)

    gt_hsi = (
        normalize_image(
            scene,
            lower_percentile=case.lower_percentile,
            upper_percentile=case.upper_percentile,
        )
        if case.normalize_inputs
        else np.asarray(scene, dtype=np.float32)
    )

    if case.user_psf is not None:
        psf = np.asarray(case.user_psf, dtype=np.float32)
    else:
        psf = make_psf(
            case.psf_name,
            sigma=case.psf_sigma,
            kernel_radius=case.psf_kernel_radius,
        )

    lr_hsi = spatial_degradation(
        image=scene,
        psf=psf,
        downsample_ratio=case.downsample_ratio,
        snr_db=case.spatial_snr_db,
        normalize=case.normalize_inputs,
        lower_percentile=case.lower_percentile,
        upper_percentile=case.upper_percentile,
        rng=rng_spatial,
    )

    hr_msi, srf, band_specs, wavelengths = spectral_degradation(
        image=scene,
        snr_db=case.spectral_snr_db,
        num_bands=case.msi_band_count,
        fwhm_factor=case.fwhm_factor,
        user_srf=case.user_srf,
        normalize=case.normalize_inputs,
        lower_percentile=case.lower_percentile,
        upper_percentile=case.upper_percentile,
        rng=rng_spectral,
    )

    return SyntheticCase(
        case=case,
        gt_hsi=gt_hsi.astype(np.float32),
        lr_hsi=lr_hsi.astype(np.float32),
        hr_msi=hr_msi.astype(np.float32),
        srf=None if srf is None else srf.astype(np.float32),
        psf=None if psf is None else psf.astype(np.float32),
        wavelengths=None if wavelengths is None else wavelengths.astype(np.float32),
        band_specs=None if band_specs is None else band_specs.astype(np.float32),
    )


def _make_result_row_base(
    synthetic_case: SyntheticCase,
    adapter: BaseAdapter,
    config: BenchmarkConfig,
) -> Dict[str, Any]:
    case = synthetic_case.case
    scene_name = Path(case.scene_path).stem

    row = {
        "scene_name": scene_name,
        "shape_policy": adapter.config.get("shape_policy"),
        "method_name": adapter.config.get("adapter_name"),
        "psf_name": case.psf_name,
        "sigma": case.psf_sigma,
        "kernel_size": 2 * case.psf_kernel_radius + 1,
        "downsampling_ratio": case.downsample_ratio,
        "msi_band_count": case.msi_band_count,
        "spatial_snr": case.spatial_snr_db,
        "spectral_snr": case.spectral_snr_db,
        "fwhm_factor": case.fwhm_factor,
        "seed": case.seed,
        "gt_shape": tuple(int(x) for x in synthetic_case.gt_hsi.shape),
        "lr_hsi_shape": tuple(int(x) for x in synthetic_case.lr_hsi.shape),
        "hr_msi_shape": tuple(int(x) for x in synthetic_case.hr_msi.shape),
        "prediction_clipped": bool(config.clip_prediction_to_unit_interval),
        "prediction_clip_min": float(config.prediction_clip_min),
        "prediction_clip_max": float(config.prediction_clip_max),
    }

    static_stats = adapter.csv_metadata
    if static_stats:
        row.update(static_stats)

    return row


def _make_case_log_metadata(case: BenchmarkCase) -> Dict[str, Any]:
    return {
        "psf": case.psf_name,
        "sigma": case.psf_sigma,
        "kernel": case.psf_kernel_radius,
        "r": case.downsample_ratio,
        "c": case.msi_band_count,
        "snr_spatial": case.spatial_snr_db,
        "snr_spectral": case.spectral_snr_db,
        "seed": case.seed,
    }


def _split_prediction_and_stats(result: Any):
    if isinstance(result, tuple):
        if len(result) != 2 or not isinstance(result[1], dict):
            raise ValueError(
                "Adapters may return either prediction or (prediction, stats_dict)."
            )
        return result[0], result[1]
    return result, {}


def _sanitize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    clean = {}
    for key, value in stats.items():
        if isinstance(value, np.generic):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


def _clip_prediction_if_needed(prediction: Array, config: BenchmarkConfig) -> Array:
    """Clip prediction to the configured numeric range if enabled."""
    prediction = np.asarray(prediction, dtype=np.float32)

    if config.clip_prediction_to_unit_interval:
        prediction = np.clip(
            prediction,
            config.prediction_clip_min,
            config.prediction_clip_max,
        )

    return prediction


def run_benchmark(
    adapter: BaseAdapter,
    config: BenchmarkConfig,
    logger: Optional[logging.Logger] = None,
    log_level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> BenchmarkResults:
    if not isinstance(adapter, BaseAdapter):
        raise ValueError(
            "adapter must be an instance of BaseAdapter, got {}".format(
                type(adapter).__name__
            )
        )

    if logger is None:
        logger = get_logger()
        configure_logger(logger, level=log_level, log_file=log_file)
    else:
        logger.setLevel(log_level)

    set_random_seed(config.seed)

    logger.info(
        "Loading scene from '%s'%s",
        config.scene_path,
        "" if config.scene_key is None else " with key '{}'".format(config.scene_key),
    )
    scene = load_hsi(config.scene_path, key=config.scene_key)

    logger.info("Generating benchmark cases from config.")
    cases = generate_cases(config)

    if config.save_csv and config.overwrite_csv_on_start:
        csv_path = Path(config.output_csv_path)
        if csv_path.exists():
            csv_path.unlink()
        logger.info("Initialized benchmark CSV at '%s'.", csv_path)

    log_benchmark_start(
        logger,
        num_cases=len(cases),
        adapter_name=adapter.name,
    )

    results = BenchmarkResults()

    for case in cases:
        synthetic_case = None  # type: Optional[SyntheticCase]
        inputs = None  # type: Optional[ReconstructionInputs]
        prediction = None  # type: Optional[Array]
        metric_values = None  # type: Optional[Dict[str, float]]

        start_time = time.perf_counter()

        log_case_start(
            logger,
            case.case_id,
            metadata=_make_case_log_metadata(case),
        )

        try:
            synthetic_case = _build_synthetic_case(case, scene)

            inputs = ReconstructionInputs(
                lr_hsi=synthetic_case.lr_hsi,
                hr_msi=synthetic_case.hr_msi,
                srf=synthetic_case.srf,
                psf=synthetic_case.psf,
                metadata={
                    **case.metadata,
                    "case_id": case.case_id,
                    "scene_path": str(case.scene_path),
                    "scene_key": case.scene_key,
                    "downsample_ratio": case.downsample_ratio,
                    "msi_band_count": case.msi_band_count,
                    "spatial_snr_db": case.spatial_snr_db,
                    "spectral_snr_db": case.spectral_snr_db,
                    "psf_name": case.psf_name,
                    "psf_sigma": case.psf_sigma,
                    "psf_kernel_radius": case.psf_kernel_radius,
                    "fwhm_factor": case.fwhm_factor,
                    "seed": case.seed,
                },
            )

            adapter_result = adapter.predict(inputs)
            prediction, row_stats = _split_prediction_and_stats(adapter_result)
            row_stats = _sanitize_stats(row_stats)

            prediction = validate_prediction_shape(
                prediction,
                synthetic_case.gt_hsi,
                prediction_name="prediction",
                reference_name="gt_hsi",
            )

            prediction = _clip_prediction_if_needed(prediction, config)

            metric_values = evaluate_metrics(
                synthetic_case.gt_hsi,
                prediction,
                metrics=config.metrics,
            )

            runtime_seconds = time.perf_counter() - start_time

            row = _make_result_row_base(synthetic_case, adapter, config)
            if row_stats:
                row.update(row_stats)
            row.update(metric_values)
            row["status"] = "success"
            row["error"] = ""
            row["runtime_seconds"] = runtime_seconds

            results.add_row(row)

            if config.save_csv and config.flush_csv_after_each_case:
                results.append_row_to_csv(config.output_csv_path, row)

            log_case_success(
                logger,
                case.case_id,
                runtime_seconds=runtime_seconds,
                metrics=metric_values,
            )

        except Exception as exc:
            runtime_seconds = time.perf_counter() - start_time

            if synthetic_case is not None:
                row = _make_result_row_base(synthetic_case, adapter, config)
            else:
                row = {
                    "scene_name": Path(case.scene_path).stem,
                    "shape_policy": adapter.config.get("shape_policy"),
                    "method_name": adapter.config.get("adapter_name"),
                    "psf_name": case.psf_name,
                    "sigma": case.psf_sigma,
                    "kernel_size": 2 * case.psf_kernel_radius + 1,
                    "downsampling_ratio": case.downsample_ratio,
                    "msi_band_count": case.msi_band_count,
                    "spatial_snr": case.spatial_snr_db,
                    "spectral_snr": case.spectral_snr_db,
                    "fwhm_factor": case.fwhm_factor,
                    "seed": case.seed,
                    "prediction_clipped": bool(config.clip_prediction_to_unit_interval),
                    "prediction_clip_min": float(config.prediction_clip_min),
                    "prediction_clip_max": float(config.prediction_clip_max),
                }

            static_stats = adapter.csv_metadata
            if static_stats:
                row.update(static_stats)

            row["status"] = "failed"
            row["error"] = "{}: {}".format(type(exc).__name__, exc)
            row["runtime_seconds"] = runtime_seconds

            results.add_row(row)

            if config.save_csv and config.flush_csv_after_each_case:
                results.append_row_to_csv(config.output_csv_path, row)

            log_case_failure(
                logger,
                case.case_id,
                runtime_seconds=runtime_seconds,
                error=exc,
            )

            if config.fail_fast:
                logger.exception("Fail-fast enabled; aborting benchmark run.")
                raise

        finally:
            metric_values = None
            prediction = None
            inputs = None
            synthetic_case = None

    if config.save_csv and not config.flush_csv_after_each_case:
        output_path = results.to_csv(config.output_csv_path)
        logger.info("Saved benchmark results to '%s'.", output_path)
    elif config.save_csv:
        logger.info(
            "Benchmark results were incrementally written to '%s'.",
            config.output_csv_path,
        )

    success_count = sum(1 for row in results.rows if row.get("status") == "success")
    failure_count = sum(1 for row in results.rows if row.get("status") == "failed")

    logger.info(
        "Benchmark run complete. total=%d success=%d failed=%d",
        len(results.rows),
        success_count,
        failure_count,
    )

    return results