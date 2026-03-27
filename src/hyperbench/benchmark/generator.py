# Author: Ritik Shah

"""Benchmark case generation utilities for HyperBench."""

from __future__ import annotations

from itertools import product

from hyperbench.utils.validation import validate_band_count, validate_downsample_ratio

from .case import BenchmarkCase, BenchmarkConfig, DegradationSpec


def _make_case_id(index: int) -> str:
    return f"case_{index:05d}"


def _validate_degradation_spec(spec: DegradationSpec) -> DegradationSpec:
    """Validate and normalize one explicit degradation spec."""
    validate_downsample_ratio(spec.downsample_ratio)
    validate_band_count(spec.msi_band_count)

    return DegradationSpec(
        downsample_ratio=int(spec.downsample_ratio),
        msi_band_count=int(spec.msi_band_count),
        spatial_snr_db=float(spec.spatial_snr_db),
        spectral_snr_db=float(spec.spectral_snr_db),
    )


def _expand_degradation_specs(config: BenchmarkConfig) -> list[DegradationSpec]:
    """Return the list of degradation specs from either explicit or sweep mode."""
    if config.degradation_specs is not None:
        if not config.degradation_specs:
            raise ValueError(
                "config.degradation_specs was provided but is empty. "
                "Provide at least one DegradationSpec."
            )
        return [_validate_degradation_spec(spec) for spec in config.degradation_specs]

    # Sweep mode
    if not config.msi_band_counts:
        raise ValueError("config.msi_band_counts must contain at least one band count")
    if not config.downsample_ratio_to_spatial_snr_db:
        raise ValueError(
            "config.downsample_ratio_to_spatial_snr_db must contain at least one mapping"
        )
    if not config.spectral_snr_dbs:
        raise ValueError("config.spectral_snr_dbs must contain at least one value")

    specs: list[DegradationSpec] = []

    for ratio, spatial_snr_db in config.downsample_ratio_to_spatial_snr_db.items():
        validate_downsample_ratio(ratio)

        for msi_band_count, spectral_snr_db in product(
            config.msi_band_counts,
            config.spectral_snr_dbs,
        ):
            validate_band_count(msi_band_count)
            specs.append(
                DegradationSpec(
                    downsample_ratio=int(ratio),
                    msi_band_count=int(msi_band_count),
                    spatial_snr_db=float(spatial_snr_db),
                    spectral_snr_db=float(spectral_snr_db),
                )
            )

    return specs


def generate_cases(config: BenchmarkConfig) -> list[BenchmarkCase]:
    """Expand a benchmark config into a list of concrete benchmark cases."""
    if not config.psf_names:
        raise ValueError("config.psf_names must contain at least one PSF name")
    if not config.psf_sigmas:
        raise ValueError("config.psf_sigmas must contain at least one sigma")
    if not config.psf_kernel_radii:
        raise ValueError("config.psf_kernel_radii must contain at least one kernel radius")

    degradation_specs = _expand_degradation_specs(config)

    cases: list[BenchmarkCase] = []

    combinations = product(
        config.psf_names,
        config.psf_sigmas,
        config.psf_kernel_radii,
        degradation_specs,
    )

    for index, (
        psf_name,
        psf_sigma,
        psf_kernel_radius,
        degradation_spec,
    ) in enumerate(combinations):
        cases.append(
            BenchmarkCase(
                case_id=_make_case_id(index),
                scene_path=config.scene_path,
                scene_key=config.scene_key,
                psf_name=psf_name,
                psf_sigma=float(psf_sigma),
                psf_kernel_radius=int(psf_kernel_radius),
                downsample_ratio=int(degradation_spec.downsample_ratio),
                msi_band_count=int(degradation_spec.msi_band_count),
                spatial_snr_db=float(degradation_spec.spatial_snr_db),
                spectral_snr_db=float(degradation_spec.spectral_snr_db),
                fwhm_factor=float(config.fwhm_factor),
                seed=int(config.seed),
                normalize_inputs=bool(config.normalize_inputs),
                lower_percentile=float(config.lower_percentile),
                upper_percentile=float(config.upper_percentile),
                user_srf=config.user_srf,
                user_psf=config.user_psf,
                metadata=dict(config.metadata),
            )
        )

    return cases