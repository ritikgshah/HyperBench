# Author: Ritik Shah

"""Configuration loading utilities for HyperBench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from hyperbench.benchmark.case import BenchmarkConfig, DegradationSpec
from hyperbench.exceptions import ConfigurationError


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigurationError(
            "Failed to parse JSON config '{}': {}".format(path, exc)
        ) from exc


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ConfigurationError(
            "YAML config requested but PyYAML is not installed. "
            "Install it separately or use JSON."
        ) from exc

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigurationError(
            "Failed to parse YAML config '{}': {}".format(path, exc)
        ) from exc

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ConfigurationError(
            "Top-level config in '{}' must be a mapping/dictionary.".format(path)
        )

    return data


def load_config_dict(path) -> Dict[str, Any]:
    """Load a config dictionary from JSON or YAML."""
    path = Path(path)

    if not path.exists():
        raise ConfigurationError("Config file does not exist: '{}'".format(path))

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    elif suffix in {".yaml", ".yml"}:
        return _load_yaml(path)
    else:
        raise ConfigurationError(
            "Unsupported config extension '{}'. Use .json, .yaml, or .yml.".format(suffix)
        )


def _parse_degradation_specs(raw_specs):
    if raw_specs is None:
        return None

    if not isinstance(raw_specs, list):
        raise ConfigurationError("degradation_specs must be a list of dictionaries.")

    specs = []
    for i, item in enumerate(raw_specs):
        if not isinstance(item, dict):
            raise ConfigurationError(
                "degradation_specs[{}] must be a dictionary.".format(i)
            )

        try:
            spec = DegradationSpec(
                downsample_ratio=item["downsample_ratio"],
                msi_band_count=item["msi_band_count"],
                spatial_snr_db=item["spatial_snr_db"],
                spectral_snr_db=item.get("spectral_snr_db", 40.0),
            )
        except KeyError as exc:
            raise ConfigurationError(
                "Missing key {} in degradation_specs[{}].".format(exc, i)
            ) from exc

        specs.append(spec)

    return specs


def benchmark_config_from_dict(data: Dict[str, Any]) -> BenchmarkConfig:
    """Construct a BenchmarkConfig from a plain dictionary."""
    if not isinstance(data, dict):
        raise ConfigurationError("Config data must be a dictionary.")

    payload = dict(data)

    payload["degradation_specs"] = _parse_degradation_specs(
        payload.get("degradation_specs")
    )

    try:
        return BenchmarkConfig(**payload)
    except TypeError as exc:
        raise ConfigurationError(
            "Failed to construct BenchmarkConfig: {}".format(exc)
        ) from exc


def load_benchmark_config(path) -> BenchmarkConfig:
    """Load a BenchmarkConfig from JSON or YAML."""
    data = load_config_dict(path)
    return benchmark_config_from_dict(data)