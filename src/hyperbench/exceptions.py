# Author: Ritik Shah

"""Custom exceptions for HyperBench."""

from __future__ import annotations


class HyperBenchError(Exception):
    """Base exception for all HyperBench-specific errors."""


class ConfigurationError(HyperBenchError):
    """Raised when benchmark or adapter configuration is invalid."""


class IOValidationError(HyperBenchError):
    """Raised when an input file or loaded scene is invalid."""


class SceneKeyError(IOValidationError):
    """Raised when the requested MATLAB variable key is missing or ambiguous."""


class ShapeMismatchError(HyperBenchError):
    """Raised when a model prediction shape does not match expectations."""


class UnsupportedPSFError(HyperBenchError):
    """Raised when a requested PSF name is unsupported."""


class UnsupportedSRFError(HyperBenchError):
    """Raised when a requested SRF/MSI band configuration is unsupported."""


class AdapterError(HyperBenchError):
    """Raised when an adapter cannot run properly."""


class AdapterOutputError(AdapterError):
    """Raised when an adapter returns an invalid output."""


class PipelineError(AdapterError):
    """Raised when a user pipeline or run_pipeline contract fails."""


class MetricsError(HyperBenchError):
    """Raised when metric computation fails."""


class FrameworkAvailabilityError(HyperBenchError):
    """Raised when a required framework is not installed or unavailable."""