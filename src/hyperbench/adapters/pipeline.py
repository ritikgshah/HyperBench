# Author: Ritik Shah

"""Generic pipeline adapter for HyperBench.

This adapter is the recommended path for real-world research models.

Expected user-side contract:
- an object with a `run_pipeline(...)` method, or
- a callable function itself

The pipeline should accept:
    run_pipeline(HR_MSI, LR_HSI, srf, psf=None, metadata=None)

Valid returns:
- prediction
- (prediction, stats_dict)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from hyperbench.utils import (
    convert_prediction_to_numpy_hwc,
    get_preferred_tensorflow_device,
    get_preferred_torch_device,
    numpy_hwc_to_tf_image,
    numpy_hwc_to_torch_image,
    numpy_to_tf_matrix,
    numpy_to_torch_matrix,
    numpy_prediction_to_hwc,
)

from .base import BaseAdapter, ReconstructionInputs, ShapePolicy


Array = np.ndarray


class PipelineAdapter(BaseAdapter):
    """Generic adapter for model wrappers exposing `run_pipeline(...)`."""

    def __init__(
        self,
        pipeline: Any,
        name: str = "pipeline_model",
        input_backend: str = "numpy",
        output_backend: Optional[str] = None,
        add_batch_dim: bool = True,
        device: str = "auto",
        shape_policy: ShapePolicy = "strict",
        hr_multiple: int = 1,
        lr_multiple: int = 1,
    ) -> None:
        super().__init__(
            name=name,
            shape_policy=shape_policy,
            hr_multiple=hr_multiple,
            lr_multiple=lr_multiple,
        )

        backend = input_backend.lower()
        if backend not in {"numpy", "tensorflow", "torch"}:
            raise ValueError(
                "input_backend must be one of {'numpy', 'tensorflow', 'torch'}, got {!r}".format(
                    input_backend
                )
            )

        out_backend = output_backend.lower() if output_backend is not None else backend
        if out_backend not in {"numpy", "tensorflow", "torch"}:
            raise ValueError(
                "output_backend must be one of {'numpy', 'tensorflow', 'torch'}, got {!r}".format(
                    output_backend
                )
            )

        self.pipeline = pipeline
        self.input_backend = backend
        self.output_backend = out_backend
        self.add_batch_dim = bool(add_batch_dim)
        self.device = device

        self.config.update(
            {
                "input_backend": self.input_backend,
                "output_backend": self.output_backend,
                "add_batch_dim": self.add_batch_dim,
                "device": self.device,
            }
        )

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device

        if self.input_backend == "torch":
            return get_preferred_torch_device()
        elif self.input_backend == "tensorflow":
            return get_preferred_tensorflow_device()
        else:
            return "cpu"

    def _prepare_image_input(self, image: Array, device: str) -> Any:
        if self.input_backend == "numpy":
            out = np.asarray(image, dtype=np.float32)
            if self.add_batch_dim:
                out = np.expand_dims(out, axis=0)
            return out
        elif self.input_backend == "tensorflow":
            return numpy_hwc_to_tf_image(image, add_batch_dim=self.add_batch_dim)
        elif self.input_backend == "torch":
            return numpy_hwc_to_torch_image(
                image,
                add_batch_dim=self.add_batch_dim,
                device=device,
            )
        else:
            raise ValueError("Unsupported input_backend {!r}".format(self.input_backend))

    def _prepare_matrix_input(self, matrix: Optional[Array], device: str) -> Any:
        if matrix is None:
            return None

        if self.input_backend == "numpy":
            return np.asarray(matrix, dtype=np.float32)
        elif self.input_backend == "tensorflow":
            return numpy_to_tf_matrix(matrix)
        elif self.input_backend == "torch":
            return numpy_to_torch_matrix(matrix, device=device)
        else:
            raise ValueError("Unsupported input_backend {!r}".format(self.input_backend))

    def _get_pipeline_callable(self):
        if hasattr(self.pipeline, "run_pipeline"):
            return self.pipeline.run_pipeline
        elif callable(self.pipeline):
            return self.pipeline
        else:
            raise ValueError(
                "pipeline must be a callable or an object exposing run_pipeline(...)."
            )

    def _normalize_result(self, result: Any, prepared: ReconstructionInputs):
        if isinstance(result, tuple):
            if len(result) != 2 or not isinstance(result[1], dict):
                raise ValueError(
                    "PipelineAdapter pipelines may return either prediction or "
                    "(prediction, stats_dict)."
                )
            prediction, stats = result
        else:
            prediction, stats = result, None

        if self.output_backend == "numpy":
            pred_np = numpy_prediction_to_hwc(prediction, remove_batch_dim=self.add_batch_dim)
        else:
            pred_np = convert_prediction_to_numpy_hwc(
                prediction,
                output_backend=self.output_backend,
                remove_batch_dim=self.add_batch_dim,
            )

        pred_np = self._restore_output_shape(pred_np, prepared)
        pred_np = np.asarray(pred_np, dtype=np.float32)

        if stats is None:
            return pred_np
        return pred_np, stats

    def predict(self, inputs: ReconstructionInputs) -> Any:
        prepared = self.prepare_inputs(inputs)
        device = self._resolve_device()

        LR_HSI = self._prepare_image_input(prepared.lr_hsi, device=device)
        HR_MSI = self._prepare_image_input(prepared.hr_msi, device=device)
        srf = self._prepare_matrix_input(prepared.srf, device=device)
        psf = self._prepare_matrix_input(prepared.psf, device=device)

        pipeline_fn = self._get_pipeline_callable()

        result = pipeline_fn(
            HR_MSI=HR_MSI,
            LR_HSI=LR_HSI,
            srf=srf,
            psf=psf,
            metadata=prepared.metadata,
        )

        return self._normalize_result(result, prepared)