# Author: Ritik Shah

"""Callable adapter for HyperBench."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np

from hyperbench.utils import numpy_prediction_to_hwc

from .base import BaseAdapter, ReconstructionInputs, ShapePolicy


Array = np.ndarray


class CallableAdapter(BaseAdapter):
    """Wrap a plain Python callable as a HyperBench adapter."""

    def __init__(
        self,
        fn: Callable,
        name: str = "callable_model",
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
        self.fn = fn

    def predict(self, inputs: ReconstructionInputs) -> Any:
        prepared = self.prepare_inputs(inputs)

        result = self.fn(
            prepared.lr_hsi,
            prepared.hr_msi,
            srf=prepared.srf,
            psf=prepared.psf,
            metadata=prepared.metadata,
        )

        if isinstance(result, tuple):
            if len(result) != 2 or not isinstance(result[1], dict):
                raise ValueError(
                    "CallableAdapter functions may return either prediction or "
                    "(prediction, stats_dict)."
                )
            prediction, stats = result
            prediction = numpy_prediction_to_hwc(prediction, remove_batch_dim=True)
            prediction = self._restore_output_shape(prediction, prepared)
            return prediction.astype(np.float32), stats

        prediction = numpy_prediction_to_hwc(result, remove_batch_dim=True)
        prediction = self._restore_output_shape(prediction, prepared)
        return prediction.astype(np.float32)