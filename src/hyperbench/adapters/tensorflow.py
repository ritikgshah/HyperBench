# Author: Ritik Shah

"""TensorFlow model adapter for HyperBench."""

from __future__ import annotations

import numpy as np

from .base import BaseAdapter, ReconstructionInputs, ShapePolicy


Array = np.ndarray


class TensorFlowModelAdapter(BaseAdapter):
    """Adapter for TensorFlow / Keras models."""

    def __init__(
        self,
        model,
        name: str = "tensorflow_model",
        shape_policy: ShapePolicy = "strict",
        hr_multiple: int = 1,
        lr_multiple: int = 1,
        add_batch_dim: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            shape_policy=shape_policy,
            hr_multiple=hr_multiple,
            lr_multiple=lr_multiple,
        )
        self.model = model
        self.add_batch_dim = add_batch_dim

        self.config.update(
            {
                "add_batch_dim": self.add_batch_dim,
            }
        )

    def predict(self, inputs: ReconstructionInputs) -> Array:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is not installed. Install it separately before using "
                "TensorFlowModelAdapter."
            ) from exc

        from hyperbench.utils import (
            numpy_hwc_to_tf_image,
            numpy_to_tf_matrix,
            tf_image_to_numpy_hwc,
        )

        prepared = self.prepare_inputs(inputs)

        lr = numpy_hwc_to_tf_image(prepared.lr_hsi, add_batch_dim=self.add_batch_dim)
        hr = numpy_hwc_to_tf_image(prepared.hr_msi, add_batch_dim=self.add_batch_dim)
        srf = numpy_to_tf_matrix(prepared.srf) if prepared.srf is not None else None
        psf = numpy_to_tf_matrix(prepared.psf) if prepared.psf is not None else None

        output = self.model(lr, hr, srf=srf, psf=psf)
        output = tf_image_to_numpy_hwc(output, remove_batch_dim=self.add_batch_dim)
        output = self._restore_output_shape(output, prepared)
        return np.asarray(output, dtype=np.float32)