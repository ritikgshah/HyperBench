# Author: Ritik Shah

"""PyTorch model adapter for HyperBench."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseAdapter, ReconstructionInputs, ShapePolicy


Array = np.ndarray


class TorchModelAdapter(BaseAdapter):
    """Adapter for PyTorch models.

    This adapter assumes the wrapped model is already instantiated and ready
    for inference. Framework installation remains the user's responsibility.
    """

    def __init__(
        self,
        model,
        device: str = "cpu",
        name: str = "torch_model",
        shape_policy: ShapePolicy = "strict",
        hr_multiple: int = 1,
        lr_multiple: int = 1,
        input_layout: str = "hwc",
        output_layout: str = "hwc",
        add_batch_dim: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            shape_policy=shape_policy,
            hr_multiple=hr_multiple,
            lr_multiple=lr_multiple,
        )
        self.model = model
        self.device = device
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.add_batch_dim = add_batch_dim

        self.config.update(
            {
                "device": self.device,
                "input_layout": self.input_layout,
                "output_layout": self.output_layout,
                "add_batch_dim": self.add_batch_dim,
            }
        )

    def predict(self, inputs: ReconstructionInputs) -> Array:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is not installed. Install it separately before using "
                "TorchModelAdapter."
            ) from exc

        from hyperbench.utils import (
            numpy_hwc_to_torch_image,
            numpy_to_torch_matrix,
            torch_image_to_numpy_hwc,
        )

        prepared = self.prepare_inputs(inputs)

        self.model.eval()
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        lr = numpy_hwc_to_torch_image(
            prepared.lr_hsi,
            add_batch_dim=self.add_batch_dim,
            device=self.device,
        )
        hr = numpy_hwc_to_torch_image(
            prepared.hr_msi,
            add_batch_dim=self.add_batch_dim,
            device=self.device,
        )

        srf = (
            numpy_to_torch_matrix(prepared.srf, device=self.device)
            if prepared.srf is not None
            else None
        )
        psf = (
            numpy_to_torch_matrix(prepared.psf, device=self.device)
            if prepared.psf is not None
            else None
        )

        with torch.no_grad():
            output = self.model(lr, hr, srf=srf, psf=psf)

        output = torch_image_to_numpy_hwc(
            output,
            remove_batch_dim=self.add_batch_dim,
        )
        output = self._restore_output_shape(output, prepared)
        return output.astype(np.float32)