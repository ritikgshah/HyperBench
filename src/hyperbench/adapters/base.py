# Author: Ritik Shah

"""Base adapter abstractions for HyperBench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np


Array = np.ndarray
ShapePolicy = Literal["strict", "crop", "pad"]


@dataclass
class ReconstructionInputs:
    """Canonical inference-time input bundle for HyperBench."""

    lr_hsi: Array
    hr_msi: Array
    srf: Optional[Array] = None
    psf: Optional[Array] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAdapter:
    """Base class for all HyperBench adapters."""

    def __init__(
        self,
        name: str,
        shape_policy: ShapePolicy = "strict",
        hr_multiple: int = 1,
        lr_multiple: int = 1,
    ) -> None:
        if shape_policy not in {"strict", "crop", "pad"}:
            raise ValueError(
                "shape_policy must be one of {'strict', 'crop', 'pad'}, got {!r}".format(
                    shape_policy
                )
            )

        self.name = name
        self.shape_policy = shape_policy
        self.hr_multiple = int(hr_multiple)
        self.lr_multiple = int(lr_multiple)

        self.config = {
            "adapter_name": self.name,
            "adapter_type": type(self).__name__,
            "shape_policy": self.shape_policy,
            "hr_multiple": self.hr_multiple,
            "lr_multiple": self.lr_multiple,
        }

    def predict(self, inputs: ReconstructionInputs) -> Any:
        """Run model inference for one HyperBench case.

        Valid returns:
        - prediction
        - (prediction, stats_dict)
        """
        raise NotImplementedError

    def _validate_multiple(self, value: int, name: str) -> None:
        if value < 1:
            raise ValueError("{} must be >= 1, got {}".format(name, value))

    def prepare_inputs(self, inputs: ReconstructionInputs) -> ReconstructionInputs:
        """Apply shape policy before inference."""
        self._validate_multiple(self.hr_multiple, "hr_multiple")
        self._validate_multiple(self.lr_multiple, "lr_multiple")

        if self.shape_policy == "strict":
            self._ensure_valid_shapes(inputs)
            return inputs
        elif self.shape_policy == "crop":
            return self._crop_inputs_to_valid(inputs)
        elif self.shape_policy == "pad":
            return self._pad_inputs_to_valid(inputs)
        else:
            raise ValueError("Unsupported shape_policy {!r}".format(self.shape_policy))

    def _ensure_valid_shapes(self, inputs: ReconstructionInputs) -> None:
        hr_h, hr_w = inputs.hr_msi.shape[:2]
        lr_h, lr_w = inputs.lr_hsi.shape[:2]

        if hr_h % self.hr_multiple != 0 or hr_w % self.hr_multiple != 0:
            raise ValueError(
                "HR input shape {} is not divisible by hr_multiple={}".format(
                    inputs.hr_msi.shape[:2], self.hr_multiple
                )
            )

        if lr_h % self.lr_multiple != 0 or lr_w % self.lr_multiple != 0:
            raise ValueError(
                "LR input shape {} is not divisible by lr_multiple={}".format(
                    inputs.lr_hsi.shape[:2], self.lr_multiple
                )
            )

    def _crop_inputs_to_valid(self, inputs: ReconstructionInputs) -> ReconstructionInputs:
        hr_h, hr_w = inputs.hr_msi.shape[:2]
        lr_h, lr_w = inputs.lr_hsi.shape[:2]

        new_hr_h = hr_h - (hr_h % self.hr_multiple)
        new_hr_w = hr_w - (hr_w % self.hr_multiple)
        new_lr_h = lr_h - (lr_h % self.lr_multiple)
        new_lr_w = lr_w - (lr_w % self.lr_multiple)

        if new_hr_h <= 0 or new_hr_w <= 0:
            raise ValueError("Cropping would result in invalid HR shape")
        if new_lr_h <= 0 or new_lr_w <= 0:
            raise ValueError("Cropping would result in invalid LR shape")

        return ReconstructionInputs(
            lr_hsi=inputs.lr_hsi[:new_lr_h, :new_lr_w, :],
            hr_msi=inputs.hr_msi[:new_hr_h, :new_hr_w, :],
            srf=inputs.srf,
            psf=inputs.psf,
            metadata=dict(inputs.metadata),
        )

    def _compute_padding(self, size: int, multiple: int) -> int:
        remainder = size % multiple
        return 0 if remainder == 0 else multiple - remainder

    def _pad_inputs_to_valid(self, inputs: ReconstructionInputs) -> ReconstructionInputs:
        lr_h, lr_w, lr_c = inputs.lr_hsi.shape
        hr_h, hr_w, hr_c = inputs.hr_msi.shape

        pad_lr_h = self._compute_padding(lr_h, self.lr_multiple)
        pad_lr_w = self._compute_padding(lr_w, self.lr_multiple)
        pad_hr_h = self._compute_padding(hr_h, self.hr_multiple)
        pad_hr_w = self._compute_padding(hr_w, self.hr_multiple)

        lr_padded = np.pad(
            inputs.lr_hsi,
            ((0, pad_lr_h), (0, pad_lr_w), (0, 0)),
            mode="reflect",
        ).astype(np.float32)

        hr_padded = np.pad(
            inputs.hr_msi,
            ((0, pad_hr_h), (0, pad_hr_w), (0, 0)),
            mode="reflect",
        ).astype(np.float32)

        metadata = dict(inputs.metadata)
        metadata["_original_lr_shape"] = (lr_h, lr_w, lr_c)
        metadata["_original_hr_shape"] = (hr_h, hr_w, hr_c)

        return ReconstructionInputs(
            lr_hsi=lr_padded,
            hr_msi=hr_padded,
            srf=inputs.srf,
            psf=inputs.psf,
            metadata=metadata,
        )

    def _restore_output_shape(self, output: Array, inputs: ReconstructionInputs) -> Array:
        """Restore output to original HR shape when using pad policy."""
        if self.shape_policy != "pad":
            return output

        original_hr_shape = inputs.metadata.get("_original_hr_shape")
        if original_hr_shape is None:
            return output

        h, w, c = original_hr_shape
        return output[:h, :w, :c]

    @property
    def csv_metadata(self) -> Dict[str, Any]:
        """Optional static metadata to merge into CSV rows."""
        return {}