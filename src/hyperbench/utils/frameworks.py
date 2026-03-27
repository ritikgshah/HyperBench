# Author: Ritik Shah

"""Framework conversion and device helpers for HyperBench."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


Array = np.ndarray


def is_torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def is_tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def get_torch_device_info() -> Dict[str, Any]:
    info = {
        "available": False,
        "gpu_available": False,
        "gpu_count": 0,
        "device": "cpu",
        "device_names": [],
    }

    try:
        import torch  # type: ignore
    except ImportError:
        return info

    info["available"] = True

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        info["gpu_available"] = True
        info["gpu_count"] = count
        info["device"] = "cuda"
        info["device_names"] = [torch.cuda.get_device_name(i) for i in range(count)]

    return info


def get_tensorflow_device_info() -> Dict[str, Any]:
    info = {
        "available": False,
        "gpu_available": False,
        "gpu_count": 0,
        "device": "/CPU:0",
        "device_names": [],
    }

    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        return info

    info["available"] = True

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        info["gpu_available"] = True
        info["gpu_count"] = len(gpus)
        info["device"] = "/GPU:0"
        info["device_names"] = [d.name for d in gpus]

    return info


def get_preferred_torch_device() -> str:
    return get_torch_device_info()["device"]


def get_preferred_tensorflow_device() -> str:
    return get_tensorflow_device_info()["device"]


def print_framework_device_summary() -> None:
    torch_info = get_torch_device_info()
    tf_info = get_tensorflow_device_info()

    print("Framework device summary")
    print("-" * 40)

    print("PyTorch:")
    print("  installed: {}".format(torch_info["available"]))
    print("  gpu_available: {}".format(torch_info["gpu_available"]))
    print("  gpu_count: {}".format(torch_info["gpu_count"]))
    print("  preferred_device: {}".format(torch_info["device"]))
    if torch_info["device_names"]:
        for i, name in enumerate(torch_info["device_names"]):
            print("  gpu[{}]: {}".format(i, name))

    print()
    print("TensorFlow:")
    print("  installed: {}".format(tf_info["available"]))
    print("  gpu_available: {}".format(tf_info["gpu_available"]))
    print("  gpu_count: {}".format(tf_info["gpu_count"]))
    print("  preferred_device: {}".format(tf_info["device"]))
    if tf_info["device_names"]:
        for i, name in enumerate(tf_info["device_names"]):
            print("  gpu[{}]: {}".format(i, name))


def _validate_hwc_image(array: Array, name: str = "array") -> Array:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("{} must have shape (H, W, C), got {}".format(name, array.shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains non-finite values".format(name))
    return array


def _validate_2d_array(array: Array, name: str = "array") -> Array:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("{} must have shape (N, M), got {}".format(name, array.shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains non-finite values".format(name))
    return array


# ---------------------------------------------------------------------
# TensorFlow helpers
# ---------------------------------------------------------------------
def numpy_hwc_to_tf_image(array: Array, add_batch_dim: bool = False) -> Any:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is not installed. Install it separately before using "
            "TensorFlow conversion helpers."
        ) from exc

    array = _validate_hwc_image(array, name="array")
    if add_batch_dim:
        array = np.expand_dims(array, axis=0)
    return tf.convert_to_tensor(array, dtype=tf.float32)


def tf_image_to_numpy_hwc(tensor: Any, remove_batch_dim: bool = False) -> Array:
    array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    array = np.asarray(array, dtype=np.float32)

    if remove_batch_dim:
        if array.ndim != 4 or array.shape[0] != 1:
            raise ValueError(
                "Expected batched TensorFlow image tensor with shape (1, H, W, C), "
                "got {}".format(array.shape)
            )
        array = array[0]

    return _validate_hwc_image(array, name="tensor")


def numpy_to_tf_matrix(array: Array) -> Any:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is not installed. Install it separately before using "
            "TensorFlow conversion helpers."
        ) from exc

    array = _validate_2d_array(array, name="array")
    return tf.convert_to_tensor(array, dtype=tf.float32)


def tf_matrix_to_numpy(array: Any) -> Array:
    out = array.numpy() if hasattr(array, "numpy") else np.asarray(array)
    return _validate_2d_array(out, name="array")


# ---------------------------------------------------------------------
# PyTorch helpers
# ---------------------------------------------------------------------
def numpy_hwc_to_torch_image(array: Array, add_batch_dim: bool = False, device: str = "cpu") -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Install it separately before using "
            "PyTorch conversion helpers."
        ) from exc

    array = _validate_hwc_image(array, name="array")
    array = np.transpose(array, (2, 0, 1))

    if add_batch_dim:
        array = np.expand_dims(array, axis=0)

    return torch.as_tensor(array, dtype=torch.float32, device=device)


def torch_image_to_numpy_hwc(tensor: Any, remove_batch_dim: bool = False) -> Array:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Install it separately before using "
            "PyTorch conversion helpers."
        ) from exc

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor, dtype=torch.float32)

    array = tensor.detach().to("cpu").float().numpy()

    if remove_batch_dim:
        if array.ndim != 4 or array.shape[0] != 1:
            raise ValueError(
                "Expected batched PyTorch image tensor with shape (1, C, H, W), got {}".format(
                    array.shape
                )
            )
        array = array[0]

    if array.ndim != 3:
        raise ValueError(
            "Expected unbatched PyTorch image tensor with shape (C, H, W), got {}".format(
                array.shape
            )
        )

    array = np.transpose(array, (1, 2, 0))
    return _validate_hwc_image(array, name="tensor")


def numpy_to_torch_matrix(array: Array, device: str = "cpu") -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Install it separately before using "
            "PyTorch conversion helpers."
        ) from exc

    array = _validate_2d_array(array, name="array")
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def torch_matrix_to_numpy(array: Any) -> Array:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Install it separately before using "
            "PyTorch conversion helpers."
        ) from exc

    if not isinstance(array, torch.Tensor):
        array = torch.as_tensor(array, dtype=torch.float32)

    out = array.detach().to("cpu").float().numpy()
    return _validate_2d_array(out, name="array")


# ---------------------------------------------------------------------
# Generic helpers for adapter outputs
# ---------------------------------------------------------------------
def numpy_prediction_to_hwc(array: Array, remove_batch_dim: bool = True) -> Array:
    """Normalize a NumPy prediction into HyperBench HWC format.

    Accepts:
    - HWC
    - 1HWC (batch size 1)
    """
    array = np.asarray(array, dtype=np.float32)

    if remove_batch_dim and array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(
                "Expected NumPy prediction with batch size 1 when ndim=4, got {}".format(
                    array.shape
                )
            )
        array = array[0]

    return _validate_hwc_image(array, name="prediction")


def convert_prediction_to_numpy_hwc(
    prediction: Any,
    output_backend: str,
    remove_batch_dim: bool = True,
) -> Array:
    """Convert a prediction from numpy / tensorflow / torch into HWC NumPy."""
    backend = output_backend.lower()

    if backend == "numpy":
        return numpy_prediction_to_hwc(prediction, remove_batch_dim=remove_batch_dim)
    elif backend == "tensorflow":
        return tf_image_to_numpy_hwc(prediction, remove_batch_dim=remove_batch_dim)
    elif backend == "torch":
        return torch_image_to_numpy_hwc(prediction, remove_batch_dim=remove_batch_dim)
    else:
        raise ValueError(
            "Unsupported output_backend {!r}. Expected one of {'numpy', 'tensorflow', 'torch'}.".format(
                output_backend
            )
        )