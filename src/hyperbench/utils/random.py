# Author: Ritik Shah

"""Reproducibility utilities for HyperBench."""

from __future__ import annotations

import os
import random

import numpy as np


def set_random_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """Set random seeds across supported libraries.

    This function always seeds:
    - Python's ``random``
    - NumPy

    If installed, it will also try to seed:
    - PyTorch
    - TensorFlow

    Parameters
    ----------
    seed:
        Random seed value.
    deterministic:
        Whether to request deterministic behavior where supported.
        This is best-effort and depends on the installed backend.
    """
    if not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (optional)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # TensorFlow (optional)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)

        if deterministic:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Create a NumPy random generator.

    Parameters
    ----------
    seed:
        Optional integer seed. If None, NumPy will use fresh entropy.

    Returns
    -------
    numpy.random.Generator
        Generator instance for local reproducible randomness.
    """
    if seed is not None and not isinstance(seed, int):
        raise ValueError(f"seed must be an integer or None, got {seed!r}")

    return np.random.default_rng(seed)