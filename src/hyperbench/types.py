# Author: Ritik Shah

"""Shared type aliases for HyperBench."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np


Array = np.ndarray
PathLike = Union[str, Path]
Metadata = Dict[str, Any]
MappingStrAny = Mapping[str, Any]
MutableMappingStrAny = MutableMapping[str, Any]

Shape2D = Tuple[int, int]
Shape3D = Tuple[int, int, int]

Numeric = Union[int, float, np.number]
OptionalArray = Optional[Array]
SequenceStr = Sequence[str]