"""Shared type aliases.

Isolated in their own module so every part of the library speaks the same
vocabulary for numeric input without importing the estimator classes.
"""

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Numeric: TypeAlias = int | float
NumericValues: TypeAlias = Sequence[Numeric]
FloatArray: TypeAlias = NDArray[np.float64]
# Row positions, for selecting a subset of observations.
IndexArray: TypeAlias = NDArray[np.intp]
# What public methods accept: a plain sequence of numbers OR a float array.
NumericInput: TypeAlias = NumericValues | FloatArray
