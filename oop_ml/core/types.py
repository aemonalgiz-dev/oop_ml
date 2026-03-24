"""Shared type aliases.

Isolated in their own module so every part of the library speaks the same
vocabulary for numeric input without importing the estimator classes.
"""

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import DTypeLike, NDArray

Numeric: TypeAlias = int | float
NumericValues: TypeAlias = Sequence[Numeric]
FloatArray: TypeAlias = NDArray[np.float64]
# Row positions, for selecting a subset of observations.
IndexArray: TypeAlias = NDArray[np.intp]
# Which rows satisfy a condition, for splitting a set of observations in two.
MaskArray: TypeAlias = NDArray[np.bool_]
# What public methods accept: a plain sequence of numbers OR a float array.
NumericInput: TypeAlias = NumericValues | FloatArray


def array_for_protocol(
    values: FloatArray, dtype: DTypeLike | None = None, copy: bool | None = None
) -> FloatArray:
    """The numpy-2 ``__array__`` contract, implemented once for every wrapper.

    numpy trusts an object whose ``__array__`` accepts ``copy`` to honour it,
    and adds no copy of its own -- so a body that declares the parameter and
    ignores it hands ``np.array(wrapper)`` (whose default is ``copy=True``) the
    internal buffer while the caller believes they hold a private copy. For a
    frozen buffer their first write then raises a bare read-only ValueError
    from an ordinary idiom; for a writable one it silently corrupts the
    validated object. Measured on numpy 2.5.1 before this existed:
    ``np.shares_memory(np.array(p), p.values)`` was ``True``.

    The three cases, per the protocol:

    * ``copy=True`` -- always return a fresh array.
    * ``copy=None`` -- a view is allowed; copy only if the dtype forces it.
    * ``copy=False`` -- a view is required; raise if the dtype forces a copy.
    """
    if dtype is not None and dtype != values.dtype:
        if copy is False:
            raise ValueError(
                "a copy is required to honour the requested dtype, and "
                "copy=False forbids one"
            )

        return values.astype(dtype)

    if copy:
        return values.copy()

    return values
