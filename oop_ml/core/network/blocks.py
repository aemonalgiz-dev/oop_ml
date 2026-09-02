"""Reading a block of any rank as float64, which every layer does first.

Why this is one module and not a line in each layer
----------------------------------------------------
Every entry point in the network package reaches straight for ``.ndim`` or
``.shape``, which turns an ordinary mistake, handing in a list of rows, or a
list of pictures, into a bare ``AttributeError`` from numpy rather than a typed
refusal. Coercing first keeps every failure inside the ``MLLibError`` hierarchy
and lets a nested list work, which is what a caller writing a small example
reasonably expects. That coercion was written identically in three modules and
the finiteness scan in two, byte for byte, which is the argument for one copy.

:func:`~oop_ml.core.validation.to_float_array` is the library's coercion
boundary for a *column* and it is deliberately not reused here. It refuses
anything that is not one-dimensional, and a layer's block is anything from a row
to a stack of pictures. It also names its column with a closed
:class:`~oop_ml.core.validation.ValueRole`, where a layer's roles are phrases
such as "a flattening layer's arriving slopes" that only ever appear in a
message. A role here is a label for an error, not a key that selects behaviour,
so it stays a plain string.

Why the finiteness scan is separate from the coercion
------------------------------------------------------
Because not every block should pay for it. A layer scans what it is *given*,
and deliberately not what it *produces*: a finite block times finite weights
can still overflow, and that overflow is refused at the next layer, one join
later than it began, rather than every block being scanned twice. Restoring the
scan on the way in, after the matrix multiply had silently taken it away, was
measured against the multiply it guards at 1.2% at 1024x64 into 128, 4.3% at
256x32 into 64, and 29.6% on a 32x8 toy where the multiply is microseconds.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.types import FloatArray


def as_block(values: object, role: str) -> FloatArray:
    """Read ``values`` as a private float64 copy of any rank.

    Raises
    ------
    InvalidValuesError
        If the values cannot be read as a float array at all.
    """
    try:
        return np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise InvalidValuesError(f"{role} must be readable as a float array") from error


def check_finite(values: FloatArray, role: str) -> None:
    """Refuse a block carrying a non-finite entry.

    Raises
    ------
    InvalidValuesError
        If any entry is ``nan`` or infinite.
    """
    if not np.isfinite(values).all():
        raise InvalidValuesError(f"{role} must contain only finite values")


def as_per_feature(values: object, n_features: int, role: str) -> FloatArray:
    """Read one per-feature vector, frozen, refusing in the library's own words.

    Raises
    ------
    InvalidValuesError
        If it cannot be read as a float array, or carries a non-finite entry.
    ShapeMismatchError
        If it is not one-dimensional with exactly ``n_features`` entries.
    """
    block = as_block(values, role)

    if block.ndim != 1 or block.shape[0] != n_features:
        raise ShapeMismatchError(
            f"{role} is one value per feature, so ({n_features},), got {block.shape}"
        )
    check_finite(block, role)

    block.setflags(write=False)
    return block
