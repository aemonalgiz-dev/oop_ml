"""Validate at the numerical boundary, including callers outside the API."""

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.types import FloatArray


def finite(values: object, ndim: int, name: str) -> FloatArray:
    """Copy finite arrays so operations cannot mutate a caller's parameters."""
    try:
        result = np.array(values, dtype=float, copy=True)
    except (ValueError, TypeError):
        raise InvalidValuesError(f"{name} must contain real numbers") from None
    if result.ndim != ndim:
        raise ShapeMismatchError(f"{name} must have {ndim} dimensions")
    if not result.size or not np.isfinite(result).all():
        raise InvalidValuesError(f"{name} must be nonempty and finite")
    return result


def positive(value: float, name: str) -> float:
    """A denominator or learning rate must be finite and strictly positive."""
    if not np.isfinite(value) or value <= 0:
        raise InvalidValuesError(f"{name} must be finite and positive")
    return float(value)


def count(value: int, name: str, minimum: int = 1) -> int:
    """Reject fractional counts rather than silently truncating them."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise InvalidValuesError(f"{name} must be an integer")
    if value < minimum:
        raise InvalidValuesError(f"{name} must be at least {minimum}")
    return int(value)
