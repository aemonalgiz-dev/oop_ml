"""Reusable input guards.

These functions convert loose user input into the canonical form the estimators
work with (a finite, one-dimensional ``float64`` array) and enforce the shape
invariants models rely on. Keeping them here means every model validates the
same way instead of re-deriving these checks.

Every guard takes a :class:`ValueRole` saying which column it is inspecting, so
the error it raises can name the offending input. The role is a closed
enumeration rather than a free-form string: a caller cannot invent a label, and
a typo is a type error instead of a misleading message at runtime.
"""

from enum import StrEnum

import numpy as np

from oop_ml.core.exceptions import (
    AllSameValuesError,
    EmptyValuesError,
    InvalidValuesError,
    NonBinaryLabelsError,
    NonEqualArrayLengthError,
    SingleClassError,
    TooFewValuesError,
)
from oop_ml.core.types import FloatArray, NumericInput


class ValueRole(StrEnum):
    """Which column a guard is inspecting, as it should appear in an error."""

    INPUT_VALUES = "input_values"
    TARGET_VALUES = "target_values"
    ACTUAL_VALUES = "actual_values"
    PREDICTED_VALUES = "predicted_values"
    FEATURE_VALUES = "feature_values"
    LABEL_VALUES = "label_values"
    WEIGHT_VALUES = "weight_values"


def to_float_array(values: NumericInput, role: ValueRole) -> FloatArray:
    """Coerce ``values`` to a finite, one-dimensional ``float64`` array.

    Raises
    ------
    InvalidValuesError
        If the values cannot be coerced to floats, are not one-dimensional,
        or contain NaN/inf.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidValuesError(f"{role} must be numeric") from error

    if array.ndim != 1:
        raise InvalidValuesError(f"{role} must be one-dimensional")

    if not np.all(np.isfinite(array)):
        raise InvalidValuesError(f"{role} must contain only finite values")

    return array


def check_non_empty(array: FloatArray, role: ValueRole) -> None:
    """Raise :class:`EmptyValuesError` if ``array`` has no elements."""
    if array.size == 0:
        raise EmptyValuesError(f"{role} must not be empty")


def check_has_variance(array: FloatArray, role: ValueRole) -> None:
    """Raise :class:`AllSameValuesError` if every element is identical.

    A predictor with no variance carries no information to regress on (and, for
    simple regression, drives the slope's denominator to zero). Call after
    :func:`check_non_empty` so ``array`` is guaranteed to have a first element.
    """
    if array.size and bool(np.all(array == array.flat[0])):
        raise AllSameValuesError(f"{role} must not be constant (zero variance)")


def check_equal_length(first_array: FloatArray, second_array: FloatArray) -> None:
    """Raise :class:`NonEqualArrayLengthError` if the arrays differ in length."""
    if len(first_array) != len(second_array):
        raise NonEqualArrayLengthError(
            f"arrays must have equal length, "
            f"got {len(first_array)} and {len(second_array)}"
        )


def check_min_length(array: FloatArray, minimum_length: int, role: ValueRole) -> None:
    """Raise :class:`TooFewValuesError` if ``array`` is shorter than the minimum."""
    if len(array) < minimum_length:
        raise TooFewValuesError(
            f"{role} needs at least {minimum_length} samples, got {len(array)}"
        )


def check_is_binary(values: FloatArray, role: ValueRole) -> None:
    """Raise unless every value is exactly 0 or exactly 1.

    Binary classification needs a target it can read as "the event happened" or
    "it did not". Anything else, whether that is a probability someone forgot to
    threshold or a three-class label encoded as 0, 1 and 2, would be silently
    reinterpreted by the arithmetic rather than rejected, so it is caught here.

    Raises
    ------
    NonBinaryLabelsError
        If any value is neither 0 nor 1.
    """
    if not bool(np.all((values == 0.0) | (values == 1.0))):
        offenders = np.unique(values[(values != 0.0) & (values != 1.0)])
        raise NonBinaryLabelsError(
            f"{role} must contain only 0 or 1, found "
            f"{', '.join(str(value) for value in offenders[:5])}"
        )


def check_is_label_encoded(values: FloatArray, role: ValueRole) -> None:
    """Raise unless every value is a whole class position in ``0 .. n - 1``.

    A multi-class target holds positions, not quantities. Insisting they run
    densely from zero means class ``k`` is column ``k`` of a probability matrix,
    with no lookup table to keep in step -- and a gap in the run is nearly
    always a filtered dataset rather than a decision.

    Raises
    ------
    NonBinaryLabelsError
        If a value is negative or is not a whole number.
    SingleClassError
        If fewer than two classes are present, or the classes present do not
        form a dense run from zero.
    """
    if values.size and (np.any(values < 0.0) or np.any(values != np.floor(values))):
        raise NonBinaryLabelsError(
            f"{role} must hold whole class positions starting at 0, "
            f"and holds at least one value that is negative or fractional"
        )

    present = np.unique(values)

    if present.size < 2:
        raise SingleClassError(
            f"{role} holds only class {present.tolist()}, so there is nothing "
            f"to discriminate between"
        )

    if not np.array_equal(present, np.arange(present.size, dtype=np.float64)):
        raise SingleClassError(
            f"{role} must use every class from 0 to {present.size - 1}; "
            f"got {present.tolist()}, which leaves a gap"
        )


def check_are_class_positions(
    values: FloatArray, n_classes: int, role: ValueRole
) -> None:
    """Raise unless every value is a whole class position inside a stated width.

    The looser sibling of :func:`check_is_label_encoded`, for the callers that
    already know how many classes the problem has. The dense-run rule exists to
    make an *inferred* width trustworthy; once the width is stated, whole
    positions inside it is the whole requirement -- a bootstrap resample or a
    held-out fold missing a class is ordinary, not suspect. Even a single
    class is allowed: a resample can come out that way, and a tree fitted to
    it is a lone leaf rather than an error.

    Raises
    ------
    NonBinaryLabelsError
        If a value is negative or is not a whole number.
    InvalidValuesError
        If a value names a class at or beyond ``n_classes``.
    """
    if values.size and (np.any(values < 0.0) or np.any(values != np.floor(values))):
        raise NonBinaryLabelsError(
            f"{role} must hold whole class positions starting at 0, "
            f"and holds at least one value that is negative or fractional"
        )

    if values.size and values.max() >= n_classes:
        raise InvalidValuesError(
            f"{role} names class {int(values.max())}, which is outside a "
            f"problem with {n_classes} classes"
        )


def check_has_both_classes(values: FloatArray, role: ValueRole) -> None:
    """Raise unless both 0 and 1 appear at least once.

    A target that is entirely one class has no boundary to find. Every
    coefficient could be zero and the model would still be right on every row it
    was shown, which tells you nothing and generalises to nothing.

    Raises
    ------
    SingleClassError
        If only one of the two classes is present.
    """
    if not (bool(np.any(values == 1.0)) and bool(np.any(values == 0.0))):
        present = "1" if bool(np.any(values == 1.0)) else "0"
        raise SingleClassError(
            f"{role} contains only class {present}; a classifier needs both"
        )
