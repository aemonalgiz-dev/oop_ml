"""A named column of observations, which is the alternative to a design matrix.

An anonymous ``(n_samples, n_features)`` matrix asks you to remember what column
three was, and I did not want that memory to live outside the model. A
:class:`Feature` pairs a name with a validated vector of values, and that buys
two things a bare numpy column cannot.

The first is coefficients addressed by name rather than by position. A model
built from features exposes ``model.coefficients["age"]`` instead of
``model.coef_[2]``, so the variable keeps its identity the whole way through
fitting and out the other side.

The second is alignment errors phrased in your vocabulary. When two columns
disagree in length, the model can tell you that age has four rows while price
has five, rather than surfacing a raw shape mismatch from somewhere deep inside
numpy.

A :class:`Feature` is an immutable value object. Its values are validated and
frozen at construction, and two features are equal exactly when they share a
name and the same values. It carries no learning logic of its own, since
transforms such as standardization belong with preprocessing.

I made this a plain class rather than a Pydantic model, which runs against the
convention elsewhere in the library, for three concrete reasons that this
particular contract runs into.

Pydantic compares models by their field dictionary, and with a numpy array
sitting in there, ``__eq__`` ends up evaluating an array and raising "truth
value of an array with more than one element is ambiguous" rather than returning
a bool. For the same reason the generated ``__hash__`` fails outright, since a
numpy array is unhashable, so equal features could neither share a hash nor live
in a set. And Pydantic's ``__init__`` is keyword-only, which would turn every
``Feature("age", values)`` into ``Feature(name="age", values=values)``.

The Pydantic convention in this repo is aimed at estimators, where
hyperparameters genuinely do need validating at construction. Value objects such
as this one validate themselves.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.data.column import Column
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.types import FloatArray, NumericInput
from oop_ml.core.validation import ValueRole


class Feature:
    """A named, finite, one-dimensional column of ``float64`` observations.

    Parameters
    ----------
    name:
        A non-empty label identifying the variable (surrounding whitespace is
        stripped). This is what a model uses to key the fitted coefficient.
    values:
        The observations. Coerced to a finite, one-dimensional ``float64`` array
        and then frozen, so a constructed feature can never silently change.

    Raises
    ------
    InvalidValuesError
        If ``name`` is not a non-empty string, or ``values`` cannot be coerced
        to a finite one-dimensional float array.
    EmptyValuesError
        If ``values`` contains no observations.
    """

    __slots__ = ("_column", "_name")

    def __init__(self, name: str, values: NumericInput) -> None:
        if not isinstance(name, str) or not name.strip():
            raise InvalidValuesError("Feature name must be a non-empty string")

        # A feature *is* a column that knows its own name: Column owns the
        # coercion, copying, freezing, and emptiness check, so none of that is
        # repeated here.
        self._name = name.strip()
        self._column = Column.of(values, ValueRole.FEATURE_VALUES)

    @property
    def name(self) -> str:
        """The variable's identifying label."""
        return self._name

    @property
    def column(self) -> Column:
        """The underlying validated column of observations."""
        return self._column

    @property
    def values(self) -> FloatArray:
        """The observations as a read-only ``float64`` array."""
        return self._column.values

    @property
    def n_samples(self) -> int:
        """Number of observations in the column."""
        return self._column.n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Feature):
            return NotImplemented
        return self._name == other._name and np.array_equal(self.values, other.values)

    def __hash__(self) -> int:
        # Values are frozen, so a content hash is stable for the object's
        # life. Adding 0.0 first normalises -0.0 to +0.0: __eq__ goes through
        # array_equal, which calls the two zeros equal, so their bytes must
        # hash equal too or the hash contract breaks in a set.
        return hash((self._name, (self.values + 0.0).tobytes()))

    def __repr__(self) -> str:
        return f"Feature(name={self._name!r}, n_samples={self.n_samples})"
