"""The K separate fits behind a one-vs-rest classifier.

The model looks like one estimator and is K of them, and that is the whole
thing worth seeing about it. Each was fitted on the same features against a
different recoded target -- "is this row class 0", then class 1, and so on --
and each knows nothing about the others.

Which is exactly why its probabilities do not sum to one. Nothing in the
arithmetic makes them; the K models were never introduced. Measured on the
three-class fixture the row totals run from 0.41 to 1.78, and holding the
sub-fits makes that a fact a caller can check rather than a claim in a
docstring.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from oop_ml.classification.linear_classifier import LinearClassifier
from oop_ml.core.data.feature import Feature


class ClassFit:
    """One class's binary model, and the target it was fitted against.

    Parameters
    ----------
    class_index:
        Which class this model answers for.
    recoded_target:
        The target as this model saw it: 1 where the row was this class, 0
        everywhere else. Named for the question it encodes, so a caller
        reading several of them can tell them apart.
    model:
        The fitted binary classifier. A deep copy per class, because fitting
        one shared prototype K times would leave every entry holding whichever
        class went last.
    """

    __slots__ = ("_class_index", "_model", "_recoded_target")

    def __init__(
        self, class_index: int, recoded_target: Feature, model: LinearClassifier
    ) -> None:
        self._class_index = class_index
        self._recoded_target = recoded_target
        self._model = model

    @property
    def class_index(self) -> int:
        """Which class this model answers for."""
        return self._class_index

    @property
    def recoded_target(self) -> Feature:
        """The 0/1 target this model was fitted against."""
        return self._recoded_target

    @property
    def model(self) -> LinearClassifier:
        """The fitted binary classifier.

        The same type ``OneVsRestClassifier.binary_model`` accepts, so a
        caller can go a level deeper and ask it for its own observation --
        a ``solver_path`` if it walks.
        """
        return self._model

    @property
    def positive_rows(self) -> int:
        """How many rows were the positive class for this fit.

        Worth having beside the model: one-vs-rest turns a balanced
        three-class problem into three unbalanced binary ones, and this is the
        number that shows it.
        """
        return int(self._recoded_target.values.sum())

    def __repr__(self) -> str:
        return (
            f"ClassFit(class {self._class_index}, {self.positive_rows} positive rows)"
        )


class OneVsRestFits:
    """Every sub-fit a one-vs-rest classifier is made of, in class order."""

    __slots__ = ("_fits",)

    def __init__(self, fits: Sequence[ClassFit]) -> None:
        self._fits = tuple(fits)

    @property
    def result(self) -> tuple[LinearClassifier, ...]:
        """What the efficient route stores: the fitted models, in class order."""
        return tuple(fit.model for fit in self._fits)

    @property
    def fits(self) -> tuple[ClassFit, ...]:
        """The sub-fits, in class order."""
        return self._fits

    def __iter__(self) -> Iterator[ClassFit]:
        return iter(self._fits)

    def __len__(self) -> int:
        return len(self._fits)

    def __repr__(self) -> str:
        return f"OneVsRestFits({len(self._fits)} classes)"
