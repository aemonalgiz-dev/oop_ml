"""The abstract estimator hierarchy that every model in the library sits under.

``Estimator`` captures the one thing that every learner has in common, which is
that it is fit to data and that using it before fitting is an error.
``Regressor`` adds the predict and score contract on top for models producing
continuous values, and a concrete model such as ``SimpleLinearRegression``
inherits from that and supplies only its own ``fit`` and ``predict``.

Three decisions here are worth understanding before you write a model against
this hierarchy.

Hyperparameters are Pydantic fields and are therefore validated the moment you
construct the model. Learned parameters are private attributes set during
``fit`` and exposed through read-only properties, so they never appear as
constructor arguments; if we allowed that, a caller could hand a model
coefficients it never learned.

Fitted state is tracked in exactly one place, ``_fitted``, and enforced in
exactly one place, ``_check_fitted``. No model has to reinvent the "not fitted
yet" guard, which also means no model can quietly forget it.

The hierarchy is generic in both directions, in what a model consumes and in
what it is fit against. An array-shaped model and a feature-first one can then
share one contract without either of them having to widen its types to ``Any``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, PrivateAttr

from oop_ml.core.classification_evaluation import ClassificationEvaluation
from oop_ml.core.column import ColumnSource
from oop_ml.core.evaluation import RegressionEvaluation
from oop_ml.core.multiclass_evaluation import MultiClassEvaluation
from oop_ml.exceptions import NotFittedError
from oop_ml.types import FloatArray

InputT = TypeVar("InputT")
"""What a model is fit and predicted *on*: raw values, or named features."""

TargetT = TypeVar("TargetT", bound=ColumnSource)
"""What a model is fit against.

This one is bounded rather than free, because every target eventually has to
become a :class:`~oop_ml.core.column.Column` in order to be scored, and raw
values, a ``Column``, and a ``Feature`` can all manage that.
"""


class Fittable(BaseModel):
    """Anything that learns from data and must not be used before it has.

    This holds the fitted-state machinery in one place, so that estimators,
    which learn against a target, and transformers, which do not, end up
    sharing a single implementation of the guard rather than each writing their
    own slightly different version of it.
    """

    # Learned parameters are numpy scalars/arrays, so allow arbitrary types.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _fitted: bool = PrivateAttr(default=False)

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit`` has completed successfully."""
        return self._fitted

    def _mark_fitted(self) -> None:
        """Record that fitting has completed. Call at the end of ``fit``."""
        self._fitted = True

    def _check_fitted(self) -> None:
        """Guard used by fitted-only attributes and methods.

        Raises
        ------
        NotFittedError
            If called before ``fit`` has completed.
        """
        if not self._fitted:
            raise NotFittedError(
                f"{type(self).__name__} must be fit before this is available"
            )


class Estimator(Fittable, Generic[InputT, TargetT]):
    """Base class for anything that learns from data via ``fit``."""

    @abstractmethod
    def fit(self, input_values: InputT, target_values: TargetT) -> Self:
        """Learn parameters from data and return ``self`` (so calls can chain).

        Implementations should validate input, compute learned parameters,
        store them on private attributes, then call ``self._mark_fitted()``.
        """


class Regressor(Estimator[InputT, TargetT]):
    """Base class for estimators that predict continuous values.

    Note that this inherits the parameterized base rather than re-listing
    ``Generic`` alongside it. Doing both puts ``Generic`` ahead of ``Estimator``
    in the bases, at which point no consistent MRO exists and pydantic rejects
    the class outright at creation time, which means nothing in the package
    imports at all.
    """

    @abstractmethod
    def predict(self, input_values: InputT) -> FloatArray:
        """Return predictions. Must call ``_check_fitted()`` first."""

    def evaluate(
        self, input_values: InputT, actual_values: TargetT
    ) -> RegressionEvaluation:
        """Predict on ``input_values`` and pair the result with the truth.

        This is the single scoring entry point that every regressor inherits.
        Predictions are computed once and the inputs are validated once, and
        you then read as many metrics off the returned evaluation as you like.
        """
        return RegressionEvaluation(actual_values, self.predict(input_values))

    def score(self, input_values: InputT, target_values: TargetT) -> float:
        """Coefficient of determination (R^2) of the prediction on the given data.

        A thin convenience over :meth:`evaluate`, for the one metric that
        callers ask for more than any other.
        """
        return self.evaluate(input_values, target_values).r2_score


class Classifier(Estimator[InputT, TargetT]):
    """Base class for estimators that predict a class rather than a quantity.

    The split from :class:`Regressor` is not cosmetic. A classifier answers with
    a label, although what it actually computes is a probability, and both are
    worth exposing because they answer different questions. ``predict`` tells
    you which side of the boundary a row falls on, while
    ``predict_probability`` tells you how close to that boundary it was, and a
    row at 0.51 is a very different thing from a row at 0.99 even though the two
    get the same label.

    Scoring differs as well. R^2 is meaningless on labels, so ``evaluate``
    returns a :class:`~oop_ml.core.classification_evaluation.ClassificationEvaluation`
    built on a confusion matrix instead.
    """

    @abstractmethod
    def predict(self, input_values: InputT) -> FloatArray:
        """Return one predicted label per observation, each 0.0 or 1.0.

        Must call ``_check_fitted()`` first.
        """

    @abstractmethod
    def predict_probability(self, input_values: InputT) -> FloatArray:
        """Return P(class is 1) per observation, each in ``[0, 1]``.

        Must call ``_check_fitted()`` first.
        """

    def evaluate(
        self, input_values: InputT, actual_values: TargetT
    ) -> ClassificationEvaluation:
        """Predict labels on ``input_values`` and pair them with the truth.

        The single scoring entry point every classifier inherits, mirroring
        ``Regressor.evaluate``. Predictions are computed once, and every metric
        is then read off the returned evaluation.
        """
        return ClassificationEvaluation(actual_values, self.predict(input_values))

    def score(self, input_values: InputT, target_values: TargetT) -> float:
        """Accuracy of the prediction on the given data.

        A thin convenience over :meth:`evaluate`, and worth treating with more
        caution than ``Regressor.score``. Accuracy on an unbalanced target can
        look excellent while the model finds nothing, so reach for the
        evaluation object and read precision and recall before believing it.
        """
        return self.evaluate(input_values, target_values).accuracy


class MultiClassClassifier(Estimator[InputT, TargetT]):
    """Base class for estimators that choose between more than two classes.

    A sibling of :class:`Classifier` rather than a parent of it. The binary case
    hands back one probability per row and this hands back one per class per
    row, so ``predict_probability`` and ``predict_probabilities`` are different
    methods with different shapes, and no caller wanting one would accept the
    other. Collapsing them would mean every binary caller indexing a column out
    of a matrix to get the number they already had.

    ``predict`` still returns one value per row: the class with the highest
    probability, as a float on the same 0..K-1 scale the target uses.
    """

    @abstractmethod
    def predict(self, input_values: InputT) -> FloatArray:
        """Return one predicted class per observation, as ``0.0 .. K-1``.

        Must call ``_check_fitted()`` first.
        """

    @abstractmethod
    def predict_probabilities(self, input_values: InputT) -> FloatArray:
        """Return an ``(n_samples, n_classes)`` matrix of probabilities.

        Column ``k`` is P(class is k). Whether the rows sum to 1 is a property
        of the concrete model rather than of this contract: a softmax model
        guarantees it by construction, and a one-vs-rest wrapper cannot.

        Must call ``_check_fitted()`` first.
        """

    def evaluate(
        self, input_values: InputT, actual_values: TargetT
    ) -> MultiClassEvaluation:
        """Predict classes on ``input_values`` and pair them with the truth.

        The single scoring entry point, mirroring ``Classifier.evaluate``. The
        class count is taken from the fitted model rather than from the truth,
        so a held-out fold missing one class still produces a table the right
        size instead of a smaller one that quietly renumbers the rest.
        """
        return MultiClassEvaluation(
            actual_values, self.predict(input_values), self.n_classes
        )

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """How many classes the fit saw.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """

    def score(self, input_values: InputT, target_values: TargetT) -> float:
        """Accuracy of the prediction on the given data.

        Worth even more caution than in the binary case: with many classes a
        model that only ever names the commonest one can score respectably
        while getting every other class wrong every time. Read the per-class
        recalls on the evaluation before believing this number.
        """
        return self.evaluate(input_values, target_values).accuracy


class Transformer(Fittable, Generic[InputT]):
    """Base class for things that learn a reshaping of the inputs themselves.

    A transformer has no target at all. It learns from the inputs alone, a
    column's mean and spread for instance, and then rewrites inputs in terms of
    what it learned.

    The split between ``fit`` and ``transform`` is the entire point here, and it
    is not a stylistic preference. Statistics have to be learned from the
    training data only and then applied unchanged to everything else. Re-learn
    them on a test set and you have leaked information the model would never
    have had, which quietly flatters every score you compute afterwards.
    """

    @abstractmethod
    def fit(self, input_values: InputT) -> Self:
        """Learn the transformation from ``input_values`` and return ``self``."""

    @abstractmethod
    def transform(self, input_values: InputT) -> InputT:
        """Apply the learned transformation. Must call ``_check_fitted()`` first."""

    def fit_transform(self, input_values: InputT) -> InputT:
        """Fit on ``input_values`` and immediately transform them.

        This is for the training set only. Never reach for it on held-out data,
        since doing so re-fits the statistics and defeats the entire split
        described above.
        """
        return self.fit(input_values).transform(input_values)
