"""Preprocessing and a model as one object, so the fold cannot be got wrong.

What this is for
----------------
A ``Standardizer`` learns a mean and a spread. Fit it on the whole dataset and
then cross-validate, and every training fold has been told something about the
rows it is about to be scored on. That is leakage, and until now nothing in this
library made it structurally impossible -- the caller had to remember to fit the
scaler on ``split.training`` inside each fold.

Measured, that particular leak is smaller than its reputation. Standardizing a
three-column dataset before folding rather than inside each fold, with k-nearest
neighbours at five folds, averaged over thirty seeds::

    rows   fitted outside   fitted inside    flattered by
      30       +0.5679         +0.5610          +0.0068
      60       +0.7586         +0.7530          +0.0055
     200       +0.9009         +0.9007          +0.0002

Small, and vanishing with sample size, because a mean estimated from 20% more
rows barely differs. It is worth being accurate about that rather than repeating
a scare: for a transformer that never looks at the target, the bias is real and
usually tiny.

**It is not tiny for a transformer that does look at the target.** Anything that
selects columns by correlation with the outcome, or bins by it, or imputes from
it, has seen the answers -- and fitting that outside the fold can turn pure
noise into an apparently strong model. This library has no such transformer
today. A caller can write one, and then this class is the difference between a
result and a mirage.

The stronger reason, which is a capability rather than a bias
--------------------------------------------------------------
**Without this, a preprocessing choice cannot be searched at all.**
``GridSearch`` varies fields of the model it is handed, and ``degree`` is a
field of ``PolynomialFeatures``, not of ``RidgeRegression``. So "should I use
degree 2 or degree 3" was a question the search could not be asked. Wrapping
both in one object makes the whole configuration one model, and the question
becomes an ordinary candidate.

The same closure makes serving safe: a fitted pipeline holds its fitted
transformers, so ``predict`` re-applies exactly the transformation the fit
learned, with no second call site to keep in step.

Two classes, because the tasks answer differently
--------------------------------------------------
:class:`RegressionPipeline` is a ``Regressor`` and
:class:`ClassificationPipeline` is a ``MultiClassClassifier``, which is what
lets each drop into the matching ``CrossValidation`` and ``GridSearch`` entry
point with nothing else changed. One class covering both would have to decide
at runtime which evaluation to return, and the seam the rest of this library
uses is the one drawn at the type.

How a search varies a step
---------------------------
By replacing whole objects, never by naming a nested field::

    SearchSpace.over(
        RegressionPipeline,
        steps=[
            PipelineSteps.of(terms=PolynomialFeatures(degree=2)),
            PipelineSteps.of(terms=PolynomialFeatures(degree=3)),
        ],
        model=[RidgeRegression(penalty=0.1), RidgeRegression(penalty=1.0)],
    )

scikit-learn spells this ``terms__degree``, a step name and a field name joined
by a double underscore and parsed at runtime. Every value in the version above
was validated by its own constructor when it was written, and a misspelling is
a ``NameError`` in the caller's own source rather than a setting that is
silently ignored.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import ClassVar, Self

from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.estimator import (
    Fittable,
    MultiClassClassifier,
    Regressor,
)
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import ClassScores
from oop_ml.pipeline.steps import PipelineStep, PipelineSteps


class FittedChain:
    """What fitting the step chain produced: the fitted steps and their output.

    A pairing object for the reason every pairing here exists -- a method
    reaching for ``return steps, transformed`` is this class unwritten -- and
    for one more: ``fit`` must commit nothing to the pipeline until every part
    has succeeded, so the chain's results have to travel together as a value
    before either lands on ``self``.
    """

    __slots__ = ("_steps", "_transformed")

    def __init__(self, steps: PipelineSteps, transformed: list[Feature]) -> None:
        self._steps = steps
        self._transformed = transformed

    @property
    def steps(self) -> PipelineSteps:
        """The transformers, fitted, in running order."""
        return self._steps

    @property
    def transformed(self) -> list[Feature]:
        """What the last step produced, which is what the model fits on."""
        return self._transformed


class Pipeline(Fittable):
    """The shared half: run the steps in order, then hand on to the model.

    Not usable on its own -- it declares no ``fit`` and no ``predict``, because
    no single signature covers a regressor and a classifier. It holds the part
    that is identical either way, which is the chaining and the refitting rule.

    Parameters
    ----------
    steps:
        The transformers to run, in order. May be empty.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    steps: PipelineSteps = Field(default_factory=PipelineSteps)

    LEARNED_STATE: ClassVar[tuple[str, ...]] = (
        "_fitted_steps",
        "_fitted_model",
    )

    _fitted_steps: PipelineSteps | None = PrivateAttr(default=None)

    @property
    def fitted_steps(self) -> PipelineSteps:
        """The transformers as this fit left them, fitted to the training rows.

        Not the ones passed in at construction. Those are the *configuration*
        and are never fitted, so the same pipeline object can be handed to
        every fold of a cross-validation and come back describing the last one
        without the others having contaminated it.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._fitted_steps is not None
        return self._fitted_steps

    def _fitted_chain(self, input_values: Sequence[Feature]) -> FittedChain:
        """Fit each transformer on what the one before it produced.

        The whole point of the class, in five lines. Each transformer is fitted
        to the *output of the previous step*, not to the raw input, because that
        is what it will see at predict time -- a scaler placed after a
        polynomial expansion has to learn the expanded columns' means, not the
        original ones.

        Every transformer is deep-copied first. The ones held in ``steps`` are
        configuration and must stay unfitted: a search hands the same
        ``PipelineSteps`` to candidate after candidate, and a fit that mutated
        them in place would leave each candidate starting from the previous
        one's learned state.

        Returns the result rather than committing it: ``fit`` assigns nothing
        to the pipeline until the model has fitted too, so a refit that fails
        anywhere leaves the previous fit intact instead of half-replaced.
        """
        fitted = []
        carried = list(input_values)

        for step in self.steps:
            transformer = deepcopy(step.transformer)
            carried = list(transformer.fit_transform(carried))
            fitted.append(PipelineStep(step.name, transformer))

        return FittedChain(PipelineSteps(fitted), carried)

    def _apply_steps(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Run the already-fitted transformers, learning nothing.

        The prediction-time counterpart, and the reason a pipeline cannot leak
        at serving time: it transforms with what ``fit`` learned, and there is
        no second call site that could apply something else.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        carried = list(input_values)

        for step in self.fitted_steps:
            carried = list(step.transformer.transform(carried))

        return carried

    @property
    def n_steps(self) -> int:
        """How many transformers run before the model."""
        return len(self.steps)


class RegressionPipeline(Pipeline, Regressor[Sequence[Feature], Feature]):
    """Preprocessing and a regressor as one regressor.

    Being a ``Regressor`` is the point: it goes straight into
    ``CrossValidation.evaluate`` and ``GridSearch.search``, and each fold refits
    the transformers on that fold's training rows because that is simply what
    ``fit`` does.

    Parameters
    ----------
    steps:
        The transformers to run, in order.
    model:
        The regressor to fit on what the steps produce.
    """

    model: Regressor[Sequence[Feature], Feature]

    _fitted_model: Regressor[Sequence[Feature], Feature] | None = PrivateAttr(
        default=None
    )

    @property
    def fitted_model(self) -> Regressor[Sequence[Feature], Feature]:
        """The regressor as this fit left it.

        A copy, like the steps, so the configured ``model`` stays unfitted and
        reusable across folds and candidates.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._fitted_model is not None
        return self._fitted_model

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Fit every step on the training rows, then the model on the result.

        Everything here is fitted from scratch, every time. That is what makes
        a pipeline safe inside a fold: there is no state to carry over, so
        handing the same pipeline to five folds gives five independent fits.
        """
        # Compute everything, then commit everything: a refit that fails in
        # a step or in the model leaves the previous fit intact rather than
        # holding a new unfitted model beside old fitted steps.
        chain = self._fitted_chain(input_values)
        fitted_model = deepcopy(self.model)
        fitted_model.fit(chain.transformed, target_values)

        self._fitted_steps = chain.steps
        self._fitted_model = fitted_model
        self._mark_fitted()

        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """Transform with what the fit learned, then predict.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        self._check_fitted()

        return self.fitted_model.predict(self._apply_steps(input_values))


class ClassificationPipeline(
    Pipeline, MultiClassClassifier[Sequence[Feature], Feature]
):
    """Preprocessing and a classifier as one classifier.

    The sibling of :class:`RegressionPipeline`, and it exists separately for the
    reason the cross-validation entry points do: which evaluation comes back
    should be decided by which model went in.

    Parameters
    ----------
    steps:
        The transformers to run, in order.
    model:
        The classifier to fit on what the steps produce.
    """

    model: MultiClassClassifier[Sequence[Feature], Feature]

    _fitted_model: MultiClassClassifier[Sequence[Feature], Feature] | None = (
        PrivateAttr(default=None)
    )

    @property
    def fitted_model(self) -> MultiClassClassifier[Sequence[Feature], Feature]:
        """The classifier as this fit left it.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._fitted_model is not None
        return self._fitted_model

    @property
    def n_classes(self) -> int:
        """How many classes the wrapped classifier learned.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.fitted_model.n_classes

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Fit every step on the training rows, then the classifier on the result."""
        # Compute everything, then commit everything: a refit that fails in
        # a step or in the model leaves the previous fit intact rather than
        # holding a new unfitted model beside old fitted steps.
        chain = self._fitted_chain(input_values)
        fitted_model = deepcopy(self.model)
        fitted_model.fit(chain.transformed, target_values)

        self._fitted_steps = chain.steps
        self._fitted_model = fitted_model
        self._mark_fitted()

        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """Transform with what the fit learned, then predict a class per row.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        self._check_fitted()

        return self.fitted_model.predict(self._apply_steps(input_values))

    def predict_probabilities(self, input_values: Sequence[Feature]) -> ClassScores:
        """Transform with what the fit learned, then score every class.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        self._check_fitted()

        return self.fitted_model.predict_probabilities(self._apply_steps(input_values))
