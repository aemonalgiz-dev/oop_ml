"""Multi-class by fitting one binary classifier per class.

The other route to ``K`` classes, and the one that needs no new mathematics at
all. Fit ``K`` independent binary models, the ``k``-th asking "is this row class
``k``, or is it anything else?", then predict whichever class answered loudest.

Its appeal is that it reuses a binary classifier wholesale. Its cost is that
the ``K`` answers were never asked to agree. Measured on 300 rows over three
classes, the one-vs-rest probabilities summed to anywhere between 0.85 and 1.56
per row, against exactly 1.0000 for softmax, and the two disagreed on the
predicted label for 4.0% of rows.

That is not a bug to be fixed by normalising. Dividing three unrelated opinions
by their total produces something that sums to one without being a probability
of anything, and it would hide the fact that the model is uncertain in a way the
softmax model cannot be. This class reports the raw per-class probabilities and
says plainly that they are not a distribution.

The other cost is subtler. Each binary fit sees a deliberately unbalanced
problem: on classes split 77 / 41 / 182, the three fits see 26%, 14% and 61%
positive respectively. The rarest class is fitted against the hardest imbalance,
which is precisely the class you were most likely to care about.

Where it earns its place is breadth. Any binary classifier at all can be
wrapped, including ones with no multi-class formulation of their own, and each
fit is independent so they parallelise perfectly. Prefer
:class:`~oop_ml.classification.multinomial_logistic_regression.MultinomialLogisticRegression`
when the model has a genuine multi-class form, and reach for this when it does
not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import ConfigDict, PrivateAttr

from oop_ml.classification.linear_classifier import LinearClassifier
from oop_ml.core.base.estimator import MultiClassClassifier
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.types import FloatArray


class OneVsRestClassifier(MultiClassClassifier[Sequence[Feature], Feature]):
    """Wrap a binary classifier so that it answers a multi-class question.

    Parameters
    ----------
    binary_model:
        The classifier to clone once per class. It is a *prototype*: it is
        never fitted itself, and every class gets its own deep copy so that
        their learned states cannot collide. Passing an already-fitted model is
        harmless for the same reason.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    binary_model: LinearClassifier

    _n_classes: int | None = PrivateAttr(default=None)
    _fitted_models: tuple[LinearClassifier, ...] | None = PrivateAttr(default=None)

    @property
    def n_classes(self) -> int:
        """How many classes the fit saw.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._n_classes is not None
        return self._n_classes

    def model_for(self, class_index: int) -> LinearClassifier:
        """The binary model fitted for one class against all the others.

        Exposed because the per-class models are the interesting part: each one
        carries its own coefficients, and comparing them across classes is how
        you read what the wrapper actually learned.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        InvalidValuesError
            If ``class_index`` is not a class of this problem.
        """
        self._check_fitted()
        assert self._fitted_models is not None

        if not 0 <= class_index < self.n_classes:
            raise InvalidValuesError(
                f"class {class_index} is outside a problem with "
                f"{self.n_classes} classes"
            )

        return self._fitted_models[class_index]

    @staticmethod
    def _binary_target(target_values: Feature, class_index: int) -> Feature:
        """``target_values`` recoded as "is it this class" 1/0.

        The name is derived rather than carried over. A column recoded for
        class 2 holds ``outcome == 2``, not ``outcome``, and three columns all
        called ``outcome`` would be three different questions wearing one name
        -- which is the thing this library exists to stop happening to a
        feature. Pull a per-class model out with :meth:`model_for` and its
        target says which question it answered.
        """
        return Feature(
            f"{target_values.name}=={class_index}",
            (target_values.values == float(class_index)).astype(np.float64),
        )

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Fit one binary model per class, each against all the others.

        Parameters
        ----------
        input_values:
            One or more predictor columns, all the same length as the target.
        target_values:
            The classes, as whole positions running ``0 .. K - 1``.

        Returns
        -------
        Self
            This model, so calls can chain.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        AllSameValuesError
            If any predictor is constant.
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If the target holds fewer than two classes, or leaves a gap in the
            run from zero.
        """
        feature_set = self._validated_feature_set(input_values)
        feature_set.check_aligned_with(target_values)

        target_column = target_values.column
        target_column.check_is_label_encoded()

        # One model per CLASS, not per feature. Every one of them is fitted on
        # the whole feature set; the only thing that changes between them is
        # the target, recoded to "is this row class k".
        self._n_classes = target_column.n_classes
        self._fitted_models = tuple(
            # A deep copy per class. Fitting self.binary_model directly would
            # give all K classes one shared object, so the last fit would
            # overwrite every earlier one and the prototype would come back
            # carrying whichever class happened to go last.
            self.binary_model.model_copy(deep=True).fit(
                input_values, self._binary_target(target_values, class_index)
            )
            for class_index in range(self._n_classes)
        )

        self._mark_fitted()
        return self

    def predict_probabilities(self, input_values: Sequence[Feature]) -> FloatArray:
        """Each class's own probability, as ``(n_samples, n_classes)``.

        **These rows do not sum to one**, and deliberately so. Column ``k`` is
        the ``k``-th binary model's P(this row is class k), and the ``K`` models
        were fitted independently with no constraint tying them together. A row
        summing to 1.5 means two models are both confident, which is a fact
        about the fit worth seeing rather than one to normalise away.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        self._check_fitted()
        assert self._fitted_models is not None

        # Each model answers for its own class and nothing normalises across
        # them, which is the whole difference from softmax.
        return np.column_stack(
            [model.predict_probability(input_values) for model in self._fitted_models]
        )

    def predict(self, input_values: Sequence[Feature]) -> FloatArray:
        """The class whose own model was most confident, as ``0.0 .. K-1``.

        Comparing probabilities across models that were never jointly
        calibrated is the compromise at the heart of this approach. It works
        well in practice and is not the same as choosing the most probable
        class, which is what the softmax model does.

        Ties go to the lower class index.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return np.argmax(self.predict_probabilities(input_values), axis=1).astype(
            np.float64
        )

    def _validated_feature_set(self, input_values: Sequence[Feature]) -> FeatureSet:
        """The features as a validated set, for ``fit`` to check once."""
        feature_set = FeatureSet(input_values)
        feature_set.check_columns_vary()

        return feature_set
