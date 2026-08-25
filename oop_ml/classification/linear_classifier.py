"""The shared frame for classifiers whose boundary is linear in the coefficients.

Everything here is task-level plumbing that any linear classifier wants and none
of them should write twice. The design matrix, the intercept split and the
coefficient pairing come from :class:`~oop_ml.core.linear_model.LinearModel`;
what this adds is the step from a linear predictor to a probability, and from a
probability to a label.

A subclass supplies its own ``_solve`` and its own ``_sigmoid``, and nothing
else. It should not have to think about thresholds, about matching features by
name, or about the order of the guards in ``fit`` -- the last of which used to
be copied into every concrete classifier until the copies started to drift.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Self

import numpy as np

from oop_ml.core.base import Classifier
from oop_ml.core.linear_model import LinearModel
from oop_ml.data.column import Column
from oop_ml.data.feature import Feature
from oop_ml.exceptions import UndefinedMetricError
from oop_ml.types import FloatArray


class LinearClassifier(LinearModel, Classifier[Sequence[Feature], Feature]):
    """A linear decision boundary over named features.

    Parameters
    ----------
    threshold:
        Supplied by the concrete model, since where the cut falls is a modelling
        choice rather than a property of the frame.
    """

    threshold: float

    @staticmethod
    @abstractmethod
    def _sigmoid(linear_predictor: FloatArray) -> FloatArray:
        """Map the linear predictor onto a probability in ``[0, 1]``."""

    def _validated_target_column(self, target_values: Feature) -> Column:
        """The target, insisted upon as 0/1 labels carrying both classes.

        Where classification tightens the regression contract. A boundary
        fitted against labels that are not 0 or 1 would be answering a question
        nobody asked, and one fitted against a single class has nothing to
        discriminate between.

        Raises
        ------
        NonBinaryLabelsError
            If the target holds anything other than 0 and 1.
        SingleClassError
            If the target holds only one of the two classes.
        """
        target_column = super()._validated_target_column(target_values)
        target_column.check_is_binary()
        target_column.check_has_both_classes()

        return target_column

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Fit the boundary, delegating the solve itself to the subclass.

        Parameters
        ----------
        input_values:
            One or more predictor columns, all the same length as the target.
            Their names key the learned coefficients and must be unique.
        target_values:
            The labels being classified, every one of them 0 or 1.

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
        TooFewValuesError
            If there are fewer observations than parameters to estimate.
        NonBinaryLabelsError
            If the target holds anything other than 0 and 1.
        SingleClassError
            If the target holds only one of the two classes.

        Notes
        -----
        A subclass whose solver can fail in its own way documents that on its
        ``_solve`` rather than here, since the failure belongs to the method
        that can raise it.
        """
        return self._fit_linear_model(input_values, target_values)

    def predict_probability(self, input_values: Sequence[Feature]) -> FloatArray:
        """P(class is 1) for each observation.

        The linear predictor is the log-odds, so this is simply that value put
        back through the sigmoid. Features are matched to coefficients by name,
        with the same contract ``predict`` follows everywhere in the library.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return self._sigmoid(self._linear_predictor(input_values))

    def predict(self, input_values: Sequence[Feature]) -> FloatArray:
        """The predicted label for each observation, as ``0.0`` or ``1.0``.

        Apply ``threshold`` to the probabilities from
        :meth:`predict_probability`. A row at or above the threshold is called
        positive.

        Returning floats rather than booleans is deliberate: the labels go
        straight into a
        :class:`~oop_ml.evaluation.classification.ClassificationEvaluation`,
        which validates them as a 0/1 ``Column`` like any other.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return (self.predict_probability(input_values) >= self.threshold).astype(
            np.float64
        )

    def decision_boundary_at(self, feature_name: str) -> float:
        """Where this one feature crosses the boundary, holding the rest at zero.

        A row is called positive once its probability reaches ``threshold``, and
        pushing that through the logit turns it into a statement about the
        linear predictor::

            X b  >=  log(threshold / (1 - threshold))

        So the threshold is a cut on the log-odds scale, and the crossing point
        along one axis is where the linear predictor meets that cut::

            x_j  =  (log(t / (1 - t)) - intercept) / b_j

        At the default threshold of 0.5 the cut is ``log(1) = 0`` and this
        collapses to the familiar ``-intercept / b_j``. Written in the general
        form so that a model with a shifted threshold reports the boundary it
        actually uses rather than the one it would have used at 0.5.

        Every other feature is held at zero. With one predictor that is the whole
        story; with several, the boundary is a surface rather than a point and
        this is where it crosses this particular axis, which is only the number
        you want if zero is a meaningful place for the others to sit. On
        standardised columns it is, since zero is then each feature's mean.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``feature_name`` was not among the fitted features.
        UndefinedMetricError
            If this feature's coefficient is zero, leaving no crossing.
        """
        coefficient = self.coefficients[feature_name]

        if coefficient == 0.0:
            raise UndefinedMetricError(
                f"{feature_name} has a coefficient of zero, so the boundary "
                f"never crosses it"
            )

        log_odds_cut = float(np.log(self.threshold / (1.0 - self.threshold)))

        return float((log_odds_cut - self.intercept) / coefficient)

    def odds_multiplier_for(self, feature_name: str) -> float:
        """What one more unit of this feature does to the odds: ``exp(b_j)``.

        This is how a logistic coefficient is read. Not a change in probability,
        which is not constant along the curve, but a constant multiplier on the
        odds. A coefficient of 0.8637 means the odds multiply by 2.37 per unit.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        return float(np.exp(self.coefficients[feature_name]))
