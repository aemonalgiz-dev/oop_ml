"""The shared frame for classifiers whose boundary is linear in the coefficients.

Everything here is task-level plumbing that any linear classifier wants and none
of them should write twice. The design matrix, the intercept split and the
coefficient pairing come from :class:`~oop_ml.core.linear_model.LinearModel`;
what this adds is the step from a linear predictor to a probability, and from a
probability to a label.

A subclass supplies its own ``fit``, its own ``_solve``, and its own
``_sigmoid``. It should not have to think about thresholds or about matching
features by name.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import numpy as np

from oop_ml.core.base import Classifier
from oop_ml.core.feature import Feature
from oop_ml.core.linear_model import LinearModel
from oop_ml.core.types import FloatArray


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
        :class:`~oop_ml.core.classification_evaluation.ClassificationEvaluation`,
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
        raise NotImplementedError

    def decision_boundary_at(self, feature_name: str) -> float:
        """Where this one feature crosses the boundary, holding the rest at zero.

        The boundary is the surface on which the log-odds are zero, so along a
        single axis it sits at ``-intercept / coefficient``. Only meaningful for
        a model with one predictor, or when every other feature really is zero,
        which on standardised columns means every other feature is at its mean.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        UndefinedMetricError
            If this feature's coefficient is zero, leaving no crossing.
        """
        raise NotImplementedError

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
        return float(np.exp(self.coefficients_[feature_name]))
