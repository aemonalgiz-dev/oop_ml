"""Boost by making the next learner care about what the last one got wrong.

The third ensemble shape here, and it is a third shape rather than a variation
on the other two. A bagged model fits every member on a *resample* and averages
them equally. A gradient-boosted model fits each member on the *residual* the
ensemble has left. This fits every member on the same rows and changes what
those rows are *worth*, then combines the members by a weighted vote in which
the ones that did well get a louder voice.

The algorithm, which is SAMME
------------------------------
Start with every row counting the same. Then, for each round:

1. Fit a weak learner on the rows as they are currently weighted.
2. Measure its weighted error -- the share of the weight it gets wrong::

       err = sum(w_i over the rows it got wrong) / sum(w_i)

3. Give it a voice proportional to how much better than guessing that is::

       alpha = learning_rate * (log((1 - err) / err) + log(K - 1))

4. Multiply the weight of every row it got wrong by ``exp(alpha)``, and
   normalise so the weights still sum to one.

The ``log(K - 1)`` is what makes this work for more than two classes, and it is
also what sets the bar: ``alpha`` is positive exactly when ``err < 1 - 1/K``,
which is to say exactly when the learner beats guessing uniformly. A learner
that does worse than that gets a *negative* voice under the formula, which
would be the ensemble learning from a liar, so the walk stops instead.

Why the weights had to be real
-------------------------------
Step 1 is the whole of it, and it is why
:class:`~oop_ml.core.tree.weights.WeightedTargets` exists. A weak learner that
ignored the weights would fit the same stump every round and every round would
be the first one. Resampling in proportion to the weights is the other
implementation of the same idea and it is what the original paper allows; this
library weights the impurity instead, so a round is deterministic and a fit is
reproducible without a seed.

What a weak learner is, and why it should stay weak
-----------------------------------------------------
A decision stump: one question, two leaves, ``max_depth=1``. It has to be weak,
which is the counter-intuitive part. Boost a learner that already fits the
training rows perfectly and the first round has ``err = 0``, nothing is
reweighted, and every later round repeats it -- the ensemble is one tree in
fifty copies. The method's power comes from combining many models that are each
barely better than a coin, and a stump is the canonical one.

Where it is fragile
-------------------
Outliers, and for exactly the reason it works. A row that is genuinely
mislabelled cannot be fitted, so every round gets it wrong, so its weight grows
exponentially until it is most of the dataset and the ensemble is fitting one
bad row. Bagging is not like this; it is the price of paying attention.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.estimator import MultiClassClassifier
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import ClassScores, ProbabilityMatrix
from oop_ml.core.exceptions import DivergenceError
from oop_ml.core.logistic import stable_softmax
from oop_ml.core.types import FloatArray
from oop_ml.numpy.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)

SILENT_MEMBER_VOICE = 1e-10
"""Below this a member is treated as having no voice at all.

Not a fudge but the only workable reading of the bar. A learner is worth
keeping exactly when its voice is positive, and at the bar the voice is zero --
but a stump on a constant column facing three balanced classes is wrong on
``20/30``, whose voice works out to 2.2e-16 rather than to 0, because
``1 - 2/3`` divided by ``2/3`` is 0.5000000000000001 and not 0.5. Testing
against exact zero therefore admits a member worth nothing at all and lets the
ensemble fill up with them.
"""

PERFECT_MEMBER_VOICE = 1.0
"""The voice given to a learner that got everything right.

``log((1 - err) / err)`` is infinite at ``err = 0``, and an infinite voice
would drown out every other member and make every later round meaningless. A
learner that fits the rows perfectly has also ended the walk, since nothing
would be reweighted, so what its exact voice is only matters when it is the
only member -- and then any positive number gives the same answer. One is the
value scikit-learn uses.
"""


class AdaBoostClassifier(MultiClassClassifier[Sequence[Feature], Feature]):
    """Weak learners in sequence, each fitted on what the last one found hard.

    Parameters
    ----------
    n_members:
        How many rounds to run at most. Fewer are kept when the walk stops
        early, which ``n_members_fitted`` reports.
    learning_rate:
        A damper on every member's voice. Smaller values need more rounds and
        generalise better, which is the same trade gradient boosting makes.
    max_depth:
        How deep each weak learner may go. One by default, which is a stump,
        and the module docstring says why weak is the point.
    random_seed:
        Passed to every member. The walk itself is deterministic -- the
        weights, not a resample, are what change between rounds -- so this
        only matters if a member is configured to draw anything.

    Raises
    ------
    NotFittedError
        From any learned property, or from ``predict``, before ``fit``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_members: int = Field(default=50, ge=1)
    learning_rate: float = Field(default=1.0, gt=0.0)
    max_depth: int = Field(default=1, ge=1)
    random_seed: int | None = None

    LEARNED_STATE: ClassVar[tuple[str, ...]] = (
        "_feature_names",
        "_members",
        "_member_voices",
        "_member_errors",
        "_n_classes",
    )

    _feature_names: tuple[str, ...] | None = PrivateAttr(default=None)
    _members: tuple[DecisionTreeClassifier, ...] | None = PrivateAttr(default=None)
    _member_voices: FloatArray | None = PrivateAttr(default=None)
    _member_errors: FloatArray | None = PrivateAttr(default=None)
    _n_classes: int | None = PrivateAttr(default=None)

    @property
    def n_classes(self) -> int:
        """How many classes the fit saw."""
        self._check_fitted()
        assert self._n_classes is not None
        return self._n_classes

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The columns this model was fitted on, in order."""
        self._check_fitted()
        assert self._feature_names is not None
        return self._feature_names

    @property
    def members(self) -> tuple[DecisionTreeClassifier, ...]:
        """The weak learners, in the order they were fitted."""
        self._check_fitted()
        assert self._members is not None
        return self._members

    @property
    def n_members_fitted(self) -> int:
        """How many rounds actually ran, which may be fewer than asked for.

        The walk stops when a learner has nothing left to learn from, or when
        one comes back worse than guessing. Both are real outcomes rather than
        failures, and reading this beside ``n_members`` is how a caller sees
        which happened.
        """
        return len(self.members)

    @property
    def member_voices(self) -> FloatArray:
        """How loudly each member votes, one per fitted member.

        Positive throughout, since a member that would have earned a negative
        voice ends the walk instead. Larger means the member was more often
        right on the rows that were hard at the time, which is not the same as
        being more often right overall.
        """
        self._check_fitted()
        assert self._member_voices is not None
        return self._member_voices

    @property
    def member_errors(self) -> FloatArray:
        """Each member's weighted error at the moment it was fitted.

        Worth reading as a sequence rather than a summary: it should hover just
        under ``1 - 1/K`` rather than fall, because every round hands the next
        learner a harder problem on purpose. A run of errors near zero means
        the members are not weak and the ensemble is one model repeated.
        """
        self._check_fitted()
        assert self._member_errors is not None
        return self._member_errors

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Run the rounds, reweighting the rows after each.

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
        SingleClassError
            If the target holds fewer than two classes, or leaves a gap in the
            run from zero.
        DivergenceError
            If the very first learner comes back no better than guessing. Every
            later round may stop for that reason without complaint, since there
            is already an ensemble to keep; on the first there is nothing, and
            an empty ensemble that reported itself fitted would answer with a
            constant.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        target_column = target_values.column
        target_column.check_is_label_encoded()

        truth = np.asarray(target_column.values, dtype=np.float64)
        n_classes = target_column.n_classes
        guessing = 1.0 - 1.0 / n_classes

        weights = np.full(truth.size, 1.0 / truth.size, dtype=np.float64)
        members: list[DecisionTreeClassifier] = []
        voices: list[float] = []
        errors: list[float] = []

        for _ in range(self.n_members):
            member = DecisionTreeClassifier(
                max_depth=self.max_depth, random_seed=self.random_seed
            ).fit(input_values, target_values, weights)

            wrong = np.asarray(member.predict(input_values)) != truth
            error = float(weights[wrong].sum() / weights.sum())
            voice = self._voice_for(error, n_classes)

            # The test is on the voice rather than on the error, and that is
            # not a restatement. "No better than guessing" is exactly "would
            # earn a voice of zero", and comparing the error against
            # ``1 - 1/K`` puts a subtraction between the two: a stump on a
            # constant column is wrong on exactly ``2/3`` of three balanced
            # classes, which is one ulp below ``1 - 1/3``, so the error test
            # lets it through and admits a member worth nothing.
            if voice <= SILENT_MEMBER_VOICE:
                if not members:
                    raise DivergenceError(
                        f"the first weak learner was wrong on {error:.3f} of "
                        f"the weight, and guessing uniformly among "
                        f"{n_classes} classes is wrong on {guessing:.3f}, so "
                        "it has nothing to contribute and neither would the "
                        "ensemble. Give the learner more depth, or more "
                        "informative columns"
                    )
                break

            members.append(member)
            voices.append(voice)
            errors.append(error)

            if error <= 0.0:
                # Nothing was got wrong, so nothing would be reweighted and
                # every later round would repeat this member exactly.
                break

            weights = weights * np.exp(voice * wrong)
            weights = weights / weights.sum()

        self._feature_names = tuple(feature.name for feature in feature_set)
        self._members = tuple(members)
        self._member_voices = np.asarray(voices, dtype=np.float64)
        self._member_errors = np.asarray(errors, dtype=np.float64)
        self._n_classes = n_classes

        self._mark_fitted()
        return self

    def _voice_for(self, error: float, n_classes: int) -> float:
        """``learning_rate * (log((1 - err) / err) + log(K - 1))``.

        The ``log(K - 1)`` is what generalises the two-class formula, and it
        is also the bar: the whole expression is positive exactly when the
        learner beats guessing uniformly.
        """
        if error <= 0.0:
            return self.learning_rate * PERFECT_MEMBER_VOICE

        return float(
            self.learning_rate * (np.log((1.0 - error) / error) + np.log(n_classes - 1))
        )

    def vote_shares(self, input_values: Sequence[Feature]) -> FloatArray:
        """Each class's share of the members' voices, one row per query.

        The quantity the decision is made on. Every member names one class per
        row and contributes its whole voice to it, so a row's shares sum to one
        and a class with no votes gets exactly zero -- which is the difference
        from a probability and the reason this is exposed under its own name.

        Returns
        -------
        FloatArray
            ``(n_rows, n_classes)``, non-negative and summing to one.
        """
        self._check_fitted()
        assert self._members is not None
        assert self._member_voices is not None
        assert self._n_classes is not None

        n_rows = FeatureSet(input_values).n_samples
        shares = np.zeros((n_rows, self._n_classes), dtype=np.float64)

        for member, voice in zip(self._members, self._member_voices, strict=True):
            chosen = np.asarray(member.predict(input_values)).astype(np.int64)
            shares[np.arange(n_rows), chosen] += voice

        return shares / self._member_voices.sum()

    def margins(self, input_values: Sequence[Feature]) -> FloatArray:
        """Each class's share less the share spread across the others.

        ``share_k - (1 - share_k) / (K - 1)``, which runs from ``-1/(K - 1)``
        when no member picked the class to ``1`` when every member did. The
        form the method's own convergence argument is written in, and the one
        scikit-learn's ``decision_function`` reports.

        A monotone transform of :meth:`vote_shares`, so it cannot reorder the
        classes and ``predict`` is free to read either.
        """
        assert self._n_classes is not None
        shares = self.vote_shares(input_values)

        return shares - (1.0 - shares) / (self._n_classes - 1)

    def predict_probabilities(self, input_values: Sequence[Feature]) -> ClassScores:
        """The margins turned into something that behaves like a probability.

        A softmax over the margins divided by ``K - 1``. The softmax is not
        decoration: the shares themselves are hard votes, so a class no member
        picked would otherwise be given a probability of exactly zero, which no
        amount of later evidence could revise.

        Returns
        -------
        ProbabilityMatrix
            Rows summing to one, and none of them zero.
        """
        self._check_fitted()
        assert self._n_classes is not None

        return ProbabilityMatrix(
            stable_softmax(self.margins(input_values) / (self._n_classes - 1))
        )

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The class the members' voices most favour.

        Read off the shares rather than the probabilities, since the softmax
        cannot reorder them.
        """
        return Predictions.already_checked(
            np.argmax(self.vote_shares(input_values), axis=1).astype(np.float64)
        )
