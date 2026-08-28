"""Two frames for models made of other models, and why they are two.

An ensemble holds members and combines them. That much is shared, and it is
almost the only thing that is, because the two families disagree about
everything that matters: how a member is fitted, what a member should be, and
which half of the error they are attacking.

Averaging: independent members, fighting variance
-------------------------------------------------
Fit many members on bootstrap resamples and average them. Members never see
each other, so the order they are fitted in carries no meaning and they could
be fitted in parallel.

Averaging reduces variance and leaves bias alone. With ``B`` members of variance
``s^2`` and pairwise correlation ``r``::

    Var(average)  =  r * s^2  +  (1 - r) * s^2 / B

Only the second term shrinks with ``B``, so ``r`` is a floor no number of
members can get below. Bagging attacks the second term; a random forest attacks
``r`` itself, by restricting which features each *node* may consider so that the
members stop all finding the same strong split first. Measured on six features,
thirty bagged trees put the same feature at the root thirty times; allowed three
features per node, five different features appeared there.

Because averaging fights variance, the ideal member is low-bias and
high-variance -- a deep, unpruned tree. That is a direct reversal of everything
the stopping rules exist for, and it is correct: in a lone tree they are the
only defence, and in a forest the averaging has taken that job over.

Boosting: dependent members, fighting bias
------------------------------------------
Fit one member, see what it got wrong, fit the next to *that*, and add it in.
Members are a sequence and cannot be reordered or parallelised, because round
seven is defined by what rounds one through six failed to explain.

Boosting reduces bias, so the ideal member is the opposite: high-bias and
low-variance, a stump of depth one or two. Deep members here overfit fast,
because nothing is averaging their noise away -- it is being *added up*.

The learning rate is what stops that. Adding only a fraction of each member's
prediction means more rounds are needed and each one commits less, which trades
computation for a model that does not sprint past the answer. Small rate with
many rounds beats large rate with few, consistently.

Neither frame cares what its members are
----------------------------------------
Both take a prototype model and deep-copy it per member, so an ensemble of
anything is expressible. Trees are the usual choice because they have the
properties each family wants -- unstable and low-bias for averaging, cheap and
high-bias when shallow for boosting -- but nothing here requires one.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from typing import Self, TypeAlias

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.base.estimator import Estimator, Fittable, Regressor
from oop_ml.core.data.column import Column
from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.ensemble.bootstrap import BootstrapSample
from oop_ml.core.ensemble.member_predictions import (
    MemberPredictions,
    predictions_of,
)
from oop_ml.core.ensemble.out_of_bag import OutOfBagEstimate
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.importance.importances import (
    FeatureContribution,
    FeatureImportances,
    Reports,
)
from oop_ml.core.types import FloatArray, MaskArray

# What a member is, and why the two frames disagree about it.
#
# Neither is ``Fittable``. That is the shared base of ``Estimator`` and
# ``Transformer``, so it deliberately declares no ``fit`` -- the two take
# different arguments and no one signature covers both. A member learns
# features against a target, which is what ``Estimator`` says.
#
# An averaging member can be either task: a bagged classifier's members are
# classifiers, and the frame only ever asks them to fit. It reads their answers
# through ``_member_answer``, which is a seam precisely because a regressor and
# a classifier are asked different questions.
#
# A boosting member is always a *regressor*, including inside a boosted
# classifier. What a round fits is a gradient, and a gradient is a continuous
# quantity whatever the task's own target looks like -- for squared error it is
# ``y - prediction``, for log loss it is ``y - probability``, and neither is a
# class. The frame therefore calls ``predict`` on its members directly, so it
# needs the type that has one.
AveragingMember: TypeAlias = Estimator[Sequence[Feature], Feature]
BoostingMember: TypeAlias = Regressor[Sequence[Feature], Feature]


class AveragingEnsemble(Fittable):
    """Members fitted independently on resamples, then combined.

    Parameters
    ----------
    n_members:
        How many to fit. More never makes an averaging ensemble worse -- it
        only stops helping, once the correlation floor is reached. That
        one-sidedness is what separates this from every other hyperparameter
        in the library, where more of something eventually hurts.
    random_seed:
        Fixes every resample, so a fit is reproducible.
    """

    n_members: int = Field(default=100, ge=1)
    random_seed: int | None = None

    _feature_names: tuple[str, ...] | None = PrivateAttr(default=None)
    _members: tuple[AveragingMember, ...] | None = PrivateAttr(default=None)
    _samples: tuple[BootstrapSample, ...] | None = PrivateAttr(default=None)

    # Kept for the same reason NeighbourModel keeps its rows, and at the same
    # cost: the out-of-bag estimate asks what each member says about training
    # rows it never drew, and there is no way to ask that after the data has
    # been let go. Every other model here discards its training set once the
    # parameters are learned.
    _training: Dataset | None = PrivateAttr(default=None)

    @property
    def samples(self) -> tuple[BootstrapSample, ...]:
        """The resample each member was fitted on, in member order.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._samples is not None
        return self._samples

    @property
    def feature_importances(self) -> FeatureImportances:
        """The members' importances, averaged.

        This is the whole reason the measure becomes worth reading. A single
        tree's importances are as unstable as the tree itself: change a few
        rows, the root split changes, and the feature that was carrying half
        the explanation drops to nothing. Averaging over a hundred members that
        each saw a different resample is what turns a reading that moves under
        you into one that holds still.

        Averaging shares rather than raw totals, because each member's totals
        are on its own scale -- a deeper member removed more impurity in
        absolute terms without that meaning its features mattered more. Each
        member's shares go in as contributions and
        ``from_contributions`` does the totalling, which is the same entry
        point a single tree's walk uses.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        InvalidValuesError
            If the members cannot report importances. Mean decrease in impurity
            is a tree measure and a bagged linear model has coefficients
            instead, so reach for
            :class:`~oop_ml.core.importance.permutation.PermutationImportance`
            there, which does not care what it is measuring.
        """
        assert self._feature_names is not None
        contributions = []

        for member in self.members:
            if not isinstance(member, Reports):
                raise InvalidValuesError(
                    f"{type(member).__name__} cannot report feature importances; "
                    "use PermutationImportance instead"
                )
            contributions.extend(
                FeatureContribution(one.name, one.value)
                for one in member.feature_importances
            )

        return FeatureImportances.from_contributions(contributions, self._feature_names)

    @property
    def members(self) -> tuple[AveragingMember, ...]:
        """The fitted members.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._members is not None
        return self._members

    @abstractmethod
    def _prototype(self, position: int) -> AveragingMember:
        """The model member ``position`` is fitted from.

        A copy per member rather than the prototype itself, because fitting one
        shared object ``n_members`` times leaves every slot holding whichever
        member went last -- the same trap ``OneVsRestClassifier`` documents.

        It takes the position because a member can carry randomness of its own,
        and every member getting the ensemble's single ``random_seed`` would
        make that randomness identical across members. A forest is exactly that
        case: twenty trees seeded the same way draw the same feature
        restriction at every node, and the decorrelation the forest exists for
        silently does not happen. Nothing else about a member depends on which
        one it is.
        """

    @abstractmethod
    def _combine(self, member_predictions: MemberPredictions) -> FloatArray:
        """Turn every member's answer into the ensemble's answer.

        The one line separating an averaging regressor from an averaging
        classifier, mirroring what ``_combine`` does for the neighbour models.

        Parameters
        ----------
        member_predictions:
            Members on axis 0: entry ``i`` is what member ``i`` said about
            every query. The rest of the shape is whatever ``_member_answer``
            returns, so a regressor sees ``(n_members, n_queries)`` and a
            classifier ``(n_members, n_queries, n_classes)``.

        Returns
        -------
        FloatArray
            ``(n_queries,)``, one answer per query.
        """

    def _prepared(
        self, input_values: Sequence[Feature], target_values: Feature
    ) -> Dataset:
        """Validate the inputs, remember the column order, hand back one object.

        The whole of the coercion, so that ``_fit_members`` is only the algorithm.
        It checks what the frame promises to check, applies the task's own
        target rule through ``_validated_target`` -- which is where a
        classifier insists on whole class positions -- records the feature
        names ``predict`` will match against later, and returns the pairing.

        A :class:`~oop_ml.core.data.dataset.Dataset` rather than two values,
        because the two must stay row-aligned through every resample and
        ``select_rows`` is what keeps them so. To fit a member on a bootstrap
        sample::

            resample = dataset.select_rows(sample.drawn)
            member.fit(resample.input_features, resample.target_feature)

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        """
        dataset = Dataset(input_values, target_values)
        self._feature_names = tuple(feature.name for feature in dataset.input_features)

        # For its checking, and for the tightening a classifier adds to it.
        # The values are the target's own, so the dataset already holds them.
        self._validated_target(target_values)

        return dataset

    def _fit_members(
        self, input_values: Sequence[Feature], target_values: Feature
    ) -> Self:
        """Draw a resample per member, fit a copy of the prototype on each.

        Every member is fitted on ``n`` rows drawn with replacement from the
        same ``n`` rows, so they see the same *quantity* of data and a
        different *selection* of it. That difference is the entire mechanism:
        identical members would average to themselves and buy nothing.

        Draws come from one generator seeded once, so the members differ from
        each other while the fit as a whole stays reproducible.

        ``self._prepared(input_values, target_values)`` does the validation and
        hands back a :class:`~oop_ml.core.data.dataset.Dataset`, so nothing
        here has to touch an array: draw a
        :class:`~oop_ml.core.ensemble.bootstrap.BootstrapSample` per member,
        ``select_rows`` it, and fit a ``deepcopy`` of ``_prototype(position)``
        on the result. Keep the samples in ``self._samples`` alongside the
        members in ``self._members``, and finish with ``self._mark_fitted()``.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        """
        dataset = self._prepared(input_values, target_values)
        generator = np.random.default_rng(self.random_seed)

        # Every draw comes from the one generator, in order, so the members
        # differ from each other while the fit as a whole stays reproducible.
        self._samples = tuple(
            BootstrapSample.draw(dataset.n_samples, generator)
            for _ in range(self.n_members)
        )

        members = []
        for position, sample in enumerate(self._samples):
            # The same rows taken from every column at once, repeats included,
            # which is what keeps the predictors and the target aligned.
            resample = dataset.select_rows(sample.drawn)

            # A copy per member. Bagging's prototype is one shared object, so
            # fitting it n_members times would leave every slot holding
            # whichever member went last.
            member = deepcopy(self._prototype(position))
            members.append(member.fit(resample.input_features, resample.target_feature))

        self._members = tuple(members)
        self._training = dataset
        self._mark_fitted()
        return self

    def out_of_bag_estimate(self) -> OutOfBagEstimate:
        """Predict each training row using only the members that missed it.

        Line the samples up as a ``(n_members, n_rows)`` grid of "did this
        member draw this row". Reading *down* a column gives the members
        entitled to judge that row, because they are the ones that never saw
        it, and averaging only those is an honest prediction for a row the
        model was nonetheless fitted on.

        Every row has a different set of judges and the sets are different
        sizes, so this is not one ensemble answering ``n_rows`` times. Rows
        that every member happened to draw have no judges at all and no
        prediction is available for them; mark them false in ``covered``
        rather than inventing one. At a hundred members that is vanishingly
        rare, and at five it happens to roughly one row in ten.

        Combine the judges' answers with ``_combine``, the same seam
        ``predict`` uses, so a regressor averages and a classifier takes the
        most probable class. Feed it only the rows of ``member_predictions``
        belonging to that row's judges.

        Returns
        -------
        OutOfBagEstimate
            The predictions, which rows they cover, and how many members stood
            behind each one.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        member_predictions = self._training_member_predictions()
        in_bag = self._in_bag_grid()
        n_rows = in_bag.shape[1]

        # Counting the misses down each column gives every row's judge count in
        # one pass, and a row is covered exactly when that count is not zero.
        judges = (~in_bag).sum(axis=0).astype(np.float64)
        covered = judges > 0

        # NaN rather than zero for the uncovered rows. Zero is a number a
        # caller could average by accident; NaN poisons anything that forgets
        # to mask, which is the failure worth having.
        predictions = np.full(n_rows, np.nan, dtype=np.float64)

        for row in np.flatnonzero(covered):
            missed = np.flatnonzero(~in_bag[:, row])

            # for_query keeps the query axis. Indexing a single row with a
            # scalar drops it, and _combine would then reduce across members
            # and queries at once -- the right type and the wrong number.
            judged = member_predictions.for_query(int(row), missed)
            predictions[row] = self._combine(judged)[0]

        return OutOfBagEstimate(predictions, covered, judges)

    def _training_member_predictions(self) -> MemberPredictions:
        """Every member's answer about every *training* row, uncombined.

        ``(n_members, n_rows)`` for a regressor and
        ``(n_members, n_rows, n_classes)`` for a classifier, which is the same
        shape ``_combine`` takes. Members are asked about rows they drew as
        well as rows they missed; the out-of-bag rule is applied afterwards, by
        selecting rather than by re-predicting.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        self._check_fitted()
        assert self._training is not None

        return self._member_predictions(self._training.input_features)

    def _in_bag_grid(self) -> MaskArray:
        """``(n_members, n_rows)`` -- true where that member drew that row.

        Reading down column ``i`` and negating gives the members allowed to
        judge row ``i``.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        return np.array([sample.in_bag for sample in self.samples])

    def _member_predictions(self, input_values: Sequence[Feature]) -> MemberPredictions:
        """``(n_members, n_queries)`` -- every member's answer, uncombined.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        ordered = list(self._matched_rows(input_values))

        return predictions_of(
            [self._member_answer(member, ordered) for member in self.members]
        )

    @abstractmethod
    def _member_answer(
        self, member: AveragingMember, input_values: Sequence[Feature]
    ) -> FloatArray:
        """What one member says about these queries.

        A seam because the two tasks ask a member different questions: a
        regressor wants ``predict``, and a classifier wants the *probabilities*
        rather than the labels. Averaging hard labels throws away how sure each
        member was, and a member that was barely certain would then count as
        much as one that was overwhelmingly so.
        """

    def _validated_target(self, target_values: Feature) -> Column:
        """The target as a column, checked against what the task requires.

        The seam every frame in this library has. The default is the regression
        answer: a column is already numeric and finite.
        """
        return target_values.column

    def _matched_rows(self, input_values: Sequence[Feature]) -> FeatureSet:
        """The query columns, in the order the fit saw them.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        self._check_fitted()
        assert self._feature_names is not None

        return FeatureSet.matching(self._feature_names, input_values)


class BoostingEnsemble(Fittable):
    """Members fitted in sequence, each to what the ones before it missed.

    Parameters
    ----------
    n_rounds:
        How many members to add. Unlike an averaging ensemble, more *does*
        eventually hurt -- the training error keeps falling long after the
        held-out error has turned around, because each round is fitted to
        whatever is left, and noise is what is left.
    learning_rate:
        The share of each member's prediction that is added. Below 1 the
        ensemble commits less per round and needs more of them, which is
        almost always the better trade.
    """

    n_rounds: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)

    _feature_names: tuple[str, ...] | None = PrivateAttr(default=None)
    _members: tuple[BoostingMember, ...] | None = PrivateAttr(default=None)
    _initial_prediction: float | None = PrivateAttr(default=None)

    @property
    def members(self) -> tuple[BoostingMember, ...]:
        """The fitted members, in the order they were added.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._members is not None
        return self._members

    @property
    def initial_prediction(self) -> float:
        """What the ensemble predicted before any member was added.

        Boosting starts from a constant -- the target's mean for squared error
        -- and every member is a correction to it. Without it the first round
        would have to explain the target's level as well as its structure, and
        a shallow member cannot do both.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._initial_prediction is not None
        return self._initial_prediction

    @abstractmethod
    def _prototype(self, round_number: int) -> BoostingMember:
        """The model round ``round_number`` is fitted from, counting from 1.

        Shallow, for this family. A deep member drives the residuals to zero
        in one round and leaves the rest of the ensemble fitting noise.

        Nothing in plain gradient boosting varies by round -- the number is
        here because the frame mirrors the averaging one, and because the
        stochastic variant does subsample per round.
        """

    @abstractmethod
    def _residuals(
        self, target_values: FloatArray, predictions: FloatArray
    ) -> FloatArray:
        """What the ensemble has failed to explain, in the units a member fits.

        For squared error this is simply ``target - prediction``, which is why
        the family is often introduced as "fit the residuals". That is a
        special case: in general it is the negative gradient of the loss with
        respect to the prediction, which is what makes the same machinery work
        for losses whose residual is not a subtraction.
        """

    def _prepared(
        self, input_values: Sequence[Feature], target_values: Feature
    ) -> Dataset:
        """Validate the inputs, remember the column order, hand back one object.

        The whole of the coercion, so that ``_fit_rounds`` is only the algorithm.
        It checks what the frame promises to check, applies the task's own
        target rule through ``_validated_target`` -- which is where a
        classifier insists on whole class positions -- records the feature
        names ``predict`` will match against later, and returns the pairing.

        A :class:`~oop_ml.core.data.dataset.Dataset` rather than two values,
        because the two must stay row-aligned through every resample and
        ``select_rows`` is what keeps them so. To fit a member on a bootstrap
        sample::

            resample = dataset.select_rows(sample.drawn)
            member.fit(resample.input_features, resample.target_feature)

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        """
        dataset = Dataset(input_values, target_values)
        self._feature_names = tuple(feature.name for feature in dataset.input_features)

        # For its checking, and for the tightening a classifier adds to it.
        # The values are the target's own, so the dataset already holds them.
        self._validated_target(target_values)

        return dataset

    def _fit_rounds(
        self, input_values: Sequence[Feature], target_values: Feature
    ) -> Self:
        """Start from a constant, then add one correction per round.

        Each round fits a member to what is currently unexplained, adds
        ``learning_rate`` times its prediction to the running total, and
        recomputes what is left.

        ``self._prepared(input_values, target_values)`` does the validation and
        hands back a :class:`~oop_ml.core.data.dataset.Dataset`. No resampling
        here -- every round sees all the rows -- so what varies is the target:
        wrap each round's residuals in a ``Feature`` and fit a ``deepcopy`` of
        ``_prototype(round_number)`` on ``dataset.input_features`` against it.
        Set ``self._initial_prediction`` before the first round, since the
        residuals are measured against it, and finish with
        ``self._mark_fitted()``.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        """
        dataset = self._prepared(input_values, target_values)
        features = dataset.input_features
        targets = dataset.target_feature.values

        self._initial_prediction = float(targets.mean())

        running = np.full(targets.size, self._initial_prediction, dtype=np.float64)

        members: list[BoostingMember] = []
        for round_number in range(1, self.n_rounds + 1):
            # What the ensemble so far has failed to explain. This, not
            # the target, is what the round learns -- the target's level was
            # settled by the starting constant and every round after it is a
            # correction.
            residuals = self._residuals(targets, running)

            member = deepcopy(self._prototype(round_number))
            member.fit(features, Feature("residual", residuals))

            # Fold the round in before the next one measures what is left.
            # Without this the residuals never move, every round fits the same
            # member, and the ensemble is one tree repeated n_rounds times.
            running = running + self.learning_rate * member.predict(features).values
            members.append(member)

        self._members = tuple(members)
        self._mark_fitted()
        return self

    def _validated_target(self, target_values: Feature) -> Column:
        """The target as a column, checked against what the task requires."""
        return target_values.column

    def _matched_rows(self, input_values: Sequence[Feature]) -> FeatureSet:
        """The query columns, in the order the fit saw them.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        self._check_fitted()
        assert self._feature_names is not None

        return FeatureSet.matching(self._feature_names, input_values)
