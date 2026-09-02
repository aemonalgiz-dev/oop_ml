"""Reach the principal components by a rule that reads only two numbers at a time.

Why this sits beside ``PrincipalComponentAnalysis``
---------------------------------------------------
It answers the same question and shares none of its machinery, and that is the
entire reason it is here. ``PrincipalComponentAnalysis`` forms the covariance
matrix and eigendecomposes it, which is a global calculation over the whole
dataset at once. This model never forms a covariance matrix, never calls an
eigensolver, and never looks at two rows at the same time. It walks the rows one
at a time, and after each one it nudges a weight vector by an amount that
depends on nothing but the row it just saw and the weights it already held.

The two routes land on the same answer. Measured on ``TILTED_GRID`` below, whose
components are known exactly, the angle between the learned direction and the
eigenvector agrees to ``1 - |cos| = 4.9e-09`` on the first component and
``5.7e-08`` on the second. That agreement is the point of the module, and the
test asserting it is the most valuable one in the spec.

What a local learning rule is, and why it is worth a model
-----------------------------------------------------------
Every other learner in the network package here changes a weight by an amount
computed from a loss measured somewhere else and carried back to it. That is
backpropagation, and the signal a weight receives has travelled through every
layer between it and the answer.

A **local** rule has no such signal. The change to the weight joining input
``i`` to output ``j`` is a function of the value at ``i`` and the value at ``j``
and nothing else. No loss, no target, no gradient propagated from a distance.
Donald Hebb's original statement of it in 1949 was about biology rather than
arithmetic, and the usual paraphrase is that cells which fire together wire
together. Written down, that is

    weights += rate * output * row

which is the outer product of the two ends. Nothing in it knows what the network
is for.

What this model teaches is that such a rule is not merely biologically
suggestive but **sufficient**. Applied to centred data it converges on the
principal components, which is to say that the answer an eigendecomposition
gives is reachable without ever assembling the matrix the eigendecomposition
needs. That claim is surprising, it is checkable, and nothing else in this
library demonstrates it.

Plain Hebb diverges, and the fix is one subtracted term
--------------------------------------------------------
The rule above has nothing pushing back. Every update adds a multiple of the row
in the direction the weights already lean, so the weights grow, which makes the
output larger, which makes the next update larger. Measured on ``TILTED_GRID``
at a fixed rate of 0.05, starting from a unit vector, the weight vector's length
runs 1.48, 2.25 and 3.67 over the first three epochs, reaches 1193 by epoch ten
and 1.0e+07 by epoch twenty. It is not converging to a direction, it is
exploding along one.

Erkki Oja's correction is to subtract what the weights have already accounted
for. With ``output = weights . row``,

    weights += rate * output * (row - output * weights)

Over the same twenty epochs on the same fixture that stays between 1.008 and
1.018, which is the whole demonstration. The subtracted term is a decay
proportional to ``output^2``, so a weight vector that has grown is pulled back
harder than one that has not, and the fixed point of the length is exactly 1.
Nothing normalises the vector by hand. **The unit length is a result, not a
step**, which is why this implementation must never divide by the norm and why
the spec asserts the lengths land near 1 on their own. Rescaling them would
delete the evidence that the rule works.

The direction the length settles at is the first eigenvector of the covariance
of the data, and the reason is worth one line. Averaged over the rows, the
update is ``rate * (C w - (w' C w) w)`` for the covariance ``C``, which is zero
exactly when ``C w`` is parallel to ``w``. That is the definition of an
eigenvector, and only the largest eigenvalue's is stable under the perturbation.

More than one component needs deflation
----------------------------------------
Run Oja's rule on two outputs independently and both find the same direction,
because both are solving the same problem. Terence Sanger's generalised Hebbian
algorithm makes each output see only what the ones before it left behind:

    weights[i] += rate * output[i] * (row - sum over j <= i of output[j] * weights[j])

**The sum runs up to and including ``i`` itself.** That is not a detail. The
``j == i`` term is Oja's own normalisation, so it is what keeps component ``i``
at unit length, and the terms ``j < i`` are the deflation, so they are what
stops component ``i`` collapsing onto component 1. Write ``range(i)`` where
``range(i + 1)`` was meant and the first component loses its normalisation
entirely, which is to say it becomes plain Hebb. Measured on ``TILTED_GRID``
with the defaults here, that variant overflows to non-finite weights by epoch
15, with the first component's weights already at ``-1.3e+04`` by the time the
second one overflowed, while the correct rule finishes 200 epochs with lengths
1.00027 and 1.00119.

The failure is loud on this fixture and it is not loud everywhere. On data whose
first eigenvalue is small enough for the unnormalised component to grow slowly,
the same mistake produces a fit that completes and reports directions that are
merely wrong, which is why the spec pins it directly rather than trusting the
divergence to catch it.

What this buys over an eigendecomposition, stated honestly
-----------------------------------------------------------
Two things, and they are narrow.

**Memory.** The covariance matrix of ``p`` features is ``p^2`` numbers. These
weights are ``k * p``. At 1000 features that is 8.0 MB against 80 kB for ten
directions, and at 10000 features it is 800 MB against 800 kB, a thousandfold.
For a decomposition of something genuinely wide the matrix is the thing that
does not fit, and this route never builds it.

**Streaming.** The update reads one row and then forgets it. A fit could
therefore be driven by rows arriving over time, from a source too large to hold,
and the weights would track a distribution that changes underneath them. The
implementation here does hold the training block, because it is fitted from a
``FeatureSet`` like every other model in this library, but nothing in the rule
requires that and the row loop is written so it is visible.

Against those, the costs are real and larger for anything that fits in memory.
It is **slower**, by a lot. On 400 rows of 3 features, three components, 200
epochs, the walk takes 836 ms against ``PrincipalComponentAnalysis``'s 0.160 ms,
a ratio of 5231x. Most of that is an irreducible Python loop over rows, which is
what a genuinely online rule looks like and is not something numpy can vectorise
away without turning it back into a batch calculation.

It is also **less accurate**, though by less than the speed gap suggests. On
``TILTED_GRID`` the variances come back 2.85714284 and 1.14285734 against exact
values of 20/7 and 8/7, absolute errors of 1.7e-08 and 2.0e-07. On a wider
three-feature block of 400 rows the first component's variance is out by
3.4e-05, a relative error of 8.9e-06. Those are approximation errors of the walk
rather than rounding, and they shrink with more epochs and a better-chosen rate.

The rate is not scale-free, and that is the trap
-------------------------------------------------
Oja's rule is stable while the rate is small against the reciprocal of the
largest eigenvalue, and an eigenvalue carries the square of the data's units. So
a rate that behaves on data of variance 1 diverges on the same data measured in
different units. Measured while choosing the defaults here, a start rate of 0.05
that works perfectly on both fixtures below overflows to non-finite weights by
epoch 10 on a three-feature block whose leading variance is 8.65.

Two things follow. The fit refuses a diverged walk by name rather than
completing and answering ``nan`` to everything, which is the same
:class:`~oop_ml.core.exceptions.DivergenceError` the iterative solvers raise.
And the honest advice for data whose columns are measured in different things is
to standardize first, exactly as it is for
``PrincipalComponentAnalysis(standardize=True)``, with the difference that there
it is a question about which matrix is decomposed and here it is also a question
about whether the walk survives.

The rate must also decay, and the reason is in
:mod:`oop_ml.core.schedule`. A constant rate fails the second Robbins-Monro
condition, so the weights keep chasing whichever row arrived last and the fit
becomes a fact about presentation order.

Why the PCA vocabulary could not be reused
-------------------------------------------
:class:`~oop_ml.core.decomposition.components.PrincipalComponents` enforces two
invariants that a correct fit here breaks, so reusing it would refuse honest
answers. This is the same judgement kernel PCA made and for the same kind of
reason.

**Orthogonality.** That class refuses a pair of directions whose dot product
exceeds 1e-08. Sanger's deflation drives the directions apart but only to within
the residual the finite walk leaves, and the measured worst pair here is 2.4e-04
on ``TILTED_GRID`` and 3.8e-03 on that wider block, four to five orders of
magnitude outside it. Asserting exact orthogonality would be asserting something
false.

**Unit length.** That class refuses a direction whose length is more than 1e-08
from 1. The lengths here land at 1.00027 and 1.00119, which is the correct
answer for a walk of 200 epochs. Worse, a *slow* component comes back short
rather than long, at 0.476 in one measured configuration, so a tight guard would
refuse an under-converged fit as if it were a broken one when the honest report
is that it under-converged.

So :class:`HebbianDirection` keeps the learned vector at whatever length the
rule left it, exposes that length as the convergence diagnostic it is, and
offers the normalised unit vector separately for the projection.
:class:`HebbianDirections` checks what is genuinely invariant, which is that the
names are unique and every direction weights the same features, and reports
ordering and orthogonality as measurements instead of guaranteeing them.

One consequence is visible in the shares. Because the directions are neither
exactly unit nor exactly perpendicular, the variances along them can sum to
marginally more than the variance that exists. Measured on ``TILTED_GRID`` the
kept total is 1.000000045 of the true total. A share above 1 is not a bug here,
it is the price of the invariant PCA has and this does not.

Not persisted, and not invertible
----------------------------------
``LEARNED_STATE`` is empty, so this model is deliberately outside the
persistence registry. There is no codec for a ``HebbianDirections`` and adding
one is a format change to a shared file rather than a property of this module.

There is no ``inverse_transform`` either, and that is a refusal rather than an
omission. PCA can rebuild the original features with a transpose only because
its directions are exactly orthonormal, and these are not. Handing back
``coordinates @ directions`` and calling it a reconstruction would quietly
return something a few parts in ten thousand away from the closest point in the
kept subspace, which is the sort of small wrong answer nothing downstream can
notice.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import ClassVar, Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.convergent_fit import ConvergentFit
from oop_ml.core.base.estimator import Transformer
from oop_ml.core.data.coefficients import Coefficient, Coefficients
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.exceptions import (
    AllSameValuesError,
    DivergenceError,
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
    TooFewValuesError,
)
from oop_ml.core.schedule import ExponentialDecaySchedule, Schedule
from oop_ml.core.types import FloatArray
from oop_ml.decomposition.principal_component_analysis import COMPONENT_NAME_PREFIX

SMALLEST_USABLE_LENGTH = 1e-12
"""How short a learned vector may be and still name a direction.

A direction is a way of pointing, so the one thing it cannot be is nothing. A
zero vector has no direction to normalise towards and would make every
projection zero while every shape still conformed. This is the only length rule
here, and it is deliberately not a check that the length is near 1 -- see the
module docstring on why that check belongs to
:class:`~oop_ml.core.decomposition.components.PrincipalComponent` and not to
this family.
"""


class HebbianDirection:
    """One weight vector a local rule learned, and the variance along it.

    The difference from
    :class:`~oop_ml.core.decomposition.components.PrincipalComponent` is that
    this holds the vector at whatever length the walk left it rather than
    insisting it is a unit vector. That length is not noise to be tidied away.
    Oja's rule drives it to 1 on its own, so how close it sits to 1 is the
    clearest single number saying whether this component converged, and
    normalising the stored vector would throw that number away.

    Parameters
    ----------
    name:
        What this direction is called. Named by position, as a component is,
        because a blend of several features has no name of its own.
    weights:
        The learned vector, one weight per feature bound to that feature's
        name. These are synaptic weights rather than loadings, which is why
        they are not normalised.
    variance:
        The variance of the training data along the *unit* direction, so that
        it is comparable with an eigenvalue whatever length the weights came
        out at. Non-negative, since it is a variance.

    Raises
    ------
    InvalidValuesError
        If ``name`` is not a non-empty string, if ``variance`` is negative or
        not finite, or if the weights are not a finite vector long enough to
        point anywhere.
    """

    __slots__ = ("_name", "_variance", "_weights")

    def __init__(self, name: str, weights: Coefficients, variance: float) -> None:
        if not isinstance(name, str) or not name.strip():
            raise InvalidValuesError("HebbianDirection name must be a non-empty string")

        if not np.isfinite(variance):
            raise InvalidValuesError(f"{name} has a non-finite variance {variance}")

        if variance < 0.0:
            raise InvalidValuesError(
                f"{name} has variance {variance}, and a variance cannot be negative"
            )

        self._name = name.strip()
        self._weights = weights
        self._variance = float(variance)
        self._check_points_somewhere()

    def _check_points_somewhere(self) -> None:
        """Raise unless the weights are finite and long enough to be a direction.

        The whole of what is refused here. A non-finite weight is a diverged
        walk that got past its own guard, and a vector of length zero names no
        direction at all.
        """
        vector = self.vector

        if not np.all(np.isfinite(vector)):
            raise InvalidValuesError(
                f"{self._name} has non-finite weights, which is a diverged walk"
            )

        if self.length < SMALLEST_USABLE_LENGTH:
            raise InvalidValuesError(
                f"{self._name} has length {self.length}, which points nowhere"
            )

    @property
    def name(self) -> str:
        """What this direction is called."""
        return self._name

    @property
    def weights(self) -> Coefficients:
        """The learned vector, one weight per feature name, unnormalised."""
        return self._weights

    @property
    def variance(self) -> float:
        """The variance of the fitted data along this direction."""
        return self._variance

    @property
    def vector(self) -> FloatArray:
        """The learned weights as a plain array, in the features' order."""
        return np.array(
            [coefficient.value for coefficient in self._weights], dtype=np.float64
        )

    @property
    def length(self) -> float:
        """How long the learned vector is, which Oja's rule drives to 1.

        The convergence diagnostic for this component on its own. A length near
        1 means the normalising term reached its fixed point; a length well
        below it means the walk stopped early; a length well above it means the
        rate was too large for the data.
        """
        return float(np.linalg.norm(self.vector))

    @property
    def direction(self) -> FloatArray:
        """The learned vector scaled to length 1, which is what it means.

        Normalised here rather than at learning time, so that :attr:`length`
        survives to be read. This is the vector a projection uses and the one
        that is comparable with an eigenvector.
        """
        return self.vector / self.length

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The features this direction weights, in order."""
        return tuple(coefficient.name for coefficient in self._weights)

    def weight_for(self, name: str) -> float:
        """How much this direction leans on one feature, before normalising.

        Raises
        ------
        InvalidValuesError
            If ``name`` is not one of the features this direction weights.
        """
        return self._weights[name]

    def __repr__(self) -> str:
        return (
            f"HebbianDirection({self._name!r}, variance={self._variance:.4f}, "
            f"length={self.length:.6f})"
        )


class HebbianDirections:
    """The directions one walk learned, with the total variance they came from.

    What this checks and what it deliberately does not is the whole of its
    design, and the module docstring argues both. It checks the structural
    facts, which are that the names are unique and every direction weights the
    same features in the same order, because a projection that summed across
    mismatched columns would be arithmetic rather than a decomposition.

    It does **not** check that the variances decrease and it does not check that
    the directions are perpendicular.
    :class:`~oop_ml.core.decomposition.components.PrincipalComponents` checks
    both, and can, because an eigendecomposition guarantees both exactly.
    Sanger's rule guarantees neither. It drives the directions towards ordered
    and perpendicular and arrives within the residual a finite walk leaves, so
    the honest thing is to report where it got to. :attr:`worst_orthogonality`
    and the variances are those reports.

    Parameters
    ----------
    directions:
        The learned directions, in the order the rule produced them, which is
        the order deflation intends to be by decreasing variance.
    total_variance:
        The variance of the whole fitted dataset, which is the sum of every
        feature's variance after centring. This is the denominator of every
        share reported here. It is supplied rather than summed for the reason
        :class:`~oop_ml.core.decomposition.components.PrincipalComponents`
        gives, and for a second reason as well: these directions are not
        exactly perpendicular, so their variances do not partition anything and
        summing them would be adding up overlapping quantities.

    Raises
    ------
    EmptyValuesError
        If no directions are supplied.
    NonUniqueFeaturesError
        If two directions share a name.
    InvalidValuesError
        If the directions weight different features, or if ``total_variance``
        is not positive.
    """

    __slots__ = ("_directions", "_total_variance")

    def __init__(
        self, directions: Sequence[HebbianDirection], total_variance: float
    ) -> None:
        if not directions:
            raise EmptyValuesError("a decomposition needs at least one direction")

        self._directions = tuple(directions)
        self._total_variance = float(total_variance)

        self._check_names_are_unique()
        self._check_directions_weight_the_same_features()
        self._check_total_variance_is_positive()

    def _check_names_are_unique(self) -> None:
        names = [direction.name for direction in self._directions]

        if len(set(names)) != len(names):
            raise NonUniqueFeaturesError(f"direction names must be unique; got {names}")

    def _check_directions_weight_the_same_features(self) -> None:
        """Raise unless every direction spans the same features in one order."""
        expected = self._directions[0].feature_names

        for direction in self._directions[1:]:
            if direction.feature_names != expected:
                raise InvalidValuesError(
                    f"{direction.name} weights {direction.feature_names}, but "
                    f"{self._directions[0].name} weights {expected}"
                )

    def _check_total_variance_is_positive(self) -> None:
        """Raise unless there is any spread for the shares to be shares of.

        Zero total variance is data with no spread in any direction, where
        every share would be a division by zero and no direction means
        anything.
        """
        if self._total_variance <= 0.0:
            raise InvalidValuesError(
                f"total variance must be positive; got {self._total_variance}"
            )

    @property
    def n_components(self) -> int:
        """How many directions were learned."""
        return len(self._directions)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The original features every direction weights, in order."""
        return self._directions[0].feature_names

    @property
    def n_features(self) -> int:
        """How many features the walk started from."""
        return len(self.feature_names)

    @property
    def total_variance(self) -> float:
        """The variance of the whole fitted dataset, summed over its columns."""
        return self._total_variance

    @property
    def kept_variance(self) -> float:
        """The variance along the directions held here.

        Read this as a total rather than as a part. Perpendicular directions
        would partition the variance and these are only nearly perpendicular,
        so on a fit where two directions overlap slightly this counts the
        overlap twice.
        """
        return float(sum(direction.variance for direction in self._directions))

    @property
    def directions(self) -> FloatArray:
        """The unit directions as a matrix, one per row.

        Shape ``(n_components, n_features)``, which is the orientation a
        projection wants: ``centred_rows @ directions.T`` gives one column per
        direction. Normalised, so that a coordinate means a distance along the
        direction rather than a distance scaled by however long the walk left
        the vector.
        """
        return np.array(
            [direction.direction for direction in self._directions], dtype=np.float64
        )

    @property
    def vectors(self) -> FloatArray:
        """The learned weights as a matrix, one per row, unnormalised."""
        return np.array(
            [direction.vector for direction in self._directions], dtype=np.float64
        )

    @property
    def lengths(self) -> tuple[float, ...]:
        """Each learned vector's length, which Oja's rule drives to 1.

        The convergence report for the set. Read alongside ``converged``, since
        a walk can stop moving because its rate decayed rather than because it
        arrived, and these lengths are what says which happened.
        """
        return tuple(direction.length for direction in self._directions)

    @property
    def worst_orthogonality(self) -> float:
        """The largest absolute dot product between two different directions.

        Zero for a real eigendecomposition and never quite zero here. Measured
        on the fixtures in the spec it is 2.4e-04 and 3.8e-03, where
        :class:`~oop_ml.core.decomposition.components.PrincipalComponents`
        refuses anything above 1e-08. Exposed rather than checked because
        refusing it would refuse correct fits, and because a caller comparing
        this model with PCA wants the number rather than a guarantee.

        A single direction has no pair, so the answer there is 0.0.
        """
        if self.n_components < 2:
            return 0.0

        products = self.directions @ self.directions.T
        np.fill_diagonal(products, 0.0)

        return float(np.max(np.abs(products)))

    @property
    def variance_shares(self) -> tuple[float, ...]:
        """Each direction's variance over the whole dataset's variance.

        These can sum to marginally more than 1 even when every direction was
        kept, measured at 1.000000045 on the spec's hand-checkable fixture,
        because near-perpendicular directions overlap a little and the overlap
        is counted twice. That excess is a report on the fit rather than an
        error in it.
        """
        return tuple(
            direction.variance / self._total_variance for direction in self._directions
        )

    @property
    def cumulative_shares(self) -> tuple[float, ...]:
        """The running total of :attr:`variance_shares`."""
        return tuple(float(total) for total in np.cumsum(self.variance_shares))

    @property
    def is_ordered(self) -> bool:
        """Whether the variances came out decreasing, as deflation intends.

        A question rather than an invariant. Deflation makes each direction see
        only what the earlier ones left, so a converged walk comes out ordered,
        but two directions through nearly equal variance can settle either way
        round and neither is wrong. A caller who needs the order asserted can
        assert this; the constructor does not, because refusing it would refuse
        a correct fit on symmetric data.
        """
        variances = [direction.variance for direction in self._directions]

        return all(
            later <= earlier
            for earlier, later in zip(variances, variances[1:], strict=False)
        )

    def value_for(self, name: str) -> HebbianDirection:
        """The direction called ``name``.

        Raises
        ------
        InvalidValuesError
            If no direction has that name.
        """
        for direction in self._directions:
            if direction.name == name:
                return direction

        raise InvalidValuesError(
            f"unknown direction {name!r}; this set holds "
            f"{[direction.name for direction in self._directions]}"
        )

    def __getitem__(self, name: str) -> HebbianDirection:
        return self.value_for(name)

    def __contains__(self, name: object) -> bool:
        return any(direction.name == name for direction in self._directions)

    def __iter__(self) -> Iterator[HebbianDirection]:
        return iter(self._directions)

    def __len__(self) -> int:
        return self.n_components

    def __repr__(self) -> str:
        return (
            f"HebbianDirections(n_components={self.n_components}, "
            f"explained={self.cumulative_shares[-1]:.4f})"
        )


class HebbianPrincipalComponents(Transformer[Sequence[Feature]], ConvergentFit):
    """Find the principal components by Hebbian learning rather than by algebra.

    A :class:`~oop_ml.core.base.estimator.Transformer`, because it takes no
    target and rewrites the columns, and a
    :class:`~oop_ml.core.base.convergent_fit.ConvergentFit`, because it arrives
    at its answer over a number of passes rather than jumping to it. Neither
    base needed changing to hold it, which is what those two frames existing
    separately was for.

    Parameters
    ----------
    n_components:
        How many directions to learn. At 1 this is exactly Oja's rule, which is
        the historical starting point and the whole of the normalisation
        argument. Above 1 it is Sanger's generalised Hebbian algorithm, which
        is Oja's rule plus deflation. Must not exceed the number of features,
        since a direction beyond the last has nothing left to see.
    learning_rate:
        How far each row moves the weights, as a
        :class:`~oop_ml.core.schedule.Schedule` rather than a number. It has to
        decay for the walk to settle at all, for the Robbins-Monro reason that
        module gives, and the default falls geometrically from 0.05 to 0.0005
        because what matters about a rate is its order of magnitude. A
        :class:`~oop_ml.core.schedule.ConstantSchedule` is accepted, and
        produces a walk that keeps chasing whichever row arrived last, which is
        worth being able to demonstrate rather than being prevented from
        expressing.
    max_epochs:
        How many passes over the rows at most. Also the denominator the
        schedule decays against, so shortening the walk steepens the decay
        rather than truncating it.
    tolerance:
        Stop once no weight moved further than this in a whole pass. Looser
        than :class:`~oop_ml.core.base.convergent_fit.ConvergentFit`'s default
        of 1e-8, because a stochastic walk over ``n`` rows accumulates ``n``
        updates per pass and does not stand still the way a batch solver does.
    random_seed:
        Fixes both the starting weights and the order the rows are presented
        in. Presentation order genuinely matters for a rule applied one row at
        a time, which is why it is shuffled every epoch and why pinning it is
        what makes a fit reproducible.

    Raises
    ------
    pydantic.ValidationError
        If any count is below its minimum, or the tolerance is not positive.
        Field bounds are pydantic's to enforce, so the error is pydantic's too.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_components: int = Field(default=1, ge=1)
    learning_rate: Schedule = ExponentialDecaySchedule(start=0.05, end=0.0005)
    max_epochs: int = Field(default=200, ge=1)
    tolerance: float = Field(default=1e-06, gt=0.0)
    random_seed: int | None = None

    # Deliberately empty, so this model stays outside the persistence registry.
    # A saved document would need a codec for HebbianDirections, and adding one
    # is a change to a shared format file rather than a property of this module.
    LEARNED_STATE: ClassVar[tuple[str, ...]] = ()

    _directions: HebbianDirections | None = PrivateAttr(default=None)
    _feature_means: dict[str, float] = PrivateAttr(default_factory=dict)

    @property
    def _pass_limit(self) -> int:
        """The cap on passes, which this model calls epochs."""
        return self.max_epochs

    @property
    def directions(self) -> HebbianDirections:
        """The directions this walk learned, with their variances.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._directions is not None
        return self._directions

    @property
    def epochs_run(self) -> int:
        """How many passes over the rows the walk took.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self._completed_passes

    @property
    def n_features_in(self) -> int:
        """How many features the fit saw.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.directions.n_features

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Learn ``n_components`` directions by walking the rows one at a time.

        The plumbing around :meth:`_updated_weights`: validate the features,
        learn and apply the centring, walk, then measure the variance along
        each direction the walk arrived at.

        Centring is not optional and this model does it, for the reason
        ``PrincipalComponentAnalysis`` gives. The update is proportional to the
        row, so on uncentred data the weights are pulled towards wherever the
        cloud sits rather than towards how it is shaped, and with means of 10
        and 100 that is most of what they would learn. The means are stored,
        because ``transform`` has to repeat this preparation on rows the fit
        never saw.

        Neither ``_walk`` nor this method sets ``_fitted`` before the
        directions are built and stored, which is the commit-nothing-until-
        everything-succeeded ordering every other fit here follows.

        Parameters
        ----------
        input_values:
            The features to decompose. At least two rows, since a single row
            has no spread to find a direction in, and at least one of them must
            actually vary.

        Returns
        -------
        Self

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonEqualArrayLengthError
            If the features are different lengths.
        NonUniqueFeaturesError
            If two features share a name.
        TooFewValuesError
            If there are fewer than two rows.
        InvalidValuesError
            If ``n_components`` exceeds the number of features supplied.
        AllSameValuesError
            If every feature is constant, since there is then no spread in any
            direction and every share would divide by zero.
        DivergenceError
            If the weights overflow to non-finite values, which means the rate
            was too large for the data's scale.
        """
        feature_set = FeatureSet(input_values)

        if feature_set.n_samples < 2:
            raise TooFewValuesError(
                f"a decomposition needs at least two rows to have any spread; "
                f"got {feature_set.n_samples}"
            )

        if self.n_components > feature_set.n_features:
            raise InvalidValuesError(
                f"cannot learn {self.n_components} directions from "
                f"{feature_set.n_features} features"
            )

        centred = self._prepared_for_fitting(feature_set)
        total_variance = self._total_variance_of(centred)

        if total_variance <= 0.0:
            raise AllSameValuesError(
                "every feature is constant, so there is no direction to find"
            )

        generator = np.random.default_rng(self.random_seed)
        weights = self._walk(centred, generator)

        self._directions = self._built_directions(centred, weights, total_variance)
        self._mark_fitted()

        return self

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Project rows onto the learned directions.

        One output feature per direction, named ``component_1`` upward, in the
        order the walk produced them. The names are the ones
        ``PrincipalComponentAnalysis`` uses on purpose, so that a caller who
        swaps one model for the other does not have to rename a column
        downstream. That the two are interchangeable at all is the module's
        claim.

        Projected onto the *unit* directions rather than the raw weight
        vectors, so a coordinate is a distance along the direction. Projecting
        onto an unnormalised vector would scale every coordinate by how far
        that component happened to get, which is a fact about the walk leaking
        into the answer.

        Centred with the means learned during ``fit``, never with these rows'
        own means, which is the leak the fit/transform split exists to prevent.

        Parameters
        ----------
        input_values:
            Rows over exactly the features the fit saw, in any order.

        Returns
        -------
        list[Feature]
            One feature per direction, each holding one coordinate per input
            row.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        centred = self._prepared_for_transforming(input_values)
        coordinates = centred.values @ self.directions.directions.T

        return [
            Feature(direction.name, coordinates[:, position])
            for position, direction in enumerate(self.directions)
        ]

    def _updated_weights(
        self, weights: FloatArray, centred_row: FloatArray, rate: float
    ) -> FloatArray:
        """Apply the Hebbian update for one row and hand back the new weights.

        The concept, and the only part of this class that is the learning rule
        rather than the bookkeeping. Everything else here centres data, loops,
        counts passes and names columns; this is what makes the model what it
        is.

        Parameters
        ----------
        weights:
            The current weights, shape ``(n_components, n_features)``. Row
            ``i`` is component ``i``'s vector. Do not modify it in place -- the
            epoch loop holds the previous epoch's copy to measure movement
            against, and mutating this would make that measurement zero.
        centred_row:
            One observation, shape ``(n_features,)``, already mean-subtracted.
            The rule is defined on centred data and nothing downstream centres
            it again.
        rate:
            How far this row is allowed to move the weights, already read off
            the schedule for the current epoch.

        Returns
        -------
        FloatArray
            The updated weights, a new array of shape
            ``(n_components, n_features)``.

        Notes
        -----
        Start with the outputs, which are what this row makes each component
        say. For component ``i``,

            output[i] = weights[i] . centred_row

        so all of them together are ``weights @ centred_row``, a vector of
        length ``n_components``.

        **Plain Hebbian learning** would then be

            weights[i] += rate * output[i] * centred_row

        and it diverges, for the reason the module docstring measures. Do not
        write that.

        **Oja's rule** subtracts what the component has already accounted for:

            weights[i] += rate * output[i] * (centred_row - output[i] * weights[i])

        The subtracted vector is this component's own reconstruction of the
        row, so the term in brackets is the part of the row the component does
        not yet explain. That is what drives the length to 1 rather than to
        infinity.

        **Sanger's rule**, which is what to implement, extends the subtraction
        to every earlier component as well:

            explained = sum over j from 0 to i inclusive of output[j] * weights[j]

            weights[i] += rate * output[i] * (centred_row - explained)

        Three things about that sum, in order of how easy they are to get
        wrong.

        The upper limit is ``i`` **inclusive**, so the loop runs
        ``range(component + 1)`` and not ``range(component)``. Dropping the
        ``j == i`` term removes Oja's normalisation from every component, which
        turns component 0 into plain Hebb and diverges; the module docstring
        has the measurement.

        ``output`` and ``weights`` are the values from the **start** of this
        row's update, not values partly updated as the loop walks the
        components. Computing all the outputs once before the loop and
        building the new weights into a separate array is the simplest way to
        keep that true.

        The sum runs over *earlier and current* components only, never over
        later ones. That asymmetry is the whole of the deflation. Component 0
        sees the raw row and finds the direction of greatest variance;
        component 1 sees the row minus what component 0 explains, so the
        largest direction is already gone and it finds the next one; and so on
        down. Summing over every component instead would leave each one seeing
        the same residual and give them no reason to differ.

        A plain Python loop over the components is a perfectly good answer, and
        ``n_components`` is small.
        """
        raise NotImplementedError

    def _walk(self, centred: RowBlock, generator: np.random.Generator) -> FloatArray:
        """Present the rows repeatedly until the weights stop moving.

        The loop around :meth:`_updated_weights`, and the place the walk is
        recorded. It calls ``_record_walk`` on its way out rather than leaving
        that to ``fit``, which is safe because ``converged`` and ``epochs_run``
        are both guarded by ``_check_fitted`` and ``fit`` marks the model
        fitted last of all.

        The rows are reshuffled every epoch. A rule applied one row at a time is
        genuinely sensitive to the order they arrive in, and a fixed order lets
        the last row of each pass have the final say every time.

        What ``converged`` means here is the weaker of the two readings
        :class:`~oop_ml.core.base.convergent_fit.ConvergentFit` describes. The
        rate decays, so a pass late in the walk moves the weights less than an
        early one would even if nothing had been learned in between. A settled
        walk therefore means the weights stopped moving, and whether they
        stopped because they arrived or because the rate ran out is a question
        :attr:`HebbianDirections.lengths` answers and this flag does not.
        """
        weights = self._initial_weights(centred.n_features, generator)
        epochs_run = 0
        converged = False

        while epochs_run < self.max_epochs:
            epochs_run += 1
            rate = self.learning_rate.value_at(epochs_run, self.max_epochs)
            before = weights.copy()

            for position in generator.permutation(centred.n_rows):
                weights = self._updated_weights(weights, centred.row(position), rate)

            self._check_is_finite(weights, epochs_run)

            if self._has_converged(weights - before):
                converged = True
                break

        self._record_walk(epochs_run, converged)

        return weights

    def _check_is_finite(self, weights: FloatArray, epochs_run: int) -> None:
        """Raise if the walk overflowed, naming the cause rather than the symptom.

        A rate too large for the data's scale makes every update bigger than
        the last, and numpy overflows to ``inf`` and then ``nan`` without
        raising anything. Left alone that is a fit which completes and answers
        ``nan`` to every question three calls later.

        Raises
        ------
        DivergenceError
            If any weight is not finite.
        """
        if not np.all(np.isfinite(weights)):
            raise DivergenceError(
                f"the weights overflowed on epoch {epochs_run}; the learning "
                f"rate is too large for this data's scale, so lower it or "
                f"standardize the features first"
            )

    def _initial_weights(
        self, n_features: int, generator: np.random.Generator
    ) -> FloatArray:
        """Random unit vectors, one per component, to start the walk from.

        Random rather than fixed, because identical starting rows would leave
        every component with the same outputs on the first row and the
        deflation with nothing to separate. Scaled to unit length so that the
        starting point is already on the sphere the rule converges to, which
        keeps the first few updates the same size as the last few.
        """
        weights = generator.normal(0.0, 1.0, (self.n_components, n_features))

        return weights / np.linalg.norm(weights, axis=1, keepdims=True)

    def _built_directions(
        self, centred: RowBlock, weights: FloatArray, total_variance: float
    ) -> HebbianDirections:
        """Pair each learned vector with its feature names and its variance.

        The variance is measured along the *unit* direction, by projecting the
        centred training rows onto it and taking the sample variance of the
        result. That is the same quantity an eigenvalue reports, computed from
        the data rather than from a matrix, so the two are directly comparable.

        Measuring it along the raw vector instead would scale it by the square
        of that vector's length, which is a fact about how far the walk got
        rather than about the data.
        """
        unit_directions = weights / np.linalg.norm(weights, axis=1, keepdims=True)
        coordinates = centred.values @ unit_directions.T

        return HebbianDirections(
            [
                HebbianDirection(
                    self.name_for(position),
                    Coefficients(
                        [
                            Coefficient(name, float(weight))
                            for name, weight in zip(
                                centred.feature_names, weights[position], strict=True
                            )
                        ]
                    ),
                    float(np.var(coordinates[:, position], ddof=1)),
                )
                for position in range(weights.shape[0])
            ],
            total_variance,
        )

    @staticmethod
    def _total_variance_of(centred: RowBlock) -> float:
        """The variance of the whole dataset, summed one column at a time.

        The trace of the covariance matrix, reached without forming the
        covariance matrix, which is the point of the module. Every direction's
        share is measured against this.
        """
        return float(np.sum(np.var(centred.values, axis=0, ddof=1)))

    def _prepared_for_fitting(self, feature_set: FeatureSet) -> RowBlock:
        """Learn the column means and subtract them.

        The means are stored because ``transform`` has to repeat exactly this
        preparation on rows the fit never saw.
        """
        self._feature_means = {
            feature.name: float(np.mean(feature.values)) for feature in feature_set
        }

        return self._centred(feature_set)

    def _centred(self, feature_set: FeatureSet) -> RowBlock:
        """Subtract the learned mean from every column, in the fitted order.

        Row-major, via ``rows_of``, because this model walks rows one at a time
        and never forms ``X.T @ v``. That is the opposite of what a linear
        model wants and it is the layout question the row block exists to
        settle in one place.
        """
        names = tuple(self._feature_means)
        ordered = FeatureSet.matching(names, list(feature_set))

        return rows_of(
            np.column_stack(
                [
                    ordered.column(name).values - self._feature_means[name]
                    for name in names
                ]
            ),
            names,
        )

    def _prepared_for_transforming(self, input_values: Sequence[Feature]) -> RowBlock:
        """Repeat the fit's centring on new rows, learning nothing new."""
        self._check_fitted()
        self._check_features_match(input_values)

        return self._centred(FeatureSet(list(input_values)))

    def _check_features_match(self, input_values: Sequence[Feature]) -> None:
        """Raise unless the supplied features are exactly the fitted ones.

        Exactly, in both directions. A missing column makes a direction's blend
        unevaluable, and an extra one has no weight to be weighted by. Order is
        free, because everything downstream reorders by name.
        """
        supplied = {feature.name for feature in input_values}
        fitted = set(self._feature_means)

        if supplied != fitted:
            raise InvalidValuesError(
                f"expected exactly the fitted features "
                f"{sorted(fitted)}; got {sorted(supplied)}"
            )

    def component_names(self) -> tuple[str, ...]:
        """What ``transform`` will call its output features.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        return tuple(direction.name for direction in self.directions)

    @staticmethod
    def name_for(position: int) -> str:
        """The name of the direction at ``position``, counting from zero.

        The same names ``PrincipalComponentAnalysis`` gives, drawn from the same
        constant so the two cannot drift apart, because a caller swapping one
        model for the other should not have to rename anything.
        """
        return f"{COMPONENT_NAME_PREFIX}_{position + 1}"

    def __repr__(self) -> str:
        if not self.is_fitted:
            return (
                f"HebbianPrincipalComponents(n_components={self.n_components}, "
                f"unfitted)"
            )

        return (
            f"HebbianPrincipalComponents(n_components={self.n_components}, "
            f"epochs_run={self.epochs_run}, converged={self.converged})"
        )
