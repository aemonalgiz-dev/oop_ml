"""Learn what the data looks like, using a rule that reads only two numbers.

What this teaches that nothing else here does
----------------------------------------------
Every other layered model in this library learns by backpropagation. A loss is
computed at the far end of the network, and the blame for it is carried back
through every layer until it reaches a weight, so the change to that weight is a
function of quantities the weight itself never sees. That is what makes a deep
network trainable and it is also what makes it a single indivisible object.

A restricted Boltzmann machine learns by a **local** rule instead. The change to
the weight joining visible unit ``i`` and hidden unit ``j`` is built from the
values *at its two ends* and from nothing else. No signal travels back from a
distant objective, because there is no distant objective. The rule is

    change to weight(i, j)  =  rate * ( <v_i h_j>_data  -  <v_i h_j>_model )

and both terms are correlations between the two units the weight actually
connects. That is the whole concept this family exists to demonstrate, and it is
worth seeing it work before accepting that backpropagation is the only way a
layered model can be trained.

The energy, and what "restricted" restricts
--------------------------------------------
Visible units hold the data and hidden units hold whatever the model invents to
explain it. Both are binary. A joint configuration is scored by an energy, low
energy meaning plausible:

    energy(visible, hidden)
        = -(visible_bias . visible)
          - (hidden_bias . hidden)
          - visible . weights . hidden

and the probability of a configuration is proportional to ``exp(-energy)``.

The restriction is that there are no within-layer connections. No visible unit
is joined to another visible unit and no hidden unit to another hidden unit, so
the energy contains no ``v_i v_k`` or ``h_j h_l`` term at all. **That is why the
units of one layer are conditionally independent given the other layer**, and it
is the most important sentence in this file. Fix the visible layer and every
hidden unit's remaining energy contribution depends only on its own state, so
the conditional distribution factorises and the whole layer can be sampled in a
single step:

    probability(hidden = 1 | visible) = logistic(hidden_bias + visible . weights)
    probability(visible = 1 | hidden) = logistic(visible_bias + hidden . weights.T)

Without the restriction those conditionals would each require their own
intractable inner sampling loop. With it, alternating between the two lines
above is an exact block Gibbs sampler over the joint distribution, and one line
of matrix arithmetic is a whole layer resampled.

Contrastive divergence, and the honest description of it
---------------------------------------------------------
The gradient of the log likelihood of the data is the difference of two
correlations, one measured with the visible units clamped to the data and one
measured under the model's own distribution. The first is cheap. The second
requires running the Gibbs chain to equilibrium, and how long that takes is not
knowable in advance, which makes the true gradient intractable rather than
merely expensive.

Contrastive divergence gives up on equilibrium. It starts the chain **at the
data** instead of at random, runs ``n_gibbs_steps`` alternations, and uses
whatever it has arrived at as the negative statistic:

    weights      += rate * (visible_0.T @ hidden_0 - visible_k.T @ hidden_k) / n_rows
    visible_bias += rate * mean over rows of (visible_0 - visible_k)
    hidden_bias  += rate * mean over rows of (hidden_0 - hidden_k)

**This is not the gradient of the likelihood, and it is not an estimate of it
that improves with more data.** It is a biased approximation whose bias is a
property of the truncation, so collecting a million more rows shrinks the
sampling noise and leaves the bias exactly where it was. One Gibbs step is the
usual choice and it works, which is a statement about what the resulting
representation is good for rather than a claim that the truncation was harmless.

That has a direct consequence for :attr:`~oop_ml.core.base.convergent_fit.\\
ConvergentFit.converged`, and the base class already warns that each model owes
its reader this sentence. Here ``converged`` means **the weights stopped
moving**. It does not mean a maximum of the likelihood was reached, because
nothing in the walk was ever climbing the likelihood. On real data with a
constant rate it is normally ``False``, since the sampling noise keeps every
weight jittering below any tolerance worth setting, and that is the expected
outcome rather than a failed fit. Decay the rate towards zero and it becomes
``True`` for the trivial reason that the steps became too small to see.

Sample the chain, but use probabilities for the statistics
-----------------------------------------------------------
The asymmetry is easy to get wrong in either direction, so both halves are
stated.

The chain **must sample**. Feeding mean-field probabilities back into the next
conditional is not a Markov chain over binary states, it is a deterministic map
with a fixed point, and the negative statistic it produces is a fact about that
map rather than about the model's distribution.

The statistics **should not sample**. ``hidden_0`` is the expected value of the
hidden units given the data, and using the expectation rather than one draw from
it removes variance from the update without changing what it estimates. Measured
on the fixture in the spec, over 400 repeats of a single step, the mean per-entry
standard deviation of the weight change is 0.0229 from the probabilities and
0.0395 from one draw, so sampling carries 1.72 times the noise at the start of a
fit. Once the probabilities have saturated the gap narrows to 1.10 times, 0.0133
against 0.0147, which is the same fact seen from the other end. A Bernoulli
variable's variance goes to zero as its parameter goes to either bound.

What that noise costs is smaller than the rule of thumb implies, and recording
the measurement is better than repeating the rule. Over ten seeds on this
fixture both choices reach a reconstruction error of 0.0408 by 1000 epochs and
are indistinguishable there, and at 300 epochs, still mid-training, the
**sampled** version is ahead, 0.0480 against 0.0595, because the extra noise
shortens the flat start that small initial weights create. So the argument for
probabilities is that the update estimates the same quantity with less variance,
which is a claim about the estimator. It is not a claim that a fixture this
small would notice, and it was not one until the two were run against each
other.

What the model is actually judged by, and what it is not
---------------------------------------------------------
**Reconstruction error is not the objective.** It is the mean squared difference
between a row and the visible probabilities you get by pushing it once through
the hidden layer and back, and nothing in the update rule is descending it. It
is reported here because it is the only cheap diagnostic available, and because
a fit whose reconstruction error does not fall has plainly learned nothing.
Measured on the spec's fixture, it falls from 0.2500 after one epoch to 0.0487
after 300 and settles at 0.0408 by 1000. It is a smoke alarm and not a
scoreboard, and a model can lower it by learning the identity through a wide
hidden layer while modelling nothing at all.

The shape of that fall is worth seeing, because it is not the shape a gradient
descent produces. At 10 epochs the error is 0.2498 and at 50 it is 0.2494,
barely moved from where it started, and then it collapses. The weights begin as
noise with standard deviation 0.01, so every hidden unit starts near 0.5 for
every row and the correlations at both ends of the chain very nearly cancel. The
first useful epochs are spent breaking that symmetry, and a run cut short at 50
epochs would report a model that had learned nothing while being three quarters
of the way to learning everything.

The quantity that *is* comparable between configurations is the free energy,

    free_energy(visible)
        = -(visible_bias . visible)
          - sum over j of softplus(hidden_bias_j + (visible . weights)_j)

which is what remains after summing ``exp(-energy)`` over every hidden
configuration and taking the negative logarithm. Rows the model finds plausible
have lower free energy than rows it does not. It is not a probability, because
the normalising constant over all ``2^n_visible_units`` visible configurations
is never computed, and differences of free energies are therefore meaningful
while a single value on its own is not.

Decisions worth naming
-----------------------
The visible values are required to lie in ``[0, 1]``. The energy above describes
binary units, and a value strictly between the two is read as a mean rather than
a state, which is how greyscale intensities are used in practice and is exactly
what the positive statistic needs. Anything outside the interval is refused,
because there is no reading of the energy under which it means anything.

One update per epoch over all the rows, rather than mini-batches. Mini-batching
is the standard practical choice and is a straightforward optimisation of this,
not a correction to it, and leaving it out keeps the epoch loop readable.

This walk is unusually hard to blow up, and that is a fact about the rule rather
than about the defaults. Every quantity in the update is a mean of values in
``[0, 1]``, so a single step is bounded by the rate however badly the rate was
chosen, and once the logistic saturates the two ends of the chain start to agree
and the accumulated change stops growing. Measured at a constant rate of 1e307
over 200 epochs, the largest weight settles at 4.9e306, about half the rate, and
at 1.79e308, the largest rate a float can hold, it reaches 8.8e307 and still
does not overflow. So :class:`~oop_ml.core.exceptions.DivergenceError` is
reachable here only through a hand-built set of parameters, unlike gradient
descent, where the step is proportional to an error that the step itself makes
larger. The guard stays because the type owns the invariant, not because the
epoch loop is expected to trip it.

Weights start as small normal noise with standard deviation 0.01 and both biases
start at zero. Symmetry has to be broken or every hidden unit computes the same
function forever, and the scale has to be small or the logistic saturates before
the first update. Initialising the visible bias to the log odds of each column,
which is Hinton's recommendation and does speed the first few epochs, is
declined here because an all-zero or all-one column makes those log odds
infinite and the guard for it costs more than the epochs it saves.

Where the pieces would live if there were two of these
-------------------------------------------------------
:class:`BoltzmannParameters`, :class:`GibbsState` and
:class:`ContrastiveDivergenceUpdate` are value objects rather than models, so by
this library's layout rule they belong under ``core``. They sit here because
this is the only model that has them so far, and moving them is what a second
energy-based model should do rather than something to do in advance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.convergent_fit import ConvergentFit
from oop_ml.core.base.estimator import Transformer
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import ClassScores
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.exceptions import (
    DivergenceError,
    InvalidValuesError,
    ShapeMismatchError,
)
from oop_ml.core.logistic import stable_logistic
from oop_ml.core.schedule import ConstantSchedule, Schedule
from oop_ml.core.types import FloatArray

HIDDEN_UNIT_NAME_PREFIX = "hidden"
"""How hidden units are named: ``hidden_1``, ``hidden_2``, and so on.

One-indexed, matching the components and the clusters, and a name rather than a
bare position so that a transformed table reads as columns rather than as
numbers whose meaning has to be remembered from somewhere else.
"""

INITIAL_WEIGHT_SPREAD = 0.01
"""Standard deviation of the normal noise the weights start from.

Small enough that no logistic starts saturated, and non-zero because identical
weights leave every hidden unit computing the identical function for the whole
of training.
"""


class BoltzmannParameters:
    """The weights and the two bias vectors, plus the conditionals they define.

    Immutable, so that a fit computes a new one per epoch rather than mutating
    the model's state part way through a step that might still fail. That is the
    same commit-nothing-until-everything-succeeded pattern the serving audit
    established for every other fit here, expressed as a type rather than as a
    convention.

    Parameters
    ----------
    weights:
        ``(n_visible_units, n_hidden_units)``. Entry ``[i, j]`` joins visible
        unit ``i`` to hidden unit ``j``, and there is no other connection in the
        model.
    visible_bias:
        ``(n_visible_units,)``.
    hidden_bias:
        ``(n_hidden_units,)``.

    Raises
    ------
    InvalidValuesError
        If the weights are not two-dimensional or either bias is not
        one-dimensional.
    ShapeMismatchError
        If a bias does not have one entry per unit of its own layer.
    DivergenceError
        If any value is not finite. Refusing it here is what stops a diverged
        walk from producing a fitted model that answers ``nan`` to every
        question. Measured, the epoch loop does not reach this at any rate a
        float can hold, for the reason the module docstring gives, so in
        practice it guards a hand-built set of parameters. The invariant belongs
        to the type either way.
    """

    __slots__ = ("_hidden_bias", "_visible_bias", "_weights")

    def __init__(
        self, weights: FloatArray, visible_bias: FloatArray, hidden_bias: FloatArray
    ) -> None:
        if weights.ndim != 2:
            raise InvalidValuesError(
                f"a weight matrix is two-dimensional, got {weights.ndim}"
            )
        if visible_bias.ndim != 1:
            raise InvalidValuesError(
                f"a visible bias is one-dimensional, got {visible_bias.ndim}"
            )
        if hidden_bias.ndim != 1:
            raise InvalidValuesError(
                f"a hidden bias is one-dimensional, got {hidden_bias.ndim}"
            )
        if visible_bias.size != weights.shape[0]:
            raise ShapeMismatchError(
                f"{visible_bias.size} visible biases for "
                f"{weights.shape[0]} visible units"
            )
        if hidden_bias.size != weights.shape[1]:
            raise ShapeMismatchError(
                f"{hidden_bias.size} hidden biases for {weights.shape[1]} hidden units"
            )

        for name, values in (
            ("weights", weights),
            ("visible bias", visible_bias),
            ("hidden bias", hidden_bias),
        ):
            if not np.all(np.isfinite(values)):
                raise DivergenceError(
                    f"the {name} left the finite numbers; lower the learning rate"
                )

        # Frozen copies, so that the caller keeps their own arrays and a fitted
        # model cannot be edited through a reference handed back out.
        self._weights = self._frozen(weights)
        self._visible_bias = self._frozen(visible_bias)
        self._hidden_bias = self._frozen(hidden_bias)

    @staticmethod
    def _frozen(values: FloatArray) -> FloatArray:
        copied = np.array(values, dtype=np.float64)
        copied.setflags(write=False)

        return copied

    @property
    def weights(self) -> FloatArray:
        """``(n_visible_units, n_hidden_units)``, read-only."""
        return self._weights

    @property
    def visible_bias(self) -> FloatArray:
        """``(n_visible_units,)``, read-only."""
        return self._visible_bias

    @property
    def hidden_bias(self) -> FloatArray:
        """``(n_hidden_units,)``, read-only."""
        return self._hidden_bias

    @property
    def n_visible_units(self) -> int:
        """How many units hold the data."""
        return int(self._weights.shape[0])

    @property
    def n_hidden_units(self) -> int:
        """How many units hold whatever the model invented to explain it."""
        return int(self._weights.shape[1])

    def hidden_given(self, visible: FloatArray) -> FloatArray:
        """``probability(hidden = 1 | visible)``, a whole layer at once.

        Parameters
        ----------
        visible:
            ``(n_rows, n_visible_units)``.

        Returns
        -------
        FloatArray
            ``(n_rows, n_hidden_units)``, every entry in ``[0, 1]``.
        """
        return stable_logistic(self._hidden_bias + visible @ self._weights)

    def visible_given(self, hidden: FloatArray) -> FloatArray:
        """``probability(visible = 1 | hidden)``, a whole layer at once.

        Parameters
        ----------
        hidden:
            ``(n_rows, n_hidden_units)``.

        Returns
        -------
        FloatArray
            ``(n_rows, n_visible_units)``, every entry in ``[0, 1]``.
        """
        return stable_logistic(self._visible_bias + hidden @ self._weights.T)

    def free_energy_of(self, visible: FloatArray) -> FloatArray:
        """The free energy of each visible row, one number per row.

        The hidden layer summed out exactly rather than sampled. Because the
        hidden units are conditionally independent, the sum of ``exp(-energy)``
        over all ``2^n_hidden_units`` hidden configurations factorises into one
        two-term sum per unit, and the negative logarithm of the product is the
        softplus expression below.

        ``numpy.logaddexp(0, z)`` is the softplus, written that way because the
        literal ``log(1 + exp(z))`` overflows for large ``z`` where the answer is
        simply ``z``.

        Parameters
        ----------
        visible:
            ``(n_rows, n_visible_units)``.

        Returns
        -------
        FloatArray
            ``(n_rows,)``. Lower means the model finds the row more plausible.
            Only differences carry meaning, since the normalising constant over
            all visible configurations is never computed.
        """
        hidden_input = self._hidden_bias + visible @ self._weights

        return -(visible @ self._visible_bias) - np.sum(
            np.logaddexp(0.0, hidden_input), axis=1
        )

    def shifted_by(self, update: ContrastiveDivergenceUpdate) -> BoltzmannParameters:
        """A new set of parameters with one contrastive divergence step applied.

        Raises
        ------
        ShapeMismatchError
            If the update was built for a differently shaped model.
        DivergenceError
            If the step took any value out of the finite numbers.
        """
        return BoltzmannParameters(
            self._weights + update.weight_change,
            self._visible_bias + update.visible_bias_change,
            self._hidden_bias + update.hidden_bias_change,
        )

    def __repr__(self) -> str:
        return (
            f"BoltzmannParameters({self.n_visible_units} visible, "
            f"{self.n_hidden_units} hidden)"
        )


class GibbsState:
    """One end of a Gibbs chain, as the two blocks a statistic is built from.

    The positive end pairs the data with the hidden probabilities the data
    implies; the negative end pairs the chain's sampled visible states with the
    hidden probabilities *those* imply. Both ends have the same shape and feed
    the same three correlations, which is why one type serves both and why the
    update rule reads as a subtraction of two of these rather than as six
    separate arrays.

    Parameters
    ----------
    visible:
        ``(n_rows, n_visible_units)``.
    hidden:
        ``(n_rows, n_hidden_units)``.

    Raises
    ------
    InvalidValuesError
        If either block is not two-dimensional.
    ShapeMismatchError
        If the two blocks describe different numbers of rows.
    """

    __slots__ = ("_hidden", "_visible")

    def __init__(self, visible: FloatArray, hidden: FloatArray) -> None:
        if visible.ndim != 2:
            raise InvalidValuesError(
                f"a visible block is two-dimensional, got {visible.ndim}"
            )
        if hidden.ndim != 2:
            raise InvalidValuesError(
                f"a hidden block is two-dimensional, got {hidden.ndim}"
            )
        if visible.shape[0] != hidden.shape[0]:
            raise ShapeMismatchError(
                f"{visible.shape[0]} visible rows against {hidden.shape[0]} hidden rows"
            )

        self._visible = visible
        self._hidden = hidden

    @property
    def visible(self) -> FloatArray:
        """``(n_rows, n_visible_units)``."""
        return self._visible

    @property
    def hidden(self) -> FloatArray:
        """``(n_rows, n_hidden_units)``."""
        return self._hidden

    @property
    def n_rows(self) -> int:
        """How many rows this end of the chain describes."""
        return int(self._visible.shape[0])

    @property
    def correlations(self) -> FloatArray:
        """``<v_i h_j>`` averaged over the rows, shape ``(n_v, n_h)``.

        The statistic a weight's update is built from, and the reason the rule
        is called local. Entry ``[i, j]`` reads visible unit ``i`` and hidden
        unit ``j`` and nothing else in the model.
        """
        return self._visible.T @ self._hidden / self.n_rows

    @property
    def visible_means(self) -> FloatArray:
        """``<v_i>`` averaged over the rows, shape ``(n_visible_units,)``."""
        return np.mean(self._visible, axis=0)

    @property
    def hidden_means(self) -> FloatArray:
        """``<h_j>`` averaged over the rows, shape ``(n_hidden_units,)``."""
        return np.mean(self._hidden, axis=0)

    def __repr__(self) -> str:
        return (
            f"GibbsState({self.n_rows} rows, {self._visible.shape[1]} visible, "
            f"{self._hidden.shape[1]} hidden)"
        )


class ContrastiveDivergenceUpdate:
    """The three changes one contrastive divergence step makes.

    A class rather than a tuple, for the reason the charter gives: a caller
    handed ``(a, b, c)`` has to know the positional order, and two of the three
    here are bias vectors that differ only in length. On a model whose layers
    happen to be the same width, swapping them type-checks, runs, and trains
    something else. This constructor refuses the swap whenever the widths differ
    and names the two lengths when it does.

    Parameters
    ----------
    weight_change:
        ``(n_visible_units, n_hidden_units)``.
    visible_bias_change:
        ``(n_visible_units,)``.
    hidden_bias_change:
        ``(n_hidden_units,)``.

    Raises
    ------
    InvalidValuesError
        If the weight change is not two-dimensional, or either bias change is
        not one-dimensional.
    ShapeMismatchError
        If a bias change does not have one entry per unit of its own layer.
    """

    __slots__ = ("_hidden_bias_change", "_visible_bias_change", "_weight_change")

    def __init__(
        self,
        weight_change: FloatArray,
        visible_bias_change: FloatArray,
        hidden_bias_change: FloatArray,
    ) -> None:
        if weight_change.ndim != 2:
            raise InvalidValuesError(
                f"a weight change is two-dimensional, got {weight_change.ndim}"
            )
        if visible_bias_change.ndim != 1:
            raise InvalidValuesError(
                f"a visible bias change is one-dimensional, "
                f"got {visible_bias_change.ndim}"
            )
        if hidden_bias_change.ndim != 1:
            raise InvalidValuesError(
                f"a hidden bias change is one-dimensional, "
                f"got {hidden_bias_change.ndim}"
            )
        if visible_bias_change.size != weight_change.shape[0]:
            raise ShapeMismatchError(
                f"{visible_bias_change.size} visible bias changes for "
                f"{weight_change.shape[0]} visible units"
            )
        if hidden_bias_change.size != weight_change.shape[1]:
            raise ShapeMismatchError(
                f"{hidden_bias_change.size} hidden bias changes for "
                f"{weight_change.shape[1]} hidden units"
            )

        self._weight_change = weight_change
        self._visible_bias_change = visible_bias_change
        self._hidden_bias_change = hidden_bias_change

    @property
    def weight_change(self) -> FloatArray:
        """``(n_visible_units, n_hidden_units)``, to be added to the weights."""
        return self._weight_change

    @property
    def visible_bias_change(self) -> FloatArray:
        """``(n_visible_units,)``, to be added to the visible bias."""
        return self._visible_bias_change

    @property
    def hidden_bias_change(self) -> FloatArray:
        """``(n_hidden_units,)``, to be added to the hidden bias."""
        return self._hidden_bias_change

    @property
    def largest_movement(self) -> float:
        """The biggest single change this step makes, in the weights' own units.

        What the walk's convergence is measured on, for the reason
        :meth:`~oop_ml.core.base.convergent_fit.ConvergentFit._has_converged`
        gives. Note what a small value licences here and what it does not. It
        says the parameters stopped moving; it says nothing about a maximum of
        the likelihood, because contrastive divergence was never climbing one.
        """
        return float(
            max(
                np.max(np.abs(self._weight_change), initial=0.0),
                np.max(np.abs(self._visible_bias_change), initial=0.0),
                np.max(np.abs(self._hidden_bias_change), initial=0.0),
            )
        )

    def __repr__(self) -> str:
        return f"ContrastiveDivergenceUpdate(largest={self.largest_movement:.6g})"


class RestrictedBoltzmannMachine(Transformer[Sequence[Feature]], ConvergentFit):
    """A generative model of binary rows, trained by a local learning rule.

    A :class:`~oop_ml.core.base.estimator.Transformer`, because it takes no
    target and answers by rewriting the columns. ``transform`` hands back one
    feature per hidden unit holding ``probability(hidden = 1 | row)``, which is
    the representation the model learned, and it is narrower than the input
    whenever there are fewer hidden units than features.

    It is also a
    :class:`~oop_ml.core.base.convergent_fit.ConvergentFit`, which supplies
    ``tolerance``, ``converged`` and the pass counter. Read the module
    docstring on what ``converged`` is entitled to mean here before trusting it.

    Parameters
    ----------
    n_hidden_units:
        How many hidden units to learn. This is the width of the learned
        representation, and it is given rather than learned.
    learning_rate:
        A :class:`~oop_ml.core.schedule.Schedule` rather than a number, so that
        a rate decaying across the run is expressible without a second field.
        The default is a constant 0.1. A decaying rate makes ``converged`` come
        out ``True`` for the trivial reason that the steps became small, which
        is worth knowing before reading anything into it.
    n_gibbs_steps:
        How many visible-hidden alternations to run before reading the negative
        statistic. One is the usual choice, and raising it reduces the bias of
        the approximation without ever removing it.
    max_epochs:
        The cap on full passes over the data. One update per epoch, over all the
        rows at once.
    random_seed:
        Fixes the initial weights and every sample drawn afterwards, so a fit
        and the sampling done through it are both reproducible.
    tolerance:
        Inherited. The walk is called settled once no single parameter moved
        further than this in a whole epoch.

    Raises
    ------
    pydantic.ValidationError
        If any count is below its minimum, or the tolerance is not positive.
        Field bounds are pydantic's to enforce, so the error is pydantic's too.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_hidden_units: int = Field(default=8, ge=1)
    learning_rate: Schedule = ConstantSchedule(value=0.1)
    n_gibbs_steps: int = Field(default=1, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    random_seed: int | None = None

    LEARNED_STATE: ClassVar[tuple[str, ...]] = (
        "_parameters",
        "_feature_names",
        "_passes_run",
        "_converged",
    )
    """The fitted self, and the format contract if this is ever registered.

    ``_generator`` is deliberately absent. It is runtime state, the way a tree's
    generator is, and a document that carried it would be promising to restore a
    position in a random stream that nothing about the model depends on.
    """

    _parameters: BoltzmannParameters | None = PrivateAttr(default=None)
    _feature_names: tuple[str, ...] = PrivateAttr(default=())
    _generator: np.random.Generator | None = PrivateAttr(default=None)

    @property
    def _pass_limit(self) -> int:
        """``max_epochs``, under the name this model gives it."""
        return self.max_epochs

    @property
    def parameters(self) -> BoltzmannParameters:
        """The weights and biases this fit settled on.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._parameters is not None
        return self._parameters

    @property
    def weights(self) -> FloatArray:
        """``(n_visible_units, n_hidden_units)``, read-only.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.parameters.weights

    @property
    def visible_bias(self) -> FloatArray:
        """``(n_visible_units,)``, read-only.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.parameters.visible_bias

    @property
    def hidden_bias(self) -> FloatArray:
        """``(n_hidden_units,)``, read-only.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.parameters.hidden_bias

    @property
    def n_visible_units(self) -> int:
        """How many features the fit saw, which is the visible layer's width.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.parameters.n_visible_units

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The visible units' names, in the order the fit saw them.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        return self._feature_names

    @property
    def epochs_run(self) -> int:
        """How many full passes over the data the fit took.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self._completed_passes

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Learn weights and biases from ``input_values`` by contrastive divergence.

        The epoch loop and the bookkeeping around
        :meth:`_contrastive_divergence_update`. Each epoch builds both ends of
        the Gibbs chain, asks for the changes they imply, applies them, and asks
        whether anything still moved.

        Nothing is assigned to the model until every epoch has succeeded, so a
        walk that diverges part way through leaves the previous fit intact
        rather than replacing it with a half-trained one.

        Parameters
        ----------
        input_values:
            The visible units, one feature per unit, every value in ``[0, 1]``.

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
        InvalidValuesError
            If any value falls outside ``[0, 1]``, where the energy function
            gives it no reading at all.
        DivergenceError
            If the weights or biases leave the finite numbers. Every statistic
            in the update is bounded, so this is far harder to provoke than it
            is for a gradient walk; see the module docstring for the measured
            rates that fail to reach it.
        """
        feature_set = FeatureSet(input_values)
        self._check_values_are_bounded(feature_set)

        feature_names = tuple(feature.name for feature in feature_set)
        rows = rows_of(
            np.column_stack([feature.values for feature in feature_set]), feature_names
        )

        generator = np.random.default_rng(self.random_seed)
        parameters = self._initial_parameters(rows.n_features, generator)

        epochs_run = 0
        converged = False

        for epoch in range(1, self.max_epochs + 1):
            rate = self.learning_rate.value_at(epoch, self.max_epochs)
            positive = GibbsState(rows.values, parameters.hidden_given(rows.values))
            negative = self._chain_end(parameters, positive.hidden, generator)
            update = self._contrastive_divergence_update(positive, negative, rate)

            parameters = parameters.shifted_by(update)
            epochs_run = epoch

            if self._has_converged(update.largest_movement):
                converged = True
                break

        self._parameters = parameters
        self._feature_names = feature_names
        self._generator = generator
        self._record_walk(epochs_run, converged)
        self._mark_fitted()

        return self

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Rewrite rows as the hidden layer's probabilities.

        One output feature per hidden unit, named ``hidden_1`` upward. This is
        the learned representation, and it is what a downstream model is
        normally fitted on.

        Deterministic, and deliberately so. ``transform`` is called on held-out
        data inside a pipeline and inside a fold, and a representation that
        differed between two calls on the same rows would make every score
        downstream a fact about the random stream. Use :meth:`sample_hidden`
        when a draw is what is wanted.

        Parameters
        ----------
        input_values:
            Rows over exactly the features the fit saw, in any order.

        Returns
        -------
        list[Feature]
            One feature per hidden unit, each holding one probability per input
            row.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or any
            value falls outside ``[0, 1]``.
        """
        probabilities = self.hidden_probabilities(input_values)

        return [
            Feature(self.name_for(position), probabilities.values[:, position])
            for position in range(probabilities.n_classes)
        ]

    def hidden_probabilities(self, input_values: Sequence[Feature]) -> ClassScores:
        """``probability(hidden = 1 | row)`` for every row and every hidden unit.

        Returns :class:`~oop_ml.core.data.probabilities.ClassScores` rather than
        :class:`~oop_ml.core.data.probabilities.Probabilities`, which is a
        ``(n_rows,)`` vector and therefore the wrong shape outright, and rather
        than :class:`~oop_ml.core.data.probabilities.ProbabilityMatrix`, whose
        extra guarantee would be a lie here. Hidden units are not alternatives
        competing for one row's allegiance. They are simultaneous, conditionally
        independent switches, and a row that turns on four of them has four
        probabilities near one which sum to four. Declining the row-sum claim is
        the same decision
        :class:`~oop_ml.numpy.classification.multiclass.one_vs_rest.OneVsRestClassifier`
        makes for the same reason.

        What ``ClassScores`` actually guarantees is two-dimensional, non-empty,
        and every entry bounded to ``[0, 1]``, which is precisely the contract a
        block of Bernoulli parameters wants. The name is the one wart, since it
        was written for a caller whose columns were classes, and the guarantees
        rather than the name are what a type is for.

        Parameters
        ----------
        input_values:
            Rows over exactly the features the fit saw, in any order.

        Returns
        -------
        ClassScores
            ``(n_rows, n_hidden_units)``.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or any
            value falls outside ``[0, 1]``.
        """
        rows = self._visible_rows(input_values)

        return ClassScores(self.parameters.hidden_given(rows.values))

    def visible_probabilities(self, hidden_values: Sequence[Feature]) -> ClassScores:
        """``probability(visible = 1 | hidden)`` for every row and visible unit.

        The other conditional, and the one that makes the model generative
        rather than merely a feature extractor. Hand it hidden states and it
        says what data those states would produce.

        Parameters
        ----------
        hidden_values:
            One feature per hidden unit, named ``hidden_1`` upward, every value
            in ``[0, 1]``.

        Returns
        -------
        ClassScores
            ``(n_rows, n_visible_units)``, in the fitted feature order.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly this model's hidden units,
            or any value falls outside ``[0, 1]``.
        """
        block = self._hidden_rows(hidden_values)

        return ClassScores(self.parameters.visible_given(block.values))

    def sample_hidden(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Draw one binary hidden state per unit per row.

        A single Bernoulli draw from :meth:`hidden_probabilities`, which is
        legitimate in one step precisely because the restriction makes the
        hidden units conditionally independent given the visible ones. Without
        it this would need its own inner sampling loop.

        Consumes randomness from the generator the fit left behind, so two
        consecutive calls differ and a fit with the same ``random_seed``
        reproduces the whole sequence.

        Parameters
        ----------
        input_values:
            Rows over exactly the features the fit saw, in any order.

        Returns
        -------
        list[Feature]
            One feature per hidden unit, every value exactly 0.0 or 1.0.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or any
            value falls outside ``[0, 1]``.
        """
        drawn = self._sampled(self.hidden_probabilities(input_values).values)

        return [
            Feature(self.name_for(position), drawn[:, position])
            for position in range(drawn.shape[1])
        ]

    def sample_visible(self, hidden_values: Sequence[Feature]) -> list[Feature]:
        """Draw one binary visible state per unit per row.

        The mirror of :meth:`sample_hidden`, and what generating a row from the
        model looks like once the hidden states are in hand.

        Parameters
        ----------
        hidden_values:
            One feature per hidden unit, named ``hidden_1`` upward. Binary
            states are what the conditional describes, although any value in
            ``[0, 1]`` is accepted and read as a mean.

        Returns
        -------
        list[Feature]
            One feature per fitted visible feature, in the fitted order, every
            value exactly 0.0 or 1.0.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly this model's hidden units,
            or any value falls outside ``[0, 1]``.
        """
        drawn = self._sampled(self.visible_probabilities(hidden_values).values)

        return [
            Feature(name, drawn[:, position])
            for position, name in enumerate(self.feature_names)
        ]

    def reconstruct(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Push rows through the hidden layer and back, and report what returns.

        Deterministic, because both halves use probabilities rather than draws.
        That makes the second half strictly speaking not a conditional
        probability, since it is fed hidden means rather than hidden states, and
        it is what every implementation uses for the diagnostic because a
        reconstruction that changed on every call would be unreadable as one.

        Parameters
        ----------
        input_values:
            Rows over exactly the features the fit saw, in any order.

        Returns
        -------
        list[Feature]
            One feature per fitted visible feature, in the fitted order, every
            value in ``[0, 1]``.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or any
            value falls outside ``[0, 1]``.
        """
        rows = self._visible_rows(input_values)
        hidden = self.parameters.hidden_given(rows.values)
        rebuilt = self.parameters.visible_given(hidden)

        return [
            Feature(name, rebuilt[:, position])
            for position, name in enumerate(self._feature_names)
        ]

    def reconstruction_error(self, input_values: Sequence[Feature]) -> float:
        """Mean squared difference between the rows and their reconstruction.

        The available diagnostic, and **not** the objective. Nothing in the
        update rule descends this, and a model can lower it by learning the
        identity through a wide hidden layer while modelling the data's
        structure not at all. What it is good for is the negative case. A fit
        whose reconstruction error does not fall has learned nothing, and that
        is worth being able to see in one number.

        Parameters
        ----------
        input_values:
            Rows over exactly the features the fit saw, in any order.

        Returns
        -------
        float
            Zero when every value is rebuilt exactly, and at most one.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or any
            value falls outside ``[0, 1]``.
        """
        rows = self._visible_rows(input_values)
        rebuilt = np.column_stack(
            [feature.values for feature in self.reconstruct(input_values)]
        )

        return float(np.mean((rows.values - rebuilt) ** 2))

    def free_energy(self, input_values: Sequence[Feature]) -> Predictions:
        """The free energy of each row, which is what compares configurations.

        The hidden layer summed out exactly rather than sampled, which the
        restriction makes possible in closed form. Lower means the model finds
        the row more plausible, and only *differences* mean anything, since the
        normalising constant over every visible configuration is never computed.

        Returned as :class:`~oop_ml.core.data.predictions.Predictions` because
        that is this library's type for one finite value per row in query order,
        which is exactly what this is. The alternative,
        :class:`~oop_ml.core.data.column.Column`, would have to borrow a
        :class:`~oop_ml.core.validation.ValueRole` that does not describe a free
        energy.

        Parameters
        ----------
        input_values:
            Rows over exactly the features the fit saw, in any order.

        Returns
        -------
        Predictions
            ``(n_rows,)`` free energies, in the order the rows were supplied.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or any
            value falls outside ``[0, 1]``.
        """
        rows = self._visible_rows(input_values)

        return Predictions.already_checked(self.parameters.free_energy_of(rows.values))

    def _contrastive_divergence_update(
        self,
        positive: GibbsState,
        negative: GibbsState,
        learning_rate: float,
    ) -> ContrastiveDivergenceUpdate:
        """The change one contrastive divergence step makes to every parameter.

        Parameters
        ----------
        positive:
            The data end of the chain. ``positive.visible`` is the data itself
            and ``positive.hidden`` is ``probability(hidden = 1 | data)``.
        negative:
            The far end of the chain, after ``n_gibbs_steps`` alternations.
            ``negative.visible`` holds sampled binary states and
            ``negative.hidden`` holds the probabilities those states imply.
        learning_rate:
            This epoch's rate, already read off the schedule. Non-negative.

        Returns
        -------
        ContrastiveDivergenceUpdate
            The three changes, to be *added* to the current parameters. Do not
            apply them here and do not set ``_fitted``; ``fit`` owns both.

        Raises
        ------
        ShapeMismatchError
            From the returned object's own constructor, if a bias change is
            built for the wrong layer. That is the swap this return type exists
            to catch.

        Notes
        -----
        Three quantities are wanted, and each is the same subtraction. Take a
        statistic measured at the data end, subtract the same statistic measured
        at the far end of the chain, and scale by the rate:

            weight_change        = rate * (positive.correlations
                                           - negative.correlations)
            visible_bias_change  = rate * (positive.visible_means
                                           - negative.visible_means)
            hidden_bias_change   = rate * (positive.hidden_means
                                           - negative.hidden_means)

        :class:`GibbsState` already owns all six statistics, so this method is
        the rule and none of the arithmetic. ``correlations`` is
        ``visible.T @ hidden / n_rows``, which is the ``(n_visible, n_hidden)``
        table of ``<v_i h_j>``; the two ``means`` are the per-unit averages down
        the rows. Every division by ``n_rows`` has already happened, so nothing
        here divides.

        Why it is a subtraction, and why that direction
        -----------------------------------------------
        The gradient of the log likelihood with respect to weight ``(i, j)`` is

            <v_i h_j>_data  -  <v_i h_j>_model

        The first term raises the probability of configurations the data
        actually shows. The second lowers the probability of configurations the
        model currently believes in, whether or not the data agrees, which is
        what stops the first term simply inflating every weight without bound.
        Contrastive divergence keeps that shape exactly and substitutes a chain
        truncated after ``n_gibbs_steps`` for the intractable model term. Adding
        rather than subtracting the negative statistic makes both terms push the
        same way, and the weights then grow until they overflow and
        :class:`BoltzmannParameters` refuses them.

        Note that the rate scales all three changes identically. It is a single
        number for the whole step, not one per layer.
        """
        return ContrastiveDivergenceUpdate(
            weight_change=learning_rate
            * (positive.correlations - negative.correlations),
            visible_bias_change=learning_rate
            * (positive.visible_means - negative.visible_means),
            hidden_bias_change=learning_rate
            * (positive.hidden_means - negative.hidden_means),
        )

    def _chain_end(
        self,
        parameters: BoltzmannParameters,
        hidden_probabilities: FloatArray,
        generator: np.random.Generator,
    ) -> GibbsState:
        """Run the Gibbs chain from the data and report where it arrived.

        The chain starts at ``hidden_probabilities``, which were computed from
        the data, and that is what makes this contrastive divergence rather than
        an honest sample from the model. Then ``n_gibbs_steps`` alternations,
        each one a whole layer resampled in a single step because the
        restriction makes the layer's units conditionally independent.

        **The chain samples.** Feeding probabilities forward instead would make
        this a deterministic mean-field map with a fixed point rather than a
        Markov chain over binary states, and the statistic it produced would
        describe the map. The final hidden block is the exception and is left as
        probabilities, because it is a statistic rather than a state and the
        expectation carries less noise than one draw from it.

        Parameters
        ----------
        parameters:
            The current weights and biases.
        hidden_probabilities:
            ``probability(hidden = 1 | data)``, the positive end's hidden block.
        generator:
            The fit's generator, advanced in place.

        Returns
        -------
        GibbsState
            Sampled visible states paired with the hidden probabilities they
            imply.
        """
        hidden_states = self._draw(hidden_probabilities, generator)
        visible_states = self._draw(parameters.visible_given(hidden_states), generator)
        arrived = parameters.hidden_given(visible_states)

        for _ in range(self.n_gibbs_steps - 1):
            hidden_states = self._draw(arrived, generator)
            visible_states = self._draw(
                parameters.visible_given(hidden_states), generator
            )
            arrived = parameters.hidden_given(visible_states)

        return GibbsState(visible_states, arrived)

    def _initial_parameters(
        self, n_visible_units: int, generator: np.random.Generator
    ) -> BoltzmannParameters:
        """Small normal weights and zero biases.

        Symmetry has to be broken, or every hidden unit computes the same
        function for the whole of training and the model has one unit however
        many were asked for. The scale has to stay small, or the logistic
        saturates and the first updates are multiplied by a derivative that has
        already gone to zero.
        """
        return BoltzmannParameters(
            generator.normal(
                0.0, INITIAL_WEIGHT_SPREAD, (n_visible_units, self.n_hidden_units)
            ),
            np.zeros(n_visible_units),
            np.zeros(self.n_hidden_units),
        )

    def _sampled(self, probabilities: FloatArray) -> FloatArray:
        """One Bernoulli draw per entry, using the model's own generator."""
        return self._draw(probabilities, self._sampling_generator())

    @staticmethod
    def _draw(probabilities: FloatArray, generator: np.random.Generator) -> FloatArray:
        """One Bernoulli draw per entry, as 0.0 or 1.0.

        Comparing a uniform draw against the probability rather than calling a
        binomial, because it is one array operation on the whole layer and the
        layer is the unit of work here.
        """
        return (generator.random(probabilities.shape) < probabilities).astype(
            np.float64
        )

    def _sampling_generator(self) -> np.random.Generator:
        """The generator the public sampling methods draw from.

        Created lazily, so that a model restored from a document, which
        deliberately does not carry a random stream position, still samples
        rather than raising.
        """
        if self._generator is None:
            self._generator = np.random.default_rng(self.random_seed)

        return self._generator

    def _visible_rows(self, input_values: Sequence[Feature]) -> RowBlock:
        """The supplied features as a checked row block, in the fitted order."""
        self._check_fitted()
        self._check_features_match(input_values)

        feature_set = FeatureSet.matching(self._feature_names, list(input_values))
        self._check_values_are_bounded(feature_set)

        return rows_of(
            np.column_stack(
                [feature_set.column(name).values for name in self._feature_names]
            ),
            self._feature_names,
        )

    def _hidden_rows(self, hidden_values: Sequence[Feature]) -> RowBlock:
        """The supplied hidden units as a checked row block, in unit order."""
        self._check_fitted()
        self._check_hidden_units_match(hidden_values)

        names = self.hidden_unit_names()
        feature_set = FeatureSet.matching(names, list(hidden_values))
        self._check_values_are_bounded(feature_set)

        return rows_of(
            np.column_stack([feature_set.column(name).values for name in names]), names
        )

    @staticmethod
    def _check_values_are_bounded(feature_set: FeatureSet) -> None:
        """Raise unless every value lies in ``[0, 1]``.

        The energy function describes binary units. A value in between is read
        as a mean, which is how greyscale intensities are used in practice and
        is exactly what the positive statistic wants. A value outside the
        interval has no reading at all, and admitting one would let a caller
        train on unscaled data and get a model that runs and means nothing.
        """
        for feature in feature_set:
            if not np.all((feature.values >= 0.0) & (feature.values <= 1.0)):
                raise InvalidValuesError(
                    f"{feature.name} must lie in [0, 1]; a Boltzmann unit is "
                    f"binary, and a value between is read as a mean"
                )

    def _check_features_match(self, input_values: Sequence[Feature]) -> None:
        """Raise unless the supplied features are exactly the fitted ones."""
        supplied = {feature.name for feature in input_values}
        fitted = set(self._feature_names)

        if supplied != fitted:
            raise InvalidValuesError(
                f"expected exactly the fitted features {sorted(fitted)}; "
                f"got {sorted(supplied)}"
            )

    def _check_hidden_units_match(self, hidden_values: Sequence[Feature]) -> None:
        """Raise unless the supplied features are exactly this model's units."""
        supplied = {feature.name for feature in hidden_values}
        expected = set(self.hidden_unit_names())

        if supplied != expected:
            raise InvalidValuesError(
                f"expected exactly this model's hidden units {sorted(expected)}; "
                f"got {sorted(supplied)}"
            )

    def hidden_unit_names(self) -> tuple[str, ...]:
        """What ``transform`` will call its output features.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        self._check_fitted()

        return tuple(self.name_for(position) for position in range(self.n_hidden_units))

    @staticmethod
    def name_for(position: int) -> str:
        """The name of the hidden unit at ``position``, counting from zero.

        One place deciding what a unit is called, so that ``transform`` and
        ``sample_visible`` cannot drift apart about it.
        """
        return f"{HIDDEN_UNIT_NAME_PREFIX}_{position + 1}"

    def __repr__(self) -> str:
        if not self.is_fitted:
            return (
                f"RestrictedBoltzmannMachine("
                f"n_hidden_units={self.n_hidden_units}, unfitted)"
            )

        return (
            f"RestrictedBoltzmannMachine("
            f"{self.n_visible_units}x{self.n_hidden_units}, "
            f"epochs_run={self.epochs_run}, converged={self.converged})"
        )
