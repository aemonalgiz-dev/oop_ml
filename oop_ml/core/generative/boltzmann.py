"""What a restricted Boltzmann machine holds, and what one step changes.

The fitted self of such a machine is a weight block joining the visible units
to the hidden ones, plus a bias for each side, and that triple is the same
whichever backend learned it. The update is modelled as its own object rather
than as three returned arrays, because three arrays in a tuple is a pairing
whose meaning depends on position and which the caller then has to remember
the order of.

Nothing here fits anything. Applying an update answers with a new set of
parameters rather than changing these, so a caller holding the parameters from
before a step still holds them afterwards, which is what lets a training loop
record its own history without copying anything by hand.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.exceptions import (
    DivergenceError,
    InvalidValuesError,
    ShapeMismatchError,
)
from oop_ml.core.logistic import stable_logistic
from oop_ml.core.types import FloatArray

HIDDEN_UNIT_NAME_PREFIX = "hidden"
"""How hidden units are named: ``hidden_1``, ``hidden_2``, and so on.

One-indexed, matching the components and the clusters, and a name rather than a
bare position so that a transformed table reads as columns rather than as
numbers whose meaning has to be remembered from somewhere else.
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
