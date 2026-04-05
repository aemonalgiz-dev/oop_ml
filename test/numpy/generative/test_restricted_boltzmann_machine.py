"""Spec for the restricted Boltzmann machine, red until the update rule lands.

What carries this file
-----------------------
The forward half of this model is a pair of logistic conditionals, and both are
checkable against a definition rather than against themselves. The energy
function is written out in the module docstring, the joint distribution over a
three-unit hidden layer has eight configurations, and eight terms can be summed
in a Python loop. So :func:`brute_force_free_energy` and
:func:`brute_force_hidden_probability` are written from ``exp(-energy)`` alone
and know nothing about softplus or about the logistic. They are the oracles, in
the sense ``test/core/distance/test_distance.py`` established, and they are what
turns "the conditionals factorise because there are no within-layer
connections" from a claim in prose into a number that either agrees to 1e-12 or
does not.

Those oracle tests run against hand-written parameters and need no fit at all,
which is deliberate. Everything that can be pinned without training is pinned
without training, so that the part of the file that goes red is exactly the part
that depends on the learning rule.

The learning half cannot be checked that way, and the reason is the concept
itself. Contrastive divergence is a biased approximation to a gradient nobody
can compute, so there is no reference implementation to agree with and no
finite-difference oracle to fall back on, because the quantity being descended
does not exist in closed form. What can be asserted is that the fit does
something, and this file asserts it in three independent ways rather than one.
Reconstruction error falls from 0.2500 after a single epoch to 0.0487 after
three hundred. The two generating patterns come back through the hidden layer on
the right side of 0.5 in every one of their six units, at 0.861 and 0.159 for
the first pattern's two halves. And the two patterns get free energies of -5.70
and -6.50 against -2.87 for a row that mixes them, which is the quantity that
actually compares configurations.

Three of them, because each is weak alone. Reconstruction error is not the
objective and a wide enough hidden layer lowers it by learning the identity.
Rebuilding the training patterns is memorisation as much as it is modelling.
Free energy is only comparable within one fit. Together they are hard to satisfy
by accident.

That the file discriminates, measured rather than assumed
----------------------------------------------------------
Five wrong update rules were written and the whole file run against each, which
is the measurement that matters, since a claim about coverage is only evidence
if some assertion is reading it. Failing tests out of 82, with a correct rule
passing clean: the subtraction the wrong way round 6, the two bias changes
swapped 30, the negative statistic dropped 7, the negative statistic added
rather than subtracted 7, and the rate ignored 3.

The rate case is the thin one, and it is thin for a reason worth writing down. A
rule that ignores the rate still trains, and trains well, because ignoring a
constant 0.1 is the same as choosing 1.0. Only the tests that vary the rate can
see it at all, and the first version of
``test_the_rate_scales_the_first_step_linearly`` could not, because with the
rate ignored all three of its fits come out identical and every proportionality
it asserts holds between zeros. The non-degeneracy line was added for exactly
that, and it took the count from 2 to 3.

What the shapes are chosen to catch
------------------------------------
Six visible units and three hidden ones, never equal and never a multiple. Two
of the three changes a contrastive divergence step makes are bias vectors, and
on a model whose layers are the same width, returning them the wrong way round
type-checks, runs, and trains a different model in silence. Here it is a
``ShapeMismatchError`` from ``ContrastiveDivergenceUpdate``'s own constructor,
and ``TestTheUpdateType`` pins that the constructor really does refuse it rather
than leaving the guard to a comment.

``FIRST_PATTERN`` and ``SECOND_PATTERN`` are complementary halves of a six-unit
row, which is the smallest fixture with structure a three-unit hidden layer
cannot express by copying. Five percent of the bits are flipped, so the data is
not separable by memorising two rows, and the flips come from a fixed seed so
the fixture is a constant rather than a draw.

The feature names are ``first`` through ``sixth`` rather than anything
alphabetical, because ``_feature_names`` carries the fitted column order and a
sort creeping in anywhere would reorder the visible layer while every shape
still conformed. PCA shipped exactly that bug once, through ``sort_keys=True``
in a saved document.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    DivergenceError,
    InvalidValuesError,
    NonUniqueFeaturesError,
    NotFittedError,
    ShapeMismatchError,
)
from oop_ml.core.schedule import (
    ConstantSchedule,
    ExponentialDecaySchedule,
    Schedule,
)
from oop_ml.core.types import FloatArray
from oop_ml.numpy.generative.restricted_boltzmann_machine import (
    BoltzmannParameters,
    ContrastiveDivergenceUpdate,
    GibbsState,
    RestrictedBoltzmannMachine,
)

FEATURE_NAMES = ("first", "second", "third", "fourth", "fifth", "sixth")
"""Six visible units, named so that alphabetical order is not fitted order."""

FIRST_PATTERN = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
SECOND_PATTERN = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
MIXED_ROW = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
"""A row belonging to neither pattern, for the free energy comparison."""

N_ROWS = 120
FLIP_CHANCE = 0.05
FIXTURE_SEED = 20260901

DEFAULT_RATE = ConstantSchedule(value=0.1)
"""The model's own default, restated so ``fitted`` can name every argument."""


def _generated_block() -> FloatArray:
    """The training rows, drawn once from a fixed seed so they are a constant."""
    generator = np.random.default_rng(FIXTURE_SEED)
    patterns = np.array([FIRST_PATTERN, SECOND_PATTERN], dtype=np.float64)
    chosen = patterns[generator.integers(0, 2, size=N_ROWS)]
    flipped = (generator.random(chosen.shape) < FLIP_CHANCE).astype(np.float64)

    return np.abs(chosen - flipped)


TWO_BLOCK_ROWS = _generated_block()

TWO_BLOCKS = [
    Feature(name, TWO_BLOCK_ROWS[:, position])
    for position, name in enumerate(FEATURE_NAMES)
]
"""120 rows over six binary units, each row one of two patterns with 5% noise."""


ON_RATE_NAMES = ("often", "nearly_never", "nearly_always", "seldom")
ON_RATES = (0.9, 0.0, 1.0, 0.1)
"""Four units switched on at four different rates, for what the visible bias is.

Deliberately neither sorted nor reverse-sorted by rate, and named so that
alphabetical order is a third arrangement again. The claim below is about the
*order* the fitted biases come out in, so a fixture whose rates already run
along the columns would be satisfied by a bias vector that had merely been
sorted, and one whose names run with the rates would be satisfied by a sort
creeping in through a name mapping.

Two of the rates are 0 and 1 exactly, which is a constant column. This model
never calls ``check_columns_vary``, and it is right not to: a unit that is
always on is data a generative model should be able to describe, and the whole
point of a visible bias is to be where that description lives.
"""

ON_RATE_SEED = 20260902


def _on_rate_block() -> FloatArray:
    """The four columns, drawn once from a fixed seed so they are a constant."""
    generator = np.random.default_rng(ON_RATE_SEED)

    return np.column_stack(
        [(generator.random(N_ROWS) < rate).astype(np.float64) for rate in ON_RATES]
    )


ON_RATE_ROWS = _on_rate_block()

ON_RATE_BLOCKS = [
    Feature(name, ON_RATE_ROWS[:, position])
    for position, name in enumerate(ON_RATE_NAMES)
]


def as_features(*rows: tuple[float, ...]) -> list[Feature]:
    """Hand-written visible rows as features, in the fixture's own order."""
    block = np.array(rows, dtype=np.float64)

    return [
        Feature(name, block[:, position]) for position, name in enumerate(FEATURE_NAMES)
    ]


def fitted(
    n_hidden_units: int = 3,
    max_epochs: int = 300,
    random_seed: int | None = 0,
    learning_rate: Schedule = DEFAULT_RATE,
    n_gibbs_steps: int = 1,
) -> RestrictedBoltzmannMachine:
    """A machine trained on the two-pattern fixture."""
    return RestrictedBoltzmannMachine(
        n_hidden_units=n_hidden_units,
        max_epochs=max_epochs,
        random_seed=random_seed,
        learning_rate=learning_rate,
        n_gibbs_steps=n_gibbs_steps,
    ).fit(TWO_BLOCKS)


def as_block(features: Sequence[Feature]) -> FloatArray:
    """A list of features back as an ``(n_rows, n_features)`` array."""
    return np.column_stack([feature.values for feature in features])


ORACLE_PARAMETERS = BoltzmannParameters(
    np.array([[0.5, -1.2, 0.3], [-0.8, 0.4, 1.1]]),
    np.array([0.2, -0.5]),
    np.array([0.1, 0.7, -0.3]),
)
"""Two visible units and three hidden ones, small enough to sum by brute force.

The numbers are arbitrary and asymmetric on purpose. A symmetric weight matrix
would let a transposed ``visible_given`` agree with the honest one.
"""

ORACLE_VISIBLE_ROWS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
ORACLE_HIDDEN_ROWS = ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0))


def energy_of(
    parameters: BoltzmannParameters, visible: FloatArray, hidden: FloatArray
) -> float:
    """The energy of one joint configuration, written from the definition.

    Deliberately a plain Python transcription of

        -(visible_bias . visible) - (hidden_bias . hidden)
        - visible . weights . hidden

    with nothing shared with the implementation. Every oracle below is built out
    of this one function.
    """
    interaction = 0.0
    for visible_unit in range(parameters.n_visible_units):
        for hidden_unit in range(parameters.n_hidden_units):
            interaction += (
                visible[visible_unit]
                * parameters.weights[visible_unit, hidden_unit]
                * hidden[hidden_unit]
            )

    visible_term = sum(
        parameters.visible_bias[unit] * visible[unit]
        for unit in range(parameters.n_visible_units)
    )
    hidden_term = sum(
        parameters.hidden_bias[unit] * hidden[unit]
        for unit in range(parameters.n_hidden_units)
    )

    return -float(visible_term) - float(hidden_term) - interaction


def hidden_configurations(parameters: BoltzmannParameters) -> list[FloatArray]:
    """Every binary hidden configuration, all ``2^n_hidden_units`` of them."""
    return [
        np.array(states, dtype=np.float64)
        for states in itertools.product((0.0, 1.0), repeat=parameters.n_hidden_units)
    ]


def visible_configurations(parameters: BoltzmannParameters) -> list[FloatArray]:
    """Every binary visible configuration."""
    return [
        np.array(states, dtype=np.float64)
        for states in itertools.product((0.0, 1.0), repeat=parameters.n_visible_units)
    ]


def brute_force_free_energy(
    parameters: BoltzmannParameters, visible: FloatArray
) -> float:
    """``-log sum over hidden of exp(-energy)``, summed one term at a time.

    The definition of free energy, and the oracle for the softplus expression
    the implementation uses instead. Nothing here knows that the sum factorises.
    """
    total = sum(
        math.exp(-energy_of(parameters, visible, hidden))
        for hidden in hidden_configurations(parameters)
    )

    return -math.log(total)


def brute_force_hidden_probability(
    parameters: BoltzmannParameters, visible: FloatArray, unit: int
) -> float:
    """``P(hidden[unit] = 1 | visible)`` by enumerating the joint distribution.

    The oracle for the conditional independence claim itself. If the units were
    *not* independent given the visible layer, this marginal would still be
    correct and the logistic form would not, so the agreement is the evidence
    for the claim rather than a restatement of it.
    """
    weights = [
        math.exp(-energy_of(parameters, visible, hidden))
        for hidden in hidden_configurations(parameters)
    ]
    switched_on = [
        weight
        for weight, hidden in zip(
            weights, hidden_configurations(parameters), strict=True
        )
        if hidden[unit] == 1.0
    ]

    return sum(switched_on) / sum(weights)


def brute_force_visible_probability(
    parameters: BoltzmannParameters, hidden: FloatArray, unit: int
) -> float:
    """``P(visible[unit] = 1 | hidden)``, the same enumeration the other way."""
    weights = [
        math.exp(-energy_of(parameters, visible, hidden))
        for visible in visible_configurations(parameters)
    ]
    switched_on = [
        weight
        for weight, visible in zip(
            weights, visible_configurations(parameters), strict=True
        )
        if visible[unit] == 1.0
    ]

    return sum(switched_on) / sum(weights)


class TestTheConditionals:
    """The forward half, against an enumeration of the joint distribution."""

    @pytest.mark.parametrize("visible_row", ORACLE_VISIBLE_ROWS)
    @pytest.mark.parametrize("unit", (0, 1, 2))
    def test_hidden_conditional_matches_the_enumerated_marginal(
        self, visible_row: tuple[float, ...], unit: int
    ) -> None:
        """This is the conditional independence claim, measured.

        The implementation computes a whole layer with one logistic. The oracle
        sums eight ``exp(-energy)`` terms and marginalises. They agree only
        because the energy contains no hidden-to-hidden term.
        """
        visible = np.array([visible_row], dtype=np.float64)

        computed = ORACLE_PARAMETERS.hidden_given(visible)[0, unit]
        expected = brute_force_hidden_probability(
            ORACLE_PARAMETERS, np.array(visible_row), unit
        )

        assert computed == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("hidden_row", ORACLE_HIDDEN_ROWS)
    @pytest.mark.parametrize("unit", (0, 1))
    def test_visible_conditional_matches_the_enumerated_marginal(
        self, hidden_row: tuple[float, ...], unit: int
    ) -> None:
        """The other direction, which is where a missing transpose hides."""
        hidden = np.array([hidden_row], dtype=np.float64)

        computed = ORACLE_PARAMETERS.visible_given(hidden)[0, unit]
        expected = brute_force_visible_probability(
            ORACLE_PARAMETERS, np.array(hidden_row), unit
        )

        assert computed == pytest.approx(expected, abs=1e-12)

    def test_a_saturated_unit_is_certain(self) -> None:
        """The one conditional whose answer needs no oracle at all."""
        saturated = BoltzmannParameters(np.zeros((2, 3)), np.full(2, 40.0), np.zeros(3))

        assert np.allclose(saturated.visible_given(np.zeros((1, 3))), 1.0)

    def test_a_whole_block_is_conditioned_in_one_call(self) -> None:
        """Four rows at once must equal four rows one at a time."""
        block = np.array(ORACLE_VISIBLE_ROWS, dtype=np.float64)

        together = ORACLE_PARAMETERS.hidden_given(block)
        apart = np.vstack(
            [ORACLE_PARAMETERS.hidden_given(row[None, :]) for row in block]
        )

        assert np.allclose(together, apart, atol=1e-15)


class TestFreeEnergy:
    """The quantity that compares configurations, against its own definition."""

    @pytest.mark.parametrize("visible_row", ORACLE_VISIBLE_ROWS)
    def test_matches_the_enumerated_partition_function(
        self, visible_row: tuple[float, ...]
    ) -> None:
        """Softplus per unit against eight summed exponentials."""
        computed = ORACLE_PARAMETERS.free_energy_of(
            np.array([visible_row], dtype=np.float64)
        )[0]
        expected = brute_force_free_energy(ORACLE_PARAMETERS, np.array(visible_row))

        assert computed == pytest.approx(expected, abs=1e-12)

    def test_a_large_hidden_input_does_not_overflow(self) -> None:
        """``log(1 + exp(z))`` overflows where the answer is simply ``z``."""
        huge = BoltzmannParameters(
            np.array([[900.0]]), np.array([0.0]), np.array([0.0])
        )

        energy = huge.free_energy_of(np.array([[1.0]]))[0]

        assert np.isfinite(energy)
        assert energy == pytest.approx(-900.0, abs=1e-9)

    def test_the_fit_scores_its_patterns_below_a_mixed_row(self) -> None:
        """A row belonging to neither pattern should be the implausible one."""
        model = fitted()

        energies = np.asarray(
            model.free_energy(as_features(FIRST_PATTERN, SECOND_PATTERN, MIXED_ROW))
        )

        assert energies[0] < energies[2]
        assert energies[1] < energies[2]

    def test_free_energy_is_one_finite_value_per_row(self) -> None:
        model = fitted()

        energies = model.free_energy(TWO_BLOCKS)

        assert len(energies) == N_ROWS
        assert np.all(np.isfinite(np.asarray(energies)))


class TestWhatItLearns:
    """That the update rule does something, asserted three independent ways."""

    def test_reconstruction_error_falls_substantially(self) -> None:
        """Measured on this fixture: 0.2500 after one epoch, 0.0487 after 300.

        The factor asserted is three, which sits under the measured factor of
        5.1 and still fails outright for a rule that moves the weights in the
        wrong direction or not at all. The margin is deliberately not tighter,
        because the fall is late rather than steady. At 10 epochs the error is
        0.2498 and at 50 it is still 0.2494, and only then does it collapse, so
        an assertion pinned near the measured value would be pinned to where
        one particular seed happened to be on the cliff.
        """
        early = fitted(max_epochs=1)
        late = fitted(max_epochs=300)

        assert (
            late.reconstruction_error(TWO_BLOCKS)
            < early.reconstruction_error(TWO_BLOCKS) / 3.0
        )

    def test_it_rebuilds_both_generating_patterns(self) -> None:
        """Every unit of both patterns comes back on the right side of 0.5."""
        model = fitted()
        patterns = as_features(FIRST_PATTERN, SECOND_PATTERN)

        rebuilt = as_block(model.reconstruct(patterns))

        assert np.array_equal(
            np.round(rebuilt), np.array([FIRST_PATTERN, SECOND_PATTERN])
        )

    def test_the_hidden_layer_tells_the_two_patterns_apart(self) -> None:
        """A representation that maps both patterns to one code learned nothing."""
        model = fitted()

        codes = np.asarray(
            model.hidden_probabilities(as_features(FIRST_PATTERN, SECOND_PATTERN))
        )

        assert np.max(np.abs(codes[0] - codes[1])) > 0.5

    def test_more_gibbs_steps_still_learns_the_patterns(self) -> None:
        """The truncation length is a knob, not a precondition."""
        model = fitted(n_gibbs_steps=3)

        rebuilt = as_block(
            model.reconstruct(as_features(FIRST_PATTERN, SECOND_PATTERN))
        )

        assert np.array_equal(
            np.round(rebuilt), np.array([FIRST_PATTERN, SECOND_PATTERN])
        )

    def test_the_visible_bias_learns_how_often_each_unit_is_on(self) -> None:
        """What the visible bias is *for*, asserted without redoing its sum.

        The claim comes from the rule's meaning rather than from its
        arithmetic. The positive term of the visible bias change is the data's
        on-rate for that unit and the negative term is the model's, so a unit
        the data switches on more often is pushed up harder for as long as the
        model under-produces it. The fitted biases therefore have to come out
        in the same order as the on-rates, and no term of the update is
        recomputed here to say so.

        This is the only assertion in the file that reads the bias half of the
        update at all. Measured: a rule that fits the weights correctly and
        leaves both biases at zero passes every other test here, and fails this
        one, because a bias vector of four zeros has no order.
        """
        model = RestrictedBoltzmannMachine(
            n_hidden_units=3, max_epochs=200, random_seed=0
        ).fit(ON_RATE_BLOCKS)

        by_rate = np.argsort(ON_RATES)

        assert list(np.argsort(model.visible_bias)) == list(by_rate)
        assert np.all(np.diff(np.asarray(model.visible_bias)[by_rate]) > 0.0)

    def test_reconstruction_error_is_a_bounded_mean_square(self) -> None:
        error = fitted().reconstruction_error(TWO_BLOCKS)

        assert isinstance(error, float)
        assert 0.0 <= error <= 1.0


class TestTransform:
    """The learned representation, as the transformer contract sees it."""

    def test_it_produces_one_feature_per_hidden_unit(self) -> None:
        transformed = fitted().transform(TWO_BLOCKS)

        assert [feature.name for feature in transformed] == [
            "hidden_1",
            "hidden_2",
            "hidden_3",
        ]

    def test_every_value_is_a_probability(self) -> None:
        block = as_block(fitted().transform(TWO_BLOCKS))

        assert np.all(block >= 0.0)
        assert np.all(block <= 1.0)

    def test_it_agrees_with_the_hidden_probabilities(self) -> None:
        """Two routes to one number, which is the habit this library keeps."""
        model = fitted()

        assert np.array_equal(
            as_block(model.transform(TWO_BLOCKS)),
            np.asarray(model.hidden_probabilities(TWO_BLOCKS)),
        )

    def test_it_is_deterministic(self) -> None:
        """A representation that varied per call would poison every fold."""
        model = fitted()

        assert np.array_equal(
            as_block(model.transform(TWO_BLOCKS)),
            as_block(model.transform(TWO_BLOCKS)),
        )

    def test_column_order_does_not_matter(self) -> None:
        model = fitted()

        assert np.allclose(
            as_block(model.transform(TWO_BLOCKS)),
            as_block(model.transform(list(reversed(TWO_BLOCKS)))),
        )

    def test_fit_transform_matches_fitting_then_transforming(self) -> None:
        together = RestrictedBoltzmannMachine(
            n_hidden_units=3, max_epochs=300, random_seed=0
        ).fit_transform(TWO_BLOCKS)
        apart = fitted().transform(TWO_BLOCKS)

        assert np.array_equal(as_block(together), as_block(apart))


class TestSampling:
    """Drawing states, which is what the chain does and what generating means."""

    def test_sampled_hidden_states_are_binary(self) -> None:
        block = as_block(fitted().sample_hidden(TWO_BLOCKS))

        assert np.array_equal(block, np.round(block))
        assert set(np.unique(block)) <= {0.0, 1.0}

    def test_sampled_visible_states_are_binary(self) -> None:
        model = fitted()
        hidden = model.sample_hidden(TWO_BLOCKS)

        block = as_block(model.sample_visible(hidden))

        assert set(np.unique(block)) <= {0.0, 1.0}

    def test_sampling_visible_returns_the_fitted_features(self) -> None:
        model = fitted()
        hidden = model.sample_hidden(TWO_BLOCKS)

        names = [feature.name for feature in model.sample_visible(hidden)]

        assert tuple(names) == FEATURE_NAMES

    def test_two_draws_differ(self) -> None:
        """Sampling consumes the stream, so it is not a cached answer."""
        model = fitted()

        first = as_block(model.sample_hidden(TWO_BLOCKS))
        second = as_block(model.sample_hidden(TWO_BLOCKS))

        assert not np.array_equal(first, second)

    def test_the_same_seed_reproduces_the_whole_run(self) -> None:
        first = fitted().sample_hidden(TWO_BLOCKS)
        second = fitted().sample_hidden(TWO_BLOCKS)

        assert np.array_equal(as_block(first), as_block(second))

    def test_a_different_seed_gives_a_different_fit(self) -> None:
        assert not np.array_equal(
            fitted(random_seed=0).weights, fitted(random_seed=1).weights
        )


class TestConvergence:
    """What settled means here, which is weaker than it means elsewhere."""

    def test_a_normal_fit_runs_out_of_epochs_rather_than_settling(self) -> None:
        """The expected outcome, and the reason the docstring labours the point.

        Contrastive divergence never stops moving, because the negative
        statistic is resampled every epoch. ``converged`` being ``False`` here
        is not a failed fit.
        """
        model = fitted(max_epochs=50)

        assert model.epochs_run == 50
        assert model.converged is False

    def test_a_zero_rate_settles_on_the_first_epoch(self) -> None:
        """Nothing moved, so the walk is settled, and that is all it means."""
        model = fitted(learning_rate=ConstantSchedule(value=0.0))

        assert model.converged is True
        assert model.epochs_run == 1

    def test_a_zero_rate_leaves_the_initial_weights_untouched(self) -> None:
        """The other half of the same claim, since converged alone is cheap."""
        model = fitted(learning_rate=ConstantSchedule(value=0.0))

        assert np.array_equal(model.visible_bias, np.zeros(6))
        assert np.array_equal(model.hidden_bias, np.zeros(3))

    def test_the_rate_scales_the_first_step_linearly(self) -> None:
        """One epoch is one step, and one step is proportional to the rate.

        The three fits share a seed, so they start from identical weights and
        their chains draw identical states, because every draw happens before
        any change is applied. The only thing that differs is the multiplier,
        and it multiplies all three changes rather than one of them. A rule
        that dropped the rate, or applied it to the weights alone, is off by a
        factor here while still training something that looks reasonable.

        The three non-degeneracy assertions are load-bearing rather than
        decorative, and there is one per part for the same reason there are
        three proportionalities. A rule that ignores the rate entirely makes
        all three fits identical, at which point every difference below is zero
        and every proportionality holds vacuously. A rule that moves the
        weights correctly and leaves a bias where it started is the same
        vacuity confined to one part, and it is otherwise invisible to this
        whole file: measured, a rule updating the weights correctly and neither
        bias at all passed every other test here.
        """
        unmoved = fitted(max_epochs=1, learning_rate=ConstantSchedule(value=0.0))
        small = fitted(max_epochs=1, learning_rate=ConstantSchedule(value=0.05))
        large = fitted(max_epochs=1, learning_rate=ConstantSchedule(value=0.1))

        assert np.max(np.abs(large.weights - unmoved.weights)) > 0.0
        assert np.max(np.abs(large.visible_bias - unmoved.visible_bias)) > 0.0
        assert np.max(np.abs(large.hidden_bias - unmoved.hidden_bias)) > 0.0
        assert np.allclose(
            large.weights - unmoved.weights,
            2.0 * (small.weights - unmoved.weights),
        )
        assert np.allclose(
            large.visible_bias - unmoved.visible_bias,
            2.0 * (small.visible_bias - unmoved.visible_bias),
        )
        assert np.allclose(
            large.hidden_bias - unmoved.hidden_bias,
            2.0 * (small.hidden_bias - unmoved.hidden_bias),
        )

    def test_a_decaying_rate_is_accepted_and_used(self) -> None:
        """A schedule rather than a number is the whole reason for the field."""
        model = fitted(
            max_epochs=100,
            learning_rate=ExponentialDecaySchedule(start=0.2, end=0.001),
        )

        assert model.epochs_run == 100
        assert np.all(np.isfinite(model.weights))

    def test_an_absurd_rate_still_cannot_diverge(self) -> None:
        """A bounded statistic makes this walk very hard to blow up.

        The first version of this test asserted a ``DivergenceError`` at a rate
        of 1e307 and did not get one, which is how the property below was
        found. Every quantity in the update is a mean of values in ``[0, 1]``,
        so one step is bounded by the rate however wrong the rate is, and once
        the logistic saturates the two ends of the chain agree and the
        accumulation stops growing. Measured at 1e307 over 200 epochs the
        largest weight settles at 4.9e306, and at 1.79e308, the largest rate a
        float can hold, at 8.8e307. Gradient descent has no such protection,
        which is what ``DivergenceError`` was written for.
        """
        model = fitted(max_epochs=50, learning_rate=ConstantSchedule(value=1e307))

        assert np.all(np.isfinite(model.weights))
        assert np.max(np.abs(model.weights)) < 1e307


class TestTheParameterType:
    """``BoltzmannParameters``, which needs no fit to be pinned."""

    def test_it_refuses_a_bias_of_the_wrong_length(self) -> None:
        with pytest.raises(ShapeMismatchError):
            BoltzmannParameters(np.zeros((6, 3)), np.zeros(5), np.zeros(3))

    def test_it_refuses_a_hidden_bias_of_the_wrong_length(self) -> None:
        with pytest.raises(ShapeMismatchError):
            BoltzmannParameters(np.zeros((6, 3)), np.zeros(6), np.zeros(2))

    def test_it_refuses_a_one_dimensional_weight_matrix(self) -> None:
        with pytest.raises(InvalidValuesError):
            BoltzmannParameters(np.zeros(6), np.zeros(6), np.zeros(3))

    def test_it_refuses_a_non_finite_weight(self) -> None:
        """Where a diverged walk is stopped, before it becomes a fitted model."""
        weights = np.zeros((6, 3))
        weights[0, 0] = np.inf

        with pytest.raises(DivergenceError):
            BoltzmannParameters(weights, np.zeros(6), np.zeros(3))

    def test_shifting_by_an_infinite_update_raises(self) -> None:
        start = BoltzmannParameters(np.zeros((6, 3)), np.zeros(6), np.zeros(3))
        update = ContrastiveDivergenceUpdate(
            np.full((6, 3), np.inf), np.zeros(6), np.zeros(3)
        )

        with pytest.raises(DivergenceError):
            start.shifted_by(update)

    def test_shifting_adds_the_changes(self) -> None:
        start = BoltzmannParameters(np.ones((6, 3)), np.zeros(6), np.full(3, 2.0))
        update = ContrastiveDivergenceUpdate(
            np.full((6, 3), 0.5), np.full(6, -1.0), np.full(3, 0.25)
        )

        shifted = start.shifted_by(update)

        assert np.allclose(shifted.weights, 1.5)
        assert np.allclose(shifted.visible_bias, -1.0)
        assert np.allclose(shifted.hidden_bias, 2.25)

    def test_the_stored_arrays_are_frozen(self) -> None:
        parameters = BoltzmannParameters(np.zeros((6, 3)), np.zeros(6), np.zeros(3))

        with pytest.raises(ValueError):
            parameters.weights[0, 0] = 1.0

    def test_it_copies_what_it_was_given(self) -> None:
        """A caller keeping their own array must not be able to edit the model."""
        weights = np.zeros((6, 3))
        parameters = BoltzmannParameters(weights, np.zeros(6), np.zeros(3))

        weights[0, 0] = 99.0

        assert parameters.weights[0, 0] == 0.0


class TestTheUpdateType:
    """``ContrastiveDivergenceUpdate``, and the swap it exists to refuse."""

    def test_it_refuses_the_two_bias_changes_the_wrong_way_round(self) -> None:
        """Six visible units and three hidden ones, so the swap is a length."""
        with pytest.raises(ShapeMismatchError):
            ContrastiveDivergenceUpdate(np.zeros((6, 3)), np.zeros(3), np.zeros(6))

    def test_it_refuses_a_two_dimensional_bias_change(self) -> None:
        with pytest.raises(InvalidValuesError):
            ContrastiveDivergenceUpdate(np.zeros((6, 3)), np.zeros((6, 1)), np.zeros(3))

    def test_the_largest_movement_reads_every_part(self) -> None:
        """A convergence measure that ignored the biases would stop too early."""
        update = ContrastiveDivergenceUpdate(
            np.full((6, 3), 0.01), np.full(6, -0.4), np.full(3, 0.02)
        )

        assert update.largest_movement == pytest.approx(0.4)


class TestTheGibbsStateType:
    """``GibbsState``, whose statistics the update rule is written in terms of."""

    def test_correlations_are_the_averaged_outer_products(self) -> None:
        """Computed by hand: two rows, so each entry is a mean of two products."""
        state = GibbsState(
            np.array([[1.0, 0.0], [1.0, 1.0]]),
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        )

        assert np.allclose(
            state.correlations,
            np.array([[0.5, 0.5, 1.0], [0.0, 0.5, 0.5]]),
        )

    def test_the_means_are_per_unit_averages(self) -> None:
        state = GibbsState(
            np.array([[1.0, 0.0], [1.0, 1.0]]),
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        )

        assert np.allclose(state.visible_means, [1.0, 0.5])
        assert np.allclose(state.hidden_means, [0.5, 0.5, 1.0])

    def test_it_refuses_two_blocks_of_different_heights(self) -> None:
        with pytest.raises(ShapeMismatchError):
            GibbsState(np.zeros((4, 2)), np.zeros((3, 3)))

    def test_it_refuses_a_one_dimensional_block(self) -> None:
        with pytest.raises(InvalidValuesError):
            GibbsState(np.zeros(4), np.zeros((4, 3)))


class TestEncapsulation:
    """What a caller can reach, and what they cannot write through."""

    def test_the_learned_weights_are_read_only(self) -> None:
        model = fitted()

        with pytest.raises(ValueError):
            model.weights[0, 0] = 1.0

    def test_the_learned_biases_are_read_only(self) -> None:
        model = fitted()

        with pytest.raises(ValueError):
            model.visible_bias[0] = 1.0


class TestWhatItRefuses:
    """Guards, each raising from the ``MLLibError`` hierarchy."""

    def test_reading_the_weights_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = RestrictedBoltzmannMachine(n_hidden_units=3).weights

    def test_reading_converged_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = RestrictedBoltzmannMachine(n_hidden_units=3).converged

    def test_reading_the_epoch_count_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = RestrictedBoltzmannMachine(n_hidden_units=3).epochs_run

    def test_transforming_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            RestrictedBoltzmannMachine(n_hidden_units=3).transform(TWO_BLOCKS)

    def test_free_energy_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            RestrictedBoltzmannMachine(n_hidden_units=3).free_energy(TWO_BLOCKS)

    def test_duplicate_feature_names_are_rejected(self) -> None:
        with pytest.raises(NonUniqueFeaturesError):
            RestrictedBoltzmannMachine(n_hidden_units=2).fit(
                [Feature("same", [1.0, 0.0]), Feature("same", [0.0, 1.0])]
            )

    def test_a_value_above_one_is_rejected(self) -> None:
        """Unscaled data has no reading under the energy function."""
        with pytest.raises(InvalidValuesError):
            RestrictedBoltzmannMachine(n_hidden_units=2).fit(
                [Feature("first", [0.0, 4.0]), Feature("second", [1.0, 0.0])]
            )

    def test_a_negative_value_is_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            RestrictedBoltzmannMachine(n_hidden_units=2).fit(
                [Feature("first", [0.0, -1.0]), Feature("second", [1.0, 0.0])]
            )

    def test_a_fractional_value_is_accepted_and_read_as_a_mean(self) -> None:
        """Greyscale intensities are the case this admits on purpose."""
        model = RestrictedBoltzmannMachine(
            n_hidden_units=2, max_epochs=5, random_seed=0
        ).fit([Feature("first", [0.0, 0.3, 0.9]), Feature("second", [1.0, 0.5, 0.2])])

        assert model.epochs_run == 5

    def test_transforming_without_every_fitted_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().transform(TWO_BLOCKS[:3])

    def test_transforming_with_an_unknown_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().transform([*TWO_BLOCKS, Feature("extra", np.zeros(N_ROWS))])

    def test_sampling_visible_from_the_wrong_names_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().sample_visible([Feature("wrong", [1.0]), Feature("names", [0.0])])

    def test_zero_hidden_units_is_refused_at_construction(self) -> None:
        with pytest.raises(ValidationError):
            RestrictedBoltzmannMachine(n_hidden_units=0)

    def test_zero_gibbs_steps_is_refused_at_construction(self) -> None:
        with pytest.raises(ValidationError):
            RestrictedBoltzmannMachine(n_gibbs_steps=0)

    def test_a_misspelled_field_is_refused(self) -> None:
        """``extra="forbid"``, which is what stops a default running silently."""
        with pytest.raises(ValidationError):
            RestrictedBoltzmannMachine(hidden_units=3)  # pyright: ignore[reportCallIssue]
