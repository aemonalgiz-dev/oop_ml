"""Spec for the Hopfield network, red until the Hebbian storage rule lands.

Only one method is stubbed, ``_stored_weights``, and it is the whole of what
this model learns. Everything else here, the settling, the energy, the two
update rules, the observed route and every refusal, is already written, so a
failure in this file that is not a ``NotImplementedError`` is a real one.

The fixtures, and why they are the sizes they are
--------------------------------------------------
``THREE_PICTURES`` is three mutually orthogonal sixteen-unit patterns, which is
a load of 0.188. Orthogonality is not decoration. For patterns that are exactly
perpendicular the crosstalk between them vanishes, so each one is a fixed point
by arithmetic rather than by luck, and a spec asserting recall is asserting the
implementation rather than the fixture's good fortune.

The basin was measured rather than assumed, over twenty different unit
orderings each. Every one of the 3 x 16 one-flip probes and every one of the
3 x 120 two-flip probes returns its own pattern, under every ordering, which is
8160 recalls with no exceptions. At three flips 622 of the 1680 probes do not,
so the spec asserts repair at one and two flips and asserts the *limit* at
three, in the way ``test_k_means.py`` asserts what concentric rings do to
k-means. A spec that only showed what works would be advertising.

``ORTHOGONAL_FOUR`` is two four-unit patterns whose weight matrix was worked out
on paper before any of this ran, and it is the strongest oracle here because
nothing computed it. ``ANTI_CORRELATED`` is one two-unit pattern, and it exists
entirely to make the synchronous oscillation visible: presented with ``(+1,
+1)`` the synchronous rule alternates forever between ``(-1, -1)`` and ``(+1,
+1)`` while the asynchronous rule settles on the first pass.

The oracles are written from the definitions
---------------------------------------------
``weights_by_definition`` and ``energy_by_definition`` are plain Python double
loops transcribed from the formulas, following the rule this repository learned
the hard way when a test built ``X.T diag(w) X`` and the implementation became
character-identical to it. The four-unit matrix is checked against numbers
written down by hand as well, which no amount of shared reasoning can make
tautological.

What each part of the file is for
----------------------------------
``TestWhatItStores`` is the storage rule and nothing else. ``TestWhatItRecalls``
is the settling, and ``TestWhereRecallStops`` is the same thing asserting a
failure on purpose. ``TestEnergy`` pins the quantity the convergence proof is
about, including the exact ``-2 * |weighted_sum|`` step the proof claims.
``TestSynchronousUpdate`` is the trap, demonstrated rather than described.
``TestTheZeroDiagonal`` is the quiet bug: a self-connected network fits, reports
an ordinary load, and remembers nothing. ``TestTheNegationIsAlsoStored`` and
``TestSpuriousMinima`` are the two families of fixed point nobody asked for.
"""

import itertools
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    NotFittedError,
    TooFewValuesError,
)
from oop_ml.numpy.associative_memory.hopfield_network import (
    BipolarPattern,
    HebbianWeights,
    HopfieldNetwork,
    RecallStop,
    UpdateRule,
)

N_UNITS = 16

STRIPES = [1.0, -1.0] * 8
PAIRS = [1.0, 1.0, -1.0, -1.0] * 4
QUARTERS = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0] * 2

THREE_PICTURES = (STRIPES, PAIRS, QUARTERS)
PICTURE_NAMES = ("stripes", "pairs", "quarters")
PICTURE_UNITS = tuple(f"unit_{position + 1:02d}" for position in range(N_UNITS))

ORTHOGONAL_FOUR = ([1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0])
FOUR_UNITS = ("unit_1", "unit_2", "unit_3", "unit_4")
HAND_WORKED_WEIGHTS = [
    [0.0, 0.0, 0.0, -0.5],
    [0.0, 0.0, -0.5, 0.0],
    [0.0, -0.5, 0.0, 0.0],
    [-0.5, 0.0, 0.0, 0.0],
]
"""The four-unit matrix, computed on paper.

Two patterns over four units, so each entry is the average of two products
divided by four. Units 1 and 4 disagree in both patterns and units 2 and 3
disagree in both, giving ``-2 / 4``; every other off-diagonal pair agrees once
and disagrees once, giving zero; the diagonal is forced to zero whatever the
products say.
"""

ANTI_CORRELATED = ([1.0, -1.0],)
TWO_UNITS = ("left", "right")
OSCILLATING_PROBE = [1.0, 1.0]

SELF_WEIGHT = 3.0
"""How hard a broken diagonal has to push to pin every unit where it is.

Measured on ``THREE_PICTURES``: the largest absolute row sum away from the
diagonal is 1.3125, so a self weight of 3.0 outvotes every other connection a
unit has, whatever state the rest of the network is in.
"""


def features_of(patterns, unit_names=PICTURE_UNITS) -> list[Feature]:
    """The patterns as one feature per unit, which is how a fit is handed them.

    Rows are patterns and columns are units, so this is a transpose: unit
    ``k``'s feature holds the value that unit takes in each pattern in turn.
    """
    values = np.array(patterns, dtype=np.float64)

    return [
        Feature(name, values[:, position]) for position, name in enumerate(unit_names)
    ]


PICTURE_FEATURES = features_of(THREE_PICTURES)


def fitted(random_seed: int | None = 0, **overrides) -> HopfieldNetwork:
    """A network storing the three orthogonal pictures."""
    return HopfieldNetwork(random_seed=random_seed, **overrides).fit(PICTURE_FEATURES)


def oscillator(
    update_rule: UpdateRule, max_passes: int = 10, random_seed: int = 0
) -> HopfieldNetwork:
    """The two-unit network that oscillates under one rule and not the other."""
    return HopfieldNetwork(
        update_rule=update_rule, max_passes=max_passes, random_seed=random_seed
    ).fit(features_of(ANTI_CORRELATED, TWO_UNITS))


def with_flips(pattern, *positions) -> list[float]:
    """A copy of ``pattern`` with the named units reversed."""
    corrupted = list(pattern)

    for position in positions:
        corrupted[position] = -corrupted[position]

    return corrupted


def weights_by_definition(patterns) -> list[list[float]]:
    """The Hebbian matrix, transcribed from the formula in plain Python.

    An oracle has to be written from the definition rather than derived from the
    implementation, so this is a double loop over pairs of units summing
    ``pattern[i] * pattern[j]`` and dividing by the unit count, with the
    diagonal set to zero because the rule says so.
    """
    n_units = len(patterns[0])
    matrix = [[0.0] * n_units for _ in range(n_units)]

    for first in range(n_units):
        for second in range(n_units):
            if first == second:
                continue

            agreement = sum(pattern[first] * pattern[second] for pattern in patterns)
            matrix[first][second] = agreement / n_units

    return matrix


def energy_by_definition(matrix, state) -> float:
    """``-0.5 * state . weights . state``, as a double loop over every pair."""
    total = 0.0

    for first, first_value in enumerate(state):
        for second, second_value in enumerate(state):
            total += matrix[first][second] * first_value * second_value

    return -0.5 * total


class TestWhatItStores:
    """The Hebbian rule, and nothing about recall."""

    def test_the_four_unit_matrix_is_the_one_worked_out_by_hand(self) -> None:
        """The strongest oracle here, because a person computed it."""
        network = HopfieldNetwork().fit(features_of(ORTHOGONAL_FOUR, FOUR_UNITS))

        assert np.asarray(network.weights) == pytest.approx(
            np.array(HAND_WORKED_WEIGHTS)
        )

    def test_every_weight_is_the_average_agreement_over_the_stored_patterns(
        self,
    ) -> None:
        """Checked pair by pair against the definition, not against a matmul."""
        network = fitted()
        expected = weights_by_definition(THREE_PICTURES)

        for first, first_name in enumerate(PICTURE_UNITS):
            for second, second_name in enumerate(PICTURE_UNITS):
                assert network.weights.weight_between(
                    first_name, second_name
                ) == pytest.approx(expected[first][second])

    def test_the_diagonal_is_zero(self) -> None:
        """The load-bearing half of the rule. See TestTheZeroDiagonal."""
        assert not fitted().weights.has_self_connections
        assert np.all(np.asarray(fitted().weights).diagonal() == 0.0)

    def test_the_matrix_is_symmetric(self) -> None:
        """``pattern[i] * pattern[j]`` and its reverse are the same number.

        Not decoration: the settling proof combines the row and the column
        contributions into one factor, and cannot if they differ.
        """
        matrix = np.asarray(fitted().weights)

        assert np.array_equal(matrix, matrix.T)

    def test_it_reports_the_load_it_was_given(self) -> None:
        network = fitted()

        assert network.n_units == N_UNITS
        assert network.n_stored_patterns == len(THREE_PICTURES)
        assert network.load == pytest.approx(len(THREE_PICTURES) / N_UNITS)

    def test_it_keeps_no_copy_of_the_patterns(self) -> None:
        """The memory is the weights, and the format contract says so.

        A Hopfield network genuinely cannot hand back what it was shown except
        by settling, and an attribute holding the originals would let a caller
        read something the model does not have.
        """
        assert HopfieldNetwork.LEARNED_STATE == ("_weights", "_n_stored_patterns")

    def test_the_units_keep_their_names(self) -> None:
        assert fitted().unit_names == PICTURE_UNITS
        assert fitted().weights.unit_names == PICTURE_UNITS


@pytest.mark.parametrize("pattern", THREE_PICTURES, ids=PICTURE_NAMES)
class TestWhatItRecalls:
    """Settling, on a fixture whose patterns are exactly perpendicular."""

    def test_a_stored_pattern_is_a_fixed_point(self, pattern) -> None:
        """Stronger than settling once: no unit would move, in any order."""
        assert fitted().is_fixed_point(BipolarPattern(pattern))

    def test_a_stored_pattern_recalls_itself(self, pattern) -> None:
        network = fitted()
        stored = BipolarPattern(pattern)

        assert network.recall(stored) == stored

    @pytest.mark.parametrize("position", range(N_UNITS))
    def test_one_flipped_unit_is_repaired(self, pattern, position) -> None:
        """Every single-unit corruption of every pattern, exhaustively."""
        network = fitted()
        probe = BipolarPattern(with_flips(pattern, position))

        assert network.recall(probe) == BipolarPattern(pattern)

    @pytest.mark.parametrize("positions", [(0, 1), (0, 8), (3, 11), (5, 6), (14, 15)])
    def test_two_flipped_units_are_repaired(self, pattern, positions) -> None:
        network = fitted()
        probe = BipolarPattern(with_flips(pattern, *positions))

        assert network.recall(probe) == BipolarPattern(pattern)

    @pytest.mark.parametrize("random_seed", range(6))
    def test_the_visiting_order_does_not_change_the_answer_here(
        self, pattern, random_seed
    ) -> None:
        """A pattern with a real basin is reached whatever order units move in.

        Which is not true in general, and ``TestReproducibility`` shows a probe
        where the order decides. It is true here because a one-flip probe of an
        orthogonal pattern has only one attractor near it.
        """
        network = fitted(random_seed=random_seed)
        probe = BipolarPattern(with_flips(pattern, 4))

        assert network.recall(probe) == BipolarPattern(pattern)


class TestWhereRecallStops:
    """The limit, asserted rather than left for a user to discover."""

    def test_three_flipped_units_are_not_promised(self) -> None:
        """Measured over twenty orderings: 622 of the 1680 three-flip probes
        do not return their own pattern, against none of the 1680 probes at one
        or two flips. The basin has an edge and it sits between two and three.
        """
        network = fitted()
        stored = BipolarPattern(STRIPES)

        lost = next(
            (
                positions
                for positions in itertools.combinations(range(N_UNITS), 3)
                if network.recall(BipolarPattern(with_flips(STRIPES, *positions)))
                != stored
            ),
            None,
        )

        assert lost is not None

    def test_a_probe_that_is_lost_still_settles_somewhere(self) -> None:
        """Failing to find the pattern is not failing to settle."""
        network = fitted()
        walk = network.recall_walk(
            BipolarPattern(with_flips(STRIPES, 0, 1, 2, 3, 4, 5, 6))
        )

        assert walk.settled
        assert network.is_fixed_point(walk.result)


class TestEnergy:
    """The quantity the convergence proof is about."""

    def test_energy_matches_the_definition(self) -> None:
        network = fitted()
        matrix = weights_by_definition(THREE_PICTURES)

        for pattern in THREE_PICTURES:
            assert network.energy_of(BipolarPattern(pattern)) == pytest.approx(
                energy_by_definition(matrix, pattern)
            )

    def test_a_stored_pattern_sits_below_a_corrupted_one(self) -> None:
        """Recall walks downhill, so the memories have to be at the bottom."""
        network = fitted()

        assert network.energy_of(BipolarPattern(STRIPES)) < network.energy_of(
            BipolarPattern(with_flips(STRIPES, 0, 5, 9))
        )

    def test_a_state_and_its_negation_have_exactly_the_same_energy(self) -> None:
        """Quadratic in the state, so the two minus signs cancel."""
        network = fitted()
        state = BipolarPattern(with_flips(PAIRS, 2, 7))

        assert network.energy_of(state) == network.energy_of(state.flipped())

    def test_one_flip_lowers_the_energy_by_twice_the_weighted_sum(self) -> None:
        """The exact step the convergence proof claims, checked directly.

        Find a unit that disagrees with the sign of what its connections are
        telling it, flip that one unit, and the energy must fall by exactly
        ``2 * |weighted_sum|``. That identity is what makes the walk strictly
        downhill, and it holds only because the diagonal is zero and the matrix
        is symmetric.
        """
        network = fitted()
        state = BipolarPattern(with_flips(STRIPES, 3))
        weighted_sums = network.weights.weighted_sums_for(state)

        unhappy = [
            unit
            for unit in range(N_UNITS)
            if state.values[unit] * weighted_sums[unit] < 0.0
        ]
        assert unhappy, "the probe should have at least one unit wanting to move"

        unit = unhappy[0]
        moved = BipolarPattern(with_flips(list(state.values), unit))

        assert network.energy_of(moved) - network.energy_of(state) == pytest.approx(
            -2.0 * abs(weighted_sums[unit])
        )

    @pytest.mark.parametrize("pattern", THREE_PICTURES, ids=PICTURE_NAMES)
    def test_energy_never_rises_during_an_asynchronous_recall(self, pattern) -> None:
        walk = fitted().recall_walk(BipolarPattern(with_flips(pattern, 0, 9)))
        energies = [walk.initial_energy, *walk.energies]

        assert energies == sorted(energies, reverse=True)


class TestTheNegationIsAlsoStored:
    """Storing a pattern stores its opposite, and nothing can prevent it."""

    @pytest.mark.parametrize("pattern", THREE_PICTURES, ids=PICTURE_NAMES)
    def test_the_negation_is_a_fixed_point_too(self, pattern) -> None:
        assert fitted().is_fixed_point(BipolarPattern(pattern).flipped())

    @pytest.mark.parametrize("pattern", THREE_PICTURES, ids=PICTURE_NAMES)
    def test_the_negation_recalls_itself(self, pattern) -> None:
        network = fitted()
        negated = BipolarPattern(pattern).flipped()

        assert network.recall(negated) == negated

    @pytest.mark.parametrize("pattern", THREE_PICTURES, ids=PICTURE_NAMES)
    def test_a_mostly_reversed_probe_settles_into_the_negation(self, pattern) -> None:
        """Which is the correct answer rather than a near miss.

        The negation sits at the identical energy, so a probe closer to it than
        to the original has landed in the deeper of the two nearest wells, not
        in a failure.
        """
        network = fitted()
        negated = BipolarPattern(pattern).flipped()
        probe = BipolarPattern(with_flips(list(negated.values), 6))

        assert network.recall(probe) == negated
        assert negated.agreements_with(BipolarPattern(pattern)) == 0


class TestSpuriousMinima:
    """Fixed points nobody stored, which are minima and merely shallower ones."""

    @staticmethod
    def mixture() -> BipolarPattern:
        """``sign(first + second + third)``, an odd mixture of all three.

        Three bipolar values cannot sum to zero, so the sign is always defined
        and no tie convention is involved.
        """
        return BipolarPattern(np.sign(np.sum(THREE_PICTURES, axis=0)))

    def test_the_mixture_was_never_stored(self) -> None:
        mixture = self.mixture()

        for pattern in THREE_PICTURES:
            assert mixture != BipolarPattern(pattern)
            assert mixture != BipolarPattern(pattern).flipped()

    def test_the_mixture_is_a_fixed_point_anyway(self) -> None:
        assert fitted().is_fixed_point(self.mixture())

    @pytest.mark.parametrize("random_seed", range(6))
    def test_recall_from_the_mixture_returns_the_mixture(self, random_seed) -> None:
        network = fitted(random_seed=random_seed)

        assert network.recall(self.mixture()) == self.mixture()

    def test_it_sits_higher_than_a_real_memory(self) -> None:
        """Measured: -4.5 for the mixture against -6.5 for each stored pattern."""
        network = fitted()

        assert network.energy_of(self.mixture()) == pytest.approx(-4.5)
        assert network.energy_of(BipolarPattern(STRIPES)) == pytest.approx(-6.5)


class TestSynchronousUpdate:
    """The trap, demonstrated on two units rather than described in prose."""

    def test_asynchronous_settles_at_once(self) -> None:
        """Two passes and not one, and the second is the cost of noticing.

        The first sweep moves the single unit that wants to move; the second
        moves nothing, which is how settling is detected at all. So a network
        that has already arrived still spends one sweep confirming it.
        """
        walk = oscillator(UpdateRule.ASYNCHRONOUS).recall_walk(
            BipolarPattern(OSCILLATING_PROBE)
        )

        assert walk.settled
        assert walk.stopped_because is RecallStop.SETTLED
        assert walk.passes_run == 2

    def test_asynchronous_lands_on_a_stored_pattern(self) -> None:
        """Either the pattern or its negation, since the probe is equidistant."""
        settled = oscillator(UpdateRule.ASYNCHRONOUS).recall(
            BipolarPattern(OSCILLATING_PROBE)
        )
        stored = BipolarPattern(ANTI_CORRELATED[0])

        assert settled in (stored, stored.flipped())

    def test_synchronous_never_settles_on_the_same_probe(self) -> None:
        walk = oscillator(UpdateRule.SYNCHRONOUS).recall_walk(
            BipolarPattern(OSCILLATING_PROBE)
        )

        assert not walk.settled
        assert walk.stopped_because is RecallStop.PASS_LIMIT_REACHED
        assert walk.passes_run == 10

    def test_synchronous_alternates_between_two_states(self) -> None:
        """Period two, exactly, and for as long as the pass limit allows."""
        walk = oscillator(UpdateRule.SYNCHRONOUS, max_passes=8).recall_walk(
            BipolarPattern(OSCILLATING_PROBE)
        )
        states = [recall_pass.state_after for recall_pass in walk]

        assert states[0] == BipolarPattern([-1.0, -1.0])
        assert states[1] == BipolarPattern([1.0, 1.0])
        assert all(
            state == states[position % 2] for position, state in enumerate(states)
        )

    def test_synchronous_never_gets_below_the_energy_it_started_at(self) -> None:
        """Where the asynchronous route drops to -0.5 on its first pass.

        Both oscillating states sit at 0.5, so this is not a slow descent that
        needed more passes. It is a walk that goes nowhere.
        """
        synchronous = oscillator(UpdateRule.SYNCHRONOUS).recall_walk(
            BipolarPattern(OSCILLATING_PROBE)
        )
        asynchronous = oscillator(UpdateRule.ASYNCHRONOUS).recall_walk(
            BipolarPattern(OSCILLATING_PROBE)
        )

        assert list(synchronous.energies) == pytest.approx(
            [0.5] * synchronous.passes_run
        )
        assert asynchronous.energies[-1] == pytest.approx(-0.5)

    def test_a_stored_pattern_still_sits_still_under_synchronous_update(self) -> None:
        """The rule is broken on probes, not on everything."""
        network = oscillator(UpdateRule.SYNCHRONOUS)
        stored = BipolarPattern(ANTI_CORRELATED[0])

        assert network.recall(stored) == stored


class TestTheZeroDiagonal:
    """The quiet bug: a self-connected network fits and remembers nothing."""

    def test_the_self_weight_chosen_here_really_does_dominate(self) -> None:
        """Measured, so the demonstration below is not resting on a guess."""
        matrix = np.abs(np.asarray(fitted().weights))

        assert float(matrix.sum(axis=1).max()) == pytest.approx(1.3125)
        assert float(matrix.sum(axis=1).max()) < SELF_WEIGHT

    def test_the_honest_network_repairs_the_probe(self) -> None:
        """The control, so the next test is about the diagonal and nothing else."""
        probe = BipolarPattern(with_flips(STRIPES, 0))

        assert fitted().recall(probe) == BipolarPattern(STRIPES)

    def test_a_self_connected_network_hands_every_probe_straight_back(self) -> None:
        """Every state becomes its own attractor, so recall recalls nothing.

        Note what does not happen. Nothing raises, no value is infinite, the
        fit reports the same load, and the energy landscape is not even
        changed, since a self weight adds the same constant to every state.
        Only the update is broken.
        """
        network = fitted()
        network._weights = HebbianWeights(
            np.asarray(network.weights) + SELF_WEIGHT * np.eye(N_UNITS),
            PICTURE_UNITS,
        )
        probe = BipolarPattern(with_flips(STRIPES, 0))

        assert network.weights.has_self_connections
        assert network.recall(probe) == probe
        assert network.recall(BipolarPattern(with_flips(STRIPES, 1, 4, 12))) == (
            BipolarPattern(with_flips(STRIPES, 1, 4, 12))
        )


class TestTheObservedRoute:
    """The efficient route and the watchable one are one calculation."""

    @pytest.mark.parametrize(
        "probe",
        [
            STRIPES,
            with_flips(PAIRS, 0),
            with_flips(QUARTERS, 2, 9),
            with_flips(STRIPES, 1, 3, 5, 7, 11),
        ],
        ids=("stored", "one-flip", "two-flip", "five-flip"),
    )
    def test_the_two_routes_agree(self, probe) -> None:
        """A fast path and a slow path with nothing between them are two
        implementations rather than one calculation observed two ways.
        """
        network = fitted()
        pattern = BipolarPattern(probe)

        assert network.recall_walk(pattern).result == network.recall(pattern)

    def test_the_walk_starts_at_the_probe(self) -> None:
        probe = BipolarPattern(with_flips(PAIRS, 6))
        walk = fitted().recall_walk(probe)

        assert walk.probe == probe
        assert next(iter(walk)).state_before == probe

    def test_it_records_one_pass_per_sweep_in_order(self) -> None:
        walk = fitted().recall_walk(BipolarPattern(with_flips(QUARTERS, 0, 7)))
        numbers = [recall_pass.pass_number for recall_pass in walk]

        assert numbers == list(range(1, walk.passes_run + 1))
        assert len(walk) == walk.passes_run

    def test_the_last_pass_is_the_one_that_moved_nothing(self) -> None:
        """Which is how settling is detected, so it costs a sweep to find out."""
        walk = fitted().recall_walk(BipolarPattern(with_flips(STRIPES, 4)))
        passes = list(walk)

        assert walk.settled
        assert not passes[-1].changed_anything
        assert all(recall_pass.changed_anything for recall_pass in passes[:-1])

    def test_each_pass_hands_its_state_to_the_next(self) -> None:
        passes = list(fitted().recall_walk(BipolarPattern(with_flips(PAIRS, 1, 10))))

        for earlier, later in zip(passes, passes[1:], strict=False):
            assert later.state_before == earlier.state_after
            assert later.energy_before == earlier.energy_after

    def test_the_recorded_energies_are_the_states_own_energies(self) -> None:
        network = fitted()
        walk = network.recall_walk(BipolarPattern(with_flips(QUARTERS, 3)))

        for recall_pass in walk:
            assert recall_pass.energy_after == pytest.approx(
                network.energy_of(recall_pass.state_after)
            )

    def test_it_counts_the_units_each_pass_moved(self) -> None:
        """Repairing two flipped units takes at least two moves, and no pass
        can move a unit it never visits, so the total across the walk is the
        number of repairs the settling actually performed.
        """
        walk = fitted().recall_walk(BipolarPattern(with_flips(STRIPES, 0, 1)))
        moves = [recall_pass.n_units_changed for recall_pass in walk]

        assert sum(moves) >= 2
        assert moves[-1] == 0
        assert walk.result == BipolarPattern(STRIPES)


class TestTransforming:
    """The named route, where features go in and features come out."""

    def test_it_returns_one_feature_per_unit_in_the_fitted_order(self) -> None:
        recalled = fitted().transform(PICTURE_FEATURES)

        assert [feature.name for feature in recalled] == list(PICTURE_UNITS)

    def test_the_stored_patterns_come_back_unchanged(self) -> None:
        recalled = fitted().transform(PICTURE_FEATURES)

        for original, answer in zip(PICTURE_FEATURES, recalled, strict=True):
            assert np.array_equal(original.values, answer.values)

    def test_a_corrupted_table_comes_back_repaired(self) -> None:
        corrupted = features_of(
            [
                with_flips(STRIPES, 0),
                with_flips(PAIRS, 5, 9),
                with_flips(QUARTERS, 14),
            ]
        )
        recalled = fitted().transform(corrupted)

        for original, answer in zip(PICTURE_FEATURES, recalled, strict=True):
            assert np.array_equal(original.values, answer.values)

    def test_fit_transform_reproduces_what_it_was_given(self) -> None:
        together = HopfieldNetwork(random_seed=0).fit_transform(PICTURE_FEATURES)

        for original, answer in zip(PICTURE_FEATURES, together, strict=True):
            assert np.array_equal(original.values, answer.values)

    def test_column_order_does_not_matter(self) -> None:
        network = fitted()
        shuffled = list(reversed(PICTURE_FEATURES))

        forwards = network.transform(PICTURE_FEATURES)
        backwards = network.transform(shuffled)

        for first, second in zip(forwards, backwards, strict=True):
            assert first == second

    def test_patterns_in_and_features_from_are_inverses(self) -> None:
        network = fitted()
        patterns = network.patterns_in(PICTURE_FEATURES)

        assert [list(pattern.values) for pattern in patterns] == [
            list(pattern) for pattern in THREE_PICTURES
        ]
        assert network.features_from(patterns) == PICTURE_FEATURES


class TestReproducibility:
    """What the seed fixes, and what it cannot."""

    def test_the_same_seed_recalls_the_same_state(self) -> None:
        probe = BipolarPattern(with_flips(STRIPES, 0, 1, 2, 3, 4, 5, 6, 7))

        assert fitted(random_seed=11).recall(probe) == fitted(random_seed=11).recall(
            probe
        )

    def test_a_second_call_repeats_the_first(self) -> None:
        """A recall builds its own generator, so calls do not drift apart."""
        network = fitted(random_seed=11)
        probe = BipolarPattern(with_flips(PAIRS, 0, 2, 4, 6, 8, 10))

        assert network.recall(probe) == network.recall(probe)

    def test_the_order_decides_when_a_probe_is_equally_close_to_two_wells(
        self,
    ) -> None:
        """Not a defect, and the reason recall takes a seed at all.

        ``(+1, +1)`` sits exactly as far from ``(+1, -1)`` as from ``(-1, +1)``,
        and whichever unit is visited first is the one that gives way. Both
        answers are correct fixed points, and over ten seeds both turn up.
        """
        answers = {
            oscillator(UpdateRule.ASYNCHRONOUS, random_seed=seed).recall(
                BipolarPattern(OSCILLATING_PROBE)
            )
            for seed in range(10)
        }

        assert len(answers) == 2


class TestCapacity:
    """Load is what predicts failure, and the fall is gradual."""

    def test_every_pattern_survives_at_this_load(self) -> None:
        network = fitted()

        assert all(
            network.is_fixed_point(BipolarPattern(pattern))
            for pattern in THREE_PICTURES
        )

    def test_an_overloaded_network_loses_some_of_what_it_was_given(self) -> None:
        """Eight random patterns in sixteen units is a load of 0.5.

        Measured: four of the eight are still fixed points and four are not,
        which is the erosion the module docstring tabulates at a hundred units.
        The assertion is the inequality rather than the four, since the count is
        a property of one particular draw.
        """
        drawn = np.random.default_rng(0).choice(
            np.array([-1.0, 1.0]), size=(8, N_UNITS)
        )
        network = HopfieldNetwork(random_seed=0).fit(features_of(drawn.tolist()))

        surviving = sum(
            network.is_fixed_point(BipolarPattern(pattern)) for pattern in drawn
        )

        assert network.load == pytest.approx(0.5)
        assert surviving < len(drawn)


class TestBipolarPattern:
    """The value object that makes a non-bipolar state unrepresentable."""

    @pytest.mark.parametrize("values", [[1.0, 0.0], [1.0, 0.5], [2.0, -1.0], [0.0]])
    def test_anything_that_is_not_minus_one_or_one_is_refused(self, values) -> None:
        with pytest.raises(InvalidValuesError):
            BipolarPattern(values)

    def test_an_empty_state_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            BipolarPattern([])

    def test_a_two_dimensional_state_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            BipolarPattern(np.ones((2, 2)))

    def test_flipping_reverses_every_unit(self) -> None:
        pattern = BipolarPattern([1.0, -1.0, 1.0])

        assert pattern.flipped() == BipolarPattern([-1.0, 1.0, -1.0])
        assert pattern.flipped().flipped() == pattern

    def test_agreements_count_the_units_that_match(self) -> None:
        pattern = BipolarPattern([1.0, -1.0, 1.0, 1.0])

        assert pattern.agreements_with(pattern) == 4
        assert pattern.agreements_with(pattern.flipped()) == 0
        assert pattern.agreements_with(BipolarPattern([1.0, 1.0, 1.0, 1.0])) == 3

    def test_comparing_states_of_different_widths_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            BipolarPattern([1.0, 1.0]).agreements_with(BipolarPattern([1.0]))

    def test_equality_is_one_verdict_rather_than_one_per_unit(self) -> None:
        """Unlike ``Predictions``, and the docstring argues why."""
        answer = BipolarPattern([1.0, -1.0]) == BipolarPattern([1.0, -1.0])

        assert answer is True

    def test_it_defers_to_things_that_are_not_patterns(self) -> None:
        assert BipolarPattern([1.0, -1.0]) != [1.0, -1.0]

    def test_equal_states_hash_alike(self) -> None:
        assert len({BipolarPattern([1.0, -1.0]), BipolarPattern([1.0, -1.0])}) == 1

    def test_numpy_gets_a_copy_when_it_asks_for_one(self) -> None:
        """The ``__array__`` contract, and the aliasing it exists to prevent."""
        pattern = BipolarPattern([1.0, -1.0])

        assert not np.shares_memory(np.array(pattern), pattern.values)

    def test_its_values_cannot_be_written_to(self) -> None:
        pattern = BipolarPattern([1.0, -1.0])

        with pytest.raises(ValueError):
            pattern.values[0] = -1.0


class TestHebbianWeights:
    """What the container refuses, and the one thing it deliberately does not."""

    @staticmethod
    def matrix() -> np.ndarray:
        return np.array([[0.0, 0.5], [0.5, 0.0]])

    def test_a_non_square_matrix_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            HebbianWeights(np.zeros((2, 3)), ("left", "right", "extra"))

    def test_a_one_dimensional_matrix_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            HebbianWeights(np.zeros(4), ("left",))

    def test_an_asymmetric_matrix_is_refused(self) -> None:
        """Without symmetry there is no settling proof, so it is structural."""
        with pytest.raises(InvalidValuesError):
            HebbianWeights(np.array([[0.0, 1.0], [-1.0, 0.0]]), TWO_UNITS)

    def test_a_non_finite_entry_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            HebbianWeights(np.array([[0.0, np.nan], [np.nan, 0.0]]), TWO_UNITS)

    def test_the_wrong_number_of_names_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            HebbianWeights(self.matrix(), ("only_one",))

    def test_duplicate_names_are_refused(self) -> None:
        with pytest.raises(NonUniqueFeaturesError):
            HebbianWeights(self.matrix(), ("same", "same"))

    def test_a_self_connected_matrix_is_accepted_on_purpose(self) -> None:
        """The diagonal is the storage rule's responsibility, not this one's.

        Refusing it here would move the lesson into the container and leave
        nothing able to demonstrate what a self connection does to recall.
        """
        weights = HebbianWeights(np.array([[1.0, 0.5], [0.5, 1.0]]), TWO_UNITS)

        assert weights.has_self_connections

    def test_an_unknown_unit_name_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            HebbianWeights(self.matrix(), TWO_UNITS).weight_between("left", "nowhere")

    def test_a_state_of_the_wrong_width_is_refused(self) -> None:
        weights = HebbianWeights(self.matrix(), TWO_UNITS)

        with pytest.raises(InvalidValuesError):
            weights.energy_of(BipolarPattern([1.0, -1.0, 1.0]))

    def test_it_copies_the_matrix_it_is_given(self) -> None:
        source = self.matrix()
        weights = HebbianWeights(source, TWO_UNITS)
        source[0, 1] = 99.0

        assert weights.weight_between("left", "right") == pytest.approx(0.5)

    def test_the_weighted_sums_are_the_matrix_applied_to_the_state(self) -> None:
        weights = HebbianWeights(self.matrix(), TWO_UNITS)
        sums = weights.weighted_sums_for(BipolarPattern([1.0, -1.0]))

        assert list(sums) == pytest.approx([-0.5, 0.5])


class TestWhatItRefuses:
    """Guards, each raising from the MLLibError hierarchy."""

    def test_reading_the_weights_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = HopfieldNetwork().weights

    @pytest.mark.parametrize(
        "read", ["unit_names", "n_units", "n_stored_patterns", "load"]
    )
    def test_every_learned_attribute_refuses_before_fitting(self, read) -> None:
        with pytest.raises(NotFittedError):
            getattr(HopfieldNetwork(), read)

    def test_recalling_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            HopfieldNetwork().recall(BipolarPattern(STRIPES))

    def test_the_observed_route_refuses_before_fitting_too(self) -> None:
        with pytest.raises(NotFittedError):
            HopfieldNetwork().recall_walk(BipolarPattern(STRIPES))

    def test_transforming_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            HopfieldNetwork().transform(PICTURE_FEATURES)

    def test_a_single_unit_is_refused(self) -> None:
        """One unit has nothing to be connected to, so there is no network."""
        with pytest.raises(TooFewValuesError):
            HopfieldNetwork().fit([Feature("alone", [1.0, -1.0])])

    def test_a_non_bipolar_unit_is_refused_by_name(self) -> None:
        with pytest.raises(InvalidValuesError, match="second"):
            HopfieldNetwork().fit(
                [Feature("first", [1.0, -1.0]), Feature("second", [1.0, 0.0])]
            )

    def test_duplicate_unit_names_are_refused(self) -> None:
        with pytest.raises(NonUniqueFeaturesError):
            HopfieldNetwork().fit(
                [Feature("same", [1.0, -1.0]), Feature("same", [-1.0, 1.0])]
            )

    def test_units_of_different_lengths_are_refused(self) -> None:
        with pytest.raises(NonEqualArrayLengthError):
            HopfieldNetwork().fit(
                [Feature("first", [1.0, -1.0]), Feature("second", [1.0])]
            )

    def test_no_units_at_all_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            HopfieldNetwork().fit([])

    def test_a_probe_of_the_wrong_width_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().recall(BipolarPattern([1.0, -1.0]))

    def test_transforming_without_every_fitted_unit_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().transform(PICTURE_FEATURES[:-1])

    def test_transforming_with_an_unknown_unit_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().transform(
                [*PICTURE_FEATURES, Feature("extra", [1.0] * len(THREE_PICTURES))]
            )

    def test_transforming_a_non_bipolar_table_is_refused(self) -> None:
        smudged = features_of([STRIPES, PAIRS, with_flips(QUARTERS, 0)])
        smudged[3] = Feature(PICTURE_UNITS[3], [1.0, 0.25, -1.0])

        with pytest.raises(InvalidValuesError):
            fitted().transform(smudged)

    def test_asking_for_features_from_no_patterns_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().features_from([])


class TestConstruction:
    """Hyperparameters, which pydantic validates the moment you name them."""

    def test_an_unknown_keyword_is_refused(self) -> None:
        """The failure ``extra="forbid"`` exists to stop: a misspelled name
        silently leaving the field at a plausible default.
        """
        # Through a loosely typed alias, because a type checker catches this one
        # statically and the point of the test is what happens at runtime when
        # it does not.
        network_type: Any = HopfieldNetwork

        with pytest.raises(ValidationError):
            network_type(maximum_passes=5)

    @pytest.mark.parametrize("max_passes", [0, -1])
    def test_a_pass_limit_below_one_is_refused(self, max_passes) -> None:
        with pytest.raises(ValidationError):
            HopfieldNetwork(max_passes=max_passes)

    def test_the_default_rule_is_the_one_with_a_proof_behind_it(self) -> None:
        assert HopfieldNetwork().update_rule is UpdateRule.ASYNCHRONOUS

    def test_an_unfitted_network_still_describes_itself(self) -> None:
        assert "unfitted" in repr(HopfieldNetwork())
