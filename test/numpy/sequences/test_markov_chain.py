"""Spec for the Markov chain, where the stationary distribution carries the file.

Fitting is counting and dividing, and it is easy to get right. The questions
asked of a fitted chain are where the mathematics has edges, and the fixtures
here are built one per edge, each worked on paper before the chain was asked.

The fixtures
------------
``WEATHER`` is one sequence, read left to right::

    sunny x 10, rainy, rainy, sunny

so sunny -> sunny nine times, sunny -> rainy once, rainy -> rainy once and
rainy -> sunny once. The rows are ``(0.9, 0.1)`` and ``(0.5, 0.5)``, and the
balance ``pi_sunny * 0.1 = pi_rainy * 0.5`` gives a stationary distribution of
``(5/6, 1/6)``. Every entry of the table is positive, so the chain is
irreducible and aperiodic, and the walk converges to that distribution from
anywhere.

``COIN_THAT_ALWAYS_TURNS`` alternates heads and tails. Its table is the flip,
``[[0, 1], [1, 0]]``, whose stationary distribution is ``(0.5, 0.5)`` and whose
walk never settles: from heads it is certainly heads after an even number of
steps and certainly tails after an odd one, forever. The distribution exists
and is unique; the limit does not.

``TWO_ISLANDS`` is two sequences that never meet, north and south in one and
east and west in the other. Each island is a closed group, so a walk started on
one stays there, and each has its own stationary distribution. Any mixture of
the two is stationary as well, so "the" stationary distribution is undefined
and asking for it is refused by name.

``DRAIN`` leaves ``start`` once and never returns to it. The chain is
reducible, and still has exactly one stationary distribution, ``(0, 1)``,
because only one group is closed. That fixture is here because the refusal is
about several closed groups rather than about reducibility, and a chain that
refused every reducible table would refuse this one wrongly.

``DEAD_END`` ends on ``closed`` and never leaves it, so the row for ``closed``
has no counts in it. At a smoothing of zero that row is zero over zero and the
fit is refused; above zero it is spread evenly.

The oracles
-----------
Every probability is a fraction worked on paper and recorded next to the test
that reads it. The stationary distribution is checked three ways that owe
nothing to how it was computed: against the hand-worked balance, against the
fixed-point equation ``pi P = pi`` evaluated in a plain loop, and against a
walk of many single steps, also in a plain loop. The method solves a linear
system; the loop iterates; the two are independent routes to one number.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueStationaryDistributionError,
    NotFittedError,
    TooFewValuesError,
)
from oop_ml.core.persistence.document import ModelDocument
from oop_ml.core.persistence.store import build_model, model_document
from oop_ml.core.sequences.transitions import StateDistribution
from oop_ml.numpy.sequences.markov_chain import MarkovChain

WEATHER = [["sunny"] * 10 + ["rainy", "rainy", "sunny"]]
COIN_THAT_ALWAYS_TURNS = [["heads", "tails", "heads", "tails", "heads"]]
TWO_ISLANDS = [
    ["north", "north", "south", "north"],
    ["east", "east", "west", "east"],
]
DRAIN = [["start", "sink", "sink", "sink"]]
DEAD_END = [["open", "closed"]]


def weather_chain(smoothing: float = 0.0) -> MarkovChain:
    return MarkovChain(smoothing=smoothing).fit(WEATHER)


def one_step_by_loop(
    distribution: Sequence[float], table: Sequence[Sequence[float]]
) -> list[float]:
    """``distribution @ table``, written as the sum it is."""
    n_states = len(distribution)
    return [
        sum(distribution[source] * table[source][target] for source in range(n_states))
        for target in range(n_states)
    ]


def walked_by_loop(
    start: Sequence[float], table: Sequence[Sequence[float]], n_steps: int
) -> list[float]:
    """Many single steps, each by :func:`one_step_by_loop`."""
    current = list(start)
    for _ in range(n_steps):
        current = one_step_by_loop(current, table)
    return current


class TestConstruction:
    def test_the_default_smoothing_is_zero(self) -> None:
        assert MarkovChain().smoothing == 0.0

    @pytest.mark.parametrize("smoothing", [-1.0, -1e-12, math.nan, math.inf])
    def test_a_smoothing_that_is_not_a_finite_count_of_at_least_zero_is_refused(
        self, smoothing: float
    ) -> None:
        with pytest.raises(ValidationError):
            MarkovChain(smoothing=smoothing)

    def test_a_misspelled_keyword_is_refused_rather_than_ignored(self) -> None:
        with pytest.raises(ValidationError):
            MarkovChain(smooting=1.0)  # type: ignore[call-arg]

    def test_it_is_not_fitted_until_fit(self) -> None:
        assert MarkovChain().is_fitted is False

    def test_fit_returns_the_chain(self) -> None:
        chain = MarkovChain()

        assert chain.fit(WEATHER) is chain
        assert chain.is_fitted is True


class TestWhatFitRefuses:
    def test_a_single_string_is_refused(self) -> None:
        """A string is a sequence of one-character strings, so this would
        otherwise learn a chain over letters and raise nothing."""
        with pytest.raises(InvalidValuesError):
            MarkovChain().fit("sunny rainy sunny")  # type: ignore[arg-type]

    def test_a_list_of_strings_is_refused_as_a_list_of_sequences(self) -> None:
        """The same trap one level down, and the likelier one to be written:
        ``["sunny", "rainy"]`` is two sequences of letters, not one sequence of
        two states."""
        with pytest.raises(InvalidValuesError):
            MarkovChain().fit(["sunny", "rainy", "sunny"])

    def test_no_sequences_at_all_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            MarkovChain().fit([])

    def test_an_empty_sequence_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            MarkovChain().fit([["sunny", "rainy"], []])

    @pytest.mark.parametrize(
        "bad_state", [3, None, ""], ids=["number", "none", "empty"]
    )
    def test_a_state_that_is_not_a_non_empty_string_is_refused(
        self, bad_state: object
    ) -> None:
        with pytest.raises(InvalidValuesError):
            MarkovChain().fit([["sunny", bad_state]])  # type: ignore[list-item]


class TestTheStatesItKnows:
    def test_they_are_in_order_of_first_appearance(self) -> None:
        assert weather_chain().states == ("sunny", "rainy")

    def test_first_appearance_is_not_alphabetical(self) -> None:
        chain = MarkovChain().fit([["zebra", "apple", "zebra"]])

        assert chain.states == ("zebra", "apple")

    def test_the_order_runs_across_sequences(self) -> None:
        chain = MarkovChain().fit([["b", "a", "b"], ["c", "a", "c"]])

        assert chain.states == ("b", "a", "c")
        assert chain.n_states == 3

    def test_a_state_seen_only_in_a_sequence_of_one_is_still_known(self) -> None:
        chain = MarkovChain(smoothing=1.0).fit([["a", "b", "a"], ["c"]])

        assert chain.states == ("a", "b", "c")


class TestTheCounts:
    def test_they_are_the_hand_count(self) -> None:
        counts = weather_chain().transition_counts

        assert counts.count_of("sunny", "sunny") == 9
        assert counts.count_of("sunny", "rainy") == 1
        assert counts.count_of("rainy", "rainy") == 1
        assert counts.count_of("rainy", "sunny") == 1
        assert counts.n_transitions == 12

    def test_the_count_table_is_addressable_as_an_array(self) -> None:
        assert np.array_equal(
            np.asarray(weather_chain().transition_counts), [[9, 1], [1, 1]]
        )

    def test_counting_does_not_run_across_the_join_between_sequences(self) -> None:
        """``[a, b]`` then ``[b, a]`` joined would count a ``b -> b`` that
        nobody observed."""
        counts = MarkovChain().fit([["a", "b"], ["b", "a"]]).transition_counts

        assert counts.count_of("b", "b") == 0
        assert counts.n_transitions == 2

    def test_smoothing_does_not_change_what_was_counted(self) -> None:
        assert np.array_equal(
            np.asarray(weather_chain(smoothing=5.0).transition_counts),
            np.asarray(weather_chain().transition_counts),
        )


class TestTheTransitionTable:
    def test_it_is_the_hand_worked_table(self) -> None:
        assert np.allclose(
            np.asarray(weather_chain().transitions), [[0.9, 0.1], [0.5, 0.5]]
        )

    def test_every_row_sums_to_one(self) -> None:
        for row in weather_chain(smoothing=0.3).transitions:
            assert float(np.sum(np.asarray(row))) == pytest.approx(1.0)

    def test_it_is_addressable_by_state_name(self) -> None:
        table = weather_chain().transitions

        assert table["sunny"]["rainy"] == pytest.approx(0.1)
        assert table.probability_of("rainy", "sunny") == pytest.approx(0.5)

    def test_the_chain_answers_the_same_probability_directly(self) -> None:
        probability = weather_chain().probability_of("sunny", "sunny")

        assert probability == pytest.approx(0.9)
        assert type(probability) is float

    def test_an_unknown_state_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            weather_chain().probability_of("sunny", "snowy")

    def test_additive_smoothing_is_the_hand_worked_table(self) -> None:
        """``(count + 1) / (row total + 1 * 2)``: sunny is ``10/12, 2/12`` and
        rainy is ``2/4, 2/4``."""
        assert np.allclose(
            np.asarray(weather_chain(smoothing=1.0).transitions),
            [[10.0 / 12.0, 2.0 / 12.0], [0.5, 0.5]],
        )

    def test_a_state_never_left_is_refused_at_a_smoothing_of_zero(self) -> None:
        """Its row is zero over zero. Refused by name, rather than patched."""
        with pytest.raises(TooFewValuesError, match="closed"):
            MarkovChain().fit(DEAD_END)

    def test_smoothing_gives_the_state_never_left_an_even_row(self) -> None:
        """``open`` is ``(0 + 0.5) / (1 + 1), (1 + 0.5) / (1 + 1)`` and
        ``closed`` is ``0.5 / 1`` each way."""
        chain = MarkovChain(smoothing=0.5).fit(DEAD_END)

        assert np.allclose(np.asarray(chain.transitions), [[0.25, 0.75], [0.5, 0.5]])

    def test_a_failed_refit_leaves_the_previous_fit_intact(self) -> None:
        chain = weather_chain()

        with pytest.raises(TooFewValuesError):
            chain.fit(DEAD_END)

        assert chain.states == ("sunny", "rainy")
        assert chain.probability_of("sunny", "sunny") == pytest.approx(0.9)


class TestTheDistributionAfterSomeSteps:
    def test_zero_steps_is_the_start_itself(self) -> None:
        after = weather_chain().distribution_after("sunny", 0)

        assert np.array_equal(np.asarray(after), [1.0, 0.0])

    def test_one_step_is_the_start_s_row(self) -> None:
        after = weather_chain().distribution_after("sunny", 1)

        assert np.allclose(np.asarray(after), [0.9, 0.1])

    def test_two_steps_is_the_hand_worked_sum(self) -> None:
        """From sunny: ``0.9 * 0.9 + 0.1 * 0.5 = 0.86`` and ``0.14``."""
        after = weather_chain().distribution_after("sunny", 2)

        assert after["sunny"] == pytest.approx(0.86)
        assert after["rainy"] == pytest.approx(0.14)

    def test_a_starting_distribution_is_walked_as_a_mixture(self) -> None:
        """Half and half: ``0.5 * 0.9 + 0.5 * 0.5 = 0.7`` and ``0.3``."""
        start = StateDistribution(["sunny", "rainy"], [0.5, 0.5])

        after = weather_chain().distribution_after(start, 1)

        assert np.allclose(np.asarray(after), [0.7, 0.3])

    def test_a_starting_distribution_is_matched_by_name_not_position(self) -> None:
        start = StateDistribution(["rainy", "sunny"], [1.0, 0.0])

        after = weather_chain().distribution_after(start, 1)

        assert after["sunny"] == pytest.approx(0.5)
        assert after.states == ("sunny", "rainy")

    def test_a_starting_distribution_over_other_states_is_refused(self) -> None:
        start = StateDistribution(["sunny", "snowy"], [0.5, 0.5])

        with pytest.raises(InvalidValuesError):
            weather_chain().distribution_after(start, 1)

    def test_an_unknown_starting_state_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            weather_chain().distribution_after("snowy", 1)

    @pytest.mark.parametrize("n_steps", [-1, 2.5, True, "3"])
    def test_a_step_count_that_is_not_a_whole_number_of_at_least_zero_is_refused(
        self, n_steps: object
    ) -> None:
        with pytest.raises(InvalidValuesError):
            weather_chain().distribution_after("sunny", n_steps)  # type: ignore[arg-type]

    def test_a_numpy_whole_number_is_a_step_count(self) -> None:
        after = weather_chain().distribution_after("sunny", np.int64(2))  # type: ignore[arg-type]

        assert after["sunny"] == pytest.approx(0.86)

    @pytest.mark.parametrize("n_steps", [3, 7, 20])
    def test_it_agrees_with_that_many_single_steps_in_a_loop(
        self, n_steps: int
    ) -> None:
        chain = MarkovChain(smoothing=0.2).fit([["a", "b", "c", "a", "a", "c", "b"]])
        table = np.asarray(chain.transitions).tolist()

        after = chain.distribution_after("b", n_steps)

        assert np.allclose(
            np.asarray(after), walked_by_loop([0.0, 1.0, 0.0], table, n_steps)
        )


class TestTheStationaryDistribution:
    def test_it_is_the_hand_worked_balance(self) -> None:
        stationary = weather_chain().stationary_distribution()

        assert stationary["sunny"] == pytest.approx(5.0 / 6.0)
        assert stationary["rainy"] == pytest.approx(1.0 / 6.0)

    def test_one_step_leaves_it_where_it_was(self) -> None:
        """``pi P = pi``, checked in a plain loop."""
        chain = weather_chain()
        stationary = np.asarray(chain.stationary_distribution()).tolist()

        assert np.allclose(
            one_step_by_loop(stationary, np.asarray(chain.transitions).tolist()),
            stationary,
        )

    @pytest.mark.parametrize("start", ["sunny", "rainy"])
    def test_an_aperiodic_walk_converges_to_it_from_anywhere(self, start: str) -> None:
        chain = weather_chain()

        walked = chain.distribution_after(start, 200)

        assert np.allclose(
            np.asarray(walked), np.asarray(chain.stationary_distribution()), atol=1e-12
        )

    def test_solving_agrees_with_iterating_on_a_larger_chain(self) -> None:
        """The pin: a linear solve against two thousand steps of a plain loop."""
        generator = np.random.default_rng(3)
        names = ["w", "x", "y", "z"]
        sequence = [names[int(index)] for index in generator.integers(0, 4, size=400)]
        chain = MarkovChain().fit([sequence])
        table = np.asarray(chain.transitions).tolist()

        iterated = walked_by_loop([1.0, 0.0, 0.0, 0.0], table, 2000)
        solved = np.asarray(chain.stationary_distribution())

        assert np.allclose(solved, iterated, atol=1e-12)

    def test_it_follows_the_chain_s_own_state_order(self) -> None:
        assert weather_chain().stationary_distribution().states == ("sunny", "rainy")

    def test_a_periodic_chain_has_one_and_the_walk_never_settles_on_it(self) -> None:
        """The flip: ``(0.5, 0.5)`` exists and is unique, and no walk ends there."""
        chain = MarkovChain().fit(COIN_THAT_ALWAYS_TURNS)

        stationary = chain.stationary_distribution()
        even = chain.distribution_after("heads", 1000)
        odd = chain.distribution_after("heads", 1001)

        assert np.allclose(np.asarray(stationary), [0.5, 0.5])
        assert np.array_equal(np.asarray(even), [1.0, 0.0])
        assert np.array_equal(np.asarray(odd), [0.0, 1.0])

    def test_on_a_periodic_chain_it_is_the_long_run_share_of_time(self) -> None:
        """Averaged over the walk rather than read off its end, which is the
        sense in which the flip still has a stationary distribution."""
        chain = MarkovChain().fit(COIN_THAT_ALWAYS_TURNS)

        visited = [
            np.asarray(chain.distribution_after("heads", n)) for n in range(1000)
        ]

        assert np.allclose(np.mean(visited, axis=0), [0.5, 0.5])

    def test_two_closed_groups_are_refused_by_name(self) -> None:
        chain = MarkovChain().fit(TWO_ISLANDS)

        with pytest.raises(NonUniqueStationaryDistributionError, match="north"):
            chain.stationary_distribution()

    def test_the_refusal_names_every_closed_group(self) -> None:
        chain = MarkovChain().fit(TWO_ISLANDS)

        with pytest.raises(NonUniqueStationaryDistributionError, match="east"):
            chain.stationary_distribution()

    def test_the_two_islands_really_do_settle_in_two_different_places(self) -> None:
        """Why the refusal is right. Each island's own balance is ``(2/3, 1/3)``,
        and a walk started on one never leaves it, so two starts settle at two
        different fixed points and neither is "the" answer."""
        chain = MarkovChain().fit(TWO_ISLANDS)
        table = np.asarray(chain.transitions).tolist()

        from_north = np.asarray(chain.distribution_after("north", 200))
        from_east = np.asarray(chain.distribution_after("east", 200))

        assert chain.states == ("north", "south", "east", "west")
        assert np.allclose(from_north, [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0])
        assert np.allclose(from_east, [0.0, 0.0, 2.0 / 3.0, 1.0 / 3.0])
        assert np.allclose(one_step_by_loop(from_north.tolist(), table), from_north)
        assert np.allclose(one_step_by_loop(from_east.tolist(), table), from_east)

    def test_smoothing_joins_the_islands_and_gives_one_answer(self) -> None:
        stationary = (
            MarkovChain(smoothing=0.1).fit(TWO_ISLANDS).stationary_distribution()
        )

        assert float(np.sum(np.asarray(stationary))) == pytest.approx(1.0)

    def test_a_reducible_chain_with_one_closed_group_still_has_one(self) -> None:
        """``start`` is left and never returned to, so it holds no share at all."""
        stationary = MarkovChain().fit(DRAIN).stationary_distribution()

        assert stationary["start"] == pytest.approx(0.0, abs=1e-15)
        assert stationary["sink"] == pytest.approx(1.0)


class TestTheLogProbabilityOfASequence:
    def test_it_is_the_sum_of_the_hand_worked_steps(self) -> None:
        answer = weather_chain().log_probability_of(["sunny", "sunny", "rainy"])

        assert answer == pytest.approx(math.log(0.9) + math.log(0.1))
        assert type(answer) is float

    def test_the_first_state_is_given_rather_than_scored(self) -> None:
        """Only the one step is scored; the chain learns nothing about starts."""
        answer = weather_chain().log_probability_of(["rainy", "sunny"])

        assert answer == pytest.approx(math.log(0.5))

    def test_a_single_state_has_nothing_to_score(self) -> None:
        assert weather_chain().log_probability_of(["rainy"]) == 0.0

    def test_it_is_each_step_s_log_probability_added_up(self) -> None:
        chain = weather_chain(smoothing=0.5)
        sequence = ["rainy", "sunny", "sunny", "rainy", "rainy", "sunny"]

        by_steps = sum(
            math.log(chain.probability_of(source, target))
            for source, target in zip(sequence, sequence[1:], strict=False)
        )

        assert chain.log_probability_of(sequence) == pytest.approx(by_steps)

    def test_an_impossible_step_is_minus_infinity(self) -> None:
        """Heads never followed heads, so at a smoothing of zero it cannot."""
        chain = MarkovChain().fit(COIN_THAT_ALWAYS_TURNS)

        assert chain.log_probability_of(["heads", "heads"]) == -math.inf

    def test_smoothing_makes_the_impossible_merely_unlikely(self) -> None:
        chain = MarkovChain(smoothing=1.0).fit(COIN_THAT_ALWAYS_TURNS)

        assert math.isfinite(chain.log_probability_of(["heads", "heads"]))

    def test_a_single_string_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            weather_chain().log_probability_of("sunny")  # type: ignore[arg-type]

    def test_an_empty_sequence_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            weather_chain().log_probability_of([])

    def test_an_unknown_state_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            weather_chain().log_probability_of(["sunny", "snowy"])


class TestDrawingASequence:
    def test_it_has_the_length_asked_for_and_begins_at_the_start(self) -> None:
        drawn = weather_chain().sample("rainy", 25, random_seed=0)

        assert len(drawn) == 25
        assert drawn[0] == "rainy"
        assert type(drawn) is tuple
        assert set(drawn) <= {"sunny", "rainy"}

    def test_a_length_of_one_is_the_start_alone(self) -> None:
        assert weather_chain().sample("sunny", 1, random_seed=0) == ("sunny",)

    def test_the_same_seed_draws_the_same_sequence(self) -> None:
        chain = weather_chain()

        assert chain.sample("sunny", 200, random_seed=4) == chain.sample(
            "sunny", 200, random_seed=4
        )

    def test_different_seeds_draw_different_sequences(self) -> None:
        chain = weather_chain()

        assert chain.sample("sunny", 200, random_seed=4) != chain.sample(
            "sunny", 200, random_seed=5
        )

    def test_it_never_takes_a_step_of_probability_zero(self) -> None:
        drawn = (
            MarkovChain().fit(COIN_THAT_ALWAYS_TURNS).sample("heads", 9, random_seed=0)
        )

        assert drawn == ("heads", "tails") * 4 + ("heads",)

    def test_a_long_draw_spends_the_stationary_share_of_its_time_in_each_state(
        self,
    ) -> None:
        drawn = weather_chain().sample("rainy", 20000, random_seed=0)

        sunny_share = drawn.count("sunny") / len(drawn)

        assert sunny_share == pytest.approx(5.0 / 6.0, abs=0.02)

    @pytest.mark.parametrize("length", [0, -3, 2.5, True])
    def test_a_length_that_is_not_a_whole_number_of_at_least_one_is_refused(
        self, length: object
    ) -> None:
        with pytest.raises(InvalidValuesError):
            weather_chain().sample("sunny", length, random_seed=0)  # type: ignore[arg-type]

    def test_an_unknown_start_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            weather_chain().sample("snowy", 5, random_seed=0)


UNFITTED_QUESTIONS: list[Callable[[MarkovChain], object]] = [
    lambda chain: chain.states,
    lambda chain: chain.n_states,
    lambda chain: chain.transition_counts,
    lambda chain: chain.transitions,
    lambda chain: chain.probability_of("sunny", "rainy"),
    lambda chain: chain.distribution_after("sunny", 1),
    lambda chain: chain.stationary_distribution(),
    lambda chain: chain.log_probability_of(["sunny", "rainy"]),
    lambda chain: chain.sample("sunny", 3, random_seed=0),
]


@pytest.mark.parametrize(
    "question",
    UNFITTED_QUESTIONS,
    ids=[
        "states",
        "n_states",
        "transition_counts",
        "transitions",
        "probability_of",
        "distribution_after",
        "stationary_distribution",
        "log_probability_of",
        "sample",
    ],
)
def test_every_question_before_fit_is_refused_in_the_library_s_own_words(
    question: Callable[[MarkovChain], object],
) -> None:
    with pytest.raises(NotFittedError):
        question(MarkovChain())


class TestPersistence:
    def test_a_saved_chain_answers_identically(self) -> None:
        fitted = weather_chain(smoothing=0.5)

        rebuilt = build_model(ModelDocument.from_json(model_document(fitted).to_json()))

        assert rebuilt.smoothing == 0.5
        assert rebuilt.states == fitted.states
        assert np.array_equal(
            np.asarray(rebuilt.transitions), np.asarray(fitted.transitions)
        )

    def test_a_tampered_count_is_refused_on_load(self) -> None:
        """A document is untrusted input, and a negative count is not a count."""
        document = model_document(weather_chain())
        learned = document.learned
        learned["_counts"]["counts"]["values"][0][0] = -4

        with pytest.raises(InvalidValuesError):
            build_model(
                ModelDocument(document.model_type, document.hyperparameters, learned)
            )
