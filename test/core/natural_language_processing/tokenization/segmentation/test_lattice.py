"""Spec for the lattice segmenter and its value objects -- every reading, and the best path.

The two 研究生命起源 frequency tables are worked by hand here, in logs, and
the same lattice is shown to answer either way depending only on them.
"""

import math

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.dictionary import (
    DictionaryEntry,
    WordDictionary,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.hidden_markov import (
    HiddenMarkovSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.lattice import (
    DictionaryLatticeSegmenter,
    Lattice,
    LatticeEdge,
    LatticePath,
    candidate_beats,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word
from test.core.natural_language_processing.tokenization.segmentation.fixtures import (
    LIFE_FAVOURING_FREQUENCIES,
    LIFE_READING,
    RESEARCH_SENTENCE,
    STUDENT_FAVOURING_FREQUENCIES,
    STUDENT_READING,
)


def dictionary_of(frequencies: dict[str, int]) -> WordDictionary:
    return WordDictionary(
        [DictionaryEntry(word, frequency) for word, frequency in frequencies.items()]
    )


LIFE_FAVOURING = dictionary_of(LIFE_FAVOURING_FREQUENCIES)
STUDENT_FAVOURING = dictionary_of(STUDENT_FAVOURING_FREQUENCIES)
THE_CAT = WordDictionary.from_words(["the", "cat", "sat", "on"])


def log_score_of(reading: tuple[str, ...], dictionary: WordDictionary) -> float:
    """The unigram log score of a reading, from the definition."""
    return sum(
        math.log(dictionary.frequency_of(word) / dictionary.total_frequency)
        for word in reading
    )


class TestCandidateBeats:
    def test_a_higher_score_wins_whatever_the_length(self):
        assert candidate_beats(-1.0, 1, -2.0, 3)
        assert not candidate_beats(-2.0, 3, -1.0, 1)

    def test_an_equal_score_goes_to_the_longer_word(self):
        assert candidate_beats(-1.0, 2, -1.0, 1)
        assert not candidate_beats(-1.0, 1, -1.0, 2)

    def test_a_full_tie_keeps_the_incumbent(self):
        assert not candidate_beats(-1.0, 2, -1.0, 2)


class TestLatticeEdge:
    def test_holds_word_span_and_score(self):
        edge = LatticeEdge("研究", 0, 2, -1.5)

        assert edge.word == "研究"
        assert edge.start == 0
        assert edge.end == 2
        assert edge.length == 2
        assert edge.log_score == -1.5

    def test_a_span_that_does_not_fit_the_word_is_refused(self):
        with pytest.raises(InvalidValuesError):
            LatticeEdge("研究", 0, 3, -1.0)
        with pytest.raises(InvalidValuesError):
            LatticeEdge("研究", -1, 1, -1.0)

    def test_an_empty_word_is_refused(self):
        with pytest.raises(EmptyValuesError):
            LatticeEdge("", 0, 0, -1.0)

    def test_a_word_with_whitespace_is_refused(self):
        with pytest.raises(InvalidValuesError):
            LatticeEdge("a b", 0, 3, -1.0)

    @pytest.mark.parametrize("log_score", [math.inf, -math.inf, math.nan])
    def test_a_non_finite_score_is_refused(self, log_score):
        with pytest.raises(InvalidValuesError):
            LatticeEdge("a", 0, 1, log_score)

    def test_equality_is_by_value(self):
        assert LatticeEdge("a", 0, 1, -1.0) == LatticeEdge("a", 0, 1, -1.0)
        assert LatticeEdge("a", 0, 1, -1.0) != LatticeEdge("a", 0, 1, -2.0)
        assert LatticeEdge("a", 0, 1, -1.0).__eq__("a") is NotImplemented


class TestLatticePath:
    def test_total_is_the_sum_of_its_edges(self):
        path = LatticePath(
            [LatticeEdge("ab", 0, 2, -1.5), LatticeEdge("c", 2, 3, -2.0)]
        )

        assert path.total_log_score == pytest.approx(-3.5)
        assert path.words == ("ab", "c")
        assert path.n_edges == 2
        assert len(path) == 2
        assert path[1].word == "c"

    def test_an_empty_path_totals_zero(self):
        assert LatticePath([]).total_log_score == 0.0
        assert LatticePath([]).words == ()

    def test_overlapping_edges_are_refused(self):
        with pytest.raises(InvalidValuesError):
            LatticePath([LatticeEdge("ab", 0, 2, -1.0), LatticeEdge("bc", 1, 3, -1.0)])

    def test_a_gap_between_edges_is_allowed_because_whitespace_has_none(self):
        path = LatticePath(
            [LatticeEdge("ab", 0, 2, -1.0), LatticeEdge("cd", 3, 5, -1.0)]
        )

        assert path.words == ("ab", "cd")

    def test_equality_is_by_edges(self):
        first = LatticePath([LatticeEdge("a", 0, 1, -1.0)])

        assert first == LatticePath([LatticeEdge("a", 0, 1, -1.0)])
        assert first != LatticePath([])
        assert first.__eq__(("a",)) is NotImplemented


def hand_lattice() -> Lattice:
    """abc with a, ab, b, bc, c: a | bc totals -2.2, ab | c -2.5, a | b | c -3.0."""
    return Lattice(
        "abc",
        [
            LatticeEdge("a", 0, 1, -1.0),
            LatticeEdge("ab", 0, 2, -1.5),
            LatticeEdge("b", 1, 2, -1.0),
            LatticeEdge("bc", 1, 3, -1.2),
            LatticeEdge("c", 2, 3, -1.0),
        ],
    )


class TestLattice:
    def test_groups_candidates_by_start_position(self):
        lattice = hand_lattice()

        assert lattice.text == "abc"
        assert lattice.n_positions == 3
        assert lattice.n_edges == 5
        assert len(lattice) == 5
        assert lattice.positions == (0, 1, 2)
        assert [edge.word for edge in lattice.edges_from(0)] == ["a", "ab"]
        assert [edge.word for edge in lattice.edges_from(1)] == ["b", "bc"]
        assert lattice.edges_from(5) == ()

    def test_iterates_every_candidate_by_position_then_listing_order(self):
        assert [edge.word for edge in hand_lattice()] == ["a", "ab", "b", "bc", "c"]

    def test_best_path_is_the_hand_worked_maximum(self):
        path = hand_lattice().best_path()

        assert path.words == ("a", "bc")
        assert path.total_log_score == pytest.approx(-2.2)

    def test_best_path_ties_go_to_the_longer_word(self):
        """ab | c and a | b | c both total -3.0; ab is longer."""
        lattice = Lattice(
            "abc",
            [
                LatticeEdge("a", 0, 1, -1.0),
                LatticeEdge("ab", 0, 2, -2.0),
                LatticeEdge("b", 1, 2, -1.0),
                LatticeEdge("c", 2, 3, -1.0),
            ],
        )

        assert lattice.best_path().words == ("ab", "c")

    def test_best_path_steps_over_whitespace_at_no_cost(self):
        lattice = Lattice(
            "ab cd",
            [
                LatticeEdge("a", 0, 1, -1.0),
                LatticeEdge("b", 1, 2, -1.0),
                LatticeEdge("c", 3, 4, -1.0),
                LatticeEdge("cd", 3, 5, -1.5),
                LatticeEdge("d", 4, 5, -1.0),
            ],
        )

        path = lattice.best_path()

        assert path.words == ("a", "b", "cd")
        assert [(edge.start, edge.end) for edge in path] == [(0, 1), (1, 2), (3, 5)]
        assert path.total_log_score == pytest.approx(-3.5)

    def test_an_edge_that_does_not_match_the_text_is_refused(self):
        with pytest.raises(InvalidValuesError):
            Lattice(
                "abc",
                [
                    LatticeEdge("x", 0, 1, -1.0),
                    LatticeEdge("b", 1, 2, -1.0),
                    LatticeEdge("c", 2, 3, -1.0),
                ],
            )

    def test_an_edge_past_the_end_is_refused(self):
        with pytest.raises(InvalidValuesError):
            Lattice("ab", [LatticeEdge("a", 0, 1, -1.0), LatticeEdge("bc", 1, 3, -1.0)])

    def test_two_candidates_over_one_span_are_refused(self):
        with pytest.raises(InvalidValuesError):
            Lattice("a", [LatticeEdge("a", 0, 1, -1.0), LatticeEdge("a", 0, 1, -2.0)])

    def test_a_non_whitespace_position_with_no_candidate_is_refused(self):
        with pytest.raises(InvalidValuesError):
            Lattice("ab", [LatticeEdge("a", 0, 1, -1.0)])

    def test_whitespace_positions_need_no_candidate(self):
        lattice = Lattice(" a ", [LatticeEdge("a", 1, 2, -1.0)])

        assert lattice.positions == (1,)
        assert lattice.best_path().words == ("a",)

    def test_a_blank_text_has_an_empty_best_path(self):
        assert Lattice("   ", []).best_path() == LatticePath([])

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            Lattice(["a"], [])  # type: ignore[arg-type]


class TestTheResearchSentence:
    def test_life_favouring_frequencies_pick_the_life_reading(self):
        segmenter = DictionaryLatticeSegmenter(dictionary=LIFE_FAVOURING)

        assert segmenter.split(RESEARCH_SENTENCE).texts == LIFE_READING

    def test_the_two_readings_score_as_the_hand_worked_logs(self):
        """log(10/33) + log(8/33) + log(5/33) against log(6/33) + log(4/33) + log(5/33)."""
        life = math.log(10 / 33) + math.log(8 / 33) + math.log(5 / 33)
        student = math.log(6 / 33) + math.log(4 / 33) + math.log(5 / 33)

        assert life == pytest.approx(-4.4981, abs=1e-4)
        assert student == pytest.approx(-5.7020, abs=1e-4)
        assert life - student == pytest.approx(math.log(10 / 3))

        path = DictionaryLatticeSegmenter(dictionary=LIFE_FAVOURING).best_path(
            RESEARCH_SENTENCE
        )

        assert path.words == LIFE_READING
        assert path.total_log_score == pytest.approx(life)

    def test_student_favouring_frequencies_pick_the_student_reading(self):
        """20 * 12 * 5 = 1200 against 10 * 8 * 5 = 400, total 55."""
        segmenter = DictionaryLatticeSegmenter(dictionary=STUDENT_FAVOURING)
        student = math.log(20 / 55) + math.log(12 / 55) + math.log(5 / 55)

        assert segmenter.split(RESEARCH_SENTENCE).texts == STUDENT_READING
        assert segmenter.best_path(RESEARCH_SENTENCE).total_log_score == pytest.approx(
            student
        )
        assert student == pytest.approx(-4.9319, abs=1e-4)

    def test_the_lattice_holds_every_dictionary_reading_and_every_single_character(
        self,
    ):
        lattice = DictionaryLatticeSegmenter(dictionary=LIFE_FAVOURING).lattice_of(
            RESEARCH_SENTENCE
        )

        assert [edge.word for edge in lattice] == [
            "研",
            "研究",
            "研究生",
            "究",
            "生",
            "生命",
            "命",
            "起",
            "起源",
            "源",
        ]
        assert lattice.n_edges == 10

    def test_a_dictionary_word_scores_its_share_and_a_fallback_scores_one_count(self):
        lattice = DictionaryLatticeSegmenter(dictionary=LIFE_FAVOURING).lattice_of(
            RESEARCH_SENTENCE
        )
        by_word = {edge.word: edge for edge in lattice}

        assert by_word["研究"].log_score == pytest.approx(math.log(10 / 33))
        assert by_word["命"].log_score == pytest.approx(math.log(4 / 33))
        assert by_word["研"].log_score == pytest.approx(math.log(1 / 33))

    def test_the_best_path_beats_the_other_reading_built_from_the_same_edges(self):
        lattice = DictionaryLatticeSegmenter(dictionary=LIFE_FAVOURING).lattice_of(
            RESEARCH_SENTENCE
        )
        by_span = {(edge.start, edge.end): edge for edge in lattice}
        student = LatticePath([by_span[(0, 3)], by_span[(3, 4)], by_span[(4, 6)]])

        assert student.words == STUDENT_READING
        assert lattice.best_path().total_log_score > student.total_log_score

    def test_a_dictionary_with_every_frequency_one_prefers_fewer_words(self):
        segmenter = DictionaryLatticeSegmenter(
            dictionary=WordDictionary.from_words(
                ["研究", "研究生", "生命", "命", "起源"]
            )
        )

        path = segmenter.best_path(RESEARCH_SENTENCE)

        assert path.n_edges == 3
        assert path.total_log_score == pytest.approx(3 * math.log(1 / 5))


class TestAgreementBetweenSplitAndBestPath:
    @pytest.mark.parametrize(
        "text",
        [
            RESEARCH_SENTENCE,
            "研究 生命起源",
            "研究生命起源很好",
            "命",
            "  研究  ",
            "很好",
            "",
        ],
    )
    @pytest.mark.parametrize("dictionary", [LIFE_FAVOURING, STUDENT_FAVOURING])
    def test_split_answers_the_words_of_the_best_path(self, text, dictionary):
        segmenter = DictionaryLatticeSegmenter(dictionary=dictionary)

        assert segmenter.split(text).texts == segmenter.best_path(text).words

    def test_the_path_total_is_the_sum_of_its_edges_and_of_the_definition(self):
        segmenter = DictionaryLatticeSegmenter(dictionary=LIFE_FAVOURING)

        path = segmenter.best_path("研究生命起源 命")

        assert path.total_log_score == pytest.approx(
            sum(edge.log_score for edge in path)
        )
        assert path.total_log_score == pytest.approx(
            log_score_of((*LIFE_READING, "命"), LIFE_FAVOURING)
        )

    def test_the_path_edges_carry_true_offsets(self):
        path = DictionaryLatticeSegmenter(dictionary=THE_CAT).best_path("the catsat")

        assert [(edge.word, edge.start, edge.end) for edge in path] == [
            ("the", 0, 3),
            ("cat", 4, 7),
            ("sat", 7, 10),
        ]


class TestUnknownRuns:
    def test_without_an_unknown_segmenter_unknown_characters_stay_single(self):
        segmenter = DictionaryLatticeSegmenter(dictionary=THE_CAT)

        assert segmenter.split("thecatsatonmat").texts == (
            "the",
            "cat",
            "sat",
            "on",
            "m",
            "a",
            "t",
        )

    def test_a_fitted_unknown_segmenter_regroups_the_unknown_run(self):
        tagger = HiddenMarkovSegmenter().fit(
            [["the", "mat"], ["a", "mat"], ["on", "mat"]]
        )
        segmenter = DictionaryLatticeSegmenter(
            dictionary=THE_CAT, unknown_segmenter=tagger
        )

        assert list(segmenter.split("thecatsatonmat")) == [
            Word("the", 0, 3),
            Word("cat", 3, 6),
            Word("sat", 6, 9),
            Word("on", 9, 11),
            Word("mat", 11, 14),
        ]

    def test_the_best_path_still_shows_the_lattice_answer_before_the_handoff(self):
        tagger = HiddenMarkovSegmenter().fit([["the", "mat"]])
        segmenter = DictionaryLatticeSegmenter(
            dictionary=THE_CAT, unknown_segmenter=tagger
        )

        assert segmenter.best_path("thecatmat").words == ("the", "cat", "m", "a", "t")
        assert segmenter.split("thecatmat").texts == ("the", "cat", "mat")

    def test_a_dictionary_single_character_ends_an_unknown_run(self):
        """x and y are unknown, a is a dictionary word between them."""
        dictionary = WordDictionary.from_words(["the", "a"])
        tagger = HiddenMarkovSegmenter().fit([["xay"], ["xy"], ["the", "xy"]])
        segmenter = DictionaryLatticeSegmenter(
            dictionary=dictionary, unknown_segmenter=tagger
        )

        assert segmenter.split("thexay").texts == ("the", "x", "a", "y")

    def test_a_lone_unknown_character_is_left_alone(self):
        tagger = HiddenMarkovSegmenter().fit([["the", "mat"]])
        segmenter = DictionaryLatticeSegmenter(
            dictionary=THE_CAT, unknown_segmenter=tagger
        )

        assert segmenter.split("thezcat").texts == ("the", "z", "cat")

    def test_an_unfitted_unknown_segmenter_constructs_and_refuses_at_split(self):
        segmenter = DictionaryLatticeSegmenter(
            dictionary=THE_CAT, unknown_segmenter=HiddenMarkovSegmenter()
        )

        with pytest.raises(NotFittedError):
            segmenter.split("thecat")

    def test_the_refusal_comes_even_when_no_unknown_run_is_present(self):
        segmenter = DictionaryLatticeSegmenter(
            dictionary=THE_CAT, unknown_segmenter=HiddenMarkovSegmenter()
        )

        with pytest.raises(NotFittedError):
            segmenter.split("thecat")

    def test_fitting_the_shared_tagger_afterwards_is_enough(self):
        tagger = HiddenMarkovSegmenter()
        segmenter = DictionaryLatticeSegmenter(
            dictionary=THE_CAT, unknown_segmenter=tagger
        )
        tagger.fit([["the", "mat"]])

        assert segmenter.split("thecatmat").texts == ("the", "cat", "mat")

    def test_the_observed_routes_do_not_need_the_tagger(self):
        segmenter = DictionaryLatticeSegmenter(
            dictionary=THE_CAT, unknown_segmenter=HiddenMarkovSegmenter()
        )

        assert segmenter.best_path("thecat").words == ("the", "cat")


class TestWhitespace:
    def test_whitespace_ends_a_run_and_offsets_are_true(self):
        assert list(
            DictionaryLatticeSegmenter(dictionary=THE_CAT).split("the cat")
        ) == [
            Word("the", 0, 3),
            Word("cat", 4, 7),
        ]

    def test_no_edge_crosses_whitespace(self):
        dictionary = WordDictionary.from_words(["the", "thecat"])
        lattice = DictionaryLatticeSegmenter(dictionary=dictionary).lattice_of(
            "the cat"
        )

        assert "thecat" not in {edge.word for edge in lattice}
        assert lattice.edges_from(3) == ()

    def test_a_blank_text_has_no_words(self):
        assert DictionaryLatticeSegmenter(dictionary=THE_CAT).split("  ").n_words == 0

    def test_a_non_string_is_refused_by_every_route(self):
        segmenter = DictionaryLatticeSegmenter(dictionary=THE_CAT)

        with pytest.raises(InvalidValuesError):
            segmenter.split(["the"])  # type: ignore[arg-type]
        with pytest.raises(InvalidValuesError):
            segmenter.lattice_of(["the"])  # type: ignore[arg-type]
        with pytest.raises(InvalidValuesError):
            segmenter.best_path(["the"])  # type: ignore[arg-type]


class TestConstruction:
    def test_the_dictionary_is_required(self):
        with pytest.raises(ValidationError):
            DictionaryLatticeSegmenter()  # type: ignore[call-arg]

    def test_the_unknown_segmenter_must_be_a_hidden_markov_segmenter(self):
        with pytest.raises(ValidationError):
            DictionaryLatticeSegmenter(dictionary=THE_CAT, unknown_segmenter="hmm")  # type: ignore[arg-type]

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            DictionaryLatticeSegmenter(dictionary=THE_CAT, hmm=None)  # type: ignore[call-arg]

    def test_the_default_has_no_unknown_segmenter(self):
        assert DictionaryLatticeSegmenter(dictionary=THE_CAT).unknown_segmenter is None
