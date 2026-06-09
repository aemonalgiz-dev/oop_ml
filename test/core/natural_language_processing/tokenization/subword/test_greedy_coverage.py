"""Spec for GreedyCoverageTokenizer -- the vocabulary as weighted maximum coverage.

The four choices on Sennrich's corpus are worked by hand in the module
docstring, in symbols rather than characters, and the unit is what decides the
first of them.
"""

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
    UnknownTokenError,
    VocabularyTooSmallError,
)
from oop_ml.core.natural_language_processing.tokenization.encoding import Encoding
from oop_ml.core.natural_language_processing.tokenization.subword.greedy_coverage import (
    ChosenPiece,
    ChosenPieces,
    CoverageSegmentation,
    GreedyCoverageTokenizer,
    cover,
    learn_chosen_pieces,
    newly_covered_positions,
)
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    SpelledWord,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.network.purpose import PassPurpose
from test.core.natural_language_processing.fixtures import (
    SENNRICH_ALPHABET,
    SENNRICH_CORPUS,
    SENNRICH_WORD_COUNTS,
)

# The four choices, in order, with the weighted positions each newly covered.
SENNRICH_CHOICES = [
    ("newest</w>", 36),
    ("widest</w>", 18),
    ("low</w>", 15),
    ("lower</w>", 10),
]

# 5 * 3 + 2 * 5 + 6 * 6 + 3 * 6: every weighted position, all of them covered.
EVERY_POSITION = 79

# The unknown token, eleven symbols, four pieces: where every fit stops.
FOUR_PIECES = len(SENNRICH_ALPHABET) + 1 + 4

AB = ChosenPiece(("a", "b"), 1)
BC = ChosenPiece(("b", "c"), 1)


def spelled_sennrich() -> list[SpelledWord]:
    return [
        SpelledWord((*word[:-1], word[-1] + "</w>"), count)
        for word, count in SENNRICH_WORD_COUNTS.items()
    ]


def symbols_of(piece: str) -> tuple[str, ...]:
    if piece.endswith("</w>"):
        return (*piece[:-5], piece[-5:])
    return tuple(piece)


def bare_coverage_of(piece: str) -> int:
    """What ``piece`` covers on the untouched corpus, weighted by word count."""
    return sum(
        word.count
        * len(
            newly_covered_positions(
                word.symbols, [False] * len(word.symbols), symbols_of(piece)
            )
        )
        for word in spelled_sennrich()
    )


def fit_sennrich(vocabulary_size: int, **keywords: object) -> GreedyCoverageTokenizer:
    return GreedyCoverageTokenizer(vocabulary_size=vocabulary_size, **keywords).fit(  # type: ignore[arg-type]
        SENNRICH_CORPUS
    )


class TestChosenPiece:
    def test_carries_symbols_and_coverage(self):
        chosen = ChosenPiece(("e", "s", "t</w>"), 27)

        assert chosen.symbols == ("e", "s", "t</w>")
        assert chosen.piece == "est</w>"
        assert chosen.n_symbols == 3
        assert chosen.coverage == 27
        assert chosen == ChosenPiece(["e", "s", "t</w>"], 27)
        assert chosen.__eq__("est</w>") is NotImplemented

    def test_a_single_symbol_is_never_a_chosen_piece(self):
        with pytest.raises(InvalidValuesError):
            ChosenPiece(("a",), 1)

    def test_needs_positive_coverage_and_non_empty_symbols(self):
        with pytest.raises(InvalidValuesError):
            ChosenPiece(("a", "b"), 0)
        with pytest.raises(EmptyValuesError):
            ChosenPiece(("a", ""), 1)


class TestChosenPieces:
    def test_keeps_rank_order_and_answers_coverage_by_piece(self):
        chosen = ChosenPieces([ChosenPiece(("a", "b"), 5), ChosenPiece(("c", "d"), 2)])

        assert chosen.pieces == ("ab", "cd")
        assert chosen.n_pieces == 2
        assert chosen.coverage_of("cd") == 2
        assert chosen.total_coverage == 7
        assert chosen[0].piece == "ab"
        assert [piece.piece for piece in chosen] == ["ab", "cd"]
        assert "ab" in chosen and "ef" not in chosen

    def test_may_be_empty(self):
        assert ChosenPieces([]).n_pieces == 0
        assert ChosenPieces([]).total_coverage == 0

    def test_refuses_a_piece_chosen_twice(self):
        with pytest.raises(InvalidValuesError):
            ChosenPieces([AB, ChosenPiece(("a", "b"), 3)])

    def test_an_unchosen_piece_raises(self):
        with pytest.raises(UnknownTokenError):
            ChosenPieces([AB]).coverage_of("bc")


class TestCoverageSegmentation:
    def test_carries_pieces_and_the_covered_count(self):
        segmentation = CoverageSegmentation(["ab", "c"], 2)

        assert segmentation.pieces == ("ab", "c")
        assert segmentation.n_tokens == 2
        assert segmentation.n_covered_symbols == 2
        assert list(segmentation) == ["ab", "c"]
        assert segmentation.__eq__("ab") is NotImplemented

    def test_refuses_no_pieces_and_a_negative_count(self):
        with pytest.raises(EmptyValuesError):
            CoverageSegmentation([], 0)
        with pytest.raises(InvalidValuesError):
            CoverageSegmentation(["a"], -1)


class TestNewlyCoveredPositions:
    def test_claims_leftmost_non_overlapping_occurrences(self):
        assert newly_covered_positions(("a", "a", "a"), [False] * 3, ("a", "a")) == (
            0,
            1,
        )

    def test_skips_positions_already_covered(self):
        assert newly_covered_positions(
            ("a", "a", "a"), [True, False, False], ("a", "a")
        ) == (
            1,
            2,
        )

    def test_a_partly_covered_occurrence_is_not_claimed(self):
        assert (
            newly_covered_positions(("a", "b", "c"), [False, True, False], ("a", "b"))
            == ()
        )

    def test_refuses_flags_that_do_not_match_the_symbols(self):
        with pytest.raises(InvalidValuesError):
            newly_covered_positions(("a", "b"), [False], ("a", "b"))


class TestCover:
    def test_applies_pieces_in_rank_order_and_leaves_the_rest_bare(self):
        assert cover(("a", "b", "c"), ChosenPieces([AB, BC])) == CoverageSegmentation(
            ["ab", "c"], 2
        )

    def test_rank_beats_position(self):
        assert cover(("a", "b", "c"), ChosenPieces([BC, AB])) == CoverageSegmentation(
            ["a", "bc"], 2
        )

    def test_a_piece_claims_every_bare_occurrence(self):
        assert cover(("a", "b", "a", "b"), ChosenPieces([AB])) == CoverageSegmentation(
            ["ab", "ab"], 4
        )

    def test_nothing_chosen_spells_every_symbol_alone(self):
        assert cover(("a", "b"), ChosenPieces([])) == CoverageSegmentation(
            ["a", "b"], 0
        )

    def test_refuses_no_symbols(self):
        with pytest.raises(EmptyValuesError):
            cover((), ChosenPieces([AB]))


class TestLearn:
    def test_the_four_choices_on_sennrichs_corpus_in_order(self):
        chosen = learn_chosen_pieces(spelled_sennrich(), 100, 16)

        assert [(piece.piece, piece.coverage) for piece in chosen] == SENNRICH_CHOICES

    def test_the_first_choice_beats_its_rivals_counted_in_symbols(self):
        """In characters with the marker as one, ``est</w>`` would tie at 36."""
        assert bare_coverage_of("newest</w>") == 36
        assert bare_coverage_of("ewest</w>") == 30
        assert bare_coverage_of("est</w>") == 27
        assert bare_coverage_of("west</w>") == 24
        assert 4 * 9 == 36

    def test_the_unmarked_low_never_wins_because_low_ends_in_a_marked_symbol(self):
        assert bare_coverage_of("low") == 6
        assert bare_coverage_of("low</w>") == 15
        assert "low" not in learn_chosen_pieces(spelled_sennrich(), 100, 16)

    def test_stops_when_nothing_is_bare(self):
        chosen = learn_chosen_pieces(spelled_sennrich(), 100, 16)

        assert chosen.n_pieces == 4
        assert chosen.total_coverage == EVERY_POSITION

    def test_stops_at_the_size_asked_for(self):
        assert learn_chosen_pieces(spelled_sennrich(), 2, 16).pieces == (
            "newest</w>",
            "widest</w>",
        )
        assert learn_chosen_pieces(spelled_sennrich(), 0, 16).n_pieces == 0

    def test_the_marginal_gains_telescope_to_the_covered_positions(self):
        chosen = learn_chosen_pieces(spelled_sennrich(), 100, 16)

        by_encoding = sum(
            word.count * cover(word.symbols, chosen).n_covered_symbols
            for word in spelled_sennrich()
        )

        assert by_encoding == chosen.total_coverage == EVERY_POSITION

    def test_ties_go_to_the_lexicographically_smaller_piece(self):
        chosen = learn_chosen_pieces(
            [SpelledWord(("c", "d</w>"), 1), SpelledWord(("a", "b</w>"), 1)], 5, 16
        )

        assert chosen.pieces == ("ab</w>", "cd</w>")
        assert chosen.coverage_of("ab</w>") == 2

    def test_a_piece_length_of_one_leaves_nothing_to_choose(self):
        assert learn_chosen_pieces(spelled_sennrich(), 100, 1).n_pieces == 0

    def test_refuses_no_words_and_bad_sizes(self):
        with pytest.raises(EmptyValuesError):
            learn_chosen_pieces([], 1, 16)
        with pytest.raises(InvalidValuesError):
            learn_chosen_pieces(spelled_sennrich(), -1, 16)
        with pytest.raises(InvalidValuesError):
            learn_chosen_pieces(spelled_sennrich(), 1, 0)


class TestFit:
    def test_the_vocabulary_is_unknown_then_alphabet_then_pieces_in_rank_order(self):
        assert list(fit_sennrich(FOUR_PIECES).vocabulary) == [
            "[UNK]",
            *SENNRICH_ALPHABET,
            *(piece for piece, _ in SENNRICH_CHOICES),
        ]

    def test_stops_early_once_nothing_is_bare(self):
        tokenizer = fit_sennrich(100)

        assert tokenizer.vocabulary.n_tokens == FOUR_PIECES
        assert tokenizer.n_chosen_pieces == 4

    def test_coverage_is_answered_by_piece(self):
        tokenizer = fit_sennrich(100)

        for piece, coverage in SENNRICH_CHOICES:
            assert tokenizer.coverage_of(piece) == coverage
        with pytest.raises(UnknownTokenError):
            tokenizer.coverage_of("est</w>")

    def test_a_smaller_vocabulary_takes_the_first_choices(self):
        assert list(fit_sennrich(13).vocabulary)[12:] == ["newest</w>"]

    def test_a_vocabulary_below_the_alphabet_is_refused(self):
        with pytest.raises(VocabularyTooSmallError):
            fit_sennrich(11)

    def test_exactly_the_alphabet_chooses_nothing(self):
        tokenizer = fit_sennrich(12)

        assert list(tokenizer.vocabulary) == ["[UNK]", *SENNRICH_ALPHABET]
        assert tokenizer.n_chosen_pieces == 0

    def test_the_same_vocabulary_whatever_the_text_order(self):
        reversed_corpus = [" ".join(reversed(SENNRICH_CORPUS[0].split()))]

        assert (
            GreedyCoverageTokenizer(vocabulary_size=100)
            .fit(reversed_corpus)
            .chosen_pieces
            == fit_sennrich(100).chosen_pieces
        )

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            GreedyCoverageTokenizer(vocabulary_size=5).fit("low lower")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            GreedyCoverageTokenizer(vocabulary_size=5).fit(["  ", ""])

    def test_fit_returns_self(self):
        tokenizer = GreedyCoverageTokenizer(vocabulary_size=FOUR_PIECES)

        assert tokenizer.fit(SENNRICH_CORPUS) is tokenizer


class TestEncode:
    def test_a_training_word_is_its_own_piece(self):
        assert fit_sennrich(100).encode("lower newest").texts == (
            "lower</w>",
            "newest</w>",
        )

    def test_an_unseen_word_no_piece_occurs_in_is_bare(self):
        segmentation = fit_sennrich(100).best_segmentation("lowest")

        assert segmentation == CoverageSegmentation(
            ["l", "o", "w", "e", "s", "t</w>"], 0
        )

    def test_a_chosen_piece_covers_the_part_of_an_unseen_word_it_occurs_in(self):
        tokenizer = fit_sennrich(100)

        assert tokenizer.best_segmentation("unwidest") == CoverageSegmentation(
            ["u", "n", "widest</w>"], 6
        )
        assert tokenizer.encode("unwidest").texts == ("[UNK]", "n", "widest</w>")

    def test_every_words_pieces_concatenate_to_the_marked_word(self):
        tokenizer = fit_sennrich(100)

        for word in [*SENNRICH_WORD_COUNTS, "lowest", "unwidest"]:
            assert "".join(tokenizer.best_segmentation(word)) == word + "</w>"

    def test_encode_and_best_segmentation_agree(self):
        tokenizer = fit_sennrich(100)

        for word in [*SENNRICH_WORD_COUNTS, "lowest"]:
            assert (
                tokenizer.encode(word).texts == tokenizer.best_segmentation(word).pieces
            )

    def test_ids_are_vocabulary_positions(self):
        tokenizer = fit_sennrich(100)

        assert tokenizer.encode("lower").ids == tokenizer.vocabulary.ids_of(
            ("lower</w>",)
        )

    def test_a_symbol_the_corpus_never_used_is_unknown(self):
        encoding = fit_sennrich(100).encode("xyz")

        assert encoding.texts == ("[UNK]", "[UNK]", "[UNK]")
        assert set(encoding.ids) == {0}

    def test_a_blank_text_encodes_to_nothing(self):
        assert fit_sennrich(100).encode("   ") == Encoding([])

    def test_the_purpose_changes_nothing(self):
        tokenizer = fit_sennrich(100)

        assert tokenizer.encode("lowest", PassPurpose.TRAINING) == tokenizer.encode(
            "lowest"
        )

    def test_before_fit_raises_not_fitted(self):
        tokenizer = GreedyCoverageTokenizer(vocabulary_size=30)

        with pytest.raises(NotFittedError):
            tokenizer.encode("low")
        with pytest.raises(NotFittedError):
            _ = tokenizer.vocabulary
        with pytest.raises(NotFittedError):
            _ = tokenizer.chosen_pieces
        with pytest.raises(NotFittedError):
            tokenizer.coverage_of("low</w>")
        with pytest.raises(NotFittedError):
            tokenizer.best_segmentation("low")

    def test_best_segmentation_refuses_an_empty_word(self):
        with pytest.raises(EmptyValuesError):
            fit_sennrich(100).best_segmentation("")


class TestDecode:
    def test_round_trips_a_word_the_corpus_never_held(self):
        tokenizer = fit_sennrich(100)

        assert (
            tokenizer.decode(tokenizer.encode("lowest widest").ids) == "lowest widest"
        )

    def test_markers_become_the_spaces_between_words(self):
        tokenizer = fit_sennrich(100)

        assert tokenizer.decode(tokenizer.encode("low lower newest").ids) == (
            "low lower newest"
        )

    def test_a_custom_marker_is_honoured(self):
        tokenizer = fit_sennrich(100, end_of_word_marker="_")

        assert tokenizer.encode("lower").texts == ("lower_",)
        assert tokenizer.decode(tokenizer.encode("low lower").ids) == "low lower"


class TestConstruction:
    @pytest.mark.parametrize(
        "keywords",
        [
            {"vocabulary_size": 1},
            {"vocabulary_size": 10, "max_piece_length": 0},
            {"vocabulary_size": 10, "end_of_word_marker": ""},
            {"vocabulary_size": 10, "unknown_token": ""},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            GreedyCoverageTokenizer(**keywords)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            GreedyCoverageTokenizer(vocab_size=10)  # type: ignore[call-arg]

    def test_the_vocabulary_is_a_vocabulary(self):
        assert isinstance(fit_sennrich(FOUR_PIECES).vocabulary, Vocabulary)
