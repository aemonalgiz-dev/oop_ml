"""Spec for ShortestPathTokenizer -- the fewest pieces a vocabulary can spell a word in.

The corpus totals on Sennrich's corpus are worked in the module docstring:
seventy-nine tokens when every word is spelled in single symbols, sixteen when
each of the four words is a piece of its own.
"""

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
    VocabularyTooSmallError,
)
from oop_ml.core.natural_language_processing.tokenization.encoding import Encoding
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    SpelledWord,
)
from oop_ml.core.natural_language_processing.tokenization.subword.shortest_path import (
    LearnedPieces,
    ShortestPathLattice,
    ShortestPathTokenizer,
    ShortestSegmentation,
    corpus_token_count,
    learn_shortest_path_pieces,
    shortest_path,
    token_count_increases_if_removed,
    token_count_of,
)
from oop_ml.core.natural_language_processing.tokenization.subword.unigram import (
    seed_pieces,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.network.purpose import PassPurpose
from test.core.natural_language_processing.fixtures import (
    SENNRICH_ALPHABET,
    SENNRICH_CORPUS,
    SENNRICH_WORD_COUNTS,
)

# ``abc`` spelled two ways in two pieces: the longer first piece must win.
TIE_PIECES = {"a", "b", "c", "ab", "bc"}

# The unknown token plus eleven symbols, then the four whole words.
FOUR_PIECES = len(SENNRICH_ALPHABET) + 1 + 4

# Every training word as one piece: sixteen occurrences, sixteen tokens.
ONE_TOKEN_PER_OCCURRENCE = sum(SENNRICH_WORD_COUNTS.values())

# Every word in single symbols: 5 * 3 + 2 * 5 + 6 * 6 + 3 * 6.
ALL_SINGLES = 79


def spelled_sennrich() -> list[SpelledWord]:
    return [
        SpelledWord((*word[:-1], word[-1] + "</w>"), count)
        for word, count in SENNRICH_WORD_COUNTS.items()
    ]


def seed_sennrich() -> set[str]:
    return set(seed_pieces(spelled_sennrich(), 1000, 16).pieces)


def fit_sennrich(vocabulary_size: int, **keywords: object) -> ShortestPathTokenizer:
    return ShortestPathTokenizer(vocabulary_size=vocabulary_size, **keywords).fit(  # type: ignore[arg-type]
        SENNRICH_CORPUS
    )


class TestShortestSegmentation:
    def test_carries_pieces_and_their_count(self):
        segmentation = ShortestSegmentation(["ab", "c"])

        assert segmentation.pieces == ("ab", "c")
        assert segmentation.n_tokens == 2
        assert list(segmentation) == ["ab", "c"]
        assert len(segmentation) == 2
        assert segmentation == ShortestSegmentation(("ab", "c"))
        assert segmentation.__eq__("ab") is NotImplemented

    @pytest.mark.parametrize("pieces", [[], ["a", ""]])
    def test_needs_non_empty_pieces(self, pieces):
        with pytest.raises(EmptyValuesError):
            ShortestSegmentation(pieces)


class TestLattice:
    def test_an_equal_count_goes_to_the_longer_piece_at_the_earliest_difference(self):
        assert shortest_path(("a", "b", "c"), TIE_PIECES) == ShortestSegmentation(
            ["ab", "c"]
        )

    def test_fewer_pieces_beats_a_longer_first_piece(self):
        assert shortest_path(
            ("a", "b", "c"), {"a", "b", "c", "ab", "bc", "abc"}
        ).pieces == ("abc",)

    def test_the_count_route_and_the_spelling_route_agree(self):
        pieces = seed_sennrich()

        for word in [
            *spelled_sennrich(),
            SpelledWord(("l", "o", "w", "e", "s", "t</w>"), 1),
        ]:
            lattice = ShortestPathLattice(word.symbols, pieces)

            assert lattice.n_tokens == lattice.segmentation.n_tokens
            assert token_count_of(word.symbols, pieces) == len(
                shortest_path(word.symbols, pieces)
            )

    def test_the_spelling_joins_back_to_the_word(self):
        pieces = seed_sennrich()

        for word in spelled_sennrich():
            assert "".join(shortest_path(word.symbols, pieces)) == "".join(word.symbols)

    def test_an_absent_single_symbol_is_one_step_spelled_as_itself(self):
        assert shortest_path(("a", "x", "b"), {"a", "b", "ab"}) == ShortestSegmentation(
            ["a", "x", "b"]
        )
        assert token_count_of(("x",), set()) == 1

    def test_refuses_no_symbols(self):
        with pytest.raises(EmptyValuesError):
            shortest_path((), TIE_PIECES)
        with pytest.raises(EmptyValuesError):
            ShortestPathLattice(["a", ""], TIE_PIECES)


class TestCorpusCounts:
    def test_the_alphabet_alone_spells_everything_in_single_symbols(self):
        assert (
            corpus_token_count(spelled_sennrich(), set(SENNRICH_ALPHABET))
            == ALL_SINGLES
        )

    def test_the_seed_holds_every_word_whole(self):
        assert corpus_token_count(spelled_sennrich(), seed_sennrich()) == (
            ONE_TOKEN_PER_OCCURRENCE
        )

    def test_the_increase_of_a_whole_word_piece_is_its_count(self):
        """Losing ``newest</w>`` makes six occurrences two tokens each: +6."""
        increases = token_count_increases_if_removed(
            spelled_sennrich(), seed_sennrich()
        )

        assert increases.score_of("newest</w>") == 6.0
        assert increases.score_of("low</w>") == 5.0
        assert increases.score_of("widest</w>") == 3.0
        assert increases.score_of("lower</w>") == 2.0
        assert increases.pieces[:4] == (
            "newest</w>",
            "low</w>",
            "widest</w>",
            "lower</w>",
        )

    def test_a_piece_on_no_shortest_path_has_no_increase(self):
        increases = token_count_increases_if_removed(
            spelled_sennrich(), seed_sennrich()
        )

        assert increases.score_of("est</w>") == 0.0
        assert increases.score_of("lo") == 0.0

    def test_single_symbols_are_never_candidates(self):
        increases = token_count_increases_if_removed(
            spelled_sennrich(), seed_sennrich()
        )

        assert not set(SENNRICH_ALPHABET) & set(increases)

    def test_removing_a_piece_never_decreases_the_total(self):
        spelled = spelled_sennrich()
        seed = seed_sennrich()
        with_everything = corpus_token_count(spelled, seed)

        for piece in seed - set(SENNRICH_ALPHABET):
            assert corpus_token_count(spelled, seed - {piece}) >= with_everything


class TestLearn:
    def test_lands_on_the_target_exactly(self):
        learned = learn_shortest_path_pieces(spelled_sennrich(), 15, 1000, 16, 0.75)

        assert len(learned) == 15
        assert learned.n_pruning_rounds == 4
        assert learned.total_token_count == ONE_TOKEN_PER_OCCURRENCE
        assert set(learned) - set(SENNRICH_ALPHABET) == {
            "low</w>",
            "lower</w>",
            "newest</w>",
            "widest</w>",
        }

    def test_a_seed_within_the_target_is_never_pruned(self):
        learned = learn_shortest_path_pieces(spelled_sennrich(), 1000, 1000, 16, 0.75)

        assert learned.n_pruning_rounds == 0
        assert set(learned) == seed_sennrich()

    def test_the_answer_does_not_depend_on_word_order(self):
        assert learn_shortest_path_pieces(
            spelled_sennrich(), 15, 1000, 16, 0.75
        ) == learn_shortest_path_pieces(spelled_sennrich()[::-1], 15, 1000, 16, 0.75)

    def test_refuses_a_target_below_the_alphabet(self):
        with pytest.raises(VocabularyTooSmallError):
            learn_shortest_path_pieces(spelled_sennrich(), 10, 1000, 16, 0.75)

    @pytest.mark.parametrize("shrinking_factor", [0.0, 1.0])
    def test_refuses_a_shrinking_factor_that_would_never_terminate(
        self, shrinking_factor
    ):
        with pytest.raises(InvalidValuesError):
            learn_shortest_path_pieces(
                spelled_sennrich(), 12, 1000, 16, shrinking_factor
            )

    def test_refuses_no_words(self):
        with pytest.raises(EmptyValuesError):
            learn_shortest_path_pieces([], 12, 1000, 16, 0.75)

    def test_learned_pieces_carry_their_counts(self):
        learned = LearnedPieces({"b", "a"}, 2, 7)

        assert learned.pieces == ("a", "b")
        assert "a" in learned and len(learned) == 2
        with pytest.raises(EmptyValuesError):
            LearnedPieces([], 0, 0)
        with pytest.raises(InvalidValuesError):
            LearnedPieces(["a"], -1, 0)


class TestFit:
    def test_the_vocabulary_is_unknown_then_alphabet_then_pieces_in_codepoint_order(
        self,
    ):
        assert list(fit_sennrich(FOUR_PIECES).vocabulary) == [
            "[UNK]",
            *SENNRICH_ALPHABET,
            "low</w>",
            "lower</w>",
            "newest</w>",
            "widest</w>",
        ]

    def test_the_alphabet_alone_costs_seventy_nine_tokens(self):
        tokenizer = fit_sennrich(12)

        assert tokenizer.total_token_count == ALL_SINGLES
        assert tokenizer.n_pruning_rounds == 5
        assert list(tokenizer.vocabulary) == ["[UNK]", *SENNRICH_ALPHABET]

    def test_four_whole_words_reach_one_token_per_occurrence(self):
        tokenizer = fit_sennrich(FOUR_PIECES)

        assert tokenizer.total_token_count == ONE_TOKEN_PER_OCCURRENCE
        assert tokenizer.n_pruning_rounds == 4

    def test_a_larger_vocabulary_cannot_go_below_the_floor(self):
        assert fit_sennrich(30).total_token_count == ONE_TOKEN_PER_OCCURRENCE

    def test_a_seed_within_the_target_gives_a_smaller_vocabulary(self):
        tokenizer = ShortestPathTokenizer(vocabulary_size=100).fit(["abz bcz"])

        assert tokenizer.vocabulary.n_tokens == 11
        assert tokenizer.n_pruning_rounds == 0

    def test_a_vocabulary_below_the_alphabet_is_refused(self):
        with pytest.raises(VocabularyTooSmallError):
            fit_sennrich(11)

    def test_the_same_vocabulary_whatever_the_text_order(self):
        reversed_corpus = [" ".join(reversed(SENNRICH_CORPUS[0].split()))]

        assert (
            ShortestPathTokenizer(vocabulary_size=20).fit(reversed_corpus).vocabulary
            == fit_sennrich(20).vocabulary
        )

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            ShortestPathTokenizer(vocabulary_size=5).fit("low lower")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            ShortestPathTokenizer(vocabulary_size=5).fit(["  ", ""])

    def test_fit_returns_self(self):
        tokenizer = ShortestPathTokenizer(vocabulary_size=FOUR_PIECES)

        assert tokenizer.fit(SENNRICH_CORPUS) is tokenizer


class TestEncode:
    def test_an_unseen_word_without_a_piece_of_its_own_is_single_symbols(self):
        assert fit_sennrich(FOUR_PIECES).encode("lowest").texts == (
            "l",
            "o",
            "w",
            "e",
            "s",
            "t</w>",
        )

    def test_a_larger_vocabulary_spells_it_in_fewer(self):
        tokenizer = fit_sennrich(30)

        assert tokenizer.encode("lowest").texts == ("l", "o", "w", "est</w>")
        assert tokenizer.token_count_of("lowest") == 4
        assert tokenizer.best_segmentation("lowest").n_tokens == 4

    def test_the_longer_first_piece_wins_a_tie_on_a_fitted_vocabulary(self):
        """``a b c z</w>`` is ``ab cz</w>`` or ``a bcz</w>``, two pieces either way."""
        tokenizer = ShortestPathTokenizer(vocabulary_size=100).fit(["abz bcz"])

        assert tokenizer.encode("abcz").texts == ("ab", "cz</w>")

    def test_a_training_word_is_one_token(self):
        assert fit_sennrich(FOUR_PIECES).encode("newest lower").texts == (
            "newest</w>",
            "lower</w>",
        )

    def test_ids_are_vocabulary_positions(self):
        tokenizer = fit_sennrich(30)

        assert tokenizer.encode("lowest").ids == tokenizer.vocabulary.ids_of(
            ("l", "o", "w", "est</w>")
        )

    def test_a_symbol_the_corpus_never_used_is_unknown(self):
        encoding = fit_sennrich(30).encode("xyz")

        assert encoding.texts == ("[UNK]", "[UNK]", "[UNK]")
        assert set(encoding.ids) == {0}

    def test_a_blank_text_encodes_to_nothing(self):
        assert fit_sennrich(30).encode("   ") == Encoding([])

    def test_the_purpose_changes_nothing(self):
        tokenizer = fit_sennrich(30)

        assert tokenizer.encode("lowest", PassPurpose.TRAINING) == tokenizer.encode(
            "lowest"
        )

    def test_before_fit_raises_not_fitted(self):
        tokenizer = ShortestPathTokenizer(vocabulary_size=30)

        with pytest.raises(NotFittedError):
            tokenizer.encode("low")
        with pytest.raises(NotFittedError):
            _ = tokenizer.vocabulary
        with pytest.raises(NotFittedError):
            _ = tokenizer.total_token_count
        with pytest.raises(NotFittedError):
            _ = tokenizer.n_pruning_rounds
        with pytest.raises(NotFittedError):
            tokenizer.best_segmentation("low")
        with pytest.raises(NotFittedError):
            tokenizer.token_count_of("low")

    def test_best_segmentation_refuses_an_empty_word(self):
        with pytest.raises(EmptyValuesError):
            fit_sennrich(30).best_segmentation("")


class TestDecode:
    def test_round_trips_a_word_the_corpus_never_held(self):
        tokenizer = fit_sennrich(30)

        assert tokenizer.decode(tokenizer.encode("lowest").ids) == "lowest"

    def test_markers_become_the_spaces_between_words(self):
        tokenizer = fit_sennrich(30)

        assert tokenizer.decode(tokenizer.encode("low lower newest").ids) == (
            "low lower newest"
        )

    def test_a_custom_marker_is_honoured(self):
        tokenizer = fit_sennrich(30, end_of_word_marker="_")

        assert tokenizer.encode("lowest").texts == ("l", "o", "w", "est_")
        assert tokenizer.decode(tokenizer.encode("low lower").ids) == "low lower"


class TestConstruction:
    @pytest.mark.parametrize(
        "keywords",
        [
            {"vocabulary_size": 1},
            {"vocabulary_size": 10, "max_piece_length": 0},
            {"vocabulary_size": 10, "seed_size": 0},
            {"vocabulary_size": 10, "shrinking_factor": 0.0},
            {"vocabulary_size": 10, "shrinking_factor": 1.0},
            {"vocabulary_size": 10, "end_of_word_marker": ""},
            {"vocabulary_size": 10, "unknown_token": ""},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            ShortestPathTokenizer(**keywords)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            ShortestPathTokenizer(vocab_size=10)  # type: ignore[call-arg]

    def test_the_vocabulary_is_a_vocabulary(self):
        assert isinstance(fit_sennrich(FOUR_PIECES).vocabulary, Vocabulary)
