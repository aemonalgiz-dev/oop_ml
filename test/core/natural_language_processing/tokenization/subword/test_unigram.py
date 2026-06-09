"""Spec for the unigram language model and the lattice arithmetic it shares.

The expectation-maximisation step is worked by hand in the module docstring on
the one-word corpus ``ab``, and every number here was derived there first. The
Sennrich pins were computed by the implementation and are recorded so that a
change in any tie rule or in the pruning order is caught.
"""

import math
import random

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
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    SpelledWord,
)
from oop_ml.core.natural_language_processing.tokenization.subword.unigram import (
    UNKNOWN_SYMBOL_PENALTY,
    LearnedPieceTable,
    PieceScores,
    PieceTable,
    Segmentation,
    UnigramLanguageModel,
    checked_symbols,
    checked_word,
    learn_piece_table,
    log_sum_exp,
    seed_pieces,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.network.purpose import PassPurpose
from test.core.natural_language_processing.fixtures import (
    SENNRICH_ALPHABET,
    SENNRICH_CORPUS,
    SENNRICH_WORD_COUNTS,
)

# The one-word corpus the module docstring works by hand: ``ab`` once.
AB_SYMBOLS = ("a", "b</w>")
AB_WORD = [SpelledWord(AB_SYMBOLS, 1)]
THIRD = math.log(1 / 3)

# After one EM round on that corpus: 0.6, 0.2, 0.2.
ONE_ROUND = PieceTable(
    {"a": math.log(0.2), "b</w>": math.log(0.2), "ab</w>": math.log(0.6)}
)

# The unknown token plus eleven symbols, then four whole-word pieces.
FOUR_PIECES = len(SENNRICH_ALPHABET) + 1 + 4


def spelled_sennrich() -> list[SpelledWord]:
    return [
        SpelledWord((*word[:-1], word[-1] + "</w>"), count)
        for word, count in SENNRICH_WORD_COUNTS.items()
    ]


def fit_sennrich(vocabulary_size: int, **keywords: object) -> UnigramLanguageModel:
    return UnigramLanguageModel(vocabulary_size=vocabulary_size, **keywords).fit(  # type: ignore[arg-type]
        SENNRICH_CORPUS
    )


class TestLogSumExp:
    def test_matches_the_definition(self):
        values = [-1.0, -2.5, 0.3]

        assert log_sum_exp(values) == pytest.approx(
            math.log(sum(map(math.exp, values)))
        )

    def test_does_not_overflow_on_large_values(self):
        assert log_sum_exp([1000.0, 1000.0]) == pytest.approx(1000.0 + math.log(2))

    def test_no_values_is_minus_infinity(self):
        assert log_sum_exp([]) == -math.inf
        assert log_sum_exp([-math.inf, -math.inf]) == -math.inf


class TestCheckedSymbols:
    def test_passes_a_spelling_through_as_a_tuple(self):
        assert checked_symbols(["l", "o", "w</w>"]) == ("l", "o", "w</w>")

    @pytest.mark.parametrize("symbols", [[], ["a", ""]])
    def test_refuses_nothing_and_an_empty_symbol(self, symbols):
        with pytest.raises(EmptyValuesError):
            checked_symbols(symbols)

    def test_checked_word_refuses_a_non_string_and_an_empty_string(self):
        with pytest.raises(InvalidValuesError):
            checked_word(["low"])
        with pytest.raises(EmptyValuesError):
            checked_word("")


class TestSegmentation:
    def test_carries_pieces_and_log_probability(self):
        segmentation = Segmentation(["lo", "w</w>"], -1.5)

        assert segmentation.pieces == ("lo", "w</w>")
        assert segmentation.log_probability == -1.5
        assert segmentation.n_tokens == 2
        assert list(segmentation) == ["lo", "w</w>"]
        assert len(segmentation) == 2

    @pytest.mark.parametrize("pieces", [[], ["a", ""]])
    def test_needs_non_empty_pieces(self, pieces):
        with pytest.raises(EmptyValuesError):
            Segmentation(pieces, -1.0)

    @pytest.mark.parametrize("log_probability", [0.5, math.inf, math.nan])
    def test_refuses_a_log_probability_that_is_not_one(self, log_probability):
        with pytest.raises(InvalidValuesError):
            Segmentation(["a"], log_probability)

    def test_compares_exactly_and_defers_to_other_types(self):
        assert Segmentation(["a"], -1.0) == Segmentation(["a"], -1.0)
        assert Segmentation(["a"], -1.0) != Segmentation(["a"], -1.5)
        assert Segmentation(["a"], -1.0).__eq__("a") is NotImplemented


class TestPieceScores:
    def test_ranks_highest_first_with_ties_lexicographic(self):
        scores = PieceScores({"b": 2.0, "a": 2.0, "c": 5.0, "d": 0.0})

        assert scores.pieces == ("c", "a", "b", "d")
        assert list(scores) == ["c", "a", "b", "d"]
        assert scores.total == 9.0
        assert scores.score_of("a") == 2.0
        assert "d" in scores and "e" not in scores

    def test_may_be_empty(self):
        assert PieceScores({}).n_pieces == 0
        assert PieceScores({}).total == 0.0

    def test_an_unscored_piece_raises(self):
        with pytest.raises(UnknownTokenError):
            PieceScores({"a": 1.0}).score_of("b")

    @pytest.mark.parametrize("score", [-0.1, math.inf, math.nan])
    def test_refuses_a_negative_or_non_finite_score(self, score):
        with pytest.raises(InvalidValuesError):
            PieceScores({"a": score})

    def test_refuses_an_empty_piece(self):
        with pytest.raises(InvalidValuesError):
            PieceScores({"": 1.0})


class TestPieceTable:
    def test_holds_pieces_in_codepoint_order_and_by_probability(self):
        table = PieceTable({"b": -1.0, "a": -2.0, "c": -0.5})

        assert table.pieces == ("a", "b", "c")
        assert table.pieces_by_probability == ("c", "b", "a")
        assert table.n_pieces == 3
        assert list(table) == ["a", "b", "c"]
        assert "a" in table and "z" not in table
        assert table.log_probability_of("b") == -1.0

    def test_refuses_no_pieces(self):
        with pytest.raises(EmptyValuesError):
            PieceTable({})

    @pytest.mark.parametrize("log_probability", [0.1, math.inf, -math.inf, math.nan])
    def test_refuses_a_log_probability_that_is_not_one(self, log_probability):
        with pytest.raises(InvalidValuesError):
            PieceTable({"a": log_probability})

    def test_refuses_an_empty_piece(self):
        with pytest.raises(InvalidValuesError):
            PieceTable({"": -1.0})

    def test_an_absent_piece_raises(self):
        with pytest.raises(UnknownTokenError):
            PieceTable({"a": -1.0}).log_probability_of("b")

    def test_normalised_sums_to_one(self):
        table = PieceTable({"a": -1.0, "b": -1.0}).normalised()

        assert sum(math.exp(table.log_probability_of(piece)) for piece in table) == (
            pytest.approx(1.0)
        )
        assert table.log_probability_of("a") == pytest.approx(math.log(0.5))

    def test_without_removes_one_piece_and_nothing_else(self):
        table = PieceTable({"a": -1.0, "b": -2.0})

        assert table.without("b") == PieceTable({"a": -1.0})
        with pytest.raises(UnknownTokenError):
            table.without("z")
        with pytest.raises(EmptyValuesError):
            PieceTable({"a": -1.0}).without("a")

    def test_the_unknown_symbol_scores_the_penalty_below_the_rarest_piece(self):
        table = PieceTable({"a": -1.0, "b": -3.0})

        assert table.unknown_symbol_score == -3.0 - UNKNOWN_SYMBOL_PENALTY


class TestSeed:
    def test_the_ab_corpus_seeds_three_pieces_at_equal_frequency(self):
        seed = seed_pieces(AB_WORD, seed_size=1000, max_piece_length=16)

        assert set(seed) == {"a", "b</w>", "ab</w>"}
        assert all(seed.score_of(piece) == 1.0 for piece in seed)
        assert PieceTable.seeded_from(AB_WORD, 1000, 16) == PieceTable(
            {"a": THIRD, "b</w>": THIRD, "ab</w>": THIRD}
        )

    def test_candidates_are_ranked_by_frequency_times_length_in_symbols(self):
        """``newest</w>`` is six symbols seen six times, 36; nothing else reaches 30."""
        seed = seed_pieces(spelled_sennrich(), seed_size=1, max_piece_length=16)

        assert sorted(seed) == sorted([*SENNRICH_ALPHABET, "newest</w>"])
        assert seed.score_of("newest</w>") == 6.0

    def test_every_single_symbol_is_kept_whatever_the_seed_size(self):
        seed = seed_pieces(spelled_sennrich(), seed_size=1, max_piece_length=16)

        assert set(SENNRICH_ALPHABET) <= set(seed)

    def test_a_piece_length_of_one_seeds_only_the_alphabet(self):
        seed = seed_pieces(spelled_sennrich(), seed_size=1000, max_piece_length=1)

        assert sorted(seed) == SENNRICH_ALPHABET

    def test_frequencies_are_weighted_by_word_count(self):
        seed = seed_pieces(spelled_sennrich(), seed_size=1000, max_piece_length=16)

        assert seed.score_of("est</w>") == 9.0
        assert seed.score_of("lo") == 7.0

    def test_refuses_nothing_to_seed_from_and_a_zero_size(self):
        with pytest.raises(EmptyValuesError):
            seed_pieces([], 1, 1)
        with pytest.raises(InvalidValuesError):
            seed_pieces(AB_WORD, 0, 1)
        with pytest.raises(InvalidValuesError):
            seed_pieces(AB_WORD, 1, 0)


class TestExpectationMaximisation:
    """Worked by hand in the module docstring: p = 1/3 each, so the one-piece
    spelling has posterior (1/3) / (1/3 + 1/9) = 3/4."""

    def test_the_marginal_of_ab_is_four_ninths(self):
        table = PieceTable.seeded_from(AB_WORD, 1000, 16)

        assert table.marginal_log_likelihood(AB_SYMBOLS) == pytest.approx(
            math.log(4 / 9)
        )

    def test_forward_and_backward_totals_agree(self):
        table = PieceTable.seeded_from(spelled_sennrich(), 1000, 16)

        for word in spelled_sennrich():
            assert table._forward(word.symbols, 1.0)[-1] == pytest.approx(
                table._backward(word.symbols, 1.0)[0]
            )

    def test_expected_counts_are_three_quarters_and_two_quarters(self):
        expected = PieceTable.seeded_from(AB_WORD, 1000, 16).expected_counts(AB_WORD)

        assert expected.score_of("ab</w>") == pytest.approx(0.75)
        assert expected.score_of("a") == pytest.approx(0.25)
        assert expected.score_of("b</w>") == pytest.approx(0.25)
        assert expected.total == pytest.approx(1.25)

    def test_one_round_gives_six_tenths_and_two_tenths(self):
        table = PieceTable.seeded_from(
            AB_WORD, 1000, 16
        ).expectation_maximisation_round(AB_WORD)

        assert table.pieces == ("a", "ab</w>", "b</w>")
        assert math.exp(table.log_probability_of("ab</w>")) == pytest.approx(0.6)
        assert math.exp(table.log_probability_of("a")) == pytest.approx(0.2)
        assert math.exp(table.log_probability_of("b</w>")) == pytest.approx(0.2)

    def test_a_second_round_moves_further_towards_one_piece(self):
        """0.6 / (0.6 + 0.04) = 0.9375 of one expected count, then normalised."""
        table = ONE_ROUND.expectation_maximisation_round(AB_WORD)

        assert math.exp(table.log_probability_of("ab</w>")) == pytest.approx(
            0.9375 / 1.0625
        )
        assert math.exp(table.log_probability_of("a")) == pytest.approx(0.0625 / 1.0625)

    def test_expected_counts_are_weighted_by_word_count(self):
        expected = PieceTable.seeded_from(AB_WORD, 1000, 16).expected_counts(
            [SpelledWord(AB_SYMBOLS, 4)]
        )

        assert expected.score_of("ab</w>") == pytest.approx(3.0)

    def test_a_piece_no_word_can_use_takes_the_floor_rather_than_minus_infinity(self):
        table = PieceTable(
            {"a": THIRD, "b</w>": THIRD, "ab</w>": THIRD, "zz": THIRD}
        ).expectation_maximisation_round(AB_WORD)

        assert table.log_probability_of("zz") == table.log_probability_of("a")
        assert math.exp(table.log_probability_of("a")) == pytest.approx(0.2)


class TestViterbi:
    def test_an_equal_score_goes_to_fewer_pieces(self):
        table = PieceTable({"a": -1.0, "b": -1.0, "ab": -2.0})

        assert table.best_segmentation(("a", "b")) == Segmentation(["ab"], -2.0)

    def test_an_equal_count_goes_to_the_longer_piece_at_the_earliest_difference(self):
        table = PieceTable({"a": -1.0, "b": -1.0, "c": -1.0, "ab": -1.0, "bc": -1.0})

        assert table.best_segmentation(("a", "b", "c")).pieces == ("ab", "c")

    def test_the_log_probability_is_the_sum_of_the_pieces(self):
        table = PieceTable.seeded_from(spelled_sennrich(), 1000, 16)

        for symbols in [
            *(word.symbols for word in spelled_sennrich()),
            tuple("lowes") + ("t</w>",),
        ]:
            best = table.best_segmentation(symbols)

            assert best.log_probability == pytest.approx(
                sum(table.log_probability_of(piece) for piece in best)
            )
            assert "".join(best) == "".join(symbols)

    def test_an_absent_single_symbol_is_spelled_as_itself_at_the_penalty(self):
        table = PieceTable({"a": -1.0})

        best = table.best_segmentation(("a", "x"))

        assert best.pieces == ("a", "x")
        assert best.log_probability == pytest.approx(-1.0 + table.unknown_symbol_score)

    def test_a_spelling_that_avoids_the_unknown_symbol_wins(self):
        table = PieceTable({"a": -1.0, "ax": -8.0})

        assert table.best_segmentation(("a", "x")).pieces == ("ax",)

    def test_refuses_no_symbols(self):
        with pytest.raises(EmptyValuesError):
            PieceTable({"a": -1.0}).best_segmentation(())


class TestMarginal:
    def test_is_never_below_the_viterbi_score(self):
        table = PieceTable.seeded_from(spelled_sennrich(), 1000, 16)

        for word in spelled_sennrich():
            assert table.marginal_log_likelihood(word.symbols) >= (
                table.best_segmentation(word.symbols).log_probability
            )

    def test_equals_the_viterbi_score_when_only_one_spelling_exists(self):
        table = PieceTable(dict.fromkeys(SENNRICH_ALPHABET, math.log(1 / 11)))
        symbols = ("l", "o", "w</w>")

        assert table.marginal_log_likelihood(symbols) == pytest.approx(
            table.best_segmentation(symbols).log_probability
        )

    def test_exceeds_the_viterbi_score_when_two_spellings_exist(self):
        table = PieceTable.seeded_from(AB_WORD, 1000, 16)

        assert table.marginal_log_likelihood(AB_SYMBOLS) == pytest.approx(
            math.log(4 / 9)
        )
        assert table.best_segmentation(AB_SYMBOLS).log_probability == pytest.approx(
            THIRD
        )


class TestSampling:
    def test_every_draw_joins_to_the_word(self):
        generator = random.Random(0)

        for _ in range(50):
            drawn = ONE_ROUND.sample_segmentation(AB_SYMBOLS, 1.0, generator)

            assert "".join(drawn) == "ab</w>"

    def test_draws_follow_the_posterior_at_temperature_one(self):
        """P(ab</w>) = 0.6 / 0.64 = 0.9375, so 1875 of 2000 are expected; 1871 drawn."""
        generator = random.Random(0)

        n_one_piece = sum(
            ONE_ROUND.sample_segmentation(AB_SYMBOLS, 1.0, generator).pieces
            == ("ab</w>",)
            for _ in range(2000)
        )

        assert n_one_piece == 1871

    def test_the_drawn_log_probability_is_the_sum_of_its_pieces(self):
        generator = random.Random(1)

        for _ in range(20):
            drawn = ONE_ROUND.sample_segmentation(AB_SYMBOLS, 1.0, generator)

            assert drawn.log_probability == pytest.approx(
                sum(ONE_ROUND.log_probability_of(piece) for piece in drawn)
            )

    def test_a_high_temperature_recovers_viterbi_on_every_draw(self):
        generator = random.Random(0)
        best = ONE_ROUND.best_segmentation(AB_SYMBOLS)

        assert all(
            ONE_ROUND.sample_segmentation(AB_SYMBOLS, 50.0, generator) == best
            for _ in range(20)
        )

    def test_a_seed_reproduces_the_draws(self):
        first = random.Random(7)
        second = random.Random(7)

        assert [
            ONE_ROUND.sample_segmentation(AB_SYMBOLS, 1.0, first) for _ in range(20)
        ] == [ONE_ROUND.sample_segmentation(AB_SYMBOLS, 1.0, second) for _ in range(20)]

    def test_refuses_a_temperature_that_is_not_positive(self):
        with pytest.raises(InvalidValuesError):
            ONE_ROUND.sample_segmentation(AB_SYMBOLS, 0.0, random.Random(0))


class TestPruning:
    def test_the_loss_of_a_piece_is_the_viterbi_gap_times_the_count(self):
        """Removing ``ab</w>`` forces ``a b</w>``: log 0.6 - 2 log 0.2 = 2.7081."""
        losses = ONE_ROUND.losses_if_removed(AB_WORD)

        assert losses.score_of("ab</w>") == pytest.approx(
            math.log(0.6) - 2 * math.log(0.2)
        )
        assert losses.score_of("ab</w>") == pytest.approx(2.7080502)

    def test_the_loss_is_weighted_by_the_word_count(self):
        assert ONE_ROUND.losses_if_removed([SpelledWord(AB_SYMBOLS, 3)]).score_of(
            "ab</w>"
        ) == pytest.approx(3 * (math.log(0.6) - 2 * math.log(0.2)))

    def test_single_symbols_are_not_candidates(self):
        losses = ONE_ROUND.losses_if_removed(AB_WORD)

        assert set(losses) == {"ab</w>"}

    def test_a_piece_on_no_best_path_has_no_loss(self):
        table = PieceTable({"a": -1.0, "b</w>": -1.0, "ab</w>": -1.0, "zz": -1.0})

        assert table.losses_if_removed(AB_WORD).score_of("zz") == 0.0

    def test_pruning_keeps_the_alphabet_and_the_most_costly_pieces_renormalised(self):
        table = PieceTable({"a": -1.0, "b</w>": -1.0, "ab</w>": -1.0, "zz": -1.0})

        pruned = table.pruned_to(3, AB_WORD)

        assert pruned.pieces == ("a", "ab</w>", "b</w>")
        assert (
            pruned
            == PieceTable({"a": -1.0, "b</w>": -1.0, "ab</w>": -1.0}).normalised()
        )

    def test_pruning_never_drops_a_single_symbol(self):
        assert ONE_ROUND.pruned_to(1, AB_WORD).pieces == ("a", "b</w>")

    def test_a_table_within_the_size_is_only_normalised(self):
        assert ONE_ROUND.pruned_to(5, AB_WORD) == ONE_ROUND.normalised()


class TestLearnPieceTable:
    def test_a_seed_within_the_target_is_estimated_and_never_pruned(self):
        learned = learn_piece_table(AB_WORD, 3, 1000, 16, 0.75, 1)

        assert learned.n_pruning_rounds == 0
        assert learned.table == PieceTable.seeded_from(
            AB_WORD, 1000, 16
        ).expectation_maximisation_round(AB_WORD)

    def test_lands_on_the_target_exactly(self):
        learned = learn_piece_table(spelled_sennrich(), 15, 1000, 16, 0.75, 2)

        assert learned.table.n_pieces == 15
        assert learned.n_pruning_rounds == 4

    def test_the_answer_does_not_depend_on_word_order(self):
        forward = learn_piece_table(spelled_sennrich(), 15, 1000, 16, 0.75, 2)
        backward = learn_piece_table(spelled_sennrich()[::-1], 15, 1000, 16, 0.75, 2)

        assert forward == backward

    def test_refuses_a_target_below_the_alphabet(self):
        with pytest.raises(VocabularyTooSmallError):
            learn_piece_table(spelled_sennrich(), 10, 1000, 16, 0.75, 2)

    @pytest.mark.parametrize("shrinking_factor", [0.0, 1.0])
    def test_refuses_a_shrinking_factor_that_would_never_terminate(
        self, shrinking_factor
    ):
        with pytest.raises(InvalidValuesError):
            learn_piece_table(AB_WORD, 2, 1000, 16, shrinking_factor, 1)

    def test_refuses_no_rounds_and_no_words(self):
        with pytest.raises(InvalidValuesError):
            learn_piece_table(AB_WORD, 2, 1000, 16, 0.75, 0)
        with pytest.raises(EmptyValuesError):
            learn_piece_table([], 2, 1000, 16, 0.75, 1)

    def test_the_learned_table_carries_its_round_count(self):
        learned = LearnedPieceTable(ONE_ROUND, 3)

        assert learned.table is ONE_ROUND
        assert learned.n_pruning_rounds == 3
        with pytest.raises(InvalidValuesError):
            LearnedPieceTable(ONE_ROUND, -1)


class TestFit:
    def test_the_vocabulary_is_unknown_then_alphabet_then_pieces_by_probability(self):
        vocabulary = fit_sennrich(FOUR_PIECES).vocabulary

        assert list(vocabulary) == [
            "[UNK]",
            *SENNRICH_ALPHABET,
            "newest</w>",
            "low</w>",
            "widest</w>",
            "lower</w>",
        ]
        assert vocabulary.unknown_token == "[UNK]"

    def test_reaches_the_requested_size_exactly(self):
        tokenizer = fit_sennrich(FOUR_PIECES)

        assert tokenizer.vocabulary.n_tokens == FOUR_PIECES
        assert tokenizer.piece_table.n_pieces == FOUR_PIECES - 1
        assert tokenizer.n_pruning_rounds == 4

    def test_pieces_are_ranked_by_falling_probability(self):
        tokenizer = fit_sennrich(FOUR_PIECES)

        assert tokenizer.log_probability_of(
            "newest</w>"
        ) > tokenizer.log_probability_of("low</w>")
        assert tokenizer.log_probability_of("low</w>") > tokenizer.log_probability_of(
            "lower</w>"
        )

    def test_a_seed_within_the_target_gives_a_smaller_vocabulary(self):
        tokenizer = UnigramLanguageModel(vocabulary_size=100).fit(["ab"])

        assert list(tokenizer.vocabulary) == ["[UNK]", "a", "b</w>", "ab</w>"]
        assert tokenizer.n_pruning_rounds == 0

    def test_a_vocabulary_below_the_alphabet_is_refused(self):
        with pytest.raises(VocabularyTooSmallError):
            fit_sennrich(11)

    def test_exactly_the_alphabet_learns_no_pieces(self):
        tokenizer = fit_sennrich(12)

        assert list(tokenizer.vocabulary) == ["[UNK]", *SENNRICH_ALPHABET]
        assert tokenizer.encode("low").texts == ("l", "o", "w</w>")

    def test_the_same_vocabulary_whatever_the_text_order(self):
        reversed_corpus = [" ".join(reversed(SENNRICH_CORPUS[0].split()))]

        assert (
            UnigramLanguageModel(vocabulary_size=20).fit(reversed_corpus).piece_table
            == fit_sennrich(20).piece_table
        )

    def test_counts_across_several_texts(self):
        assert (
            UnigramLanguageModel(vocabulary_size=20)
            .fit(SENNRICH_CORPUS[0].split())
            .piece_table
            == fit_sennrich(20).piece_table
        )

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            UnigramLanguageModel(vocabulary_size=5).fit("low lower")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            UnigramLanguageModel(vocabulary_size=5).fit(["  ", ""])

    def test_fit_returns_self(self):
        tokenizer = UnigramLanguageModel(vocabulary_size=FOUR_PIECES)

        assert tokenizer.fit(SENNRICH_CORPUS) is tokenizer

    def test_a_one_character_word_is_its_marked_character(self):
        tokenizer = UnigramLanguageModel(vocabulary_size=3).fit(["a a a"])

        assert list(tokenizer.vocabulary) == ["[UNK]", "a</w>"]


class TestEncode:
    def test_a_training_word_becomes_one_piece_at_a_generous_size(self):
        assert fit_sennrich(30).encode("newest").texts == ("newest</w>",)
        assert fit_sennrich(FOUR_PIECES).encode("newest").texts == ("newest</w>",)

    def test_an_unseen_word_is_spelled_from_learned_pieces_with_the_marker_last(self):
        tokenizer = fit_sennrich(30)

        assert tokenizer.encode("lowest").texts == ("l", "o", "w", "est</w>")
        assert "est</w>" in tokenizer.vocabulary

    def test_encode_and_best_segmentation_agree(self):
        tokenizer = fit_sennrich(30)

        for word in ["lowest", "newest", "lower", "widest"]:
            assert (
                tokenizer.encode(word).texts == tokenizer.best_segmentation(word).pieces
            )

    def test_the_best_segmentation_carries_the_sum_of_its_pieces(self):
        tokenizer = fit_sennrich(30)
        best = tokenizer.best_segmentation("lowest")

        assert best.log_probability == pytest.approx(
            sum(tokenizer.log_probability_of(piece) for piece in best)
        )
        assert tokenizer.marginal_log_likelihood("lowest") >= best.log_probability

    def test_ids_are_vocabulary_positions(self):
        tokenizer = fit_sennrich(30)
        encoding = tokenizer.encode("lowest")

        assert encoding.ids == tokenizer.vocabulary.ids_of(("l", "o", "w", "est</w>"))

    def test_a_symbol_the_corpus_never_used_is_unknown(self):
        encoding = fit_sennrich(30).encode("xyz")

        assert encoding.texts == ("[UNK]", "[UNK]", "[UNK]")
        assert set(encoding.ids) == {0}

    def test_a_blank_text_encodes_to_nothing(self):
        assert fit_sennrich(30).encode("   ") == Encoding([])

    def test_before_fit_raises_not_fitted(self):
        tokenizer = UnigramLanguageModel(vocabulary_size=30)

        with pytest.raises(NotFittedError):
            tokenizer.encode("low")
        with pytest.raises(NotFittedError):
            _ = tokenizer.vocabulary
        with pytest.raises(NotFittedError):
            _ = tokenizer.piece_table
        with pytest.raises(NotFittedError):
            _ = tokenizer.n_pruning_rounds
        with pytest.raises(NotFittedError):
            tokenizer.log_probability_of("low</w>")
        with pytest.raises(NotFittedError):
            tokenizer.best_segmentation("low")

    def test_best_segmentation_refuses_an_empty_word_and_a_non_string(self):
        tokenizer = fit_sennrich(30)

        with pytest.raises(EmptyValuesError):
            tokenizer.best_segmentation("")
        with pytest.raises(InvalidValuesError):
            tokenizer.best_segmentation(["low"])  # type: ignore[arg-type]


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

    def test_an_unknown_last_symbol_loses_its_word_boundary(self):
        tokenizer = fit_sennrich(30)

        assert tokenizer.decode(tokenizer.encode("lox low").ids) == "lo[UNK]low"


class TestSubwordRegularisation:
    def fit_ab(self, **keywords: object) -> UnigramLanguageModel:
        return UnigramLanguageModel(
            vocabulary_size=4,
            n_expectation_maximisation_rounds=1,
            **keywords,  # type: ignore[arg-type]
        ).fit(["ab"])

    def test_predicting_never_samples(self):
        """Ten predictions consume no randomness: the next training draw equals a
        fresh tokenizer's first."""
        predicted_first = self.fit_ab(random_seed=0)
        fresh = self.fit_ab(random_seed=0)
        for _ in range(10):
            assert predicted_first.encode("ab").texts == ("ab</w>",)

        assert [
            predicted_first.encode("ab", PassPurpose.TRAINING) for _ in range(5)
        ] == [fresh.encode("ab", PassPurpose.TRAINING) for _ in range(5)]

    def test_training_meets_both_spellings_in_their_posterior_share(self):
        """P(ab</w>) = 0.9375 after one round; 94 of 100 draws at seed 0."""
        tokenizer = self.fit_ab(random_seed=0)

        draws = [tokenizer.encode("ab", PassPurpose.TRAINING).texts for _ in range(100)]

        assert set(draws) == {("ab</w>",), ("a", "b</w>")}
        assert draws.count(("ab</w>",)) == 94

    def test_every_training_spelling_decodes_to_the_same_text(self):
        tokenizer = self.fit_ab(random_seed=3)

        for _ in range(50):
            encoding = tokenizer.encode("ab ab", PassPurpose.TRAINING)

            assert tokenizer.decode(encoding.ids) == "ab ab"

    def test_a_high_temperature_recovers_viterbi_on_twenty_draws(self):
        tokenizer = self.fit_ab(random_seed=0, sampling_temperature=50.0)

        assert all(
            tokenizer.encode("ab", PassPurpose.TRAINING) == tokenizer.encode("ab")
            for _ in range(20)
        )

    def test_a_seed_reproduces_the_sequence_of_spellings(self):
        first = self.fit_ab(random_seed=7)
        second = self.fit_ab(random_seed=7)

        assert [first.encode("ab", PassPurpose.TRAINING) for _ in range(20)] == [
            second.encode("ab", PassPurpose.TRAINING) for _ in range(20)
        ]

    def test_sample_segmentation_draws_at_the_configured_temperature(self):
        tokenizer = self.fit_ab(random_seed=0, sampling_temperature=50.0)

        assert tokenizer.sample_segmentation("ab").pieces == ("ab</w>",)
        with pytest.raises(NotFittedError):
            UnigramLanguageModel(vocabulary_size=4).sample_segmentation("ab")

    def test_training_on_sennrich_decodes_to_the_same_text(self):
        tokenizer = fit_sennrich(30, random_seed=5)

        for _ in range(20):
            encoding = tokenizer.encode("lowest newest", PassPurpose.TRAINING)

            assert tokenizer.decode(encoding.ids) == "lowest newest"


class TestConstruction:
    @pytest.mark.parametrize(
        "keywords",
        [
            {"vocabulary_size": 1},
            {"vocabulary_size": 10, "seed_size": 0},
            {"vocabulary_size": 10, "max_piece_length": 0},
            {"vocabulary_size": 10, "shrinking_factor": 0.0},
            {"vocabulary_size": 10, "shrinking_factor": 1.0},
            {"vocabulary_size": 10, "n_expectation_maximisation_rounds": 0},
            {"vocabulary_size": 10, "sampling_temperature": 0.0},
            {"vocabulary_size": 10, "end_of_word_marker": ""},
            {"vocabulary_size": 10, "unknown_token": ""},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            UnigramLanguageModel(**keywords)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            UnigramLanguageModel(vocab_size=10)  # type: ignore[call-arg]

    def test_the_vocabulary_is_a_vocabulary(self):
        assert isinstance(fit_sennrich(FOUR_PIECES).vocabulary, Vocabulary)
