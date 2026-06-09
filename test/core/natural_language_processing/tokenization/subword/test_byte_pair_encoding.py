"""Spec for BytePairEncoding -- a vocabulary grown one most-frequent pair at a time.

Pinned against Sennrich et al.'s toy corpus, whose first ten merges are worked
by hand in the module docstring and in ``fixtures.py``.
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
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.merges import Merges
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.network.purpose import PassPurpose
from test.core.natural_language_processing.fixtures import (
    SENNRICH_ALPHABET,
    SENNRICH_CORPUS,
    SENNRICH_FIRST_TEN_MERGES,
)

# The unknown token plus eleven symbols, then ten merges.
TEN_MERGES = len(SENNRICH_ALPHABET) + 1 + 10


def fit_ten_merges() -> BytePairEncoding:
    return BytePairEncoding(vocabulary_size=TEN_MERGES).fit(SENNRICH_CORPUS)


class TestFit:
    def test_learns_sennrichs_merges_in_order(self):
        learned = [
            (merge.left, merge.right, int(merge.score))
            for merge in fit_ten_merges().merges
        ]

        assert learned == SENNRICH_FIRST_TEN_MERGES

    def test_the_vocabulary_is_unknown_then_alphabet_then_merges(self):
        vocabulary = fit_ten_merges().vocabulary

        assert list(vocabulary)[:12] == ["[UNK]", *SENNRICH_ALPHABET]
        assert list(vocabulary)[12:] == [
            left + right for left, right, _ in SENNRICH_FIRST_TEN_MERGES
        ]
        assert vocabulary.unknown_token == "[UNK]"

    def test_reaches_the_requested_size_exactly(self):
        assert fit_ten_merges().vocabulary.n_tokens == TEN_MERGES
        assert fit_ten_merges().n_merges == 10

    def test_stops_when_no_pair_reaches_the_minimum_frequency(self):
        """After the tenth merge every remaining pair scores 2."""
        tokenizer = BytePairEncoding(vocabulary_size=100, minimum_pair_frequency=3).fit(
            SENNRICH_CORPUS
        )

        assert tokenizer.n_merges == 10
        assert tokenizer.vocabulary.n_tokens == TEN_MERGES

    def test_stops_when_nothing_is_left_to_merge(self):
        tokenizer = BytePairEncoding(vocabulary_size=100, minimum_pair_frequency=1).fit(
            SENNRICH_CORPUS
        )

        assert tokenizer.n_merges == 13
        assert "lower</w>" in tokenizer.vocabulary

    def test_ties_go_to_the_lexicographically_smaller_pair_whatever_the_text_order(
        self,
    ):
        reversed_corpus = [" ".join(reversed(SENNRICH_CORPUS[0].split()))]

        assert (
            BytePairEncoding(vocabulary_size=TEN_MERGES).fit(reversed_corpus).merges
            == fit_ten_merges().merges
        )

    def test_a_vocabulary_below_the_alphabet_is_refused(self):
        with pytest.raises(VocabularyTooSmallError):
            BytePairEncoding(vocabulary_size=11).fit(SENNRICH_CORPUS)

    def test_exactly_the_alphabet_learns_no_merges(self):
        tokenizer = BytePairEncoding(vocabulary_size=12).fit(SENNRICH_CORPUS)

        assert tokenizer.merges == Merges([])
        assert tokenizer.encode("low").texts == ("l", "o", "w</w>")

    def test_counts_across_several_texts(self):
        one_text = BytePairEncoding(vocabulary_size=TEN_MERGES).fit(SENNRICH_CORPUS)
        many_texts = BytePairEncoding(vocabulary_size=TEN_MERGES).fit(
            SENNRICH_CORPUS[0].split()
        )

        assert many_texts.merges == one_text.merges

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            BytePairEncoding(vocabulary_size=5).fit("low lower")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            BytePairEncoding(vocabulary_size=5).fit(["  ", ""])

    def test_fit_returns_self(self):
        tokenizer = BytePairEncoding(vocabulary_size=TEN_MERGES)

        assert tokenizer.fit(SENNRICH_CORPUS) is tokenizer

    def test_a_one_character_word_is_its_marked_character(self):
        tokenizer = BytePairEncoding(vocabulary_size=3).fit(["a a a"])

        assert list(tokenizer.vocabulary) == ["[UNK]", "a</w>"]


class TestEncode:
    def test_a_word_never_seen_is_spelled_from_learned_pieces(self):
        assert fit_ten_merges().encode("lowest").texts == ("lo", "w", "est</w>")

    def test_ids_are_vocabulary_positions(self):
        tokenizer = fit_ten_merges()
        encoding = tokenizer.encode("lowest")

        assert encoding.ids == tokenizer.vocabulary.ids_of(("lo", "w", "est</w>"))

    def test_a_training_word_becomes_one_token_once_fully_merged(self):
        tokenizer = BytePairEncoding(vocabulary_size=100, minimum_pair_frequency=1).fit(
            SENNRICH_CORPUS
        )

        assert tokenizer.encode("lower newest").texts == ("lower</w>", "newest</w>")

    def test_a_symbol_the_corpus_never_used_is_unknown(self):
        encoding = fit_ten_merges().encode("xyz")

        assert encoding.texts == ("[UNK]", "[UNK]", "[UNK]")
        assert set(encoding.ids) == {0}

    def test_a_blank_text_encodes_to_nothing(self):
        assert fit_ten_merges().encode("   ") == Encoding([])

    def test_merges_never_cross_a_word_boundary(self):
        """``e`` ending one word and ``s`` starting the next never join."""
        tokenizer = BytePairEncoding(vocabulary_size=100, minimum_pair_frequency=1).fit(
            ["e s e s e s ex"]
        )

        assert [merge.merged for merge in tokenizer.merges] == ["ex</w>"]
        assert tokenizer.encode("es").texts == ("e", "s</w>")

    def test_before_fit_raises_not_fitted(self):
        tokenizer = BytePairEncoding(vocabulary_size=TEN_MERGES)

        with pytest.raises(NotFittedError):
            tokenizer.encode("low")
        with pytest.raises(NotFittedError):
            _ = tokenizer.vocabulary
        with pytest.raises(NotFittedError):
            _ = tokenizer.merges


class TestDecode:
    def test_round_trips_a_word_the_corpus_never_held(self):
        tokenizer = fit_ten_merges()

        assert tokenizer.decode(tokenizer.encode("lowest").ids) == "lowest"

    def test_markers_become_the_spaces_between_words(self):
        tokenizer = fit_ten_merges()

        assert tokenizer.decode(tokenizer.encode("low lower newest").ids) == (
            "low lower newest"
        )

    def test_a_custom_marker_is_honoured(self):
        tokenizer = BytePairEncoding(
            vocabulary_size=TEN_MERGES, end_of_word_marker="_"
        ).fit(SENNRICH_CORPUS)

        assert tokenizer.encode("lowest").texts == ("lo", "w", "est_")
        assert tokenizer.decode(tokenizer.encode("low lower").ids) == "low lower"

    def test_an_unknown_last_symbol_loses_its_word_boundary(self):
        """Documented rather than hidden: the unknown token carries no marker."""
        tokenizer = fit_ten_merges()

        assert tokenizer.decode(tokenizer.encode("lox low").ids) == "lo[UNK]low"


class TestMergeDropout:
    def test_predicting_ignores_the_dropout(self):
        plain = fit_ten_merges()
        dropped = BytePairEncoding(
            vocabulary_size=TEN_MERGES, merge_dropout=0.9, random_seed=0
        ).fit(SENNRICH_CORPUS)

        assert dropped.encode("newest") == plain.encode("newest")

    def test_training_skips_merges_so_the_word_arrives_in_more_pieces(self):
        dropped = BytePairEncoding(
            vocabulary_size=TEN_MERGES, merge_dropout=0.9, random_seed=0
        ).fit(SENNRICH_CORPUS)

        lengths = {
            len(dropped.encode("newest", PassPurpose.TRAINING)) for _ in range(20)
        }

        assert len(dropped.encode("newest")) == 1
        assert max(lengths) > 1

    def test_every_training_spelling_decodes_to_the_same_text(self):
        dropped = BytePairEncoding(
            vocabulary_size=TEN_MERGES, merge_dropout=0.5, random_seed=3
        ).fit(SENNRICH_CORPUS)

        for _ in range(20):
            encoding = dropped.encode("newest lower", PassPurpose.TRAINING)
            assert dropped.decode(encoding.ids) == "newest lower"

    def test_a_seed_reproduces_the_sequence_of_spellings(self):
        first = BytePairEncoding(
            vocabulary_size=TEN_MERGES, merge_dropout=0.5, random_seed=7
        ).fit(SENNRICH_CORPUS)
        second = BytePairEncoding(
            vocabulary_size=TEN_MERGES, merge_dropout=0.5, random_seed=7
        ).fit(SENNRICH_CORPUS)

        assert [first.encode("newest", PassPurpose.TRAINING) for _ in range(10)] == [
            second.encode("newest", PassPurpose.TRAINING) for _ in range(10)
        ]

    def test_zero_dropout_trains_exactly_as_it_predicts(self):
        tokenizer = fit_ten_merges()

        assert tokenizer.encode("newest", PassPurpose.TRAINING) == tokenizer.encode(
            "newest"
        )


class TestConstruction:
    @pytest.mark.parametrize(
        "keywords",
        [
            {"vocabulary_size": 1},
            {"vocabulary_size": 10, "merge_dropout": 1.0},
            {"vocabulary_size": 10, "merge_dropout": -0.1},
            {"vocabulary_size": 10, "minimum_pair_frequency": 0},
            {"vocabulary_size": 10, "end_of_word_marker": ""},
            {"vocabulary_size": 10, "unknown_token": ""},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            BytePairEncoding(**keywords)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            BytePairEncoding(vocab_size=10)  # type: ignore[call-arg]

    def test_the_vocabulary_is_a_vocabulary(self):
        assert isinstance(fit_ten_merges().vocabulary, Vocabulary)
