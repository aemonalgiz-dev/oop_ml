"""Spec for SentencePiece -- whitespace as a symbol, two algorithms behind one class.

Pinned against Sennrich et al.'s toy corpus under both algorithms, so that the
framework's normalisation is held to the same words the marked-at-the-end
tokenizers are.
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
from oop_ml.core.natural_language_processing.tokenization.subword.merges import Merges
from oop_ml.core.natural_language_processing.tokenization.subword.sentence_piece import (
    WHITESPACE_MARKER,
    SentencePiece,
    SubwordAlgorithm,
)
from oop_ml.core.natural_language_processing.tokenization.subword.unigram import (
    PieceTable,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.network.purpose import PassPurpose
from test.core.natural_language_processing.fixtures import SENNRICH_CORPUS

# The ten characters of the corpus plus the marker, in codepoint order.
MARKED_ALPHABET = ["d", "e", "i", "l", "n", "o", "r", "s", "t", "w", "▁"]

# The first merges over ``▁ l o w`` style units, with the count that chose each.
FIRST_MERGES = [
    ("e", "s", 9),
    ("es", "t", 9),
    ("l", "o", 7),
    ("lo", "w", 7),
    ("▁", "low", 7),
]

ALGORITHMS = [SubwordAlgorithm.BYTE_PAIR, SubwordAlgorithm.UNIGRAM]


def fit_sennrich(
    algorithm: SubwordAlgorithm, vocabulary_size: int = 30
) -> SentencePiece:
    return SentencePiece(
        vocabulary_size=vocabulary_size, algorithm=algorithm, random_seed=0
    ).fit(SENNRICH_CORPUS)


class TestNormalisation:
    def test_the_marker_is_the_lower_one_eighth_block(self):
        assert WHITESPACE_MARKER == "▁"
        assert SentencePiece(vocabulary_size=5).whitespace_marker == "▁"

    def test_units_carry_the_marker_in_front_and_collapse_whitespace(self):
        tokenizer = SentencePiece(vocabulary_size=5)

        assert tokenizer.units_of("  hello   world\t") == ("▁hello", "▁world")
        assert tokenizer.units_of("") == ()

    def test_units_of_refuses_a_non_string(self):
        with pytest.raises(InvalidValuesError):
            SentencePiece(vocabulary_size=5).units_of(["hello"])  # type: ignore[arg-type]

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_two_spaces_and_one_space_encode_identically(self, algorithm):
        """A documented loss: runs of whitespace collapse before anything is learned."""
        tokenizer = fit_sennrich(algorithm)

        assert tokenizer.encode("low  lower") == tokenizer.encode("low lower")
        assert tokenizer.encode(" low lower ") == tokenizer.encode("low lower")

    def test_a_marker_of_several_characters_is_one_symbol(self):
        tokenizer = SentencePiece(vocabulary_size=30, whitespace_marker="__").fit(
            SENNRICH_CORPUS
        )

        assert "__" in tokenizer.vocabulary
        assert tokenizer.encode("low").texts == ("__low",)
        assert tokenizer.decode(tokenizer.encode("low lower").ids) == "low lower"

    def test_the_framework_takes_no_pre_tokenizer(self):
        with pytest.raises(ValidationError):
            SentencePiece(vocabulary_size=10, pre_tokenizer=WhitespacePreTokenizer())  # type: ignore[call-arg]


class TestFit:
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_the_alphabet_includes_the_marker(self, algorithm):
        vocabulary = fit_sennrich(algorithm)

        assert list(vocabulary.vocabulary)[:12] == ["[UNK]", *MARKED_ALPHABET]

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_the_smallest_vocabulary_is_the_marked_alphabet_plus_unknown(
        self, algorithm
    ):
        with pytest.raises(VocabularyTooSmallError):
            fit_sennrich(algorithm, vocabulary_size=11)

        assert list(fit_sennrich(algorithm, vocabulary_size=12).vocabulary) == [
            "[UNK]",
            *MARKED_ALPHABET,
        ]

    def test_byte_pair_learns_sennrichs_merges_over_marked_units(self):
        merges = fit_sennrich(SubwordAlgorithm.BYTE_PAIR).merges

        assert merges is not None
        assert [(merge.left, merge.right, int(merge.score)) for merge in merges][
            :5
        ] == (FIRST_MERGES)

    def test_byte_pair_stops_when_no_pair_reaches_the_minimum_frequency(self):
        tokenizer = fit_sennrich(SubwordAlgorithm.BYTE_PAIR, vocabulary_size=30)

        assert tokenizer.vocabulary.n_tokens == 27
        assert tokenizer.n_pruning_rounds == 0

    def test_unigram_lands_on_the_requested_size(self):
        tokenizer = fit_sennrich(SubwordAlgorithm.UNIGRAM, vocabulary_size=30)

        assert tokenizer.vocabulary.n_tokens == 30
        assert tokenizer.n_pruning_rounds == 3

    def test_unigram_makes_every_training_word_one_marked_piece(self):
        vocabulary = fit_sennrich(SubwordAlgorithm.UNIGRAM).vocabulary

        assert list(vocabulary)[12:16] == ["▁newest", "▁low", "▁widest", "▁lower"]

    def test_the_two_algorithms_learn_different_vocabularies(self):
        unigram = set(fit_sennrich(SubwordAlgorithm.UNIGRAM).vocabulary)
        byte_pair = set(fit_sennrich(SubwordAlgorithm.BYTE_PAIR).vocabulary)

        assert len(unigram & byte_pair) == 23
        assert unigram - byte_pair == {"id", "ide", "ides", "de", "des", "ewe", "ewes"}
        assert byte_pair - unigram == {"lo", "low", "newest", "widest"}

    def test_each_algorithm_exposes_only_what_it_learned(self):
        byte_pair = fit_sennrich(SubwordAlgorithm.BYTE_PAIR)
        unigram = fit_sennrich(SubwordAlgorithm.UNIGRAM)

        assert isinstance(byte_pair.merges, Merges) and byte_pair.piece_table is None
        assert isinstance(unigram.piece_table, PieceTable) and unigram.merges is None

    def test_the_default_algorithm_is_unigram(self):
        assert SentencePiece(vocabulary_size=5).algorithm is SubwordAlgorithm.UNIGRAM

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            SentencePiece(vocabulary_size=5).fit("low lower")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            SentencePiece(vocabulary_size=5).fit(["  ", ""])

    def test_fit_returns_self(self):
        tokenizer = SentencePiece(vocabulary_size=30)

        assert tokenizer.fit(SENNRICH_CORPUS) is tokenizer


class TestEncode:
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_the_first_piece_of_every_word_begins_with_the_marker(self, algorithm):
        tokenizer = SentencePiece(vocabulary_size=40, algorithm=algorithm).fit(
            ["hello world", *SENNRICH_CORPUS]
        )

        encoding = tokenizer.encode("hello world")

        assert encoding.texts[0].startswith("▁")
        assert sum(piece.startswith("▁") for piece in encoding.texts) == 2

    def test_unigram_makes_a_word_seen_once_a_whole_piece(self):
        """Byte pair encoding cannot: every pair in ``hello`` is seen once, below
        the minimum pair frequency of two, so it stays ``▁ h e l lo``."""
        unigram = SentencePiece(vocabulary_size=40).fit(
            ["hello world", *SENNRICH_CORPUS]
        )
        byte_pair = SentencePiece(
            vocabulary_size=40, algorithm=SubwordAlgorithm.BYTE_PAIR
        ).fit(["hello world", *SENNRICH_CORPUS])

        assert unigram.encode("hello world").texts == ("▁hello", "▁world")
        assert byte_pair.encode("hello").texts == ("▁", "h", "e", "l", "lo")

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_both_algorithms_spell_lowest_the_same_way(self, algorithm):
        assert fit_sennrich(algorithm).encode("lowest").texts == ("▁low", "est")

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_ids_are_vocabulary_positions(self, algorithm):
        tokenizer = fit_sennrich(algorithm)

        assert tokenizer.encode("lowest").ids == tokenizer.vocabulary.ids_of(
            ("▁low", "est")
        )

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_an_unknown_character_is_unknown_but_the_marker_survives(self, algorithm):
        assert fit_sennrich(algorithm).encode("xyz").texts == (
            "▁",
            "[UNK]",
            "[UNK]",
            "[UNK]",
        )

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_a_blank_text_encodes_to_nothing(self, algorithm):
        assert fit_sennrich(algorithm).encode("   ") == Encoding([])

    def test_unigram_best_segmentation_agrees_with_encode(self):
        tokenizer = fit_sennrich(SubwordAlgorithm.UNIGRAM)

        assert tokenizer.best_segmentation("lowest").pieces == ("▁low", "est")

    def test_byte_pair_has_no_best_segmentation(self):
        with pytest.raises(InvalidValuesError):
            fit_sennrich(SubwordAlgorithm.BYTE_PAIR).best_segmentation("lowest")

    def test_before_fit_raises_not_fitted(self):
        tokenizer = SentencePiece(vocabulary_size=30)

        with pytest.raises(NotFittedError):
            tokenizer.encode("low")
        with pytest.raises(NotFittedError):
            _ = tokenizer.vocabulary
        with pytest.raises(NotFittedError):
            _ = tokenizer.merges
        with pytest.raises(NotFittedError):
            _ = tokenizer.piece_table
        with pytest.raises(NotFittedError):
            _ = tokenizer.n_pruning_rounds
        with pytest.raises(NotFittedError):
            tokenizer.best_segmentation("low")


class TestDecode:
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_round_trips_ordinary_text(self, algorithm):
        tokenizer = fit_sennrich(algorithm)

        assert (
            tokenizer.decode(tokenizer.encode("low lower newest").ids)
            == "low lower newest"
        )
        assert tokenizer.decode(tokenizer.encode("lowest").ids) == "lowest"

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_a_run_of_spaces_decodes_as_one(self, algorithm):
        """Documented rather than hidden: the normalisation collapsed it."""
        tokenizer = fit_sennrich(algorithm)

        assert tokenizer.decode(tokenizer.encode("  low   lower ").ids) == "low lower"

    def test_the_leading_marker_becomes_no_space_at_all(self):
        tokenizer = fit_sennrich(SubwordAlgorithm.UNIGRAM)

        assert tokenizer.decode(tokenizer.vocabulary.ids_of(("▁low",))) == "low"
        assert (
            tokenizer.decode(tokenizer.vocabulary.ids_of(("▁low", "▁low"))) == "low low"
        )


class TestSubwordRegularisation:
    def test_byte_pair_trains_exactly_as_it_predicts(self):
        tokenizer = fit_sennrich(SubwordAlgorithm.BYTE_PAIR)

        assert tokenizer.encode("lowest", PassPurpose.TRAINING) == tokenizer.encode(
            "lowest"
        )

    def test_every_unigram_training_spelling_decodes_to_the_same_text(self):
        tokenizer = fit_sennrich(SubwordAlgorithm.UNIGRAM)

        for _ in range(20):
            encoding = tokenizer.encode("lowest newest", PassPurpose.TRAINING)

            assert tokenizer.decode(encoding.ids) == "lowest newest"

    def test_a_seed_reproduces_the_unigram_draws(self):
        first = SentencePiece(vocabulary_size=30, random_seed=7).fit(SENNRICH_CORPUS)
        second = SentencePiece(vocabulary_size=30, random_seed=7).fit(SENNRICH_CORPUS)

        assert [first.encode("lowest", PassPurpose.TRAINING) for _ in range(10)] == [
            second.encode("lowest", PassPurpose.TRAINING) for _ in range(10)
        ]

    def test_a_high_temperature_recovers_the_best_spelling(self):
        tokenizer = SentencePiece(
            vocabulary_size=30, sampling_temperature=50.0, random_seed=0
        ).fit(SENNRICH_CORPUS)

        assert all(
            tokenizer.encode("lowest", PassPurpose.TRAINING)
            == tokenizer.encode("lowest")
            for _ in range(20)
        )


class TestConstruction:
    @pytest.mark.parametrize(
        "keywords",
        [
            {"vocabulary_size": 1},
            {"vocabulary_size": 10, "whitespace_marker": ""},
            {"vocabulary_size": 10, "unknown_token": ""},
            {"vocabulary_size": 10, "minimum_pair_frequency": 0},
            {"vocabulary_size": 10, "seed_size": 0},
            {"vocabulary_size": 10, "max_piece_length": 0},
            {"vocabulary_size": 10, "shrinking_factor": 1.0},
            {"vocabulary_size": 10, "n_expectation_maximisation_rounds": 0},
            {"vocabulary_size": 10, "sampling_temperature": 0.0},
            {"vocabulary_size": 10, "algorithm": "wordpiece"},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            SentencePiece(**keywords)

    def test_the_algorithm_may_be_named_by_its_string(self):
        assert SentencePiece(vocabulary_size=10, algorithm="byte_pair").algorithm is (  # type: ignore[arg-type]
            SubwordAlgorithm.BYTE_PAIR
        )

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            SentencePiece(vocab_size=10)  # type: ignore[call-arg]

    def test_the_vocabulary_is_a_vocabulary(self):
        assert isinstance(fit_sennrich(SubwordAlgorithm.UNIGRAM).vocabulary, Vocabulary)
