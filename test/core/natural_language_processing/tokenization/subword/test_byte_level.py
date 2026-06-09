"""Spec for ByteLevelBytePairEncoding -- GPT-2's alphabet of 256 byte symbols.

Pinned against Sennrich's toy corpus read as one text, so that the first ``low``
has no space before it and the other four do. The merges are worked by hand in
the module docstring.
"""

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
    VocabularyTooSmallError,
)
from oop_ml.core.natural_language_processing.tokenization.characters.byte_symbols import (
    BYTE_SYMBOLS,
)
from oop_ml.core.natural_language_processing.tokenization.encoding import Encoding
from oop_ml.core.natural_language_processing.tokenization.subword.byte_level import (
    N_BYTE_VALUES,
    ByteLevelBytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.merges import Merges
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.network.purpose import PassPurpose
from test.core.natural_language_processing.fixtures import SENNRICH_CORPUS

SPACE = BYTE_SYMBOLS[ord(" ")]

# The first ten merges on the corpus as one text, with the count that chose each.
# ASCII letters name themselves, so within a word the pairs are Sennrich's;
# ``low`` opens the text once without a space and ``Ġlow`` follows four times.
BYTE_LEVEL_FIRST_TEN_MERGES: list[tuple[str, str, int]] = [
    ("e", "s", 9),
    ("es", "t", 9),
    ("l", "o", 7),
    ("lo", "w", 7),
    ("e", "w", 6),
    ("ew", "est", 6),
    ("n", "ewest", 6),
    (SPACE, "low", 6),
    (SPACE, "newest", 6),
    ("d", "est", 3),
]

TEN_MERGES = N_BYTE_VALUES + 10


def fit_ten_merges() -> ByteLevelBytePairEncoding:
    return ByteLevelBytePairEncoding(vocabulary_size=TEN_MERGES).fit(SENNRICH_CORPUS)


class TestFit:
    def test_learns_the_merges_in_order(self):
        learned = [
            (merge.left, merge.right, int(merge.score))
            for merge in fit_ten_merges().merges
        ]

        assert learned == BYTE_LEVEL_FIRST_TEN_MERGES

    def test_the_vocabulary_is_every_byte_symbol_then_the_merges(self):
        vocabulary = fit_ten_merges().vocabulary

        assert list(vocabulary)[:256] == list(BYTE_SYMBOLS)
        assert list(vocabulary)[256:] == [
            left + right for left, right, _ in BYTE_LEVEL_FIRST_TEN_MERGES
        ]

    def test_the_vocabulary_has_no_unknown_token(self):
        vocabulary = fit_ten_merges().vocabulary

        assert vocabulary.unknown_token is None
        assert not vocabulary.has_unknown

    def test_a_byte_symbols_id_is_its_byte_value(self):
        vocabulary = fit_ten_merges().vocabulary

        assert vocabulary.id_of("a") == ord("a")
        assert vocabulary.id_of(SPACE) == 32
        assert vocabulary.token_of(255) == BYTE_SYMBOLS[255]

    def test_reaches_the_requested_size_exactly(self):
        assert fit_ten_merges().vocabulary.n_tokens == TEN_MERGES
        assert fit_ten_merges().n_merges == 10

    def test_a_vocabulary_below_the_byte_alphabet_is_refused_at_fit(self):
        with pytest.raises(VocabularyTooSmallError):
            ByteLevelBytePairEncoding(vocabulary_size=255).fit(SENNRICH_CORPUS)

    def test_exactly_the_byte_alphabet_learns_no_merges(self):
        tokenizer = ByteLevelBytePairEncoding(vocabulary_size=256).fit(SENNRICH_CORPUS)

        assert tokenizer.merges == Merges([])
        assert tokenizer.encode("low").texts == ("l", "o", "w")

    def test_stops_when_no_pair_reaches_the_minimum_frequency(self):
        """After the ninth merge every remaining pair scores 3 or less."""
        tokenizer = ByteLevelBytePairEncoding(
            vocabulary_size=1000, minimum_pair_frequency=4
        ).fit(SENNRICH_CORPUS)

        assert tokenizer.n_merges == 9

    def test_the_space_makes_a_word_at_the_head_of_a_text_a_different_word(self):
        """``low`` once and ``Ġlow`` four times, where byte pair encoding sees five."""
        tokenizer = ByteLevelBytePairEncoding(
            vocabulary_size=1000, minimum_pair_frequency=1
        ).fit(SENNRICH_CORPUS)

        assert "low" in tokenizer.vocabulary
        assert SPACE + "low" in tokenizer.vocabulary

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            ByteLevelBytePairEncoding(vocabulary_size=300).fit("low lower")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            ByteLevelBytePairEncoding(vocabulary_size=300).fit(["  ", ""])

    def test_fit_returns_self(self):
        tokenizer = ByteLevelBytePairEncoding(vocabulary_size=TEN_MERGES)

        assert tokenizer.fit(SENNRICH_CORPUS) is tokenizer


class TestEncode:
    def test_a_word_never_seen_is_spelled_from_learned_pieces(self):
        assert fit_ten_merges().encode("low lowest").texts == (
            "low",
            SPACE + "low",
            "est",
        )

    def test_the_space_before_a_word_belongs_to_it(self):
        assert fit_ten_merges().encode("low lower").texts == (
            "low",
            SPACE + "low",
            "e",
            "r",
        )

    def test_the_second_word_of_a_pair_starts_with_the_space_symbol(self):
        assert fit_ten_merges().encode("a b").texts[1].startswith(SPACE)

    def test_every_space_before_a_word_is_kept(self):
        tokenizer = fit_ten_merges()

        assert tokenizer.encode("a  b").texts == ("a", SPACE, SPACE, "b")
        assert tokenizer.encode(" a").texts == (SPACE, "a")

    def test_a_character_outside_the_corpus_is_its_bytes_not_an_error(self):
        """``日`` is three UTF-8 bytes, and every byte has a symbol."""
        encoding = fit_ten_merges().encode("日")

        assert encoding.n_tokens == 3
        assert all(token_id < 256 for token_id in encoding.ids)

    def test_ids_are_vocabulary_positions(self):
        tokenizer = fit_ten_merges()

        assert tokenizer.encode("low lowest").ids == tokenizer.vocabulary.ids_of(
            ("low", SPACE + "low", "est")
        )

    def test_a_blank_text_encodes_to_nothing(self):
        assert fit_ten_merges().encode("   ") == Encoding([])

    def test_before_fit_raises_not_fitted(self):
        tokenizer = ByteLevelBytePairEncoding(vocabulary_size=TEN_MERGES)

        with pytest.raises(NotFittedError):
            tokenizer.encode("low")
        with pytest.raises(NotFittedError):
            _ = tokenizer.vocabulary
        with pytest.raises(NotFittedError):
            _ = tokenizer.merges


class TestDecode:
    @pytest.mark.parametrize(
        "text",
        [
            "hello world",
            "low lowest",
            "café crème",
            "日本語 テキスト",
            "  leading spaces",
            "tabs\tand\nnewlines between",
            "a  b",
        ],
    )
    def test_round_trips_text_the_corpus_never_held(self, text):
        tokenizer = fit_ten_merges()

        assert tokenizer.decode(tokenizer.encode(text).ids) == text

    def test_whitespace_after_the_last_word_is_dropped(self):
        tokenizer = fit_ten_merges()

        assert tokenizer.decode(tokenizer.encode("hello ").ids) == "hello"
        assert tokenizer.decode(tokenizer.encode("hello\n").ids) == "hello"

    def test_a_multibyte_character_cut_between_its_bytes_decodes_to_the_replacement(
        self,
    ):
        """A model choosing ids one at a time can emit half a character."""
        tokenizer = fit_ten_merges()
        first_byte_only = tokenizer.encode("日").ids[:1]

        assert tokenizer.decode(first_byte_only) == "�"


class TestMergeDropout:
    def test_predicting_ignores_the_dropout(self):
        plain = fit_ten_merges()
        dropped = ByteLevelBytePairEncoding(
            vocabulary_size=TEN_MERGES, merge_dropout=0.9, random_seed=0
        ).fit(SENNRICH_CORPUS)

        assert dropped.encode("newest") == plain.encode("newest")

    def test_training_skips_merges_so_the_word_arrives_in_more_pieces(self):
        dropped = ByteLevelBytePairEncoding(
            vocabulary_size=TEN_MERGES, merge_dropout=0.9, random_seed=0
        ).fit(SENNRICH_CORPUS)

        lengths = {
            len(dropped.encode("newest", PassPurpose.TRAINING)) for _ in range(20)
        }

        assert len(dropped.encode("newest")) == 1
        assert max(lengths) > 1

    def test_every_training_spelling_decodes_to_the_same_text(self):
        dropped = ByteLevelBytePairEncoding(
            vocabulary_size=TEN_MERGES, merge_dropout=0.5, random_seed=3
        ).fit(SENNRICH_CORPUS)

        for _ in range(20):
            encoding = dropped.encode("newest lower", PassPurpose.TRAINING)
            assert dropped.decode(encoding.ids) == "newest lower"

    def test_a_seed_reproduces_the_sequence_of_spellings(self):
        first = ByteLevelBytePairEncoding(
            vocabulary_size=TEN_MERGES, merge_dropout=0.5, random_seed=7
        ).fit(SENNRICH_CORPUS)
        second = ByteLevelBytePairEncoding(
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
            {"vocabulary_size": 0},
            {"vocabulary_size": 300, "merge_dropout": 1.0},
            {"vocabulary_size": 300, "merge_dropout": -0.1},
            {"vocabulary_size": 300, "minimum_pair_frequency": 0},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            ByteLevelBytePairEncoding(**keywords)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            ByteLevelBytePairEncoding(vocab_size=300)  # type: ignore[call-arg]

    def test_there_is_no_unknown_token_to_configure(self):
        with pytest.raises(ValidationError):
            ByteLevelBytePairEncoding(vocabulary_size=300, unknown_token="[UNK]")  # type: ignore[call-arg]

    def test_there_is_no_end_of_word_marker_to_configure(self):
        with pytest.raises(ValidationError):
            ByteLevelBytePairEncoding(vocabulary_size=300, end_of_word_marker="</w>")  # type: ignore[call-arg]

    def test_the_vocabulary_is_a_vocabulary(self):
        assert isinstance(fit_ten_merges().vocabulary, Vocabulary)
