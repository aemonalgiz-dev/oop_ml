"""Spec for ByteTokenizer -- ByT5's scheme, the 256 byte values as the vocabulary."""

import pytest
from pydantic import ValidationError

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.exceptions import InvalidValuesError, UnknownTokenError
from oop_ml.core.natural_language_processing.tokenization.characters.byte import (
    BYTE_VOCABULARY,
    ByteTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.characters.byte_symbols import (
    BYTE_SYMBOLS,
)
from oop_ml.core.natural_language_processing.tokenization.encoding import Encoding
from oop_ml.core.natural_language_processing.tokenization.tokenizer import Tokenizer
from oop_ml.core.network.purpose import PassPurpose


class TestTheVocabulary:
    def test_holds_exactly_the_256_byte_symbols_in_byte_order(self):
        vocabulary = ByteTokenizer().vocabulary

        assert vocabulary.n_tokens == 256
        assert tuple(vocabulary) == BYTE_SYMBOLS
        assert vocabulary is BYTE_VOCABULARY

    def test_is_closed_with_no_unknown_token(self):
        assert ByteTokenizer().vocabulary.unknown_token is None
        assert not ByteTokenizer().vocabulary.has_unknown

    def test_the_id_of_a_symbol_is_its_byte_value(self):
        vocabulary = ByteTokenizer().vocabulary

        assert all(
            vocabulary.id_of(BYTE_SYMBOLS[value]) == value for value in range(256)
        )

    def test_is_a_plain_tokenizer_with_nothing_to_fit(self):
        tokenizer = ByteTokenizer()

        assert isinstance(tokenizer, Tokenizer)
        assert not isinstance(tokenizer, Fittable)
        assert not hasattr(tokenizer, "fit")


class TestEncode:
    def test_ids_are_the_utf8_byte_values(self):
        assert ByteTokenizer().encode("hello").ids == tuple(b"hello")

    def test_ascii_text_is_spelled_as_itself(self):
        assert ByteTokenizer().encode("hello").texts == ("h", "e", "l", "l", "o")

    def test_a_space_is_spelled_by_its_gpt2_name(self):
        assert ByteTokenizer().encode(" a").texts == ("Ġ", "a")

    def test_a_two_byte_character_is_two_tokens(self):
        encoding = ByteTokenizer().encode("é")

        assert encoding.n_tokens == 2
        assert encoding.ids == (0xC3, 0xA9)

    def test_a_three_byte_character_is_three_tokens(self):
        encoding = ByteTokenizer().encode("字")

        assert encoding.n_tokens == 3
        assert encoding.ids == (0xE5, 0xAD, 0x97)

    def test_a_four_byte_character_is_four_tokens(self):
        assert ByteTokenizer().encode("😀").n_tokens == 4

    def test_an_empty_text_encodes_to_nothing(self):
        assert ByteTokenizer().encode("") == Encoding([])

    def test_the_purpose_changes_nothing(self):
        tokenizer = ByteTokenizer()

        assert tokenizer.encode("naïve", PassPurpose.TRAINING) == tokenizer.encode(
            "naïve"
        )

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            ByteTokenizer().encode(b"hello")  # type: ignore[arg-type]

    def test_nothing_is_ever_unknown(self):
        """Every codepoint from every plane spells in bytes the table names."""
        text = "".join(
            chr(codepoint)
            for codepoint in (0, 0x7F, 0x80, 0x7FF, 0x800, 0xFFFF, 0x10000, 0x10FFFF)
        )
        encoding = ByteTokenizer().encode(text)

        assert encoding.n_tokens == len(text.encode("utf-8"))
        assert all(0 <= token_id < 256 for token_id in encoding.ids)


class TestDecode:
    @pytest.mark.parametrize(
        "text", ["hello world", "naïve café", "日本語", "😀 emoji", "\t\n\x00", ""]
    )
    def test_round_trips_exactly(self, text):
        tokenizer = ByteTokenizer()

        assert tokenizer.decode(tokenizer.encode(text).ids) == text

    def test_decodes_raw_byte_values_as_ids(self):
        assert ByteTokenizer().decode(list(b"bytes")) == "bytes"

    def test_a_lone_continuation_byte_becomes_the_replacement_character(self):
        assert ByteTokenizer().decode([0xA9]) == "�"

    def test_a_character_cut_short_becomes_one_replacement_character(self):
        first_byte_of_e_acute = ByteTokenizer().encode("é").ids[0]

        assert ByteTokenizer().decode([first_byte_of_e_acute]) == "�"

    def test_valid_bytes_around_an_invalid_one_survive(self):
        assert ByteTokenizer().decode([*b"a", 0xA9, *b"b"]) == "a�b"

    def test_an_id_past_the_table_raises(self):
        with pytest.raises(UnknownTokenError):
            ByteTokenizer().decode([256])

    def test_a_negative_id_raises(self):
        with pytest.raises(UnknownTokenError):
            ByteTokenizer().decode([-1])


class TestConstruction:
    def test_takes_no_parameters(self):
        with pytest.raises(ValidationError):
            ByteTokenizer(vocabulary_size=256)  # type: ignore[call-arg]

    def test_every_instance_is_the_same_tokenizer(self):
        assert ByteTokenizer() == ByteTokenizer()
        assert ByteTokenizer().vocabulary == ByteTokenizer().vocabulary
