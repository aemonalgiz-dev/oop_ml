"""Spec for the byte-to-symbol table -- GPT-2's names for the 256 byte values."""

import pytest

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.characters.byte_symbols import (
    BYTE_SYMBOLS,
    byte_of_symbol,
    symbols_of_text,
    text_of_symbols,
)


class TestTheTable:
    def test_names_every_byte_value_once(self):
        assert len(BYTE_SYMBOLS) == 256
        assert len(set(BYTE_SYMBOLS)) == 256

    def test_every_name_is_one_character(self):
        assert all(len(symbol) == 1 for symbol in BYTE_SYMBOLS)

    def test_printable_latin_1_stands_for_itself(self):
        assert BYTE_SYMBOLS[ord("h")] == "h"
        assert BYTE_SYMBOLS[ord("!")] == "!"
        assert BYTE_SYMBOLS[ord("~")] == "~"
        assert BYTE_SYMBOLS[0xA1] == "¡"
        assert BYTE_SYMBOLS[0xFF] == "ÿ"

    def test_the_space_is_gpt2s_capital_g_with_dot(self):
        assert BYTE_SYMBOLS[ord(" ")] == "Ġ"

    def test_the_first_control_byte_takes_codepoint_256(self):
        assert BYTE_SYMBOLS[0] == chr(256)

    def test_exactly_68_values_are_renamed(self):
        renamed = [value for value in range(256) if ord(BYTE_SYMBOLS[value]) != value]

        assert len(renamed) == 68
        assert [ord(BYTE_SYMBOLS[value]) for value in renamed] == list(range(256, 324))

    def test_no_name_is_whitespace(self):
        assert not any(symbol.isspace() for symbol in BYTE_SYMBOLS)


class TestRoundTrips:
    def test_ascii_text_is_spelled_as_itself(self):
        assert symbols_of_text("hello") == ("h", "e", "l", "l", "o")

    def test_a_space_becomes_its_name(self):
        assert symbols_of_text(" a") == ("Ġ", "a")

    def test_a_multibyte_character_is_several_symbols(self):
        assert len(symbols_of_text("é")) == 2
        assert len(symbols_of_text("字")) == 3

    @pytest.mark.parametrize(
        "text", ["hello world", "naïve café", "日本語", "\t\n\x00"]
    )
    def test_text_round_trips_exactly(self, text):
        assert text_of_symbols("".join(symbols_of_text(text))) == text

    def test_byte_of_symbol_inverts_the_table(self):
        assert all(byte_of_symbol(BYTE_SYMBOLS[value]) == value for value in range(256))

    def test_an_unnamed_character_is_refused(self):
        with pytest.raises(InvalidValuesError):
            byte_of_symbol("字")

    def test_an_incomplete_multibyte_run_becomes_the_replacement_character(self):
        """A model choosing ids one at a time can stop half-way through a character."""
        first_byte_only = symbols_of_text("é")[0]

        assert text_of_symbols(first_byte_only) == "�"
