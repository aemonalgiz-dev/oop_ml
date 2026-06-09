"""Spec for Token / Encoding -- pieces paired with the ids a model reads."""

import pytest

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.encoding import (
    Encoding,
    Token,
)


class TestToken:
    def test_carries_text_and_id(self):
        token = Token("est</w>", 13)

        assert token.text == "est</w>"
        assert token.token_id == 13

    def test_empty_text_raises(self):
        with pytest.raises(EmptyValuesError):
            Token("", 0)

    def test_a_negative_id_raises(self):
        with pytest.raises(InvalidValuesError):
            Token("est", -1)

    def test_equal_when_text_and_id_match(self):
        assert Token("est", 3) == Token("est", 3)
        assert hash(Token("est", 3)) == hash(Token("est", 3))

    def test_unequal_when_the_id_differs(self):
        assert Token("est", 3) != Token("est", 4)

    def test_compares_unequal_to_a_bare_string(self):
        assert Token("est", 3) != "est"


class TestEncoding:
    def test_reads_ids_and_texts_off_in_order(self):
        encoding = Encoding([Token("low", 23), Token("est</w>", 13)])

        assert encoding.ids == (23, 13)
        assert encoding.texts == ("low", "est</w>")

    def test_iterates_token_objects(self):
        encoding = Encoding([Token("low", 23), Token("est</w>", 13)])

        assert list(encoding) == [Token("low", 23), Token("est</w>", 13)]

    def test_counts_and_indexes(self):
        encoding = Encoding([Token("low", 23), Token("est</w>", 13)])

        assert len(encoding) == 2
        assert encoding.n_tokens == 2
        assert encoding[1] == Token("est</w>", 13)

    def test_may_be_empty(self):
        assert Encoding([]).ids == ()
        assert len(Encoding([])) == 0

    def test_equal_when_every_token_matches(self):
        assert Encoding([Token("a", 1)]) == Encoding([Token("a", 1)])
        assert hash(Encoding([Token("a", 1)])) == hash(Encoding([Token("a", 1)]))

    def test_compares_unequal_to_a_bare_tuple(self):
        assert Encoding([Token("a", 1)]) != (1,)
