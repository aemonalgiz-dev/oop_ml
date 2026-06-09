"""Spec for Vocabulary -- the closed, ordered set of tokens a model can see."""

import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueTokensError,
    UnknownTokenError,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary


def make_open_vocabulary() -> Vocabulary:
    return Vocabulary(["[UNK]", "the", "fox"], unknown_token="[UNK]")


def make_closed_vocabulary() -> Vocabulary:
    return Vocabulary(["the", "fox"])


class TestLookups:
    def test_ids_are_positions(self):
        vocabulary = make_open_vocabulary()

        assert vocabulary.id_of("the") == 1
        assert vocabulary.id_of("fox") == 2
        assert vocabulary["fox"] == 2

    def test_tokens_are_read_back_by_position(self):
        assert make_open_vocabulary().token_of(2) == "fox"

    def test_several_at_once_in_order(self):
        vocabulary = make_open_vocabulary()

        assert vocabulary.ids_of(["fox", "the"]) == (2, 1)
        assert vocabulary.tokens_of([2, 1]) == ("fox", "the")

    def test_an_unknown_token_falls_back_to_the_unknown_id(self):
        vocabulary = make_open_vocabulary()

        assert vocabulary.id_of("cat") == vocabulary.unknown_id == 0

    def test_a_closed_vocabulary_refuses_an_unknown_token(self):
        with pytest.raises(UnknownTokenError):
            make_closed_vocabulary().id_of("cat")

    @pytest.mark.parametrize("token_id", [-1, 3])
    def test_an_id_no_token_owns_raises(self, token_id):
        with pytest.raises(UnknownTokenError):
            make_open_vocabulary().token_of(token_id)

    def test_membership_is_exact(self):
        vocabulary = make_open_vocabulary()

        assert "fox" in vocabulary
        assert "cat" not in vocabulary

    def test_iterates_tokens_in_id_order(self):
        assert list(make_open_vocabulary()) == ["[UNK]", "the", "fox"]

    def test_counts_its_tokens(self):
        vocabulary = make_open_vocabulary()

        assert vocabulary.n_tokens == 3
        assert len(vocabulary) == 3


class TestUnknownToken:
    def test_reports_the_unknown_token_and_its_id(self):
        vocabulary = make_open_vocabulary()

        assert vocabulary.has_unknown
        assert vocabulary.unknown_token == "[UNK]"
        assert vocabulary.unknown_id == 0

    def test_a_closed_vocabulary_has_none(self):
        vocabulary = make_closed_vocabulary()

        assert not vocabulary.has_unknown
        assert vocabulary.unknown_token is None
        assert vocabulary.unknown_id is None

    def test_the_unknown_token_must_itself_be_a_token(self):
        with pytest.raises(InvalidValuesError):
            Vocabulary(["the", "fox"], unknown_token="[UNK]")


class TestConstruction:
    def test_empty_raises(self):
        with pytest.raises(EmptyValuesError):
            Vocabulary([])

    def test_a_repeated_token_raises(self):
        with pytest.raises(NonUniqueTokensError):
            Vocabulary(["the", "fox", "the"])

    @pytest.mark.parametrize("token", ["", 3])
    def test_a_token_must_be_a_non_empty_string(self, token):
        with pytest.raises(InvalidValuesError):
            Vocabulary(["the", token])  # type: ignore[list-item]

    def test_equal_when_tokens_and_unknown_match(self):
        assert make_open_vocabulary() == make_open_vocabulary()
        assert hash(make_open_vocabulary()) == hash(make_open_vocabulary())

    def test_unequal_when_the_order_differs(self):
        """Order is the id assignment, so a reordering is a different table."""
        assert Vocabulary(["the", "fox"]) != Vocabulary(["fox", "the"])

    def test_unequal_when_only_the_unknown_differs(self):
        assert Vocabulary(["a", "b"], unknown_token="a") != Vocabulary(["a", "b"])

    def test_compares_unequal_to_a_bare_list(self):
        assert make_closed_vocabulary() != ["the", "fox"]
