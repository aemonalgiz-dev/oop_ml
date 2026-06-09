"""Spec for the PreTokenizer / Tokenizer / LearnedTokenizer templates.

The templates hold the one copy of every lookup rule, so they are tested with
the smallest possible concrete subclasses rather than through a real tokenizer:
a failure here is a failure of the rule, not of any cut or glue.
"""

from collections.abc import Sequence
from typing import Self

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    InvalidValuesError,
    NotFittedError,
    UnknownTokenError,
)
from oop_ml.core.natural_language_processing.tokenization.encoding import (
    Encoding,
    Token,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import (
    LearnedTokenizer,
    PreTokenizer,
    Tokenizer,
    checked_text,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words
from oop_ml.core.network.purpose import PassPurpose


class CharacterSplitter(PreTokenizer):
    def _words_of(self, text: str) -> Words:
        return Words(
            [Word.of(text, position, position + 1) for position in range(len(text))]
        )


class CharacterTokens(Tokenizer):
    """Every character is a piece; the vocabulary is whatever it was given."""

    tokens: tuple[str, ...]
    unknown_token: str | None = None

    @property
    def vocabulary(self) -> Vocabulary:
        return Vocabulary(list(self.tokens), unknown_token=self.unknown_token)

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        return tuple(text)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return "".join(pieces)


class PurposeSpy(CharacterTokens):
    """Reports which purpose reached the cut."""

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        return (str(purpose),)


class LearnsItsAlphabet(LearnedTokenizer):
    def fit(self, corpus: Sequence[str]) -> Self:
        self._alphabet = sorted({character for text in corpus for character in text})
        self._mark_fitted()
        return self

    @property
    def vocabulary(self) -> Vocabulary:
        self._check_fitted()
        return Vocabulary(self._alphabet)

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        self._check_fitted()
        return tuple(text)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return "".join(pieces)


class TestCheckedText:
    def test_passes_a_string_through(self):
        assert checked_text("fox") == "fox"

    @pytest.mark.parametrize("not_text", [b"fox", ["fox"], 3, None])
    def test_refuses_anything_else(self, not_text):
        with pytest.raises(InvalidValuesError):
            checked_text(not_text)


class TestPreTokenizerTemplate:
    def test_split_delegates_to_the_rule(self):
        assert CharacterSplitter().split("ab").texts == ("a", "b")

    def test_split_refuses_a_non_string(self):
        with pytest.raises(InvalidValuesError):
            CharacterSplitter().split(["ab"])  # type: ignore[arg-type]

    def test_an_unknown_keyword_is_refused_at_construction(self):
        with pytest.raises(ValidationError):
            CharacterSplitter(pattern="x")  # type: ignore[call-arg]

    def test_the_base_cannot_be_constructed(self):
        with pytest.raises(TypeError):
            PreTokenizer()  # type: ignore[abstract]


class TestEncodeTemplate:
    def test_pairs_each_piece_with_its_id(self):
        tokenizer = CharacterTokens(tokens=("a", "b"))

        assert tokenizer.encode("ba") == Encoding([Token("b", 1), Token("a", 0)])

    def test_an_unknown_piece_becomes_the_unknown_token_text_and_id(self):
        tokenizer = CharacterTokens(tokens=("[UNK]", "a"), unknown_token="[UNK]")

        assert tokenizer.encode("az") == Encoding([Token("a", 1), Token("[UNK]", 0)])

    def test_an_unknown_piece_against_a_closed_vocabulary_raises(self):
        with pytest.raises(UnknownTokenError):
            CharacterTokens(tokens=("a",)).encode("az")

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            CharacterTokens(tokens=("a",)).encode(3)  # type: ignore[arg-type]

    def test_the_default_purpose_is_predicting(self):
        spy = PurposeSpy(tokens=("predicting", "training"))

        assert spy.encode("anything").texts == ("predicting",)

    def test_the_purpose_reaches_the_cut(self):
        spy = PurposeSpy(tokens=("predicting", "training"))

        assert spy.encode("anything", PassPurpose.TRAINING).texts == ("training",)


class TestDecodeTemplate:
    def test_glues_the_tokens_the_ids_name(self):
        assert CharacterTokens(tokens=("a", "b")).decode([1, 0, 1]) == "bab"

    def test_an_id_no_token_owns_raises(self):
        with pytest.raises(UnknownTokenError):
            CharacterTokens(tokens=("a", "b")).decode([2])

    def test_round_trips_an_encoding(self):
        tokenizer = CharacterTokens(tokens=("a", "b"))

        assert tokenizer.decode(tokenizer.encode("abba").ids) == "abba"


class TestLearnedTokenizer:
    def test_is_not_fitted_until_fit(self):
        tokenizer = LearnsItsAlphabet()

        assert not tokenizer.is_fitted
        assert tokenizer.fit(["ba"]).is_fitted

    def test_encode_before_fit_raises_not_fitted(self):
        with pytest.raises(NotFittedError):
            LearnsItsAlphabet().encode("a")

    def test_vocabulary_before_fit_raises_not_fitted(self):
        with pytest.raises(NotFittedError):
            _ = LearnsItsAlphabet().vocabulary

    def test_decode_before_fit_raises_not_fitted(self):
        with pytest.raises(NotFittedError):
            LearnsItsAlphabet().decode([0])

    def test_fit_returns_self_so_calls_chain(self):
        tokenizer = LearnsItsAlphabet()

        assert tokenizer.fit(["ba"]) is tokenizer

    def test_encodes_after_fit(self):
        assert LearnsItsAlphabet().fit(["ba"]).encode("ab").ids == (0, 1)

    def test_an_unknown_keyword_is_refused_at_construction(self):
        with pytest.raises(ValidationError):
            LearnsItsAlphabet(vocabulary_size=3)  # type: ignore[call-arg]
