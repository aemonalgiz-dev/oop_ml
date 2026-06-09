"""Spec for CharacterTokenizer -- every character the corpus used is a token."""

import pytest
from pydantic import ValidationError

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueTokensError,
    NotFittedError,
    UnknownTokenError,
)
from oop_ml.core.natural_language_processing.tokenization.characters.character import (
    CharacterTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.encoding import (
    Encoding,
    Token,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import (
    LearnedTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.network.purpose import PassPurpose
from test.core.natural_language_processing.fixtures import PROSE_CORPUS

# "cab bad": six distinct characters including the space, in codepoint order.
SMALL_CORPUS = ["cab bad"]
SMALL_ALPHABET = [" ", "a", "b", "c", "d"]


def fit_small() -> CharacterTokenizer:
    return CharacterTokenizer().fit(SMALL_CORPUS)


class TestFit:
    def test_the_vocabulary_is_unknown_then_characters_in_codepoint_order(self):
        vocabulary = fit_small().vocabulary

        assert list(vocabulary) == ["[UNK]", *SMALL_ALPHABET]
        assert vocabulary.unknown_token == "[UNK]"
        assert vocabulary.unknown_id == 0

    def test_the_space_is_a_character_like_any_other(self):
        assert " " in fit_small().vocabulary

    def test_n_characters_excludes_the_unknown_token(self):
        assert fit_small().n_characters == 5
        assert fit_small().vocabulary.n_tokens == 6

    def test_counts_characters_across_every_text(self):
        tokenizer = CharacterTokenizer().fit(["ab", "cd", "  "])

        assert list(tokenizer.vocabulary) == ["[UNK]", " ", "a", "b", "c", "d"]

    def test_text_order_does_not_change_the_vocabulary(self):
        forward = CharacterTokenizer().fit(["ab", "cd"])
        backward = CharacterTokenizer().fit(["dc", "ba"])

        assert forward.vocabulary == backward.vocabulary

    def test_learns_a_prose_corpus(self):
        vocabulary = CharacterTokenizer().fit(PROSE_CORPUS).vocabulary

        assert vocabulary.n_tokens == 1 + len({*"".join(PROSE_CORPUS)})
        assert "T" in vocabulary
        assert "z" in vocabulary
        assert ";" in vocabulary

    def test_a_single_character_unknown_token_the_corpus_uses_is_refused(self):
        with pytest.raises(NonUniqueTokensError):
            CharacterTokenizer(unknown_token="a").fit(SMALL_CORPUS)

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            CharacterTokenizer().fit("cab bad")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            CharacterTokenizer().fit(["  ", ""])

    def test_fit_returns_self(self):
        tokenizer = CharacterTokenizer()

        assert tokenizer.fit(SMALL_CORPUS) is tokenizer

    def test_is_a_learned_tokenizer(self):
        assert isinstance(CharacterTokenizer(), LearnedTokenizer)
        assert isinstance(CharacterTokenizer(), Fittable)
        assert not CharacterTokenizer().is_fitted
        assert fit_small().is_fitted


class TestEncode:
    def test_one_token_per_character(self):
        assert fit_small().encode("bad cab").texts == tuple("bad cab")

    def test_ids_are_vocabulary_positions(self):
        assert fit_small().encode("cab").ids == (4, 2, 3)

    def test_an_unseen_character_becomes_the_unknown_token(self):
        encoding = fit_small().encode("bax")

        assert encoding == Encoding([Token("b", 3), Token("a", 2), Token("[UNK]", 0)])

    def test_a_combining_sequence_is_two_characters(self):
        tokenizer = CharacterTokenizer().fit(["é"])

        assert tokenizer.encode("é").n_tokens == 2
        assert tokenizer.n_characters == 2

    def test_an_empty_text_encodes_to_nothing(self):
        assert fit_small().encode("") == Encoding([])

    def test_the_purpose_changes_nothing(self):
        tokenizer = fit_small()

        assert tokenizer.encode("cab", PassPurpose.TRAINING) == tokenizer.encode("cab")

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            fit_small().encode(["cab"])  # type: ignore[arg-type]

    def test_before_fit_raises_not_fitted(self):
        tokenizer = CharacterTokenizer()

        with pytest.raises(NotFittedError):
            tokenizer.encode("cab")
        with pytest.raises(NotFittedError):
            _ = tokenizer.vocabulary
        with pytest.raises(NotFittedError):
            _ = tokenizer.n_characters
        with pytest.raises(NotFittedError):
            tokenizer.decode([0])


class TestDecode:
    @pytest.mark.parametrize("text", ["cab", "bad cab", "  a  ", "dcba"])
    def test_round_trips_a_seen_text_exactly(self, text):
        tokenizer = fit_small()

        assert tokenizer.decode(tokenizer.encode(text).ids) == text

    def test_an_unseen_character_decodes_to_the_unknown_tokens_spelling(self):
        tokenizer = fit_small()

        assert tokenizer.decode(tokenizer.encode("cax").ids) == "ca[UNK]"

    def test_an_id_no_token_owns_raises(self):
        with pytest.raises(UnknownTokenError):
            fit_small().decode([6])


class TestConstruction:
    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            CharacterTokenizer(vocabulary_size=10)  # type: ignore[call-arg]

    def test_an_empty_unknown_token_is_refused(self):
        with pytest.raises(ValidationError):
            CharacterTokenizer(unknown_token="")

    def test_the_vocabulary_is_a_vocabulary(self):
        assert isinstance(fit_small().vocabulary, Vocabulary)
