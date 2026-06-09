"""Spec for byte fallback, which is what makes "no unknown pieces" true.

Byte pair encoding is usually sold as having no out-of-vocabulary problem,
because any word it has never seen still decomposes into characters it has.
That is true of unseen *words* and false of unseen *characters*: a character
absent from the training corpus has no row, and the closed vocabulary has
nothing to answer with but the unknown token. One accented letter, one
alphabet the corpus did not contain, and a piece of the text is simply lost,
which the round trip then shows as corrupted text rather than as an error.

Byte fallback is SentencePiece's ``--byte_fallback``, the option Llama and
Gemma are trained with. Every character has a UTF-8 spelling, so a character
with no row is emitted as the rows of its bytes, and 256 extra rows make an
unknown piece unrepresentable rather than unlikely.

The corpus below is deliberately narrow, so that ordinary text lands outside
it in three different ways: a capital letter, a letter of another script, and
an accented letter.
"""

import pytest

from oop_ml.core.exceptions import VocabularyTooSmallError
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)

CORPUS = Corpus(
    [
        "the report was expected on monday",
        "the costs were lower than the first estimate",
        "we reviewed the results and rewrote the summary",
    ]
)

# 'z' never occurs, nor a capital, nor an accent, nor any Greek.
OUTSIDE = "Dr. Alvarez wrote naïve prose in ελληνικά"


def fitted(byte_fallback: bool, vocabulary_size: int = 320) -> BytePairEncoding:
    return BytePairEncoding(
        vocabulary_size=vocabulary_size, byte_fallback=byte_fallback
    ).fit(CORPUS)


class TestTheDefault:
    def test_is_on(self):
        """Because a silent loss is the wrong thing to make a caller opt out of."""
        assert BytePairEncoding(vocabulary_size=320).byte_fallback is True


class TestWithoutIt:
    """Sennrich's method exactly, pinned so the repair is visible against it."""

    def test_an_unseen_character_becomes_the_unknown_token(self):
        model = fitted(byte_fallback=False, vocabulary_size=60)
        encoded = model.encode(OUTSIDE)

        assert model.vocabulary.unknown_token in encoded.texts

    def test_and_the_round_trip_silently_loses_it(self):
        model = fitted(byte_fallback=False, vocabulary_size=60)

        assert model.decode(model.encode(OUTSIDE).ids) != OUTSIDE


class TestWithIt:
    def test_no_piece_is_unknown(self):
        model = fitted(byte_fallback=True)
        encoded = model.encode(OUTSIDE)

        assert model.vocabulary.unknown_token not in encoded.texts

    def test_the_round_trip_is_exact(self):
        model = fitted(byte_fallback=True)

        assert model.decode(model.encode(OUTSIDE).ids) == OUTSIDE

    @pytest.mark.parametrize(
        "text",
        [
            "naïve",
            "ελληνικά",
            "Dr. Alvarez",
            "\U0001f600",
            "the report was expected on monday",
        ],
    )
    def test_every_text_survives_the_round_trip(self, text):
        model = fitted(byte_fallback=True)

        assert model.decode(model.encode(text).ids) == text

    def test_the_vocabulary_gains_a_row_for_every_byte(self):
        with_fallback = fitted(byte_fallback=True)
        without = fitted(byte_fallback=False)

        assert len(with_fallback.vocabulary) - len(without.vocabulary) == 257

    def test_a_character_the_corpus_knows_is_still_spelled_as_itself(self):
        """Fallback is for what has no row, and must not replace what has one."""
        model = fitted(byte_fallback=True)

        assert "<0x74>" not in model.encode("the").texts

    def test_the_byte_rows_do_not_eat_the_vocabulary_budget(self):
        """The accounting that lets the default be on.

        SentencePiece counts its byte rows against the requested size, which
        would put a floor of 257 under every vocabulary here and make a small
        one impossible to ask for. The rows are a floor instead, so the number
        of merges a given size buys is the same either way.
        """
        with_fallback = fitted(byte_fallback=True, vocabulary_size=40)
        without = fitted(byte_fallback=False, vocabulary_size=40)

        assert with_fallback.n_merges == without.n_merges

    def test_a_vocabulary_smaller_than_the_byte_rows_still_fits(self):
        model = fitted(byte_fallback=True, vocabulary_size=40)

        assert model.decode(model.encode(OUTSIDE).ids) == OUTSIDE

    def test_a_size_below_the_alphabet_is_still_refused(self):
        with pytest.raises(VocabularyTooSmallError, match="smallest workable"):
            BytePairEncoding(vocabulary_size=3, byte_fallback=True).fit(CORPUS)
