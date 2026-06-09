"""Byte-level tokenization: the 256 byte values are the whole vocabulary.

Nothing to learn, nothing unknown
---------------------------------
Every string is some sequence of UTF-8 bytes, and there are exactly 256 byte
values, so a vocabulary of the 256 of them spells every text that can be
written and needs no corpus to be learned from. Xue et al. (2022) built ByT5 on
exactly that: a transformer reading raw bytes, with no tokenizer to train, to
ship or to go stale when a new script or a new emoji arrives. This class is
that scheme's tokenizer. It is a plain
:class:`~oop_ml.core.natural_language_processing.tokenization.tokenizer.Tokenizer`
rather than a learned one, because a ``fit`` with nothing to fit and a
``NotFittedError`` guarding a table that was complete at import would both be
small lies.

The vocabulary is closed, with no unknown token, for the reason the
:mod:`~oop_ml.core.natural_language_processing.tokenization.vocabulary` module
gives: nothing can be unknown to it, and a fallback it could never legitimately
reach would only hide a bug in the byte table. Token ``b`` is the byte value
``b``, so an id *is* the byte, and the token's text is the byte's one-character
name from
:mod:`~oop_ml.core.natural_language_processing.tokenization.characters.byte_symbols`,
because a token in this library is a printable string and byte ``0x20`` is not
one.

What it costs
-------------
A text is as many tokens as it has bytes, and that is more than characters
outside ASCII: ``é`` is two tokens and ``字`` three, so a Chinese sentence is
three times the tokens of the same characters under a character tokenizer, and
a model's attention cost grows with the square of that. ByT5 pays for its
tokenizer-free coverage with sequences three to four times longer than a
subword model's, which is the trade the byte patching module exists to soften.

Decoding what a model emits
---------------------------
A model chooses ids one at a time and can stop half-way through a multibyte
character, or emit a continuation byte with nothing to continue. Such a run is
not valid UTF-8, and :meth:`ByteTokenizer.decode` gives the replacement
character ``�`` for it rather than raising, because a decoder that refused
would refuse most of a half-finished generation. A lone ``0xA9``, the second
byte of ``é``, decodes to exactly one ``�``.
"""

from __future__ import annotations

from collections.abc import Sequence

from oop_ml.core.natural_language_processing.tokenization.characters.byte_symbols import (  # noqa: E501
    BYTE_SYMBOLS,
    symbols_of_text,
    text_of_symbols,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import Tokenizer
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.network.purpose import PassPurpose

BYTE_VOCABULARY: Vocabulary = Vocabulary(BYTE_SYMBOLS)
"""The 256 byte symbols in byte order, closed. Id ``b`` names byte value ``b``."""


class ByteTokenizer(Tokenizer):
    """One token per UTF-8 byte, against the fixed vocabulary of all 256.

    Takes no parameters, since there is nothing to configure: the table is the
    same for every instance and complete before any text is seen.
    """

    @property
    def vocabulary(self) -> Vocabulary:
        """The 256 byte symbols in byte order; no unknown token."""
        return BYTE_VOCABULARY

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        return symbols_of_text(text)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return text_of_symbols("".join(pieces))
