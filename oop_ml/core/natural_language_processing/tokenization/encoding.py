"""What a text became: its tokens, each paired with the id it goes in as.

A tokenizer's answer is two parallel lists, the pieces and their ids, and two
parallel lists are the shape this library refuses everywhere else -- a caller
has to trust that position three of one still describes position three of the
other. :class:`Token` is the pair, and :class:`Encoding` is the ordered run of
them, with ``ids`` and ``texts`` read off it for the callers that want one side
only. Nothing downstream has to zip anything.

A token's text is the vocabulary's spelling of it, not the source's. A byte
pair piece carries its end-of-word marker, a WordPiece continuation its ``##``,
a SentencePiece piece its ``▁``, and an unknown word appears as the unknown
token itself rather than as the word that was unknown. That is deliberate: the
text is what the id *means*, and decoding is a function of exactly these
strings. The words the text was cut into, with their offsets, are the
:class:`~oop_ml.core.natural_language_processing.tokenization.words.Words` a
pre-tokenizer produced one step earlier.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError


class Token:
    """One piece of an encoded text, with the id a model would read.

    Parameters
    ----------
    text:
        The vocabulary's spelling of the piece. At least one character.
    token_id:
        Its position in the vocabulary. Non-negative.

    Raises
    ------
    EmptyValuesError
        If ``text`` is empty.
    InvalidValuesError
        If ``token_id`` is negative.
    """

    __slots__ = ("_text", "_token_id")

    def __init__(self, text: str, token_id: int) -> None:
        if not isinstance(text, str) or not text:
            raise EmptyValuesError("a token must hold at least one character")

        if token_id < 0:
            raise InvalidValuesError(f"a token id is a position, got {token_id}")

        self._text = text
        self._token_id = int(token_id)

    @property
    def text(self) -> str:
        """The vocabulary's spelling of the piece."""
        return self._text

    @property
    def token_id(self) -> int:
        """Its position in the vocabulary."""
        return self._token_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Token):
            return NotImplemented
        return self._text == other._text and self._token_id == other._token_id

    def __hash__(self) -> int:
        return hash((self._text, self._token_id))

    def __repr__(self) -> str:
        return f"Token({self._text!r}, {self._token_id})"


class Encoding:
    """The tokens one text became, in order.

    Parameters
    ----------
    tokens:
        The tokens, in text order. May be empty: a text with nothing in it that
        the pre-tokenizer counts as a word encodes to nothing, and saying so is
        more honest than inventing a token for it.
    """

    __slots__ = ("_tokens",)

    def __init__(self, tokens: Sequence[Token]) -> None:
        self._tokens = tuple(tokens)

    @property
    def ids(self) -> tuple[int, ...]:
        """The ids alone, in order: what a model reads."""
        return tuple(token.token_id for token in self._tokens)

    @property
    def texts(self) -> tuple[str, ...]:
        """The pieces alone, in order: what a person reads."""
        return tuple(token.text for token in self._tokens)

    @property
    def n_tokens(self) -> int:
        """How many tokens the text became."""
        return len(self._tokens)

    def __iter__(self) -> Iterator[Token]:
        """Iterate the tokens themselves, not their ids or their texts."""
        return iter(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, position: int) -> Token:
        return self._tokens[position]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Encoding):
            return NotImplemented
        return self._tokens == other._tokens

    def __hash__(self) -> int:
        return hash(self._tokens)

    def __repr__(self) -> str:
        return f"Encoding({list(self.texts)!r})"
