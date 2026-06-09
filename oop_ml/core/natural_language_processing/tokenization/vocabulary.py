"""The closed set of tokens a model can see, each bound to the id it goes in as.

What a vocabulary is
--------------------
A neural model reads numbers, and a tokenizer's whole job ends in a lookup
table: token in, whole number out. That table is what this class is. It is
*closed* in the sense that once built it has a fixed size and a fixed order,
because the size becomes the width of an embedding table and the order decides
which row of that table each token owns. Change either after the fact and
every model trained against it reads the wrong rows.

The unknown token, and who needs one
------------------------------------
A word-level or subword vocabulary can always meet a symbol it has never seen
-- a character from a script absent from the training corpus is enough -- and
the honest answer is a single reserved token meaning "something else". A
vocabulary built with an ``unknown_token`` hands that token's id back for any
lookup it cannot satisfy.

A vocabulary built without one refuses instead, with
:class:`~oop_ml.core.exceptions.UnknownTokenError`. That is not a gap. The
byte-level vocabularies in this package are closed by construction, since every
string is some sequence of the 256 byte values, so nothing is ever unknown to
them and a fallback would be a lie about their coverage. Giving them one
anyway would hide a real bug: a byte-level tokenizer that ever reached its
unknown token would be a tokenizer whose byte mapping had gone wrong.

Ids are positions
-----------------
Token ``i`` is whichever token the sequence held at position ``i``. There is no
separate id table to keep in step with the token list, which is the same
argument :func:`~oop_ml.core.validation.check_is_label_encoded` makes for class
positions: with one list there is nothing that can drift.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueTokensError,
    UnknownTokenError,
)


class Vocabulary:
    """A fixed, ordered set of tokens, each addressable by its position.

    Parameters
    ----------
    tokens:
        Every token, in id order. Position ``i`` is id ``i``. Non-empty, every
        entry a non-empty string, no repeats.
    unknown_token:
        The token any unknown lookup falls back to. Must itself be among
        ``tokens``, so its id is a position like any other. ``None`` means the
        vocabulary is closed and an unknown lookup is an error.

    Raises
    ------
    EmptyValuesError
        If no tokens are supplied.
    InvalidValuesError
        If a token is not a non-empty string, or the unknown token is not
        among the tokens.
    NonUniqueTokensError
        If a token appears twice.
    """

    __slots__ = ("_ids_by_token", "_tokens", "_unknown_token")

    def __init__(self, tokens: Sequence[str], unknown_token: str | None = None) -> None:
        if len(tokens) == 0:
            raise EmptyValuesError("a vocabulary needs at least one token")

        ids_by_token: dict[str, int] = {}
        for position, token in enumerate(tokens):
            if not isinstance(token, str) or not token:
                raise InvalidValuesError(
                    f"every token must be a non-empty string, got {token!r} "
                    f"at position {position}"
                )
            if token in ids_by_token:
                raise NonUniqueTokensError(
                    f"token {token!r} appears at positions "
                    f"{ids_by_token[token]} and {position}"
                )
            ids_by_token[token] = position

        if unknown_token is not None and unknown_token not in ids_by_token:
            raise InvalidValuesError(
                f"the unknown token {unknown_token!r} must itself be a token"
            )

        self._tokens = tuple(tokens)
        self._ids_by_token = ids_by_token
        self._unknown_token = unknown_token

    @property
    def n_tokens(self) -> int:
        """How many tokens there are, which is the width of an embedding table."""
        return len(self._tokens)

    @property
    def unknown_token(self) -> str | None:
        """The fallback token, or ``None`` for a closed vocabulary."""
        return self._unknown_token

    @property
    def unknown_id(self) -> int | None:
        """The fallback token's id, or ``None`` for a closed vocabulary."""
        if self._unknown_token is None:
            return None
        return self._ids_by_token[self._unknown_token]

    @property
    def has_unknown(self) -> bool:
        """Whether an unknown lookup has somewhere to go."""
        return self._unknown_token is not None

    def id_of(self, token: str) -> int:
        """The id ``token`` goes in as.

        Raises
        ------
        UnknownTokenError
            If the token is absent and the vocabulary has no unknown token.
        """
        token_id = self._ids_by_token.get(token)
        if token_id is not None:
            return token_id

        if self._unknown_token is not None:
            return self._ids_by_token[self._unknown_token]

        raise UnknownTokenError(
            f"token {token!r} is not in this vocabulary of {self.n_tokens}, "
            f"which has no unknown token to fall back on"
        )

    def token_of(self, token_id: int) -> str:
        """The token at position ``token_id``.

        Raises
        ------
        UnknownTokenError
            If no token has that id.
        """
        if not 0 <= token_id < len(self._tokens):
            raise UnknownTokenError(
                f"no token has id {token_id}; ids run from 0 to {self.n_tokens - 1}"
            )
        return self._tokens[token_id]

    def ids_of(self, tokens: Sequence[str]) -> tuple[int, ...]:
        """The ids of several tokens, in order. See :meth:`id_of`."""
        return tuple(self.id_of(token) for token in tokens)

    def tokens_of(self, token_ids: Sequence[int]) -> tuple[str, ...]:
        """The tokens at several ids, in order. See :meth:`token_of`."""
        return tuple(self.token_of(token_id) for token_id in token_ids)

    def __getitem__(self, token: str) -> int:
        """The id for ``token``, so that ``vocabulary["the"]`` reads well."""
        return self.id_of(token)

    def __contains__(self, token: object) -> bool:
        return token in self._ids_by_token

    def __iter__(self) -> Iterator[str]:
        """Iterate the tokens in id order."""
        return iter(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vocabulary):
            return NotImplemented
        return (
            self._tokens == other._tokens
            and self._unknown_token == other._unknown_token
        )

    def __hash__(self) -> int:
        return hash((self._tokens, self._unknown_token))

    def __repr__(self) -> str:
        return (
            f"Vocabulary(n_tokens={self.n_tokens}, "
            f"unknown_token={self._unknown_token!r})"
        )
