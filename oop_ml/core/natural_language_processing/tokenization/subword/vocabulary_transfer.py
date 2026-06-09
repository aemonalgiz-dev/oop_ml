"""Fast vocabulary transfer: which old ids each token of a new vocabulary is made of.

The problem
-----------
A model trained against one vocabulary owns an embedding row per token of it.
Retokenize its domain with a smaller or better-fitting vocabulary and every one
of those rows is addressed by an id that now means something else, so the model
has to start its embeddings again, from noise, and relearn what it knew about
``the``. Gee et al. (2022) noticed that most of what it knew is still there:
a new token is either an old token, in which case its row can be copied, or it
is a string the old tokenizer can spell in a few of its own pieces, in which
case the mean of those pieces' rows is a far better start than noise. That is
the whole method, and it is fast because it needs no training at all.

The half that lives here
------------------------
The method has a tokenizer half and a model half, and only the first is a fact
about tokenizers. :meth:`VocabularyTransfer.between` answers, for every token
of the target vocabulary, which source ids it corresponds to:

1. The target's unknown token maps to the source's unknown token, if the source
   has one. An unknown token is a label for "something else", not text, and its
   spelling is not the thing being transferred.
2. A token spelled identically in both vocabularies, marker and all, maps to
   the source id of that spelling. ``est</w>`` in a byte pair vocabulary maps to
   ``est</w>`` in another, and not to whatever a re-encoding of ``est`` might
   yield.
3. Anything else is decoded to the plain text it stands for and encoded by the
   source, and maps to the ids that come back.

The model half -- a new embedding table whose row ``target_id`` is the mean of
the old table's rows ``source_ids`` -- belongs with the model that owns the
table, and is deliberately not here. What this module does say is that a
mapping may have *no* source ids, and a caller averaging zero rows gets ``nan``
and a warning rather than an error; such tokens are reported by
:attr:`TokenMappings.unmapped_tokens` and need a row from somewhere else.

What the third rule loses
-------------------------
A piece is not a word. Decoding ``lo`` from a byte pair vocabulary gives the
text ``lo``, and re-encoding that text spells a *whole word* ``lo``, whose last
character now carries an end-of-word marker the piece never had. On the
Sennrich corpus, where no word ends in ``o``, ``o</w>`` is not a symbol at all,
so a source vocabulary lacking ``lo`` maps it to ``l`` followed by the unknown
token. A continuation piece of a WordPiece vocabulary loses its ``##`` the same
way and is re-read as word-initial. The marker was the piece's information
about its position, decoding is precisely the operation that discards it, and
there is no text a piece can be turned into that keeps it. This is why the
second rule exists and runs first: within one convention it maps every shared
piece exactly, and the lossy route is taken only for tokens the source has
never seen in any form.

A token whose text encodes to nothing
-------------------------------------
A token can decode to a text the source finds no words in: a byte-level
vocabulary's ``Ġ``, the name of a lone space, or any token that is only a
marker. Such a token is reported as *unmapped*, with an empty ``source_ids``,
rather than mapped to nothing and left for the model half to divide by zero
over.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    UnknownTokenError,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import Tokenizer


class TokenMapping:
    """One target token, its id, and the source ids whose rows should seed it.

    Parameters
    ----------
    target_token:
        The token as the target vocabulary spells it. Non-empty.
    target_id:
        Its position in the target vocabulary. Non-negative.
    source_ids:
        The source vocabulary positions it corresponds to, in order. May be
        empty, which is what :attr:`is_mapped` reports.

    Raises
    ------
    EmptyValuesError
        If ``target_token`` is empty.
    InvalidValuesError
        If ``target_id`` or any source id is negative.
    """

    __slots__ = ("_source_ids", "_target_id", "_target_token")

    def __init__(
        self, target_token: str, target_id: int, source_ids: Sequence[int]
    ) -> None:
        if not isinstance(target_token, str) or not target_token:
            raise EmptyValuesError("a mapped token must hold at least one character")

        if target_id < 0:
            raise InvalidValuesError(f"a token id is a position, got {target_id}")

        for source_id in source_ids:
            if source_id < 0:
                raise InvalidValuesError(f"a token id is a position, got {source_id}")

        self._target_token = target_token
        self._target_id = int(target_id)
        self._source_ids = tuple(int(source_id) for source_id in source_ids)

    @property
    def target_token(self) -> str:
        """The token as the target vocabulary spells it."""
        return self._target_token

    @property
    def target_id(self) -> int:
        """Its position in the target vocabulary: the row to be seeded."""
        return self._target_id

    @property
    def source_ids(self) -> tuple[int, ...]:
        """The source positions whose rows seed it, in order. Empty if unmapped."""
        return self._source_ids

    @property
    def n_source_ids(self) -> int:
        """How many source rows the seed averages over."""
        return len(self._source_ids)

    @property
    def is_mapped(self) -> bool:
        """Whether there is at least one source row to seed from."""
        return len(self._source_ids) > 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenMapping):
            return NotImplemented
        return (
            self._target_token == other._target_token
            and self._target_id == other._target_id
            and self._source_ids == other._source_ids
        )

    def __hash__(self) -> int:
        return hash((self._target_token, self._target_id, self._source_ids))

    def __repr__(self) -> str:
        return (
            f"TokenMapping({self._target_token!r}, {self._target_id}, "
            f"source_ids={list(self._source_ids)!r})"
        )


class TokenMappings:
    """One mapping per target token, in target id order, addressable by token.

    Parameters
    ----------
    mappings:
        One :class:`TokenMapping` per target token. Non-empty, no token twice.

    Raises
    ------
    EmptyValuesError
        If there are no mappings.
    InvalidValuesError
        If a target token is mapped twice.
    """

    __slots__ = ("_mappings", "_mappings_by_token")

    def __init__(self, mappings: Sequence[TokenMapping]) -> None:
        if len(mappings) == 0:
            raise EmptyValuesError("a vocabulary transfer needs at least one mapping")

        mappings_by_token: dict[str, TokenMapping] = {}
        for mapping in mappings:
            if mapping.target_token in mappings_by_token:
                raise InvalidValuesError(
                    f"the token {mapping.target_token!r} is mapped twice"
                )
            mappings_by_token[mapping.target_token] = mapping

        self._mappings = tuple(mappings)
        self._mappings_by_token = mappings_by_token

    @property
    def n_mappings(self) -> int:
        """How many target tokens there are, mapped or not."""
        return len(self._mappings)

    @property
    def n_unmapped(self) -> int:
        """How many target tokens have no source id to seed from."""
        return sum(1 for mapping in self._mappings if not mapping.is_mapped)

    @property
    def unmapped_tokens(self) -> tuple[str, ...]:
        """The target tokens with no source id, in target id order."""
        return tuple(
            mapping.target_token for mapping in self._mappings if not mapping.is_mapped
        )

    def for_token(self, token: str) -> TokenMapping:
        """The mapping for one target token.

        Raises
        ------
        UnknownTokenError
            If the token is not in the target vocabulary.
        """
        mapping = self._mappings_by_token.get(token)
        if mapping is None:
            raise UnknownTokenError(
                f"the token {token!r} is not among the {self.n_mappings} target tokens"
            )
        return mapping

    def __iter__(self) -> Iterator[TokenMapping]:
        """Iterate the mappings in target id order."""
        return iter(self._mappings)

    def __len__(self) -> int:
        return len(self._mappings)

    def __contains__(self, token: object) -> bool:
        return token in self._mappings_by_token

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenMappings):
            return NotImplemented
        return self._mappings == other._mappings

    def __hash__(self) -> int:
        return hash(self._mappings)

    def __repr__(self) -> str:
        return (
            f"TokenMappings(n_mappings={self.n_mappings}, n_unmapped={self.n_unmapped})"
        )


class VocabularyTransfer:
    """The tokenizer half of fast vocabulary transfer (Gee et al., 2022)."""

    @staticmethod
    def between(source: Tokenizer, target: Tokenizer) -> TokenMappings:
        """For every token of ``target``, the ids of ``source`` it is made of.

        Parameters
        ----------
        source:
            The tokenizer the existing model was trained against.
        target:
            The tokenizer the model is moving to. Both must be fitted, if they
            are the kind that fits.

        Raises
        ------
        NotFittedError
            If either tokenizer is learned and has not been fit.
        UnknownTokenError
            If the source has a closed vocabulary that cannot spell the text of
            some target token. A byte-level source can spell anything, so this
            is reachable only with a closed source over a narrower alphabet.
        """
        source_vocabulary = source.vocabulary
        target_vocabulary = target.vocabulary

        mappings: list[TokenMapping] = []
        for target_id, target_token in enumerate(target_vocabulary):
            if (
                target_token == target_vocabulary.unknown_token
                and source_vocabulary.unknown_id is not None
            ):
                source_ids: tuple[int, ...] = (source_vocabulary.unknown_id,)
            elif target_token in source_vocabulary:
                source_ids = (source_vocabulary.id_of(target_token),)
            else:
                source_ids = source.encode(target.decode([target_id])).ids
            mappings.append(TokenMapping(target_token, target_id, source_ids))
        return TokenMappings(mappings)
