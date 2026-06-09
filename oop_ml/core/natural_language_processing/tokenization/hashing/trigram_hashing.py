"""Trigram hashing: a word becomes the set of buckets its character trigrams hit.

Giving up the vocabulary from the other side
--------------------------------------------
A subword vocabulary spells ``cat`` and ``cats`` as two unrelated ids, and the
model has to learn from data that they mean nearly the same thing -- once per
inflection, once per language, once per typo. Deiseroth et al. (2024)'s T-FREE
removes the vocabulary and with it the problem. A word is wrapped in a boundary
marker, ``_cat_``, and cut into its overlapping character trigrams, ``_ca``,
``cat``, ``at_``; each trigram is hashed into one of ``n_buckets`` buckets;
and the word is the *set* of buckets that lit up. The model embeds buckets, and
a word's embedding is the sum of its active buckets' rows, so ``cats``, whose
trigrams are ``_ca``, ``cat``, ``ats``, ``ts_``, shares two of its four terms
with ``cat`` before any training has happened. That overlap between
morphological neighbours is the point of the method, and it comes free from the
spelling.

It is a set rather than a sequence, so a repeated trigram counts once --
``_aaaa_`` has three distinct trigrams among its four -- and two trigrams of
one word that collide in the hash light one bucket between them. The set is
kept sorted, so that two constructions of the same word compare equal.

Counting trigrams
-----------------
A word of ``n`` characters wrapped in one marker on each side is ``n + 2``
characters and has ``n`` trigrams: ``word`` gives four, ``_wo``, ``wor``,
``ord``, ``rd_``, and a one-character word gives one, ``_a_``, in which both
markers appear at once. The marker is one character so that count holds; a
marker no word contains is the intent, and T-FREE's own is the space, which a
whitespace split guarantees. The default here is ``_``, which a word in
``snake_case`` does contain, making an inner trigram look like a boundary one;
a caller with such words should choose another.

Why the hash is written out
---------------------------
Python's ``hash`` of a string is salted per process, so the same trigram lands
in a different bucket every run and a model trained in one process would read
the wrong rows in the next. The hash here is a polynomial in the codepoints,
``c_0 m^2 + c_1 m + c_2`` for a trigram, computed exactly and then reduced
modulo ``n_buckets``: ``_ca`` under the first multiplier ``1000003`` is
``95 * 1000003^2 + 99 * 1000003 + 97 = 95000669001249``. It is fixed, and it
spreads: the 19,683 trigrams over the twenty-six lowercase letters and the
marker fall into 8,122 of 8,192 buckets with a heaviest load of five against a
mean of 2.4. With ``n_hash_functions`` above one each trigram is hashed under
that many distinct multipliers and lights that many buckets, so a collision
under one multiplier is not a collision under another; there are eight
multipliers, and more hashes than that are refused at construction.

Worked, at 8,192 buckets and one hash
-------------------------------------
``cat`` lights ``2002, 5665, 6052`` and ``cats`` lights ``1676, 2002, 5665,
6072``; the two shared buckets are the two shared trigrams, and
:func:`overlap` reports 2.

Why there is no decode
----------------------
A set of buckets is not a word; it is a fingerprint of one, and many words
share a fingerprint. T-FREE recovers a word by nearest neighbour: it keeps a
dictionary of candidate words, hashes each, and answers with the candidate
whose active set is nearest the model's output. That dictionary is the
caller's, since it is a fact about the task rather than the tokenizer, so this
module offers the similarity that lookup uses, :func:`overlap`, and no
``decode``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.tokenizer import (
    PreTokenizer,
    checked_text,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)

HASH_MULTIPLIERS: tuple[int, ...] = (
    1000003,
    1000033,
    1000037,
    1000039,
    1000081,
    1000099,
    1000117,
    1000121,
)
"""The multiplier of hash ``k`` is position ``k``. Distinct primes near a million."""

TRIGRAM_LENGTH: int = 3


def polynomial_hash(text: str, multiplier: int) -> int:
    """``sum(codepoint_i * multiplier ** (len - 1 - i))``, exactly.

    A fixed function of the text, unlike Python's salted ``hash``. Not reduced
    here, so that the caller chooses the modulus.
    """
    value = 0
    for character in text:
        value = value * multiplier + ord(character)
    return value


class WordActivations:
    """One word and the sorted set of buckets its trigrams lit.

    Parameters
    ----------
    word:
        The word, as the pre-tokenizer spelled it. At least one character.
    bucket_ids:
        The active buckets, strictly increasing, each non-negative. At least
        one, since every word has at least one trigram.

    Raises
    ------
    EmptyValuesError
        If the word is empty or no bucket is active.
    InvalidValuesError
        If the buckets are not strictly increasing or one is negative.
    """

    __slots__ = ("_bucket_ids", "_word")

    def __init__(self, word: str, bucket_ids: Sequence[int]) -> None:
        if not isinstance(word, str) or not word:
            raise EmptyValuesError("a word holds at least one character")

        if len(bucket_ids) == 0:
            raise EmptyValuesError("a word lights at least one bucket")

        previous = -1
        for bucket_id in bucket_ids:
            if bucket_id <= previous:
                raise InvalidValuesError(
                    f"bucket ids are a sorted set of non-negative positions, got "
                    f"{list(bucket_ids)!r}"
                )
            previous = bucket_id

        self._word = word
        self._bucket_ids = tuple(int(bucket_id) for bucket_id in bucket_ids)

    @property
    def word(self) -> str:
        """The word."""
        return self._word

    @property
    def bucket_ids(self) -> tuple[int, ...]:
        """The active buckets, ascending."""
        return self._bucket_ids

    @property
    def n_active(self) -> int:
        """How many buckets are active."""
        return len(self._bucket_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self._bucket_ids)

    def __len__(self) -> int:
        return len(self._bucket_ids)

    def __contains__(self, bucket_id: object) -> bool:
        return bucket_id in self._bucket_ids

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WordActivations):
            return NotImplemented
        return self._word == other._word and self._bucket_ids == other._bucket_ids

    def __hash__(self) -> int:
        return hash((self._word, self._bucket_ids))

    def __repr__(self) -> str:
        return f"WordActivations({self._word!r}, {list(self._bucket_ids)!r})"


def overlap(first: WordActivations, second: WordActivations) -> int:
    """How many buckets two words both light: the similarity a lookup ranks by."""
    return len(set(first.bucket_ids) & set(second.bucket_ids))


class TextActivations:
    """The activations of every word of one text, in order.

    Parameters
    ----------
    activations:
        One :class:`WordActivations` per word. May be empty, for a text the
        pre-tokenizer finds no words in.
    """

    __slots__ = ("_activations",)

    def __init__(self, activations: Sequence[WordActivations]) -> None:
        self._activations = tuple(activations)

    @property
    def words(self) -> tuple[str, ...]:
        """The words, in order."""
        return tuple(activations.word for activations in self._activations)

    @property
    def bucket_ids(self) -> tuple[tuple[int, ...], ...]:
        """Every word's active buckets, in order: what a model reads."""
        return tuple(activations.bucket_ids for activations in self._activations)

    @property
    def n_words(self) -> int:
        """How many words the text held."""
        return len(self._activations)

    def __iter__(self) -> Iterator[WordActivations]:
        return iter(self._activations)

    def __len__(self) -> int:
        return len(self._activations)

    def __getitem__(self, position: int) -> WordActivations:
        return self._activations[position]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TextActivations):
            return NotImplemented
        return self._activations == other._activations

    def __hash__(self) -> int:
        return hash(self._activations)

    def __repr__(self) -> str:
        return f"TextActivations({list(self.words)!r})"


class TrigramHashTokenizer(BaseModel):
    """Every word becomes the sorted set of buckets its trigrams hash to.

    T-FREE's tokenizer-side. Not a
    :class:`~oop_ml.core.natural_language_processing.tokenization.tokenizer.Tokenizer`:
    a word becomes a set of ids rather than one, there is no vocabulary, and
    nothing decodes.

    Parameters
    ----------
    n_buckets:
        How many buckets a trigram can land in: the height of the embedding
        table.
    n_hash_functions:
        How many multipliers each trigram is hashed under, each lighting one
        bucket. At most ``len(HASH_MULTIPLIERS)``, which is 8.
    pre_tokenizer:
        Decides where the words are.
    boundary_marker:
        The one character wrapped around each word before its trigrams are
        taken, so that a word's first and last trigrams know they are.

    Raises
    ------
    pydantic.ValidationError
        If more hash functions are asked for than there are multipliers, or a
        field is out of range. A ``ValueError`` raised inside a validator is
        what pydantic reports, which is the library's rule for every
        refusal made at construction.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_buckets: int = Field(ge=2)
    n_hash_functions: int = Field(default=1, ge=1)
    pre_tokenizer: PreTokenizer = Field(default_factory=WhitespacePreTokenizer)
    boundary_marker: str = Field(default="_", min_length=1, max_length=1)

    @model_validator(mode="after")
    def _check_a_multiplier_exists_for_every_hash(self) -> Self:
        if self.n_hash_functions > len(HASH_MULTIPLIERS):
            raise ValueError(
                f"n_hash_functions={self.n_hash_functions} but only "
                f"{len(HASH_MULTIPLIERS)} multipliers are held, one per hash function"
            )
        return self

    @property
    def multipliers(self) -> tuple[int, ...]:
        """The multiplier of each hash function, in hash order."""
        return HASH_MULTIPLIERS[: self.n_hash_functions]

    def trigrams_of(self, word: str) -> tuple[str, ...]:
        """The word's character trigrams once wrapped in the marker, in order.

        A word of ``n`` characters has ``n`` of them, repeats included.

        Raises
        ------
        InvalidValuesError
            If ``word`` is not a string.
        EmptyValuesError
            If ``word`` is empty.
        """
        if not checked_text(word):
            raise EmptyValuesError("a word holds at least one character")

        wrapped = self.boundary_marker + word + self.boundary_marker
        return tuple(
            wrapped[position : position + TRIGRAM_LENGTH]
            for position in range(len(wrapped) - TRIGRAM_LENGTH + 1)
        )

    def activations_of(self, word: str) -> WordActivations:
        """The sorted set of buckets ``word``'s trigrams light.

        Raises
        ------
        InvalidValuesError
            If ``word`` is not a string.
        EmptyValuesError
            If ``word`` is empty.
        """
        bucket_ids = {
            polynomial_hash(trigram, multiplier) % self.n_buckets
            for trigram in self.trigrams_of(word)
            for multiplier in self.multipliers
        }
        return WordActivations(word, sorted(bucket_ids))

    def encode(self, text: str) -> TextActivations:
        """The activations of every word the pre-tokenizer finds in ``text``.

        Raises
        ------
        InvalidValuesError
            If ``text`` is not a string.
        """
        return TextActivations(
            [self.activations_of(word) for word in self.pre_tokenizer.split(text).texts]
        )
