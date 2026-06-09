"""Hashed character representation: a codepoint becomes a few bucket ids.

The table that cannot be built
------------------------------
The open-vocabulary character tokenizer is ``id = ord(character)``, and it is
not in this library because Unicode has 1,114,112 codepoints: a materialised
embedding table over all of them is 855,638,016 parameters at a width of 768,
almost every row of it never touched. Clark et al. (2022)'s CANINE keeps the
open vocabulary and drops the table. A codepoint is hashed into one of
``n_buckets`` buckets, and the model embeds the *bucket*, so the table has
``n_buckets`` rows whatever the script. At CANINE's 16,384 buckets and the same
width that is 12,582,912 parameters, sixty-eight times fewer, and a character
from a script the training data never contained still has a row.

Several hashes, and what they are for
-------------------------------------
One hash sends two different characters to one bucket with probability about
``1 / n_buckets``, and there the model cannot tell them apart at all. CANINE
uses ``n_hash_functions`` hashes instead of one, so a character is
``n_hash_functions`` bucket ids, each with its own embedding table of width
``d / n_hash_functions``, and the character's embedding is those slices
concatenated. The argument is that a collision in one hash is unlikely to be a
collision in all of them, so two characters that share one slice still differ
in the others; the Bloom-embedding variant sums full-width vectors instead of
concatenating slices and rests on the same argument. The tokenizer-side is the
same either way: this class produces the bucket ids, and how the model
combines them is the model's decision.

The hash is the one CANINE's released code uses, multiplicative hashing with a
small prime per function::

    bucket_k = ((codepoint + 1) * PRIME_k) mod n_buckets

with the primes ``31, 43, 59, 61, 73, 97, 103, 113`` for the default eight,
so that the default configuration reproduces CANINE's bucket ids exactly. The
letter ``a``, codepoint 97, hashes to ``98 * 31 = 3038``, ``98 * 43 = 4214``,
and so on: ``3038, 4214, 5782, 5978, 7154, 9506, 10094, 11074``, none of them
reduced because all are below 16,384. Twenty-four further primes follow the
eight, so up to thirty-two hashes can be asked for; more is refused at
construction, since a hash without a prime has no definition.

What the independence argument assumes, measured
------------------------------------------------
Multiplying by a prime coprime to ``n_buckets`` permutes the residues modulo
``n_buckets``, so hash ``k`` separates two codepoints exactly when their
difference is not a multiple of ``n_buckets / gcd(PRIME_k, n_buckets)``. With
the default 16,384 buckets, a power of two, every prime is coprime, every hash
separates exactly the same pairs, and two codepoints collide in one hash
precisely when they are congruent modulo 16,384 -- in which case they collide
in all eight. ``A``, codepoint 65, and ``U+4041``, codepoint 16,449, share
every one of their eight buckets, ``2046, 2838, 3894, 4026, 4818, 6402, 6798,
7458``. Counted over all 1,114,112 codepoints, eight hashes produce exactly
16,384 distinct bucket tuples, the same number one hash does. So with this
hash family and any bucket count that at least one prime is coprime to, the
extra hashes buy no separation at all; they buy it only when every prime
shares a factor with the bucket count, as at 93 buckets, where the first prime
31 collapses the codepoints to three classes and the second, 43, still
separates all 93. The argument is sound for hashes whose collisions are
independent, and multiplicative hashes sharing one modulus are not. Recorded
here because it is measurable and because a reader would otherwise take the
eight hashes as eight chances.

What it costs
-------------
There is no vocabulary, so there is no way to name a token: a bucket id is
shared by every codepoint that hashes to it, and nothing about which one was
meant survives. That is why this class is not a
:class:`~oop_ml.core.natural_language_processing.tokenization.tokenizer.Tokenizer`:
a character becomes ``n_hash_functions`` ids rather than one, and decoding
needs the codepoints rather than the ids. :class:`HashedCharacters` therefore
keeps every character's codepoint beside its buckets, and its ``text`` is
rebuilt from those. CANINE's own decoder does the same, predicting characters
rather than buckets.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.tokenizer import checked_text

HASH_PRIMES: tuple[int, ...] = (
    31,
    43,
    59,
    61,
    73,
    97,
    103,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
    233,
    239,
    241,
    251,
)
"""The multiplier of hash ``k`` is position ``k``. The first eight are CANINE's."""


class HashedCharacter:
    """One character: its codepoint, and the bucket each hash sent it to.

    Parameters
    ----------
    codepoint:
        The character's Unicode codepoint, ``0`` to ``0x10FFFF``.
    bucket_ids:
        One bucket id per hash function, in hash order. At least one, each
        non-negative.

    Raises
    ------
    EmptyValuesError
        If there are no bucket ids.
    InvalidValuesError
        If the codepoint is outside Unicode or a bucket id is negative.
    """

    __slots__ = ("_bucket_ids", "_codepoint")

    def __init__(self, codepoint: int, bucket_ids: Sequence[int]) -> None:
        if not 0 <= codepoint <= sys.maxunicode:
            raise InvalidValuesError(
                f"a codepoint lies in 0 to {sys.maxunicode:#x}, got {codepoint}"
            )

        if len(bucket_ids) == 0:
            raise EmptyValuesError("a hashed character has at least one bucket")

        for bucket_id in bucket_ids:
            if bucket_id < 0:
                raise InvalidValuesError(f"a bucket id is a position, got {bucket_id}")

        self._codepoint = int(codepoint)
        self._bucket_ids = tuple(int(bucket_id) for bucket_id in bucket_ids)

    @property
    def codepoint(self) -> int:
        """The Unicode codepoint."""
        return self._codepoint

    @property
    def character(self) -> str:
        """The character itself."""
        return chr(self._codepoint)

    @property
    def bucket_ids(self) -> tuple[int, ...]:
        """The bucket each hash chose, in hash order."""
        return self._bucket_ids

    @property
    def n_hash_functions(self) -> int:
        """How many hashes were applied."""
        return len(self._bucket_ids)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashedCharacter):
            return NotImplemented
        return (
            self._codepoint == other._codepoint
            and self._bucket_ids == other._bucket_ids
        )

    def __hash__(self) -> int:
        return hash((self._codepoint, self._bucket_ids))

    def __repr__(self) -> str:
        return f"HashedCharacter({self.character!r}, {list(self._bucket_ids)!r})"


class HashedCharacters:
    """The hashed characters one text became, in order.

    Parameters
    ----------
    characters:
        One :class:`HashedCharacter` per character of the text. May be empty.
    """

    __slots__ = ("_characters",)

    def __init__(self, characters: Sequence[HashedCharacter]) -> None:
        self._characters = tuple(characters)

    @property
    def codepoints(self) -> tuple[int, ...]:
        """Every codepoint, in order: what decoding needs."""
        return tuple(character.codepoint for character in self._characters)

    @property
    def bucket_ids(self) -> tuple[tuple[int, ...], ...]:
        """Every character's buckets, in order: what a model reads."""
        return tuple(character.bucket_ids for character in self._characters)

    @property
    def text(self) -> str:
        """The text, rebuilt from the codepoints."""
        return "".join(character.character for character in self._characters)

    @property
    def n_characters(self) -> int:
        """How many characters the text held."""
        return len(self._characters)

    def __iter__(self) -> Iterator[HashedCharacter]:
        return iter(self._characters)

    def __len__(self) -> int:
        return len(self._characters)

    def __getitem__(self, position: int) -> HashedCharacter:
        return self._characters[position]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashedCharacters):
            return NotImplemented
        return self._characters == other._characters

    def __hash__(self) -> int:
        return hash(self._characters)

    def __repr__(self) -> str:
        return f"HashedCharacters({self.text!r})"


class HashedCharacterTokenizer(BaseModel):
    """Every character becomes ``n_hash_functions`` bucket ids. CANINE's hashing.

    Not a
    :class:`~oop_ml.core.natural_language_processing.tokenization.tokenizer.Tokenizer`:
    a character maps to several ids rather than one, there is no vocabulary
    and so no unknown token, and decoding reads codepoints rather than ids.

    Parameters
    ----------
    n_hash_functions:
        How many buckets each character gets. At most ``len(HASH_PRIMES)``,
        which is 32.
    n_buckets:
        How many buckets each hash ranges over: the height of each embedding
        table.

    Raises
    ------
    pydantic.ValidationError
        If more hash functions are asked for than there are primes, or a
        field is out of range. A ``ValueError`` raised inside a validator is
        what pydantic reports, which is the library's rule for every
        refusal made at construction.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_hash_functions: int = Field(default=8, ge=1)
    n_buckets: int = Field(default=16384, ge=2)

    @model_validator(mode="after")
    def _check_a_prime_exists_for_every_hash(self) -> Self:
        if self.n_hash_functions > len(HASH_PRIMES):
            raise ValueError(
                f"n_hash_functions={self.n_hash_functions} but only "
                f"{len(HASH_PRIMES)} primes are held, one per hash function"
            )
        return self

    @property
    def primes(self) -> tuple[int, ...]:
        """The multiplier of each hash function, in hash order."""
        return HASH_PRIMES[: self.n_hash_functions]

    def bucket_ids_of(self, codepoint: int) -> tuple[int, ...]:
        """The bucket each hash sends ``codepoint`` to.

        ``((codepoint + 1) * prime) mod n_buckets`` for each prime.

        Raises
        ------
        InvalidValuesError
            If ``codepoint`` is outside Unicode.
        """
        if not 0 <= codepoint <= sys.maxunicode:
            raise InvalidValuesError(
                f"a codepoint lies in 0 to {sys.maxunicode:#x}, got {codepoint}"
            )
        return tuple(
            ((codepoint + 1) * prime) % self.n_buckets for prime in self.primes
        )

    def encode(self, text: str) -> HashedCharacters:
        """Every character of ``text`` with its codepoint and its buckets.

        Raises
        ------
        InvalidValuesError
            If ``text`` is not a string.
        """
        return HashedCharacters(
            [
                HashedCharacter(ord(character), self.bucket_ids_of(ord(character)))
                for character in checked_text(text)
            ]
        )
