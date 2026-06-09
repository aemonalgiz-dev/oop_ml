"""Byte patching: grouping a text's bytes into blocks a model reads as units.

The problem with bytes
----------------------
A byte-level model has the smallest possible vocabulary and the longest
possible sequences: a text is as many tokens as it has UTF-8 bytes, three to
four times a subword model's count, and attention costs the square of that.
The hierarchical byte models keep the vocabulary and shorten the sequence by
reading the bytes in *patches*: a local model handles the bytes inside each
patch, and a global model sees one vector per patch. Where the patch
boundaries fall is the tokenizer-side of that design, and this module holds
the two ways of choosing them.

A patch is several ids, not one, which is why neither patcher here is a
:class:`~oop_ml.core.natural_language_processing.tokenization.tokenizer.Tokenizer`.
An :class:`~oop_ml.core.natural_language_processing.tokenization.encoding.Encoding`
pairs each piece of text with the single id a model reads, and a patch has no
single id: it is a run of byte ids with a start and an end. So the answer is a
:class:`Patches`, an ordered run of :class:`Patch` objects that tile the text's
bytes exactly, and the ids inside each patch are plain byte values, since a
patch is a block of the byte tokenizer's output and not a vocabulary of its
own.

Fixed patching
--------------
Yu et al. (2023) built MegaByte on the simplest rule: cut the byte sequence
into consecutive blocks of a fixed size, the last block shorter if the count
does not divide. Ten bytes at a patch size of four are patches of 4, 4 and 2,
starting at offsets 0, 4 and 8. Nawrot et al. (2022)'s Hourglass downsamples
by the same partition inside the model rather than before it, so this class
is the tokenizer-side of both. The rule knows nothing about the text, so a
patch can cut through a multibyte character: ``é`` is two bytes and a patch
size of one puts them in different patches. That is the design, not a
defect -- MegaByte's patches do the same -- and it is what the local model
exists to repair.

Entropy patching
----------------
Pagnoni et al. (2024)'s Byte Latent Transformer spends its patches where the
text is hard to predict. A small byte-level language model estimates, at every
position, the entropy of the next byte given the bytes before it; a new patch
begins wherever that entropy is high, so predictable runs -- the inside of a
word the model has seen a thousand times -- are swallowed into long patches
and the uncertain positions, which are mostly the first byte of a word, get a
patch each. Two rules for "high" are offered, both from the paper.
:attr:`PatchingRule.GLOBAL_THRESHOLD` starts a patch where the entropy exceeds
``threshold``; :attr:`PatchingRule.RELATIVE_INCREASE`, the paper's
"approximate monotonic" constraint, starts one where the entropy exceeds the
previous position's by more than ``threshold``, so a long run of uniformly
high entropy is one patch rather than many. On a text the model has never
seen, every position after the first is exactly 8 bits -- the first is scored
under the start-of-text context, which every text shares -- so the first rule
makes every byte its own patch at any threshold below 8 and the second makes
the whole text one patch, because a plateau has no rises. The first byte
always starts a patch under either rule.

Where BLT has a small transformer, this class has a byte n-gram model, because
the patching rule is the idea and the estimator is interchangeable. ``fit``
counts, across the corpus, how often each context of ``order`` bytes was
followed by each byte value; the start of each text is padded with ``order``
copies of :data:`START_OF_TEXT`, a value of 256 that no byte can equal, so
that the first bytes of a text have a context too and a text's opening is a
context of its own. The next-byte distribution after a context is the counts
plus ``smoothing`` for every one of the 256 byte values, divided by the total,
and the entropy is Shannon's, in bits, so it lies between 0 and exactly 8.
A context never seen has every byte at ``1/256`` and is 8 bits precisely.

Worked, on a corpus of ``aab`` at order 1 and smoothing 1
--------------------------------------------------------
The start context was followed by ``a`` once, so ``a`` has probability
``2/257`` and each of the other 255 values ``1/257``, and the entropy is
``(2/257) log2(257/2) + (255/257) log2(257) = 7.99784`` bits. The context
``a`` was followed by ``a`` once and ``b`` once: ``2/258`` each, the other 254
at ``1/258``, ``7.99572`` bits. The context ``b`` was never seen, 8 bits. So
the entropies of the text ``aabb`` are ``7.99784, 7.99572, 7.99572, 8.0``.

Those numbers are all near 8 for a reason worth knowing before choosing a
threshold. Add-one smoothing spreads 255 pseudo-observations across the
unseen values, and a context has to be observed about 2,294 times with the
same continuation before that continuation reaches probability 0.9. So on a
small corpus the default smoothing makes every position look uncertain, and
the specs demonstrate word alignment at a smoothing of 0.001 instead: twenty
copies of ``the cat sat on the mat`` at order 2 put the inside of a word at
0.198 bits, a context seen forty times at 0.106, and the first byte of
``cat``, ``sat``, ``on`` and ``mat`` at 1.099, since ``e␣`` and ``t␣`` were each
followed by two different words. At a threshold of 0.5 the patches are
``the␣``, ``cat␣``, ``sat␣``, ``on the␣`` and ``mat`` -- ``on the`` stays one
patch because ``on`` was only ever followed by ``the``, which is exactly the
behaviour the rule promises.

Cost
----
Fitting is one pass over the corpus's bytes. Each entropy is a sum over 256
values, so scoring a text costs ``256 n`` operations for ``n`` bytes; the
usual repair is to sum the seen values and multiply the unseen contribution
by their count, which is an optimisation of the same sum.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.tokenizer import checked_text

N_BYTE_VALUES: int = 256
"""How many outcomes a next-byte distribution ranges over."""

START_OF_TEXT: int = 256
"""The context value standing before a text's first byte; no byte equals it."""


class Patch:
    """One block of consecutive bytes, with its span in the text's bytes.

    Parameters
    ----------
    byte_ids:
        The byte values in the block, each ``0`` to ``255``. At least one.
    start:
        Offset of the first byte in the text's UTF-8 bytes.
    end:
        Offset one past the last, so that ``end - start`` is the block's size.

    Raises
    ------
    EmptyValuesError
        If there are no bytes.
    InvalidValuesError
        If a value is not a byte, ``start`` is negative, or the span does not
        match the count.
    """

    __slots__ = ("_byte_ids", "_end", "_start")

    def __init__(self, byte_ids: Sequence[int], start: int, end: int) -> None:
        if len(byte_ids) == 0:
            raise EmptyValuesError("a patch holds at least one byte")

        for byte_id in byte_ids:
            if not 0 <= byte_id < N_BYTE_VALUES:
                raise InvalidValuesError(
                    f"a patch holds byte values 0 to 255, got {byte_id}"
                )

        if start < 0 or end - start != len(byte_ids):
            raise InvalidValuesError(
                f"a patch of {len(byte_ids)} bytes must span [start, start + "
                f"{len(byte_ids)}) from a non-negative start, got [{start}, {end})"
            )

        self._byte_ids = tuple(int(byte_id) for byte_id in byte_ids)
        self._start = int(start)
        self._end = int(end)

    @property
    def byte_ids(self) -> tuple[int, ...]:
        """The byte values, in order."""
        return self._byte_ids

    @property
    def start(self) -> int:
        """Offset of the first byte."""
        return self._start

    @property
    def end(self) -> int:
        """Offset one past the last byte."""
        return self._end

    @property
    def n_bytes(self) -> int:
        """How many bytes the patch holds."""
        return len(self._byte_ids)

    @property
    def text(self) -> str:
        """The bytes decoded as UTF-8, for reading.

        A patch can cut through a multibyte character, in which case the cut
        character appears as the replacement character ``�``. The byte ids are
        the truth; this is a convenience.
        """
        return bytes(self._byte_ids).decode("utf-8", errors="replace")

    def __iter__(self) -> Iterator[int]:
        return iter(self._byte_ids)

    def __len__(self) -> int:
        return len(self._byte_ids)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Patch):
            return NotImplemented
        return (
            self._byte_ids == other._byte_ids
            and self._start == other._start
            and self._end == other._end
        )

    def __hash__(self) -> int:
        return hash((self._byte_ids, self._start, self._end))

    def __repr__(self) -> str:
        return f"Patch({list(self._byte_ids)!r}, {self._start}, {self._end})"


class Patches:
    """The patches one text's bytes were cut into, tiling them exactly.

    Parameters
    ----------
    patches:
        The patches in order. The first starts at offset 0 and each starts
        where the previous ended, so that together they cover every byte once.
        May be empty, for a text with no bytes.

    Raises
    ------
    InvalidValuesError
        If the patches leave a gap, overlap, or do not begin at offset 0.
    """

    __slots__ = ("_patches",)

    def __init__(self, patches: Sequence[Patch]) -> None:
        expected_start = 0
        for patch in patches:
            if patch.start != expected_start:
                raise InvalidValuesError(
                    f"patches must tile the bytes from offset 0 without gaps or "
                    f"overlaps; {patch!r} should start at {expected_start}"
                )
            expected_start = patch.end

        self._patches = tuple(patches)

    @property
    def n_patches(self) -> int:
        """How many patches there are: the length of the sequence a global
        model reads."""
        return len(self._patches)

    @property
    def n_bytes(self) -> int:
        """How many bytes the patches cover between them."""
        return sum(patch.n_bytes for patch in self._patches)

    @property
    def byte_ids(self) -> tuple[int, ...]:
        """Every byte value in order, the patch boundaries forgotten."""
        return tuple(byte_id for patch in self._patches for byte_id in patch)

    @property
    def boundaries(self) -> tuple[int, ...]:
        """The offset at which each patch starts, in order."""
        return tuple(patch.start for patch in self._patches)

    @property
    def sizes(self) -> tuple[int, ...]:
        """How many bytes each patch holds, in order."""
        return tuple(patch.n_bytes for patch in self._patches)

    def __iter__(self) -> Iterator[Patch]:
        return iter(self._patches)

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, position: int) -> Patch:
        return self._patches[position]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Patches):
            return NotImplemented
        return self._patches == other._patches

    def __hash__(self) -> int:
        return hash(self._patches)

    def __repr__(self) -> str:
        return f"Patches(sizes={list(self.sizes)!r})"


def patches_starting_at(byte_values: Sequence[int], starts: Sequence[int]) -> Patches:
    """``byte_values`` cut into patches beginning at each offset in ``starts``.

    Parameters
    ----------
    byte_values:
        A text's bytes.
    starts:
        Where each patch begins, strictly increasing. The first must be 0 when
        there are any bytes at all; for no bytes there are no patches and
        ``starts`` must be empty too.

    Raises
    ------
    InvalidValuesError
        If ``starts`` does not begin at 0, is not strictly increasing, or
        names an offset past the last byte.
    """
    if len(byte_values) == 0:
        if len(starts) != 0:
            raise InvalidValuesError("no bytes means no patches, so no starts")
        return Patches([])

    if len(starts) == 0 or starts[0] != 0:
        raise InvalidValuesError("the first patch must start at offset 0")

    patches: list[Patch] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(byte_values)
        if end <= start or start >= len(byte_values):
            raise InvalidValuesError(
                f"patch starts must be strictly increasing and within the "
                f"{len(byte_values)} bytes, got {list(starts)!r}"
            )
        patches.append(Patch(byte_values[start:end], start, end))
    return Patches(patches)


class FixedSizePatcher(BaseModel):
    """Consecutive blocks of ``patch_size`` bytes, the last one shorter.

    MegaByte's patching. A pydantic model so that the size is validated at
    construction, like every other hyperparameter.

    Parameters
    ----------
    patch_size:
        How many bytes each patch holds, except the last.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    patch_size: int = Field(ge=1)

    def patch(self, text: str) -> Patches:
        """The text's UTF-8 bytes in blocks of ``patch_size``.

        Raises
        ------
        InvalidValuesError
            If ``text`` is not a string.
        """
        byte_values = checked_text(text).encode("utf-8")
        return patches_starting_at(
            byte_values, range(0, len(byte_values), self.patch_size)
        )


class PatchingRule(StrEnum):
    """How an entropy decides that a new patch begins at a position."""

    GLOBAL_THRESHOLD = "global_threshold"
    """The entropy at the position exceeds the threshold."""

    RELATIVE_INCREASE = "relative_increase"
    """The entropy at the position exceeds the previous position's by more
    than the threshold. Pagnoni et al.'s approximate monotonic constraint."""


class EntropyPatcher(Fittable):
    """New patches where the next byte is hard to predict.

    The Byte Latent Transformer's entropy patching with a byte n-gram model as
    the estimator. Construction configures the model and the rule; ``fit``
    counts the corpus; :meth:`entropies_of` is the observable route and
    :meth:`patch` the answer.

    Parameters
    ----------
    order:
        How many preceding bytes the next-byte distribution is conditioned on.
    threshold:
        In bits. Under ``GLOBAL_THRESHOLD`` a position whose entropy exceeds
        this starts a patch; under ``RELATIVE_INCREASE`` a position whose
        entropy exceeds the previous one's by more than this does.
    smoothing:
        Added to every byte value's count in every context, so no continuation
        has probability zero and an unseen context is uniform. Add-one by
        default, which on a small corpus keeps every entropy near 8 bits;
        see the module docstring before choosing a threshold.
    rule:
        Which of the two rules decides a boundary.
    """

    order: int = Field(default=2, ge=1)
    threshold: float = Field(gt=0)
    smoothing: float = Field(default=1.0, gt=0)
    rule: PatchingRule = PatchingRule.GLOBAL_THRESHOLD

    _counts_by_context: dict[tuple[int, ...], dict[int, int]] = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Count, for every context of ``order`` bytes, what followed it.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty or every text is blank.
        """
        counts_by_context: dict[tuple[int, ...], dict[int, int]] = {}
        for text in Corpus.of(corpus):
            values = self._padded_bytes_of(text)
            for position in range(self.order, len(values)):
                context = tuple(values[position - self.order : position])
                following = values[position]
                counts = counts_by_context.setdefault(context, {})
                counts[following] = counts.get(following, 0) + 1

        self._counts_by_context = counts_by_context
        self._mark_fitted()
        return self

    @property
    def n_contexts(self) -> int:
        """How many distinct contexts the corpus contained.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return len(self._counts_by_context)

    def entropies_of(self, text: str) -> tuple[float, ...]:
        """The entropy, in bits, of the next-byte distribution at each byte.

        Position ``i`` holds the uncertainty about byte ``i`` given the
        ``order`` bytes before it, the start of the text padded with
        :data:`START_OF_TEXT`. Every value lies in ``[0, 8]``.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``text`` is not a string.
        """
        self._check_fitted()
        values = self._padded_bytes_of(checked_text(text))
        return tuple(
            self._entropy_after(tuple(values[position - self.order : position]))
            for position in range(self.order, len(values))
        )

    def patch(self, text: str) -> Patches:
        """The text's bytes, a new patch beginning wherever the rule fires.

        The first byte always begins a patch.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``text`` is not a string.
        """
        self._check_fitted()
        byte_values = checked_text(text).encode("utf-8")
        entropies = self.entropies_of(text)
        starts = [
            position
            for position in range(len(entropies))
            if position == 0 or self._begins_patch(entropies, position)
        ]
        return patches_starting_at(byte_values, starts)

    def _begins_patch(self, entropies: Sequence[float], position: int) -> bool:
        if self.rule is PatchingRule.RELATIVE_INCREASE:
            return entropies[position] - entropies[position - 1] > self.threshold
        return entropies[position] > self.threshold

    def _entropy_after(self, context: tuple[int, ...]) -> float:
        """Shannon entropy in bits of the smoothed distribution after ``context``."""
        counts = self._counts_by_context.get(context, {})
        total = sum(counts.values()) + N_BYTE_VALUES * self.smoothing
        bits = 0.0
        for value in range(N_BYTE_VALUES):
            probability = (counts.get(value, 0) + self.smoothing) / total
            bits -= probability * math.log2(probability)
        return bits

    def _padded_bytes_of(self, text: str) -> list[int]:
        """The text's bytes with ``order`` start-of-text values in front."""
        return [START_OF_TEXT] * self.order + list(text.encode("utf-8"))
