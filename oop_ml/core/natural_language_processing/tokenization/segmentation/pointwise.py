"""Pointwise segmentation: one yes-or-no per gap, decided by an averaged perceptron.

The idea, and why it exists
---------------------------
The hidden Markov segmenter decides a sentence at once: the tag of each
character depends on the tag of its neighbour, and Viterbi finds the best
whole sequence. KyTea (Neubig, Nakata and Mori, 2011) throws that dependency
away on purpose. Between every two adjacent characters there is a gap, and
each gap is asked one question -- is there a word boundary here -- answered by
a linear classifier reading only the characters around it. ``研究生命起源`` has
five gaps; the pointwise segmenter answers ``no yes no yes no`` and the words
are ``研究 / 生命 / 起源``.

The reason to give up the sequence model is *what can be learned from*. Every
count the HMM needs -- which tag starts a sentence, which tag follows which,
which tag emits which character -- requires the tag of a character to be
known, and a character's tag is known only when the boundaries on *both* sides
of it are. A corpus in which an annotator has marked some gaps and left others
unmarked gives the HMM no complete tag anywhere near an unmarked gap. To a
pointwise classifier the same corpus is simply fewer training instances: each
marked gap is one example, each unmarked gap is nothing, and no example needs
any other. So a corpus can be annotated one gap at a time, wherever the
annotator is sure or wherever the current model is least sure, which is what
lets KyTea be trained cheaply for a new domain and is the whole reason it was
written. The ``fit`` here takes fully segmented sentences because that is the
form the other segmenters in this family share; nothing in the learner would
change for partial ones, and that is the claim this docstring is making.

The features
------------
Number the characters around a gap so that ``c[-1]`` is the one just before it
and ``c[0]`` the one just after. Within ``window`` characters on either side,
each character contributes a unigram feature, ``c[-1]=研``, and each adjacent
pair a bigram feature, ``c[0..1]=究生``. The same offsets contribute the
character's *type* -- letter, digit, punctuation, Han, hiragana, katakana or
other, read from :mod:`unicodedata` -- so that a boundary between a katakana
run and a Han run can be learned once rather than once per character pair. A
constant ``bias`` feature carries the base rate. Every feature is a string, and
the whole model is a mapping from those strings to weights; :func:`gap_features`
is public so that the spec can count them by hand. Positions off either end of
the run contribute nothing, which is the simplest choice and is stated so.

The learner
-----------
An averaged perceptron. The plain perceptron adds the features of a
misclassified gap to the weights, with the sign of the correct label; the
averaged one answers with the *mean* of every weight vector the plain one
passed through, which Freund and Schapire (1999) showed generalises far better
than the final one for the same cost. The mean is kept lazily: each feature
records when its weight last changed, the running total is charged
``(now - then) * old_weight`` at each change and once more at the end, and the
total over the number of steps is the averaged weight. Nothing is stored per
step. A gap whose score is exactly zero counts as a mistake during training,
since zero is right for neither label, and is *not* cut at prediction time,
since cutting nothing is the conservative error.

Worked, on one gap
------------------
With ``window = 1`` the gap in ``ab`` has seven features: ``bias``, ``c[-1]=a``,
``c[0]=b``, ``c[-1..0]=ab``, and the three type features that say all of those
are letters. Fit for one epoch on the single sentence ``a / b``, the first and
only step meets a score of zero, calls it a mistake, and adds ``+1`` to all
seven; the average over one step is that same vector, so the gap then scores
``7.0``. Fit on ``ab`` as one word instead and every weight is ``-1``, score
``-7.0``. Two gaps and one epoch pin the averaging itself, in the spec.

Order and the seed
------------------
Each epoch visits the training gaps in a freshly shuffled order drawn from
``random.Random(random_seed)``, because a perceptron's final weights depend on
the order it met its mistakes in. Two fits with the same seed are identical;
with ``None`` they need not be.

Cost is ``O(epochs * gaps * features per gap)`` and the feature count per gap
is a small constant in ``window``, so nothing here is worth optimising.
"""

from __future__ import annotations

import random
import unicodedata
from collections.abc import Sequence
from enum import StrEnum
from typing import Self

from pydantic import Field, PrivateAttr

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.exceptions import InvalidValuesError, TooFewValuesError
from oop_ml.core.natural_language_processing.tokenization.segmentation.dictionary import (  # noqa: E501
    SegmentedCorpus,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.runs import (
    RunSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import checked_text
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    NON_WHITESPACE_RUN,
)

BIAS_FEATURE = "bias"
"""The constant feature every gap carries, so the weights can learn a base rate."""


class CharacterType(StrEnum):
    """The coarse class of a character, so that a boundary rule can generalise.

    Attributes
    ----------
    HAN:
        A CJK unified or compatibility ideograph.
    HIRAGANA, KATAKANA:
        The two Japanese syllabaries, told apart by the Unicode name.
    LETTER:
        Any other letter, in Unicode's sense.
    DIGIT:
        Any Unicode number.
    PUNCTUATION:
        Any Unicode punctuation.
    OTHER:
        Everything else: symbols, marks, whitespace, controls.
    """

    HAN = "han"
    HIRAGANA = "hiragana"
    KATAKANA = "katakana"
    LETTER = "letter"
    DIGIT = "digit"
    PUNCTUATION = "punctuation"
    OTHER = "other"


def character_type_of(character: str) -> CharacterType:
    """The :class:`CharacterType` of one character, from its Unicode name and category.

    The three script types are read from the character's name, because
    Unicode's general category calls a Han character, a hiragana and a Latin
    letter all ``Lo`` or ``Ll`` and the boundary behaviour of the three could
    not be more different.
    """
    name = unicodedata.name(character, "")
    if name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")):
        return CharacterType.HAN
    if name.startswith("HIRAGANA"):
        return CharacterType.HIRAGANA
    if name.startswith("KATAKANA"):
        return CharacterType.KATAKANA

    category = unicodedata.category(character)
    if category.startswith("L"):
        return CharacterType.LETTER
    if category.startswith("N"):
        return CharacterType.DIGIT
    if category.startswith("P"):
        return CharacterType.PUNCTUATION
    return CharacterType.OTHER


def gap_features(run: str, gap: int, window: int) -> tuple[str, ...]:
    """The features of the gap between ``run[gap]`` and ``run[gap + 1]``.

    Offset ``k`` names ``run[gap + 1 + k]``, so ``c[-1]`` is the character
    before the gap and ``c[0]`` the one after. Unigrams and types are taken at
    offsets ``-window`` to ``window - 1``; bigrams and type bigrams at every
    adjacent pair within that range. Offsets off either end of the run are
    skipped.

    Raises
    ------
    InvalidValuesError
        If ``gap`` is not a gap of ``run``, or ``window`` is below one.
    """
    if window < 1:
        raise InvalidValuesError(f"the window must be at least one, got {window}")
    if not 0 <= gap < len(run) - 1:
        raise InvalidValuesError(
            f"a run of {len(run)} characters has gaps 0 to {len(run) - 2}, got {gap}"
        )

    features = [BIAS_FEATURE]
    for offset in range(-window, window):
        index = gap + 1 + offset
        if 0 <= index < len(run):
            features.append(f"c[{offset}]={run[index]}")
            features.append(f"t[{offset}]={character_type_of(run[index])}")
    for offset in range(-window, window - 1):
        index = gap + 1 + offset
        if index >= 0 and index + 1 < len(run):
            left, right = run[index], run[index + 1]
            features.append(f"c[{offset}..{offset + 1}]={left}{right}")
            features.append(
                f"t[{offset}..{offset + 1}]="
                f"{character_type_of(left)},{character_type_of(right)}"
            )
    return tuple(features)


class _LabelledGap:
    """One training instance: a gap's features and whether a boundary sits there."""

    __slots__ = ("features", "label")

    def __init__(self, features: tuple[str, ...], label: int) -> None:
        self.features = features
        self.label = label


class PointwiseSegmenter(RunSegmenter, Fittable):
    """A boundary-or-not decision at every gap, by an averaged perceptron.

    Parameters
    ----------
    window:
        How many characters on each side of a gap the features read.
    epochs:
        How many passes over the training gaps to make.
    random_seed:
        Seeds the per-epoch shuffle of the training gaps. Two fits with one
        seed are identical.
    """

    window: int = Field(default=3, ge=1)
    epochs: int = Field(default=10, ge=1)
    random_seed: int | None = None

    _weights: dict[str, float] = PrivateAttr()
    _n_features: int = PrivateAttr()
    _n_updates_by_epoch: tuple[int, ...] = PrivateAttr()

    def fit(self, sentences: SegmentedCorpus | Sequence[Sequence[str]]) -> Self:
        """Learn the gap classifier from segmented sentences.

        Raises
        ------
        InvalidValuesError
            If ``sentences`` is one string, a sentence is one string, a word is
            not a string, or a word contains whitespace.
        EmptyValuesError
            If there are no sentences, a sentence is empty, or a word is empty.
        TooFewValuesError
            If no sentence has two characters, so there is no gap to learn from.
        """
        gaps = self._labelled_gaps(SegmentedCorpus.of(sentences))
        if not gaps:
            raise TooFewValuesError(
                "every sentence is a single character, so there is no gap between "
                "characters to learn a boundary decision from"
            )

        n_features = len({feature for gap in gaps for feature in gap.features})
        generator = random.Random(self.random_seed)
        weights: dict[str, float] = {}
        totals: dict[str, float] = {}
        last_changed: dict[str, int] = {}
        order = list(range(len(gaps)))
        step = 0
        n_updates_by_epoch: list[int] = []

        for _ in range(self.epochs):
            generator.shuffle(order)
            n_updates = 0
            for index in order:
                gap = gaps[index]
                step += 1
                score = sum(weights.get(feature, 0.0) for feature in gap.features)
                if score * gap.label > 0:
                    continue
                n_updates += 1
                for feature in gap.features:
                    current = weights.get(feature, 0.0)
                    totals[feature] = (
                        totals.get(feature, 0.0)
                        + (step - last_changed.get(feature, 0)) * current
                    )
                    weights[feature] = current + gap.label
                    last_changed[feature] = step
            n_updates_by_epoch.append(n_updates)

        averaged = {
            feature: (totals[feature] + (step - last_changed[feature] + 1) * current)
            / step
            for feature, current in weights.items()
        }

        self._weights = averaged
        self._n_features = n_features
        self._n_updates_by_epoch = tuple(n_updates_by_epoch)
        self._mark_fitted()
        return self

    @property
    def n_features(self) -> int:
        """How many distinct features the training gaps produced.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._n_features

    @property
    def n_updates_by_epoch(self) -> tuple[int, ...]:
        """How many gaps each epoch corrected; a final zero means the fit converged.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._n_updates_by_epoch

    def weight_of(self, feature: str) -> float:
        """The averaged weight of one feature string, or zero for one never updated.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        self._check_fitted()
        return self._weights.get(feature, 0.0)

    def gap_scores(self, text: str) -> tuple[float, ...]:
        """One score per gap between adjacent non-whitespace characters of one run.

        Positive means a boundary. The observed route: :meth:`split` cuts
        exactly where these are positive. A gap that spans whitespace is
        already a boundary and has no score.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``text`` is not a string.
        """
        self._check_fitted()
        scores: list[float] = []
        for match in NON_WHITESPACE_RUN.finditer(checked_text(text)):
            scores.extend(self._scores_of_run(match.group()))
        return tuple(scores)

    def _words_of_run(self, run: str) -> tuple[str, ...]:
        self._check_fitted()
        pieces: list[str] = []
        start = 0
        for gap, score in enumerate(self._scores_of_run(run)):
            if score > 0:
                pieces.append(run[start : gap + 1])
                start = gap + 1
        pieces.append(run[start:])
        return tuple(pieces)

    def _scores_of_run(self, run: str) -> list[float]:
        return [
            sum(
                self._weights.get(feature, 0.0)
                for feature in gap_features(run, gap, self.window)
            )
            for gap in range(len(run) - 1)
        ]

    def _labelled_gaps(self, corpus: SegmentedCorpus) -> list[_LabelledGap]:
        """Every gap of every sentence: ``+1`` at a boundary, ``-1`` elsewhere."""
        gaps: list[_LabelledGap] = []
        for sentence in corpus:
            run = "".join(sentence)
            boundaries: set[int] = set()
            end = 0
            for word in sentence[:-1]:
                end += len(word)
                boundaries.add(end)
            for gap in range(len(run) - 1):
                label = 1 if gap + 1 in boundaries else -1
                gaps.append(_LabelledGap(gap_features(run, gap, self.window), label))
        return gaps
