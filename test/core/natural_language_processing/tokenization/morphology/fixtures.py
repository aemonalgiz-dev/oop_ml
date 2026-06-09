"""Corpora and hand-built morphologies shared by the morphology specs.

Every number quoted beside a fixture was computed with the implementation and
then checked against the hand formula in the module docstring, never the other
way round.
"""

from __future__ import annotations

from oop_ml.core.natural_language_processing.tokenization.morphology.finite_state import (
    END,
    FiniteStateAnalyzer,
    Lexicon,
    LexiconEntry,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words

# Three verbs in four forms each. At three repeats Morfessor finds exactly the
# three stems and the three suffixes; at five it is stuck at whole words.
INFLECTION_WORDS: list[str] = [
    "walk",
    "walks",
    "walked",
    "walking",
    "talk",
    "talks",
    "talked",
    "talking",
    "play",
    "plays",
    "played",
    "playing",
]


def inflection_corpus(repeats: int) -> list[str]:
    """The twelve inflected forms, each ``repeats`` times, as one text."""
    return [" ".join(word for word in INFLECTION_WORDS for _ in range(repeats))]


INFLECTION_CORPUS: list[str] = inflection_corpus(3)

# Six words sharing no substring of two or more characters: nothing to gain
# from any split.
UNRELATED_WORDS: list[str] = ["cat", "dog", "fish", "bird", "house", "tree"]
UNRELATED_CORPUS: list[str] = [
    " ".join(word for word in UNRELATED_WORDS for _ in range(3))
]

# Two prefixes and two roots, every combination, so plain byte pair encoding
# has a frequent pair spanning the prefix-root boundary to merge.
UNDO_WORDS: list[str] = ["undo", "untie", "redo", "retie"]
UNDO_MORPHS: dict[str, tuple[str, ...]] = {
    "undo": ("un", "do"),
    "untie": ("un", "tie"),
    "redo": ("re", "do"),
    "retie": ("re", "tie"),
}
UNDO_CORPUS: list[str] = [" ".join(word for word in UNDO_WORDS for _ in range(10))]


class MorphTable(PreTokenizer):
    """A morph splitter that looks each word up; a word it lacks stays whole.

    The smallest honest ``morph_splitter``: the cuts are stated, so a spec can
    reason about exactly which boundaries the merges must respect.
    """

    morphs_by_word: dict[str, tuple[str, ...]]

    def _words_of(self, text: str) -> Words:
        words: list[Word] = []
        position = 0
        for morph in self.morphs_by_word.get(text, (text,)):
            words.append(Word(morph, position, position + len(morph)))
            position += len(morph)
        return Words(words)


UNDO_TABLE = MorphTable(morphs_by_word=UNDO_MORPHS)

# A toy English morphology. Bare stems reach END through a second root entry,
# which is the lexc idiom for a zero suffix.
VERB_SUFFIXES = Lexicon(
    "VerbSuffix",
    [
        LexiconEntry("s", "+3SG", END),
        LexiconEntry("ed", "+PAST", END),
        LexiconEntry("ing", "+PROG", END),
    ],
)
NOUN_SUFFIXES = Lexicon("NounSuffix", [LexiconEntry("s", "+PL", END)])

# Naive: ``bake`` continues to every verb suffix, so ``baked`` fails on
# ``bakeed`` and ``bakeed`` itself is accepted.
NAIVE_ENGLISH = FiniteStateAnalyzer(
    root="Root",
    lexicons=(
        Lexicon(
            "Root",
            [
                LexiconEntry("walk", "walk+V", "VerbSuffix"),
                LexiconEntry("walk", "walk+V", END),
                LexiconEntry("talk", "talk+V", "VerbSuffix"),
                LexiconEntry("talk", "talk+V", END),
                LexiconEntry("bake", "bake+V", "VerbSuffix"),
                LexiconEntry("bake", "bake+V", END),
                LexiconEntry("walk", "walk+N", "NounSuffix"),
                LexiconEntry("walk", "walk+N", END),
            ],
        ),
        VERB_SUFFIXES,
        NOUN_SUFFIXES,
    ),
)

# Repaired: the allomorph ``bak`` takes the vowel-initial suffixes and the full
# stem takes only ``s``, which is what a two-level e-deletion rule would compile
# to.
REPAIRED_ENGLISH = FiniteStateAnalyzer(
    root="Root",
    lexicons=(
        Lexicon(
            "Root",
            [
                LexiconEntry("walk", "walk+V", "VerbSuffix"),
                LexiconEntry("walk", "walk+V", END),
                LexiconEntry("talk", "talk+V", "VerbSuffix"),
                LexiconEntry("talk", "talk+V", END),
                LexiconEntry("bake", "bake+V", "ConsonantSuffix"),
                LexiconEntry("bake", "bake+V", END),
                LexiconEntry("bak", "bake+V", "VowelSuffix"),
                LexiconEntry("walk", "walk+N", "NounSuffix"),
                LexiconEntry("walk", "walk+N", END),
            ],
        ),
        VERB_SUFFIXES,
        Lexicon("ConsonantSuffix", [LexiconEntry("s", "+3SG", END)]),
        Lexicon(
            "VowelSuffix",
            [LexiconEntry("ed", "+PAST", END), LexiconEntry("ing", "+PROG", END)],
        ),
        NOUN_SUFFIXES,
    ),
)

# ``a`` and ``aa``, each continuing to itself or to the end: a word of n ``a``s
# has as many analyses as there are ordered ways to write n as ones and twos,
# which is the (n + 1)th Fibonacci number.
ONES_AND_TWOS = FiniteStateAnalyzer(
    root="Root",
    lexicons=(
        Lexicon(
            "Root",
            [
                LexiconEntry("a", "1", "Root"),
                LexiconEntry("a", "1", END),
                LexiconEntry("aa", "2", "Root"),
                LexiconEntry("aa", "2", END),
            ],
        ),
    ),
)
