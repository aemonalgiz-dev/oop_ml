"""Dictionaries and segmented sentences shared by the segmentation specs, with what is known about each.

Every number here was derived on paper before the implementation existed and
then confirmed by it; the derivations are in the module docstrings of the
segmenters and repeated beside the fixture that pins them.
"""

from __future__ import annotations

from fractions import Fraction

# ---------------------------------------------------------------------------
# The classic ambiguity: 研究生命起源 is "research / life / origin" or
# "graduate student / fate / origin", and every piece is a real word.
# ---------------------------------------------------------------------------

RESEARCH_SENTENCE = "研究生命起源"
RESEARCH_WORDS: list[str] = ["研究", "研究生", "生命", "命", "起源"]
LIFE_READING: tuple[str, ...] = ("研究", "生命", "起源")
STUDENT_READING: tuple[str, ...] = ("研究生", "命", "起源")

# Frequencies under which 研究 / 生命 / 起源 scores 10 * 8 * 5 = 400 against
# 研究生 / 命 / 起源 at 6 * 4 * 5 = 120, total 33.
LIFE_FAVOURING_FREQUENCIES: dict[str, int] = {
    "研究": 10,
    "生命": 8,
    "起源": 5,
    "研究生": 6,
    "命": 4,
}

# The same words with the other reading favoured: 20 * 12 * 5 = 1200 against
# 400, total 55.
STUDENT_FAVOURING_FREQUENCIES: dict[str, int] = {
    "研究": 10,
    "生命": 8,
    "起源": 5,
    "研究生": 20,
    "命": 12,
}

# ---------------------------------------------------------------------------
# The classic maximum-matching disagreement: "we play at the wildlife park".
# Forward is led into 在野 ("out of office") and 生动 ("vivid"), both real
# words the sentence does not contain; backward meets 动物园 whole.
# ---------------------------------------------------------------------------

ZOO_SENTENCE = "我们在野生动物园玩"
ZOO_WORDS: list[str] = [
    "我们",
    "在",
    "在野",
    "野生",
    "生动",
    "动物园",
    "物",
    "园",
    "玩",
]
ZOO_FORWARD: tuple[str, ...] = ("我们", "在野", "生动", "物", "园", "玩")
ZOO_BACKWARD: tuple[str, ...] = ("我们", "在", "野生", "动物园", "玩")

# ---------------------------------------------------------------------------
# The two-sentence corpus whose smoothed HMM tables are worked by hand in the
# hidden_markov module docstring. Tags: B E S and B M E.
# ---------------------------------------------------------------------------

TWO_SENTENCE_CORPUS: list[list[str]] = [["ab", "c"], ["abc"]]

# Initial tags: B twice, S never, smoothing 1 over the two admissible tags.
HAND_INITIAL: dict[str, Fraction] = {
    "B": Fraction(2 + 1, 2 + 2),
    "S": Fraction(0 + 1, 2 + 2),
}

# Transitions, smoothing 1 over the two admissible successors of each tag.
HAND_TRANSITIONS: dict[tuple[str, str], Fraction] = {
    ("B", "M"): Fraction(1 + 1, 2 + 2),
    ("B", "E"): Fraction(1 + 1, 2 + 2),
    ("M", "M"): Fraction(0 + 1, 1 + 2),
    ("M", "E"): Fraction(1 + 1, 1 + 2),
    ("E", "B"): Fraction(0 + 1, 1 + 2),
    ("E", "S"): Fraction(1 + 1, 1 + 2),
    ("S", "B"): Fraction(0 + 1, 0 + 2),
    ("S", "S"): Fraction(0 + 1, 0 + 2),
}

# Emissions over the alphabet {a, b, c} plus one unseen slot, four in all.
# B emitted a twice; M emitted b once; E emitted b and c once each; S emitted
# c once.
HAND_EMISSIONS: dict[tuple[str, str], Fraction] = {
    ("B", "a"): Fraction(2 + 1, 2 + 4),
    ("B", "b"): Fraction(0 + 1, 2 + 4),
    ("B", "z"): Fraction(0 + 1, 2 + 4),
    ("M", "b"): Fraction(1 + 1, 1 + 4),
    ("M", "a"): Fraction(0 + 1, 1 + 4),
    ("E", "b"): Fraction(1 + 1, 2 + 4),
    ("E", "c"): Fraction(1 + 1, 2 + 4),
    ("E", "a"): Fraction(0 + 1, 2 + 4),
    ("S", "c"): Fraction(1 + 1, 1 + 4),
    ("S", "z"): Fraction(0 + 1, 1 + 4),
}

# ---------------------------------------------------------------------------
# A corpus in which a, c, e only ever begin a word and b, d, f only ever end
# one, so that a sentence of words it never saw is still segmentable.
# ---------------------------------------------------------------------------

BOUNDARY_BEHAVIOUR_CORPUS: list[list[str]] = [["ab", "cd"], ["ab", "ef"], ["cd", "ef"]]
NOVEL_SENTENCE = "adcfeb"
NOVEL_SEGMENTATION: tuple[str, ...] = ("ad", "cf", "eb")

# A small real-script corpus for the taggers, consistent with itself: no
# character sequence is a word in one sentence and two words in another.
CJK_SEGMENTED_CORPUS: list[list[str]] = [
    ["研究", "生命", "起源"],
    ["学生", "很", "多"],
    ["生命", "很", "好"],
    ["起源", "研究"],
    ["学生", "研究", "生命"],
]

# The same corpus with one sentence that contradicts it: 研究 is a word three
# times and 研究生 once. A smoothed HMM sides with the majority at smoothing
# 1.0 and 0.5, and memorises the minority sentence at 0.1 and below.
CONTRADICTORY_CJK_CORPUS: list[list[str]] = [
    ["研究", "生命", "起源"],
    ["研究生", "很", "多"],
    ["生命", "很", "好"],
    ["起源", "研究"],
]
MINORITY_SENTENCE = "研究生很多"
MAJORITY_READING: tuple[str, ...] = ("研究", "生", "很", "多")
MINORITY_READING: tuple[str, ...] = ("研究生", "很", "多")

# ---------------------------------------------------------------------------
# Pointwise: a separable toy in which b|c and d|a are always boundaries and
# a|b, c|d never are. With window 1 its gaps produce exactly 16 features:
# four c[-1], four c[0], four bigrams, three type features, and the bias.
# ---------------------------------------------------------------------------

SEPARABLE_CORPUS: list[list[str]] = [
    ["ab", "cd"],
    ["cd", "ab"],
    ["ab"],
    ["cd"],
    ["ab", "cd", "ab"],
]
SEPARABLE_N_FEATURES_AT_WINDOW_ONE = 16

# Every gap here is between characters the corpus has seen on both sides of
# a boundary, but 命起 as a pair never occurs.
CJK_POINTWISE_CORPUS: list[list[str]] = [
    ["研究", "生命"],
    ["生命", "研究"],
    ["起源", "研究"],
    ["研究", "起源"],
]
