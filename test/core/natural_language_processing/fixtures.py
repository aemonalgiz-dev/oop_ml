"""Texts shared by the tokenization specs, each paired with what is known about it.

The subword trainers are pinned against Sennrich, Haddow and Birch's own toy
corpus, because its merge sequence can be worked by hand and is worked by hand
in the byte pair encoding module docstring. Every count below was derived on
paper before the implementation existed and then confirmed by it.
"""

from __future__ import annotations

# Sennrich et al. (2016), section 3.2: low x5, lower x2, newest x6, widest x3.
SENNRICH_WORD_COUNTS: dict[str, int] = {
    "low": 5,
    "lower": 2,
    "newest": 6,
    "widest": 3,
}

SENNRICH_CORPUS: list[str] = [
    " ".join(word for word, count in SENNRICH_WORD_COUNTS.items() for _ in range(count))
]

# The eleven symbols the corpus is spelled in once every word's last character
# carries the end-of-word marker, in codepoint order.
SENNRICH_ALPHABET: list[str] = [
    "d",
    "e",
    "i",
    "l",
    "n",
    "o",
    "r</w>",
    "s",
    "t</w>",
    "w",
    "w</w>",
]

# The first ten merges, with the weighted pair count that chose each. Ties go
# to the lexicographically smaller pair: (e, s) beats (s, t</w>) at 9, and
# (e, w) beats (n, e) and (w, est</w>) at 6.
SENNRICH_FIRST_TEN_MERGES: list[tuple[str, str, int]] = [
    ("e", "s", 9),
    ("es", "t</w>", 9),
    ("l", "o", 7),
    ("e", "w", 6),
    ("ew", "est</w>", 6),
    ("n", "ewest</w>", 6),
    ("lo", "w</w>", 5),
    ("d", "est</w>", 3),
    ("i", "dest</w>", 3),
    ("w", "idest</w>", 3),
]

# A short prose corpus for the splitters and the pre-tokenizer specs.
PROSE_CORPUS: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "The dog didn't mind; it was asleep.",
    "Foxes, unlike dogs, are rarely asleep at noon.",
]
