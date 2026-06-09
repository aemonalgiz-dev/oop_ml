"""Corpora shared by the prediction-based embedding specs, paired with what they hold.

Two word lists that never mix, so a model that reads context has something
to find: sentences are drawn from one list or the other, never both, and the
claim every spec makes is that words from one list end up nearer to each
other than to words from the other. The lists are chosen so that FastText has
something to find as well. The verb list holds inflections that share pieces,
``play``, ``played``, ``plays``, ``player``, and it leaves out ``playing``, so
that word is unseen while every piece of it was seen in some other form. The
finance list shares no piece of ``playing`` at all.
"""

from __future__ import annotations

import random

VERB_WORDS: tuple[str, ...] = (
    "play",
    "played",
    "plays",
    "player",
    "stay",
    "stayed",
    "stays",
    "staying",
    "say",
    "said",
    "says",
    "saying",
    "walk",
    "walked",
    "walks",
    "walking",
    "talk",
    "talked",
    "talks",
    "talking",
)
"""One topic: twenty verb forms, several sharing pieces with ``playing``."""

FINANCE_WORDS: tuple[str, ...] = (
    "stock",
    "bond",
    "market",
    "price",
    "trade",
    "fund",
    "bank",
    "cash",
    "debt",
    "yield",
    "share",
)
"""The other topic: eleven nouns sharing no piece of ``playing``."""

UNSEEN_VERB = "playing"
"""Absent from every sentence; every one of its pieces appears in a verb form."""


def two_topic_sentences(n_sentences_per_topic: int = 100, seed: int = 11) -> list[str]:
    """Sentences of five to eight words, each drawn from one list only.

    Alternates topics so that neither list is clustered at one end of the
    corpus, and draws with a seeded ``random.Random`` so that every spec sees
    the same texts.
    """
    draw = random.Random(seed)
    sentences: list[str] = []
    for _ in range(n_sentences_per_topic):
        for words in (VERB_WORDS, FINANCE_WORDS):
            length = draw.randint(5, 8)
            sentences.append(" ".join(draw.choice(words) for _ in range(length)))
    return sentences


TWO_TOPIC_CORPUS: list[str] = two_topic_sentences()
"""Two hundred sentences, a hundred from each list, alternating."""
