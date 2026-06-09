"""Corpora shared by the count-based embedding specs, each with what is known about it.

The two-topic corpus is a design, not prose
--------------------------------------------
Twelve documents about cooking and twelve about sailing, each topic spelled
from ten words of its own, sharing only ``and``, ``the`` and ``we``. Document
``i`` of a topic uses that topic's words ``i .. i + 4`` modulo ten, so every
topic word keeps the same company as every other and appears in six of the
twelve documents. The sentences read as lists because that is what they are.

The design is what makes the claims below testable. A term-document matrix
is non-negative, so its leading singular direction is the one every document
shares (Perron's theorem, applied to ``X X^T``) and every document has the
same sign on it. The topic contrast can only be the *second* direction, and
it is the second direction only if no structure inside a topic carries more
variance than the difference between the topics. Prose written naturally has
such structure -- ``bake``/``oven`` against ``whisk``/``eggs`` -- and on two
earlier drafts the second component split those sub-themes instead. The cyclic
design removes them.

The function words are kept to one per document for the opposite reason. Raw
co-occurrence counts, which random indexing reads unweighted, are dominated by
whatever word every other word sits beside: with five function words per
document the cooking and sailing words came out 0.69 within a topic and 0.67
across it, which is no separation at all. One per document is enough coupling
to give the term-document matrix a single leading direction and little enough
that a topic word's context is mostly its topic.
"""

from __future__ import annotations

COOKING_WORDS: list[str] = [
    "flour",
    "sugar",
    "butter",
    "eggs",
    "oven",
    "bake",
    "stir",
    "whisk",
    "dough",
    "pan",
]

SAILING_WORDS: list[str] = [
    "sail",
    "wind",
    "boat",
    "harbour",
    "anchor",
    "tide",
    "mast",
    "rope",
    "deck",
    "crew",
]

SHARED_WORDS: list[str] = ["and", "the", "we"]

COOKING_DOCUMENTS: list[str] = [
    "flour sugar and butter eggs oven",
    "sugar butter eggs the oven bake",
    "we butter eggs oven bake stir",
    "eggs oven and bake stir whisk",
    "oven bake stir the whisk dough",
    "we bake stir whisk dough pan",
    "stir whisk and dough pan flour",
    "whisk dough pan the flour sugar",
    "we dough pan flour sugar butter",
    "pan flour and sugar butter eggs",
    "flour sugar butter the eggs oven",
    "we sugar butter eggs oven bake",
]

SAILING_DOCUMENTS: list[str] = [
    "sail wind and boat harbour anchor",
    "wind boat harbour the anchor tide",
    "we boat harbour anchor tide mast",
    "harbour anchor and tide mast rope",
    "anchor tide mast the rope deck",
    "we tide mast rope deck crew",
    "mast rope and deck crew sail",
    "rope deck crew the sail wind",
    "we deck crew sail wind boat",
    "crew sail and wind boat harbour",
    "sail wind boat the harbour anchor",
    "we wind boat harbour anchor tide",
]

TWO_TOPIC_CORPUS: list[str] = COOKING_DOCUMENTS + SAILING_DOCUMENTS
N_COOKING_DOCUMENTS = len(COOKING_DOCUMENTS)

# Three sentences, window 1, uniform weighting: the pointwise mutual information
# module docstring works every count by hand. Vocabulary, commonest first with
# ties alphabetical: the (3), cat (2), sat (2), dog (1), ran (1).
TINY_PMI_CORPUS: list[str] = ["the cat sat", "the dog sat", "the cat ran"]

# Four documents whose term-document matrix has singular values sqrt(7),
# sqrt(3), 1, 1 exactly; worked in the latent semantic analysis docstring.
TINY_LSA_CORPUS: list[str] = [
    "the cat sat",
    "the cat ran",
    "the boat sailed",
    "the boat sank",
]

# The term-document module docstring's example: ``the``, ``cat`` and ``sat``
# in two documents of three, ``dog`` and ``a`` in one.
THREE_DOCUMENTS: list[str] = ["the cat sat", "the dog sat", "a cat"]
