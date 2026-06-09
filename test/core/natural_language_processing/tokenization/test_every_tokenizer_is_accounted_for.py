"""Every tokenizer the request named is provided as its mechanism or declined by name.

The package was built from a list of seven families and some fifty named
methods, products and models. This test is that list, one entry per item,
mapped to the classes that implement the item's mechanism -- or to nothing,
which means the package must say why in ``NOT_PROVIDED``. It is the tokenization
counterpart of ``test/contract/test_every_model_is_accounted_for.py``: an item
can be provided or it can be declined, and it cannot be quietly dropped, and it
cannot be both.

The mapping is deliberately by mechanism rather than by product. Jieba is a
dictionary lattice with a hidden Markov model over the unknown runs, so Jieba
is provided by those two classes, and its dictionary is data. That is the rule
the package's docstring states, and this is where it is checked.
"""

from __future__ import annotations

import pytest

from oop_ml.core.natural_language_processing import tokenization

REQUESTED: dict[str, tuple[str, ...]] = {
    # 1. Word-level, rule-based pre-tokenizers
    "whitespace splitting": ("WhitespacePreTokenizer",),
    "regex or pattern splitting, GPT-2's pattern included": ("PatternPreTokenizer",),
    "Penn Treebank tokenizer": ("PennTreebankPreTokenizer",),
    "Moses tokenizer": ("MosesPreTokenizer",),
    "NLTK and spaCy rule-plus-exception-list tokenizers": (
        "RuleAndExceptionPreTokenizer",
        "PennTreebankPreTokenizer",
    ),
    "UAX #29 Unicode word segmentation": ("UnicodeWordPreTokenizer",),
    # 2. Dictionary and statistical word segmentation
    "maximum matching, forward, backward and bidirectional": (
        "MaximumMatchingSegmenter",
        "MatchingDirection",
    ),
    "Jieba's dictionary lattice with a hidden Markov model for unknown runs": (
        "DictionaryLatticeSegmenter",
        "HiddenMarkovSegmenter",
    ),
    "MeCab, Juman++ and Sudachi's lattice search": ("DictionaryLatticeSegmenter",),
    "conditional random field training of lattice costs": (),
    "KyTea's pointwise classification": ("PointwiseSegmenter",),
    "Khmer and Thai dictionary-lattice and classifier segmentation": (
        "DictionaryLatticeSegmenter",
        "PointwiseSegmenter",
    ),
    # 3. Subword
    "byte pair encoding": ("BytePairEncoding",),
    "byte-level byte pair encoding": ("ByteLevelBytePairEncoding",),
    "WordPiece": ("WordPiece",),
    "unigram language model": ("UnigramLanguageModel",),
    "SentencePiece": ("SentencePiece", "SubwordAlgorithm"),
    "BPE-dropout": ("BytePairEncoding",),
    "subword regularisation": ("UnigramLanguageModel",),
    "fast vocabulary transfer": ("VocabularyTransfer",),
    "trans-tokenization": (),
    "PathPiece": ("ShortestPathTokenizer",),
    "SaGe": (),
    "GreedTok": ("GreedyCoverageTokenizer",),
    "SuperBPE": ("SuperBytePairEncoding",),
    # 4. Morphology-aware
    "Morfessor Baseline": ("MorfessorBaseline",),
    "Morfessor FlatCat": (),
    "Morfessor EM+Prune": ("UnigramLanguageModel",),
    "morphologically informed BPE and MorphPiece": (
        "MorphemeConstrainedBytePairEncoding",
    ),
    "finite-state morphological analysis, HFST and Buckwalter": (
        "FiniteStateAnalyzer",
        "Lexicon",
        "LexiconEntry",
    ),
    # 5. Character- and byte-level
    "character tokenization": ("CharacterTokenizer",),
    "CANINE's hashed characters": ("HashedCharacterTokenizer",),
    "ByT5's byte tokenization": ("ByteTokenizer",),
    "MegaByte and Hourglass fixed patching": ("FixedSizePatcher",),
    "Byte Latent Transformer entropy patching": ("EntropyPatcher", "PatchingRule"),
    # 6. Tokenizer-free and learned segmentation
    "Charformer's gradient-based subword tokenization": (),
    "Dynamic Token Pooling": (),
    "T-FREE trigram hashing": ("TrigramHashTokenizer",),
    # 7. Non-text analogues
    "PIXEL's rendering of text as images": (),
    "VQ-VAE and VQGAN codebook lookup": ("CodebookQuantizer", "Codebook"),
    "VQ-VAE and VQGAN codebook learning": (),
    "finite scalar quantisation": ("FiniteScalarQuantizer",),
    "HuBERT and wav2vec 2.0 speech units": ("CodebookQuantizer",),
}

PROVIDED = {item: classes for item, classes in REQUESTED.items() if classes}
DECLINED = tuple(item for item, classes in REQUESTED.items() if not classes)


def test_the_request_is_not_empty() -> None:
    """A guard on the guard: an emptied table would pass everything below."""
    assert len(REQUESTED) >= 40
    assert len(PROVIDED) >= 30
    assert len(DECLINED) >= 5


@pytest.mark.parametrize("item", sorted(PROVIDED))
def test_a_provided_item_is_exported_and_not_declined(item: str) -> None:
    for class_name in PROVIDED[item]:
        assert hasattr(tokenization, class_name), (
            f"{item!r} is said to be provided by {class_name}, which the package "
            f"does not export"
        )
        assert class_name in tokenization.__all__
    assert item not in tokenization.NOT_PROVIDED, (
        f"{item!r} is both provided and declined; resolve the contradiction"
    )


@pytest.mark.parametrize("item", DECLINED)
def test_a_declined_item_names_a_reason(item: str) -> None:
    assert item in tokenization.NOT_PROVIDED, (
        f"{item!r} is neither provided nor declined; add the mechanism or a "
        f"reason to NOT_PROVIDED"
    )
    assert tokenization.NOT_PROVIDED[item].strip()


def test_nothing_is_declined_that_was_not_requested() -> None:
    assert set(tokenization.NOT_PROVIDED) <= set(REQUESTED)


def test_every_reason_names_where_the_mechanism_would_live() -> None:
    """A reason that names only an absent class invites a reader to reach for
    the near miss. Each one here names a class that is present, a package
    where the missing half belongs, or the data it would need."""
    for item, reason in tokenization.NOT_PROVIDED.items():
        names_something = any(
            marker in reason
            for marker in (
                "Segmenter",
                "Tokenizer",
                "Quantizer",
                "Baseline",
                "VocabularyTransfer",
                "backend",
                "network",
                "font",
            )
        )
        assert names_something, f"the reason for {item!r} names no mechanism"


def test_every_export_is_a_real_attribute() -> None:
    missing = [name for name in tokenization.__all__ if not hasattr(tokenization, name)]

    assert not missing
