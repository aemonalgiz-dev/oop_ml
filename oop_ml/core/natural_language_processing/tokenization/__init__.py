"""Tokenization: from text to the ids a model reads, and back.

The families sit in subpackages named for the question each answers.
``word_level`` decides where the words are for a script that writes spaces;
``segmentation`` does the same for one that does not; ``subword`` learns a
vocabulary of pieces from a corpus; ``morphology`` constrains those pieces to
respect the units a language is actually built from; ``characters`` and
``hashing`` give up on a learned vocabulary from two directions; and
``quantisation`` is the same problem asked of vectors instead of strings.

Two contracts run through all of them. A
:class:`~oop_ml.core.natural_language_processing.tokenization.tokenizer.PreTokenizer`
reads one text and answers with
:class:`~oop_ml.core.natural_language_processing.tokenization.words.Words`, each word
carrying its span in the source. A
:class:`~oop_ml.core.natural_language_processing.tokenization.tokenizer.Tokenizer`
owns a
:class:`~oop_ml.core.natural_language_processing.tokenization.vocabulary.Vocabulary`
and turns text into an
:class:`~oop_ml.core.natural_language_processing.tokenization.encoding.Encoding` and
back; the ones that learn their vocabulary are
:class:`~oop_ml.core.natural_language_processing.tokenization.tokenizer.LearnedTokenizer`
and fit the way every model in this library fits.

Named systems are provided as their mechanism
---------------------------------------------
The literature names tokenizers after products and models -- Jieba, MeCab,
spaCy, CANINE, ByT5, HuBERT -- and a product is a mechanism plus data plus a
model around it. This package implements the mechanism and nothing else. Jieba
is a dictionary lattice searched by dynamic programming with a hidden Markov
model over the unknown runs, so :class:`DictionaryLatticeSegmenter` and
:class:`HiddenMarkovSegmenter` are Jieba; its 350,000-entry dictionary is data a
caller supplies. CANINE hashes each codepoint into several buckets, so
:class:`HashedCharacterTokenizer` is CANINE's tokenizer; the transformer that
reads the buckets is a network and belongs beside one.

What is declined, by name
-------------------------
A handful of named methods have no mechanism here at all, and each is listed in
:data:`NOT_PROVIDED` with the reason, following the rule the scikit backend set:
a method can be provided or it can be declined, and it cannot be forgotten. The
test ``test_every_tokenizer_is_accounted_for`` holds every item of the request
this package was built from to exactly one of the two. Every reason names the
mechanism that is missing and where it would live, because a reason that named
only an absent class would invite a reader to reach for the near miss.
"""

from __future__ import annotations

from oop_ml.core.natural_language_processing.tokenization.characters.byte import (
    ByteTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.characters.character import (
    CharacterTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.characters.patching import (
    EntropyPatcher,
    FixedSizePatcher,
    Patch,
    Patches,
    PatchingRule,
)
from oop_ml.core.natural_language_processing.tokenization.corpus import (
    Corpus,
    WordCount,
    WordCounts,
)
from oop_ml.core.natural_language_processing.tokenization.encoding import (
    Encoding,
    Token,
)
from oop_ml.core.natural_language_processing.tokenization.hashing.hashed_characters import (  # noqa: E501
    HashedCharacter,
    HashedCharacters,
    HashedCharacterTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.hashing.trigram_hashing import (  # noqa: E501
    TextActivations,
    TrigramHashTokenizer,
    WordActivations,
)
from oop_ml.core.natural_language_processing.tokenization.morphology.constrained_merges import (  # noqa: E501
    MorphemeConstrainedBytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.morphology.finite_state import (  # noqa: E501
    END,
    Analyses,
    Analysis,
    FiniteStateAnalyzer,
    Lexicon,
    LexiconEntry,
)
from oop_ml.core.natural_language_processing.tokenization.morphology.morfessor import (
    DescriptionLength,
    MorfessorBaseline,
    Morph,
    Morphs,
    MorphSegmentation,
)
from oop_ml.core.natural_language_processing.tokenization.quantisation.codebook import (
    CodeAssignment,
    Codebook,
    CodebookQuantizer,
)
from oop_ml.core.natural_language_processing.tokenization.quantisation.finite_scalar import (  # noqa: E501
    FiniteScalarQuantizer,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.dictionary import (  # noqa: E501
    DictionaryEntry,
    SegmentedCorpus,
    WordDictionary,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.hidden_markov import (  # noqa: E501
    BoundaryTag,
    HiddenMarkovSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.lattice import (
    DictionaryLatticeSegmenter,
    Lattice,
    LatticeEdge,
    LatticePath,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.maximum_matching import (  # noqa: E501
    MatchingDirection,
    MaximumMatchingSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.pointwise import (  # noqa: E501
    CharacterType,
    PointwiseSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.runs import (
    RunSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_level import (
    ByteLevelBytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (  # noqa: E501
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.greedy_coverage import (  # noqa: E501
    ChosenPiece,
    ChosenPieces,
    CoverageSegmentation,
    GreedyCoverageTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.subword.merges import (
    Merge,
    Merges,
)
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    PairScoring,
    SpelledWord,
)
from oop_ml.core.natural_language_processing.tokenization.subword.sentence_piece import (  # noqa: E501
    SentencePiece,
    SubwordAlgorithm,
)
from oop_ml.core.natural_language_processing.tokenization.subword.shortest_path import (
    ShortestPathTokenizer,
    ShortestSegmentation,
)
from oop_ml.core.natural_language_processing.tokenization.subword.super_byte_pair_encoding import (  # noqa: E501
    SuperBytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.unigram import (
    PieceTable,
    Segmentation,
    UnigramLanguageModel,
)
from oop_ml.core.natural_language_processing.tokenization.subword.vocabulary_transfer import (  # noqa: E501
    TokenMapping,
    TokenMappings,
    VocabularyTransfer,
)
from oop_ml.core.natural_language_processing.tokenization.subword.word_piece import (
    WordPiece,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import (
    LearnedTokenizer,
    PreTokenizer,
    Tokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.moses import (
    MosesPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.pattern import (
    GPT2_PATTERN,
    WORD_OR_PUNCTUATION_PATTERN,
    PatternPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.penn_treebank import (  # noqa: E501
    PennTreebankPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.rule_based import (
    RuleAndExceptionPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.unicode_words import (  # noqa: E501
    UnicodeWordPreTokenizer,
    WordBreakClass,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words

NOT_PROVIDED: dict[str, str] = {
    "conditional random field training of lattice costs": (
        "MeCab and Sudachi estimate the word and connection costs of their "
        "lattice with a conditional random field over part-of-speech features "
        "from a tagged corpus, and Juman++ scores its lattice with a recurrent "
        "language model; KyTea and the Thai and Khmer segmenters that use a "
        "sequence model train one the same way. The lattice and its search are "
        "DictionaryLatticeSegmenter, whose costs are counted from segmented "
        "text, and the pointwise alternative is PointwiseSegmenter, whose "
        "perceptron needs only boundary labels. A CRF needs a tagged corpus and "
        "a part-of-speech layer for its features to read, and neither exists in "
        "this library; it is a sequence model rather than a tokenizer, and it "
        "would belong beside the classifiers"
    ),
    "trans-tokenization": (
        "maps a target vocabulary onto a source one through word alignments "
        "over a parallel corpus, so it needs an aligner and parallel text, and "
        "the aligner is itself a statistical model. The half that needs "
        "neither, initialising each new token from the source tokens that "
        "spell it, is VocabularyTransfer"
    ),
    "SaGe": (
        "chooses a vocabulary by a contextual skip-gram objective, so building "
        "one means training skip-gram embeddings over every candidate token and "
        "scoring vocabularies by the resulting loss; the embedding training is "
        "a model and belongs to a backend. The two objective-driven builders "
        "that need no embeddings, ShortestPathTokenizer and "
        "GreedyCoverageTokenizer, are here"
    ),
    "Morfessor FlatCat": (
        "adds a hidden Markov model over four morph categories (prefix, stem, "
        "suffix and non-morpheme) to the Baseline's lexicon, and its training "
        "alternates the category tagging with the segmentation. The Baseline is "
        "MorfessorBaseline; the category model is a second learned model on top "
        "of it, and the recursive split search would have to be rewritten to "
        "score a tagging as well as a lexicon"
    ),
    "Charformer's gradient-based subword tokenization": (
        "a differentiable layer that scores candidate blocks at several sizes "
        "and softly pools them, learned by backpropagation with the model. It "
        "reads embeddings and answers embeddings, so it is a network layer that "
        "belongs beside Embedding in oop_ml.core.network, not a text-to-id step"
    ),
    "Dynamic Token Pooling": (
        "learns where to segment inside the model from a boundary predictor "
        "trained jointly with the language model, through the model's own "
        "gradient. It is a layer whose input is the model's hidden states, not "
        "text, so it belongs beside Embedding in oop_ml.core.network, and "
        "nothing about it can run before a model exists"
    ),
    "PIXEL's rendering of text as images": (
        "renders text with a font and reads patches of pixels, so its tokenizer "
        "is a rasteriser: there is no text-to-id step to implement, and the "
        "rendering needs a font and a rasteriser this library does not ship"
    ),
    "VQ-VAE and VQGAN codebook learning": (
        "the codes move under the encoder's gradient with a commitment loss, "
        "which is a network's training loop, or under k-means, which is KMeans "
        "in the numpy backend and whose centroids are exactly a Codebook. The "
        "lookup half, a vector to the id of its nearest code, is "
        "CodebookQuantizer; FiniteScalarQuantizer is the codebook that needs no "
        "learning at all"
    ),
}
"""Named methods with no mechanism here, each with the reason.

Every key is an item of the request this package was built from, and the
accounting test holds the two lists, provided and declined, to cover that
request exactly once each.
"""

__all__ = [
    # The vocabulary every family speaks
    "Word",
    "Words",
    "Vocabulary",
    "Token",
    "Encoding",
    "Corpus",
    "WordCount",
    "WordCounts",
    "PreTokenizer",
    "Tokenizer",
    "LearnedTokenizer",
    # Word level: where the words are, for a script that writes spaces
    "WhitespacePreTokenizer",
    "PatternPreTokenizer",
    "GPT2_PATTERN",
    "WORD_OR_PUNCTUATION_PATTERN",
    "PennTreebankPreTokenizer",
    "MosesPreTokenizer",
    "RuleAndExceptionPreTokenizer",
    "UnicodeWordPreTokenizer",
    "WordBreakClass",
    # Segmentation: where the words are, for a script that does not
    "DictionaryEntry",
    "WordDictionary",
    "SegmentedCorpus",
    "RunSegmenter",
    "MatchingDirection",
    "MaximumMatchingSegmenter",
    "BoundaryTag",
    "HiddenMarkovSegmenter",
    "LatticeEdge",
    "LatticePath",
    "Lattice",
    "DictionaryLatticeSegmenter",
    "CharacterType",
    "PointwiseSegmenter",
    # Subword: a vocabulary of pieces learned from a corpus
    "Merge",
    "Merges",
    "PairScoring",
    "SpelledWord",
    "BytePairEncoding",
    "ByteLevelBytePairEncoding",
    "WordPiece",
    "SuperBytePairEncoding",
    "UnigramLanguageModel",
    "PieceTable",
    "Segmentation",
    "SentencePiece",
    "SubwordAlgorithm",
    "ShortestPathTokenizer",
    "ShortestSegmentation",
    "GreedyCoverageTokenizer",
    "ChosenPiece",
    "ChosenPieces",
    "CoverageSegmentation",
    "TokenMapping",
    "TokenMappings",
    "VocabularyTransfer",
    # Morphology: pieces that respect the units a language is built from
    "MorfessorBaseline",
    "Morph",
    "Morphs",
    "MorphSegmentation",
    "DescriptionLength",
    "MorphemeConstrainedBytePairEncoding",
    "LexiconEntry",
    "Lexicon",
    "Analysis",
    "Analyses",
    "FiniteStateAnalyzer",
    "END",
    # Characters and bytes: no learned vocabulary, or the smallest one
    "CharacterTokenizer",
    "ByteTokenizer",
    "Patch",
    "Patches",
    "FixedSizePatcher",
    "PatchingRule",
    "EntropyPatcher",
    # Hashing: no vocabulary at all
    "HashedCharacter",
    "HashedCharacters",
    "HashedCharacterTokenizer",
    "WordActivations",
    "TextActivations",
    "TrigramHashTokenizer",
    # Quantisation: the same question asked of vectors
    "Codebook",
    "CodeAssignment",
    "CodebookQuantizer",
    "FiniteScalarQuantizer",
    # What is declined, by name
    "NOT_PROVIDED",
]
