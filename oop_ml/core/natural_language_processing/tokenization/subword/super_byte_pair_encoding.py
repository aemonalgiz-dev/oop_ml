"""SuperBPE: byte pair encoding that stops respecting the spaces part-way through.

The boundary every subword tokenizer keeps
------------------------------------------
Byte pair encoding merges pairs *within* a word and never across one, because
the pre-tokenizer cut the text into words first and the trainer only ever sees
one word at a time. That keeps every token a piece of a word, which is tidy and
is also a limit: ``of the``, which English produces more often than most words,
can never be one token, and a model pays two positions for it every time.

Liu et al. (2025) lift the boundary, but not from the start. Stage one is
ordinary byte pair encoding, merging within words until the vocabulary holds
``transition_size`` tokens, so that the common subwords exist before anything
larger is attempted. Stage two removes the pre-tokenization: every text of the
corpus becomes one symbol sequence, its words' stage-one spellings laid end to
end with their end-of-word markers kept, and merging continues over those
sequences to ``vocabulary_size``. A pair such as ``of</w>`` followed by
``the</w>`` is now adjacent and counted, and when it merges the result is a
*superword* token spanning two words, marker and all. The order matters: lifted
from the start, the trainer would spend its first merges on ``e</w> t`` and
``s</w> a`` -- the letters that end one common word and start another -- and
never build the words themselves.

Encoding applies every merge, stage one's and stage two's, to the whole text's
symbol sequence in learned order. That is exactly right rather than
approximately so: a stage-one merge never has a marker-bearing symbol on its
left, since such a symbol ends a word and is never a left operand within one,
so applied to a whole text it can only ever join what it joined in training,
and every stage-two merge ranks after every stage-one merge, so the boundary
is lifted at the same point it was lifted while learning. Decoding is byte
pair encoding's: join the pieces, turn each marker into a space.

Worked, on a corpus of ``of the`` three times
---------------------------------------------
Two words, ``o f</w>`` and ``t h e</w>``, five symbols, each pair counted
3 times. With ``transition_size=9`` stage one learns three merges -- ties at
3 go to the lexicographically smaller pair, so ``h e</w>`` first, then
``o f</w>``, then ``t he</w>`` -- and every word is one token. With
``vocabulary_size=10`` stage two has room for one merge over the sequence
``of</w> the</w>``, and learns ``of</w>the</w>`` with count 3. ``of the`` then
encodes to one token and decodes to ``of the``, the space recovered from the
marker in the middle.

With ``transition_size=8`` stage one stops after ``he</w>`` and ``of</w>``,
and stage two starts from ``of</w> t he</w>``. Its two candidates tie at 3,
``of</w> t`` and ``t he</w>``, and the smaller pair wins: the first superword
token is ``of</w>t``, spanning the boundary and stopping mid-word, and only
the next merge completes it to ``of</w>the</w>``. Lifting the boundary forbids
nothing, so a stage-two merge may cross a word and stop anywhere.

Two things the transition is not
--------------------------------
It is not a guarantee about where stage two begins. Stage one stops early if
every remaining pair is below ``minimum_pair_frequency`` or nothing is left to
merge, and the boundary is lifted wherever that happened; stage two is given
whatever room remains up to ``vocabulary_size``, so the target is the target.
And it is not a guarantee that every stage-two merge crosses a word: the
boundary is lifted, not inverted, so a within-word pair that ranks highest
still wins. :attr:`SuperBytePairEncoding.superword_tokens` lists the tokens
that actually span one, which is a fact about the vocabulary rather than about
the stage.

Costs
-----
Stage two's "words" are whole texts, so every merge step walks every symbol of
the corpus once to count pairs and once to merge, ``O(n_merges * corpus)``. A
pair-count index updated incrementally at each merge is the usual repair, and
is an optimisation of this loop rather than a different algorithm.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Self

from pydantic import Field, PrivateAttr, model_validator

from oop_ml.core.exceptions import VocabularyTooSmallError
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.subword.merges import Merges
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    SpelledWord,
    alphabet_of,
    apply_merges,
    learn_merges,
    with_pair_merged,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import (
    LearnedTokenizer,
    PreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.network.purpose import PassPurpose


class SuperBytePairEncoding(LearnedTokenizer):
    """Byte pair encoding whose later merges may span several words.

    Parameters
    ----------
    vocabulary_size:
        How many tokens to learn in all, counting the unknown token and the
        alphabet.
    transition_size:
        The vocabulary size at which the word boundary is lifted. Merges
        learned before it lie within words; merges learned after it may not.
        Must be below ``vocabulary_size``, or there is no second stage.
    pre_tokenizer:
        Decides where the words are for stage one. Stage two ignores the
        boundaries it drew, but keeps the markers they left.
    minimum_pair_frequency:
        A pair seen fewer times than this is never merged, in either stage.
    end_of_word_marker:
        Appended to each word's last character, so a superword token shows
        where the words inside it end and decoding can put the spaces back.
    unknown_token:
        Stands in for any symbol the corpus never used.
    """

    vocabulary_size: int = Field(ge=2)
    transition_size: int = Field(ge=2)
    pre_tokenizer: PreTokenizer = Field(default_factory=WhitespacePreTokenizer)
    minimum_pair_frequency: int = Field(default=2, ge=1)
    end_of_word_marker: str = Field(default="</w>", min_length=1)
    unknown_token: str = Field(default="[UNK]", min_length=1)

    _vocabulary: Vocabulary = PrivateAttr()
    _merges: Merges = PrivateAttr()
    _n_word_merges: int = PrivateAttr()

    @model_validator(mode="after")
    def _check_the_transition_precedes_the_target(self) -> Self:
        if self.transition_size >= self.vocabulary_size:
            raise ValueError(
                f"transition_size={self.transition_size} must be below "
                f"vocabulary_size={self.vocabulary_size}, or the word boundary is "
                f"never lifted; for ordinary byte pair encoding use BytePairEncoding"
            )
        return self

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn the within-word merges, then the merges across words.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        VocabularyTooSmallError
            If ``transition_size`` is below the alphabet plus the unknown
            token, so that stage one could not even hold the symbols.
        NonUniqueTokensError
            If the unknown token collides with a symbol of the alphabet.
        """
        checked = Corpus.of(corpus)
        counts = checked.word_counts(self.pre_tokenizer)
        spelled = [
            SpelledWord(self._symbols_of(word_count.word), word_count.count)
            for word_count in counts
        ]
        alphabet = alphabet_of(spelled)

        smallest_possible = len(alphabet) + 1
        if self.transition_size < smallest_possible:
            raise VocabularyTooSmallError(
                f"transition_size={self.transition_size} cannot hold the "
                f"{len(alphabet)} symbols this corpus is spelled in plus the unknown "
                f"token; the smallest workable transition is {smallest_possible}"
            )

        word_merges = learn_merges(
            spelled,
            n_merges=self.transition_size - smallest_possible,
            minimum_pair_frequency=self.minimum_pair_frequency,
        )
        stage_one_spellings = {
            word_count.word: self._replayed(
                self._symbols_of(word_count.word), word_merges
            )
            for word_count in counts
        }

        text_counter: Counter[tuple[str, ...]] = Counter()
        for text in checked:
            sequence = tuple(
                symbol
                for word in self.pre_tokenizer.split(text).texts
                for symbol in stage_one_spellings[word]
            )
            if sequence:
                text_counter[sequence] += 1
        superword_merges = learn_merges(
            [SpelledWord(sequence, count) for sequence, count in text_counter.items()],
            n_merges=self.vocabulary_size - smallest_possible - word_merges.n_merges,
            minimum_pair_frequency=self.minimum_pair_frequency,
        )

        merges = Merges([*word_merges, *superword_merges])
        tokens = [self.unknown_token, *alphabet, *(merge.merged for merge in merges)]

        self._vocabulary = Vocabulary(tokens, unknown_token=self.unknown_token)
        self._merges = merges
        self._n_word_merges = word_merges.n_merges
        self._mark_fitted()
        return self

    @property
    def vocabulary(self) -> Vocabulary:
        """The unknown token, the alphabet in codepoint order, the within-word
        merges, then the merges learned with the boundary lifted.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._vocabulary

    @property
    def merges(self) -> Merges:
        """Every merge of both stages, in the order learned, which is the order applied.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._merges

    @property
    def n_merges(self) -> int:
        """How many merges both stages learned together. See :attr:`merges`."""
        return self.merges.n_merges

    @property
    def word_merges(self) -> Merges:
        """The merges learned within words, before the boundary was lifted."""
        return Merges(list(self.merges)[: self._n_word_merges])

    @property
    def superword_merges(self) -> Merges:
        """The merges learned after the boundary was lifted. One of these may
        still lie within a word; see :attr:`superword_tokens` for the ones that
        do not."""
        return Merges(list(self.merges)[self._n_word_merges :])

    @property
    def n_word_merges(self) -> int:
        """How many merges stage one learned."""
        self._check_fitted()
        return self._n_word_merges

    @property
    def n_superword_merges(self) -> int:
        """How many merges stage two learned."""
        return self.n_merges - self.n_word_merges

    @property
    def superword_tokens(self) -> tuple[str, ...]:
        """The tokens that span a word boundary: an end-of-word marker sits
        somewhere before their end. A marker occurrence that does not end at the
        token's last character lies wholly inside ``token[:-1]``."""
        return tuple(
            token for token in self.vocabulary if self.end_of_word_marker in token[:-1]
        )

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        self._check_fitted()
        symbols = [
            symbol
            for word in self.pre_tokenizer.split(text).texts
            for symbol in self._symbols_of(word)
        ]
        return apply_merges(symbols, self._merges)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return "".join(pieces).replace(self.end_of_word_marker, " ").rstrip(" ")

    def _symbols_of(self, word: str) -> tuple[str, ...]:
        """A word as characters, the last one carrying the end-of-word marker."""
        return (*word[:-1], word[-1] + self.end_of_word_marker)

    @staticmethod
    def _replayed(symbols: tuple[str, ...], merges: Merges) -> tuple[str, ...]:
        """The spelling the trainer left: every merge applied to every occurrence,
        in learned order. The one-occurrence-at-a-time route the encoder takes
        reaches the same spelling, and the spec says so."""
        for merge in merges:
            symbols = with_pair_merged(symbols, merge)
        return symbols
