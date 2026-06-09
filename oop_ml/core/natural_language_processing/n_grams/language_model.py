"""A language model from counts: the probability of the next word given the
last few, and everything that follows from having it.

What the model is
-----------------
``P(w_i | w_1 ... w_{i-1})`` is the probability of a sentence, one word at a
time, and it is unknowable in full because almost every prefix is new. The
n-gram model replaces the whole prefix with its last ``n - 1`` words, so a
trigram model believes ``P(sat | the cat sat on the) = P(sat | on the)``. That
is the Markov assumption, false and useful: it makes the model a table of
counts, and a table of counts can be filled from any text and read in constant
time. The order is the whole trade. A unigram model knows only which words are
common; a trigram model knows which words follow which; a five-gram model
knows phrases and has seen almost none of the ones a new text contains, which
is where the smoothing comes in.

What one fitted model answers
-----------------------------
Four questions, and each is a method rather than a formula a caller assembles:

* :meth:`probability_of` -- one number, ``P(word | context)``, with the
  context reduced to the last ``order - 1`` words and padded with sentence
  starts if there are fewer.
* :meth:`next_word_distribution` -- the same number for every word of the
  vocabulary at once, which is what generation samples from and what entropy
  is read off.
* :meth:`evaluate` -- the probability of whole held-out texts, as a perplexity
  and a cross-entropy, with the unknown-word share reported beside them.
* :meth:`generate` -- a sentence drawn from the model, word by word from the
  start markers until it draws the end marker or runs out of patience.

Worked, on three sentences
--------------------------
``the cat sat / the cat ran / a dog sat``, a bigram model, maximum likelihood.
``P(the | <s>) = 2 / 3``, ``P(cat | the) = 2 / 2``, ``P(sat | cat) = 1 / 2``,
``P(</s> | sat) = 2 / 2``, so the training sentence ``the cat sat`` has
probability ``1 / 3`` and, over its four predicted words, perplexity
``3 ** (1 / 4) = 1.3161``. The same model gives ``the dog sat`` probability
zero, because ``the dog`` never occurred, and perplexity ``inf``. Under
Laplace smoothing with eight words in the vocabulary (the six seen, the end
marker and the unknown word) the first sentence's probability falls to
``(3 / 11)(3 / 10)(2 / 10)(3 / 10) = 54 / 11000`` and its perplexity rises to
``3.7779``, and the second sentence becomes finite, which is the trade every
smoothing method makes.

Rare words become the unknown word before counting
--------------------------------------------------
A word below ``minimum_count`` is replaced by ``<unk>`` in every sentence
before any n-gram is counted, so the model learns how often an unknown word
appears and what tends to surround one. Without that, ``<unk>`` would be in
the vocabulary with a count of zero and every held-out word the model never
saw would get whatever the smoothing gives an n-gram seen never, which is the
same small number for all of them. Replacing rare training words is how a
closed vocabulary learns about the open one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.exceptions import InvalidValuesError, UndefinedMetricError
from oop_ml.core.natural_language_processing.embeddings.embedder import (
    TokenisedCorpus,
)
from oop_ml.core.natural_language_processing.n_grams.counts import NGramCounts
from oop_ml.core.natural_language_processing.n_grams.distribution import (
    NextWordDistribution,
)
from oop_ml.core.natural_language_processing.n_grams.evaluation import (
    LanguageModelEvaluation,
)
from oop_ml.core.natural_language_processing.n_grams.grams import (
    RESERVED_WORDS,
    SENTENCE_END,
    SENTENCE_START,
    UNKNOWN_WORD,
    padded,
)
from oop_ml.core.natural_language_processing.n_grams.smoothing.base import (
    AdditiveSmoothing,
    Smoothing,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import (
    PreTokenizer,
    checked_text,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (  # noqa: E501
    WhitespacePreTokenizer,
)


class NGramLanguageModel(Fittable):
    """``P(word | the last order - 1 words)``, from counts, under a smoothing rule.

    Parameters
    ----------
    order:
        How many words each n-gram spans: one for a unigram model, two for a
        bigram model. The context is ``order - 1`` words.
    smoothing:
        How the counts become probabilities. Laplace's additive rule by
        default, because it is the one whose arithmetic can be checked by
        hand. Maximum likelihood is the other, kept as the control. The rules
        worth reaching for in practice, interpolated Kneser-Ney above all, are
        not implemented here; the marginals they need are counted and waiting.
    pre_tokenizer:
        Decides where the words are in each text.
    minimum_count:
        A word seen fewer times than this becomes the unknown word before
        counting, so the model learns what an unknown word looks like.
    """

    order: int = Field(default=2, ge=1)
    smoothing: Smoothing = Field(default_factory=AdditiveSmoothing)
    pre_tokenizer: PreTokenizer = Field(default_factory=WhitespacePreTokenizer)
    minimum_count: int = Field(default=1, ge=1)

    _counts: NGramCounts = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Count every n-gram of ``corpus`` up to ``order`` and return ``self``.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string, holds a non-string, or contains
            one of the reserved markers ``<s>``, ``</s>``, ``<unk>`` as a
            word.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        TooFewValuesError
            If no word reaches ``minimum_count``.
        """
        tokenised = TokenisedCorpus.from_texts(corpus, self.pre_tokenizer)
        reserved = sorted(
            word for word in tokenised.word_counts.words if word in RESERVED_WORDS
        )
        if reserved:
            raise InvalidValuesError(
                f"the corpus contains the reserved marker(s) {reserved}, which "
                f"stand for sentence edges and the unknown word"
            )

        known = tokenised.vocabulary(self.minimum_count)
        sentences = [
            tuple(word if word in known else UNKNOWN_WORD for word in sentence)
            for sentence in tokenised.sentences
        ]

        self._counts = NGramCounts.from_sentences(sentences, self.order)
        self._mark_fitted()
        return self

    @property
    def counts(self) -> NGramCounts:
        """Every n-gram counted, with the marginals the smoothing reads.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._counts

    @property
    def vocabulary(self) -> Vocabulary:
        """Every word the model can predict, the end marker and the unknown
        word included. See :attr:`counts`."""
        return self.counts.vocabulary

    def probability_of(self, word: str, context: Sequence[str] = ()) -> float:
        """``P(word | context)``.

        The context is reduced to its last ``order - 1`` words, padded with
        sentence starts on the left if shorter, and every word outside the
        vocabulary, ``word`` included, is read as the unknown word.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``word`` is not a non-empty string.
        """
        counts = self.counts
        return self.smoothing.probability_of(
            counts, self._framed_context(context), self._known(word)
        )

    def next_word_distribution(
        self, context: Sequence[str] = ()
    ) -> NextWordDistribution:
        """``P(word | context)`` for every word of the vocabulary at once.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        counts = self.counts
        framed = self._framed_context(context)
        values = np.array(
            [
                self.smoothing.probability_of(counts, framed, word)
                for word in counts.vocabulary
            ]
        )
        return NextWordDistribution(
            counts.vocabulary, values, normalised=self.smoothing.is_normalised
        )

    def log_probability_of(self, text: str) -> float:
        """The natural log of the probability of ``text`` as one sentence.

        Every word is predicted from the ``order - 1`` before it, the end
        marker included; ``-inf`` if any word has probability zero.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``text`` is not a string.
        """
        self._check_fitted()
        return self._log_probability_of_words(self._known_words(text))

    def evaluate(self, texts: Sequence[str]) -> LanguageModelEvaluation:
        """Score held-out ``texts``, each as one sentence.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        UndefinedMetricError
            If the smoothing answers scores rather than probabilities, since
            a perplexity of scores is not a perplexity.
        InvalidValuesError
            If ``texts`` is a single string or holds a non-string.
        EmptyValuesError
            If ``texts`` is empty.
        """
        self._check_fitted()
        if not self.smoothing.is_normalised:
            raise UndefinedMetricError(
                f"{type(self.smoothing).__name__} answers scores that do not sum "
                f"to one, so a perplexity under it is undefined"
            )

        checked = TokenisedCorpus.from_texts(texts, self.pre_tokenizer)
        log_probability = 0.0
        n_words = 0
        n_unknown = 0
        for sentence in checked.sentences:
            known = tuple(self._known(word) for word in sentence)
            n_unknown += sum(1 for word in known if word == UNKNOWN_WORD)
            n_words += len(known) + 1
            log_probability += self._log_probability_of_words(known)
        return LanguageModelEvaluation(log_probability, n_words, n_unknown)

    def perplexity(self, texts: Sequence[str]) -> float:
        """The perplexity of held-out ``texts``. See :meth:`evaluate`."""
        return self.evaluate(texts).perplexity

    def generate(
        self,
        max_words: int = 20,
        random_seed: int | None = None,
        temperature: float = 1.0,
    ) -> tuple[str, ...]:
        """Draw a sentence from the model, word by word.

        Starts from the sentence-start context and stops at the end marker or
        after ``max_words`` words, whichever comes first. The words are a
        tuple because they are like items; the end marker is not among them.

        Parameters
        ----------
        max_words:
            The most words to draw. At least one.
        random_seed:
            Seeds the draws, so a seeded call is reproducible.
        temperature:
            Reshapes each draw as
            :meth:`~oop_ml.core.natural_language_processing.n_grams.distribution.NextWordDistribution.sample`
            describes. One is the model's own distribution.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``max_words`` is below one or the temperature is not positive.
        """
        if max_words < 1:
            raise InvalidValuesError(f"max_words must be at least 1, got {max_words}")
        self._check_fitted()

        generator = np.random.default_rng(random_seed)
        context: tuple[str, ...] = (SENTENCE_START,) * (self.order - 1)
        words: list[str] = []
        for _ in range(max_words):
            word = self.next_word_distribution(context).sample(generator, temperature)
            if word == SENTENCE_END:
                break
            words.append(word)
            context = (*context, word)[max(0, len(context) + 1 - (self.order - 1)) :]
            if self.order == 1:
                context = ()
        return tuple(words)

    def _known(self, word: str) -> str:
        """``word`` if the model can predict it, else the unknown word."""
        if not isinstance(word, str) or not word:
            raise InvalidValuesError("a word is a non-empty string")
        return word if word in self._counts.vocabulary else UNKNOWN_WORD

    def _known_words(self, text: str) -> tuple[str, ...]:
        return tuple(
            self._known(word)
            for word in self.pre_tokenizer.split(checked_text(text)).texts
        )

    def _framed_context(self, context: Sequence[str]) -> tuple[str, ...]:
        """The last ``order - 1`` words, unknown words replaced, padded with
        sentence starts on the left."""
        if isinstance(context, str):
            raise InvalidValuesError("a context is a sequence of words, not one string")
        width = self.order - 1
        recent = tuple(
            word if word == SENTENCE_START else self._known(word) for word in context
        )[len(context) - width if len(context) > width else 0 :]
        return (SENTENCE_START,) * (width - len(recent)) + recent

    def _log_probability_of_words(self, words: Sequence[str]) -> float:
        framed = padded(words, self.order)
        total = 0.0
        for position in range(self.order - 1, len(framed)):
            probability = self.smoothing.probability_of(
                self._counts,
                framed[position - self.order + 1 : position],
                framed[position],
            )
            if probability <= 0.0:
                return -math.inf
            total += math.log(probability)
        return total
