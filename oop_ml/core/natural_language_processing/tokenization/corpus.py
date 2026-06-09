"""The texts a tokenizer learns from, and the word counts every trainer starts at.

The coercion boundary for text
------------------------------
:class:`~oop_ml.core.data.column.Column` is where a loose numeric input becomes
the one shape the library computes on, and every model downstream trusts it.
:class:`Corpus` is the same boundary for text. A caller hands ``fit`` a
sequence of strings; the corpus checks that it *is* a sequence of strings and
not one string -- which is the mistake that iterates a sentence as forty
single-character "texts" and fits a vocabulary of letters without raising --
that there is at least one text, and that at least one of them holds something
other than whitespace. Every trainer then works on a ``Corpus`` and re-validates
nothing.

Why the counts are an object
----------------------------
Nearly every subword trainer begins the same way: split the corpus into words
and count them, because a merge or a piece is worth exactly the total count of
the words it appears in. That dictionary of word to count is what byte pair
encoding merges over, what WordPiece scores over and what the unigram model
seeds its candidates from. It is asked for by one method, :meth:`Corpus.word_counts`,
and it comes back as :class:`WordCounts`, which knows its own total and is
iterated as :class:`WordCount` objects, so no trainer ever holds a bare
``dict[str, int]`` and none of them can disagree about how a word was counted.

The words are counted as the pre-tokenizer spells them, and nothing here
lower-cases or strips punctuation. Normalisation is a rule about a language and
belongs to the pre-tokenizer that knows the language; a corpus that quietly
folded case would make ``The`` and ``the`` one word for every tokenizer at once,
whether or not that was wanted.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer


class WordCount:
    """One distinct word, and how many times the corpus used it.

    Parameters
    ----------
    word:
        The word, as the pre-tokenizer spelled it.
    count:
        How many times it appeared. At least one, since a word that never
        appeared is not a word of the corpus.
    """

    __slots__ = ("_count", "_word")

    def __init__(self, word: str, count: int) -> None:
        if not isinstance(word, str) or not word:
            raise EmptyValuesError("a counted word must hold at least one character")

        if count < 1:
            raise InvalidValuesError(
                f"a word of the corpus appeared at least once, got count {count}"
            )

        self._word = word
        self._count = int(count)

    @property
    def word(self) -> str:
        """The word."""
        return self._word

    @property
    def count(self) -> int:
        """How many times the corpus used it."""
        return self._count

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WordCount):
            return NotImplemented
        return self._word == other._word and self._count == other._count

    def __hash__(self) -> int:
        return hash((self._word, self._count))

    def __repr__(self) -> str:
        return f"WordCount({self._word!r}, {self._count})"


class WordCounts:
    """Every distinct word of a corpus with its count, addressable by word.

    Parameters
    ----------
    counts:
        One :class:`WordCount` per distinct word. Non-empty, no word twice.

    Raises
    ------
    EmptyValuesError
        If there are no counts.
    InvalidValuesError
        If a word is counted twice.
    """

    __slots__ = ("_counts_by_word", "_total")

    def __init__(self, counts: Sequence[WordCount]) -> None:
        if len(counts) == 0:
            raise EmptyValuesError("word counts need at least one word")

        counts_by_word: dict[str, WordCount] = {}
        for word_count in counts:
            if word_count.word in counts_by_word:
                raise InvalidValuesError(
                    f"the word {word_count.word!r} is counted twice"
                )
            counts_by_word[word_count.word] = word_count

        self._counts_by_word = counts_by_word
        self._total = sum(word_count.count for word_count in counts)

    @property
    def n_words(self) -> int:
        """How many distinct words there are."""
        return len(self._counts_by_word)

    @property
    def total(self) -> int:
        """How many word occurrences there were, distinct or not."""
        return self._total

    @property
    def words(self) -> tuple[str, ...]:
        """The distinct words, in first-seen order."""
        return tuple(self._counts_by_word)

    def count_of(self, word: str) -> int:
        """How many times ``word`` appeared, or zero if it never did."""
        word_count = self._counts_by_word.get(word)
        return 0 if word_count is None else word_count.count

    def __getitem__(self, word: str) -> int:
        """The count for ``word``, so that ``counts["the"]`` reads well."""
        return self.count_of(word)

    def __contains__(self, word: object) -> bool:
        return word in self._counts_by_word

    def __iter__(self) -> Iterator[WordCount]:
        """Iterate the counts themselves, not the words or bare numbers."""
        return iter(self._counts_by_word.values())

    def __len__(self) -> int:
        return self.n_words

    def __repr__(self) -> str:
        return f"WordCounts(n_words={self.n_words}, total={self._total})"


class Corpus:
    """The texts a tokenizer is fit on, checked once.

    Parameters
    ----------
    texts:
        The texts, one string each. At least one, and at least one of them
        not blank.

    Raises
    ------
    InvalidValuesError
        If ``texts`` is a single string rather than a sequence of them, or an
        entry is not a string.
    EmptyValuesError
        If there are no texts, or every text is blank.
    """

    __slots__ = ("_texts",)

    def __init__(self, texts: Sequence[str]) -> None:
        if isinstance(texts, str):
            raise InvalidValuesError(
                "a corpus is a sequence of texts, not one string; wrap a single "
                "text in a list"
            )

        if len(texts) == 0:
            raise EmptyValuesError("a corpus needs at least one text")

        for position, text in enumerate(texts):
            if not isinstance(text, str):
                raise InvalidValuesError(
                    f"every text must be a str, got {type(text).__name__} "
                    f"at position {position}"
                )

        if all(not text.strip() for text in texts):
            raise EmptyValuesError("every text in the corpus is blank")

        self._texts = tuple(texts)

    @classmethod
    def of(cls, source: Corpus | Sequence[str]) -> Corpus:
        """``source`` as a corpus, validating once and never twice.

        Idempotent, like :meth:`~oop_ml.core.data.column.Column.of`, so a
        corpus passed down through a trainer's helpers costs nothing to
        re-wrap.
        """
        if isinstance(source, Corpus):
            return source
        return cls(source)

    @property
    def n_texts(self) -> int:
        """How many texts there are."""
        return len(self._texts)

    def word_counts(self, pre_tokenizer: PreTokenizer) -> WordCounts:
        """Every distinct word across the texts, as ``pre_tokenizer`` splits them.

        Raises
        ------
        EmptyValuesError
            If the pre-tokenizer finds no words in any text. A pattern that
            matches nothing in the corpus is a configuration error, and a
            vocabulary learned from nothing would be one token wide and wrong.
        """
        counter: Counter[str] = Counter()
        for text in self._texts:
            counter.update(pre_tokenizer.split(text).texts)

        if not counter:
            raise EmptyValuesError(
                f"{type(pre_tokenizer).__name__} found no words in any of the "
                f"{self.n_texts} texts"
            )

        return WordCounts([WordCount(word, count) for word, count in counter.items()])

    def __iter__(self) -> Iterator[str]:
        return iter(self._texts)

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, position: int) -> str:
        return self._texts[position]

    def __repr__(self) -> str:
        return f"Corpus(n_texts={self.n_texts})"
