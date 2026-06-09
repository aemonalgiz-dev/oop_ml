"""What a segmenter learns from: a word dictionary, or the sentences it is counted from.

The problem this family exists for
----------------------------------
Chinese, Japanese and Thai are written without spaces, so "where are the words"
is not a rule about the writing system at all; it is a question the reader
answers from knowledge of the language. ``研究生命起源`` is either
``研究 / 生命 / 起源`` (research / life / origin) or ``研究生 / 命 / 起源``
(graduate student / fate / origin), and every character sequence in it is a
real word. A segmenter has to carry that knowledge in some form, and the two
forms it comes in are a dictionary and a segmented corpus.

:class:`WordDictionary` is the first. Every entry pairs a word with a
frequency, because a plain word list can say only that a reading is *possible*
and the lattice segmenter needs to know which possible reading is *likely*:
with ``研究`` at 10, ``生命`` at 8, ``研究生`` at 6 and ``命`` at 4, the first
reading scores ``10 * 8 = 80`` against the second's ``6 * 4 = 24``. A
dictionary built with :meth:`WordDictionary.from_words` has every frequency at
one, which is the honest statement that nothing is known beyond the list.

:class:`SegmentedCorpus` is the second, and it is the coercion boundary for
segmented text in the way
:class:`~oop_ml.core.natural_language_processing.tokenization.corpus.Corpus` is for
plain text. A caller hands ``fit`` a sequence of sentences, each a sequence of
words; the corpus checks that it is not one string, that no sentence is one
string (which would iterate as single-character words and fit a model in which
every word is one character long, without raising), that nothing is empty, and
that no word contains whitespace. That last rule is a fact about this family
rather than a preference: every segmenter here treats whitespace as ending a
run before it looks at anything, so a "word" with a space inside could never
be produced and a model taught one would be taught something it cannot say.

Why a repeated word is the same failure as a repeated token
-----------------------------------------------------------
A dictionary holding ``研究`` twice with two frequencies would have to choose
one of them silently. That is exactly the failure
:class:`~oop_ml.core.natural_language_processing.tokenization.vocabulary.Vocabulary`
refuses with :class:`~oop_ml.core.exceptions.NonUniqueTokensError`, and it is
refused here with the same exception so that a reader meeting it is sent to
the same idea. Counting a corpus with :meth:`WordDictionary.from_segmented_corpus`
cannot produce the collision, since a word seen twice is one entry counted
twice.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueTokensError,
)


def checked_word(word: object, role: str) -> str:
    """The one place a word of this family is confirmed to be usable.

    Parameters
    ----------
    word:
        The candidate.
    role:
        What the word is being checked as, for the message: ``"dictionary
        word"`` or ``"word of a segmented sentence"``.

    Raises
    ------
    InvalidValuesError
        If ``word`` is not a string, or contains whitespace.
    EmptyValuesError
        If ``word`` is the empty string.
    """
    if not isinstance(word, str):
        raise InvalidValuesError(
            f"every {role} must be a str, got {type(word).__name__}"
        )
    if not word:
        raise EmptyValuesError(f"a {role} must hold at least one character")
    if any(character.isspace() for character in word):
        raise InvalidValuesError(
            f"a {role} cannot contain whitespace, got {word!r}; whitespace ends "
            f"a run before any segmenter looks at it, so such a word could "
            f"never be matched or produced"
        )
    return word


class DictionaryEntry:
    """One word a segmenter may produce, and how often the language uses it.

    Parameters
    ----------
    word:
        The word. At least one character, no whitespace.
    frequency:
        How often it occurs. At least one, because a word that never occurs
        is not a word of the dictionary; ``from_words`` uses one for every
        entry when nothing more is known.

    Raises
    ------
    InvalidValuesError
        If ``word`` is not a string, contains whitespace, or ``frequency`` is
        below one.
    EmptyValuesError
        If ``word`` is empty.
    """

    __slots__ = ("_frequency", "_word")

    def __init__(self, word: str, frequency: int) -> None:
        self._word = checked_word(word, "dictionary word")

        if frequency < 1:
            raise InvalidValuesError(
                f"a dictionary word occurs at least once, got frequency "
                f"{frequency} for {word!r}"
            )
        self._frequency = int(frequency)

    @property
    def word(self) -> str:
        """The word."""
        return self._word

    @property
    def frequency(self) -> int:
        """How often the language uses it."""
        return self._frequency

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DictionaryEntry):
            return NotImplemented
        return self._word == other._word and self._frequency == other._frequency

    def __hash__(self) -> int:
        return hash((self._word, self._frequency))

    def __repr__(self) -> str:
        return f"DictionaryEntry({self._word!r}, {self._frequency})"


class WordDictionary:
    """Every word a dictionary-driven segmenter may produce, with its frequency.

    Parameters
    ----------
    entries:
        One entry per word. Non-empty, no word twice.

    Raises
    ------
    EmptyValuesError
        If there are no entries.
    NonUniqueTokensError
        If a word appears in two entries.
    """

    __slots__ = ("_entries_by_word", "_longest_word_length", "_total_frequency")

    def __init__(self, entries: Sequence[DictionaryEntry]) -> None:
        if len(entries) == 0:
            raise EmptyValuesError("a word dictionary needs at least one entry")

        entries_by_word: dict[str, DictionaryEntry] = {}
        for entry in entries:
            if entry.word in entries_by_word:
                raise NonUniqueTokensError(
                    f"the word {entry.word!r} appears twice in the dictionary, "
                    f"with frequencies {entries_by_word[entry.word].frequency} "
                    f"and {entry.frequency}"
                )
            entries_by_word[entry.word] = entry

        self._entries_by_word = entries_by_word
        self._total_frequency = sum(entry.frequency for entry in entries)
        self._longest_word_length = max(len(entry.word) for entry in entries)

    @classmethod
    def from_words(cls, words: Sequence[str]) -> WordDictionary:
        """A dictionary in which every word has frequency one.

        For a caller who has a word list and nothing more; the lattice
        segmenter then prefers fewer words, since every word costs the same.
        """
        return cls([DictionaryEntry(word, 1) for word in words])

    @classmethod
    def from_segmented_corpus(
        cls, sentences: SegmentedCorpus | Sequence[Sequence[str]]
    ) -> WordDictionary:
        """A dictionary counting every word of an already segmented corpus.

        Raises
        ------
        InvalidValuesError, EmptyValuesError
            As :class:`SegmentedCorpus` raises them.
        """
        counter: Counter[str] = Counter()
        for sentence in SegmentedCorpus.of(sentences):
            counter.update(sentence)
        return cls(
            [DictionaryEntry(word, frequency) for word, frequency in counter.items()]
        )

    @property
    def n_words(self) -> int:
        """How many distinct words the dictionary holds."""
        return len(self._entries_by_word)

    @property
    def total_frequency(self) -> int:
        """The sum of every frequency, which is what a frequency is a share of."""
        return self._total_frequency

    @property
    def longest_word_length(self) -> int:
        """How many characters the longest word has, which bounds every match."""
        return self._longest_word_length

    @property
    def words(self) -> tuple[str, ...]:
        """The words, in the order their entries were given."""
        return tuple(self._entries_by_word)

    def frequency_of(self, word: str) -> int:
        """How often ``word`` occurs, or zero if the dictionary lacks it."""
        entry = self._entries_by_word.get(word)
        return 0 if entry is None else entry.frequency

    def __contains__(self, word: object) -> bool:
        return word in self._entries_by_word

    def __iter__(self) -> Iterator[DictionaryEntry]:
        """Iterate the entries themselves, not the words or bare frequencies."""
        return iter(self._entries_by_word.values())

    def __len__(self) -> int:
        return self.n_words

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WordDictionary):
            return NotImplemented
        return self._entries_by_word == other._entries_by_word

    def __hash__(self) -> int:
        return hash(frozenset(self._entries_by_word.values()))

    def __repr__(self) -> str:
        return (
            f"WordDictionary(n_words={self.n_words}, "
            f"total_frequency={self._total_frequency})"
        )


class SegmentedCorpus:
    """Sentences already cut into words, checked once, for the segmenters that learn.

    Parameters
    ----------
    sentences:
        The sentences, each a sequence of its words in order. At least one
        sentence, every sentence at least one word, every word a non-empty
        string without whitespace.

    Raises
    ------
    InvalidValuesError
        If ``sentences`` is one string, a sentence is one string, a word is
        not a string, or a word contains whitespace.
    EmptyValuesError
        If there are no sentences, a sentence has no words, or a word is
        empty.
    """

    __slots__ = ("_sentences",)

    def __init__(self, sentences: Sequence[Sequence[str]]) -> None:
        if isinstance(sentences, str):
            raise InvalidValuesError(
                "a segmented corpus is a sequence of sentences, each a sequence "
                "of words, not one string"
            )

        if len(sentences) == 0:
            raise EmptyValuesError("a segmented corpus needs at least one sentence")

        checked: list[tuple[str, ...]] = []
        for position, sentence in enumerate(sentences):
            if isinstance(sentence, str):
                raise InvalidValuesError(
                    f"a segmented sentence is a sequence of words, not one string; "
                    f"got {sentence!r} at position {position}"
                )
            if len(sentence) == 0:
                raise EmptyValuesError(
                    f"the sentence at position {position} has no words"
                )
            checked.append(
                tuple(
                    checked_word(word, "word of a segmented sentence")
                    for word in sentence
                )
            )

        self._sentences = tuple(checked)

    @classmethod
    def of(cls, source: SegmentedCorpus | Sequence[Sequence[str]]) -> SegmentedCorpus:
        """``source`` as a segmented corpus, validating once and never twice.

        Idempotent, like
        :meth:`~oop_ml.core.natural_language_processing.tokenization.corpus.Corpus.of`.
        """
        if isinstance(source, SegmentedCorpus):
            return source
        return cls(source)

    @property
    def n_sentences(self) -> int:
        """How many sentences there are."""
        return len(self._sentences)

    @property
    def n_words(self) -> int:
        """How many word occurrences there are across every sentence."""
        return sum(len(sentence) for sentence in self._sentences)

    def __iter__(self) -> Iterator[tuple[str, ...]]:
        """Iterate the sentences, each as its words in order."""
        return iter(self._sentences)

    def __len__(self) -> int:
        return self.n_sentences

    def __getitem__(self, position: int) -> tuple[str, ...]:
        return self._sentences[position]

    def __repr__(self) -> str:
        return (
            f"SegmentedCorpus(n_sentences={self.n_sentences}, n_words={self.n_words})"
        )
