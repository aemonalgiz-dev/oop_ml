"""Every n-gram of every order up to ``n``, counted once, with the marginals
every smoothing method reads.

What is counted
---------------
Each sentence is framed with ``order - 1`` sentence starts and one sentence
end, and then every window of one to ``order`` words is counted -- except
windows ending in a sentence start, which is never predicted and so is never
an event. From ``the cat sat`` under a bigram model the windows are the
unigrams ``the cat sat </s>`` and the bigrams ``<s> the``, ``the cat``,
``cat sat``, ``sat </s>``. A unigram model frames with only the end marker.

What is derived, and why here
-----------------------------
Every smoothing method in this package is a rule for turning these counts into
a probability, and between them they ask five questions of the table. How
often did this n-gram occur (``count_of``). How often was this context
followed by anything (``context_total``). How many *distinct* words followed it
(``n_followers``, Witten-Bell's ``T(h)``). How many distinct words preceded
this n-gram (``n_preceding_types``, Kneser-Ney's continuation count, which
asks how many contexts a word completes rather than how often it occurs).
And how many n-gram types were seen exactly ``r`` times
(``frequency_of_frequencies``, which Good-Turing and the modified Kneser-Ney
discounts are estimated from). Each is answered once, here, from the one pass
over the corpus, so no smoothing method carries its own copy of the counting
and two methods cannot disagree about what a count is.

What is implemented today, and what these are for
-------------------------------------------------
Two rules exist: maximum likelihood and additive smoothing. Both read only
``count_of`` and ``context_total``. So ``n_followers``, ``n_preceding_types``,
``n_continuation_types`` and ``frequency_of_frequencies`` currently have **no
caller anywhere in the library**. They are derived here rather than left to
the rule that will want them, because they are properties of the table and a
rule that computed its own would be a second opinion about what a count is.
Recorded so that a reader meeting an unused method knows it is a foundation
rather than dead weight.

Why the context total is a sum and not a count
----------------------------------------------
``context_total(h)`` is the number of times ``h`` was followed by *something*,
``sum_w C(h w)``, and not the count of ``h`` as an n-gram of its own. Inside a
sentence the two agree: in ``the cat sat / the cat ran`` the unigram ``cat``
occurs twice and is followed by something twice. They part at the edges.
``</s>`` occurs twice as a unigram and is followed by nothing, so its context
total is zero; ``<s>`` is never counted as a unigram at all and is followed by
something twice. A probability is a count over the number of trials, and the
trials are the times the context stood before a next word.

The prediction vocabulary
-------------------------
Every word that can be predicted, which is every ordinary word seen plus the
end marker and the unknown word, in that order: end marker, unknown word, then
the ordinary words commonest first. The unknown word is in the vocabulary
whether or not the corpus produced it, so a held-out text can always be
scored; it is the unknown token of the
:class:`~oop_ml.core.natural_language_processing.tokenization.vocabulary.Vocabulary`,
so a lookup of a word never seen answers its id.

Dense tables would be ``V ** n`` cells and are not built; the tables are
dictionaries keyed by the n-grams that occurred, which is the only honest size
for a model whose whole point is that most n-grams never occur.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.n_grams.grams import (
    FRAMING_MARKERS,
    RESERVED_WORDS,
    SENTENCE_END,
    SENTENCE_START,
    UNKNOWN_WORD,
    NGram,
    checked_order,
    padded,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary


class CountFrequency:
    """How many n-gram types of one order were seen exactly ``count`` times.

    Parameters
    ----------
    count:
        A number of occurrences, at least one.
    n_types:
        How many distinct n-grams occurred exactly that often, at least one.

    Raises
    ------
    InvalidValuesError
        If either figure is below one.
    """

    __slots__ = ("_count", "_n_types")

    def __init__(self, count: int, n_types: int) -> None:
        if count < 1 or n_types < 1:
            raise InvalidValuesError(
                f"a frequency of frequencies pairs a count of at least one with "
                f"at least one type, got count {count} and {n_types} types"
            )
        self._count = int(count)
        self._n_types = int(n_types)

    @property
    def count(self) -> int:
        """The number of occurrences."""
        return self._count

    @property
    def n_types(self) -> int:
        """How many distinct n-grams occurred exactly that often."""
        return self._n_types

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CountFrequency):
            return NotImplemented
        return self._count == other._count and self._n_types == other._n_types

    def __hash__(self) -> int:
        return hash((self._count, self._n_types))

    def __repr__(self) -> str:
        return f"CountFrequency(count={self._count}, n_types={self._n_types})"


class FrequencyOfFrequencies:
    """For one order, how many n-gram types were seen ``r`` times, for each ``r``.

    Good-Turing's table: ``N_1`` is the number of n-grams seen once, ``N_2``
    twice, and so on. The share of the next text that will be n-grams never
    seen at all is estimated as ``N_1 / N``, which is the table's first entry
    over its total.

    Parameters
    ----------
    order:
        Which order the table is over.
    n_types_by_count:
        ``count -> n_types``, every entry positive.

    Raises
    ------
    InvalidValuesError
        If the order is below one, or an entry is not positive.
    """

    __slots__ = ("_n_types_by_count", "_order")

    def __init__(self, order: int, n_types_by_count: Mapping[int, int]) -> None:
        self._order = checked_order(order)
        checked: dict[int, int] = {}
        for count, n_types in sorted(n_types_by_count.items()):
            entry = CountFrequency(count, n_types)
            checked[entry.count] = entry.n_types
        self._n_types_by_count = checked

    @property
    def order(self) -> int:
        """Which order the table is over."""
        return self._order

    @property
    def counts(self) -> tuple[int, ...]:
        """Every count that some type was seen exactly that often, ascending."""
        return tuple(self._n_types_by_count)

    @property
    def n_types(self) -> int:
        """How many distinct n-grams of this order were seen at all."""
        return sum(self._n_types_by_count.values())

    @property
    def n_tokens(self) -> int:
        """How many occurrences there were in all: ``sum r N_r``."""
        return sum(count * n_types for count, n_types in self._n_types_by_count.items())

    def n_types_with_count(self, count: int) -> int:
        """``N_r``: how many types were seen exactly ``count`` times, or zero."""
        return self._n_types_by_count.get(count, 0)

    def __iter__(self) -> Iterator[CountFrequency]:
        return (
            CountFrequency(count, n_types)
            for count, n_types in self._n_types_by_count.items()
        )

    def __len__(self) -> int:
        return len(self._n_types_by_count)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrequencyOfFrequencies):
            return NotImplemented
        return (
            self._order == other._order
            and self._n_types_by_count == other._n_types_by_count
        )

    def __hash__(self) -> int:
        return hash((self._order, tuple(self._n_types_by_count.items())))

    def __repr__(self) -> str:
        return (
            f"FrequencyOfFrequencies(order={self._order}, "
            f"n_types={self.n_types}, n_tokens={self.n_tokens})"
        )


class NGramCounts:
    """Every n-gram of orders one to ``order``, counted over framed sentences.

    Built through :meth:`from_sentences`; the constructor is the validated
    state it produces. Every lookup takes a sequence of words and answers a
    number, and an n-gram that never occurred answers zero rather than raising,
    because "never seen" is the ordinary case and the whole subject of
    smoothing.

    Raises
    ------
    InvalidValuesError
        If a lookup's order is outside what was counted.
    """

    __slots__ = (
        "_context_totals",
        "_continuation_totals",
        "_counts",
        "_followers",
        "_frequency_tables",
        "_n_sentences",
        "_order",
        "_preceding",
        "_totals",
        "_vocabulary",
    )

    def __init__(
        self,
        order: int,
        counts: Sequence[Mapping[tuple[str, ...], int]],
        n_sentences: int,
    ) -> None:
        self._order = checked_order(order)
        if len(counts) != self._order:
            raise InvalidValuesError(
                f"an order-{self._order} table needs one count table per order, "
                f"got {len(counts)}"
            )
        if n_sentences < 1:
            raise EmptyValuesError("n-gram counts need at least one sentence")

        self._counts: tuple[dict[tuple[str, ...], int], ...] = tuple(
            dict(table) for table in counts
        )
        self._n_sentences = int(n_sentences)

        context_totals: dict[tuple[str, ...], int] = defaultdict(int)
        followers: dict[tuple[str, ...], set[str]] = defaultdict(set)
        preceding: dict[tuple[str, ...], set[str]] = defaultdict(set)
        continuation_totals: dict[tuple[str, ...], int] = defaultdict(int)
        for table in self._counts:
            for gram, count in table.items():
                if count < 1:
                    raise InvalidValuesError(
                        f"a counted n-gram occurred at least once, got {count} "
                        f"for {gram!r}"
                    )
                context_totals[gram[:-1]] += count
                followers[gram[:-1]].add(gram[-1])
                if len(gram) >= 2:
                    preceding[gram[1:]].add(gram[0])
                    continuation_totals[gram[1:-1]] += 1

        self._context_totals = dict(context_totals)
        self._followers = {context: len(words) for context, words in followers.items()}
        self._preceding = {gram: len(words) for gram, words in preceding.items()}
        self._continuation_totals = dict(continuation_totals)
        self._totals = tuple(sum(table.values()) for table in self._counts)
        self._frequency_tables = tuple(
            FrequencyOfFrequencies(position + 1, Counter(table.values()))
            for position, table in enumerate(self._counts)
        )

        unigram_counts = self._counts[0]
        ordinary = sorted(
            (word for (word,) in unigram_counts if word not in RESERVED_WORDS),
            key=lambda word: (-unigram_counts[(word,)], word),
        )
        self._vocabulary = Vocabulary(
            [SENTENCE_END, UNKNOWN_WORD, *ordinary], unknown_token=UNKNOWN_WORD
        )

    @classmethod
    def from_sentences(
        cls, sentences: Sequence[Sequence[str]], order: int
    ) -> NGramCounts:
        """Frame every sentence and count every window of one to ``order`` words.

        Parameters
        ----------
        sentences:
            Each a sequence of words, already reduced to the model's
            vocabulary (rare words replaced by the unknown word). A sentence
            may be empty; it still contributes an end marker.
        order:
            The highest order to count.

        Raises
        ------
        InvalidValuesError
            If the order is below one, ``sentences`` is a single string, a
            word is not a non-empty string, or a word is one of the framing
            markers ``<s>`` or ``</s>``, which the framing adds itself. The
            unknown word is *accepted* here, because sentences arrive already
            reduced to a vocabulary and refusing it would make that reduction
            impossible to express.
        EmptyValuesError
            If there are no sentences.
        """
        checked = checked_order(order)
        if isinstance(sentences, str):
            raise InvalidValuesError(
                "sentences are a sequence of word sequences, not one string"
            )
        if len(sentences) == 0:
            raise EmptyValuesError("n-gram counts need at least one sentence")

        tables: list[dict[tuple[str, ...], int]] = [{} for _ in range(checked)]
        for position, sentence in enumerate(sentences):
            if isinstance(sentence, str):
                raise InvalidValuesError(
                    f"sentence {position} is a single string; a sentence is a "
                    f"sequence of words"
                )
            for word in sentence:
                if not isinstance(word, str) or not word:
                    raise InvalidValuesError(
                        f"every word is a non-empty string, got {word!r} in "
                        f"sentence {position}"
                    )
                if word in FRAMING_MARKERS:
                    raise InvalidValuesError(
                        f"{word!r} is a reserved marker and cannot appear in a "
                        f"sentence, because the framing adds it"
                    )

            framed = padded(sentence, checked)
            for size in range(1, checked + 1):
                table = tables[size - 1]
                for start in range(len(framed) - size + 1):
                    gram = framed[start : start + size]
                    if gram[-1] == SENTENCE_START:
                        continue
                    table[gram] = table.get(gram, 0) + 1

        return cls(checked, tables, len(sentences))

    @property
    def order(self) -> int:
        """The highest order counted."""
        return self._order

    @property
    def vocabulary(self) -> Vocabulary:
        """Every word that can be predicted: the end marker, the unknown word,
        then the ordinary words commonest first."""
        return self._vocabulary

    @property
    def n_sentences(self) -> int:
        """How many sentences were counted."""
        return self._n_sentences

    def count_of(self, gram: Sequence[str]) -> int:
        """How often this n-gram occurred; zero if never.

        Raises
        ------
        InvalidValuesError
            If the n-gram's order is outside one to ``order``.
        """
        words = tuple(gram)
        return self._table_for(len(words)).get(words, 0)

    def context_total(self, context: Sequence[str]) -> int:
        """How often ``context`` was followed by anything: ``sum_w C(context w)``.

        The empty context's total is the number of unigram events.

        Raises
        ------
        InvalidValuesError
            If the context is as long as the order or longer.
        """
        words = self._checked_context(context)
        return self._context_totals.get(words, 0)

    def n_followers(self, context: Sequence[str]) -> int:
        """How many distinct words ever followed ``context``.

        Raises
        ------
        InvalidValuesError
            If the context is as long as the order or longer.
        """
        words = self._checked_context(context)
        return self._followers.get(words, 0)

    def n_preceding_types(self, gram: Sequence[str]) -> int:
        """How many distinct words ever stood before this n-gram.

        Kneser-Ney's continuation count ``N1+(. gram)``: how many contexts the
        n-gram completes, rather than how often it occurs. Defined for orders
        one to ``order - 1``, since a preceding word makes a longer n-gram.

        Raises
        ------
        InvalidValuesError
            If the n-gram's order is not between one and ``order - 1``.
        """
        words = tuple(gram)
        if not 1 <= len(words) <= self._order - 1:
            raise InvalidValuesError(
                f"a preceding-type count is for orders 1 to {self._order - 1}, "
                f"got an n-gram of {len(words)} words"
            )
        return self._preceding.get(words, 0)

    def n_continuation_types(self, context: Sequence[str]) -> int:
        """``sum_w N1+(. context w)``: how many distinct n-grams have this
        context in the middle.

        The denominator of Kneser-Ney's lower-order estimate. Defined for
        contexts of zero to ``order - 2`` words.

        Raises
        ------
        InvalidValuesError
            If the context is longer than ``order - 2`` words.
        """
        words = tuple(context)
        if len(words) > self._order - 2:
            raise InvalidValuesError(
                f"a continuation total is for contexts of at most "
                f"{self._order - 2} words, got {len(words)}"
            )
        return self._continuation_totals.get(words, 0)

    def n_types(self, order: int) -> int:
        """How many distinct n-grams of ``order`` were seen."""
        return len(self._table_for(order))

    def total(self, order: int) -> int:
        """How many n-gram occurrences of ``order`` there were in all."""
        self._table_for(order)
        return self._totals[order - 1]

    def grams(self, order: int) -> tuple[NGram, ...]:
        """Every distinct n-gram of ``order``, in a fixed order."""
        return tuple(NGram(words) for words in sorted(self._table_for(order)))

    def frequency_of_frequencies(self, order: int) -> FrequencyOfFrequencies:
        """How many types of ``order`` were seen ``r`` times, for each ``r``."""
        self._table_for(order)
        return self._frequency_tables[order - 1]

    def _table_for(self, order: int) -> dict[tuple[str, ...], int]:
        if not 1 <= order <= self._order:
            raise InvalidValuesError(
                f"only orders 1 to {self._order} were counted, got {order}"
            )
        return self._counts[order - 1]

    def _checked_context(self, context: Sequence[str]) -> tuple[str, ...]:
        words = tuple(context)
        if len(words) > self._order - 1:
            raise InvalidValuesError(
                f"a context has at most {self._order - 1} words under an "
                f"order-{self._order} table, got {len(words)}"
            )
        return words

    def __repr__(self) -> str:
        return (
            f"NGramCounts(order={self._order}, "
            f"n_types={[self.n_types(size) for size in range(1, self._order + 1)]})"
        )
