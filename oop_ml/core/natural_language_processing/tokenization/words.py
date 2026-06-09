"""A word as a pre-tokenizer hands it on: the text, and where in the source it was.

Why a word carries its position
-------------------------------
Every splitter in this package answers the same question, "where does one word
stop and the next begin", and answers it with a list of strings. A list of
strings has lost something the splitter knew a moment earlier, which is where
each one came from. Two tokenizers that disagree about a sentence cannot be
compared without it, a highlighted span in an interface cannot be drawn without
it, and a segmenter for a script with no spaces has literally nothing else to
report, since its whole output *is* the set of boundaries it chose.

So a :class:`Word` is the text together with its ``[start, end)`` span in the
source, and :class:`Words` is the ordered, non-overlapping run of them that one
text became. The order and the non-overlap are checked in the constructor,
because a "segmentation" whose pieces overlap or run backwards is not a
segmentation of anything, and an object whose type carries that guarantee cannot
be handed around in a broken state. That is the
:class:`~oop_ml.core.data.feature_set.FeatureSet` pattern applied to text.

Why the text may differ from the slice
--------------------------------------
``end - start`` is not required to equal ``len(text)``. The Penn Treebank
rules turn an opening ``"`` into two backquotes, and a Moses-style splitter
escapes ``&`` as ``&amp;``, so a word can be longer or shorter than the
characters it stands for. The span says where the word came from; the text
says what it became. Requiring the two to agree would forbid every normalising
rule in the word-level family, and normalising is most of what those rules do.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError


class Word:
    """One word, and the span of the source text it stands for.

    Parameters
    ----------
    text:
        What the word became. At least one character.
    start:
        Offset of its first character in the source text.
    end:
        Offset one past its last character in the source text, so that
        ``source[start:end]`` is the slice it was cut from.

    Raises
    ------
    EmptyValuesError
        If ``text`` is empty.
    InvalidValuesError
        If the span is not ``0 <= start < end``.
    """

    __slots__ = ("_end", "_start", "_text")

    def __init__(self, text: str, start: int, end: int) -> None:
        if not isinstance(text, str) or not text:
            raise EmptyValuesError("a word must hold at least one character")

        if start < 0 or end <= start:
            raise InvalidValuesError(
                f"a word's span must satisfy 0 <= start < end, got [{start}, {end})"
            )

        self._text = text
        self._start = int(start)
        self._end = int(end)

    @classmethod
    def of(cls, source: str, start: int, end: int) -> Word:
        """The word that is exactly ``source[start:end]``.

        The ordinary case, for the splitters that cut and never rewrite.
        """
        return cls(source[start:end], start, end)

    @property
    def text(self) -> str:
        """What the word became."""
        return self._text

    @property
    def start(self) -> int:
        """Offset of the first source character."""
        return self._start

    @property
    def end(self) -> int:
        """Offset one past the last source character."""
        return self._end

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Word):
            return NotImplemented
        return (
            self._text == other._text
            and self._start == other._start
            and self._end == other._end
        )

    def __hash__(self) -> int:
        return hash((self._text, self._start, self._end))

    def __repr__(self) -> str:
        return f"Word({self._text!r}, {self._start}, {self._end})"


class Words:
    """The words one text was split into, in source order and never overlapping.

    Parameters
    ----------
    words:
        The words, in the order they appear in the source. May be empty, since
        a text of nothing but spaces genuinely holds no words and a splitter
        should be able to say so.

    Raises
    ------
    InvalidValuesError
        If a word starts before the previous one ended.
    """

    __slots__ = ("_words",)

    def __init__(self, words: Sequence[Word]) -> None:
        previous_end = 0
        for word in words:
            if word.start < previous_end:
                raise InvalidValuesError(
                    f"words must be in source order and must not overlap; "
                    f"{word!r} starts before offset {previous_end}"
                )
            previous_end = word.end

        self._words = tuple(words)

    @property
    def texts(self) -> tuple[str, ...]:
        """The words as strings, in order, for the callers that need only those."""
        return tuple(word.text for word in self._words)

    @property
    def n_words(self) -> int:
        """How many words the text became."""
        return len(self._words)

    def __iter__(self) -> Iterator[Word]:
        """Iterate the words themselves, not their strings."""
        return iter(self._words)

    def __len__(self) -> int:
        return len(self._words)

    def __getitem__(self, position: int) -> Word:
        return self._words[position]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Words):
            return NotImplemented
        return self._words == other._words

    def __hash__(self) -> int:
        return hash(self._words)

    def __repr__(self) -> str:
        return f"Words({list(self.texts)!r})"
