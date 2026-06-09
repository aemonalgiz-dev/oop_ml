"""Maximum matching: the greedy dictionary segmenter, run forward, backward, or both.

The idea
--------
Stand at the start of a run and take the longest dictionary word that begins
there; if none does, take one character; move past what was taken and repeat.
That is forward maximum matching, and it is the oldest segmenter for Chinese
that is still in use, because it is trivially fast and right far more often
than it has any business being. The reason it works is that Chinese words are
mostly two characters, and a longer word is almost always the intended reading
when it is available at all.

Where it fails is the point of the other two directions. Greedy choice cannot
look ahead, so a long word that happens to *begin* at the cursor is taken even
when the characters after it are then left with nothing to join. Running the
same greedy rule from the right end, backward maximum matching, makes the
mirror-image mistakes -- and Chinese being a head-final language, the mirror
image is measurably the rarer one. The bidirectional heuristic runs both and
keeps the one that looks more like a segmentation.

Worked, on the classic case
---------------------------
``我们在野生动物园玩`` ("we play at the wildlife park") with a dictionary of
``我们``, ``在``, ``在野``, ``野生``, ``生动``, ``动物园``, ``物``, ``园``, ``玩``::

    forward    我们 | 在野 | 生动 | 物 | 园 | 玩      6 words, 3 of them single
    backward   我们 | 在 | 野生 | 动物园 | 玩         5 words, 2 of them single

Forward is led astray twice, by ``在野`` ("out of office") and ``生动``
("vivid"), both real words that the sentence does not contain. Backward, moving
from ``玩``, meets ``动物园`` whole and everything falls into place. The
heuristic prefers fewer words and picks backward.

On ``研究生命起源`` with ``研究``, ``研究生``, ``生命``, ``命`` and ``起源``,
both directions produce three words -- ``研究生 | 命 | 起源`` forward and
``研究 | 生命 | 起源`` backward -- and the tie is broken by the second rule,
fewer single-character words: forward has one (``命``), backward none.

The heuristic, stated as the arbitrary rule it is
-------------------------------------------------
:attr:`MatchingDirection.BIDIRECTIONAL` keeps the candidate with

1. fewer words, then
2. fewer single-character words, then
3. the backward one.

None of the three is a theorem. The first says a segmentation into fewer
pieces used longer words and longer words are likelier to be intended; the
second says a stray single character is usually a sign that a word was
mis-cut; the third is the head-final argument above. All three are the
conventional choices and are recorded here so that the two backends of a
future comparison agree on them. ``ab | c`` against ``a | bc`` on a Latin toy
dictionary holding all four ties on both counts and goes backward.

A character the dictionary lacks
--------------------------------
It becomes a word of its own. That is the only thing a dictionary segmenter
*can* do with a character it has never seen, and it is also why the lattice
segmenter in this family takes an optional statistical model for exactly
those runs.

Cost
----
Each position tries at most ``longest_word_length`` substrings, so a run of
``n`` characters costs ``O(n * L)`` lookups. Nothing here is worth optimising.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from oop_ml.core.natural_language_processing.tokenization.segmentation.dictionary import (  # noqa: E501
    WordDictionary,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.runs import (
    RunSegmenter,
)


class MatchingDirection(StrEnum):
    """Which end of the run the greedy match starts from.

    Attributes
    ----------
    FORWARD:
        Left to right: the longest word beginning at the cursor.
    BACKWARD:
        Right to left: the longest word ending at the cursor.
    BIDIRECTIONAL:
        Both, keeping whichever the heuristic in the module docstring prefers.
    """

    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"


class MaximumMatchingSegmenter(RunSegmenter):
    """Greedy longest-match segmentation against a dictionary.

    Parameters
    ----------
    dictionary:
        The words that may be matched. Frequencies are ignored; this segmenter
        knows only whether a word exists.
    direction:
        Which end to match from, or both.
    """

    dictionary: WordDictionary
    direction: MatchingDirection = Field(default=MatchingDirection.FORWARD)

    def _words_of_run(self, run: str) -> tuple[str, ...]:
        if self.direction is MatchingDirection.FORWARD:
            return self._forward(run)
        if self.direction is MatchingDirection.BACKWARD:
            return self._backward(run)
        return self._preferred(self._forward(run), self._backward(run))

    def _forward(self, run: str) -> tuple[str, ...]:
        """The longest dictionary word starting at each position, else one character."""
        pieces: list[str] = []
        position = 0
        while position < len(run):
            longest = min(self.dictionary.longest_word_length, len(run) - position)
            taken = run[position]
            for length in range(longest, 1, -1):
                candidate = run[position : position + length]
                if candidate in self.dictionary:
                    taken = candidate
                    break
            pieces.append(taken)
            position += len(taken)
        return tuple(pieces)

    def _backward(self, run: str) -> tuple[str, ...]:
        """The longest dictionary word ending at each position, else one character."""
        pieces: list[str] = []
        end = len(run)
        while end > 0:
            longest = min(self.dictionary.longest_word_length, end)
            taken = run[end - 1]
            for length in range(longest, 1, -1):
                candidate = run[end - length : end]
                if candidate in self.dictionary:
                    taken = candidate
                    break
            pieces.append(taken)
            end -= len(taken)
        pieces.reverse()
        return tuple(pieces)

    @staticmethod
    def _preferred(
        forward: tuple[str, ...], backward: tuple[str, ...]
    ) -> tuple[str, ...]:
        """The heuristic: fewer words, then fewer single characters, then backward."""
        if len(forward) != len(backward):
            return forward if len(forward) < len(backward) else backward

        forward_singles = sum(1 for piece in forward if len(piece) == 1)
        backward_singles = sum(1 for piece in backward if len(piece) == 1)
        if forward_singles < backward_singles:
            return forward
        return backward
