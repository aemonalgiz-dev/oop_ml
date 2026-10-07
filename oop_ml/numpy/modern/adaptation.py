"""Low-rank parameter updates and the retrieval stage of a grounded prompt."""

import re
from dataclasses import dataclass

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.types import FloatArray
from oop_ml.numpy.modern._validation import count, finite, positive


class LowRankAdapter:
    """Apply a low-rank update to a frozen row-vector projection.

    W has shape (input, output), A is (input, rank), B is (rank, output).
    The caller supplies factors; this object demonstrates the forward operation
    and merging. It does not claim those factors have been fitted to a task.
    """

    def __init__(
        self, base: FloatArray, down: FloatArray, up: FloatArray, alpha: float = 1
    ) -> None:
        self.base = finite(base, 2, "base")
        self.down = finite(down, 2, "down")
        self.up = finite(up, 2, "up")
        self.alpha = positive(alpha, "alpha")
        if (
            self.down.shape[0] != self.base.shape[0]
            or self.up.shape[1] != self.base.shape[1]
            or self.down.shape[1] != self.up.shape[0]
        ):
            raise ShapeMismatchError("adapter factors must join the base projection")

    @property
    def delta(self) -> FloatArray:
        return self.alpha / self.down.shape[1] * (self.down @ self.up)

    @property
    def trainable_parameters(self) -> int:
        return self.down.size + self.up.size

    def respond_to(self, inputs: FloatArray) -> FloatArray:
        inputs = finite(inputs, 2, "inputs")
        if inputs.shape[1] != self.base.shape[0]:
            raise ShapeMismatchError("input width must match the projection")
        return inputs @ (self.base + self.delta)


@dataclass(frozen=True)
class RetrievedPassage:
    index: int
    text: str
    score: float


class TermOverlapRetriever:
    """Exact cosine search over case-folded word counts, without learned meaning.

    An intentionally small retrieval baseline. It returns only positive matches,
    so a query containing no indexed words does not invent a relevant passage.
    Source indices survive ranking for citations. Generation is a separate step.
    """

    def __init__(self, passages: list[str]) -> None:
        if not passages or any(
            not isinstance(p, str) or not p.strip() for p in passages
        ):
            raise InvalidValuesError("supply nonempty text passages")
        self.passages = tuple(passages)
        self.vocabulary = sorted({word for p in passages for word in self._words(p)})
        if not self.vocabulary:
            raise InvalidValuesError("passages must contain words")
        self.vectors = np.array([self._vector(p) for p in passages])

    @staticmethod
    def _words(text: str) -> list[str]:
        return re.findall(r"\w+", text.casefold())

    def _vector(self, text: str) -> FloatArray:
        words = self._words(text)
        return np.array([words.count(word) for word in self.vocabulary], dtype=float)

    def retrieve(self, query: str, limit: int = 2) -> list[RetrievedPassage]:
        if not isinstance(query, str) or not query.strip():
            raise InvalidValuesError("query must contain text")
        count(limit, "limit")
        vector = self._vector(query)
        magnitude = np.linalg.norm(vector)
        if magnitude == 0:
            return []
        norms = np.linalg.norm(self.vectors, axis=1)
        scores = np.divide(
            self.vectors @ vector,
            norms * magnitude,
            out=np.zeros(len(norms)),
            where=norms > 0,
        )
        order = np.argsort(-scores, kind="stable")
        return [
            RetrievedPassage(int(i), self.passages[i], float(scores[i]))
            for i in order[:limit]
            if scores[i] > 0
        ]
