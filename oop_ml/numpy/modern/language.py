"""A causal decoder forward pass, token selection, and the training objective.

The decoder composes the SDK's attention, positions, and row normalization. It
uses one pre-normalized block and fixed seeded parameters. It has no fit method:
its useful answer is a trace of the operations, not fluent text. The loss object
supplies the exact logit derivative independently of that forward-only model.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.network.attention import MultiHeadAttention
from oop_ml.core.network.positions import PositionalEncoding
from oop_ml.core.network.row_normalisation import LayerNormalization
from oop_ml.core.types import FloatArray
from oop_ml.numpy.modern._validation import count, finite, positive


@dataclass(frozen=True)
class TokenDistribution:
    """Turn one vocabulary row into a temperature-scaled nucleus distribution.

    The smallest probability-sorted prefix reaching top_p is retained, including
    the token crossing the threshold. Ties retain vocabulary order. Greedy
    selection is a separate operation, not a division by zero temperature.
    """

    probabilities: FloatArray
    retained: NDArray[np.bool_]

    @classmethod
    def from_logits(
        cls, logits: FloatArray, temperature: float = 1.0, top_p: float = 1.0
    ) -> "TokenDistribution":
        values = finite(logits, 1, "logits")
        temperature = positive(temperature, "temperature")
        if not np.isfinite(top_p) or not 0 < top_p <= 1:
            raise InvalidValuesError("top_p must be in (0, 1]")
        # Subtract first, so large equal logits never overflow on scaling.
        with np.errstate(over="ignore", under="ignore"):
            weights = np.exp((values - values.max()) / temperature)
        probabilities = weights / weights.sum()
        order = np.argsort(-probabilities, kind="stable")
        size = min(
            len(order),
            int(np.searchsorted(np.cumsum(probabilities[order]), top_p, side="left"))
            + 1,
        )
        retained = np.zeros(len(values), dtype=bool)
        retained[order[:size]] = True
        probabilities[~retained] = 0
        probabilities /= probabilities.sum()
        return cls(probabilities, retained)

    def sample(self, generator: np.random.Generator) -> int:
        """The caller owns the random stream, so a sequence is reproducible."""
        return int(generator.choice(len(self.probabilities), p=self.probabilities))

    def greedy(self) -> int:
        """Choose the most probable token, breaking ties by vocabulary order."""
        return int(self.probabilities.argmax())


@dataclass(frozen=True)
class TokenLoss:
    value: float
    per_token: FloatArray
    probabilities: FloatArray
    gradient: FloatArray
    included: NDArray[np.bool_]

    @property
    def perplexity(self) -> float:
        """Exponentiate mean negative log probability, using natural logs."""
        return float(np.exp(self.value))


class TokenCrossEntropy:
    """Average over scored positions only; masked rows have zero derivative.

    Used for next-token learning, response-only supervised fine-tuning, and
    held-out scoring. A loss mask does not block attention to the prompt.
    """

    def measure(
        self,
        logits: FloatArray,
        targets: NDArray[np.int_],
        included: NDArray[np.bool_] | None = None,
    ) -> TokenLoss:
        scores = finite(logits, 2, "logits")
        ids = np.asarray(targets)
        if ids.shape != (len(scores),):
            raise ShapeMismatchError("supply one target id per row")
        if (
            ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= scores.shape[1])
        ):
            raise InvalidValuesError("target ids must index the vocabulary")
        mask = (
            np.ones(len(scores), dtype=bool)
            if included is None
            else np.asarray(included)
        )
        if mask.shape != ids.shape or mask.dtype.kind != "b":
            raise ShapeMismatchError("included must be one boolean per row")
        if not mask.any():
            raise InvalidValuesError("at least one position must be scored")
        shifted = scores - scores.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        probabilities = np.exp(log_probs)
        per_token = -log_probs[np.arange(len(ids)), ids]
        gradient = probabilities.copy()
        gradient[np.arange(len(ids)), ids] -= 1
        gradient *= mask[:, None] / mask.sum()
        return TokenLoss(
            float(per_token[mask].mean()),
            per_token,
            probabilities,
            gradient,
            mask.copy(),
        )


@dataclass(frozen=True)
class DecoderTrace:
    embedded: FloatArray
    positioned: FloatArray
    attention: FloatArray
    after_attention: FloatArray
    feed_forward: FloatArray
    hidden: FloatArray
    logits: FloatArray


@dataclass(frozen=True)
class GeneratedToken:
    context: tuple[int, ...]
    distribution: FloatArray
    token_id: int


class CausalLanguageModel:
    """One inference-only decoder block, with a trace at each connection.

    Token rows share a feed-forward transformation, while causal self-attention
    exchanges information across rows. Parameters are initialized once and reused
    at every sequence length. Recomputing the whole prefix keeps this reference
    implementation clear; it does not implement a key/value cache.
    """

    def __init__(
        self,
        n_vocab: int,
        n_model: int = 4,
        n_heads: int = 2,
        max_context: int = 64,
        seed: int = 7,
    ) -> None:
        self.n_vocab = count(n_vocab, "n_vocab", 2)
        self.n_model = count(n_model, "n_model", 2)
        self.n_heads = count(n_heads, "n_heads")
        self.max_context = count(max_context, "max_context")
        if self.n_model % self.n_heads:
            raise ShapeMismatchError("head count must divide model width")
        rng = np.random.default_rng(count(seed, "seed", 0))
        width = self.n_model
        self.embeddings = rng.normal(0, 0.5, (self.n_vocab, width))
        self.projections = rng.normal(0, 1 / np.sqrt(width), (4 * width, width))
        self.expand = rng.normal(0, 1 / np.sqrt(width), (width, 2 * width))
        self.contract = rng.normal(0, 1 / np.sqrt(2 * width), (2 * width, width))
        self.output = rng.normal(0, 1 / np.sqrt(width), (width, self.n_vocab))

    def respond_to(self, token_ids: NDArray[np.int_]) -> DecoderTrace:
        """Each position predicts the token after it, with no future access."""
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or not 0 < ids.size <= self.max_context:
            raise ShapeMismatchError("supply a nonempty prefix within max_context")
        if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= self.n_vocab):
            raise InvalidValuesError("token ids must index the vocabulary")
        embedded = self.embeddings[ids]
        positioned = (
            PositionalEncoding((len(ids), self.n_model))
            .respond_to(embedded[None, :, :])
            .outputs[0]
        )
        norm = LayerNormalization(self.n_model)
        layer = MultiHeadAttention(
            reads=(len(ids), self.n_model),
            n_heads=self.n_heads,
            looks_ahead=False,
        ).with_parameters(self.projections, np.zeros(4 * self.n_model))
        attention = layer.respond_to(norm.respond_to(positioned).outputs[None])
        after_attention = positioned + attention.outputs[0]
        expanded = norm.respond_to(after_attention).outputs @ self.expand
        feed_forward = np.maximum(expanded, 0) @ self.contract
        hidden = after_attention + feed_forward
        logits = norm.respond_to(hidden).outputs @ self.output
        return DecoderTrace(
            embedded,
            positioned,
            attention.attention[0],
            after_attention,
            feed_forward,
            hidden,
            logits,
        )

    def generate(
        self,
        prefix: NDArray[np.int_],
        max_new_tokens: int = 8,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 0,
        greedy: bool = False,
        end_id: int | None = None,
    ) -> list[GeneratedToken]:
        """Predict, select, append, and repeat until a bound or end token."""
        self.respond_to(prefix)
        count(max_new_tokens, "max_new_tokens", 0)
        positive(temperature, "temperature")
        if not np.isfinite(top_p) or not 0 < top_p <= 1:
            raise InvalidValuesError("top_p must be in (0, 1]")
        if end_id is not None:
            count(end_id, "end_id", 0)
            if end_id >= self.n_vocab:
                raise InvalidValuesError("end_id must index the vocabulary")
        rng = np.random.default_rng(count(seed, "seed", 0))
        context = list(map(int, prefix))
        result = []
        for _ in range(max_new_tokens):
            if len(context) >= self.max_context:
                break
            distribution = TokenDistribution.from_logits(
                self.respond_to(np.array(context)).logits[-1], temperature, top_p
            )
            token = distribution.greedy() if greedy else distribution.sample(rng)
            result.append(
                GeneratedToken(tuple(context), distribution.probabilities, token)
            )
            context.append(token)
            if token == end_id:
                break
        return result
