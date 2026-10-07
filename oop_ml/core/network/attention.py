"""A layer where every step decides for itself which other steps to read.

What is actually new here
-------------------------
A recurrent layer carries the past forward one step at a time, so what step
twenty knows about step one has been squeezed through nineteen intermediate
states. Attention does not carry anything. Every step looks at every other step
directly and takes a weighted average of what it finds, and the weights are
computed from the content rather than fixed by position.

This gives a distant step a direct route into the current output, without
passing through each intervening recurrent state. That shortens the dependency
path; it does not guarantee that gradients remain large. A saturated softmax
or other layers can still make a gradient small.

For an already available sequence, projections, comparisons and weighted sums
can be computed in batches without a recurrent loop over positions. Generating
new tokens autoregressively still proceeds one token at a time, because the
next input depends on the token just generated.

Three roles for the same step
------------------------------
Each step is projected three ways before anything is compared, and the three
have different jobs::

    query    what this step is looking for
    key      what this step offers to anyone looking
    value    what this step hands over when chosen

A step's attention to another is its query against that other's key, so the
comparison is between what one wants and what another advertises. The answer is
built from values, so what is advertised and what is handed over can differ,
which is what lets a step be found by one thing and contribute another.

Without three separate projections the comparison would be a step's raw
similarity to another, and the layer could only ever gather things that already
resembled each other. Learning what to look for is what the projections buy.

Why the scores are divided before they are squashed
-----------------------------------------------------
A query against a key is a sum of ``head_size`` products. If the entries are
independent, centred on zero and have variance one, that sum has a spread of the
square root of ``head_size``, so the scores grow with the width. Squashing wide
scores gives an answer near zero and one, which is a step reading exactly one
other step and, worse, a slope of almost nothing everywhere, since a squashing
that has saturated has stopped responding. Dividing by the square root of
``head_size`` cancels the growth, so the scores keep the same spread whatever
the width is.

Why several heads rather than one
----------------------------------
One set of projections can express one notion of relevance. A step in a sentence
usually needs several at once, the grammatical subject and the topic and the
nearest preceding number, and one weighted average cannot hold them apart.
Several heads run the same arithmetic on different slices of the width and their
answers are laid side by side, allowing the layer to learn several notions of
relevance. At fixed model width, splitting heads preserves the leading number
of operations in the dot products and weighted sums. Memory use and execution
overhead can still change with the number of heads.

Why one weight block again
---------------------------
All four projections read ``n_model`` numbers and answer ``n_model`` numbers.
The three role projections read a step; the output projection reads the merged
heads, which is the same width again. So they stack down the rows of one matrix
of ``(4 * n_model, n_model)``, exactly as the gates of a recurrent cell do, and
:class:`~oop_ml.core.network.gradient.LayerGradient` holds this layer unchanged
too. That is now twice this arrangement has paid, which is the argument for the
gradient object not having grown to hold a list of blocks.

Why looking ahead is a choice and not an oversight
----------------------------------------------------
By default a step reads every step, which is what a layer summarising a whole
sentence wants. A layer predicting the next word must not, since a step that
can read the answer will learn to copy it and score perfectly while having
learned nothing. Setting ``looks_ahead`` to ``False`` stops each step reading
anything after itself, and it is done by pushing those scores to minus infinity
*before* the squashing rather than by zeroing the weights after it, because the
weights along a step must still add to one. Zeroing afterwards leaves them
adding to less than one, which quietly shrinks every answer near the start of
the sequence.

What is deliberately absent
----------------------------
This layer attends to its own input, which is the case a transformer's encoder
and its decoder's first block both use. Attending to some *other* sequence,
which is what a decoder does when it reads an encoder's answer, needs a layer
reading two blocks rather than one, and every layer in this package reads one.
That is a change to what a layer is rather than another layer, so it is recorded
here rather than half-built.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray, MaskArray

SEQUENCE_EXTENTS = 2
"""An attention layer reads ``(n_steps, n_model)``, which is two extents."""

QUERY = 0
KEY = 1
VALUE = 2
OUTPUT = 3
"""Which band of the weight block each projection occupies."""

PROJECTIONS = 4
"""How many bands there are, which sets the weight block's height."""


def _checked_setting(value: int, lowest: int, name: str) -> int:
    """Read one whole-number setting, refusing anything below its floor.

    Raises
    ------
    InvalidValuesError
        If the value is not a whole number, or is below its floor.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidValuesError(f"{name} is a whole number, got {value!r}")
    if value < lowest:
        raise InvalidValuesError(f"{name} is at least {lowest}, got {value}")
    return value


def _softmax_along_last(scores: FloatArray) -> FloatArray:
    """Squash each row of scores to shares that add to one.

    The largest score in a row is subtracted before exponentiating, which
    changes nothing about the answer and everything about whether it can be
    computed. Attention scores are unbounded, and ``exp`` of a few hundred is an
    infinity, so the subtraction is what stops a perfectly ordinary sequence
    answering not-a-number. It is the same guard
    :mod:`oop_ml.core.logistic` applies for the same reason.

    A row that has been masked to minus infinity everywhere but its own
    position still has a finite largest entry, so the subtraction stays well
    defined under masking.
    """
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


class AttentionResponse(LayerResponse):
    """A layer response that also remembers who read whom, and what they read.

    The backward pass needs the attention weights, the three role projections
    and the merged heads, and none is recoverable from the answer. The weights
    are worth exposing for their own sake as well, since they are the one part
    of a network that says in plain terms which step the layer thought mattered.

    ``scores`` and ``outputs`` are the same block, because this layer applies no
    bend to its answer. The squashing happens inside, on the attention scores,
    which are kept separately.
    """

    __slots__ = ("_attention", "_keys", "_merged", "_queries", "_values")

    @classmethod
    def recording(
        cls,
        inputs: FloatArray,
        outputs: FloatArray,
        attention: FloatArray,
        queries: FloatArray,
        keys: FloatArray,
        values: FloatArray,
        merged: FloatArray,
    ) -> AttentionResponse:
        """Wrap a forward pass this layer just built, along with its workings.

        The :meth:`~oop_ml.core.network.layer.LayerResponse.already_checked`
        pattern, sound for the same reason: the caller allocated every block and
        has shared none of them.
        """
        response = cls.__new__(cls)
        for block in (inputs, outputs, attention, queries, keys, values, merged):
            block.setflags(write=False)
        response._inputs = inputs
        response._scores = outputs
        response._outputs = outputs
        response._attention = attention
        response._queries = queries
        response._keys = keys
        response._values = values
        response._merged = merged
        return response

    @property
    def attention(self) -> FloatArray:
        """``(n_rows, n_heads, n_steps, n_steps)``, how much each step read each.

        Entry ``[row, head, from_step, to_step]`` is the share of
        ``from_step``'s answer taken from ``to_step``, so every row of the last
        axis adds to one.
        """
        return self._attention

    @property
    def queries(self) -> FloatArray:
        """``(n_rows, n_heads, n_steps, head_size)``, what each step looked for."""
        return self._queries

    @property
    def keys(self) -> FloatArray:
        """``(n_rows, n_heads, n_steps, head_size)``, what each step advertised."""
        return self._keys

    @property
    def values(self) -> FloatArray:
        """``(n_rows, n_heads, n_steps, head_size)``, what each step handed over."""
        return self._values

    @property
    def merged(self) -> FloatArray:
        """``(n_rows, n_steps, n_model)``, every head's answer laid side by side."""
        return self._merged

    def __eq__(self, other: object) -> bool:
        # Who read whom is the whole reason this class exists, so two responses
        # that gathered differently are not the same response however well their
        # answers line up. The base settles the type check and the blocks.
        #
        # The base answers NotImplemented when the two types differ, and
        # ``bool(NotImplemented)`` is True, so coercing it would report two
        # different types equal, which is the failure this override exists to
        # prevent. It is guarded explicitly.
        if not isinstance(other, AttentionResponse):
            return NotImplemented
        settled = super().__eq__(other)
        if settled is NotImplemented or settled is False:
            return settled
        return bool(np.array_equal(self._attention, other._attention))

    def __hash__(self) -> int:
        return NotImplemented

    def __repr__(self) -> str:
        return f"AttentionResponse(attention={self._attention.shape!r})"


class MultiHeadAttention(Layer):
    """Every step reads every step, in several notions of relevance at once.

    Parameters
    ----------
    reads:
        ``(n_steps, n_model)``. That many steps, each described by that many
        numbers.
    n_heads:
        How many notions of relevance to run at once. Each reads a slice of the
        width, so ``n_model`` must divide evenly by it, and one head reading the
        whole width is the plain case.
    looks_ahead:
        Whether a step may read steps after itself. The default is that it may,
        which is what summarising a whole sequence wants. See the module
        docstring for why a layer predicting the next step must set this to
        ``False`` and why the stopping is done before the squashing.
    random_seed:
        Fixes the starting projections.

    Raises
    ------
    ShapeMismatchError
        If ``reads`` is not a sequence of two extents.
    InvalidValuesError
        If a setting is not a whole number or is below its floor, or if the
        width does not divide evenly among the heads.
    """

    def __init__(
        self,
        reads: Sequence[int],
        n_heads: int = 1,
        looks_ahead: bool = True,
        random_seed: int = 0,
    ) -> None:
        try:
            extents = tuple(reads)
        except TypeError:
            raise ShapeMismatchError(
                "an attention layer reads (n_steps, n_model), got "
                f"{reads!r}, which is not a sequence of extents"
            ) from None
        if len(extents) != SEQUENCE_EXTENTS:
            raise ShapeMismatchError(
                "an attention layer reads (n_steps, n_model), got "
                f"{len(extents)} extents"
            )

        n_steps = _checked_setting(extents[0], 1, "n_steps")
        n_model = _checked_setting(extents[1], 1, "n_model")
        checked_heads = _checked_setting(n_heads, 1, "n_heads")

        if n_model % checked_heads != 0:
            raise InvalidValuesError(
                f"the heads split the width between them, so n_model {n_model} "
                f"must divide by n_heads {checked_heads}, and it leaves "
                f"{n_model % checked_heads}"
            )

        generator = np.random.default_rng(random_seed)
        weights = generator.normal(size=(PROJECTIONS * n_model, n_model)) * sqrt(
            1.0 / n_model
        )

        self._configure(
            shape=LayerShape((n_steps, n_model), (n_steps, n_model)),
            n_heads=checked_heads,
            looks_ahead=looks_ahead,
            weights=weights,
            biases=np.zeros(PROJECTIONS * n_model),
        )

    def _configure(
        self,
        shape: LayerShape,
        n_heads: int,
        looks_ahead: bool,
        weights: FloatArray,
        biases: FloatArray,
    ) -> None:
        """Fill every slot from an already validated configuration.

        Separated from ``__init__`` for the reason
        :meth:`~oop_ml.core.network.convolution.Conv2d._configure` gives: a
        training step builds the next layer from the last one's geometry and a
        corrected set of parameters, and rerunning the constructor would throw
        a draw away and misreport where the numbers came from.
        """
        self._shape = shape
        self._n_steps, self._n_model = shape.reads
        self._n_heads = n_heads
        self._head_size = self._n_model // n_heads
        self._looks_ahead = looks_ahead

        weight_block = np.array(weights, dtype=np.float64, copy=True)
        bias_block = np.array(biases, dtype=np.float64, copy=True)
        weight_block.setflags(write=False)
        bias_block.setflags(write=False)
        self._weights = weight_block
        self._biases = bias_block

    def with_parameters(
        self, weights: FloatArray, biases: FloatArray
    ) -> MultiHeadAttention:
        """The same layer with different projections, and nothing else changed.

        Raises
        ------
        ShapeMismatchError
            If either block is not the arrangement this layer's parameters have.
        """
        expected_weights = (PROJECTIONS * self._n_model, self._n_model)
        weight_block = np.asarray(weights, dtype=np.float64)
        bias_block = np.asarray(biases, dtype=np.float64)

        if weight_block.shape != expected_weights:
            raise ShapeMismatchError(
                f"this layer's weights are {expected_weights}, got {weight_block.shape}"
            )
        if bias_block.shape != (PROJECTIONS * self._n_model,):
            raise ShapeMismatchError(
                f"this layer's biases are {(PROJECTIONS * self._n_model,)}, got "
                f"{bias_block.shape}"
            )

        replacement = MultiHeadAttention.__new__(MultiHeadAttention)
        replacement._configure(
            shape=self._shape,
            n_heads=self._n_heads,
            looks_ahead=self._looks_ahead,
            weights=weight_block,
            biases=bias_block,
        )
        return replacement

    @property
    def shape(self) -> LayerShape:
        """``(n_steps, n_model)`` in and the same out.

        A step keeps its width, which is what lets this layer sit inside a
        residual connection and what lets several of them stack.
        """
        return self._shape

    @property
    def n_heads(self) -> int:
        """How many notions of relevance run at once."""
        return self._n_heads

    @property
    def head_size(self) -> int:
        """How wide a slice each head reads, which is the width over the count."""
        return self._head_size

    @property
    def looks_ahead(self) -> bool:
        """Whether a step may read steps after itself."""
        return self._looks_ahead

    @property
    def weights(self) -> FloatArray:
        """``(4 * n_model, n_model)``, frozen, four projections stacked."""
        return self._weights

    @property
    def biases(self) -> FloatArray:
        """``(4 * n_model,)``, frozen, aligned with the weight rows."""
        return self._biases

    def _band(self, projection: int) -> slice:
        """Which rows of the weight block belong to one projection."""
        return slice(projection * self._n_model, (projection + 1) * self._n_model)

    def _split_into_heads(self, block: FloatArray) -> FloatArray:
        """``(n_rows, n_steps, n_model)`` to ``(n_rows, n_heads, n_steps, head)``.

        The width is cut into consecutive slices, one per head, and the head
        axis is moved in front of the step axis so that each head's comparison
        of every step against every step is an ordinary matrix multiply.
        """
        n_rows = block.shape[0]
        by_head = block.reshape(n_rows, self._n_steps, self._n_heads, self._head_size)
        return np.transpose(by_head, (0, 2, 1, 3))

    def _merge_heads(self, block: FloatArray) -> FloatArray:
        """The inverse, laying every head's answer side by side again."""
        n_rows = block.shape[0]
        by_step = np.transpose(block, (0, 2, 1, 3))
        return by_step.reshape(n_rows, self._n_steps, self._n_model)

    def _blocked_from_looking_ahead(self) -> MaskArray:
        """Which pairs of steps are not allowed to see each other.

        ``True`` where the reading step comes before the read step, which is
        exactly the entries above the diagonal.
        """
        positions = np.arange(self._n_steps)
        return positions[None, :] > positions[:, None]

    def _response_for(
        self, inputs: FloatArray, purpose: PassPurpose
    ) -> AttentionResponse:
        """Project three ways, compare every step to every step, gather, project.

        The projections are done on a block flattened to one row per step of
        every observation, since a projection reads one step and knows nothing
        about which observation or which position that step came from. That is
        one matrix multiply for the whole batch rather than a loop, and it is
        the reason this layer is quick where recurrence is not.
        """
        # A projection reads one step and knows nothing about which observation
        # or which position that step came from, so flatten first and project the
        # whole batch in one multiply:
        #
        #   flat = inputs.reshape(n_rows * self._n_steps, self._n_model)
        #   roles = slice(0, (PROJECTIONS - 1) * self._n_model)
        #   projected = flat @ self._weights[roles].T + self._biases[roles]
        #   by_role = projected.reshape(n_rows, self._n_steps, PROJECTIONS - 1,
        #                               self._n_model)
        #
        # The three bands are in the order QUERY, KEY, VALUE, so
        # by_role[:, :, QUERY, :] is the query block. Push each through
        # self._split_into_heads to reach (n_rows, n_heads, n_steps, head_size).
        #
        # Then the comparison, every step's query against every step's key:
        #
        #   scores = queries @ np.swapaxes(keys, -1, -2) / sqrt(self._head_size)
        #
        # The divisor is what stops the scores growing with the width; the module
        # docstring has the argument.
        #
        # If this layer may not look ahead, push the blocked entries to minus
        # infinity BEFORE squashing, using self._blocked_from_looking_ahead().
        # Zeroing after the squash leaves the early steps' shares adding to less
        # than one, which shrinks their answers with nothing raised.
        #
        #   attention = _softmax_along_last(scores)
        #   merged = self._merge_heads(attention @ values)
        #   outputs = merged @ self._weights[output band].T + biases[output band]
        #
        # Answer AttentionResponse.recording with inputs, outputs, attention,
        # queries, keys, values and merged. The backward pass needs every one.
        n_rows = inputs.shape[0]
        flat_inputs = inputs.reshape(-1, self._n_model)
        roles = slice(0, (PROJECTIONS - 1) * self._n_model)
        projected = flat_inputs @ self._weights[roles].T + self._biases[roles]
        by_role = projected.reshape(
            n_rows, self._n_steps, PROJECTIONS - 1, self._n_model
        )
        queries = self._split_into_heads(by_role[:, :, QUERY, :])
        keys = self._split_into_heads(by_role[:, :, KEY, :])
        values = self._split_into_heads(by_role[:, :, VALUE, :])

        scores = queries @ np.swapaxes(keys, -1, -2) / sqrt(self._head_size)
        if not self._looks_ahead:
            scores = np.where(self._blocked_from_looking_ahead(), -np.inf, scores)
        attention = _softmax_along_last(scores)
        merged = self._merge_heads(attention @ values)
        output_band = self._band(OUTPUT)
        outputs = merged @ self._weights[output_band].T + self._biases[output_band]

        return AttentionResponse.recording(
            inputs=inputs,
            outputs=outputs,
            attention=attention,
            queries=queries,
            keys=keys,
            values=values,
            merged=merged,
        )

    def respond_to(
        self, inputs: FloatArray, purpose: PassPurpose = PassPurpose.PREDICTING
    ) -> AttentionResponse:
        """Push a sequence block through this layer.

        Overridden only to narrow what comes back, so a caller reading who
        attended to whom does not have to ask whether the response carries it.
        The base still owns every check, and the narrowing runs after the call
        so a bad block is refused by this library's own typed error rather than
        turning into an assertion.
        """
        response = super().respond_to(inputs, purpose)
        assert isinstance(response, AttentionResponse)
        return response

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """Walk back through the gather, the squashing and the four projections.

        The one term worth naming is the squashing. Each step's attention along
        the sequence adds to one, so raising one share lowers the others, and
        the slope of the squashing is therefore not a number per entry but a
        whole matrix per step. It reduces to::

            d score  =  attention * ( d attention  -  sum( d attention * attention ) )

        where the sum runs along the read steps. Dropping the subtracted term
        leaves a gradient that ignores the competition between steps, which
        still trains and still descends.

        A second term is easy to miss: the query and the key both feed the same
        comparison, so what the loss wants from the projections is two separate
        blocks that are then laid beside the value's and multiplied through one
        weight band each.

        Raises
        ------
        ShapeMismatchError
            If the arriving block is not what this layer answers with, or if the
            response carries no workings because it came from another layer.
        """
        # Guards first: self._checked_arriving, then refuse a response carrying
        # no workings. Then walk back in the order the forward pass built it.
        #
        # 1. The output projection. Flatten the arriving block and merged to
        #    (n_rows * n_steps, n_model):
        #
        #      d W_out   =  flat_arriving.T @ flat_merged
        #      d b_out   =  flat_arriving.sum(axis=0)
        #      d merged  =  flat_arriving @ self._weights[output band]
        #
        #    reshape d merged back and split it into heads to get d context.
        #
        # 2. The gather, context = attention @ values:
        #
        #      d attention  =  d_context @ np.swapaxes(values, -1, -2)
        #      d values     =  np.swapaxes(attention, -1, -2) @ d_context
        #
        # 3. The squashing, and this is the term worth care. Each step's shares
        #    add to one, so raising one lowers the others and the slope is a
        #    whole matrix per step rather than a number per entry. It reduces to
        #
        #      d scores = attention * ( d attention
        #                 - np.sum(d attention * attention, axis=-1, keepdims=True) )
        #
        #    then divide by sqrt(self._head_size), matching the forward divisor.
        #    Dropping the subtracted term leaves a gradient that ignores the
        #    competition between steps, and it still trains.
        #
        # 4. The comparison, scores = queries @ keys transposed. Both sides
        #    collect a slope from the same block, and they are transposes of each
        #    other's arithmetic:
        #
        #      d queries  =  d_scores @ keys
        #      d keys     =  np.swapaxes(d_scores, -1, -2) @ queries
        #
        # 5. Back through the heads and the three role projections. Merge each of
        #    d queries, d keys and d values with self._merge_heads, concatenate
        #    them along axis 2 in that order, flatten to
        #    (n_rows * n_steps, 3 * n_model), and then
        #
        #      d W_roles    =  flat_roles.T @ flat_inputs
        #      d b_roles    =  flat_roles.sum(axis=0)
        #      passed_down  =  (flat_roles @ self._weights[roles]) reshaped back
        #
        # The gradient is the role slopes stacked on top of the output slopes, in
        # the same band order the forward pass read them.
        checked = self._checked_arriving(response, arriving)
        if not isinstance(response, AttentionResponse):
            raise ShapeMismatchError("attention needs a response carrying its workings")

        flat_arriving = checked.reshape(-1, self._n_model)
        flat_merged = response.merged.reshape(-1, self._n_model)
        output_band = self._band(OUTPUT)
        output_weight_slopes = flat_arriving.T @ flat_merged
        output_bias_slopes = flat_arriving.sum(axis=0)
        merged_slopes = (flat_arriving @ self._weights[output_band]).reshape(
            checked.shape
        )
        context_slopes = self._split_into_heads(merged_slopes)

        attention_slopes = context_slopes @ np.swapaxes(response.values, -1, -2)
        value_slopes = np.swapaxes(response.attention, -1, -2) @ context_slopes
        score_slopes = response.attention * (
            attention_slopes
            - np.sum(attention_slopes * response.attention, axis=-1, keepdims=True)
        )
        # Masked shares are exactly zero, so their scores receive no gradient.
        score_slopes /= sqrt(self._head_size)
        query_slopes = score_slopes @ response.keys
        key_slopes = np.swapaxes(score_slopes, -1, -2) @ response.queries

        role_slopes = np.concatenate(
            (
                self._merge_heads(query_slopes),
                self._merge_heads(key_slopes),
                self._merge_heads(value_slopes),
            ),
            axis=2,
        ).reshape(-1, (PROJECTIONS - 1) * self._n_model)
        flat_inputs = response.inputs.reshape(-1, self._n_model)
        roles = slice(0, (PROJECTIONS - 1) * self._n_model)
        return LayerCorrection(
            passed_down=(role_slopes @ self._weights[roles]).reshape(
                response.inputs.shape
            ),
            gradient=LayerGradient(
                weights=np.concatenate(
                    (role_slopes.T @ flat_inputs, output_weight_slopes), axis=0
                ),
                biases=np.concatenate((role_slopes.sum(axis=0), output_bias_slopes)),
            ),
        )

    def stepped_by(
        self, gradient: LayerGradient | None, learning_rate: float
    ) -> MultiHeadAttention:
        """A new layer of the same geometry, with its projections nudged.

        Raises
        ------
        ShapeMismatchError
            If no gradient is supplied, since this layer has parameters, or if
            the gradient does not describe them.
        """
        if gradient is None:
            raise ShapeMismatchError(
                "an attention layer has parameters and needs a gradient to step by"
            )
        if gradient.weights.shape != self._weights.shape:
            raise ShapeMismatchError(
                f"this layer's weights are {self._weights.shape}, and the "
                f"gradient describes {gradient.weights.shape}"
            )

        return self.with_parameters(
            self._weights - learning_rate * gradient.weights,
            self._biases - learning_rate * gradient.biases,
        )

    def __repr__(self) -> str:
        return (
            f"MultiHeadAttention(reads={self._shape.reads!r}, "
            f"n_heads={self._n_heads!r}, looks_ahead={self._looks_ahead!r})"
        )
