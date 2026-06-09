"""A binary tree over the vocabulary, so a softmax becomes a walk of yes-or-no.

The problem hierarchical softmax solves
---------------------------------------
Predicting one word out of forty thousand is a forty-thousand-way choice, and
its exact gradient touches forty thousand output rows. Arrange the words as the
leaves of a binary tree instead, and reaching a word is a sequence of binary
choices, left or right at each internal node on the way down. Each internal
node gets its own vector, the probability of a word is the product of the
probabilities of the choices on its path, and the gradient touches only the
nodes on that path -- about ``log2(n)`` of them, sixteen for forty thousand
words. That is Morin and Bengio's (2005) hierarchical softmax, and the tree is
the whole trick.

Why Huffman
-----------
Any binary tree works; the *shape* decides the cost. Mikolov et al. (2013) used
Huffman's code (1952), which gives frequent words short paths and rare words
long ones, so the expected number of nodes touched per training word is the
smallest any prefix code can achieve. On counts of ``5, 2, 1, 1`` the tree
puts the commonest word one step from the root and the two rarest three steps
down: expected depth ``(5 * 1 + 2 * 2 + 1 * 3 + 1 * 3) / 9 = 1.67`` where a
balanced tree over four words costs exactly 2. The Kraft equality holds for a
full binary tree, ``sum 2 ** -depth = 1``, and a test says so.

The code as a path
------------------
:class:`HuffmanCode` carries, for one word, the internal nodes visited from the
root and the bit taken at each. Hierarchical softmax reads them in step: at
node ``k`` the model must say bit ``k``, so the node's vector is trained as a
binary logistic classifier whose label is the bit. Internal nodes are numbered
in the order the algorithm created them, ``0`` to ``n - 2``, which is what the
output-vector table is indexed by. A vocabulary of one word has a tree with no
internal nodes and an empty code, and hierarchical softmax over it has nothing
to learn; the embedder refuses that case rather than fitting silently.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator, Sequence

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError


class HuffmanCode:
    """One word's path from the root: which internal nodes, and which way at each.

    Parameters
    ----------
    word_id:
        The leaf this code reaches.
    nodes:
        The internal nodes visited, root first.
    bits:
        The choice taken at each node, ``0`` or ``1``, one per node.

    Raises
    ------
    InvalidValuesError
        If ``nodes`` and ``bits`` differ in length or a bit is not 0 or 1.
    """

    __slots__ = ("_bits", "_nodes", "_word_id")

    def __init__(self, word_id: int, nodes: Sequence[int], bits: Sequence[int]) -> None:
        if len(nodes) != len(bits):
            raise InvalidValuesError(
                f"a code takes one bit at each node, got {len(nodes)} nodes and "
                f"{len(bits)} bits"
            )
        if any(bit not in (0, 1) for bit in bits):
            raise InvalidValuesError("every bit of a Huffman code is 0 or 1")

        self._word_id = int(word_id)
        self._nodes = tuple(int(node) for node in nodes)
        self._bits = tuple(int(bit) for bit in bits)

    @property
    def word_id(self) -> int:
        """The leaf this code reaches."""
        return self._word_id

    @property
    def nodes(self) -> tuple[int, ...]:
        """The internal nodes on the path, root first."""
        return self._nodes

    @property
    def bits(self) -> tuple[int, ...]:
        """The branch taken at each node."""
        return self._bits

    @property
    def depth(self) -> int:
        """How many binary choices reach the word."""
        return len(self._nodes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HuffmanCode):
            return NotImplemented
        return (
            self._word_id == other._word_id
            and self._nodes == other._nodes
            and self._bits == other._bits
        )

    def __hash__(self) -> int:
        return hash((self._word_id, self._nodes, self._bits))

    def __repr__(self) -> str:
        bits = "".join(str(bit) for bit in self._bits)
        return f"HuffmanCode(word_id={self._word_id}, bits={bits})"


class HuffmanTree:
    """The Huffman tree over a vocabulary's counts, one code per word.

    Parameters
    ----------
    codes:
        One :class:`HuffmanCode` per word id, in id order.
    n_internal_nodes:
        How many internal nodes the tree has: ``n_words - 1``.

    Raises
    ------
    EmptyValuesError
        If there are no codes.
    InvalidValuesError
        If a code's word id is not its position, or a node is out of range.
    """

    __slots__ = ("_codes", "_n_internal_nodes")

    def __init__(self, codes: Sequence[HuffmanCode], n_internal_nodes: int) -> None:
        if len(codes) == 0:
            raise EmptyValuesError("a Huffman tree needs at least one word")

        for position, code in enumerate(codes):
            if code.word_id != position:
                raise InvalidValuesError(
                    f"the code at position {position} is for word {code.word_id}"
                )
            if any(not 0 <= node < n_internal_nodes for node in code.nodes):
                raise InvalidValuesError(
                    f"the code for word {position} visits a node outside the "
                    f"{n_internal_nodes} internal nodes"
                )

        self._codes = tuple(codes)
        self._n_internal_nodes = int(n_internal_nodes)

    @classmethod
    def from_counts(cls, counts: Sequence[int]) -> HuffmanTree:
        """Build the tree by merging the two least frequent nodes, repeatedly.

        Ties in count break by which node was created first, so the tree is a
        function of the counts alone. The first of the two merged nodes takes
        bit ``0`` and the second bit ``1``, as in the reference implementation.

        Raises
        ------
        EmptyValuesError
            If there are no counts.
        InvalidValuesError
            If a count is below one.
        """
        if len(counts) == 0:
            raise EmptyValuesError("a Huffman tree needs at least one word")

        if any(count < 1 for count in counts):
            raise InvalidValuesError(
                "every word of the vocabulary was seen at least once"
            )

        n_words = len(counts)
        # A heap entry is (count, creation order, node). Leaves are nodes
        # 0 .. n_words - 1; internal nodes continue from n_words upward and
        # are renumbered from zero for the codes.
        heap: list[tuple[int, int, int]] = [
            (int(count), position, position) for position, count in enumerate(counts)
        ]
        heapq.heapify(heap)

        parent: dict[int, int] = {}
        bit_taken: dict[int, int] = {}
        next_node = n_words
        while len(heap) > 1:
            first_count, _, first = heapq.heappop(heap)
            second_count, _, second = heapq.heappop(heap)
            parent[first] = next_node
            bit_taken[first] = 0
            parent[second] = next_node
            bit_taken[second] = 1
            heapq.heappush(heap, (first_count + second_count, next_node, next_node))
            next_node += 1

        codes: list[HuffmanCode] = []
        for word_id in range(n_words):
            nodes: list[int] = []
            bits: list[int] = []
            node = word_id
            while node in parent:
                nodes.append(parent[node] - n_words)
                bits.append(bit_taken[node])
                node = parent[node]
            codes.append(HuffmanCode(word_id, nodes[::-1], bits[::-1]))

        return cls(codes, n_words - 1)

    @property
    def n_words(self) -> int:
        """How many leaves the tree has."""
        return len(self._codes)

    @property
    def n_internal_nodes(self) -> int:
        """How many internal nodes, which is how many output vectors are needed."""
        return self._n_internal_nodes

    def code_of(self, word_id: int) -> HuffmanCode:
        """The path to ``word_id``.

        Raises
        ------
        InvalidValuesError
            If the id is outside the vocabulary.
        """
        if not 0 <= word_id < len(self._codes):
            raise InvalidValuesError(
                f"word id {word_id} is outside a tree of {self.n_words} words"
            )
        return self._codes[word_id]

    def expected_depth(self, counts: Sequence[int]) -> float:
        """The count-weighted mean number of choices per training word.

        The quantity Huffman's construction minimises over all prefix codes.

        Raises
        ------
        InvalidValuesError
            If there is not one count per word.
        """
        if len(counts) != self.n_words:
            raise InvalidValuesError(
                f"expected {self.n_words} counts, one per word, got {len(counts)}"
            )
        total = sum(counts)
        return (
            sum(
                count * code.depth
                for count, code in zip(counts, self._codes, strict=True)
            )
            / total
        )

    def __iter__(self) -> Iterator[HuffmanCode]:
        return iter(self._codes)

    def __len__(self) -> int:
        return len(self._codes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HuffmanTree):
            return NotImplemented
        return self._codes == other._codes

    def __hash__(self) -> int:
        return hash(self._codes)

    def __repr__(self) -> str:
        return (
            f"HuffmanTree(n_words={self.n_words}, "
            f"n_internal_nodes={self._n_internal_nodes})"
        )
