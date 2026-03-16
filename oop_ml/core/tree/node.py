"""What a fitted tree is made of, and where routing actually happens.

Routing is not a search
-----------------------
A query row is not compared against the leaves and matched to the best one.
It answers the question at each node it reaches, and the answers force the
path, so there is exactly one leaf it can possibly arrive at. The leaves
partition the feature space into disjoint boxes that together cover all of it,
which means "which leaf" is an address rather than a decision.

That is why :meth:`TreeNode.leaf_for` is abstract with two one-line
implementations instead of a loop over leaves anywhere. A decision node passes
the row on; a leaf is where the row stopped. Two leaves are free to carry the
same prediction -- if the tree were ranking them that would be a contradiction
needing a rule, and because it is not, they are simply two regions that agree.

Where the answer is read off
----------------------------
Only inside the leaf that was reached. A classification leaf reports the class
holding the most rows; a regression leaf reports the mean. Neither has to be
pure -- a leaf is just a node that stopped -- so a leaf holding three fails and
one pass predicts fail, is wrong about one row, and stays wrong. Fixing that by
splitting again is exactly how a tree ends up with one leaf per row.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from oop_ml.core.tree.split import Split
from oop_ml.core.types import FloatArray


class TreeNode(ABC):
    """A node of a fitted tree, internal or terminal."""

    __slots__ = ()

    @abstractmethod
    def leaf_for(self, row: FloatArray) -> LeafNode:
        """The one leaf this row reaches.

        Parameters
        ----------
        row:
            ``(n_features,)``, in the fitted column order.

        Returns
        -------
        LeafNode
            The leaf the row lands in. Never a choice between several -- the
            answers to the questions on the way down determine it entirely.
        """

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """How many training rows reached this node."""

    @property
    @abstractmethod
    def depth(self) -> int:
        """How many levels sit below this node. A leaf is zero."""

    @property
    @abstractmethod
    def n_leaves(self) -> int:
        """How many leaves sit at or below this node. A leaf is one."""

    @abstractmethod
    def description_lines(self, indent: int = 0) -> list[str]:
        """This subtree as indented text, one line per node.

        Trees are worth reading, which is most of why anyone picks one, so
        rendering is behaviour on the node rather than something a caller has
        to write against the internals.
        """


class DecisionNode(TreeNode):
    """An internal node: one question, and where each answer leads.

    Parameters
    ----------
    split:
        The question this node asks.
    left:
        Where rows below the threshold go.
    right:
        Where rows at or above it go.
    n_samples:
        How many training rows reached this node.
    impurity:
        How mixed those rows were, before the split.
    """

    __slots__ = ("_impurity", "_left", "_n_samples", "_right", "_split")

    def __init__(
        self,
        split: Split,
        left: TreeNode,
        right: TreeNode,
        n_samples: int,
        impurity: float,
    ) -> None:
        self._split = split
        self._left = left
        self._right = right
        self._n_samples = n_samples
        self._impurity = impurity

    @property
    def split(self) -> Split:
        """The question this node asks."""
        return self._split

    @property
    def left(self) -> TreeNode:
        """Where rows below the threshold go."""
        return self._left

    @property
    def right(self) -> TreeNode:
        """Where rows at or above the threshold go."""
        return self._right

    @property
    def impurity(self) -> float:
        """How mixed the rows reaching this node were."""
        return self._impurity

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def depth(self) -> int:
        return 1 + max(self._left.depth, self._right.depth)

    @property
    def n_leaves(self) -> int:
        return self._left.n_leaves + self._right.n_leaves

    def leaf_for(self, row: FloatArray) -> LeafNode:
        if row[self._split.feature_index] < self._split.threshold:
            return self._left.leaf_for(row)

        return self._right.leaf_for(row)

    def description_lines(self, indent: int = 0) -> list[str]:
        pad = "  " * indent
        name = self._split.feature_name
        threshold = self._split.threshold

        return [
            f"{pad}{name} < {threshold:g} ?  "
            f"[n={self._n_samples}, impurity={self._impurity:.4f}, "
            f"gain={self._split.gain:.4f}]",
            *self._left.description_lines(indent + 1),
            *self._right.description_lines(indent + 1),
        ]


class LeafNode(TreeNode):
    """A terminal node, holding the answer for every row that reaches it.

    Parameters
    ----------
    prediction:
        What this leaf answers. A class position for a classifier, a quantity
        for a regressor.
    n_samples:
        How many training rows reached it.
    impurity:
        How mixed those rows were. Not necessarily zero -- a leaf is a node
        that stopped, not a node that succeeded.
    """

    __slots__ = ("_impurity", "_n_samples", "_prediction")

    def __init__(self, prediction: float, n_samples: int, impurity: float) -> None:
        self._prediction = prediction
        self._n_samples = n_samples
        self._impurity = impurity

    @property
    def prediction(self) -> float:
        """What this leaf answers."""
        return self._prediction

    @property
    def impurity(self) -> float:
        """How mixed the rows reaching this leaf were."""
        return self._impurity

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def depth(self) -> int:
        return 0

    @property
    def n_leaves(self) -> int:
        return 1

    def leaf_for(self, row: FloatArray) -> LeafNode:
        return self

    def description_lines(self, indent: int = 0) -> list[str]:
        pad = "  " * indent

        return [
            f"{pad}predict {self._prediction:g}  "
            f"[n={self._n_samples}, impurity={self._impurity:.4f}]"
        ]


class ClassificationLeaf(LeafNode):
    """A leaf that also remembers how its rows were divided between classes.

    The extra field is what ``predict_probabilities`` reads. Those shares are a
    probability in the sense that they are non-negative and sum to one, and not
    in the sense a fitted model offers: a leaf holding four rows can only ever
    report multiples of a quarter, so the resolution is ``1 / n_samples`` and
    reading them as calibrated confidence is the same mistake it was for a
    neighbour vote.

    Parameters
    ----------
    prediction:
        The class position holding the most rows. Ties go to the lowest index.
    class_shares:
        ``(n_classes,)``, summing to 1. Every fitted class gets an entry even
        where no row in this leaf belongs to it, or the matrix would change
        width depending on which leaf a query happened to reach.
    n_samples:
        How many training rows reached this leaf.
    impurity:
        How mixed they were.
    """

    __slots__ = ("_class_shares",)

    def __init__(
        self,
        prediction: float,
        class_shares: FloatArray,
        n_samples: int,
        impurity: float,
    ) -> None:
        super().__init__(prediction, n_samples, impurity)
        self._class_shares = class_shares

    @property
    def class_shares(self) -> FloatArray:
        """Each class's share of this leaf's rows, summing to 1."""
        return self._class_shares
