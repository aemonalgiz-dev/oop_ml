"""What every scikit-learn wrapper does before and after the engine runs.

The regression and classification families share one set of chores: turn a
validated :class:`~oop_ml.core.data.feature_set.FeatureSet` into the row-major
array an engine reads, match query features to the fitted names, rebuild a
model from its own configuration, put an engine's separated ``coef_`` and
``intercept_`` back into the order the linear frame expects, walk an engine's
fitted tree into this library's nodes, and name this library's metrics and
kernels in the engine's vocabulary. Each of those was written once for the
regression family and is used unchanged by the classification one, so they
live here rather than in either.

Nothing here is a model. A wrapper imports what it needs and supplies the
engine, the translation of its own fields, and the reading of the engine's
fitted attributes back into the library's learned-parameter objects.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Self

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.design_matrix import DesignMatrix
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.row_block import rows_of
from oop_ml.core.distance.calculations import Distance, MinkowskiDistance
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.kernel.functions import (
    Kernel,
    LinearKernel,
    PolynomialKernel,
    RadialBasisKernel,
    SigmoidKernel,
)
from oop_ml.core.tree.node import DecisionNode, LeafNode, TreeNode
from oop_ml.core.tree.split import Split
from oop_ml.core.types import FloatArray

ENGINE_LEAF = -1
"""How scikit-learn's tree marks a node with no children."""

ENGINE_METRIC_NAMES: dict[DistanceMetric, str] = {
    DistanceMetric.EUCLIDEAN: "euclidean",
    DistanceMetric.MANHATTAN: "manhattan",
    DistanceMetric.CHEBYSHEV: "chebyshev",
    DistanceMetric.COSINE: "cosine",
    DistanceMetric.HAMMING: "hamming",
    DistanceMetric.CANBERRA: "canberra",
}
"""Each of this library's metrics under the name the engine knows it by.

The six happen to coincide with the enum's own values, and the mapping is
written out anyway so the translation is a thing a reader can see rather than
a coincidence the code leans on.
"""


def matrix_of(feature_set: FeatureSet) -> FloatArray:
    """The features as the ``(n_rows, n_features)`` array an engine reads.

    Row-major, because every scikit-learn estimator copies a column-major
    input into row-major before it starts, and doing it once here is cheaper
    than letting each engine do it on every call.
    """
    return np.ascontiguousarray(feature_set.feature_matrix, dtype=np.float64)


def matched_matrix(
    feature_names: Sequence[str], input_values: Sequence[Feature]
) -> FloatArray:
    """The query rows as an array, in the column order the fit saw.

    Raises
    ------
    EmptyValuesError
        If no features are supplied.
    NonUniqueFeaturesError
        If two supplied features share a name.
    NonEqualArrayLengthError
        If the supplied columns are not all the same length.
    InvalidValuesError
        If the supplied names do not match the fitted ones exactly.
    """
    return matrix_of(FeatureSet.matching(feature_names, input_values))


def configuration_of(model: Fittable) -> dict[str, Any]:
    """Every hyperparameter of ``model`` by name, for rebuilding it.

    Read with ``getattr`` rather than ``model_dump``, which recurses a nested
    model into a plain dict and then cannot rebuild it. The same failure
    ``Candidate.applied_to`` documents.
    """
    return {name: getattr(model, name) for name in type(model).model_fields}


class EngineMember(Fittable):
    """A wrapper that can be a member of a scikit-learn ensemble.

    The engine's bagging takes a scikit-learn estimator as its prototype and
    hands back fitted scikit-learn estimators as its members. For the
    ensemble wrappers to keep this library's contract, ``members`` has to be
    a tuple of this library's models, so a member type needs two things: an
    unfitted engine to hand the ensemble, and a way to wrap a fitted engine
    the ensemble hands back. A model that cannot do both cannot be bagged in
    this backend, and the bagging wrappers say so at construction.
    """

    @abstractmethod
    def _engine_prototype(self, n_rows: int) -> Any:
        """An unfitted scikit-learn estimator configured like this model.

        Parameters
        ----------
        n_rows:
            How many rows the engine will be fitted on. Most models ignore
            it; the lasso needs it, because its penalty translation carries
            the row count and a bagging engine fits each member on as many
            rows as the training set holds.
        """

    @abstractmethod
    def _adopting(self, engine: Any, training: Dataset) -> Self:
        """A fitted copy of this model wrapped around ``engine``.

        Parameters
        ----------
        engine:
            A scikit-learn estimator that has already been fitted.
        training:
            The rows the engine read. What a wrapper takes from it varies. A
            tree and a linear model take only the feature names; a neighbour
            model keeps every row, since the rows are what it predicts from.

        Returns
        -------
        Self
            A new model, fitted, configured as this one is.
        """


def solution_of(engine: Any, design_matrix: DesignMatrix) -> FloatArray:
    """The engine's fitted weights, in the order the design matrix uses.

    The frame expects the intercept first whenever there is one, followed by
    one weight per feature column. scikit-learn keeps the two apart as
    ``intercept_`` and ``coef_``, so this puts them back together. A
    regression engine reports a scalar intercept and a classification engine
    a one-entry array; both are read through :func:`scalar_intercept_of`.
    """
    weights = np.asarray(engine.coef_, dtype=np.float64).ravel()

    if not design_matrix.has_intercept:
        return weights

    return np.concatenate([[scalar_intercept_of(engine)], weights])


def scalar_intercept_of(engine: Any) -> float:
    """The engine's intercept as one number, whatever shape it reports it in.

    A regression engine stores a float; a binary classification engine stores
    a length-one array, and numpy no longer converts a one-element array to a
    scalar quietly.
    """
    return float(np.asarray(engine.intercept_, dtype=np.float64).ravel()[0])


def predictor_columns(design_matrix: DesignMatrix) -> FloatArray:
    """The feature columns alone, without the frame's ones column.

    Every engine here learns its own intercept, centred and unpenalised, so it
    must not also be handed a column of ones. With a penalty that column would
    be shrunk like any other weight, which is exactly the intercept bug the
    design matrix exists to make unwriteable in the numpy backend.
    """
    values = design_matrix.values

    if design_matrix.has_intercept:
        return values[:, 1:]

    return values


def fit_watching_convergence(
    engine: Any, matrix: FloatArray, targets: FloatArray
) -> bool:
    """Fit the engine and report whether it ran out of iterations.

    A ``ConvergenceWarning`` is the only signal an iterative engine gives that
    its cap was reached; ``n_iter_`` alone cannot separate a fit that settled
    on the last permitted step from one that ran out. The warning is caught
    rather than shown, because ``converged`` is where this library says that.
    Every other warning the engine issues is re-raised as it was.

    Returns
    -------
    bool
        ``True`` if the engine warned that it reached its cap.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.fit(matrix, targets)

    reached_the_cap = False
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            reached_the_cap = True
        else:
            warnings.warn(warning.message, warning.category, stacklevel=3)

    return reached_the_cap


def converted_tree(
    engine: Any,
    feature_names: Sequence[str],
    leaf_of: Callable[[Any, int], LeafNode],
) -> TreeNode:
    """The engine's fitted tree as this library's nodes.

    scikit-learn keeps its tree as parallel arrays indexed by node, and this
    walks them once into :class:`~oop_ml.core.tree.node.DecisionNode` and
    :class:`~oop_ml.core.tree.node.LeafNode` objects, so that everything the
    :class:`~oop_ml.core.base.tree_model.TreeModel` frame reads off a root,
    the depth, the leaf count, the description and the importances, reads
    this tree without knowing where it came from.

    What a leaf holds is the one thing the two tasks disagree about, so the
    caller supplies ``leaf_of``, which is handed the engine's tree and a node
    id: a regression leaf reads the mean out of ``value``, a classification
    leaf reads the class shares.

    Each decision node's gain is recovered from the impurities the engine
    stored, as the parent's impurity less the row-weighted mean of its
    children's, which is the same quantity the numpy search scores a split by
    and the same one the engine's own ``feature_importances_`` totals. The
    rows are counted through :func:`node_row_count`, by weight, because that
    is the count the impurities were computed over. Rounding can leave the
    gain a hair below zero, and it is clamped, because a negative
    contribution is refused downstream.

    One convention differs and cannot be converted away. The engine sends a
    row **at** the threshold left and this library's
    :class:`~oop_ml.core.tree.split.Split` sends it right. Every threshold is
    a midpoint between two training values, so a training row never sits on
    one, and a query lands there only by coincidence. ``predict`` asks the
    engine, so the engine's rule is the one a prediction follows.
    """
    tree = engine.tree_

    def node_at(node_id: int) -> TreeNode:
        n_samples = node_row_count(tree, node_id)
        impurity = float(tree.impurity[node_id])

        if tree.children_left[node_id] == ENGINE_LEAF:
            return leaf_of(tree, node_id)

        left_id = int(tree.children_left[node_id])
        right_id = int(tree.children_right[node_id])
        rows_left = node_row_count(tree, left_id)
        rows_right = node_row_count(tree, right_id)

        gain = (
            impurity
            - (rows_left / n_samples) * float(tree.impurity[left_id])
            - (rows_right / n_samples) * float(tree.impurity[right_id])
        )
        feature_index = int(tree.feature[node_id])

        return DecisionNode(
            split=Split(
                feature_index,
                feature_names[feature_index],
                float(tree.threshold[node_id]),
                max(gain, 0.0),
            ),
            left=node_at(left_id),
            right=node_at(right_id),
            n_samples=n_samples,
            impurity=impurity,
        )

    return node_at(0)


def node_row_count(tree: Any, node_id: int) -> int:
    """How many training rows reached this node of the engine's tree, repeats counted.

    The engine keeps two counts per node. ``n_node_samples`` is how many
    *distinct* rows arrived and ``weighted_n_node_samples`` is their total
    weight, and on a plain fit the two agree. A bagging or forest engine does
    not fit a member on the resampled rows; it fits it on every row with a
    weight equal to how often the resample drew it, so a member that drew 40
    rows of which 27 were distinct reports ``n_node_samples`` of 27 at its
    root while its impurities were computed over the weighted 40. The
    weighted count is the one the impurities, the gain and the engine's own
    ``feature_importances_`` are written against, and it is what the numpy
    backend reports, since it fits on the resampled rows outright. Measured
    on five bagged members, reading the distinct count put their importances
    off the engine's by up to 0.029; the weighted count agrees to the last
    bit and puts 40 at the root.

    Every weight an engine here is handed is a draw count, so the total is a
    whole number and rounding recovers it exactly.
    """
    return int(round(float(tree.weighted_n_node_samples[node_id])))


def pairwise_callable(distance: Distance) -> Callable[[FloatArray, FloatArray], float]:
    """One of this library's distances as the two-row callable the engine takes.

    The engine calls a metric on two one-dimensional arrays at a time, where
    a :class:`~oop_ml.core.distance.calculations.Distance` pairs whole
    blocks. Each call wraps its two rows as one-row blocks and reads the one
    entry back. The names on the blocks are placeholders, since a distance
    reads values and never consults them; they exist because a block has to
    have some.

    Slow, one Python call per pair, and the engine is told to use brute force
    since no tree can index an arbitrary callable. It is the honest route for
    a metric the engine has no name for.
    """

    def between_rows(left: FloatArray, right: FloatArray) -> float:
        names = [f"feature_{position}" for position in range(left.size)]
        left_block = rows_of(left[None, :], names)
        right_block = rows_of(right[None, :], names)

        return float(distance.between(left_block, right_block)[0, 0])

    return between_rows


def neighbour_engine_parameters(
    n_neighbours: int, metric: DistanceMetric | Distance
) -> dict[str, Any]:
    """This library's neighbour configuration in the engine's keywords.

    ``n_neighbours`` is the engine's ``n_neighbors``. A
    :class:`~oop_ml.core.distance.metric.DistanceMetric` is translated by
    name through :data:`ENGINE_METRIC_NAMES`; a
    :class:`~oop_ml.core.distance.calculations.MinkowskiDistance` becomes the
    engine's ``minkowski`` with its order as ``p``; any other
    :class:`~oop_ml.core.distance.calculations.Distance` is handed over as a
    callable through :func:`pairwise_callable`, which works and is slow.

    Shared by the neighbour regressor and the neighbour classifier, whose
    engines take identical keywords for the search and differ only in what
    they do with the rows they find.
    """
    if isinstance(metric, DistanceMetric):
        return {"n_neighbors": n_neighbours, "metric": ENGINE_METRIC_NAMES[metric]}

    if isinstance(metric, MinkowskiDistance):
        return {"n_neighbors": n_neighbours, "metric": "minkowski", "p": metric.order}

    return {
        "n_neighbors": n_neighbours,
        "metric": pairwise_callable(metric),
        "algorithm": "brute",
    }


def engine_kernel_parameters(kernel: Kernel) -> dict[str, Any]:
    """This library's kernel in the engine's keywords.

    Each kernel class maps onto one of the engine's named kernels, and the
    parameter names differ on one of them: this library's ``constant`` is the
    engine's ``coef0``. The formulas agree exactly, ``(gamma a . b +
    coef0) ** degree`` for the polynomial, ``exp(-gamma ||a - b|| ** 2)`` for
    the radial basis, ``tanh(gamma a . b + coef0)`` for the sigmoid. The
    polynomial is named ``poly``, which both ``KernelRidge`` and ``SVC``
    accept, where only the first also answers to ``polynomial``.

    Raises
    ------
    InvalidValuesError
        If the kernel is a subclass this library did not ship. The engine
        would accept a callable, but a callable over raw arrays could not
        carry the feature-name check ``Kernel.between`` makes.
    """
    if isinstance(kernel, LinearKernel):
        return {"kernel": "linear"}

    if isinstance(kernel, PolynomialKernel):
        return {
            "kernel": "poly",
            "degree": kernel.degree,
            "gamma": kernel.gamma,
            "coef0": kernel.constant,
        }

    if isinstance(kernel, RadialBasisKernel):
        return {"kernel": "rbf", "gamma": kernel.gamma}

    if isinstance(kernel, SigmoidKernel):
        return {"kernel": "sigmoid", "gamma": kernel.gamma, "coef0": kernel.constant}

    raise InvalidValuesError(
        f"scikit-learn has no equivalent of {kernel!r}; the kernels it can be "
        "handed are LinearKernel, PolynomialKernel, RadialBasisKernel and "
        "SigmoidKernel"
    )
