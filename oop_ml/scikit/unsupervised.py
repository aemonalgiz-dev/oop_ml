"""The unsupervised family, with scikit-learn doing the arithmetic.

Every class here has a namesake in ``oop_ml.numpy`` with the same name, the
same pydantic fields, the same base class and the same learned properties, so
that a caller can swap one backend for the other at the import line and change
nothing else. A clusterer still answers with a
:class:`~oop_ml.core.clustering.clustering.Clustering`, a decomposition still
hands back components addressable by name, and a Boltzmann machine still
exposes its weights through
:class:`~oop_ml.numpy.generative.restricted_boltzmann_machine.BoltzmannParameters`.
What differs is who does the sums.

The vocabulary these wrappers answer in is the numpy backend's own value
objects. Two of them, ``BoltzmannParameters`` and the kernel components, live
under ``oop_ml.numpy`` rather than ``oop_ml.core`` because until now only one
model needed each; a second model needing them is the condition their own
docstrings named for moving them, and that move is a separate commit.

Hyperparameters keep this library's names and are translated at the boundary.
Every translation is written down on the wrapper that makes it, and two of
them carry a scale factor worth reading: the k-means tolerance, which the
engine measures relative to the data's variance where the numpy backend
measures it in the features' own units, and the Boltzmann machine's batch,
which is set to the whole training set so that one epoch is one update at the
stated rate on both sides.

Refusals are this library's, in this library's words, and the words are part
of what is being kept. Three of the calls here match a set of supplied names
against a set the fit produced, and only one of those three is asking about
features, so :class:`NameVocabulary` carries the noun through to the message
and the sentence a caller reads is the one the numpy backend writes. In the
other direction, ``KMeans.fit`` drops the one engine warning this library
already has its own words for; see
:func:`fit_dropping_the_distinct_cluster_warning`.

Where a numpy model exposes something the engine cannot supply, the member is
omitted here rather than stubbed or imitated, and the wrapper lists what it
leaves out under "Not mirrored from the numpy backend". Only the kernel
decomposition has such a list. The Boltzmann machine nearly had one, for
``tolerance``, ``converged`` and ``epochs_run``, until it turned out the
engine's ``partial_fit`` runs exactly one epoch and so lets the wrapper
measure every step; its fields are the numpy backend's fields.

Who holds an engine, and who does not
--------------------------------------
A wrapper keeps its fitted scikit-learn estimator only where a later call
needs it. k-means predicts by asking the engine, and both decompositions
project by asking it, so those three keep theirs. The Boltzmann machine does
not, since everything it answers after fitting is a function of its weights
and biases, and ``BoltzmannParameters`` already owns that arithmetic; keeping
an engine nothing would read again is the dead field the serving audit removed
elsewhere. Measured, that arithmetic and the engine's ``transform`` agree to
one ulp, and the free energies agree exactly.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from sklearn.cluster import KMeans as EngineKMeans
from sklearn.decomposition import PCA as EnginePCA
from sklearn.decomposition import KernelPCA as EngineKernelPCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from oop_ml.core.base.convergent_fit import ConvergentFit
from oop_ml.core.base.estimator import Clusterer, Transformer
from oop_ml.core.clustering.centroids import Centroid, Centroids
from oop_ml.core.clustering.clustering import Clustering
from oop_ml.core.data.coefficients import Coefficient, Coefficients
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import ClassScores
from oop_ml.core.decomposition.components import (
    PrincipalComponent,
    PrincipalComponents,
)
from oop_ml.core.exceptions import InvalidValuesError, TooFewValuesError
from oop_ml.core.kernel.functions import Kernel, LinearKernel
from oop_ml.core.schedule import ConstantSchedule, Schedule
from oop_ml.core.types import FloatArray
from oop_ml.numpy.clustering.k_means import CLUSTER_NAME_PREFIX
from oop_ml.numpy.decomposition.kernel_principal_component_analysis import (
    KERNEL_COMPONENT_NAME_PREFIX,
    MINIMUM_COMPONENT_VARIANCE,
    KernelComponent,
    KernelComponents,
)
from oop_ml.numpy.decomposition.principal_component_analysis import (
    COMPONENT_NAME_PREFIX,
)
from oop_ml.numpy.generative.restricted_boltzmann_machine import (
    HIDDEN_UNIT_NAME_PREFIX,
    BoltzmannParameters,
)
from oop_ml.scikit.plumbing import (
    engine_kernel_parameters,
    matched_matrix,
    matrix_of,
)

MINIMUM_DECOMPOSITION_ROWS = 2
"""A single row has no spread to find a direction in."""

ENGINE_INITIAL_WEIGHT_SPREAD = 0.01
"""The standard deviation ``BernoulliRBM`` starts its weights from.

The engine's own figure, restated because the engine draws it inside ``fit``
and ``partial_fit`` and the Boltzmann wrapper has to draw it before either
runs. It happens to equal the numpy backend's ``INITIAL_WEIGHT_SPREAD``; it is
named for the engine because that is whose number it is.
"""


class NameVocabulary(StrEnum):
    """Which set of names a match is asking for, as it should appear in an error.

    Three different questions run through one comparison here, and only one of
    them is about features. A decomposition rebuilding its inputs wants its own
    components, and a Boltzmann machine running backwards wants its own hidden
    units, and neither of those is a column anybody trained on. A closed enum
    rather than a string, for the reason the rest of the library uses enums,
    since the wrong noun is then a name that does not exist rather than a
    sentence that sends a reader looking for columns the fit never saw.
    """

    FITTED_FEATURES = "the fitted features"
    COMPONENTS = "this model's components"
    HIDDEN_UNITS = "this model's hidden units"


def check_names_match(
    expected: Sequence[str], supplied: Sequence[Feature], vocabulary: NameVocabulary
) -> None:
    """Raise unless ``supplied`` names exactly the ``expected`` ones.

    Exactly, in both directions, and in any order. A missing column leaves a
    centre, a component or a visible layer unevaluable; an extra one has no
    coordinate, loading or unit to be read against. Every model here matches
    by name, so the order is free.

    The numpy backend keeps three refusals for the three questions
    :class:`NameVocabulary` names; this keeps one comparison and hands it the
    noun, so the sentence a reader meets is the same one on both backends.

    Parameters
    ----------
    expected:
        The names the caller must supply, in any order.
    supplied:
        What the caller actually supplied.
    vocabulary:
        What ``expected`` is, which is what the message calls it.

    Raises
    ------
    InvalidValuesError
        If the two sets of names differ.
    """
    supplied_names = {feature.name for feature in supplied}
    expected_names = set(expected)

    if supplied_names != expected_names:
        raise InvalidValuesError(
            f"expected exactly {vocabulary} {sorted(expected_names)}; "
            f"got {sorted(supplied_names)}"
        )


def fit_dropping_the_distinct_cluster_warning(engine: Any, matrix: FloatArray) -> None:
    """Fit the engine, keeping its empty-group warning out of the caller's log.

    ``KMeans.fit`` warns, in the engine's own vocabulary, when the rows hold
    fewer distinct points than the caller asked for clusters. The numpy
    backend says nothing on the same data, because leaving a group's centre
    where it is and reporting an empty group is what this library does with
    one; ``clustering.has_an_empty_cluster`` and ``clustering.sizes`` are
    where it says so, and both backends answer them identically. Measured on
    six rows holding two distinct points at ``n_clusters=4``, both report
    sizes ``[0, 0, 3, 3]``, an empty group, and an inertia of 0.0, so the
    warning adds nothing this library has not already said in its own words.

    Only that one warning is dropped, and by category rather than by its text.
    Read against scikit-learn 1.9, ``KMeans.fit`` raises exactly one
    ``ConvergenceWarning``, the distinct-cluster one, and it does not warn on
    reaching ``max_iter``, so nothing about a walk that ran out is being
    silenced. Every other warning the engine issues is re-raised as it was,
    which is the rule
    :func:`~oop_ml.scikit.plumbing.fit_watching_convergence` already keeps.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.fit(matrix)

    for warning in caught:
        if not issubclass(warning.category, ConvergenceWarning):
            warnings.warn(warning.message, warning.category, stacklevel=3)


def engine_tolerance(matrix: FloatArray, tolerance: float) -> float:
    """This library's absolute k-means tolerance in the engine's relative units.

    The numpy backend stops when no centre's squared movement exceeds
    ``tolerance``, in the features' own units. The engine multiplies its
    ``tol`` by the mean per-feature variance of the data before comparing,
    so the same number means a different distance on every dataset. Dividing
    by that variance here hands the engine a threshold that lands back on the
    stated one, and the field keeps its meaning across the two backends.

    Data with no variance at all leaves nothing to divide by. The engine's
    threshold is then zero whatever it is told, since it multiplies by the
    zero variance, so the tolerance is passed through unchanged.
    """
    mean_variance = float(np.mean(np.var(matrix, axis=0)))

    if mean_variance <= 0.0:
        return tolerance

    return tolerance / mean_variance


class KMeans(Clusterer[Sequence[Feature]]):
    """Round clusters by Lloyd's algorithm, by ``KMeans``.

    Parameters
    ----------
    n_clusters, n_initialisations, max_iterations, tolerance, random_seed:
        As on the numpy backend.

    Translation
    -----------
    ``n_initialisations`` is the engine's ``n_init``, ``max_iterations`` its
    ``max_iter`` and ``random_seed`` its ``random_state``. The engine's
    ``init`` is left at ``k-means++``, which is the seeding the numpy backend
    implements, and its ``algorithm`` at plain Lloyd.

    ``tolerance`` is the engine's ``tol`` after :func:`engine_tolerance`,
    because the engine measures its threshold relative to the data's variance
    and this library measures it in the features' own units. One difference
    survives the translation: the engine compares the *sum* over centres of
    the squared movement where the numpy backend compares the *largest*, so
    the engine's stop is at least as strict and can run a pass longer.

    The fit runs through
    :func:`fit_dropping_the_distinct_cluster_warning`, so a caller who asks
    for more clusters than the rows hold distinct points meets this library's
    report of that and not the engine's. Neither backend refuses such a fit;
    ``clustering.has_an_empty_cluster`` and ``clustering.sizes`` are where it
    is said, and they say it identically on both.

    Where the backends disagree
    ---------------------------
    Both keep the restart with the lowest inertia, but the two draw their
    seedings from different random streams, so the same ``random_seed`` names
    a different restart on each side. On separated data every restart finds
    the same partition and the disagreement is invisible; on ambiguous data
    the two backends can settle in different local minima.

    ``iterations_run`` counts the engine's Lloyd passes for the winning
    restart, which is the same quantity to within the one pass the stricter
    stopping rule above can add.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_clusters: int = Field(default=8, ge=1)
    n_initialisations: int = Field(default=10, ge=1)
    max_iterations: int = Field(default=300, ge=1)
    tolerance: float = Field(default=1e-08, gt=0.0)
    random_seed: int | None = None

    _engine: EngineKMeans | None = PrivateAttr(default=None)
    _clustering: Clustering | None = PrivateAttr(default=None)
    _feature_names: tuple[str, ...] = PrivateAttr(default=())
    _iterations_run: int | None = PrivateAttr(default=None)

    @property
    def clustering(self) -> Clustering:
        """The grouping this fit settled on: labels, centres, and inertia.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._clustering is not None
        return self._clustering

    @property
    def centroids(self) -> Centroids:
        """Where the learned groups sit.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.clustering.centroids

    @property
    def inertia(self) -> float:
        """The objective the best initialisation reached.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.clustering.inertia

    @property
    def iterations_run(self) -> int:
        """Assign/update passes taken by the initialisation that won.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._iterations_run is not None
        return self._iterations_run

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Find ``n_clusters`` groups in ``input_values``.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonEqualArrayLengthError
            If the features are different lengths.
        NonUniqueFeaturesError
            If two features share a name.
        TooFewValuesError
            If there are fewer rows than clusters, since a group would then
            have to be empty.
        """
        feature_set = FeatureSet(input_values)

        if feature_set.n_samples < self.n_clusters:
            raise TooFewValuesError(
                f"cannot find {self.n_clusters} groups in {feature_set.n_samples} rows"
            )

        feature_names = tuple(feature.name for feature in feature_set)
        matrix = matrix_of(feature_set)

        # scikit-learn is untyped, and pyright reads each unannotated default
        # as the parameter's type, so ``n_init="auto"`` would refuse an int.
        engine_type: Any = EngineKMeans
        engine = engine_type(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=self.n_initialisations,
            max_iter=self.max_iterations,
            tol=engine_tolerance(matrix, self.tolerance),
            random_state=self.random_seed,
            algorithm="lloyd",
        )
        fit_dropping_the_distinct_cluster_warning(engine, matrix)

        centroids = Centroids(
            [
                Centroid(self.name_for(position), row, feature_names)
                for position, row in enumerate(
                    np.asarray(engine.cluster_centers_, dtype=np.float64)
                )
            ]
        )
        clustering = Clustering(
            np.asarray(engine.labels_), centroids, float(engine.inertia_)
        )

        self._engine = engine
        self._clustering = clustering
        self._feature_names = feature_names
        self._iterations_run = int(engine.n_iter_)
        self._mark_fitted()

        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """Label each row with the group whose centre it is nearest.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        self._check_fitted()
        assert self._engine is not None
        check_names_match(
            self._feature_names, input_values, NameVocabulary.FITTED_FEATURES
        )

        labels = self._engine.predict(matched_matrix(self._feature_names, input_values))

        return Predictions.already_checked(np.asarray(labels).astype(np.float64))

    @staticmethod
    def name_for(position: int) -> str:
        """The name of the group at ``position``, counting from zero."""
        return f"{CLUSTER_NAME_PREFIX}_{position + 1}"

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"KMeans(n_clusters={self.n_clusters}, unfitted)"

        return (
            f"KMeans(n_clusters={self.n_clusters}, inertia={self.inertia:.4f}, "
            f"iterations_run={self.iterations_run})"
        )


class PrincipalComponentAnalysis(Transformer[Sequence[Feature]]):
    """The directions of greatest variance, by ``PCA``.

    Parameters
    ----------
    n_components, standardize:
        As on the numpy backend.

    Translation
    -----------
    ``n_components`` is the engine's ``n_components`` unchanged, ``None``
    meaning every direction on both sides. The engine's ``svd_solver`` is
    pinned to ``full`` so that the decomposition is exact and deterministic
    rather than randomised above a size threshold.

    ``standardize`` has no engine parameter. It is a ``StandardScaler``
    fitted first, which divides by the population standard deviation exactly
    as the numpy backend's ``Standardizer`` does, so the standardised columns
    agree to rounding and the decomposition that follows agrees with them.

    The engine reports each kept direction's variance. The total those are
    shared against is the sum over *every* direction, kept and discarded, and
    it is read here as the trace of the sample covariance, the sum of each
    column's variance over ``n_rows - 1``, which is the same number the engine
    divides by when it computes ``explained_variance_ratio_`` and is defined
    where recovering it from the ratios is not: on data with no spread at all
    every ratio is ``0 / 0``, and the refusal below lands before the engine
    can raise a warning about it.

    Where the backends disagree
    ---------------------------
    With fewer rows than features the engine keeps ``min(n_rows, n_features)``
    directions where the numpy backend keeps one per feature, the surplus
    carrying zero variance. Asking for more components than that is accepted
    by the numpy backend and refused here as ``InvalidValuesError``, since the
    engine cannot report directions it did not compute. The first version of
    this wrapper let the engine's own ``ValueError`` through on that call.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_components: int | None = Field(default=None, ge=1)
    standardize: bool = False

    _engine: EnginePCA | None = PrivateAttr(default=None)
    _scaler: StandardScaler | None = PrivateAttr(default=None)
    _components: PrincipalComponents | None = PrivateAttr(default=None)
    _feature_names: tuple[str, ...] = PrivateAttr(default=())

    @property
    def components(self) -> PrincipalComponents:
        """The directions this fit found, most variance first.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._components is not None
        return self._components

    @property
    def n_features_in(self) -> int:
        """How many features the fit saw.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.components.n_features

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Learn the directions of greatest variance in ``input_values``.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonEqualArrayLengthError
            If the features are different lengths.
        NonUniqueFeaturesError
            If two features share a name.
        TooFewValuesError
            If there are fewer than two rows.
        AllSameValuesError
            If ``standardize`` is set and some feature is constant, since there
            is then no spread to divide by.
        InvalidValuesError
            If ``n_components`` exceeds the number of features supplied, or
            the number of rows, which is the engine's own ceiling; or if no
            feature varies at all, so there is no variance to share out.
        """
        feature_set = FeatureSet(input_values)

        if feature_set.n_samples < MINIMUM_DECOMPOSITION_ROWS:
            raise TooFewValuesError(
                f"a decomposition needs at least two rows to have any spread; "
                f"got {feature_set.n_samples}"
            )

        if self.n_components is not None and self.n_components > feature_set.n_features:
            raise InvalidValuesError(
                f"cannot keep {self.n_components} components from "
                f"{feature_set.n_features} features"
            )

        if self.n_components is not None and self.n_components > feature_set.n_samples:
            raise InvalidValuesError(
                f"cannot keep {self.n_components} components from "
                f"{feature_set.n_samples} rows; the engine computes at most one "
                f"direction per row, where the numpy backend would pad the "
                f"surplus with directions of zero variance"
            )

        feature_names = tuple(feature.name for feature in feature_set)
        matrix = matrix_of(feature_set)

        scaler: StandardScaler | None = None
        if self.standardize:
            feature_set.check_columns_vary()
            scaler = StandardScaler().fit(matrix)
            matrix = np.asarray(scaler.transform(matrix), dtype=np.float64)

        total_variance = float(np.sum(np.var(matrix, axis=0, ddof=1)))
        if total_variance <= 0.0:
            raise InvalidValuesError(
                f"total variance must be positive; got {total_variance}"
            )

        engine = EnginePCA(n_components=self.n_components, svd_solver="full")
        engine.fit(matrix)

        components = self._read_components(engine, feature_names, total_variance)

        self._engine = engine
        self._scaler = scaler
        self._components = components
        self._feature_names = feature_names
        self._mark_fitted()

        return self

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Project rows onto the learned components.

        One output feature per kept component, named ``component_1`` upward.
        The rows are centred, and scaled if the fit was, by what the fit
        learned and never by their own statistics.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        self._check_fitted()
        assert self._engine is not None
        check_names_match(
            self._feature_names, input_values, NameVocabulary.FITTED_FEATURES
        )

        matrix = self._prepared(matched_matrix(self._feature_names, input_values))
        coordinates = np.asarray(self._engine.transform(matrix), dtype=np.float64)

        return [
            Feature(component.name, coordinates[:, position])
            for position, component in enumerate(self.components)
        ]

    def inverse_transform(self, transformed_values: Sequence[Feature]) -> list[Feature]:
        """Rebuild the original features from their component coordinates.

        Lossless only when every component was kept; otherwise the data's
        shadow on the kept subspace, exactly as on the numpy backend.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly this model's components.
        """
        self._check_fitted()
        assert self._engine is not None
        check_names_match(
            self.component_names(), transformed_values, NameVocabulary.COMPONENTS
        )

        by_name = {feature.name: feature for feature in transformed_values}
        coordinates = np.column_stack(
            [by_name[component.name].values for component in self.components]
        )
        rebuilt = np.asarray(
            self._engine.inverse_transform(coordinates), dtype=np.float64
        )

        if self._scaler is not None:
            rebuilt = np.asarray(
                self._scaler.inverse_transform(rebuilt), dtype=np.float64
            )

        return [
            Feature(name, rebuilt[:, position])
            for position, name in enumerate(self._feature_names)
        ]

    def _read_components(
        self, engine: Any, feature_names: Sequence[str], total_variance: float
    ) -> PrincipalComponents:
        """The engine's directions and variances as this library's components.

        ``total_variance`` is the trace of the sample covariance the caller
        already has in hand, which is the sum over every direction the engine
        could have kept; the components' shares are reported against it.
        """
        directions = np.asarray(engine.components_, dtype=np.float64)
        variances = np.asarray(engine.explained_variance_, dtype=np.float64)

        return PrincipalComponents(
            [
                self.component_from(
                    position,
                    directions[position],
                    float(variances[position]),
                    feature_names,
                )
                for position in range(directions.shape[0])
            ],
            total_variance,
        )

    def _prepared(self, matrix: FloatArray) -> FloatArray:
        """The rows scaled as the fit scaled them, or untouched if it did not."""
        if self._scaler is None:
            return matrix

        return np.asarray(self._scaler.transform(matrix), dtype=np.float64)

    def component_names(self) -> tuple[str, ...]:
        """What ``transform`` will call its output features.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        return tuple(component.name for component in self.components)

    def component_from(
        self,
        position: int,
        direction: FloatArray,
        variance: float,
        feature_names: Sequence[str],
    ) -> PrincipalComponent:
        """Pair one direction vector with the names its entries weight."""
        return PrincipalComponent(
            self.name_for(position),
            Coefficients(
                [
                    Coefficient(name, float(weight))
                    for name, weight in zip(feature_names, direction, strict=True)
                ]
            ),
            variance,
        )

    @staticmethod
    def name_for(position: int) -> str:
        """The name of the component at ``position``, counting from zero."""
        return f"{COMPONENT_NAME_PREFIX}_{position + 1}"

    def __repr__(self) -> str:
        if not self.is_fitted:
            return "PrincipalComponentAnalysis(unfitted)"

        return (
            f"PrincipalComponentAnalysis("
            f"n_components={self.components.n_components}, "
            f"explained={self.components.cumulative_shares[-1]:.4f})"
        )


class KernelPrincipalComponentAnalysis(Transformer[Sequence[Feature]]):
    """PCA through the Gram matrix, by ``KernelPCA``.

    Parameters
    ----------
    kernel, n_components:
        As on the numpy backend.

    Translation
    -----------
    ``kernel`` is one of this library's
    :class:`~oop_ml.core.kernel.functions.Kernel` objects and is translated by
    :func:`~oop_ml.scikit.plumbing.engine_kernel_parameters` into the engine's
    kernel name and parameters; ``constant`` becomes ``coef0`` and everything
    else keeps its name. The engine's ``eigen_solver`` is pinned to ``dense``,
    so the decomposition is exact and deterministic rather than an iterative
    solver above a size threshold.

    ``n_components`` is **not** handed to the engine. Told a count, the engine
    computes only that many eigenvalues, and the total variance is the sum
    over every one of them, kept and discarded, which is the denominator the
    shares are reported against. So the engine is always asked for the whole
    spectrum, which is what the numpy backend's ``eigh`` computes too, and the
    truncation happens here on the way into the components. The first
    attempt read the total off the trace of the centred Gram matrix instead,
    and on a sigmoid kernel that put the kept variance *above* the total: the
    trace carries the negative eigenvalues an indefinite matrix has, where
    both backends clamp them at zero before summing.

    The engine reports raw eigenvalues of the centred Gram matrix, descending
    and clamped, and unit eigenvectors. Each is read back the way the numpy
    backend reads its own: the variance is the eigenvalue over ``n_rows - 1``,
    and the coefficients are the eigenvector over the square root of the raw
    eigenvalue, which is what makes the implied direction a unit vector. A
    direction whose raw eigenvalue falls below ``MINIMUM_COMPONENT_VARIANCE``
    is dropped for the reason the numpy backend drops it.

    Where the backends disagree
    ---------------------------
    A kernel that fails Mercer's condition makes the centred Gram matrix
    indefinite. The numpy backend clamps every negative eigenvalue at zero
    and fits; the engine clamps the small ones and refuses significantly
    negative ones outright, and that refusal is re-raised as
    :class:`~oop_ml.core.exceptions.InvalidValuesError`.

    Not mirrored from the numpy backend
    -----------------------------------
    ``query_matrix``
        Returns the centred query kernel matrix. The engine builds the same
        matrix inside ``transform`` and keeps its centerer on a private
        attribute, so exposing it would mean either reaching into the engine
        or recomputing it with this library's kernels.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    kernel: Kernel = LinearKernel()
    n_components: int | None = Field(default=None, ge=1)

    _engine: EngineKernelPCA | None = PrivateAttr(default=None)
    _components: KernelComponents | None = PrivateAttr(default=None)
    _feature_names: tuple[str, ...] = PrivateAttr(default=())

    @property
    def components(self) -> KernelComponents:
        """The directions this fit found, most variance first.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._components is not None
        return self._components

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Decompose the centred Gram matrix of ``input_values``.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        TooFewValuesError
            If there are fewer than two rows.
        InvalidValuesError
            If ``n_components`` exceeds the number of rows, if the kernel is one
            the engine cannot be handed, or if the Gram matrix is indefinite.
        """
        feature_set = FeatureSet(input_values)

        if feature_set.n_samples < MINIMUM_DECOMPOSITION_ROWS:
            raise TooFewValuesError(
                f"a decomposition needs at least two rows; got {feature_set.n_samples}"
            )

        if self.n_components is not None and self.n_components > feature_set.n_samples:
            raise InvalidValuesError(
                f"cannot keep {self.n_components} components from "
                f"{feature_set.n_samples} rows"
            )

        feature_names = tuple(feature.name for feature in feature_set)
        matrix = matrix_of(feature_set)

        # The untyped engine reads ``gamma=None`` as the parameter's type.
        engine_type: Any = EngineKernelPCA
        engine = engine_type(
            n_components=None,
            eigen_solver="dense",
            **engine_kernel_parameters(self.kernel),
        )

        try:
            engine.fit(matrix)
        except ValueError as refusal:
            raise InvalidValuesError(
                f"the engine refused the Gram matrix of {self.kernel!r}: {refusal}"
            ) from refusal

        components = self._read_components(engine)

        self._engine = engine
        self._components = components
        self._feature_names = feature_names
        self._mark_fitted()

        return self

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Project rows onto the learned directions.

        One output feature per kept component, named ``kernel_component_1``
        upward. The engine centres the query rows against the training mean,
        which is the step the numpy backend's ``centred_against`` exists for.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        self._check_fitted()
        assert self._engine is not None
        check_names_match(
            self._feature_names, input_values, NameVocabulary.FITTED_FEATURES
        )

        coordinates = np.asarray(
            self._engine.transform(matched_matrix(self._feature_names, input_values)),
            dtype=np.float64,
        )

        return [
            Feature(component.name, coordinates[:, position])
            for position, component in enumerate(self.components)
        ]

    def _read_components(self, engine: Any) -> KernelComponents:
        """The engine's eigenpairs as this library's kernel components.

        The engine sorts descending and clamps rounding-sized negatives at
        zero, so the total is the sum of what it reports, and the walk down
        its eigenvalues stops at ``n_components`` or at the first one below
        the minimum, exactly as the numpy backend's does.
        """
        raw = np.asarray(engine.eigenvalues_, dtype=np.float64)
        directions = np.asarray(engine.eigenvectors_, dtype=np.float64)
        n_training_rows = directions.shape[0]

        total_variance = float(np.sum(raw) / (n_training_rows - 1))
        kept = (
            raw.size if self.n_components is None else min(self.n_components, raw.size)
        )

        components: list[KernelComponent] = []

        for position in range(kept):
            if raw[position] < MINIMUM_COMPONENT_VARIANCE:
                break

            components.append(
                KernelComponent(
                    self.name_for(position),
                    directions[:, position] / np.sqrt(raw[position]),
                    float(raw[position] / (n_training_rows - 1)),
                )
            )

        return KernelComponents(components, total_variance)

    @staticmethod
    def name_for(position: int) -> str:
        """The name of the component at ``position``, counting from zero."""
        return f"{KERNEL_COMPONENT_NAME_PREFIX}_{position + 1}"

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"KernelPrincipalComponentAnalysis({self.kernel!r}, unfitted)"

        return (
            f"KernelPrincipalComponentAnalysis({self.kernel!r}, "
            f"n_components={self.components.n_components})"
        )


class RestrictedBoltzmannMachine(Transformer[Sequence[Feature]], ConvergentFit):
    """A generative model of binary rows, by ``BernoulliRBM``.

    Parameters
    ----------
    n_hidden_units, learning_rate, n_gibbs_steps, max_epochs, random_seed,
    tolerance:
        As on the numpy backend, with one value the engine cannot honour
        refused at construction; see Raises.

    Translation
    -----------
    ``n_hidden_units`` is the engine's ``n_components`` and ``random_seed``
    its ``random_state``.

    The engine is not asked to run the whole walk. Its ``fit`` runs
    ``n_iter`` epochs at one rate and measures nothing on the way, which
    would leave ``tolerance`` with no threshold to set, ``converged`` with
    nothing to report, and a decaying schedule with no epoch to decay over.
    Its ``partial_fit`` runs exactly one update, so the wrapper drives the
    walk itself: one ``partial_fit`` per epoch, the engine's ``learning_rate``
    set to what the schedule says for that epoch first, and the largest
    movement of any parameter read off afterwards and compared against
    ``tolerance`` exactly as the numpy backend compares its own. Measured,
    ``max_epochs`` calls of ``partial_fit`` reproduce one call of ``fit`` bit
    for bit, so nothing is lost by taking the walk one step at a time.

    The engine's ``batch_size`` has no field here and is set to the number of
    training rows. The engine scales its rate by the batch size and sums the
    batch's statistics, so a batch of every row is one update per epoch of
    ``rate * (<v h>_data - <v h>_model)``, which is the numpy backend's rule
    at the numpy backend's cadence. At the engine's default of ten, an epoch
    would be a dozen full-rate steps and ``max_epochs`` would mean something
    else.

    An epoch whose rate is zero is not handed to the engine, which refuses a
    rate of zero outright. Nothing would have moved, so the epoch is recorded
    as a movement of zero, which is what the numpy backend measures on the
    same epoch and is why both settle on their first epoch at a zero rate.

    Where the backends disagree
    ---------------------------
    The negative statistic. The numpy backend runs contrastive divergence,
    restarting the Gibbs chain at the data every epoch. The engine runs
    persistent contrastive divergence, carrying the chain's hidden states
    from one epoch into the next and starting them at zero. Both are biased
    approximations to the same gradient, and on the spec's two-pattern data
    both learn the patterns, but the weights they arrive at are not the same
    weights and no test asks them to be. A zero-rate epoch is the one place
    the persistence shows: the engine's chain does not advance through an
    epoch it never ran.

    Raises
    ------
    InvalidValuesError
        If ``n_gibbs_steps`` is not one, since the engine runs exactly one
        Gibbs step per update. The default passes, so a search rebuilding
        candidates field-by-field is unaffected.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_hidden_units: int = Field(default=8, ge=1)
    learning_rate: Schedule = ConstantSchedule(value=0.1)
    n_gibbs_steps: int = Field(default=1, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    random_seed: int | None = None

    _parameters: BoltzmannParameters | None = PrivateAttr(default=None)
    _feature_names: tuple[str, ...] = PrivateAttr(default=())
    _generator: np.random.Generator | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _refuse_a_longer_chain(self) -> Self:
        """Raise on a Gibbs chain longer than the one step the engine runs."""
        if self.n_gibbs_steps != 1:
            raise InvalidValuesError(
                f"the engine runs one Gibbs step per update; got n_gibbs_steps="
                f"{self.n_gibbs_steps}"
            )

        return self

    @property
    def _pass_limit(self) -> int:
        """``max_epochs``, under the name this model gives it."""
        return self.max_epochs

    @property
    def epochs_run(self) -> int:
        """How many full passes over the data the fit took.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self._completed_passes

    @property
    def parameters(self) -> BoltzmannParameters:
        """The weights and biases this fit settled on.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._parameters is not None
        return self._parameters

    @property
    def weights(self) -> FloatArray:
        """``(n_visible_units, n_hidden_units)``, read-only.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.parameters.weights

    @property
    def visible_bias(self) -> FloatArray:
        """``(n_visible_units,)``, read-only.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.parameters.visible_bias

    @property
    def hidden_bias(self) -> FloatArray:
        """``(n_hidden_units,)``, read-only.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.parameters.hidden_bias

    @property
    def n_visible_units(self) -> int:
        """How many features the fit saw, which is the visible layer's width.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.parameters.n_visible_units

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The visible units' names, in the order the fit saw them.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        return self._feature_names

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Learn weights and biases from ``input_values``.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonEqualArrayLengthError
            If the features are different lengths.
        NonUniqueFeaturesError
            If two features share a name.
        InvalidValuesError
            If any value falls outside ``[0, 1]``.
        DivergenceError
            If the engine's weights or biases left the finite numbers, which
            :class:`BoltzmannParameters` refuses on the way in.
        """
        feature_set = FeatureSet(input_values)
        self._check_values_are_bounded(feature_set)

        feature_names = tuple(feature.name for feature in feature_set)
        matrix = matrix_of(feature_set)

        engine = self._started_engine(matrix.shape[0], matrix.shape[1])
        parameters = self._parameters_of(engine)
        epochs_run = 0
        converged = False

        for epoch in range(1, self.max_epochs + 1):
            rate = self.learning_rate.value_at(epoch, self.max_epochs)
            if rate > 0.0:
                engine.set_params(learning_rate=rate)
                engine.partial_fit(matrix)
            moved = self._parameters_of(engine)
            movement = self._largest_movement(parameters, moved)
            parameters = moved
            epochs_run = epoch

            if self._has_converged(movement):
                converged = True
                break

        self._parameters = parameters
        self._feature_names = feature_names
        self._generator = np.random.default_rng(self.random_seed)
        self._record_walk(epochs_run, converged)
        self._mark_fitted()

        return self

    def _started_engine(self, n_rows: int, n_visible_units: int) -> BernoulliRBM:
        """An engine holding its initial parameters, ready for ``partial_fit``.

        ``partial_fit`` keeps whatever ``components_``, intercepts,
        ``h_samples_`` and ``random_state_`` an engine already holds and
        creates the ones it lacks, which is the contract that lets a walk be
        resumed. It is used here to start one: the initial state is put in
        place before the first epoch exactly as the engine's ``fit`` would
        put it, small normal weights from the seeded stream, zero biases and
        a zero persistent chain, and the same stream is handed over for the
        sampling that follows. Measured, the walk that results is bit for bit
        the one ``fit`` would have run.

        Doing it here rather than letting the first ``partial_fit`` do it is
        what makes the first epoch's movement measurable, since the engine
        otherwise has no state to measure from, and what gives a walk whose
        rate is zero from the start, which never calls the engine at all,
        the weights the engine would have started from.
        """
        engine = BernoulliRBM(
            n_components=self.n_hidden_units,
            batch_size=n_rows,
            n_iter=self.max_epochs,
            random_state=self.random_seed,
        )
        stream = check_random_state(self.random_seed)

        engine.random_state_ = stream
        engine.components_ = np.asarray(
            stream.normal(
                0, ENGINE_INITIAL_WEIGHT_SPREAD, (self.n_hidden_units, n_visible_units)
            ),
            order="F",
        )
        engine.intercept_hidden_ = np.zeros(self.n_hidden_units)
        engine.intercept_visible_ = np.zeros(n_visible_units)
        engine.h_samples_ = np.zeros((n_rows, self.n_hidden_units))

        return engine

    @staticmethod
    def _parameters_of(engine: Any) -> BoltzmannParameters:
        """The engine's weights and biases as this library's parameters.

        Raises
        ------
        DivergenceError
            If any of them has left the finite numbers.
        """
        return BoltzmannParameters(
            np.asarray(engine.components_, dtype=np.float64).T,
            np.asarray(engine.intercept_visible_, dtype=np.float64),
            np.asarray(engine.intercept_hidden_, dtype=np.float64),
        )

    @staticmethod
    def _largest_movement(
        before: BoltzmannParameters, after: BoltzmannParameters
    ) -> float:
        """The biggest single change one epoch made, in the parameters' own units.

        The quantity the numpy backend's ``ContrastiveDivergenceUpdate``
        reports as ``largest_movement``, read here as a difference because the
        engine applies its update in place and hands back only the result.
        """
        return float(
            max(
                np.max(np.abs(after.weights - before.weights)),
                np.max(np.abs(after.visible_bias - before.visible_bias)),
                np.max(np.abs(after.hidden_bias - before.hidden_bias)),
            )
        )

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Rewrite rows as the hidden layer's probabilities.

        One output feature per hidden unit, named ``hidden_1`` upward, and
        deterministic for the reason the numpy backend gives.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or any
            value falls outside ``[0, 1]``.
        """
        probabilities = self.hidden_probabilities(input_values)

        return [
            Feature(self.name_for(position), probabilities.values[:, position])
            for position in range(probabilities.n_classes)
        ]

    def hidden_probabilities(self, input_values: Sequence[Feature]) -> ClassScores:
        """``probability(hidden = 1 | row)`` for every row and every hidden unit.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or any
            value falls outside ``[0, 1]``.
        """
        return ClassScores(
            self.parameters.hidden_given(self._visible_rows(input_values))
        )

    def visible_probabilities(self, hidden_values: Sequence[Feature]) -> ClassScores:
        """``probability(visible = 1 | hidden)`` for every row and visible unit.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly this model's hidden units,
            or any value falls outside ``[0, 1]``.
        """
        return ClassScores(
            self.parameters.visible_given(self._hidden_rows(hidden_values))
        )

    def sample_hidden(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Draw one binary hidden state per unit per row.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or any
            value falls outside ``[0, 1]``.
        """
        drawn = self._sampled(self.hidden_probabilities(input_values).values)

        return [
            Feature(self.name_for(position), drawn[:, position])
            for position in range(drawn.shape[1])
        ]

    def sample_visible(self, hidden_values: Sequence[Feature]) -> list[Feature]:
        """Draw one binary visible state per unit per row.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly this model's hidden units,
            or any value falls outside ``[0, 1]``.
        """
        drawn = self._sampled(self.visible_probabilities(hidden_values).values)

        return [
            Feature(name, drawn[:, position])
            for position, name in enumerate(self.feature_names)
        ]

    def reconstruct(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Push rows through the hidden layer and back, and report what returns.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or any
            value falls outside ``[0, 1]``.
        """
        rows = self._visible_rows(input_values)
        rebuilt = self.parameters.visible_given(self.parameters.hidden_given(rows))

        return [
            Feature(name, rebuilt[:, position])
            for position, name in enumerate(self._feature_names)
        ]

    def reconstruction_error(self, input_values: Sequence[Feature]) -> float:
        """Mean squared difference between the rows and their reconstruction.

        A diagnostic and not the objective, as on the numpy backend.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or any
            value falls outside ``[0, 1]``.
        """
        rows = self._visible_rows(input_values)
        rebuilt = np.column_stack(
            [feature.values for feature in self.reconstruct(input_values)]
        )

        return float(np.mean((rows - rebuilt) ** 2))

    def free_energy(self, input_values: Sequence[Feature]) -> Predictions:
        """The free energy of each row, which is what compares configurations.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or any
            value falls outside ``[0, 1]``.
        """
        return Predictions.already_checked(
            self.parameters.free_energy_of(self._visible_rows(input_values))
        )

    def hidden_unit_names(self) -> tuple[str, ...]:
        """What ``transform`` will call its output features.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        self._check_fitted()

        return tuple(self.name_for(position) for position in range(self.n_hidden_units))

    def _visible_rows(self, input_values: Sequence[Feature]) -> FloatArray:
        """The supplied features as a checked matrix, in the fitted order."""
        self._check_fitted()
        check_names_match(
            self._feature_names, input_values, NameVocabulary.FITTED_FEATURES
        )

        feature_set = FeatureSet.matching(self._feature_names, list(input_values))
        self._check_values_are_bounded(feature_set)

        return matrix_of(feature_set)

    def _hidden_rows(self, hidden_values: Sequence[Feature]) -> FloatArray:
        """The supplied hidden units as a checked matrix, in unit order."""
        names = self.hidden_unit_names()
        check_names_match(names, hidden_values, NameVocabulary.HIDDEN_UNITS)

        feature_set = FeatureSet.matching(names, list(hidden_values))
        self._check_values_are_bounded(feature_set)

        return matrix_of(feature_set)

    def _sampled(self, probabilities: FloatArray) -> FloatArray:
        """One Bernoulli draw per entry, as 0.0 or 1.0, from the model's generator.

        The generator is created lazily so that a model which has not drawn
        yet still draws, and it is seeded from ``random_seed`` so that a fit
        with the same seed reproduces the whole sequence.
        """
        if self._generator is None:
            self._generator = np.random.default_rng(self.random_seed)

        return (self._generator.random(probabilities.shape) < probabilities).astype(
            np.float64
        )

    @staticmethod
    def _check_values_are_bounded(feature_set: FeatureSet) -> None:
        """Raise unless every value lies in ``[0, 1]``.

        The energy function describes binary units, and a value outside the
        interval has no reading under it, as the numpy backend says.
        """
        for feature in feature_set:
            if not np.all((feature.values >= 0.0) & (feature.values <= 1.0)):
                raise InvalidValuesError(
                    f"{feature.name} must lie in [0, 1]; a Boltzmann unit is "
                    f"binary, and a value between is read as a mean"
                )

    @staticmethod
    def name_for(position: int) -> str:
        """The name of the hidden unit at ``position``, counting from zero."""
        return f"{HIDDEN_UNIT_NAME_PREFIX}_{position + 1}"

    def __repr__(self) -> str:
        if not self.is_fitted:
            return (
                f"RestrictedBoltzmannMachine("
                f"n_hidden_units={self.n_hidden_units}, unfitted)"
            )

        return (
            f"RestrictedBoltzmannMachine("
            f"{self.n_visible_units}x{self.n_hidden_units}, "
            f"epochs_run={self.epochs_run}, converged={self.converged})"
        )
