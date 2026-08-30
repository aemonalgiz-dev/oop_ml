"""Find the directions the data actually varies along, and keep the best few.

Theory
------
Every model before this one was handed a target and asked what predicts it.
This one is handed features and nothing else, and asks a question about their
shape: **along which direction does this cloud of points spread out most?**

Take that direction, record how much variance lies along it, then ask again
among the directions perpendicular to it, and again, until the directions run
out. That ordered, mutually perpendicular set is the principal components, and
it is a new set of axes for the same data -- a rotation, nothing thrown away
yet. What makes it useful is that the rotation *concentrates*: on real data the
first few axes usually carry most of the spread, so dropping the rest costs
little and buys a much narrower table.

Why variance is the thing maximised
-----------------------------------
Because a direction along which the points barely differ cannot tell any two of
them apart. Two features measuring nearly the same thing produce a cloud
stretched along one diagonal and flat across it, and the flat direction is
almost pure duplication. Variance is what "these rows are distinguishable" looks
like as a number.

The mechanics, which are one eigendecomposition
-----------------------------------------------
The variance along a unit direction ``v`` is ``v.T C v``, where ``C`` is the
covariance matrix of the centred data. Maximising that over unit vectors is the
textbook eigenvalue problem: the maximum is the largest eigenvalue of ``C`` and
it is attained at that eigenvalue's eigenvector. The next-best perpendicular
direction is the next eigenvector, and so on down. So the whole search is one
symmetric eigendecomposition, and the eigenvalue *is* the variance along its
eigenvector, which is why nothing extra has to be computed to report it.

Two consequences follow, and both are load-bearing:

* ``C`` is symmetric, so its eigenvectors are orthogonal and its eigenvalues
  real. That is not a lucky fact -- it is why the components come out
  perpendicular without anything having to make them so.
* ``numpy.linalg.eigh`` returns them **ascending**. PCA wants them descending.
  See :class:`~oop_ml.core.decomposition.components.PrincipalComponents` for
  what an unreversed sort does and why nothing downstream notices.

Centring is not optional, and scaling is a decision
---------------------------------------------------
The data has to be centred first, because variance is measured about the mean.
Skip it and the first component points at the mean itself -- it finds where the
cloud *is* rather than how it is *shaped*. So centring happens here always, and
the means are learned during ``fit`` and reused unchanged in ``transform``,
because re-centring held-out rows on their own mean is the same leak
``Standardizer`` exists to prevent.

Scaling is genuinely a choice, and it is a choice about units. Variance is not
unit-free: measure a length in millimetres instead of metres and its variance
grows by a million, so it wins the first component on nothing but its unit.
When the features are measured in different things, standardize first and the
decomposition is of the correlation matrix rather than the covariance one.
When they share a unit and their relative sizes are meaningful, do not, because
standardizing then throws away information you had.

What this is not
----------------
It is not feature selection. Every component is a blend of *all* the original
features, so "we kept two components" does not mean two columns were dropped --
it means the table now holds two mixtures rather than five measurements, and
what those mixtures mean is a question the loadings answer and the model does
not.

It is also unsupervised, so nothing here consults a target. A direction of large
variance is not thereby a direction that predicts anything, and the case where
the discarded low-variance direction was the one carrying the signal is real
rather than hypothetical.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.estimator import Transformer
from oop_ml.core.data.coefficients import Coefficient, Coefficients
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.decomposition.components import (
    PrincipalComponent,
    PrincipalComponents,
)
from oop_ml.core.exceptions import InvalidValuesError, TooFewValuesError
from oop_ml.core.types import FloatArray
from oop_ml.preprocessing.standardization.standardizer import Standardizer

COMPONENT_NAME_PREFIX = "component"
"""How components are named: ``component_1``, ``component_2``, and so on.

One-indexed, because "the first principal component" is what the literature and
every caller says, and a ``component_0`` would make the two disagree.
"""


class PrincipalComponentAnalysis(Transformer[Sequence[Feature]]):
    """Rotate the features onto the directions of greatest variance.

    A :class:`~oop_ml.core.base.estimator.Transformer` rather than an
    ``Estimator``, and that base already fits the shape: it takes no target,
    learns from the inputs alone, and rewrites them. No new frame was needed --
    the frame unsupervised learning still lacks here is the *clustering* one,
    where the answer is a label per row rather than a rewriting of the columns.

    Parameters
    ----------
    n_components:
        How many directions to keep. Left as ``None``, every direction is kept,
        which throws nothing away and is the right setting for asking how many
        you *should* keep -- fit once, read
        :meth:`~oop_ml.core.decomposition.components.PrincipalComponents.n_components_for`,
        refit.
    standardize:
        Whether to divide each feature by its standard deviation before
        decomposing, as well as centring it. Defaults to ``False``. The module
        docstring has the argument; the short version is that this is a question
        about units, not about defaults, and the answer is ``True`` whenever the
        features are not measured in the same thing.

    Raises
    ------
    pydantic.ValidationError
        If ``n_components`` is given and is less than 1. Field bounds are
        pydantic's to enforce, so the error is pydantic's too.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_components: int | None = Field(default=None, ge=1)
    standardize: bool = False

    _components: PrincipalComponents | None = PrivateAttr(default=None)
    _feature_means: dict[str, float] = PrivateAttr(default_factory=dict)
    _standardizer: Standardizer | None = PrivateAttr(default=None)

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

        The plumbing around :meth:`_solve`: validate the features, learn and
        apply the centring (and the scaling, if asked for), hand the prepared
        rows over, and only then mark the model fitted.

        ``_solve`` does not set ``_fitted`` and must not. That is this method's
        job, and it happens last, after the components are stored -- the same
        ordering every other fit here follows.

        Parameters
        ----------
        input_values:
            The features to decompose. At least two rows, since a single row has
            no spread to find a direction in, and at least one feature.

        Returns
        -------
        Self

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
            If ``n_components`` exceeds the number of features supplied.
        """
        feature_set = FeatureSet(input_values)

        if feature_set.n_samples < 2:
            raise TooFewValuesError(
                f"a decomposition needs at least two rows to have any spread; "
                f"got {feature_set.n_samples}"
            )

        if self.n_components is not None and self.n_components > feature_set.n_features:
            raise InvalidValuesError(
                f"cannot keep {self.n_components} components from "
                f"{feature_set.n_features} features"
            )

        prepared = self._prepared_for_fitting(feature_set)
        self._components = self._solve(prepared)
        self._mark_fitted()

        return self

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Project rows onto the learned components.

        One output feature per kept component, named ``component_1`` upward, in
        the components' own order. The output is *narrower* than the input
        whenever ``n_components`` was set, which is the whole reason to run
        this.

        Centre with the means learned during ``fit`` -- never with these rows'
        own means. Held-out rows re-centred on themselves have been told
        something about the held-out set, which is the leak the ``fit`` /
        ``transform`` split exists to prevent, and it is quieter here than it is
        for a standardizer because nothing about the output looks wrong.

        Every fitted feature must be present, and by name rather than by
        position, because a component is a weighted blend and a missing column
        makes the blend unevaluable. Extra features are refused too: a column
        the fit never saw has no loading to be weighted by.

        Parameters
        ----------
        input_values:
            Rows over exactly the features the fit saw, in any order.

        Returns
        -------
        list[Feature]
            One feature per component, each holding one coordinate per input
            row.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        centred = self._prepared_for_transforming(input_values)
        coordinates = centred.values @ self.components.directions.T

        return [
            Feature(component.name, coordinates[:, position])
            for position, component in enumerate(self.components)
        ]

    def inverse_transform(self, transformed_values: Sequence[Feature]) -> list[Feature]:
        """Rebuild the original features from their component coordinates.

        The reverse trip, and it is only lossless when every component was kept.
        Drop components and this returns the data's shadow on the kept
        subspace: the closest point to each original row that the kept
        directions can express. That gap is the reconstruction error, and it is
        exactly the variance the discarded components held.

        The arithmetic is a transpose rather than an inverse, and that is what
        the orthogonality invariant buys. For an orthonormal set the projection
        back is ``coordinates @ directions``, after which the learned centring
        is undone in the reverse of the order ``transform`` applied it.

        Parameters
        ----------
        transformed_values:
            Component coordinates, exactly the features ``transform`` produces.

        Returns
        -------
        list[Feature]
            One feature per original name, in the order the fit saw them.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly this model's components.
        """
        self._check_fitted()
        self._check_components_match(transformed_values)

        by_name = {feature.name: feature for feature in transformed_values}
        coordinates = np.column_stack(
            [by_name[component.name].values for component in self.components]
        )

        return self._restored(coordinates @ self.components.directions)

    def _solve(self, centred: RowBlock) -> PrincipalComponents:
        """Find the directions of greatest variance in already-centred rows.

        The concept, and the only part of this class that is the algorithm
        rather than the bookkeeping. Given rows whose every column already has
        mean zero:

        1. Build the covariance matrix. For centred rows that is
           ``rows.T @ rows / (n_rows - 1)``, an ``(n_features, n_features)``
           symmetric table whose ``[i, j]`` entry is how features ``i`` and
           ``j`` vary together.
        2. Eigendecompose it with ``numpy.linalg.eigh``, which exists for
           symmetric matrices and guarantees real eigenvalues and orthogonal
           eigenvectors. It returns eigenvalues ascending and eigenvectors as
           **columns**, so ``vectors[:, k]`` goes with ``values[k]``.
        3. Reverse both into descending order. This is the step whose omission
           is silent; see the components module.
        4. Keep the first ``n_components`` of them, or all if that is ``None``.
        5. Build one :class:`~oop_ml.core.decomposition.components.Principal\\
           Component` per kept direction, pairing its loadings with
           ``centred.feature_names``, and hand the lot to ``PrincipalComponents``
           along with the total variance.

        The total variance is the sum of **every** eigenvalue, not of the kept
        ones. That is the denominator the shares are reported against, and
        summing only what survived truncation would make every fit claim to
        explain all of the variance. Compute it before step 4.

        One trap that a hand-checked fixture will find for you. A covariance
        matrix is positive semi-definite, so no eigenvalue can really be
        negative -- but a direction along which the data has *no* spread comes
        back as a small negative number rather than a clean zero, because the
        solver is working in floating point. Two perfectly correlated columns
        produce exactly that, and ``PrincipalComponent`` refuses a negative
        variance, so the fit raises on data that is not wrong. Clamp the
        eigenvalues at zero when you reverse them.

        Note that an eigenvector's sign is arbitrary: ``v`` and ``-v`` describe
        the same direction and the same variance, and different platforms
        legitimately return different signs. Nothing here should try to fix a
        sign, and no test should assert one.

        Parameters
        ----------
        centred:
            Rows with every column already mean-zero, and the feature names in
            column order.

        Returns
        -------
        PrincipalComponents
            The kept directions, ordered, paired with the full total variance.
            Do not set ``_fitted`` here.
        """
        covariance_matrix = centred.values.T @ centred.values / (centred.n_rows - 1)
        eigenvalues, eigen_vectors = np.linalg.eigh(covariance_matrix)

        sorted_eigen_indices = eigenvalues.argsort(descending=True)

        sorted_eigenvalues = np.maximum(eigenvalues[sorted_eigen_indices], 0.0)
        sorted_eigen_vectors = eigen_vectors[:, sorted_eigen_indices]

        total_variance = float(np.sum(sorted_eigenvalues))

        if self.n_components:
            sorted_eigenvalues = sorted_eigenvalues[: self.n_components]
            sorted_eigen_vectors = sorted_eigen_vectors[:, : self.n_components]

        principal_components: list[PrincipalComponent] = []

        for index in range(0, sorted_eigenvalues.shape[0]):
            coefficients: list[Coefficient] = []

            for feature_index, value in enumerate(sorted_eigen_vectors[:, index]):
                coefficients.append(
                    Coefficient(
                        name=centred.feature_names[feature_index],
                        value=float(value),
                    )
                )

            principal_components.append(
                PrincipalComponent(
                    name=self.name_for(index),
                    loadings=Coefficients(coefficients),
                    variance=float(sorted_eigenvalues[index]),
                )
            )

        return PrincipalComponents(principal_components, total_variance)

    def _prepared_for_fitting(self, feature_set: FeatureSet) -> RowBlock:
        """Centre the features, scaling them first if asked, and learn how.

        Both the means and the standardizer are stored, because ``transform``
        has to repeat this exact preparation on rows the fit never saw.
        """
        if self.standardize:
            self._standardizer = Standardizer().fit(list(feature_set))
            feature_set = FeatureSet(self._standardizer.transform(list(feature_set)))

        self._feature_means = {
            feature.name: float(np.mean(feature.values)) for feature in feature_set
        }

        return self._centred(feature_set)

    def _centred(self, feature_set: FeatureSet) -> RowBlock:
        """Subtract the learned mean from every column, in the fitted order."""
        names = tuple(self._feature_means)
        ordered = FeatureSet.matching(names, list(feature_set))

        return rows_of(
            np.column_stack(
                [
                    ordered.column(name).values - self._feature_means[name]
                    for name in names
                ]
            ),
            names,
        )

    def _prepared_for_transforming(self, input_values: Sequence[Feature]) -> RowBlock:
        """Repeat the fit's preparation on new rows, learning nothing new.

        The counterpart of :meth:`_prepared_for_fitting`, and the reason both
        the means and the standardizer were kept. ``transform`` calls this and
        then does the projection.
        """
        self._check_fitted()
        self._check_features_match(input_values)

        features = list(input_values)

        if self._standardizer is not None:
            features = self._standardizer.transform(features)

        return self._centred(FeatureSet(features))

    def _check_features_match(self, input_values: Sequence[Feature]) -> None:
        """Raise unless the supplied features are exactly the fitted ones.

        Exactly, in both directions. A missing column makes a component's blend
        unevaluable; an extra one has no loading to weight it by. Order is free,
        because everything downstream reorders by name.
        """
        supplied = {feature.name for feature in input_values}
        fitted = set(self._feature_means)

        if supplied != fitted:
            raise InvalidValuesError(
                f"expected exactly the fitted features "
                f"{sorted(fitted)}; got {sorted(supplied)}"
            )

    def _check_components_match(self, transformed_values: Sequence[Feature]) -> None:
        """Raise unless the supplied features are exactly this fit's components."""
        supplied = {feature.name for feature in transformed_values}
        produced = {component.name for component in self.components}

        if supplied != produced:
            raise InvalidValuesError(
                f"expected exactly this model's components "
                f"{sorted(produced)}; got {sorted(supplied)}"
            )

    def _restored(self, rebuilt: FloatArray) -> list[Feature]:
        """Undo the fit's centring and scaling on a reconstructed block.

        The mirror of :meth:`_prepared_for_fitting`, applied in the reverse
        order: add the means back first, then let the standardizer restore the
        original units.
        """
        names = tuple(self._feature_means)
        features = [
            Feature(name, rebuilt[:, position] + self._feature_means[name])
            for position, name in enumerate(names)
        ]

        if self._standardizer is None:
            return features

        return [
            Feature(
                feature.name,
                self._standardizer.scalings[feature.name].restore(feature.values),
            )
            for feature in features
        ]

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
        """Pair one direction vector with the names its entries weight.

        The bookkeeping half of building a component, kept here so ``_solve``
        stays the arithmetic. Give it the position (counting from zero), the
        eigenvector, its eigenvalue, and the feature names in column order.
        """
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
        """The name of the component at ``position``, counting from zero.

        One place deciding what a component is called, so ``_solve`` and
        ``transform`` cannot drift apart about it.
        """
        return f"{COMPONENT_NAME_PREFIX}_{position + 1}"

    def __repr__(self) -> str:
        if not self.is_fitted:
            raise_free = "unfitted"
        else:
            raise_free = (
                f"n_components={self.components.n_components}, "
                f"explained={self.components.cumulative_shares[-1]:.4f}"
            )

        return f"PrincipalComponentAnalysis({raise_free})"
