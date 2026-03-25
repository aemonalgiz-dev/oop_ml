"""Group rows by moving centres to the middle of whatever is nearest them.

Theory
------
Pick ``k`` points to be the centres. Then repeat two steps until nothing moves:

1. **Assign.** Give every row to whichever centre it is nearest.
2. **Update.** Move every centre to the mean of the rows that chose it.

That is the whole algorithm, and it is called Lloyd's algorithm. What is worth
understanding is why it terminates and what it is actually minimising, because
neither is obvious from the two steps.

The objective, and why each step lowers it
-------------------------------------------
k-means minimises **inertia**: the total squared distance from every row to its
own group's centre.

    inertia  =  sum over rows of  ||row - centre(row)||^2

Both steps lower it, for different reasons, and neither can raise it:

* **Assign** lowers it because moving a row to a nearer centre reduces that
  row's own term and touches nothing else. If no row can improve, the step is a
  no-op.
* **Update** lowers it because of a small fact worth stating outright: **the
  mean is the point minimising the sum of squared distances to a set of
  points.** Differentiate ``sum ||x - c||^2`` with respect to ``c``, set it to
  zero, and you get ``c = mean(x)``. So the update step is not a heuristic for
  moving the centre somewhere reasonable -- it jumps straight to the optimum
  for the current assignment.

Inertia therefore falls or stays flat at every step, and there are only finitely
many ways to assign ``n`` rows to ``k`` groups, so the process cannot cycle and
must stop. That is the convergence proof, and it is unusually short.

Why the metric is not a hyperparameter here
--------------------------------------------
The neighbour models take a :class:`~oop_ml.core.distance.metric.DistanceMetric`
because "near" is genuinely a choice there. k-means cannot, and the reason is
the fact above. The update step is a *mean*, and the mean is the minimiser for
squared **Euclidean** distance specifically. Swap in Manhattan and the mean is
no longer the optimal centre -- the median is -- so the update step stops
lowering the objective and the convergence argument collapses. Using another
metric here does not give you k-means with a different notion of near; it gives
you an algorithm with no guarantee at all. (The Manhattan version, with medians,
is a real algorithm with a different name.)

What it converges *to* is not what you wanted
----------------------------------------------
Lloyd's algorithm finds a local minimum, and which one depends entirely on where
the centres started. Two runs on identical data with different seeds can produce
genuinely different groupings, and neither is a bug.

This is why ``n_initialisations`` exists: run the whole thing several times from
different starts and keep the fit with the lowest inertia. It is also why
``k-means++`` seeding matters. Rather than picking ``k`` rows uniformly, which
can easily pick two rows from the same dense blob and none from a sparse one,
k-means++ picks the first centre uniformly and each subsequent one with
probability proportional to its squared distance from the nearest centre already
chosen. Far-away rows are more likely to be picked, so the initial centres tend
to be spread out.

The assumptions hiding in the objective
----------------------------------------
Minimising squared distance to a centre means the groups this can find are
**round, similarly sized, and similarly dense**. Two concentric rings cannot be
separated by it at any ``k``, because no pair of centres carves a ring out of a
ring. Neither can two elongated diagonal streaks. That is not a failure of the
implementation -- it is what "closest centre" means as a definition of a group.

And ``k`` is given, never learned. Inertia falls monotonically as ``k`` rises
and reaches zero when every row is its own group, so it cannot choose ``k`` for
you. See :class:`~oop_ml.core.clustering.clustering.Clustering` for that.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.estimator import Clusterer
from oop_ml.core.clustering.centroids import Centroid, Centroids
from oop_ml.core.clustering.clustering import (
    Clustering,
    InitialisationAttempt,
)
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.exceptions import InvalidValuesError, TooFewValuesError
from oop_ml.core.types import FloatArray, IndexArray

CLUSTER_NAME_PREFIX = "cluster"
"""How groups are named: ``cluster_1``, ``cluster_2``, and so on.

One-indexed to match the components, and a name rather than a bare position so
that a caller reading a report is looking at ``cluster_3`` rather than at the
number ``2``, which in this library usually means a class.
"""


class KMeans(Clusterer[Sequence[Feature]]):
    """Group rows into ``k`` round clusters by minimising inertia.

    Parameters
    ----------
    n_clusters:
        How many groups to find. Given, never learned -- see the module
        docstring for why inertia cannot choose it.
    n_initialisations:
        How many times to run the whole algorithm from a fresh seeding, keeping
        the fit with the lowest inertia. Lloyd's algorithm finds a local
        minimum and which one depends on the start, so this is the cheapest
        insurance against a bad one. Ten is the usual default and the one used
        here.
    max_iterations:
        A ceiling on the assign/update passes within one initialisation.
        Convergence is guaranteed, so this is a guard against a pathological
        case rather than the normal stopping condition.
    tolerance:
        The centres are considered settled when no centre moves further than
        this in one update. Compared against the *squared* movement, since
        every other distance here is squared too.
    random_seed:
        Fixes the seeding, so a fit is reproducible. Each initialisation is
        offset from it, the way an ensemble offsets its members' seeds -- one
        seed shared across all ten restarts would run the same restart ten
        times and report it as the best of ten.

    Raises
    ------
    pydantic.ValidationError
        If any count is below its minimum. Field bounds are pydantic's to
        enforce, so the error is pydantic's too.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_clusters: int = Field(default=8, ge=1)
    n_initialisations: int = Field(default=10, ge=1)
    max_iterations: int = Field(default=300, ge=1)
    tolerance: float = Field(default=1e-08, gt=0.0)
    random_seed: int | None = None

    LEARNED_STATE: ClassVar[tuple[str, ...]] = (
        "_clustering",
        "_feature_names",
        "_iterations_run",
    )

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

        Runs the whole algorithm ``n_initialisations`` times from different
        seedings and keeps the one with the lowest inertia. Each restart is
        given its own generator, offset from ``random_seed`` by its position,
        for the reason the ensemble frame offsets its members: one seed shared
        across every restart reruns a single restart and calls it the best of
        ten.

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

        self._feature_names = tuple(feature.name for feature in feature_set)
        rows = self._as_rows(feature_set)

        best: InitialisationAttempt | None = None

        for attempt in range(self.n_initialisations):
            generator = np.random.default_rng(
                None if self.random_seed is None else self.random_seed + attempt
            )
            candidate = self._one_initialisation(rows, generator)

            if best is None or candidate.beats(best):
                best = candidate

        assert best is not None
        self._clustering = best.clustering
        self._iterations_run = best.iterations_run
        self._mark_fitted()

        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """Label each row with the group whose centre it is nearest.

        Works on rows the fit never saw, which is what makes a fitted
        clusterer reusable. Nothing is relearned: the centres stay exactly
        where ``fit`` left them, and a new row simply falls to whichever is
        closest.

        Every fitted feature must be present, by name rather than position. A
        missing one makes the distance unevaluable, and an extra one has no
        coordinate in any centre to be compared against.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        self._check_fitted()
        self._check_features_match(input_values)

        rows = self._as_rows(FeatureSet(input_values))

        return Predictions.already_checked(
            self._assign(rows, self.centroids).astype(np.float64)
        )

    def _assign(self, rows: RowBlock, centroids: Centroids) -> IndexArray:
        """Give every row to its nearest centre.

        The first half of Lloyd's algorithm, and the shorter half.
        ``centroids.squared_distances_to(rows)`` hands you an
        ``(n_rows, n_clusters)`` table where entry ``[i, k]`` is how far row
        ``i`` sits from group ``k``. The label for row ``i`` is the position of
        the smallest entry in row ``i`` of that table.

        A row equidistant from two centres has to go somewhere, and either
        answer is correct. Take the lower-numbered group, which is what
        ``argmin`` does on its own -- the point is only that the rule must be
        deterministic, or two runs on identical data disagree for no reason.

        Parameters
        ----------
        rows:
            The rows to label, over the fitted features in fitted order.
        centroids:
            The current centres.

        Returns
        -------
        IndexArray
            One whole group position per row, shape ``(n_rows,)``.
        """
        return np.argmin(centroids.squared_distances_to(rows), axis=1)

    def _updated_positions(
        self, rows: RowBlock, labels: IndexArray, current: FloatArray
    ) -> FloatArray:
        """Move every centre to the mean of the rows that chose it.

        The second half of Lloyd's algorithm, and the half the convergence
        proof rests on: the mean is the point minimising the sum of squared
        distances to a set of points, so this does not nudge the centre in a
        good direction, it jumps to the best one available for the current
        assignment.

        **An empty group has no mean.** If no row chose group ``k``, averaging
        its members divides by zero and numpy hands back ``nan`` with a warning
        rather than an error -- and a ``nan`` centre makes every later distance
        ``nan``, so every ``argmin`` picks group 0 and the fit silently
        collapses to one cluster. That is why ``current`` is a parameter:
        leave such a centre exactly where it is. It will get another chance
        next pass, and if it never does, ``Clustering.sizes`` reports the empty
        group rather than hiding it.

        Parameters
        ----------
        rows:
            The rows being clustered.
        labels:
            The assignment just produced by :meth:`_assign`.
        current:
            Where the centres are now, shape ``(n_clusters, n_features)``. Used
            only for the groups that came out empty.

        Returns
        -------
        FloatArray
            The new centres, shape ``(n_clusters, n_features)``.
        """
        moved = current.copy()

        for position in range(current.shape[0]):
            members = rows.values[labels == position]

            if members.size:
                moved[position] = np.mean(members, axis=0)

        return moved

    def _seeded_positions(
        self, rows: RowBlock, generator: np.random.Generator
    ) -> FloatArray:
        """Choose starting centres by the k-means++ rule.

        Picking ``k`` rows uniformly is the obvious thing and it is a real
        weakness: a dense blob holding most of the rows is likely to be handed
        several centres while a sparse one gets none, and Lloyd's algorithm
        cannot recover from that because it only ever moves a centre to the
        middle of what already chose it.

        k-means++ spreads the start out instead:

        1. Pick the first centre uniformly at random from the rows.
        2. For every row, find the squared distance to the **nearest centre
           chosen so far**.
        3. Pick the next centre from the rows at random, with each row's
           probability proportional to that squared distance.
        4. Repeat from step 2 until there are ``n_clusters`` centres.

        Step 3 is the whole idea. It is not "take the furthest row", which
        would be deterministic and would chase outliers -- an outlier is
        exactly the row with the largest squared distance. Sampling in
        proportion makes far rows *likely* rather than certain, so the seeding
        is spread out without being at the mercy of one strange point.

        Use ``generator.choice`` with a ``p`` argument for step 3, and
        normalise the squared distances into probabilities that sum to 1.
        Identical rows can make every distance zero once a centre sits on them;
        if the total is zero there is nothing to weight by, so fall back to a
        uniform pick.

        Parameters
        ----------
        rows:
            The rows to seed from.
        generator:
            This restart's own generator.

        Returns
        -------
        FloatArray
            ``n_clusters`` starting centres, shape
            ``(n_clusters, n_features)``.
        """
        values = rows.values
        chosen = [values[generator.integers(values.shape[0])]]

        while len(chosen) < self.n_clusters:
            gaps = values[:, None, :] - np.array(chosen)[None, :, :]
            nearest = np.min(np.sum(gaps * gaps, axis=2), axis=1)
            total = float(np.sum(nearest))

            if total <= 0.0:
                chosen.append(values[generator.integers(values.shape[0])])
                continue

            chosen.append(values[generator.choice(values.shape[0], p=nearest / total)])

        return np.array(chosen, dtype=np.float64)

    def _one_initialisation(
        self, rows: RowBlock, generator: np.random.Generator
    ) -> InitialisationAttempt:
        """Seed once, then alternate assign and update until the centres settle.

        The Lloyd loop itself, which is plumbing around the three steps above.
        Stops when no centre moves further than ``tolerance``, or at
        ``max_iterations``.
        """
        positions = self._seeded_positions(rows, generator)
        labels = self._assign(rows, self._as_centroids(positions))
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            moved = self._updated_positions(rows, labels, positions)
            shift = float(np.max(np.sum((moved - positions) ** 2, axis=1)))
            positions = moved
            labels = self._assign(rows, self._as_centroids(positions))

            if shift <= self.tolerance:
                break

        centroids = self._as_centroids(positions)

        return InitialisationAttempt(
            Clustering(labels, centroids, self._inertia_of(rows, labels, centroids)),
            iterations,
        )

    @staticmethod
    def _inertia_of(rows: RowBlock, labels: IndexArray, centroids: Centroids) -> float:
        """Total squared distance from every row to the centre it was given."""
        distances = centroids.squared_distances_to(rows)

        return float(np.sum(distances[np.arange(labels.size), labels]))

    def _as_centroids(self, positions: FloatArray) -> Centroids:
        """Wrap a positions matrix as named centres, in the fitted features."""
        return Centroids(
            [
                Centroid(self.name_for(position), row, self._feature_names)
                for position, row in enumerate(positions)
            ]
        )

    def _as_rows(self, feature_set: FeatureSet) -> RowBlock:
        """The features as a row block, in the fitted order."""
        ordered = FeatureSet.matching(self._feature_names, list(feature_set))

        return rows_of(
            np.column_stack(
                [ordered.column(name).values for name in self._feature_names]
            ),
            self._feature_names,
        )

    def _check_features_match(self, input_values: Sequence[Feature]) -> None:
        """Raise unless the supplied features are exactly the fitted ones."""
        supplied = {feature.name for feature in input_values}
        fitted = set(self._feature_names)

        if supplied != fitted:
            raise InvalidValuesError(
                f"expected exactly the fitted features {sorted(fitted)}; "
                f"got {sorted(supplied)}"
            )

    @staticmethod
    def name_for(position: int) -> str:
        """The name of the group at ``position``, counting from zero.

        One place deciding what a group is called, so the seeding and the
        report cannot drift apart about it.
        """
        return f"{CLUSTER_NAME_PREFIX}_{position + 1}"

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"KMeans(n_clusters={self.n_clusters}, unfitted)"

        return (
            f"KMeans(n_clusters={self.n_clusters}, inertia={self.inertia:.4f}, "
            f"iterations_run={self.iterations_run})"
        )
