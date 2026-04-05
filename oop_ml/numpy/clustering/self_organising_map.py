"""Learn a grid of prototypes whose neighbours on the grid stay neighbours in the data.

What is different about this model
-----------------------------------
Every learning model in this library up to the network package computes an error
against a target and then propagates it backwards through whatever produced the
answer. This one has no target and no backward pass. A unit's weights change by
an amount that depends on two things only, the value presented and the unit's own
current weights, scaled by how near that unit sits to the winner **on the grid**.
Nothing travels back from a loss, because there is no loss.

That is what a *local* learning rule means, and it is the entire reason this
model is here. The update is

    weights[unit] += rate * neighbourhood[unit] * (row - weights[unit])

and every term in it is available at the unit. A backpropagating network cannot
say that about a single weight in its first layer, whose update depends on every
parameter in every layer above it.

The algorithm
-------------
Fix a rectangular grid of units, each holding a weight vector in the input space.
Present the rows one at a time, in a fresh random order each epoch. For each row:

1. **Compete.** Find the unit whose weights are nearest the row. That is the
   winner, sometimes called the best matching unit.
2. **Cooperate.** Score every unit by how near it sits to the winner *on the
   grid*, through a Gaussian that falls off with grid distance:

       neighbourhood[unit] = exp(-(grid_distance(unit, winner) ** 2)
                                 / (2 * radius ** 2))

   The winner scores 1.0 and everything else scores less.
3. **Adapt.** Move every unit that fraction of the remaining way toward the row.

Both ``rate`` and ``radius`` shrink across the walk, and both are
:class:`~oop_ml.core.schedule.Schedule` objects rather than numbers.

The decay is not tuning, it is what makes the process converge
---------------------------------------------------------------
This is the claim worth measuring rather than repeating, because a constant rate
sounds merely suboptimal and is in fact fatal. Each update moves the winner a
fixed *fraction* of the way to the row it just saw, so the size of the step does
not shrink as the map improves. A well placed unit is dragged just as far by the
next row as a badly placed one. The map therefore never settles, it tracks
whichever row arrived most recently, and the answer you read at the end is a fact
about presentation order rather than about the data.

Measured on ``CURVE_CLOUD`` in the spec -- sixty points along an arc of radius 5,
so a cloud roughly ten across -- on a 1 x 8 chain, with the radius held at 1.0
throughout so that only the rate varies. The number is the largest distance any
unit moved during the walk's final epoch:

    epochs   constant 0.5   exponential 0.5 -> 0.01
        30         1.876                     0.0045
       100         0.950                     0.0044
       300         1.134                     0.0103
      1000         2.196                     0.0056

The constant column has no trend, which is the finding. Running the walk ten
times as long leaves the map moving as far on its last epoch as on its thirtieth,
because nothing in it is getting smaller. A unit still travelling 1.1 per epoch
across a cloud ten wide is being dragged a tenth of the way across the picture by
whichever row happened to arrive most recently, which is what "the answer is a
fact about presentation order" looks like as a number. The decaying column sits
two orders of magnitude below it at every length.

With both schedules realistic -- the radius decaying 3.0 down to 0.5 as well --
the same comparison gives 1.354, 1.414, 0.657 and 1.198 for the constant rate
against 0.058, 0.027, 0.012 and 0.004 for the decaying one. The second column
falls; the first wanders. That is the Robbins-Monro condition the schedule module
argues from, seen as a number.

k-means is the zero-radius case
--------------------------------
This is the sharpest thing that can be said about the model. Set the
neighbourhood to zero width and the Gaussian collapses to a one at the winner and
a zero everywhere else, so only the winner moves and the rule becomes

    weights[winner] += rate * (row - weights[winner])

which is an online mean of the rows that unit has won. k-means computes the same
mean in a batch, by assigning every row and then jumping each centre to the
average of what chose it. Same objective, same fixed points, different arithmetic
for getting there.

Measured on ``THREE_BLOBS`` in the spec, three units at zero radius against
``KMeans(n_clusters=3)``, over five seeds: the two agree about the grouping every
time, row for row. They agree about no label at all. At seed 0 the map answers
``2 2 2 2 0 0 0 0 1 1 1 1`` where k-means answers ``1 1 1 1 2 2 2 2 0 0 0 0``,
which is the same partition written in different numbers, and is exactly why the
comparison goes through a partition test.

They are not guaranteed to agree in general, and the spec says so rather than
overclaiming. Two differences survive switching the neighbourhood off. Online
updating visits one row at a time, so the map depends on presentation order in a
way Lloyd's algorithm does not; and the rate decays toward something small rather
than jumping straight to the optimum for the current assignment, so a capped walk
can stop short of a fixed point k-means reaches in a handful of passes. On
separated blobs neither difference has anywhere to bite, which is why the fixture
is separated. Where the local minima are real, both differences can and do move
the answer.

Topology preservation is what the neighbourhood buys
------------------------------------------------------
With the radius switched off this is k-means with extra machinery. What the
radius adds is the property k-means has no way to express: **units adjacent on
the grid end up with adjacent weight vectors.** Because a winner drags its grid
neighbours along with it, units near each other on the grid keep being pulled
toward the same region of the input space, and the grid ends up as a map of the
data in the ordinary sense of the word.

Measured on ``CURVE_CLOUD``, a 1 x 8 chain over sixty points along an arc, 200
epochs, correlating grid distance against weight distance across all 28 pairs of
units:

* radius decaying 3.0 down to 0.5: **0.9909**, and the same to four places at
  each of four seeds, because the arrangement is what the map converges to rather
  than something the seeding decides.
* radius held at zero: **-0.038, -0.212, -0.003, -0.139** at those same four
  seeds. No relationship, and a different absence of one each time.

The second row is the control and it is the point. The zero-radius map quantises
the data perfectly well and carries no arrangement whatever -- its unit 1 and unit
2 sit beside each other only because the seeding numbered them that way, so the
correlation is noise around zero and changes sign with the seed. Nothing about a
k-means fit could have been read off a picture; this can.

The same measurement on a 2 x 4 grid over the same one-dimensional data gives
**0.9034**, lower and correctly so: a two-dimensional grid laid over a curve has
to fold, and the fold costs exactly the agreement between the two distances that
the correlation is measuring.

The grid distance is not the input distance
---------------------------------------------
Both are distances, both come out of a subtraction, and on a chain over
one-dimensional data they even have the same shape. This is the central
implementation trap, and it was written and run rather than merely imagined.

Replacing the grid distance in the neighbourhood with the distance between the
two units' *weight vectors* produces a model that fits, predicts, and reports
plausible numbers. Two things then happen, both measured. The grid stops mattering
at all, so the 1 x 8 and the 2 x 4 fits come back **bit-identical** where the
honest implementation puts 4.151 between their furthest-apart weights. And the
map **collapses**: units that already agree pull each other closer, which makes
them agree more, and every unit ends on one point, so the topology correlation is
not low but undefined -- every weight distance is zero and the correlation divides
by a zero standard deviation. That runaway is the reason the neighbourhood must
be a fact about the grid, which does not change, rather than about the weights,
which the rule is busy changing.

The guard against this is the type rather than the name.
:meth:`GridPosition.distance_to` accepts a :class:`GridPosition` and nothing
else, and a weight vector is a :class:`~oop_ml.core.clustering.centroids.Centroid`,
so the confusion is not expressible rather than merely discouraged. The spec
keeps it closed from the outside as well, by fitting the same data on a 1 x 8 and
a 2 x 4 grid and asserting the answers differ.

Why the metric is not a hyperparameter, as with k-means and for a related reason
---------------------------------------------------------------------------------
:class:`~oop_ml.core.distance.metric.DistanceMetric` is a genuine choice on the
neighbour models and is refused here, and the argument is one step removed from
the one k-means makes. There the update is a mean, and the mean is the minimiser
of squared *Euclidean* distance specifically. Here the update is
``rate * (row - weights)``, which is a step along the negative gradient of
``||row - weights||^2 / 2`` -- again squared Euclidean, and again specifically.
Under Manhattan the gradient is ``sign(row - weights)``, a fixed-size step in
every coordinate at once, which is a different rule and not this one.

The variant worth naming is the incoherent one, because it is what a metric field
would actually produce. Letting the competition use Manhattan while the
adaptation stays as written means rows are assigned by one measure and prototypes
move under another, so no single quantity is being reduced by both halves of the
step. That is worse than a wrong metric; it is two metrics disagreeing inside one
model.

What ``converged`` means here, which is the weaker of the two readings
----------------------------------------------------------------------
:class:`~oop_ml.core.base.convergent_fit.ConvergentFit` requires each model to say
which of the two things its ``converged`` means, and this one means the weaker.
The weights stopped moving. It does **not** mean a maximum or minimum of anything
was reached, because the online self-organising map is not descending a global
objective at all -- for a continuous input distribution no cost function exists
whose gradient this rule follows, which is a result about the algorithm rather
than a gap in this implementation. What it is descending, at a fixed assignment
of rows to winners, is the neighbourhood-weighted squared quantisation error, and
the assignment changes underneath it as the weights move.

Since the rate decays to a small value on its own, a settled walk here is partly
the schedule running out rather than the map arriving. Read
:attr:`SelfOrganisingMap.final_epoch_movement` alongside ``converged``, which is
why it is public.

The inherited default tolerance of 1e-8 is far below anything an online walk
reaches, and that is honest rather than a misconfiguration. Measured on the
chain above, the walk settles at 86 epochs against a tolerance of 0.1 and at 117
against 0.05, and runs the full 200 against 0.01 with a final movement of 0.0154.
So ``converged`` is normally False here and ``max_epochs`` is normally what stops
the walk, which is the opposite of k-means and follows from the schedules being
expressed as fractions of the run.

A label means nothing across fits
-----------------------------------
The same warning :class:`~oop_ml.numpy.clustering.k_means.KMeans` carries, and for the
same reason: nothing ever told this model which group deserved which number. Two
correct fits can agree completely about which rows belong together and disagree
about every label. Every assertion in the spec goes through a partition
comparison for that reason.

There is one extra wrinkle here. Because the grid is an arrangement, "unit 3" is
not merely an arbitrary number, it is a *position*, and a fit seeded differently
can produce the mirror image of the same map -- the arrangement preserved, read
right to left. So even the topology claim is asserted as a correlation between
distances rather than as anything about which unit sits where.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import ClassVar, Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.convergent_fit import ConvergentFit
from oop_ml.core.base.estimator import Clusterer
from oop_ml.core.clustering.centroids import Centroid, Centroids
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
)
from oop_ml.core.schedule import (
    ExponentialDecaySchedule,
    LinearDecaySchedule,
    Schedule,
)
from oop_ml.core.types import FloatArray, IndexArray

UNIT_NAME_PREFIX = "unit"
"""How units are named: ``unit_1``, ``unit_2``, and so on.

One-indexed to match the clusters and the components, and a name rather than a
bare position for the same reason: a report saying ``unit_3`` is saying something,
and one saying ``2`` is saying a number that in this library usually means a
class.
"""


class GridPosition:
    """Where a unit sits on the map, which is not where its weights sit.

    Two whole numbers, and a type whose whole job is to be a different type from
    a point in the input space. The neighbourhood is a function of distance
    *here*, and the competition is a function of distance *there*, and the
    central implementation trap in this model is using one for the other. Both
    are Euclidean distances between vectors of numbers, so no amount of care with
    names reliably separates them; two types do.

    Parameters
    ----------
    row:
        Which row of the grid, counting from zero.
    column:
        Which column of the grid, counting from zero.

    Raises
    ------
    InvalidValuesError
        If either coordinate is negative, or is not a whole number.
    """

    __slots__ = ("_column", "_row")

    def __init__(self, row: int, column: int) -> None:
        for name, value in (("row", row), ("column", column)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidValuesError(
                    f"a grid {name} is a whole position; got {value!r}"
                )
            if value < 0:
                raise InvalidValuesError(f"a grid {name} counts from zero; got {value}")

        self._row = row
        self._column = column

    @property
    def row(self) -> int:
        """Which row of the grid this unit is in, counting from zero."""
        return self._row

    @property
    def column(self) -> int:
        """Which column of the grid this unit is in, counting from zero."""
        return self._column

    def distance_to(self, other: GridPosition) -> float:
        """How far this unit sits from ``other`` **on the grid**.

        Straight-line distance across the rectangle, so the diagonal neighbour of
        a unit is ``sqrt(2)`` away rather than 1 or 2. The alternatives are real
        choices -- counting steps along the rows and columns gives 2, and
        counting the larger of the two gives 1 -- and Euclidean is taken here
        because the Gaussian neighbourhood is stated in terms of a radius, and a
        radius that means different things along a diagonal than along an axis is
        not a radius.

        Parameters
        ----------
        other:
            Another position on the same grid.

        Returns
        -------
        float
            The straight-line grid distance, which is zero to itself.
        """
        row_gap = float(self._row - other._row)
        column_gap = float(self._column - other._column)

        return float(np.hypot(row_gap, column_gap))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridPosition):
            return NotImplemented

        return self._row == other._row and self._column == other._column

    def __hash__(self) -> int:
        return hash((self._row, self._column))

    def __repr__(self) -> str:
        return f"GridPosition(row={self._row}, column={self._column})"


class MapUnit:
    """One unit: a place on the grid, and a point in the input space.

    The pairing exists because these are two different things about one object
    and a method reaching for ``return position, weights`` is this class not yet
    written. They are also two different *kinds* of thing, which is the more
    interesting half: the position is fixed at construction and never learned,
    while the weights are the entirety of what the fit produces.

    The weights are a
    :class:`~oop_ml.core.clustering.centroids.Centroid` rather than a bare array,
    and that reuse is the zero-radius argument made structural. A unit's weight
    vector is a prototype -- a location in feature space that a set of rows is
    nearest to -- which is exactly what a k-means centre is. Reaching for a new
    class here would have asserted a difference that the mathematics says is not
    there.

    Parameters
    ----------
    position:
        Where on the grid this unit sits.
    prototype:
        The unit's weight vector, named and bound to the features it lives in.
    """

    __slots__ = ("_position", "_prototype")

    def __init__(self, position: GridPosition, prototype: Centroid) -> None:
        self._position = position
        self._prototype = prototype

    @property
    def name(self) -> str:
        """What this unit is called, ``unit_1`` upward."""
        return self._prototype.name

    @property
    def position(self) -> GridPosition:
        """Where on the grid this unit sits."""
        return self._position

    @property
    def prototype(self) -> Centroid:
        """The unit's weight vector as a named point in feature space."""
        return self._prototype

    @property
    def weights(self) -> FloatArray:
        """The weight vector as a plain array, in fitted feature order."""
        return self._prototype.coordinates

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The features the weights are coordinates in, in order."""
        return self._prototype.feature_names

    def weight_for(self, name: str) -> float:
        """This unit's weight along one named feature.

        Raises
        ------
        InvalidValuesError
            If ``name`` is not one of the fitted features.
        """
        return self._prototype.coordinate_for(name)

    def __repr__(self) -> str:
        return f"MapUnit({self.name!r}, {self._position!r})"


class UnitGrid:
    """Every unit, arranged in the rectangle that gives the model its meaning.

    Parameters
    ----------
    units:
        ``width * height`` units, in row-major order: the unit at index ``i``
        must sit at row ``i // width`` and column ``i % width``.
    width:
        How many columns the grid has.
    height:
        How many rows the grid has.

    Raises
    ------
    EmptyValuesError
        If no units are supplied.
    InvalidValuesError
        If either extent is below one, if the count does not match
        ``width * height``, if the positions are not the rectangle in row-major
        order, or if the units do not all live in the same features.
    NonUniqueFeaturesError
        If two units share a name.

    Notes
    -----
    The ordering invariant is the interesting difference from
    :class:`~oop_ml.core.clustering.centroids.Centroids`, which deliberately
    enforces none. Clusters are not ranked and imposing an order on them would
    invent a fact. Units are not ranked either -- unit 1 is no more important
    than unit 5 -- but they are *arranged*, and the arrangement is the whole
    model rather than an artefact of it. ``predict`` answers with an index, and
    :meth:`unit_at` addresses by row and column, and those two have to be talking
    about the same unit or a caller drawing the map draws it scrambled.
    """

    __slots__ = ("_height", "_units", "_width")

    def __init__(self, units: Sequence[MapUnit], width: int, height: int) -> None:
        if not units:
            raise EmptyValuesError("a map needs at least one unit")

        if width < 1 or height < 1:
            raise InvalidValuesError(
                f"a grid has at least one row and one column; got {height} x {width}"
            )

        if len(units) != width * height:
            raise InvalidValuesError(
                f"a {height} x {width} grid holds {width * height} units; got "
                f"{len(units)}"
            )

        for index, unit in enumerate(units):
            expected = GridPosition(index // width, index % width)
            if unit.position != expected:
                raise InvalidValuesError(
                    f"{unit.name} sits at {unit.position!r}, but index {index} of "
                    f"a {height} x {width} grid is {expected!r}"
                )

        # Delegated rather than repeated: uniqueness of names and agreement on
        # the features are exactly the rules Centroids already enforces, and a
        # second implementation of them here would be a second opinion.
        Centroids([unit.prototype for unit in units])

        self._units = tuple(units)
        self._width = width
        self._height = height

    @property
    def width(self) -> int:
        """How many columns the grid has."""
        return self._width

    @property
    def height(self) -> int:
        """How many rows the grid has."""
        return self._height

    @property
    def n_units(self) -> int:
        """How many units the grid holds, which is ``width * height``."""
        return len(self._units)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The features every unit's weights live in, in order."""
        return self._units[0].feature_names

    @property
    def n_features(self) -> int:
        """How many dimensions the weight vectors have."""
        return len(self.feature_names)

    @property
    def prototypes(self) -> Centroids:
        """Every unit's weight vector, as the library's named-point collection.

        What the competition is run against, and the reason a weight vector is a
        centroid: ``squared_distances_to`` is already the exact, blocked
        arithmetic k-means uses, so nothing here reimplements it.
        """
        return Centroids([unit.prototype for unit in self._units])

    @property
    def weights(self) -> FloatArray:
        """The whole map as a ``(n_units, n_features)`` matrix, in index order."""
        return np.array([unit.weights for unit in self._units], dtype=np.float64)

    def unit_at(self, row: int, column: int) -> MapUnit:
        """The unit sitting at one place on the grid.

        Raises
        ------
        InvalidValuesError
            If the place is outside the grid.
        """
        if not 0 <= row < self._height or not 0 <= column < self._width:
            raise InvalidValuesError(
                f"({row}, {column}) is outside a {self._height} x {self._width} grid"
            )

        return self._units[row * self._width + column]

    def unit_named(self, name: str) -> MapUnit:
        """The unit called ``name``.

        Raises
        ------
        InvalidValuesError
            If no unit has that name.
        """
        for unit in self._units:
            if unit.name == name:
                return unit

        raise InvalidValuesError(
            f"unknown unit {name!r}; this map holds "
            f"{[unit.name for unit in self._units]}"
        )

    def position_of(self, name: str) -> GridPosition:
        """Where the unit called ``name`` sits on the grid.

        What a caller draws the map from. Ask every unit where it is, ask it what
        its weights are, and the picture follows.

        Raises
        ------
        InvalidValuesError
            If no unit has that name.
        """
        return self.unit_named(name).position

    def __getitem__(self, name: str) -> MapUnit:
        return self.unit_named(name)

    def __contains__(self, name: object) -> bool:
        return any(unit.name == name for unit in self._units)

    def __iter__(self) -> Iterator[MapUnit]:
        return iter(self._units)

    def __len__(self) -> int:
        return self.n_units

    def __repr__(self) -> str:
        return f"UnitGrid({self._height}x{self._width}, n_features={self.n_features})"


class SelfOrganisingMap(Clusterer[Sequence[Feature]], ConvergentFit):
    """Fit a grid of prototypes by competition, cooperation and adaptation.

    Parameters
    ----------
    grid_width:
        How many columns of units. A width of ``n`` with a height of 1 is a
        chain, which is the arrangement to reach for when the structure being
        looked for is one-dimensional.
    grid_height:
        How many rows of units. The number of clusters is the product of the two,
        so a 3 x 3 map holds nine prototypes.
    learning_rate:
        How far a winning unit moves toward the row it just saw, as a fraction of
        the gap, decaying across the walk. Exponential by default, because what
        matters about a rate is its order of magnitude.
    neighbourhood_radius:
        The width of the Gaussian that decides how much a unit moves with the
        winner, measured in grid units and decaying across the walk. Linear by
        default, because a radius is measured in the grid's own units and the
        midpoint of the walk should be the middle of the range. **This default
        does not know how large your grid is**, and cannot: it is a field, and a
        field default cannot read another field. Start it near half the grid's
        longest side, so that the first epochs arrange the whole map and the last
        ones refine each unit alone.
    max_epochs:
        A ceiling on the passes over the data. Unlike k-means's, this one is
        usually what stops the walk rather than a guard against a pathological
        case, because both schedules are expressed as fractions of it.
    tolerance:
        Inherited from :class:`~oop_ml.core.base.convergent_fit.ConvergentFit`.
        The walk stops early once no unit moves further than this in a whole
        epoch.
    random_seed:
        Fixes both the seeding of the weights and the presentation order, so a
        fit is reproducible. Left ``None``, neither is.

    Raises
    ------
    pydantic.ValidationError
        If any extent or the epoch cap is below its minimum. Field bounds are
        pydantic's to enforce, so the error is pydantic's too.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    grid_width: int = Field(default=4, ge=1)
    grid_height: int = Field(default=4, ge=1)
    learning_rate: Schedule = ExponentialDecaySchedule(start=0.5, end=0.01)
    neighbourhood_radius: Schedule = LinearDecaySchedule(start=2.0, end=0.5)
    max_epochs: int = Field(default=100, ge=1)
    random_seed: int | None = None

    LEARNED_STATE: ClassVar[tuple[str, ...]] = (
        "_units",
        "_feature_names",
        "_final_epoch_movement",
        "_passes_run",
        "_converged",
    )

    _units: UnitGrid | None = PrivateAttr(default=None)
    _feature_names: tuple[str, ...] = PrivateAttr(default=())
    _final_epoch_movement: float | None = PrivateAttr(default=None)

    @property
    def _pass_limit(self) -> int:
        """The cap on epochs, under this model's own name for it."""
        return self.max_epochs

    @property
    def n_units(self) -> int:
        """How many prototypes the map holds, known before any data arrives."""
        return self.grid_width * self.grid_height

    @property
    def units(self) -> UnitGrid:
        """The fitted map: every unit's place on the grid and its weights.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._units is not None
        return self._units

    @property
    def epochs_run(self) -> int:
        """How many passes over the data the fit took.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self._completed_passes

    @property
    def final_epoch_movement(self) -> float:
        """The largest distance any unit moved during the last epoch run.

        Public because ``converged`` alone is a poor summary here. The rate
        decays on its own, so a walk can settle because the steps became tiny
        rather than because the map arrived, and this is the number that
        distinguishes a map still chasing rows at 0.54 from one refining itself
        at 0.013.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._final_epoch_movement is not None
        return self._final_epoch_movement

    def position_of(self, name: str) -> GridPosition:
        """Where the unit called ``name`` sits on the grid.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If no unit has that name.
        """
        return self.units.position_of(name)

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Organise the grid against ``input_values``.

        Seeds the weights from randomly drawn rows, then runs at most
        ``max_epochs`` passes. Each pass presents every row once, in a fresh
        random order, and applies the competitive update for each.

        **The order is reshuffled every epoch, and that is not incidental.** The
        update is online, so the last row seen has moved the map more recently
        than any other; a fixed order therefore bakes the tail of the data into
        the final answer, and two datasets differing only in row order fit
        differently. Reshuffling makes the order a nuisance that averages out
        instead of a parameter nobody chose.

        The walk stops early when no unit moved further than ``tolerance`` in a
        whole epoch, and either way ``epochs_run`` and ``converged`` record how
        it ended.

        Parameters
        ----------
        input_values:
            The features to organise. Any number of rows -- unlike k-means, a
            map with more units than rows is not refused, because a unit that
            never wins is an ordinary and informative outcome here rather than
            an empty group with no mean.

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
        """
        feature_set = FeatureSet(input_values)

        generator = np.random.default_rng(self.random_seed)
        feature_names = tuple(feature.name for feature in feature_set)
        rows = rows_of(
            np.column_stack([feature.values for feature in feature_set]), feature_names
        )

        weights = self._seeded_weights(rows, generator)

        epochs_run = 0
        converged = False
        movement = np.full(self.n_units, np.inf)

        while epochs_run < self.max_epochs:
            epochs_run += 1
            started_at = weights

            rate = self.learning_rate.value_at(epochs_run, self.max_epochs)
            radius = self.neighbourhood_radius.value_at(epochs_run, self.max_epochs)

            for row_position in generator.permutation(rows.n_rows):
                presented = rows.row(int(row_position))
                weights = self._updated_weights(
                    weights,
                    presented,
                    self._winner_for(presented, weights),
                    rate,
                    radius,
                )

            movement = np.sqrt(np.sum((weights - started_at) ** 2, axis=1))

            if self._has_converged(movement):
                converged = True
                break

        # Nothing is committed until every step has succeeded, which is the
        # pattern the serving audit established: a refit that raises inside the
        # loop leaves the previous fit whole rather than half replaced.
        units = self._as_units(weights, feature_names)

        self._feature_names = feature_names
        self._units = units
        self._final_epoch_movement = float(np.max(movement))
        self._record_walk(epochs_run, converged)
        self._mark_fitted()

        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """Label each row with the unit whose weights are nearest it.

        The competition step alone, run on rows the fit may never have seen.
        Nothing is relearned: the weights stay exactly where ``fit`` left them.

        The answer is a unit *index*, running ``0 .. n_units - 1`` in the same
        row-major order :meth:`UnitGrid.unit_at` uses, so a caller can turn a
        label back into a place on the grid.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        NonUniqueFeaturesError
            If two supplied features share a name, which is checked first so
            that a duplicate cannot pass as a match by collapsing into one entry.
        """
        self._check_fitted()
        self._check_features_match(input_values)

        rows = self._as_rows(FeatureSet(input_values))

        return Predictions.already_checked(self._winners_for(rows).astype(np.float64))

    def _updated_weights(
        self,
        weights: FloatArray,
        presented_row: FloatArray,
        winner: int,
        rate: float,
        radius: float,
    ) -> FloatArray:
        """Move every unit toward one presented row, by its share of the step.

        The competitive update, and the whole of what this model learns by. It is
        a *local* rule: the change to a unit depends on that unit's own weights,
        the row in front of it, and how near it sits to the winner on the grid.
        Nothing arrives from a loss computed elsewhere.

        Parameters
        ----------
        weights:
            The map as it stands, ``(n_units, n_features)``. Row ``u`` is unit
            ``u``'s weight vector. Not to be modified in place -- return a new
            block, so that the caller's measurement of how far the epoch moved
            things still has something to compare against.
        presented_row:
            The one row being shown, ``(n_features,)``.
        winner:
            The index of the unit nearest ``presented_row``, already found by
            :meth:`_winner_for`.
        rate:
            What fraction of the remaining gap the winner closes, this epoch's
            value from ``learning_rate``. Between 0 and 1 in ordinary use.
        radius:
            The width of the neighbourhood in grid units, this epoch's value from
            ``neighbourhood_radius``. Hand it straight to
            :meth:`_neighbourhood_at`, which turns it and the winner into the
            per-unit scores; nothing here needs to know the shape of the
            Gaussian, only that a unit's score is what scales its step.

        Returns
        -------
        FloatArray
            The new map, ``(n_units, n_features)``, same shape as ``weights``.

        Notes
        -----
        Let ``neighbourhood = self._neighbourhood_at(winner, radius)``, an
        ``(n_units,)`` block whose entry ``u`` says how much unit ``u`` moves when
        ``winner`` wins. It is exactly 1.0 at ``u == winner`` and falls off with
        distance on the grid. Then, for every unit ``u``:

            new[u] = weights[u] + rate * neighbourhood[u] * (row - weights[u])

        Reading it term by term:

        * ``(row - weights[u])`` is the gap from the unit to the row, a vector
          pointing from where the unit is to where the row is. This is where the
          Euclidean commitment lives -- the whole expression is a step along the
          negative gradient of ``||row - weights[u]||^2 / 2``, which is why no
          other metric can be swapped in without changing the rule rather than
          the measure.
        * ``rate`` shrinks that gap by a fraction. At ``rate == 1`` the unit
          lands exactly on the row and forgets everything it had learned, and at
          ``rate == 0`` nothing happens; the useful range is small and shrinking.
        * ``neighbourhood[u]`` scales the fraction down for units far from the
          winner on the grid. At the winner it is exactly 1.0, so the winner gets
          the full step. Far away it is effectively zero and the unit does not
          move.

        Three things this must **not** do, all of which type-check and run:

        * Move only the winner. That is the zero-radius case, and
          :meth:`_neighbourhood_at` already answers it correctly when the radius
          is zero. Writing it in here instead deletes the only thing that
          distinguishes this model from an online k-means, at every radius.
        * Measure the neighbourhood in the input space. ``radius`` is in grid
          units, and a fall-off computed from ``||weights[u] - weights[winner]||``
          has the right shape and the wrong meaning -- it rewards units that
          already agree instead of imposing the grid's arrangement on them.
        * Update in place. ``fit`` compares the block this returns against the one
          it passed in to measure how far the epoch moved things, so mutating the
          argument makes every epoch report a movement of zero and the first one
          report convergence.
        """
        neighbourhood = self._neighbourhood_at(winner, radius)
        gap = presented_row - weights

        step = rate * neighbourhood[:, None] * gap

        return weights + step

    def _winner_for(self, presented_row: FloatArray, weights: FloatArray) -> int:
        """Which unit's weights are nearest one row.

        The competition step, for a single presentation. Squared Euclidean
        distance, unrooted because the square root is monotonic and the answer is
        an ``argmin``.

        This is a distance in the **input** space, between a row and a weight
        vector, and it is the one the neighbourhood must not be confused with.

        A row equidistant from two units has to go somewhere and either answer is
        correct; ``argmin`` takes the lower index, which is deterministic, which
        is the only property that matters.

        Parameters
        ----------
        presented_row:
            One row, ``(n_features,)``.
        weights:
            The map, ``(n_units, n_features)``.

        Returns
        -------
        int
            The index of the nearest unit.
        """
        gaps = weights - presented_row

        return int(np.argmin(np.einsum("uf,uf->u", gaps, gaps)))

    def _winners_for(self, rows: RowBlock) -> IndexArray:
        """Which unit wins each of many rows at once.

        What ``predict`` uses, where :meth:`_winner_for` is what the fit uses one
        row at a time. Routed through
        :meth:`~oop_ml.core.clustering.centroids.Centroids.squared_distances_to`,
        which is the same expansion k-means and k-nearest neighbours share, exact
        because it shifts rows and prototypes to a common origin first.
        """
        return np.argmin(self.units.prototypes.squared_distances_to(rows), axis=1)

    def _neighbourhood_at(self, winner: int, radius: float) -> FloatArray:
        """How much each unit moves when ``winner`` wins, at one radius.

        The cooperation step. Entry ``u`` is the Gaussian

            exp(-(grid_distance(u, winner) ** 2) / (2 * radius ** 2))

        which is exactly 1.0 at the winner and falls off with distance **on the
        grid**. At ``radius = 1`` an immediate neighbour scores 0.607, a diagonal
        one 0.368, and a unit three steps away 0.011.

        The distance comes from :meth:`GridPosition.distance_to`, which is the
        one method in this module that measures the grid, and which takes a
        ``GridPosition`` and nothing else. A weight vector cannot be handed to
        it, so the trap the module docstring names is closed by the signature
        rather than by remembering.

        A radius of zero is not a special case bolted on, it is the limit of that
        Gaussian as the width goes to nothing: a one at the winner and a zero
        everywhere else. It is written separately only because the expression
        divides by ``radius``, and it earns the attention because it is the
        k-means case the module docstring rests on.

        Parameters
        ----------
        winner:
            The index of the unit that won this row.
        radius:
            This epoch's value from ``neighbourhood_radius``, in grid units.

        Returns
        -------
        FloatArray
            ``(n_units,)``, every entry in ``[0, 1]``, and 1.0 at ``winner``.
        """
        scores = np.zeros(self.n_units, dtype=np.float64)

        if radius <= 0.0:
            scores[winner] = 1.0

            return scores

        winning_position = self._position_at(winner)
        for index in range(self.n_units):
            gap = winning_position.distance_to(self._position_at(index))
            scores[index] = np.exp(-(gap**2) / (2.0 * radius**2))

        return scores

    def _position_at(self, index: int) -> GridPosition:
        """Where the unit at ``index`` sits, in row-major order.

        The single place that turns an index into a place on the grid. The
        neighbourhood reads it during the fit and :meth:`_as_units` reads it when
        the fit is wrapped up, so the arrangement the model learned under and the
        arrangement a caller draws cannot disagree. Two copies of ``index //
        width`` could, and the disagreement would be a map that trained as a
        chain and prints as a square.
        """
        return GridPosition(index // self.grid_width, index % self.grid_width)

    def _seeded_weights(
        self, rows: RowBlock, generator: np.random.Generator
    ) -> FloatArray:
        """Start every unit on a randomly drawn row.

        Drawing rows rather than sampling small random numbers, for the reason
        k-means seeds from rows: a prototype placed where no data is has to
        travel before it can win anything, and a unit that never wins never
        moves under its own competition -- only its grid neighbours can drag it.

        No k-means++ spreading here, and deliberately so. That procedure exists
        to stop two centres landing in one blob, which Lloyd's algorithm cannot
        recover from because a centre only ever moves toward what already chose
        it. A map recovers from exactly that, because the neighbourhood pulls
        units apart whether or not they have won anything, so the spreading would
        be buying insurance against a failure this model does not have.

        The measurement behind that claim is worth keeping, because it also names
        the one configuration where it is false. Sixteen units over the twelve
        blob rows, a hundred epochs: with a decaying radius, not one of the
        sixteen is still sitting on the row it was seeded from. At radius zero,
        thirteen of them are, because a unit that never wins is never moved by
        anything and duplicated draws leave the higher-indexed copy permanently
        dead. So a zero-radius map really does want k-means++ and this one does
        not, which is the same fact the module docstring states from the other
        end.

        Rows are drawn without replacement where there are enough of them, so a
        map smaller than its data starts with distinct prototypes.
        """
        drawn = generator.choice(
            rows.n_rows, size=self.n_units, replace=rows.n_rows < self.n_units
        )

        return rows.values[drawn].astype(np.float64)

    def _as_units(
        self, weights: FloatArray, feature_names: tuple[str, ...]
    ) -> UnitGrid:
        """Wrap a weight matrix as the arranged, named grid a caller reads."""
        return UnitGrid(
            [
                MapUnit(
                    self._position_at(index),
                    Centroid(self.name_for(index), row, feature_names),
                )
                for index, row in enumerate(weights)
            ],
            self.grid_width,
            self.grid_height,
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
        """Raise unless the supplied features are exactly the fitted ones.

        Raises
        ------
        InvalidValuesError
            If a fitted feature is missing or an unknown one is present.
        NonUniqueFeaturesError
            If two supplied features share a name, which is checked first so
            that a duplicate cannot pass as a match by collapsing into one entry.
        """
        names = [feature.name for feature in input_values]

        if len(set(names)) != len(names):
            raise NonUniqueFeaturesError(
                f"feature names must be unique; got {sorted(names)}"
            )

        if set(names) != set(self._feature_names):
            raise InvalidValuesError(
                f"expected exactly the fitted features "
                f"{sorted(self._feature_names)}; got {sorted(names)}"
            )

    @staticmethod
    def name_for(position: int) -> str:
        """The name of the unit at ``position``, counting from zero.

        One place deciding what a unit is called, so the fit and the report
        cannot drift apart about it.
        """
        return f"{UNIT_NAME_PREFIX}_{position + 1}"

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"SelfOrganisingMap({self.grid_height}x{self.grid_width}, unfitted)"

        return (
            f"SelfOrganisingMap({self.grid_height}x{self.grid_width}, "
            f"epochs_run={self.epochs_run}, "
            f"final_epoch_movement={self.final_epoch_movement:.4f})"
        )
