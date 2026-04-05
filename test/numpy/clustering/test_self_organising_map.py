"""Spec for the self-organising map -- red until the competitive update lands.

What carries this file is that almost every claim about a map is a claim about a
*relationship* rather than about a number, and relationships need oracles built
independently of the thing being tested.

They also need one claim that is not a relationship. ``TestTheUpdateRule`` pins
the arithmetic itself against numbers worked out on paper, because three rules
that are not the one in the docstring were measured passing every relationship
in this file -- see that class for which, and why each of them survives.

Nothing here asserts a label. A clusterer's label 0 means "whichever group this
seeding numbered first", so two correct fits can agree completely about the
grouping and disagree about every number, and ``same_partition`` below compares
which rows share a label rather than what the label is. Measured on
``THREE_BLOBS``, the zero-radius map at seed 0 answers ``2 2 2 2 0 0 0 0 1 1 1
1`` where ``KMeans`` answers ``1 1 1 1 2 2 2 2 0 0 0 0``. Same partition, no
label in common.

``topology_correlation`` is the other oracle, and it is written from the
definition rather than from the implementation. It takes every pair of units,
asks the grid how far apart they are and the weights how far apart they are, and
correlates the two lists. It reads the model only through the public accessors a
caller drawing the map would use, and it knows nothing about the Gaussian.

Two fixtures, each shaped for one question
-------------------------------------------
``THREE_BLOBS`` is twelve rows in three groups whose centres sit 10 apart while
their members sit 0.5 from their own centre, so the partition is not a judgement
call and any correct implementation finds it from any seeding. That is what makes
it usable as the comparison against ``KMeans``: on data this separated, the two
differences that survive switching the neighbourhood off -- online versus batch,
a decaying rate versus a jump to the optimum -- have nowhere to bite, so a
disagreement means a bug rather than a coin toss.

``CURVE_CLOUD`` is sixty points along a half circle of radius 5. It is there
because topology preservation needs data with a known one-dimensional shape and
blobs have none: a chain of units laid over an arc either follows it in order or
does not, and the correlation says which.

The controls are the point
---------------------------
Three claims here would be untestable without a control fitted alongside them.

* **The decay.** A decaying rate settling to 0.0045 means nothing on its own,
  since it might be the data. The constant rate on the same fixture, same seed,
  same radius, reaching 1.876 is what makes it a finding.
* **The neighbourhood.** A topology correlation of 0.99 might be an artefact of
  the correlation being asked about eight points. At zero radius the same
  measurement gives -0.038, -0.212, -0.003 and -0.139 at four seeds, which is
  noise around nothing and settles it.
* **The grid geometry.** Fitting the same data on a 1 x 8 and a 2 x 4 grid, with
  every other setting and the seed identical, is the outside test for the module's
  central trap. Those two runs differ only in the grid distances, so an
  implementation measuring the neighbourhood between weight vectors instead makes
  them bit-identical -- measured, a gap of exactly 0.0 where the honest one puts
  4.151 between their furthest-apart weights. The same-grid-twice case is
  asserted beside it, so that "they differ" cannot be passing on noise.

What is asserted about the fit is deliberately loose
------------------------------------------------------
An online map with a decaying rate lands *near* the blob centres and not on them,
where k-means lands on them exactly. Measured at 400 epochs on a 3 x 1 map:
(1.0053, 1.0069), (5.9997, 8.9860) and (10.9957, 1.0071) against true centres of
(1, 1), (6, 9) and (11, 1), a worst coordinate error of 0.014. The tolerance here
is 0.05, which is loose enough for the arithmetic and far tighter than the 0.5
that separates a blob member from its own centre.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.clustering.centroids import Centroid
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    NotFittedError,
)
from oop_ml.core.schedule import (
    ConstantSchedule,
    ExponentialDecaySchedule,
    LinearDecaySchedule,
)
from oop_ml.numpy.clustering.k_means import KMeans
from oop_ml.numpy.clustering.self_organising_map import (
    GridPosition,
    MapUnit,
    SelfOrganisingMap,
    UnitGrid,
)

BLOB_CENTRES = ((1.0, 1.0), (11.0, 1.0), (6.0, 9.0))
"""Three well separated centres, 10 apart, so the grouping is not in doubt."""

BLOB_OFFSETS = ((0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5))
"""Four members per blob, each exactly 0.5 from its centre and averaging to it."""

BLOB_POINTS = [
    (centre[0] + offset[0], centre[1] + offset[1])
    for centre in BLOB_CENTRES
    for offset in BLOB_OFFSETS
]

THREE_BLOBS = [
    Feature("first", [point[0] for point in BLOB_POINTS]),
    Feature("second", [point[1] for point in BLOB_POINTS]),
]
"""Twelve rows in three groups of four, in blob order."""

THREE_BLOBS_GROUPS = np.repeat(np.arange(len(BLOB_CENTRES)), len(BLOB_OFFSETS))
"""Which blob generated each row, for comparing partitions against."""

THREE_BLOBS_ROWS = np.column_stack([feature.values for feature in THREE_BLOBS])

CURVE_ANGLES = np.linspace(0.0, np.pi, 60)

CURVE_CLOUD = [
    Feature("first", 5.0 * np.cos(CURVE_ANGLES)),
    Feature("second", 5.0 * np.sin(CURVE_ANGLES)),
]
"""Sixty points along a half circle of radius 5: known one-dimensional shape."""


def same_partition(left, right) -> bool:
    """Whether two labellings group the rows identically, whatever they call them.

    Written from the definition rather than by renumbering: two rows are together
    under one labelling exactly when they are together under the other. That
    avoids any assumption about which label a fit gave to which group, which is
    the whole reason this exists.
    """
    left_labels = np.asarray(left)
    right_labels = np.asarray(right)

    if left_labels.size != right_labels.size:
        return False

    for position in range(left_labels.size):
        together_left = left_labels == left_labels[position]
        together_right = right_labels == right_labels[position]

        if not np.array_equal(together_left, together_right):
            return False

    return True


def topology_correlation(model: SelfOrganisingMap) -> float:
    """How strongly grid distance predicts weight distance, over every unit pair.

    The independent oracle for topology preservation. It reads the fitted model
    only through the accessors a caller drawing the map would use -- each unit's
    position and each unit's weights -- and it knows nothing about the Gaussian
    that produced them.

    Near 1 means units adjacent on the grid ended up adjacent in the data, which
    is the property the neighbourhood buys. Near 0 means the grid arrangement
    carries no information about the weights, which is what a zero-radius fit
    produces and what any k-means fit would produce.
    """
    units = list(model.units)
    grid_gaps: list[float] = []
    weight_gaps: list[float] = []

    for left in range(len(units)):
        for right in range(left + 1, len(units)):
            grid_gaps.append(units[left].position.distance_to(units[right].position))
            weight_gaps.append(
                float(np.linalg.norm(units[left].weights - units[right].weights))
            )

    return float(np.corrcoef(grid_gaps, weight_gaps)[0, 1])


def units_still_on_a_seeded_row(model: SelfOrganisingMap) -> int:
    """How many units have not moved at all since they were seeded.

    Every unit starts exactly on one of the training rows, so a unit whose
    weights still equal a training row bit for bit has never been moved by
    anything. That is the public, exact way to ask whether units other than the
    winner are being adapted.
    """
    return sum(
        any(np.array_equal(unit.weights, row) for row in THREE_BLOBS_ROWS)
        for unit in model.units
    )


def map_over(features, **overrides) -> SelfOrganisingMap:
    """A fitted map, with settings that every test can then vary one of."""
    settings = {
        "grid_width": 8,
        "grid_height": 1,
        "learning_rate": ExponentialDecaySchedule(start=0.5, end=0.01),
        "neighbourhood_radius": LinearDecaySchedule(start=3.0, end=0.5),
        "max_epochs": 200,
        "random_seed": 0,
    }
    settings.update(overrides)

    return SelfOrganisingMap(**settings).fit(features)


def blob_map(**overrides) -> SelfOrganisingMap:
    """A three-unit chain fitted to the three blobs."""
    settings = {
        "grid_width": 3,
        "grid_height": 1,
        "neighbourhood_radius": LinearDecaySchedule(start=1.5, end=0.1),
        "max_epochs": 400,
    }
    settings.update(overrides)

    return map_over(THREE_BLOBS, **settings)


def centres_found(model: SelfOrganisingMap) -> list[tuple[float, float]]:
    """The learned prototypes, sorted so they can be compared to known centres.

    Sorted because the order the units came out in is an arrangement rather than
    a ranking, and comparing it to a list of centres would be asserting the
    seeding.
    """
    return sorted(
        (unit.weight_for("first"), unit.weight_for("second")) for unit in model.units
    )


def unit_grid(width: int, height: int, **overrides) -> UnitGrid:
    """A hand-built grid, for testing the container rather than the fit."""
    names = overrides.get(
        "names", [SelfOrganisingMap.name_for(index) for index in range(width * height)]
    )
    positions = overrides.get(
        "positions",
        [
            GridPosition(index // width, index % width)
            for index in range(width * height)
        ],
    )

    return UnitGrid(
        [
            MapUnit(
                position,
                Centroid(name, np.array([float(index), 0.0]), ("first", "second")),
            )
            for index, (name, position) in enumerate(zip(names, positions, strict=True))
        ],
        width,
        height,
    )


class TestGridPosition:
    """Where a unit sits, which is the type that keeps the two distances apart."""

    def test_a_position_is_no_distance_from_itself(self) -> None:
        assert GridPosition(2, 3).distance_to(GridPosition(2, 3)) == 0.0

    @pytest.mark.parametrize(
        ("row", "column", "expected"),
        [
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, float(np.sqrt(2.0))),
            (0, 3, 3.0),
            (3, 4, 5.0),
        ],
    )
    def test_it_measures_straight_across_the_rectangle(
        self, row: int, column: int, expected: float
    ) -> None:
        """Euclidean, so the diagonal neighbour is sqrt(2) rather than 1 or 2.

        The alternatives are real choices and this pins the one taken: counting
        steps along the axes would make the diagonal 2, and taking the larger of
        the two would make it 1.
        """
        assert GridPosition(0, 0).distance_to(
            GridPosition(row, column)
        ) == pytest.approx(expected)

    def test_the_measurement_is_symmetric(self) -> None:
        here, there = GridPosition(1, 2), GridPosition(4, 0)

        assert here.distance_to(there) == pytest.approx(there.distance_to(here))

    def test_positions_compare_and_hash_by_their_coordinates(self) -> None:
        assert GridPosition(1, 2) == GridPosition(1, 2)
        assert GridPosition(1, 2) != GridPosition(2, 1)
        assert len({GridPosition(1, 2), GridPosition(1, 2)}) == 1

    def test_comparing_against_something_else_defers_rather_than_raising(self) -> None:
        """``__eq__`` returns ``NotImplemented``; it does not raise it."""
        assert GridPosition(0, 0) != "not a position"

    @pytest.mark.parametrize(("row", "column"), [(-1, 0), (0, -1), (-2, -3)])
    def test_a_negative_coordinate_is_refused(self, row: int, column: int) -> None:
        with pytest.raises(InvalidValuesError):
            GridPosition(row, column)

    @pytest.mark.parametrize("coordinate", [1.5, "1", None, True])
    def test_a_coordinate_that_is_not_a_whole_number_is_refused(
        self, coordinate: object
    ) -> None:
        """A grid has no half rows, and ``True`` is not a row either."""
        with pytest.raises(InvalidValuesError):
            GridPosition(coordinate, 0)  # type: ignore[arg-type]


class TestUnitGrid:
    """The arrangement, and the invariant k-means's centroids deliberately lack."""

    def test_it_reports_its_extents_and_its_count(self) -> None:
        grid = unit_grid(4, 2)

        assert grid.width == 4
        assert grid.height == 2
        assert grid.n_units == 8
        assert len(grid) == 8

    def test_iteration_and_row_major_addressing_agree(self) -> None:
        """Otherwise ``predict``'s index and ``unit_at`` name different units."""
        grid = unit_grid(4, 2)
        listed = list(grid)

        for index, unit in enumerate(listed):
            assert grid.unit_at(index // 4, index % 4) is unit

    def test_it_finds_a_unit_and_its_position_by_name(self) -> None:
        grid = unit_grid(3, 2)

        assert grid.unit_named("unit_5").name == "unit_5"
        assert grid.position_of("unit_5") == GridPosition(1, 1)
        assert "unit_5" in grid
        assert grid["unit_5"].name == "unit_5"

    def test_the_weights_come_out_as_one_matrix_in_index_order(self) -> None:
        grid = unit_grid(3, 2)

        assert grid.weights.shape == (6, 2)
        assert grid.weights[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_the_prototypes_are_the_librarys_named_points(self) -> None:
        """Which is what lets the competition reuse the k-means distance."""
        grid = unit_grid(2, 2)

        assert grid.prototypes.n_clusters == 4
        assert grid.prototypes.feature_names == ("first", "second")

    def test_an_empty_grid_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            UnitGrid([], 1, 1)

    def test_a_count_that_does_not_fill_the_rectangle_is_refused(self) -> None:
        units = list(unit_grid(3, 1))

        with pytest.raises(InvalidValuesError):
            UnitGrid(units, 2, 2)

    def test_positions_out_of_row_major_order_are_refused(self) -> None:
        """The invariant that makes the arrangement readable from an index."""
        scrambled = [
            GridPosition(0, 1),
            GridPosition(0, 0),
            GridPosition(1, 0),
            GridPosition(1, 1),
        ]

        with pytest.raises(InvalidValuesError):
            unit_grid(2, 2, positions=scrambled)

    def test_a_position_outside_the_rectangle_is_refused(self) -> None:
        outside = [GridPosition(0, 0), GridPosition(0, 1), GridPosition(9, 9)]

        with pytest.raises(InvalidValuesError):
            unit_grid(3, 1, positions=outside)

    def test_two_units_sharing_a_name_are_refused(self) -> None:
        with pytest.raises(NonUniqueFeaturesError):
            unit_grid(2, 1, names=["unit_1", "unit_1"])

    @pytest.mark.parametrize(("width", "height"), [(0, 1), (1, 0), (0, 0)])
    def test_an_empty_extent_is_refused(self, width: int, height: int) -> None:
        units = list(unit_grid(2, 1))

        with pytest.raises(InvalidValuesError):
            UnitGrid(units, width, height)

    def test_addressing_outside_the_grid_raises(self) -> None:
        grid = unit_grid(3, 2)

        with pytest.raises(InvalidValuesError):
            grid.unit_at(2, 0)

    def test_an_unknown_unit_name_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            unit_grid(2, 2).unit_named("unit_99")


class TestTheUpdateRule:
    """The one calculation the map learns by, against arithmetic done by hand.

    Every other claim in this file is a claim about a *relationship* -- a
    partition recovered, a correlation, a movement that shrinks -- and that is
    the right shape for almost all of them, because a relationship is what a map
    is for. It is also, measured, not enough on its own. Three rules that are
    not this one satisfy every relationship here:

    * moving each non-winner toward the **winner's weights** rather than toward
      the presented row. It preserves the topology, quantises the blobs and
      settles identically, because dragging a neighbour after the winner and
      dragging it after the row the winner just took look the same from far
      enough away. It passed all 98 of the other assertions.
    * normalising the neighbourhood to sum to one, which silently takes the full
      step away from the winner -- at a radius of 1.0 on a 3 x 1 grid the scores
      sum to 1.7419, so the winner closes 0.287 of the gap where the rule says
      0.5. That is a smaller effective rate, and a smaller rate still organises
      a map.
    * squaring the neighbourhood, which is the same Gaussian at a radius
      narrower by ``sqrt(2)`` and so is caught by nothing that varies the radius.

    So the rule itself is pinned here, straight from the definition,

        new[u] = weights[u] + rate * neighbourhood[u] * (row - weights[u])

    with the expected numbers worked out on paper rather than read off a run.
    """

    THREE_UNITS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    PRESENTED = np.array([4.0, 8.0])

    def test_it_matches_the_rule_worked_out_by_hand(self) -> None:
        """A 3 x 1 grid, unit 0 winning, rate 0.5, radius 1.0.

        Grid distances from unit 0 are 0, 1 and 2, so the Gaussian scores are
        1, ``exp(-1/2) = 0.6065306597`` and ``exp(-4/2) = 0.1353352832``, and
        half of each of those multiplies the gap to ``(4, 8)``:

            unit_1: (0, 0)  + 0.5 * 1.0        * (4, 8)   = (2, 4)
            unit_2: (10, 0) + 0.5 * 0.60653066 * (-6, 8)  = (8.18040802, 2.42612264)
            unit_3: (0, 10) + 0.5 * 0.13533528 * (4, -2)  = (0.27067057, 9.86466472)
        """
        model = SelfOrganisingMap(grid_width=3, grid_height=1)

        moved = model._updated_weights(self.THREE_UNITS, self.PRESENTED, 0, 0.5, 1.0)

        assert np.asarray(moved) == pytest.approx(
            np.array(
                [
                    [2.0, 4.0],
                    [8.18040802, 2.42612264],
                    [0.27067057, 9.86466472],
                ]
            )
        )

    @pytest.mark.parametrize(("grid_width", "radius"), [(3, 1.0), (3, 5.0), (8, 0.25)])
    def test_the_winner_always_closes_exactly_the_rate(
        self, grid_width: int, radius: float
    ) -> None:
        """The neighbourhood is 1.0 at the winner whatever else is going on.

        Neither the radius nor how many units are listening changes what the
        winner itself does, which is what makes ``rate`` mean "the fraction of
        the gap the winner closes" rather than a number scaled by the grid.
        """
        model = SelfOrganisingMap(grid_width=grid_width, grid_height=1)
        weights = np.zeros((grid_width, 2))
        weights[0] = (2.0, 6.0)

        moved = np.asarray(
            model._updated_weights(weights, self.PRESENTED, 0, 0.25, radius)
        )

        assert moved[0] == pytest.approx(np.array([2.5, 6.5]))

    def test_a_neighbour_moves_toward_the_row_and_not_toward_the_winner(self) -> None:
        """The two rules disagree in *sign* here, which is why this fixture.

        Unit 2 sits at 5 between the winner at 0 and the row at 10. Under the
        rule it moves to ``5 + 0.5 * 0.60653066 * 5 = 6.51632665``; under a rule
        dragging it after the winner's weights instead it moves the other way,
        to 3.48367335.
        """
        model = SelfOrganisingMap(grid_width=2, grid_height=1)
        weights = np.array([[0.0, 0.0], [5.0, 0.0]])

        moved = np.asarray(
            model._updated_weights(weights, np.array([10.0, 0.0]), 0, 0.5, 1.0)
        )

        assert moved[1] == pytest.approx(np.array([6.51632665, 0.0]))

    def test_at_zero_radius_only_the_winner_moves(self) -> None:
        """The k-means case, exactly rather than by counting untouched units."""
        model = SelfOrganisingMap(grid_width=3, grid_height=1)

        moved = np.asarray(
            model._updated_weights(self.THREE_UNITS, self.PRESENTED, 1, 0.5, 0.0)
        )

        assert moved[1] == pytest.approx(np.array([7.0, 4.0]))
        assert np.array_equal(moved[0], self.THREE_UNITS[0])
        assert np.array_equal(moved[2], self.THREE_UNITS[2])

    def test_it_leaves_the_block_it_was_handed_alone(self) -> None:
        """Or ``fit`` has nothing left to measure the epoch's movement against."""
        model = SelfOrganisingMap(grid_width=2, grid_height=1)
        weights = np.array([[0.0, 0.0], [5.0, 0.0]])
        before = weights.copy()

        model._updated_weights(weights, np.array([10.0, 0.0]), 0, 0.5, 1.0)

        assert np.array_equal(weights, before)


class TestWhatItLearns:
    """The map itself, on data whose grouping is not in doubt."""

    def test_the_grid_holds_one_unit_per_place(self) -> None:
        model = map_over(CURVE_CLOUD, grid_width=4, grid_height=2)

        assert model.n_units == 8
        assert model.units.n_units == 8
        assert (model.units.width, model.units.height) == (4, 2)

    def test_units_are_named_by_position_from_one(self) -> None:
        assert [unit.name for unit in blob_map().units] == [
            "unit_1",
            "unit_2",
            "unit_3",
        ]

    def test_every_unit_knows_where_it_sits(self) -> None:
        """What a caller draws the map from."""
        model = map_over(CURVE_CLOUD, grid_width=4, grid_height=2)

        assert model.position_of("unit_1") == GridPosition(0, 0)
        assert model.position_of("unit_5") == GridPosition(1, 0)
        assert model.position_of("unit_8") == GridPosition(1, 3)

    def test_weights_are_addressable_by_feature_name(self) -> None:
        """What a prototype is for: a sentence rather than two subscripts."""
        unit = blob_map().units["unit_1"]

        assert isinstance(unit.weight_for("first"), float)
        assert unit.feature_names == ("first", "second")

    def test_it_recovers_the_generating_partition_of_the_blobs(self) -> None:
        labels = np.asarray(blob_map().predict(THREE_BLOBS))

        assert same_partition(labels, THREE_BLOBS_GROUPS)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_it_recovers_that_partition_from_any_seeding(self, seed: int) -> None:
        """Blobs 10 apart leave no room for a bad start to survive."""
        labels = np.asarray(blob_map(random_seed=seed).predict(THREE_BLOBS))

        assert same_partition(labels, THREE_BLOBS_GROUPS)

    def test_the_prototypes_land_near_the_blob_centres(self) -> None:
        """Near, not on. An online walk with a decaying rate stops short.

        Measured at 400 epochs the worst coordinate error is 0.014, against the
        0.5 that separates a blob member from its own centre.
        """
        assert np.asarray(centres_found(blob_map())) == pytest.approx(
            np.asarray(sorted(BLOB_CENTRES)), abs=0.05
        )

    def test_a_map_with_more_units_than_rows_still_fits(self) -> None:
        """Unlike k-means, which refuses, because a unit need never win.

        A group with no rows has no mean and breaks Lloyd's update. A unit with
        no rows is an ordinary outcome here -- its grid neighbours still drag it
        along -- so there is nothing to refuse.
        """
        model = SelfOrganisingMap(grid_width=3, grid_height=3, random_seed=0).fit(
            [Feature("first", [0.0, 1.0]), Feature("second", [0.0, 1.0])]
        )

        assert model.units.n_units == 9


class TestTheDecayIsWhatMakesItConverge:
    """The claim that a constant rate is fatal rather than merely suboptimal.

    Every comparison here holds the radius at 1.0, so the only thing varying is
    the rate and the movement cannot be credited to the neighbourhood shrinking.
    """

    def test_a_constant_rate_leaves_the_map_still_moving_at_the_end(self) -> None:
        """1.876 across a cloud ten wide: a tenth of the picture per epoch."""
        model = map_over(
            CURVE_CLOUD,
            learning_rate=ConstantSchedule(value=0.5),
            neighbourhood_radius=ConstantSchedule(value=1.0),
            max_epochs=30,
        )

        assert model.final_epoch_movement > 0.5
        assert not model.converged

    def test_a_decaying_rate_settles_on_the_same_data(self) -> None:
        """0.0045 where the constant rate reaches 1.876, on the same seed."""
        model = map_over(
            CURVE_CLOUD,
            neighbourhood_radius=ConstantSchedule(value=1.0),
            max_epochs=30,
        )

        assert model.final_epoch_movement < 0.05

    def test_the_constant_rate_does_not_improve_with_a_longer_walk(self) -> None:
        """Which is the finding, and the reason both lengths are measured.

        Ten times the epochs leaves the constant walk moving as far on its last
        pass as before -- 1.876 at thirty epochs, 2.196 at a thousand -- because
        nothing in it is getting smaller. The decaying walk sits two orders of
        magnitude below at both lengths.
        """
        short = map_over(
            CURVE_CLOUD,
            learning_rate=ConstantSchedule(value=0.5),
            neighbourhood_radius=ConstantSchedule(value=1.0),
            max_epochs=30,
        )
        long = map_over(
            CURVE_CLOUD,
            learning_rate=ConstantSchedule(value=0.5),
            neighbourhood_radius=ConstantSchedule(value=1.0),
            max_epochs=300,
        )
        decayed = map_over(
            CURVE_CLOUD,
            neighbourhood_radius=ConstantSchedule(value=1.0),
            max_epochs=300,
        )

        assert long.final_epoch_movement > 0.5
        assert long.final_epoch_movement > 20 * decayed.final_epoch_movement
        assert short.final_epoch_movement > 20 * decayed.final_epoch_movement

    def test_a_rate_of_zero_moves_nothing_and_says_so_on_the_first_epoch(self) -> None:
        """The degenerate control, and the one that pins the pass count.

        A walk that settles immediately must report one pass and not zero. Two of
        the three copies ``ConvergentFit`` replaces got that wrong in exactly
        this case.
        """
        model = blob_map(learning_rate=ConstantSchedule(value=0.0))

        assert model.final_epoch_movement == 0.0
        assert model.converged
        assert model.epochs_run == 1


class TestKMeansIsTheZeroRadiusCase:
    """Switch the neighbourhood off and the rule is an online k-means."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_a_zero_radius_map_groups_the_blobs_as_k_means_does(
        self, seed: int
    ) -> None:
        """The partitions agree row for row, at every seed tried."""
        flat = blob_map(
            neighbourhood_radius=ConstantSchedule(value=0.0), random_seed=seed
        )
        lloyd = KMeans(n_clusters=3, random_seed=0).fit(THREE_BLOBS)

        assert same_partition(
            np.asarray(flat.predict(THREE_BLOBS)),
            np.asarray(lloyd.predict(THREE_BLOBS)),
        )

    def test_agreeing_about_the_grouping_is_not_agreeing_about_the_labels(
        self,
    ) -> None:
        """Which is why every other assertion here goes through a partition.

        Two seeds of the same model, both correct, both finding the same three
        groups, disagreeing about which group is called what.
        """
        first = np.asarray(
            blob_map(
                neighbourhood_radius=ConstantSchedule(value=0.0), random_seed=0
            ).predict(THREE_BLOBS)
        )
        second = np.asarray(
            blob_map(
                neighbourhood_radius=ConstantSchedule(value=0.0), random_seed=2
            ).predict(THREE_BLOBS)
        )

        assert same_partition(first, second)
        assert not np.array_equal(first, second)

    def test_at_zero_radius_only_the_winner_moves(self) -> None:
        """Measured exactly, since a unit that never moved is bit-identical.

        Every unit starts on a training row. With sixteen units over twelve rows
        and no neighbourhood, thirteen of them are never a winner and still sit
        on their seeded row when the walk ends.
        """
        model = map_over(
            THREE_BLOBS,
            grid_width=4,
            grid_height=4,
            neighbourhood_radius=ConstantSchedule(value=0.0),
            max_epochs=100,
        )

        assert units_still_on_a_seeded_row(model) >= 8

    def test_with_a_neighbourhood_every_unit_moves(self) -> None:
        """The other half of the same measurement, and the discriminating one.

        Same grid, same seed, same rate: not one of the sixteen units is still
        where it was seeded, because a winner drags all of them.
        """
        model = map_over(
            THREE_BLOBS,
            grid_width=4,
            grid_height=4,
            neighbourhood_radius=LinearDecaySchedule(start=2.0, end=0.5),
            max_epochs=100,
        )

        assert units_still_on_a_seeded_row(model) == 0


class TestTopologyPreservation:
    """What the neighbourhood buys, which k-means has no way to express."""

    def test_a_chain_of_units_follows_a_one_dimensional_cloud(self) -> None:
        """0.9909 over the 28 pairs of an eight-unit chain."""
        assert topology_correlation(map_over(CURVE_CLOUD)) > 0.9

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_it_finds_the_same_arrangement_from_any_seeding(self, seed: int) -> None:
        """The arrangement is what the map converges to, not what it started at."""
        assert topology_correlation(map_over(CURVE_CLOUD, random_seed=seed)) > 0.9

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_without_a_neighbourhood_the_arrangement_is_absent(self, seed: int) -> None:
        """The control, and the whole point.

        At zero radius the same fixture gives -0.038, -0.212, -0.003 and -0.139
        at these four seeds: noise around nothing, changing sign with the seed,
        because unit 1 and unit 2 are adjacent only in the numbering.
        """
        model = map_over(
            CURVE_CLOUD,
            neighbourhood_radius=ConstantSchedule(value=0.0),
            random_seed=seed,
        )

        assert topology_correlation(model) < 0.5

    def test_a_two_dimensional_grid_over_a_curve_preserves_less(self) -> None:
        """0.9034 against the chain's 0.9909, and correctly so.

        A 2 x 4 grid laid over a one-dimensional cloud has to fold, and the fold
        is exactly the disagreement between the two distances that the
        correlation measures. Still strongly positive, so the arrangement is
        there; measurably weaker, because the grid has a dimension the data
        does not.
        """
        chain = topology_correlation(map_over(CURVE_CLOUD))
        folded = topology_correlation(
            map_over(CURVE_CLOUD, grid_width=4, grid_height=2)
        )

        assert folded > 0.85
        assert chain > folded


class TestTheGridDistanceIsNotTheInputDistance:
    """The module's central trap, kept closed from outside the implementation."""

    def test_the_shape_of_the_grid_changes_the_answer(self) -> None:
        """A 1 x 8 and a 2 x 4 hold eight units and differ in nothing else here.

        Same seed, same schedules, same epochs, same seeded draw, same
        presentation order. The only difference between the two runs is the table
        of grid distances, so a neighbourhood measured between weight vectors
        instead makes them identical -- measured, a gap of exactly 0.0 where this
        puts 4.151 between their furthest-apart weights.
        """
        chain = map_over(CURVE_CLOUD, grid_width=8, grid_height=1, max_epochs=100)
        folded = map_over(CURVE_CLOUD, grid_width=4, grid_height=2, max_epochs=100)

        assert not np.allclose(chain.units.weights, folded.units.weights)

    def test_the_same_grid_twice_is_identical(self) -> None:
        """The control for the test above, so that "they differ" is not noise."""
        first = map_over(CURVE_CLOUD, grid_width=8, grid_height=1, max_epochs=100)
        second = map_over(CURVE_CLOUD, grid_width=8, grid_height=1, max_epochs=100)

        assert np.array_equal(first.units.weights, second.units.weights)


class TestPredicting:
    """Labelling rows, including rows the fit never saw."""

    def test_a_label_is_a_place_on_the_grid(self) -> None:
        """The index ``predict`` answers with addresses ``unit_at`` directly.

        Which is what makes a prediction drawable: label 6 on a 4 x 4 map is the
        unit at row 1, column 2, and that unit really is the nearest one.
        """
        model = map_over(THREE_BLOBS, grid_width=4, grid_height=4, max_epochs=100)
        labels = np.asarray(model.predict(THREE_BLOBS)).astype(int)

        for position, label in enumerate(labels):
            addressed = model.units.unit_at(label // 4, label % 4)
            gaps = np.linalg.norm(
                model.units.weights - THREE_BLOBS_ROWS[position], axis=1
            )

            assert np.linalg.norm(
                addressed.weights - THREE_BLOBS_ROWS[position]
            ) == pytest.approx(gaps.min())

    def test_labels_are_whole_positions_inside_the_grid(self) -> None:
        values = np.asarray(blob_map().predict(THREE_BLOBS))

        assert np.array_equal(values, np.floor(values))
        assert values.min() >= 0
        assert values.max() <= 2

    def test_a_new_row_falls_to_its_nearest_unit(self) -> None:
        """Nothing is relearned: the weights stay where ``fit`` left them."""
        model = blob_map()
        near_first_blob = [Feature("first", [1.1]), Feature("second", [0.9])]

        assert int(np.asarray(model.predict(near_first_blob))[0]) == int(
            np.asarray(model.predict(THREE_BLOBS))[0]
        )

    def test_column_order_does_not_matter(self) -> None:
        model = blob_map()
        first, second = THREE_BLOBS

        assert np.array_equal(
            np.asarray(model.predict([first, second])),
            np.asarray(model.predict([second, first])),
        )

    def test_fit_predict_matches_fitting_then_predicting(self) -> None:
        together = SelfOrganisingMap(
            grid_width=3,
            grid_height=1,
            neighbourhood_radius=LinearDecaySchedule(start=1.5, end=0.1),
            max_epochs=400,
            random_seed=0,
        ).fit_predict(THREE_BLOBS)

        assert np.array_equal(
            np.asarray(together), np.asarray(blob_map().predict(THREE_BLOBS))
        )


class TestConvergenceBookkeeping:
    """What the walk reports about itself, which ``ConvergentFit`` owns."""

    def test_a_walk_that_runs_out_of_epochs_says_so(self) -> None:
        """The default tolerance of 1e-8 is far below what an online walk reaches.

        That is honest rather than a misconfiguration, and it is why
        ``final_epoch_movement`` is public beside ``converged``.
        """
        model = map_over(CURVE_CLOUD, max_epochs=30)

        assert model.epochs_run == 30
        assert not model.converged

    def test_a_looser_tolerance_stops_the_walk_early(self) -> None:
        """86 epochs of a possible 200 at a tolerance of 0.1."""
        model = map_over(CURVE_CLOUD, tolerance=0.1)

        assert model.converged
        assert model.epochs_run < 200

    def test_the_reported_movement_is_below_the_tolerance_that_stopped_it(
        self,
    ) -> None:
        model = map_over(CURVE_CLOUD, tolerance=0.1)

        assert model.final_epoch_movement < 0.1

    def test_the_same_seed_gives_the_same_fit(self) -> None:
        """Both the seeded weights and the presentation order come from it."""
        assert np.array_equal(
            map_over(CURVE_CLOUD, random_seed=7).units.weights,
            map_over(CURVE_CLOUD, random_seed=7).units.weights,
        )

    def test_a_different_seed_gives_a_different_fit(self) -> None:
        """Otherwise the seed is not reaching the generator at all."""
        assert not np.allclose(
            map_over(CURVE_CLOUD, random_seed=1).units.weights,
            map_over(CURVE_CLOUD, random_seed=2).units.weights,
        )

    def test_a_single_epoch_walk_is_allowed(self) -> None:
        """The schedules are asked for pass 1 of 1, which is their start value."""
        model = blob_map(max_epochs=1)

        assert model.epochs_run == 1


class TestWhatItRefuses:
    """Guards, each raising from the MLLibError hierarchy or from pydantic."""

    @pytest.mark.parametrize(
        "attribute", ["units", "epochs_run", "converged", "final_epoch_movement"]
    )
    def test_reading_a_learned_attribute_before_fitting_raises(
        self, attribute: str
    ) -> None:
        with pytest.raises(NotFittedError):
            getattr(SelfOrganisingMap(), attribute)

    def test_predicting_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            SelfOrganisingMap().predict(THREE_BLOBS)

    def test_asking_where_a_unit_sits_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            SelfOrganisingMap().position_of("unit_1")

    def test_fitting_with_no_features_raises(self) -> None:
        with pytest.raises(EmptyValuesError):
            SelfOrganisingMap().fit([])

    def test_duplicate_feature_names_are_rejected(self) -> None:
        with pytest.raises(NonUniqueFeaturesError):
            SelfOrganisingMap().fit(
                [Feature("same", [1.0, 2.0, 3.0]), Feature("same", [4.0, 5.0, 6.0])]
            )

    def test_features_of_different_lengths_are_rejected(self) -> None:
        with pytest.raises(NonEqualArrayLengthError):
            SelfOrganisingMap().fit(
                [Feature("first", [1.0, 2.0]), Feature("second", [1.0, 2.0, 3.0])]
            )

    def test_predicting_without_every_fitted_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            blob_map().predict([THREE_BLOBS[0]])

    def test_predicting_with_an_unknown_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            blob_map().predict([*THREE_BLOBS, Feature("extra", [1.0] * 12)])

    def test_predicting_with_a_duplicated_feature_raises(self) -> None:
        """Checked before the name comparison, or a duplicate passes as a match."""
        with pytest.raises(NonUniqueFeaturesError):
            blob_map().predict([THREE_BLOBS[0], THREE_BLOBS[1], THREE_BLOBS[0]])

    def test_asking_where_an_unknown_unit_sits_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            blob_map().position_of("unit_99")

    @pytest.mark.parametrize(
        "settings",
        [
            {"grid_width": 0},
            {"grid_height": 0},
            {"grid_width": -1},
            {"max_epochs": 0},
            {"tolerance": 0.0},
            {"tolerance": -1.0},
        ],
    )
    def test_an_impossible_configuration_is_refused_at_construction(
        self, settings: dict[str, Any]
    ) -> None:
        """Field bounds are pydantic's to enforce, so the error is pydantic's."""
        with pytest.raises(ValidationError):
            SelfOrganisingMap(**settings)

    def test_an_unknown_keyword_is_refused(self) -> None:
        """``extra="forbid"``: a misspelling must not run on the default."""
        with pytest.raises(ValidationError):
            SelfOrganisingMap(gird_width=4)  # pyright: ignore[reportCallIssue]
