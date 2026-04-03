"""Spec for Hebbian principal components -- red until ``_updated_weights`` lands.

The claim this file exists to check is that a rule reading two numbers at a time
arrives where an eigendecomposition arrives.
``TestItAgreesWithPrincipalComponentAnalysis`` is therefore the centre of the
file, and everything else is either the machinery that claim rests on or a
demonstration of what breaks it.

``TILTED_GRID`` is a fixture whose decomposition was written down before any
code existed. It is the eight combinations of a coordinate ``along`` in
``{-2, -1, 1, 2}`` with a coordinate ``across`` in ``{-1, 1}``, laid onto the
perpendicular pair ``(0.6, 0.8)`` and ``(0.8, -0.6)`` so that every stored
number comes out to one decimal place. The variance along the first is
``20 / 7`` and along the second ``8 / 7``, the two sum to exactly 4.0, and the
shares are exactly ``5 / 7`` and ``2 / 7``.

Its means are 10 and 100 rather than 0, and far apart on purpose.
``test_it_centres_before_walking`` is what fails if the centring is skipped,
and it fails by pointing the first direction at ``(0.0995, 0.995)``, which is
the mean vector rather than anything about the shape.

**No test asserts a sign.** A direction and its negation are the same
direction, and which one a walk lands on depends on the random start. Every
assertion here compares absolute values or a quantity that is sign-free.

Three thresholds, and where they come from
-------------------------------------------
Every tolerance below was measured rather than guessed, over ten seeds on
``TILTED_GRID`` and three on ``STREAM``.

The agreement with PCA is asserted as ``1 - |cos| < 1e-05``. Measured, the worst
over ten seeds on ``TILTED_GRID`` is 2.9e-07 and on ``STREAM`` 2.6e-06, so there
is between four and thirty times of room beneath the threshold, and a fit that
found the wrong direction misses it by a whole number rather than by a factor.

A loading is compared entry by entry to within 5e-03, which is two orders of
magnitude looser than the cosine and is not an inconsistency. A cosine is flat
at its maximum, so an angular error of ``theta`` shows up in the cosine as
``theta^2 / 2`` and in the entries as ``theta`` itself. The measured worst entry
gap over ten seeds is 6.0e-04, and the first draft of this file asserted 1e-04
and failed on a correct fit -- which is what the satisfiability run is for.

The learned lengths are asserted within 0.02 of 1. Measured, the worst deviation
is 1.2e-03 on ``TILTED_GRID`` and 1.7e-03 on ``STREAM``. The break that test
exists for is the unnormalised rule, whose length reaches 2.48 after a single
epoch and 8790 after ten, so the discrimination is four orders of magnitude
rather than a near thing.

The worst orthogonality is asserted below 0.01, and measured it runs 7.8e-05 to
8.3e-04 on ``TILTED_GRID`` and 5.7e-04 to 2.4e-03 on ``STREAM``. That is also
the number ``test_the_directions_are_not_exactly_perpendicular`` reads in the
other direction, because
:class:`~oop_ml.core.decomposition.components.PrincipalComponents` refuses
anything above 1e-08 and a spec that quietly implied this family could satisfy
that would be advertising.

The hand-worked step is the sharpest test here
-----------------------------------------------
``TestTheUpdateRule`` calls ``_updated_weights`` directly with weights, a row
and a rate chosen so that all three plausible wrong versions of the deflation
sum give visibly different answers. Two components over three features, weights
``[[1, 0, 0], [0, 1, 0]]``, row ``(2, 3, 4)``, rate 0.1:

* correct, summing ``j`` from 0 to ``i`` inclusive, gives
  ``[[1, 0.6, 0.8], [0, 1, 1.2]]``;
* summing ``j < i`` instead, which drops Oja's normalisation, gives
  ``[[1.4, 0.6, 0.8], [0, 1.9, 1.2]]``;
* summing only ``j == i``, which is Oja per component with no deflation at all,
  gives ``[[1, 0.6, 0.8], [0.6, 1, 1.2]]``.

Every entry that differs, differs by at least 0.4, so nothing here rests on a
tolerance. That one case separates all three, which is why it is written out by
hand rather than left to the fits to catch by their symptoms.

``STREAM`` is generated rather than written out
------------------------------------------------
Three latent columns of known relative spread put through a random orthogonal
rotation, from a pinned generator, so it is reproducible without being a wall of
literals. It exists because ``TILTED_GRID`` has only two features and cannot
show a middle component being deflated by one direction and deflating another.
It is fitted once and cached, because 150 rows over 80 epochs is 12000 Python
updates and takes about 0.24 seconds.
"""

from functools import cache

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.data.coefficients import Coefficient, Coefficients
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    AllSameValuesError,
    DivergenceError,
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    NotFittedError,
    TooFewValuesError,
)
from oop_ml.core.schedule import ConstantSchedule, LinearDecaySchedule
from oop_ml.decomposition.hebbian_principal_components import (
    HebbianDirection,
    HebbianDirections,
    HebbianPrincipalComponents,
)
from oop_ml.decomposition.principal_component_analysis import (
    PrincipalComponentAnalysis,
)

# Eight rows, laid out as along * (0.6, 0.8) + across * (0.8, -0.6) and then
# shifted to means of 10 and 100. Every entry is exact at one decimal place.
TILTED_GRID_ROWS = np.array(
    [
        [8.0, 99.0],
        [9.6, 97.8],
        [8.6, 99.8],
        [10.2, 98.6],
        [9.8, 101.4],
        [11.4, 100.2],
        [10.4, 102.2],
        [12.0, 101.0],
    ]
)

TILTED_GRID_MEANS = (10.0, 100.0)
TILTED_GRID_VARIANCES = (20.0 / 7.0, 8.0 / 7.0)
TILTED_GRID_TOTAL_VARIANCE = 4.0
TILTED_GRID_SHARES = (5.0 / 7.0, 2.0 / 7.0)

# The two directions the data was built along, as magnitudes. The signed pair is
# (0.6, 0.8) and (0.8, -0.6), and the second entry is stored positive because
# every comparison here is against an absolute value -- a direction and its
# negation are the same direction, and which one a walk lands on is the seed's
# business. Writing the signed pair here and comparing it against an absolute
# value is the mistake the satisfiability run caught.
TILTED_GRID_DIRECTION_MAGNITUDES = ((0.6, 0.8), (0.8, 0.6))

# What an uncentred walk finds instead: the mean vector, normalised. Quoted so
# the centring test says what it is discriminating against.
TILTED_GRID_MEAN_DIRECTION = (0.09950371902099893, 0.9950371902099892)

MAXIMUM_ANGLE_DISAGREEMENT = 1e-05
MAXIMUM_LOADING_GAP = 5e-03
MAXIMUM_LENGTH_DEVIATION = 0.02
MAXIMUM_ORTHOGONALITY = 0.01
MAXIMUM_VARIANCE_GAP = 1e-04


def tilted_grid() -> list[Feature]:
    """The hand-computed fixture as named columns."""
    return [
        Feature("first", TILTED_GRID_ROWS[:, 0]),
        Feature("second", TILTED_GRID_ROWS[:, 1]),
    ]


def stream_rows(n_rows: int = 150, seed: int = 7) -> np.ndarray:
    """Three columns of known relative spread, rotated into general position."""
    generator = np.random.default_rng(seed)
    latent = generator.normal(0.0, 1.0, (n_rows, 3)) * np.array([2.0, 1.0, 0.4])
    rotation = np.linalg.qr(generator.normal(0.0, 1.0, (3, 3)))[0]

    return latent @ rotation.T + np.array([1.0, -3.0, 6.0])


def stream() -> list[Feature]:
    """The wider fixture as named columns."""
    rows = stream_rows()

    return [Feature(f"stream_{position}", rows[:, position]) for position in range(3)]


def fitted(n_components: int = 2, **overrides) -> HebbianPrincipalComponents:
    """A model fitted to the hand-computed fixture, seeded so it repeats."""
    return HebbianPrincipalComponents(
        n_components=n_components, random_seed=0, **overrides
    ).fit(tilted_grid())


@cache
def fitted_stream() -> HebbianPrincipalComponents:
    """A three-component fit of the wider fixture, built once and reused.

    Cached because it is twelve thousand Python updates and nothing reads it
    destructively.
    """
    return HebbianPrincipalComponents(n_components=3, max_epochs=80, random_seed=0).fit(
        stream()
    )


@cache
def tilted_grid_reference() -> PrincipalComponentAnalysis:
    """The eigendecomposition of the hand-computed fixture, as the oracle."""
    return PrincipalComponentAnalysis().fit(tilted_grid())


@cache
def stream_reference() -> PrincipalComponentAnalysis:
    """The eigendecomposition of the wider fixture, as the oracle."""
    return PrincipalComponentAnalysis().fit(stream())


def absolute_cosines(
    model: HebbianPrincipalComponents, oracle: PrincipalComponentAnalysis
) -> list[float]:
    """How closely each learned direction lines up with its eigenvector.

    One for one in order, by absolute value, since a direction and its negation
    are the same direction. Exactly 1.0 is perfect agreement.
    """
    learned = model.directions.directions
    known = oracle.components.directions

    return [
        abs(float(learned[position] @ known[position]))
        for position in range(learned.shape[0])
    ]


def centred_tilted_grid() -> np.ndarray:
    """The fixture with its known means removed, for the oracles below."""
    return TILTED_GRID_ROWS - np.array(TILTED_GRID_MEANS)


class TestTheUpdateRule:
    """One step of Sanger's rule, worked by hand and written down.

    The sharpest tests in the file, because the three ways of getting the
    deflation sum wrong all still run, all still produce weights of the right
    shape, and are separated here by whole numbers rather than by tolerances.
    """

    @staticmethod
    def stepped() -> np.ndarray:
        """Two components over three features, one step, rate 0.1."""
        model = HebbianPrincipalComponents(n_components=2)

        return model._updated_weights(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([2.0, 3.0, 4.0]),
            0.1,
        )

    def test_matches_the_hand_worked_step(self) -> None:
        """Outputs are (2, 3), and each component subtracts what is explained.

        Component 0 has explained ``2 * (1, 0, 0)``, so it sees the residual
        ``(0, 3, 4)`` and moves by ``0.1 * 2 * (0, 3, 4)``. Component 1 has
        ``2 * (1, 0, 0) + 3 * (0, 1, 0)`` explained between them, so it sees
        ``(0, 0, 4)`` and moves by ``0.1 * 3 * (0, 0, 4)``.
        """
        assert self.stepped() == pytest.approx(
            np.array([[1.0, 0.6, 0.8], [0.0, 1.0, 1.2]])
        )

    def test_the_sum_includes_the_component_itself(self) -> None:
        """Summing ``j < i`` gives ``[[1.4, 0.6, 0.8], [0, 1.9, 1.2]]``.

        Which is to say component 0 loses Oja's normalisation and becomes plain
        Hebb. The entries differ by 0.4 and 0.9, so nothing here is a rounding
        question.
        """
        assert not np.allclose(
            self.stepped(), np.array([[1.4, 0.6, 0.8], [0.0, 1.9, 1.2]])
        )

    def test_the_sum_includes_the_earlier_components(self) -> None:
        """Summing only ``j == i`` gives ``[[1, 0.6, 0.8], [0.6, 1, 1.2]]``.

        That is Oja's rule per component with no deflation between them, which
        is the version where every component finds the same direction.
        """
        assert not np.allclose(
            self.stepped(), np.array([[1.0, 0.6, 0.8], [0.6, 1.0, 1.2]])
        )

    def test_one_component_is_ojas_rule_exactly(self) -> None:
        """At ``n_components == 1`` the sum has one term and it is the own one.

        Output is 3, the explained part is ``3 * (1, 0)``, the residual is
        ``(0, 4)``, and the move is ``0.1 * 3 * (0, 4)``.
        """
        model = HebbianPrincipalComponents(n_components=1)
        stepped = model._updated_weights(
            np.array([[1.0, 0.0]]), np.array([3.0, 4.0]), 0.1
        )

        assert stepped == pytest.approx(np.array([[1.0, 1.2]]))

    def test_a_row_of_zeros_moves_nothing(self) -> None:
        """Every term is multiplied by an output, and a centred zero row has none."""
        model = HebbianPrincipalComponents(n_components=2)
        weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        stepped = model._updated_weights(weights, np.zeros(3), 0.5)

        assert stepped == pytest.approx(weights)

    def test_a_rate_of_zero_moves_nothing(self) -> None:
        """The whole update is scaled by the rate, so a schedule may reach zero."""
        model = HebbianPrincipalComponents(n_components=2)
        weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        stepped = model._updated_weights(weights, np.array([2.0, 3.0, 4.0]), 0.0)

        assert stepped == pytest.approx(weights)

    def test_it_does_not_write_into_the_weights_it_was_given(self) -> None:
        """The epoch loop keeps the previous pass's copy to measure movement.

        Update the caller's array in place and that difference is zero, so the
        walk reports itself converged on its first pass whatever it did.
        """
        model = HebbianPrincipalComponents(n_components=2)
        weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        model._updated_weights(weights, np.array([2.0, 3.0, 4.0]), 0.1)

        assert weights == pytest.approx(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def test_it_answers_with_the_shape_it_was_given(self) -> None:
        model = HebbianPrincipalComponents(n_components=3)
        stepped = model._updated_weights(np.eye(3), np.array([1.0, 2.0, 3.0]), 0.05)

        assert stepped.shape == (3, 3)


class TestWhatItLearns:
    """The directions themselves, against numbers known before the code existed."""

    def test_learns_one_direction_per_component(self) -> None:
        directions = fitted().directions

        assert directions.n_components == 2
        assert directions.n_features == 2

    def test_the_first_direction_lies_along_the_spread(self) -> None:
        """The cloud is stretched along (0.6, 0.8), which is where this lands."""
        first = fitted().directions["component_1"]

        assert np.abs(first.direction) == pytest.approx(
            np.array(TILTED_GRID_DIRECTION_MAGNITUDES[0]), abs=MAXIMUM_LOADING_GAP
        )

    def test_the_second_direction_lies_across_it(self) -> None:
        second = fitted().directions["component_2"]

        assert np.abs(second.direction) == pytest.approx(
            np.array(TILTED_GRID_DIRECTION_MAGNITUDES[1]), abs=MAXIMUM_LOADING_GAP
        )

    def test_variances_are_the_eigenvalues(self) -> None:
        """20 / 7 and 8 / 7, reached without ever forming a covariance matrix."""
        variances = [direction.variance for direction in fitted().directions]

        assert variances == pytest.approx(
            list(TILTED_GRID_VARIANCES), abs=MAXIMUM_VARIANCE_GAP
        )

    def test_the_total_variance_is_the_sum_over_the_columns(self) -> None:
        """Exactly 4.0 for this fixture, and the denominator of every share."""
        assert fitted().directions.total_variance == pytest.approx(
            TILTED_GRID_TOTAL_VARIANCE
        )

    def test_reports_the_share_of_variance_explained(self) -> None:
        directions = fitted().directions

        assert directions.variance_shares == pytest.approx(
            TILTED_GRID_SHARES, abs=MAXIMUM_VARIANCE_GAP
        )
        assert directions.cumulative_shares[-1] == pytest.approx(
            1.0, abs=MAXIMUM_VARIANCE_GAP
        )

    def test_the_components_come_out_ordered(self) -> None:
        """Deflation intends this, and nothing sorts them afterwards."""
        assert fitted().directions.is_ordered

    def test_names_the_directions_by_position(self) -> None:
        assert [direction.name for direction in fitted().directions] == [
            "component_1",
            "component_2",
        ]

    def test_weights_are_addressable_by_feature_name(self) -> None:
        """What binding a weight to a name is for: a sentence, not a subscript."""
        first = fitted().directions["component_1"]

        assert isinstance(first.weight_for("first"), float)
        assert first.feature_names == ("first", "second")

    def test_keeping_fewer_directions_than_features_reports_the_gap(self) -> None:
        """Two of three, so the cumulative share stops short of 1 and says so."""
        model = HebbianPrincipalComponents(
            n_components=2, max_epochs=80, random_seed=0
        ).fit(stream())

        assert model.directions.n_components == 2
        assert model.directions.cumulative_shares[-1] < 0.99


class TestSelfNormalisation:
    """Oja's subtracted term, which is why the length settles rather than grows."""

    def test_the_learned_vectors_end_up_near_unit_length(self) -> None:
        """Nothing in the model divides by a norm. This is the rule doing it."""
        for length in fitted().directions.lengths:
            assert abs(length - 1.0) < MAXIMUM_LENGTH_DEVIATION

    def test_the_unnormalised_rule_explodes_on_the_same_data(self) -> None:
        """The oracle here is plain Hebbian learning, written out independently.

        Same fixture, same rate, same starting vector, and the only difference
        is the subtracted term. Measured, the length reaches 2.48 after one
        epoch, 93.0 after five and 8790 after ten, against the model's 1.0003.
        A spec that only asserted the model's length would not be saying what
        that length is evidence of.
        """
        centred = centred_tilted_grid()
        weights = np.array([[0.6, 0.8]])

        for epoch in range(1, 11):
            for position in np.random.default_rng(epoch).permutation(8):
                row = centred[position]
                weights = weights + 0.05 * np.outer(weights @ row, row)

        assert float(np.linalg.norm(weights)) > 100.0
        assert max(abs(length - 1.0) for length in fitted().directions.lengths) < 0.01

    def test_the_direction_is_exactly_unit_whatever_the_vector_is(self) -> None:
        """``direction`` normalises on the way out, so a projection is a distance."""
        for direction in fitted().directions:
            assert float(np.linalg.norm(direction.direction)) == pytest.approx(1.0)

    def test_the_length_is_kept_rather_than_normalised_away(self) -> None:
        """It is the convergence diagnostic, so it must survive to be read.

        A model that stored the normalised vector would report every length as
        exactly 1.0 and lose the only number saying whether this component
        arrived.
        """
        first = fitted().directions["component_1"]

        assert first.length != 1.0
        assert float(np.linalg.norm(first.vector)) == pytest.approx(first.length)


class TestDeflation:
    """The part of the sum that makes several components differ from one."""

    def test_the_directions_come_out_nearly_perpendicular(self) -> None:
        """Nothing orthogonalises them. Subtracting what is explained does it."""
        assert fitted().directions.worst_orthogonality < MAXIMUM_ORTHOGONALITY

    def test_the_directions_are_not_exactly_perpendicular(self) -> None:
        """Which is why this family cannot reuse the PCA value objects.

        ``PrincipalComponents`` refuses a pair whose dot product exceeds 1e-08.
        Measured on this fixture at this seed the pair sits at 2.4e-04, four
        orders of magnitude outside it, and that is a correct fit.
        """
        assert fitted().directions.worst_orthogonality > 1e-08

    def test_the_second_component_is_not_the_first(self) -> None:
        """Without deflation both solve the same problem and find one direction."""
        learned = fitted().directions.directions

        assert abs(float(learned[0] @ learned[1])) < 0.01

    def test_a_middle_component_is_deflated_and_deflates(self) -> None:
        """Three components, so the middle one is on both sides of the sum."""
        directions = fitted_stream().directions

        assert directions.n_components == 3
        assert directions.worst_orthogonality < MAXIMUM_ORTHOGONALITY
        assert directions.is_ordered

    def test_the_first_component_is_the_same_whether_or_not_others_follow(
        self,
    ) -> None:
        """Nothing later feeds back into it, because the sum runs one way only."""
        alone = HebbianPrincipalComponents(n_components=1, random_seed=0).fit(
            tilted_grid()
        )
        leading = fitted().directions["component_1"]

        assert abs(
            float(alone.directions["component_1"].direction @ leading.direction)
        ) == pytest.approx(1.0, abs=MAXIMUM_ANGLE_DISAGREEMENT)


class TestItAgreesWithPrincipalComponentAnalysis:
    """The claim the module exists for, checked against the other route.

    One model eigendecomposes a covariance matrix and the other never builds
    one, and they land in the same place. Compared by absolute cosine, since a
    direction and its negation are the same direction.
    """

    def test_the_directions_agree_up_to_sign(self) -> None:
        """Measured, ``1 - |cos|`` is 4.9e-09 and 5.7e-08 at this seed."""
        for cosine in absolute_cosines(fitted(), tilted_grid_reference()):
            assert cosine == pytest.approx(1.0, abs=MAXIMUM_ANGLE_DISAGREEMENT)

    def test_the_variances_agree(self) -> None:
        """The eigenvalues, reached by projecting rather than by solving."""
        model = fitted()
        oracle = tilted_grid_reference()

        assert [direction.variance for direction in model.directions] == pytest.approx(
            [component.variance for component in oracle.components],
            abs=MAXIMUM_VARIANCE_GAP,
        )

    def test_the_shares_agree(self) -> None:
        model = fitted()
        oracle = tilted_grid_reference()

        assert model.directions.variance_shares == pytest.approx(
            oracle.components.variance_shares, abs=MAXIMUM_VARIANCE_GAP
        )

    def test_they_agree_on_a_wider_fixture_too(self) -> None:
        """Two features cannot show a middle component, and three can."""
        for cosine in absolute_cosines(fitted_stream(), stream_reference()):
            assert cosine == pytest.approx(1.0, abs=MAXIMUM_ANGLE_DISAGREEMENT)

    def test_the_agreement_does_not_depend_on_the_seed(self) -> None:
        """A different start is a different walk arriving at the same place."""
        oracle = tilted_grid_reference()

        for seed in range(5):
            model = HebbianPrincipalComponents(n_components=2, random_seed=seed).fit(
                tilted_grid()
            )

            for cosine in absolute_cosines(model, oracle):
                assert cosine == pytest.approx(1.0, abs=MAXIMUM_ANGLE_DISAGREEMENT)

    def test_both_models_name_their_output_columns_the_same(self) -> None:
        """So a caller can swap one for the other without renaming anything."""
        assert fitted().component_names() == tilted_grid_reference().component_names()


class TestCentring:
    """Learned during ``fit``, applied unchanged in ``transform``."""

    def test_it_centres_before_walking(self) -> None:
        """The means are 10 and 100, so skipping this is unmissable.

        Uncentred, the update is pulled towards wherever the cloud sits, and
        the first direction comes out at (0.0995, 0.995) -- the mean vector --
        rather than at (0.6, 0.8).
        """
        first = fitted().directions["component_1"]

        assert not np.allclose(
            np.abs(first.direction), np.array(TILTED_GRID_MEAN_DIRECTION), atol=0.01
        )
        assert np.abs(first.direction) == pytest.approx(
            np.array(TILTED_GRID_DIRECTION_MAGNITUDES[0]), abs=MAXIMUM_LOADING_GAP
        )

    def test_transforming_uses_the_fitted_means_not_the_new_rows(self) -> None:
        """Re-centring held-out rows on themselves is the leak the split prevents.

        One row whose own mean is itself must not transform to the origin.
        """
        model = fitted()
        single = [Feature("first", [12.0]), Feature("second", [101.0])]
        coordinates = np.column_stack(
            [feature.values for feature in model.transform(single)]
        )

        assert float(np.abs(coordinates).max()) > 1.0


class TestTransforming:
    """Projecting rows onto what the walk learned."""

    def test_produces_one_column_per_direction(self) -> None:
        transformed = fitted().transform(tilted_grid())

        assert [feature.name for feature in transformed] == [
            "component_1",
            "component_2",
        ]
        assert all(feature.n_samples == 8 for feature in transformed)

    def test_the_coordinates_are_the_projection_onto_the_unit_directions(
        self,
    ) -> None:
        """The oracle is written from the definition, independently of the model."""
        model = fitted()
        expected = centred_tilted_grid() @ model.directions.directions.T
        produced = np.column_stack(
            [feature.values for feature in model.transform(tilted_grid())]
        )

        assert produced == pytest.approx(expected)

    def test_each_column_has_the_variance_the_direction_reports(self) -> None:
        """Which is what makes the reported variance an eigenvalue and not a label."""
        model = fitted()
        produced = np.column_stack(
            [feature.values for feature in model.transform(tilted_grid())]
        )

        assert np.var(produced, axis=0, ddof=1) == pytest.approx(
            [direction.variance for direction in model.directions]
        )

    def test_column_order_does_not_matter(self) -> None:
        model = fitted()
        first, second = tilted_grid()

        forwards = np.column_stack(
            [feature.values for feature in model.transform([first, second])]
        )
        backwards = np.column_stack(
            [feature.values for feature in model.transform([second, first])]
        )

        assert forwards == pytest.approx(backwards)

    def test_it_works_on_rows_the_fit_never_saw(self) -> None:
        model = fitted()
        fresh = [Feature("first", [10.6, 9.4]), Feature("second", [100.8, 99.2])]

        assert [feature.n_samples for feature in model.transform(fresh)] == [2, 2]

    def test_fit_transform_matches_fitting_then_transforming(self) -> None:
        together = HebbianPrincipalComponents(
            n_components=2, random_seed=0
        ).fit_transform(tilted_grid())
        apart = fitted().transform(tilted_grid())

        for one, other in zip(together, apart, strict=True):
            assert one.name == other.name
            assert one.values == pytest.approx(other.values)


class TestTheWalk:
    """Epochs, convergence, and what ``converged`` does and does not promise."""

    def test_it_reports_how_many_passes_it_took(self) -> None:
        model = fitted()

        assert 1 <= model.epochs_run <= model.max_epochs

    def test_a_capped_walk_runs_every_pass_and_says_it_did_not_settle(self) -> None:
        """The default tolerance is tighter than 200 epochs reaches on this data."""
        model = fitted()

        assert model.epochs_run == 200
        assert not model.converged

    def test_a_looser_tolerance_settles_inside_the_cap(self) -> None:
        """Measured, 1e-04 over a cap of 400 settles on epoch 185."""
        model = fitted(tolerance=1e-04, max_epochs=400)

        assert model.converged
        assert model.epochs_run < 400

    def test_a_single_epoch_runs_exactly_one_pass(self) -> None:
        """The cap is also the schedule's denominator, so this is a legal walk."""
        model = fitted(max_epochs=1)

        assert model.epochs_run == 1

    def test_more_epochs_do_not_move_the_answer_further_from_pca(self) -> None:
        """A longer walk is a better approximation, which is the whole shape of it."""
        oracle = tilted_grid_reference()
        short = min(absolute_cosines(fitted(max_epochs=25), oracle))
        long = min(absolute_cosines(fitted(max_epochs=200), oracle))

        assert long >= short

    def test_a_decaying_rate_is_not_required_to_be_exponential(self) -> None:
        """Any schedule is accepted, and a linear one reaches the same place."""
        model = fitted(learning_rate=LinearDecaySchedule(start=0.05, end=0.0005))

        for cosine in absolute_cosines(model, tilted_grid_reference()):
            assert cosine == pytest.approx(1.0, abs=1e-03)


class TestReproducibility:
    """A stochastic walk is only usable if a seed pins it."""

    def test_the_same_seed_gives_the_same_directions(self) -> None:
        first = HebbianPrincipalComponents(n_components=2, random_seed=3).fit(
            tilted_grid()
        )
        second = HebbianPrincipalComponents(n_components=2, random_seed=3).fit(
            tilted_grid()
        )

        assert np.array_equal(first.directions.vectors, second.directions.vectors)

    def test_different_seeds_give_different_walks(self) -> None:
        """Different starting weights and a different presentation order.

        They agree about the direction to within 1e-05 and disagree in the
        digits after that, which is what a stochastic approximation looks like.
        """
        first = HebbianPrincipalComponents(n_components=2, random_seed=3).fit(
            tilted_grid()
        )
        second = HebbianPrincipalComponents(n_components=2, random_seed=4).fit(
            tilted_grid()
        )

        assert not np.array_equal(first.directions.vectors, second.directions.vectors)


class TestWhatItRefuses:
    """Guards, each raising from the MLLibError hierarchy or from pydantic."""

    def test_reading_the_directions_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = HebbianPrincipalComponents().directions

    def test_reading_the_epoch_count_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = HebbianPrincipalComponents().epochs_run

    def test_reading_convergence_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = HebbianPrincipalComponents().converged

    def test_transforming_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            HebbianPrincipalComponents().transform(tilted_grid())

    def test_naming_the_components_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            HebbianPrincipalComponents().component_names()

    def test_no_features_at_all_raises(self) -> None:
        with pytest.raises(EmptyValuesError):
            HebbianPrincipalComponents().fit([])

    def test_a_single_row_raises(self) -> None:
        with pytest.raises(TooFewValuesError):
            HebbianPrincipalComponents().fit(
                [Feature("first", [1.0]), Feature("second", [2.0])]
            )

    def test_columns_of_different_lengths_raise(self) -> None:
        with pytest.raises(NonEqualArrayLengthError):
            HebbianPrincipalComponents().fit(
                [Feature("first", [1.0, 2.0]), Feature("second", [1.0, 2.0, 3.0])]
            )

    def test_duplicate_feature_names_are_rejected(self) -> None:
        with pytest.raises(NonUniqueFeaturesError):
            HebbianPrincipalComponents().fit(
                [Feature("same", [1.0, 2.0, 3.0]), Feature("same", [4.0, 5.0, 6.0])]
            )

    def test_more_components_than_features_raises(self) -> None:
        """A direction beyond the last has nothing left to see."""
        with pytest.raises(InvalidValuesError):
            HebbianPrincipalComponents(n_components=3).fit(tilted_grid())

    def test_data_with_no_spread_at_all_raises(self) -> None:
        """Every share would divide by zero, and no direction would mean anything."""
        with pytest.raises(AllSameValuesError):
            HebbianPrincipalComponents().fit(
                [Feature("first", [4.0] * 6), Feature("second", [7.0] * 6)]
            )

    def test_a_rate_too_large_for_the_data_raises_by_name(self) -> None:
        """Not a fit that completes and answers nan to everything afterwards."""
        with pytest.raises(DivergenceError):
            HebbianPrincipalComponents(
                learning_rate=ConstantSchedule(value=50.0), max_epochs=20
            ).fit(tilted_grid())

    def test_transforming_without_every_fitted_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().transform([tilted_grid()[0]])

    def test_transforming_with_an_unknown_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().transform([*tilted_grid(), Feature("extra", [1.0] * 8)])

    @pytest.mark.parametrize(
        "overrides",
        [
            {"n_components": 0},
            {"max_epochs": 0},
            {"tolerance": 0.0},
            {"tolerance": -1.0},
        ],
    )
    def test_impossible_configuration_is_refused_at_construction(
        self, overrides: dict
    ) -> None:
        with pytest.raises(ValidationError):
            HebbianPrincipalComponents(**overrides)

    def test_an_unknown_keyword_is_refused_rather_than_ignored(self) -> None:
        """The wrong value is always the default, which is a plausible number."""
        with pytest.raises(ValidationError):
            HebbianPrincipalComponents(n_component=2)  # pyright: ignore[reportCallIssue]


class TestTheDirectionValueObject:
    """What one learned vector guarantees about itself, and what it does not."""

    @staticmethod
    def weights(*values: float) -> Coefficients:
        return Coefficients(
            [
                Coefficient(f"feature_{position}", value)
                for position, value in enumerate(values)
            ]
        )

    def test_it_keeps_the_vector_at_the_length_it_was_given(self) -> None:
        """No normalising at construction, which is the whole difference."""
        direction = HebbianDirection("component_1", self.weights(3.0, 4.0), 2.0)

        assert direction.length == pytest.approx(5.0)
        assert direction.vector == pytest.approx(np.array([3.0, 4.0]))

    def test_the_direction_is_the_vector_scaled_to_one(self) -> None:
        direction = HebbianDirection("component_1", self.weights(3.0, 4.0), 2.0)

        assert direction.direction == pytest.approx(np.array([0.6, 0.8]))

    def test_it_accepts_a_length_a_principal_component_would_refuse(self) -> None:
        """1.002 is a correct answer here and outside PCA's 1e-08 tolerance."""
        direction = HebbianDirection("component_1", self.weights(1.002, 0.0), 1.0)

        assert direction.length == pytest.approx(1.002)

    def test_a_vector_of_zeros_is_refused(self) -> None:
        """It has no direction to normalise towards, so it names nothing."""
        with pytest.raises(InvalidValuesError):
            HebbianDirection("component_1", self.weights(0.0, 0.0), 1.0)

    def test_a_negative_variance_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            HebbianDirection("component_1", self.weights(1.0, 0.0), -0.5)

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            HebbianDirection("   ", self.weights(1.0, 0.0), 1.0)

    def test_a_weight_it_does_not_hold_is_refused(self) -> None:
        direction = HebbianDirection("component_1", self.weights(1.0, 0.0), 1.0)

        with pytest.raises(InvalidValuesError):
            direction.weight_for("nonesuch")


class TestTheDirectionsValueObject:
    """The set, and the two invariants it deliberately does not carry."""

    @staticmethod
    def direction(
        name: str, first: float, second: float, variance: float
    ) -> HebbianDirection:
        return HebbianDirection(
            name,
            Coefficients([Coefficient("first", first), Coefficient("second", second)]),
            variance,
        )

    def test_an_empty_set_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            HebbianDirections([], 4.0)

    def test_duplicate_names_are_refused(self) -> None:
        with pytest.raises(NonUniqueFeaturesError):
            HebbianDirections(
                [
                    self.direction("component_1", 1.0, 0.0, 3.0),
                    self.direction("component_1", 0.0, 1.0, 1.0),
                ],
                4.0,
            )

    def test_directions_over_different_features_are_refused(self) -> None:
        """Position ``i`` of one must mean the same feature as position ``i``
        of the next, or a projection sums across mismatched columns."""
        other = HebbianDirection(
            "component_2",
            Coefficients([Coefficient("third", 0.0), Coefficient("fourth", 1.0)]),
            1.0,
        )

        with pytest.raises(InvalidValuesError):
            HebbianDirections(
                [self.direction("component_1", 1.0, 0.0, 3.0), other], 4.0
            )

    def test_a_total_variance_of_zero_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            HebbianDirections([self.direction("component_1", 1.0, 0.0, 0.0)], 0.0)

    def test_increasing_variances_are_accepted_deliberately(self) -> None:
        """``PrincipalComponents`` refuses this and would be right to.

        Here the order is an empirical outcome of a finite walk rather than an
        algebraic guarantee, so it is reported through ``is_ordered`` instead of
        being enforced. Two directions through nearly equal variance can settle
        either way round and neither is wrong.
        """
        directions = HebbianDirections(
            [
                self.direction("component_1", 1.0, 0.0, 1.0),
                self.direction("component_2", 0.0, 1.0, 3.0),
            ],
            4.0,
        )

        assert not directions.is_ordered

    def test_directions_that_are_not_perpendicular_are_accepted_deliberately(
        self,
    ) -> None:
        """Sanger's rule cannot promise perpendicular, so this cannot demand it."""
        directions = HebbianDirections(
            [
                self.direction("component_1", 1.0, 0.0, 3.0),
                self.direction("component_2", 0.6, 0.8, 1.0),
            ],
            4.0,
        )

        assert directions.worst_orthogonality == pytest.approx(0.6)

    def test_a_lone_direction_has_no_pair_to_be_orthogonal_to(self) -> None:
        directions = HebbianDirections(
            [self.direction("component_1", 1.0, 0.0, 3.0)], 4.0
        )

        assert directions.worst_orthogonality == 0.0

    def test_shares_are_measured_against_the_supplied_total(self) -> None:
        """Not against what the set happens to hold, which is a different number."""
        directions = HebbianDirections(
            [self.direction("component_1", 1.0, 0.0, 3.0)], 4.0
        )

        assert directions.variance_shares == pytest.approx((0.75,))
        assert directions.kept_variance == pytest.approx(3.0)

    def test_it_is_iterable_rather_than_handing_out_its_container(self) -> None:
        directions = HebbianDirections(
            [
                self.direction("component_1", 1.0, 0.0, 3.0),
                self.direction("component_2", 0.0, 1.0, 1.0),
            ],
            4.0,
        )

        assert len(directions) == 2
        assert [one.name for one in directions] == ["component_1", "component_2"]
        assert "component_2" in directions
        assert directions["component_2"].variance == pytest.approx(1.0)

    def test_an_unknown_name_is_refused(self) -> None:
        directions = HebbianDirections(
            [self.direction("component_1", 1.0, 0.0, 3.0)], 4.0
        )

        with pytest.raises(InvalidValuesError):
            directions.value_for("component_9")
