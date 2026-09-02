"""Spec for the half of an iterative fit that is not the step.

Two claims carry this file.

The first is that it composes. ``ConvergentFit`` is a second base for models
that already inherit ``Clusterer`` or ``Transformer``, and pydantic's metaclass
derives from ``ABCMeta``, so a diamond here is a real risk rather than a
theoretical one. This library has been bitten by exactly that once already:
listing ``Generic`` ahead of ``Estimator`` in ``Regressor``'s bases left no
consistent method resolution order, pydantic raised at class-creation time, and
nothing in the package imported at all. The failure is total and it happens at
import, so a test that merely builds such a class is worth having.

The second is the bookkeeping, which exists because it was got wrong three
times in the copies this class replaces. A walk that settles on its first pass
ran one pass, not zero, and two of the three originals incremented the counter
after the convergence break and so reported zero passes run alongside
``converged=True``. That is invisible to any test that only checks the answer.

The rest is the guard. Reading either reported value before ``fit`` has to
raise, and ``_record_walk`` is deliberately separate from ``_mark_fitted`` so
that a fit raising partway through leaves neither set, which is the
commit-nothing-until-everything-succeeded pattern the serving audit established
for every other fit here.

That this file discriminates was measured rather than assumed, by breaking the
implementation four ways and running it against each. Reading the mean movement
rather than the largest fails 1, comparing with ``<=`` rather than ``<`` fails
1, leaving the sign on the movement fails 2, and folding ``_mark_fitted`` into
``_record_walk`` fails 2. The mean-versus-largest break is the one worth noting,
because the first version of this spec missed it entirely: its fixtures used a
tolerance of 1e-8 against a stray movement of 5e-3, which is large enough that
the average exceeds the tolerance too. Catching it needs a block where the two
readings genuinely disagree, one settled parameter in a hundred still moving,
and that is what the fixture is now.
"""

from collections.abc import Sequence

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.base.convergent_fit import ConvergentFit
from oop_ml.core.base.estimator import Clusterer, Fittable, Transformer
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.exceptions import NotFittedError
from oop_ml.core.types import FloatArray


class WalkingClusterer(Clusterer[Sequence[Feature]], ConvergentFit):
    """A clusterer that walks, which is the shape a self-organising map has."""

    max_epochs: int = 50

    @property
    def _pass_limit(self) -> int:
        return self.max_epochs

    def fit(self, input_values: Sequence[Feature]) -> "WalkingClusterer":
        self._record_walk(passes_run=3, converged=True)
        self._mark_fitted()
        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        return Predictions(np.zeros(1))


class WalkingTransformer(Transformer[Sequence[Feature]], ConvergentFit):
    """A transformer that walks, which is the shape the Hebbian projection has."""

    max_epochs: int = 50

    @property
    def _pass_limit(self) -> int:
        return self.max_epochs

    def fit(self, input_values: Sequence[Feature]) -> "WalkingTransformer":
        self._record_walk(passes_run=self.max_epochs, converged=False)
        self._mark_fitted()
        return self

    def transform(self, input_values: Sequence[Feature]) -> Sequence[Feature]:
        return input_values


class RefusingClusterer(WalkingClusterer):
    """A fit that raises after recording, to pin what a failed fit leaves behind."""

    def fit(self, input_values: Sequence[Feature]) -> "RefusingClusterer":
        self._record_walk(passes_run=7, converged=True)
        raise ValueError("this fit did not finish")


class TestItComposesWithBothFrames:
    """A diamond under pydantic's metaclass, which has broken this package once.

    These tests pass by the class bodies above having been created at import
    time. That reads as though they assert nothing, and it is the opposite: a
    method resolution order failure raises during class creation, so the whole
    module fails to import and every test in it errors at once. Naming the
    claim is what makes that legible when it happens.
    """

    def test_a_clusterer_can_also_be_a_convergent_fit(self) -> None:
        assert issubclass(WalkingClusterer, Clusterer)
        assert issubclass(WalkingClusterer, ConvergentFit)

    def test_a_transformer_can_also_be_a_convergent_fit(self) -> None:
        assert issubclass(WalkingTransformer, Transformer)
        assert issubclass(WalkingTransformer, ConvergentFit)

    def test_the_resolution_order_puts_the_frame_before_the_walk(self) -> None:
        """Clusterer first, so its ``fit`` signature is the one that wins."""
        order = [one.__name__ for one in WalkingClusterer.__mro__]

        assert order.index("Clusterer") < order.index("ConvergentFit")
        assert order.index("ConvergentFit") < order.index("Fittable")

    def test_the_fitted_state_guard_is_reached_once_and_not_twice(self) -> None:
        """Fittable sits at the join of the diamond, and appearing once is what
        makes ``_check_fitted`` one implementation rather than two."""
        assert WalkingClusterer.__mro__.count(Fittable) == 1
        assert WalkingTransformer.__mro__.count(Fittable) == 1


class TestTheConfiguredFields:
    def test_tolerance_defaults_and_can_be_set(self) -> None:
        assert WalkingClusterer().tolerance == pytest.approx(1e-8)
        assert WalkingClusterer(tolerance=1e-3).tolerance == pytest.approx(1e-3)

    @pytest.mark.parametrize("tolerance", [0.0, -1e-6], ids=["zero", "negative"])
    def test_a_tolerance_that_is_not_positive_is_refused(
        self, tolerance: float
    ) -> None:
        """A tolerance of zero can never be met, so the walk would always run
        to its cap and always report that it had not converged."""
        with pytest.raises(ValidationError):
            WalkingClusterer(tolerance=tolerance)

    def test_a_misspelled_field_is_refused(self) -> None:
        """extra="forbid" is inherited, which is what stops a misspelling
        leaving a plausible default quietly in place."""
        with pytest.raises(ValidationError):
            WalkingClusterer(tolerence=1e-3)  # type: ignore[call-arg]

    def test_the_model_names_its_own_cap(self) -> None:
        """A pass is an epoch here and an iteration elsewhere, so the public
        spelling belongs to the model and only the private one is shared."""
        model = WalkingClusterer(max_epochs=120)

        assert model._pass_limit == 120


class TestNothingIsReadableBeforeFit:
    @pytest.mark.parametrize(
        "model", [WalkingClusterer(), WalkingTransformer()], ids=["clusterer", "former"]
    )
    def test_converged_raises(self, model: ConvergentFit) -> None:
        with pytest.raises(NotFittedError):
            _ = model.converged

    @pytest.mark.parametrize(
        "model", [WalkingClusterer(), WalkingTransformer()], ids=["clusterer", "former"]
    )
    def test_the_pass_count_raises(self, model: ConvergentFit) -> None:
        with pytest.raises(NotFittedError):
            _ = model._completed_passes


class TestWhatTheWalkRecords:
    def test_a_settled_walk_reports_both(self) -> None:
        model = WalkingClusterer().fit([])

        assert model.converged is True
        assert model._completed_passes == 3

    def test_a_walk_that_ran_out_reports_both(self) -> None:
        model = WalkingTransformer(max_epochs=40).fit([])

        assert model.converged is False
        assert model._completed_passes == 40

    def test_a_walk_settling_immediately_reports_one_pass_and_not_zero(self) -> None:
        """The mistake this class was extracted to stop making a fourth time.

        Two of the three copies it replaces incremented the counter after the
        convergence break, so a fit that settled at once reported zero passes
        run while also reporting that it had converged. Nothing about the
        answer looks wrong when that happens.
        """
        model = WalkingClusterer()
        model._record_walk(passes_run=1, converged=True)
        model._mark_fitted()

        assert model._completed_passes == 1
        assert model.converged is True


class TestAFitThatRaisesCommitsNothing:
    """Recording and marking fitted are separate so that a failure leaves both."""

    def test_the_model_is_not_fitted(self) -> None:
        model = RefusingClusterer()

        with pytest.raises(ValueError):
            model.fit([])

        assert model.is_fitted is False

    def test_and_so_neither_reading_is_available(self) -> None:
        model = RefusingClusterer()
        with pytest.raises(ValueError):
            model.fit([])

        with pytest.raises(NotFittedError):
            _ = model.converged


class TestWhetherThePassSettled:
    """Measured on the movement rather than on the change in the objective.

    Near an optimum the objective is flat, so the improvement it reports goes
    as the square of the parameter error and reaches zero in floating point
    while the parameters are still visibly moving. The movement is in the units
    the caller reads back and needs no reference value.
    """

    @pytest.mark.parametrize(
        ("movement", "settled"),
        [
            (np.array([[1e-9, -2e-9], [0.0, 5e-10]]), True),
            (np.array([[1e-9, -2e-9], [0.0, 5e-3]]), False),
            (np.zeros((3, 3)), True),
        ],
        ids=["all tiny", "one large", "no movement at all"],
    )
    def test_it_reads_the_largest_entry_of_a_block(
        self, movement: FloatArray, settled: bool
    ) -> None:
        assert WalkingClusterer()._has_converged(movement) is settled

    @pytest.mark.parametrize(
        ("movement", "settled"), [(1e-12, True), (0.5, False), (-0.5, False)]
    )
    def test_a_scalar_is_the_same_question_of_a_different_shape(
        self, movement: float, settled: bool
    ) -> None:
        """Not every walk here moves an array. A Hopfield settling moves a count
        of flipped units where a contrastive divergence step moves a matrix."""
        assert WalkingClusterer()._has_converged(movement) is settled

    def test_the_sign_of_the_movement_does_not_matter(self) -> None:
        model = WalkingClusterer(tolerance=1e-3)

        assert model._has_converged(np.array([-1e-6])) is True
        assert model._has_converged(np.array([-1.0])) is False

    def test_it_is_strictly_below_the_tolerance(self) -> None:
        """A movement exactly equal to the tolerance has not settled, which is
        the same comparison ``IterativeSolver`` makes."""
        model = WalkingClusterer(tolerance=0.25)

        assert model._has_converged(np.array([0.25])) is False
        assert model._has_converged(np.array([0.2499])) is True

    def test_one_parameter_still_moving_is_enough_to_say_it_has_not_settled(
        self,
    ) -> None:
        """The largest movement, not the average one, and the difference is
        the whole meaning of the check.

        A hundred settled parameters and one that is still moving is a walk
        that has not settled. An average would report otherwise, and the more
        parameters a model has the more thoroughly it would hide the one that
        matters. The fixture is built so that the two answers actually differ:
        the largest entry is ten times the tolerance and the mean is a tenth
        of it, so a version reading the mean passes every other test in this
        class and fails only here.
        """
        model = WalkingClusterer(tolerance=0.1)
        settled_but_one = np.zeros(101)
        settled_but_one[0] = 1.0

        assert float(np.max(np.abs(settled_but_one))) > model.tolerance
        assert float(np.mean(np.abs(settled_but_one))) < model.tolerance
        assert model._has_converged(settled_but_one) is False

    def test_the_tolerance_is_the_one_the_model_was_given(self) -> None:
        movement = np.array([1e-4])

        assert WalkingClusterer(tolerance=1e-6)._has_converged(movement) is False
        assert WalkingClusterer(tolerance=1e-2)._has_converged(movement) is True


class TestItCarriesNoFormatContractOfItsOwn:
    def test_learned_state_is_left_empty_deliberately(self) -> None:
        """A subclass lists ``_passes_run`` and ``_converged`` alongside its own
        learned attributes, so that a class inheriting this without being
        persistable does not acquire a format contract by accident."""
        assert ConvergentFit.LEARNED_STATE == ()


class TestTheBaseIsAbstract:
    def test_a_subclass_must_name_its_own_cap(self) -> None:
        class WithoutACap(ConvergentFit):
            pass

        with pytest.raises(TypeError):
            WithoutACap()  # type: ignore[abstract]
