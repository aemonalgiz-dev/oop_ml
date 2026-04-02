"""Spec for the activations -- red until the eight bodies land.

Four of these are worth reading rather than skimming.

The first is the shape and finiteness pair, run over every activation. An
elementwise function that quietly reduces, or that answers ``nan`` at the tails
where the naive exponential overflows, satisfies a formula test on small inputs
and fails here. ``Sigmoid`` at ``z = -800`` is the specific trap, and it is why
:func:`oop_ml.core.logistic.stable_logistic` exists.

The second is the derivative check by finite difference. Every activation is
compared against ``(f(z + h) - f(z - h)) / 2h`` on the same scores, which is an
oracle written from the definition of a derivative rather than from the
implementation, so a derivative that was copied from the wrong formula cannot
agree with it by construction. The rectifier is excluded at its kink, where no
derivative exists.

Every oracle here is built that way on purpose. Comparing ``of`` against the
function its body will call, or a derivative against ``of * (1 - of)`` when
that is what the body computes, is the tautological oracle this library
already got caught by once: both sides move together and the test cannot
fail.

The third is the sigmoid's ceiling. Its slope peaks at exactly 0.25, which is
the arithmetic behind the vanishing gradient, and asserting it here is what
stops a body that returns ``sigmoid(z)`` instead of ``sigmoid(z) * (1 -
sigmoid(z))`` from passing everything else.

The fourth is that the rectifier answers 0 rather than 1 at exactly zero. The
derivative does not exist there, this library picks the negative branch's
answer, and an implementation choosing the other one is not wrong so much as
disagreeing with its own docstring.
"""

import numpy as np
import pytest

from oop_ml.core.network.activation import (
    Activation,
    HyperbolicTangent,
    Identity,
    RectifiedLinear,
    Sigmoid,
)

EVERY_ACTIVATION = [
    Identity(),
    RectifiedLinear(),
    Sigmoid(),
    HyperbolicTangent(),
]

SMOOTH_ACTIVATIONS = [Identity(), Sigmoid(), HyperbolicTangent()]

ORDINARY_SCORES = np.array([-2.5, -1.0, -0.25, 0.5, 1.0, 3.0])

EXTREME_SCORES = np.array([-800.0, -50.0, 0.0, 50.0, 800.0])


class TestEveryActivation:
    """What holds for all of them, whatever the formula."""

    @pytest.mark.parametrize("activation", EVERY_ACTIVATION)
    def test_it_preserves_shape(self, activation: Activation) -> None:
        scores = np.array([[1.0, -2.0, 3.0], [0.5, 0.0, -0.5]])

        assert activation.of(scores).shape == scores.shape
        assert activation.derivative_at(scores).shape == scores.shape

    @pytest.mark.parametrize("activation", EVERY_ACTIVATION)
    def test_it_stays_finite_at_the_extremes(self, activation: Activation) -> None:
        assert np.all(np.isfinite(activation.of(EXTREME_SCORES)))
        assert np.all(np.isfinite(activation.derivative_at(EXTREME_SCORES)))

    @pytest.mark.parametrize("activation", EVERY_ACTIVATION)
    def test_it_works_elementwise(self, activation: Activation) -> None:
        """Entry i of the answer depends on entry i and nothing else.

        This is the property that separates these from softmax, so it is worth
        asserting rather than assuming. Every score is bent alone and the
        answers are compared against the whole row bent at once.
        """
        together = activation.of(ORDINARY_SCORES)
        alone = np.array(
            [float(activation.of(np.array([score]))[0]) for score in ORDINARY_SCORES]
        )

        assert np.allclose(together, alone)

    @pytest.mark.parametrize("activation", EVERY_ACTIVATION)
    def test_it_describes_itself(self, activation: Activation) -> None:
        assert isinstance(activation.description, str)
        assert activation.description.strip()

    @pytest.mark.parametrize("activation", SMOOTH_ACTIVATIONS)
    def test_its_derivative_matches_a_finite_difference(
        self, activation: Activation
    ) -> None:
        """An oracle written from the definition, not from the implementation.

        The rectifier is absent because its kink at zero has no derivative and
        a central difference across it answers 0.5, which is neither branch.
        """
        step = 1e-6
        approximated = (
            activation.of(ORDINARY_SCORES + step)
            - activation.of(ORDINARY_SCORES - step)
        ) / (2.0 * step)

        assert np.allclose(
            activation.derivative_at(ORDINARY_SCORES), approximated, atol=1e-6
        )


class TestIdentity:
    def test_it_returns_its_argument(self) -> None:
        assert np.allclose(Identity().of(ORDINARY_SCORES), ORDINARY_SCORES)

    def test_its_slope_is_one_everywhere(self) -> None:
        slopes = Identity().derivative_at(ORDINARY_SCORES)

        assert np.allclose(slopes, np.ones_like(ORDINARY_SCORES))


class TestRectifiedLinear:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [(-3.0, 0.0), (-0.001, 0.0), (0.0, 0.0), (0.001, 0.001), (4.0, 4.0)],
    )
    def test_it_keeps_the_positive_half(self, score: float, expected: float) -> None:
        answer = RectifiedLinear().of(np.array([score]))

        assert float(answer[0]) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("score", "expected"),
        [(-3.0, 0.0), (-0.001, 0.0), (0.001, 1.0), (4.0, 1.0)],
    )
    def test_its_slope_is_one_or_nothing(self, score: float, expected: float) -> None:
        slope = RectifiedLinear().derivative_at(np.array([score]))

        assert float(slope[0]) == pytest.approx(expected)

    def test_the_kink_takes_the_negative_branch(self) -> None:
        """No derivative exists at zero, and this library answers 0 there."""
        slope = RectifiedLinear().derivative_at(np.array([0.0]))

        assert float(slope[0]) == 0.0


class TestSigmoid:
    def test_it_matches_the_definition_where_the_definition_is_safe(self) -> None:
        expected = 1.0 / (1.0 + np.exp(-ORDINARY_SCORES))

        assert np.allclose(Sigmoid().of(ORDINARY_SCORES), expected)

    def test_it_is_one_half_at_zero(self) -> None:
        assert float(Sigmoid().of(np.array([0.0]))[0]) == pytest.approx(0.5)

    def test_it_stays_bounded(self) -> None:
        answers = Sigmoid().of(EXTREME_SCORES)

        assert np.all(answers >= 0.0)
        assert np.all(answers <= 1.0)

    def test_a_deeply_negative_score_does_not_warn(self) -> None:
        """The naive spelling overflows here and emits a RuntimeWarning.

        The value it returns is still correct, since ``1 / inf`` is zero, so
        nothing but this test separates the stable spelling from the trap.
        """
        with np.errstate(over="raise"):
            answer = Sigmoid().of(np.array([-800.0]))

        assert float(answer[0]) == pytest.approx(0.0)

    def test_its_slope_peaks_at_a_quarter(self) -> None:
        """The arithmetic behind the vanishing gradient, asserted once."""
        dense = np.linspace(-8.0, 8.0, 100_001)
        slopes = Sigmoid().derivative_at(dense)

        assert float(slopes.max()) == pytest.approx(0.25, abs=1e-9)
        assert float(dense[slopes.argmax()]) == pytest.approx(0.0, abs=1e-3)


class TestHyperbolicTangent:
    def test_it_matches_the_definition(self) -> None:
        """Built from the exponential definition, not from ``np.tanh``.

        The body will be ``np.tanh(scores)``, so an oracle that also called
        ``np.tanh`` would be the implementation compared against itself and
        could not fail. This is the same independent-oracle rule the sigmoid's
        naive-form test follows.
        """
        doubled = np.exp(2.0 * ORDINARY_SCORES)
        expected = (doubled - 1.0) / (doubled + 1.0)

        assert np.allclose(HyperbolicTangent().of(ORDINARY_SCORES), expected)

    def test_it_is_zero_at_zero(self) -> None:
        assert float(HyperbolicTangent().of(np.array([0.0]))[0]) == pytest.approx(0.0)

    def test_its_slope_peaks_at_one(self) -> None:
        """Four times the sigmoid's ceiling, which is the whole difference."""
        dense = np.linspace(-8.0, 8.0, 100_001)
        slopes = HyperbolicTangent().derivative_at(dense)

        assert float(slopes.max()) == pytest.approx(1.0, abs=1e-9)


class TestActivationIdentityAndEquality:
    """Two of the same bend are the same bend, since none of them carry state."""

    @pytest.mark.parametrize("activation", EVERY_ACTIVATION)
    def test_two_of_a_kind_are_equal(self, activation: Activation) -> None:
        twin = type(activation)()

        assert activation == twin
        assert hash(activation) == hash(twin)

    def test_different_bends_are_not_equal(self) -> None:
        assert Sigmoid() != HyperbolicTangent()

    def test_it_defers_to_anything_that_is_not_an_activation(self) -> None:
        """``__eq__`` returns NotImplemented so Python retries the other side."""
        assert Sigmoid().__eq__("sigmoid") is NotImplemented


class TestSoftmaxIsAbsentOnPurpose:
    """The module offers no elementwise softmax, and that is a design decision.

    Softmax reads a whole layer, so it cannot be an ``Activation`` and cannot
    live on a neuron, which has no neighbours to normalise against. This test
    exists so that adding one here has to be a deliberate act that breaks a
    named expectation rather than a quiet import.
    """

    def test_the_module_exposes_no_softmax_activation(self) -> None:
        from oop_ml.core.network import activation as activation_module

        assert not hasattr(activation_module, "Softmax")

    def test_the_stable_row_wise_one_lives_in_core_instead(self) -> None:
        from oop_ml.core.logistic import stable_softmax

        rows = np.array([[2.0, 1.0, 0.1]])
        answer = stable_softmax(rows)

        assert float(np.sum(answer)) == pytest.approx(1.0)
        assert float(answer[0][0]) == pytest.approx(0.659001, abs=1e-6)
