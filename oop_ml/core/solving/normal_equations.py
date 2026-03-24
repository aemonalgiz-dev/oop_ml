"""The intermediate matrices a closed-form fit builds and then discards.

``np.linalg.solve(X.T @ X, X.T @ y)`` is one line, and the one line is why the
closed-form models look like they have nothing to observe. They do: the
derivation has four named stages, three of them are matrices that never leave
the method, and each answers a question about the fit that the coefficients
alone cannot.

``X`` shows whether an intercept column was prepended. ``X.T X`` is the matrix
whose invertibility is the entire question of whether the fit is possible --
collinear predictors make it singular, and a near-singular one is how a fit
returns enormous coefficients that cancel. ``X.T y`` is each predictor's raw
alignment with the target, before any of them are adjusted for the others.

Ridge adds a fifth. Its penalty matrix is ``penalty * I`` with the intercept's
diagonal slot zeroed, and that zero is worth being able to look at: setting it
unconditionally exempts a real predictor whenever ``fit_intercept`` is false,
which is a bug this library shipped until a test caught it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from oop_ml.core.data.design_matrix import DesignMatrix
from oop_ml.core.exceptions import CollinearFeaturesError
from oop_ml.core.observation import Stage
from oop_ml.core.types import FloatArray


class NormalEquations:
    """The matrices behind a closed-form least-squares solution.

    Parameters
    ----------
    design_matrix:
        ``X``, with the leading ones column if one was wanted.
    moment_matrix:
        ``X.T X``, before any penalty.
    target_moments:
        ``X.T y``.
    penalty_matrix:
        What was added to ``X.T X``, or ``None`` for an unpenalised fit.
    solution:
        The coefficients the system was solved for.
    """

    __slots__ = (
        "_design_matrix",
        "_moment_matrix",
        "_penalty_matrix",
        "_solution",
        "_target_moments",
    )

    def __init__(
        self,
        design_matrix: DesignMatrix,
        moment_matrix: FloatArray,
        target_moments: FloatArray,
        penalty_matrix: FloatArray | None,
        solution: FloatArray,
    ) -> None:
        self._design_matrix = design_matrix
        self._moment_matrix = moment_matrix
        self._target_moments = target_moments
        self._penalty_matrix = penalty_matrix
        self._solution = solution

    @property
    def result(self) -> FloatArray:
        """What the efficient route would have returned: the coefficients."""
        return self._solution

    @property
    def design_matrix(self) -> DesignMatrix:
        """``X``. Its first column is ones where an intercept was fitted."""
        return self._design_matrix

    @property
    def moment_matrix(self) -> FloatArray:
        """``X.T X``, unpenalised."""
        return self._moment_matrix

    @property
    def target_moments(self) -> FloatArray:
        """``X.T y``: each predictor's raw alignment with the target."""
        return self._target_moments

    @property
    def penalty_matrix(self) -> FloatArray | None:
        """What was added to ``X.T X``, or ``None`` if nothing was."""
        return self._penalty_matrix

    @property
    def solved_matrix(self) -> FloatArray:
        """The left-hand side actually handed to the solver."""
        if self._penalty_matrix is None:
            return self._moment_matrix

        return self._moment_matrix + self._penalty_matrix

    @property
    def solution(self) -> FloatArray:
        """The coefficients."""
        return self._solution

    @property
    def condition_number(self) -> float:
        """How close the solved system is to being unsolvable.

        Large means the coefficients are decided by rounding as much as by the
        data -- the numerical face of collinearity, visible here and nowhere
        in the coefficients themselves. Infinite means singular, which is the
        one failure in this library that escapes as a bare ``LinAlgError``.
        """
        return float(np.linalg.cond(self.solved_matrix))

    def __iter__(self) -> Iterator[Stage]:
        """The stages in the order the derivation builds them."""
        stages = [
            Stage("design matrix", self._design_matrix),
            Stage("X.T X", self._moment_matrix),
            Stage("X.T y", self._target_moments),
        ]
        if self._penalty_matrix is not None:
            stages.append(Stage("penalty", self._penalty_matrix))
            stages.append(Stage("X.T X + penalty", self.solved_matrix))
        stages.append(Stage("solution", self._solution))

        return iter(stages)

    def __len__(self) -> int:
        return 4 if self._penalty_matrix is None else 6

    def __repr__(self) -> str:
        penalised = "" if self._penalty_matrix is None else ", penalised"
        return (
            f"NormalEquations({self._design_matrix.n_columns} parameters"
            f"{penalised}, condition {self.condition_number:.3g})"
        )


class LeastSquaresLine:
    """The two sums a single-predictor fit reduces to.

    The smallest closed form in the library, and the one whose stages are most
    worth naming, because every larger least-squares fit is this generalised.
    The slope is the two columns' co-variation over the predictor's own
    variation; the intercept then anchors the line at the pair of means, which
    is why a fitted line always passes through them.

    Parameters
    ----------
    input_mean, target_mean:
        The point the fitted line is anchored at.
    covariation:
        ``sum((x - x_mean) * (y - y_mean))``. The numerator of the slope, and
        the only place the target enters.
    input_variation:
        ``sum((x - x_mean) ** 2)``. The denominator, and the reason a constant
        predictor cannot be fitted -- it is zero, and nothing else about the
        data matters.
    slope, intercept:
        What the fit learned.
    """

    __slots__ = (
        "_covariation",
        "_input_mean",
        "_input_variation",
        "_intercept",
        "_slope",
        "_target_mean",
    )

    def __init__(
        self,
        input_mean: float,
        target_mean: float,
        covariation: float,
        input_variation: float,
        slope: float,
        intercept: float,
    ) -> None:
        self._input_mean = input_mean
        self._target_mean = target_mean
        self._covariation = covariation
        self._input_variation = input_variation
        self._slope = slope
        self._intercept = intercept

    @property
    def result(self) -> tuple[float, float]:
        """The slope and intercept the efficient route stores."""
        return self._slope, self._intercept

    @property
    def input_mean(self) -> float:
        """The predictor's mean."""
        return self._input_mean

    @property
    def target_mean(self) -> float:
        """The target's mean."""
        return self._target_mean

    @property
    def covariation(self) -> float:
        """Sum of the products of the two columns' deviations."""
        return self._covariation

    @property
    def input_variation(self) -> float:
        """Sum of the predictor's squared deviations."""
        return self._input_variation

    @property
    def slope(self) -> float:
        """Covariation over the predictor's own variation."""
        return self._slope

    @property
    def intercept(self) -> float:
        """What anchors the line at the pair of means."""
        return self._intercept

    def __iter__(self) -> Iterator[Stage]:
        return iter(
            [
                Stage("input mean", self._input_mean),
                Stage("target mean", self._target_mean),
                Stage("covariation", self._covariation),
                Stage("input variation", self._input_variation),
                Stage("slope", self._slope),
                Stage("intercept", self._intercept),
            ]
        )

    def __len__(self) -> int:
        return 6

    def __repr__(self) -> str:
        return (
            f"LeastSquaresLine(slope {self._slope:.4f}, "
            f"intercept {self._intercept:.4f})"
        )


def solved_normal_equations(
    system_matrix: FloatArray, target_vector: FloatArray
) -> FloatArray:
    """The unique solution of ``system_matrix @ coefficients = target_vector``.

    A thin wrapper around ``numpy.linalg.solve`` whose whole job is the failure
    path, written once so that the four call sites -- ordinary least squares
    and ridge, each with an efficient and an observed route -- cannot drift
    apart about what a singular system means. That is the same argument
    ``penalty_diagonal`` records: a rule written twice is a rule one copy gets
    wrong.

    Raises
    ------
    CollinearFeaturesError
        If the system is singular. For normal equations that means the
        features are collinear: some column is (nearly) a linear combination
        of the others, so infinitely many coefficient vectors fit equally
        well. The condition number is included because it is the diagnosis --
        and the message points at ridge, because a positive penalty is the
        standard way out.
    """
    try:
        return np.linalg.solve(system_matrix, target_vector)
    except np.linalg.LinAlgError as error:
        raise CollinearFeaturesError(
            f"the features are collinear -- some column is (nearly) a linear "
            f"combination of the others -- so the normal equations have no "
            f"unique solution (condition number "
            f"{float(np.linalg.cond(system_matrix)):.3g}). Remove or combine "
            f"the duplicated columns, or use RidgeRegression, whose penalty "
            f"makes the system solvable"
        ) from error
