"""Ridge and lasso: two penalties, two different things they do to a model.

Both add a price on large coefficients, and both trade a little bias for a lot
of variance. But the shapes of the two penalties differ where it matters:

* Ridge squares the weights, so the price of the last little bit of a weight
  approaches zero as the weight approaches zero. Nothing is ever worth setting
  exactly to zero, and ridge keeps every feature.
* Lasso takes absolute values, so the price stays constant all the way down.
  Below a threshold a weight is not worth carrying at all, and lasso sets it to
  exactly 0.0 -- which makes it a feature selector as well as a shrinker.

Two datasets show the two behaviours: collinear predictors, where a penalty
rescues an unstable fit, and a sparse signal, where only lasso finds the columns
that matter.
"""

from __future__ import annotations

import logging

from examples.datasets import collinear_predictors, sparse_signal
from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import LassoRegression, MultipleLinearRegression, RidgeRegression

logger = logging.getLogger(__name__)

PENALTIES = [0.0, 0.1, 1.0, 10.0, 100.0]
SELECTION_PENALTIES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def main() -> None:
    report = Report(logger)
    collinear = collinear_predictors()

    report.heading("Least squares on near-duplicate predictors")

    # 'second' is 0.95 * 'first' plus a whisper. The truth is +3.0 and -2.0, but
    # almost any pair summing to about 1.0 fits nearly as well, so the estimate
    # is at the mercy of the noise.
    least_squares = MultipleLinearRegression()
    least_squares.fit(collinear.input_features, collinear.target_feature)

    first_weight = least_squares.coefficients["first"]
    second_weight = least_squares.coefficients["second"]

    report.line("true    : first= 3.000  second=-2.000  (sum  1.000)")
    report.line(
        f"fitted  : first={first_weight:6.3f}  "
        f"second={second_weight:6.3f}  "
        f"(sum {first_weight + second_weight:6.3f})"
    )
    report.line("the sum is recovered well; the split between the two is not")

    report.heading("Ridge shrinks both, and never to zero")

    rows = []
    for penalty in PENALTIES:
        model = RidgeRegression(penalty=penalty)
        model.fit(collinear.input_features, collinear.target_feature)
        score = model.score(collinear.input_features, collinear.target_feature)
        rows.append(
            [
                f"{penalty:g}",
                f"{model.coefficients['first']:.4f}",
                f"{model.coefficients['second']:.4f}",
                f"{score:.4f}",
            ]
        )

    report.table(["penalty", "first", "second", "train R2"], rows)

    report.heading("Lasso shrinks too, but has an exact zero to reach")

    rows = []
    concerns = []
    for penalty in PENALTIES:
        model = LassoRegression(penalty=penalty)
        model.fit(collinear.input_features, collinear.target_feature)

        if not model.converged:
            concerns.append(
                f"penalty={penalty:g} exhausted all {model.max_iterations} sweeps "
                f"without converging; its coefficients are wherever the walk had "
                f"reached, not the optimum"
            )

        rows.append(
            [
                f"{penalty:g}",
                f"{model.coefficients['first']:.4f}",
                f"{model.coefficients['second']:.4f}",
                str(model.iterations_run),
                str(model.converged),
            ]
        )

    report.table(["penalty", "first", "second", "sweeps", "converged"], rows)

    for concern in concerns:
        report.warn(concern)

    report.paragraph(
        "Read the last two columns together. At penalty 0 and 0.1 lasso burns\n"
        "all 1000 sweeps and still reports converged=False -- on collinear data\n"
        "it has not finished, and its answer is short of the one np.linalg.solve\n"
        "produced instantly above. Then the penalty bites, one weight snaps to\n"
        "zero, the problem becomes one-dimensional, and it lands in 16 sweeps.\n\n"
        "Coordinate descent earns its keep only where no closed form exists --\n"
        "which is exactly the case an L1 penalty creates, and nowhere else."
    )

    report.heading("Selection: six predictors, four of them pure noise")

    sparse = sparse_signal()

    rows = []
    for penalty in SELECTION_PENALTIES:
        ridge = RidgeRegression(penalty=penalty)
        ridge.fit(sparse.input_features, sparse.target_feature)
        lasso = LassoRegression(penalty=penalty)
        lasso.fit(sparse.input_features, sparse.target_feature)

        rows.append(
            [
                f"{penalty:g}",
                str(sum(1 for weight in ridge.coefficients if weight.value == 0.0)),
                str(sum(1 for weight in lasso.coefficients if weight.value == 0.0)),
                f"{lasso.coefficients['signal_a']:.3f}",
                f"{lasso.coefficients['signal_b']:.3f}",
            ]
        )

    report.table(["penalty", "ridge zeros", "lasso zeros", "lasso a", "lasso b"], rows)

    report.paragraph(
        "Four of the six weights are truly zero. Ridge never reaches a single\n"
        "one of them at any penalty -- the column stays at 0 the whole way down.\n"
        "Lasso drops them off one at a time as the price rises, while holding\n"
        "signal_a and signal_b near their true 5.0 and -3.0."
    )

    heaviest = LassoRegression(penalty=5.0)
    heaviest.fit(sparse.input_features, sparse.target_feature)
    survivors = [weight.name for weight in heaviest.coefficients if weight.value != 0.0]
    surviving_noise = [name for name in survivors if name.startswith("noise")]

    report.line(f"\nstill standing at penalty=5: {survivors}")
    if surviving_noise:
        report.warn(
            f"{', '.join(surviving_noise)} has a true weight of zero and survived "
            f"selection anyway -- the noise happened to make it look real. "
            f"Selection is a judgement under uncertainty, not a filter."
        )


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
