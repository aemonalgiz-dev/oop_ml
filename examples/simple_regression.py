"""One predictor, one response, solved in closed form.

Start here. ``SimpleLinearRegression`` is the only model in the library that
takes bare arrays rather than named features, because with a single predictor
there is nothing to name -- and it shows the three-call shape every estimator
shares: construct, ``fit``, then ask.

The thing to notice is what ``evaluate`` returns. Not a float, and not a tuple of
metrics: a :class:`~oop_ml.evaluation.regression.RegressionEvaluation` that has already
aligned the predictions with the truth. Metrics are read off it as properties, so
predicting twice to get two numbers is impossible by construction.
"""

from __future__ import annotations

import logging

from examples.datasets import straight_line
from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import SimpleLinearRegression

logger = logging.getLogger(__name__)

FITTED_RANGE_MAXIMUM = 20.0


def main() -> None:
    report = Report(logger)
    data = straight_line()
    predictor_values = data.input_features[0].values
    target_values = data.target_feature.values

    report.detail(f"generated {len(target_values)} rows from y = 4 + 2.5x + noise")

    report.heading("Fitting y = intercept + slope * x")

    # Construction configures; fit learns. No data reaches the constructor.
    model = SimpleLinearRegression()
    model.fit(predictor_values, target_values)

    report.line("true    : intercept=4.000  slope=2.500")
    report.line(f"fitted  : intercept={model.intercept:.3f}  slope={model.slope:.3f}")

    report.heading("Reading the fit")

    # One call, one evaluation object, as many metrics as you like off it.
    evaluation = model.evaluate(predictor_values, target_values)
    report.evaluation("in-sample", evaluation)
    report.line(f"RSS={evaluation.residual_sum_of_squares:.3f}")
    report.line(f"TSS={evaluation.total_sum_of_squares:.3f}")

    # R^2 is exactly the share of TSS the model removed -- not a separate idea.
    explained_share = 1.0 - (
        evaluation.residual_sum_of_squares / evaluation.total_sum_of_squares
    )
    report.line(f"1 - RSS/TSS = {explained_share:.4f}  (== R2 above)")
    report.detail(
        f"largest residual: {max(abs(evaluation.residuals)):.4f} "
        f"over {evaluation.n_samples} rows"
    )

    report.heading("Predicting on new inputs")

    for new_value in [0.0, 10.0, 25.0]:
        prediction = float(model.predict([new_value])[0])
        report.line(f"x={new_value:>5.1f}  ->  y={prediction:8.3f}")

        if new_value > FITTED_RANGE_MAXIMUM:
            report.warn(
                f"x={new_value:g} is outside the fitted range "
                f"(0 to {FITTED_RANGE_MAXIMUM:g}); this is extrapolation and the "
                f"model has no way to know it"
            )


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
