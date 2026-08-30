"""Walking downhill instead of jumping, and what that costs you.

``GradientDescentRegression`` minimises exactly the same objective as
``MultipleLinearRegression`` -- same convex bowl, same floor. The closed form
lands on the floor in one algebraic step; descent walks there. On this problem
that is strictly worse, which is the point: it is being learned here, where the
right answer is already known, so it can be checked against the jump.

It also drags in two things the closed form never had to care about, and both
show up below: the learning rate has to be small enough not to overshoot, and
the columns have to be on comparable scales or no single rate works for all of
them.
"""

from __future__ import annotations

import logging

import numpy as np

from examples.datasets import independent_predictors
from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import (
    DivergenceError,
    GradientDescentRegression,
    MultipleLinearRegression,
    Standardizer,
)

logger = logging.getLogger(__name__)

LEARNING_RATES = [0.001, 0.01, 0.1, 0.5, 1.0]


def main() -> None:
    report = Report(logger)
    data = independent_predictors()

    report.heading("The closed form, for reference")

    closed_form = MultipleLinearRegression()
    closed_form.fit(data.input_features, data.target_feature)
    report.line(
        f"R2 = {closed_form.score(data.input_features, data.target_feature):.6f}"
    )
    report.line("one call to np.linalg.solve, no iteration, no hyperparameters")

    report.heading("The same walk on the raw columns")

    # age_years spans 0 to 60 and rooms spans about 2 to 8, so the gradient in
    # the age direction is an order of magnitude larger. One learning rate has
    # to serve both, and the rate that is safe for age crawls for rooms.
    for feature in data.input_features:
        report.detail(
            f"{feature.name}: min={feature.values.min():.2f} "
            f"max={feature.values.max():.2f}"
        )

    raw_descent = GradientDescentRegression(learning_rate=0.01, max_epochs=5_000)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            raw_descent.fit(data.input_features, data.target_feature)
        report.line(f"converged : {raw_descent.converged}")
        report.line(f"epochs    : {raw_descent.epochs_run}")
        report.line(f"intercept : {raw_descent.intercept}")
    except DivergenceError as failure:
        report.warn(
            f"the fit refused to finish: {failure}. The walk overshot on the "
            "first step and never came back -- every step on these raw "
            "columns is scaled by values in the hundreds of thousands. The "
            "failure names its cause at the fit, rather than surfacing as "
            "nan predictions three calls later."
        )

    report.heading("The same walk on standardized columns")

    standardizer = Standardizer()
    scaled_features = standardizer.fit_transform(data.input_features)

    scaled_descent = GradientDescentRegression(learning_rate=0.01, max_epochs=5_000)
    scaled_descent.fit(scaled_features, data.target_feature)

    report.line(f"converged : {scaled_descent.converged}")
    report.line(f"epochs    : {scaled_descent.epochs_run}")
    report.line(
        f"R2        : {scaled_descent.score(scaled_features, data.target_feature):.6f}"
    )
    report.paragraph(
        "Same objective, same minimum, same R2 as the closed form. The weights\n"
        "differ because they are now per standard deviation rather than per room\n"
        "or per year -- standardizing changed the units, not the model."
    )

    report.heading("Learning rate is the whole game")

    rows = []
    concerns = []
    for learning_rate in LEARNING_RATES:
        model = GradientDescentRegression(
            learning_rate=learning_rate, max_epochs=20_000
        )

        # A diverged fit refuses to finish: the walk overflows to non-finite
        # weights and the fit raises DivergenceError naming the cause, rather
        # than handing back a model that answers nan to every question.
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                model.fit(scaled_features, data.target_feature)
            outcome = f"{model.score(scaled_features, data.target_feature):.6f}"
            converged = str(model.converged)
            epochs = str(model.epochs_run)
        except DivergenceError:
            outcome = "diverged"
            converged = "False"
            epochs = "-"
            concerns.append(
                f"learning_rate={learning_rate:g} diverged: every step "
                f"overshot further than the last, and the fit refused to "
                f"finish rather than return nan coefficients"
            )
        else:
            if not model.converged:
                concerns.append(
                    f"learning_rate={learning_rate:g} ran out of patience at "
                    f"{model.epochs_run} epochs without converging"
                )

        rows.append([f"{learning_rate:g}", converged, epochs, outcome])

    report.table(["rate", "converged", "epochs", "R2"], rows)

    for concern in concerns:
        report.warn(concern)

    report.paragraph(
        "Too small and it runs out of patience before arriving; too large and\n"
        "each step overshoots further than the last. The usable band is a\n"
        "property of the data, which is why there is no sensible default."
    )


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
