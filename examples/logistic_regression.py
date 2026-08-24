"""A linear boundary fitted by maximum likelihood, and how to read it.

The first thing to notice is how little of the API changes. Features still go in
by name, coefficients still come back by name, and the fitted object still
answers with an evaluation. What changes is the question: the target is now a
label rather than a quantity, so the model has no closed form to jump to and no
R^2 to report.

The second thing is that a logistic coefficient does not mean what a regression
coefficient means. It is a multiplier on the odds, not an amount added to the
answer, and the last section here shows the difference in the numbers rather
than asserting it.
"""

from __future__ import annotations

import logging

import numpy as np

from examples.datasets import exam_outcomes, separable_outcomes
from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import LogisticRegression, Standardizer

logger = logging.getLogger(__name__)


def main() -> None:
    report = Report(logger)
    data = exam_outcomes()

    report.detail(
        f"{data.dataset.n_samples} rows, "
        f"{data.positive_rate:.1%} of them positive, "
        f"predictors {[feature.name for feature in data.input_features]}"
    )

    report.heading("Fitting a boundary rather than a line")

    # Standardised first, because a single learning rate has to serve every
    # direction and these two columns are on different scales.
    standardizer = Standardizer()
    scaled_features = standardizer.fit_transform(data.input_features)

    model = LogisticRegression(learning_rate=0.5, max_epochs=50_000, tolerance=1e-10)
    model.fit(scaled_features, data.target_feature)

    report.line(f"converged : {model.converged_} after {model.epochs_run_} epochs")
    report.line(f"intercept : {model.intercept_:.4f}")
    for coefficient in model.coefficients_:
        report.line(f"{coefficient.name:<14}: {coefficient.value:+.4f}")

    report.paragraph(
        "There was no equation to solve here. Ridge and least squares both have\n"
        "a closed form; the logistic likelihood does not, because setting its\n"
        "gradient to zero leaves the coefficients inside a sigmoid. Gradient\n"
        "ascent is not the slow way to do this, it is the only way."
    )

    report.heading("Probabilities and labels are different answers")

    probabilities = model.predict_probability(scaled_features)
    labels = model.predict(scaled_features)

    ordered = np.argsort(probabilities)
    rows = []
    for position in list(ordered[:3]) + list(ordered[-3:]):
        rows.append(
            [
                f"{data.input_features[0].values[position]:.2f}",
                f"{data.input_features[1].values[position]:.2f}",
                f"{probabilities[position]:.4f}",
                f"{labels[position]:.0f}",
                f"{data.target_feature.values[position]:.0f}",
            ]
        )

    report.table(
        ["studied", "slept", "P(pass)", "predicted", "actual"],
        rows,
    )
    report.paragraph(
        "Three least confident rows and three most confident. A row at 0.51 and\n"
        "a row at 0.99 receive the same label and are not the same claim, which\n"
        "is why both methods exist rather than just the one."
    )

    report.heading("Reading a coefficient as a multiplier on the odds")

    report.table(
        ["feature", "coefficient", "odds multiplier"],
        [
            [
                coefficient.name,
                f"{coefficient.value:+.4f}",
                f"{model.odds_multiplier_for(coefficient.name):.4f}",
            ]
            for coefficient in model.coefficients_
        ],
    )

    # The odds ratio is constant along the curve; the change in probability is
    # not, which is the whole reason the model is built on log-odds. Vary one
    # standardised feature and hold the other at zero, which on a standardised
    # column is its mean.
    strongest = max(model.coefficients_, key=lambda weight: abs(weight.value))
    rows = []
    previous_probability = None
    previous_odds = None
    for standard_deviations in [-1.0, 0.0, 1.0, 2.0]:
        log_odds = model.intercept_ + strongest.value * standard_deviations
        probability = 1.0 / (1.0 + np.exp(-log_odds))
        odds = probability / (1.0 - probability)

        rows.append(
            [
                f"{standard_deviations:+.1f}",
                f"{probability:.4f}",
                f"{odds:.4f}",
                "-" if previous_odds is None else f"{odds / previous_odds:.4f}",
                "-"
                if previous_probability is None
                else f"{probability - previous_probability:+.4f}",
            ]
        )
        previous_probability, previous_odds = probability, odds

    report.table(
        [f"{strongest.name} (sd)", "P(pass)", "odds", "odds ratio", "change in P"],
        rows,
    )
    report.paragraph(
        "The odds ratio column is constant and equals exp(coefficient). The\n"
        "change in probability is not, because the sigmoid flattens at both\n"
        "ends. An extra hour helps a borderline student far more than one who\n"
        "was already going to pass, and that is a fact about the curve rather\n"
        "than about the students."
    )

    report.heading("Separation: when there is no answer to find")

    separable = separable_outcomes()
    short_run = LogisticRegression(learning_rate=0.5, max_epochs=200)
    short_run.fit(separable.input_features, separable.target_feature)
    long_run = LogisticRegression(learning_rate=0.5, max_epochs=5_000)
    long_run.fit(separable.input_features, separable.target_feature)

    rows = []
    for model_run in (short_run, long_run):
        training_accuracy = model_run.score(
            separable.input_features, separable.target_feature
        )
        rows.append(
            [
                str(model_run.max_epochs),
                str(model_run.epochs_run_),
                str(model_run.converged_),
                f"{model_run.coefficients_['hours_studied']:.4f}",
                f"{training_accuracy:.4f}",
            ]
        )

    report.table(
        ["max_epochs", "epochs run", "converged", "coefficient", "training accuracy"],
        rows,
    )

    growth = (
        long_run.coefficients_["hours_studied"]
        / short_run.coefficients_["hours_studied"]
    )
    report.warn(
        f"neither run converged, and the coefficient grew by a factor of "
        f"{growth:.1f} between them. On perfectly separable classes the maximum "
        f"likelihood estimate does not exist, so the walk climbs forever and "
        f"only the epoch cap ends it"
    )

    report.paragraph(
        "Both runs score a perfect 1.0000 on the rows they were fitted to, which\n"
        "is exactly why that number cannot be the thing you check. converged_ is\n"
        "the attribute that tells you the difference between a model that found\n"
        "an answer and one that merely ran out of patience."
    )


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
