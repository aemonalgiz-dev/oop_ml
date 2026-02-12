"""Fitting a curve without changing the model at all.

"Linear" in linear regression means linear in the *coefficients*, never in the
predictors. So ``x ** 2`` is an ordinary column, and a curve is fitted by the
same normal equations as a straight line. That is why ``PolynomialFeatures``
lives in ``preprocessing`` and not in ``regression``: it hands the estimator more
columns and the estimator is untouched.

The second half is the cost. Degree is a complexity dial, and turning it up
always improves the fit you can measure while eventually destroying the one you
care about.
"""

from __future__ import annotations

import logging

from examples.datasets import independent_predictors, quadratic_curve
from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import MultipleLinearRegression, PolynomialFeatures, TrainTestSplitter

logger = logging.getLogger(__name__)

DEGREES = [1, 2, 3, 5, 7, 9]


def main() -> None:
    report = Report(logger)
    data = quadratic_curve()

    report.heading("A straight line through a curve")

    straight = MultipleLinearRegression()
    straight.fit(data.input_features, data.target_feature)
    report.line(f"R2 = {straight.score(data.input_features, data.target_feature):.4f}")
    report.line("the best line through a parabola, and still the wrong shape")

    report.heading("The same model, given x^2 as well")

    expansion = PolynomialFeatures(degree=2)
    expanded_features = expansion.fit_transform(data.input_features)
    report.line(f"columns produced: {list(expansion.terms_.names)}")

    curved = MultipleLinearRegression()
    curved.fit(expanded_features, data.target_feature)

    report.coefficients(
        curved.coefficients_,
        data.true_coefficients,
        curved.intercept_,
        data.true_intercept,
    )
    report.line(f"\nR2 = {curved.score(expanded_features, data.target_feature):.4f}")
    report.line("same estimator, same solve, different design matrix")

    report.heading("Interactions, when there is more than one predictor")

    two_features = independent_predictors().input_features[:2]
    with_interactions = PolynomialFeatures(degree=2, include_interactions=True)
    with_interactions.fit(two_features)
    powers_only = PolynomialFeatures(degree=2, include_interactions=False)
    powers_only.fit(two_features)

    report.line(f"with interactions : {list(with_interactions.terms_.names)}")
    report.line(f"powers only       : {list(powers_only.terms_.names)}")
    report.paragraph(
        "The cross term is the only way a linear model can say 'the effect of\n"
        "one predictor depends on the level of another'. Without it the model\n"
        "insists every effect is the same regardless of the others."
    )

    report.heading("What degree costs: train score against held-out score")

    split = TrainTestSplitter(test_fraction=0.4, random_seed=2).split(data.dataset)
    report.detail(
        f"{split.training.n_samples} training rows, {split.testing.n_samples} held out"
    )

    rows = []
    concerns = []
    for degree in DEGREES:
        # Fitted inside the split, never across it -- the expansion learns which
        # columns exist, and it learns that from the training features alone.
        expansion = PolynomialFeatures(degree=degree)
        training_features = expansion.fit_transform(split.training.input_features)
        testing_features = expansion.transform(split.testing.input_features)

        model = MultipleLinearRegression()
        model.fit(training_features, split.training.target_feature)

        training_score = model.score(training_features, split.training.target_feature)
        testing_score = model.score(testing_features, split.testing.target_feature)

        if testing_score < 0.0:
            concerns.append(
                f"degree {degree} scores {testing_score:.4f} on held-out rows "
                f"while scoring {training_score:.4f} on the rows it was fitted "
                f"to -- it is now worse than predicting the mean"
            )

        rows.append(
            [
                str(degree),
                str(expansion.terms_.n_terms),
                f"{training_score:.4f}",
                f"{testing_score:.4f}",
            ]
        )

    report.table(["degree", "terms", "train R2", "test R2"], rows)

    # Emitted after the table, so a reader has the numbers in front of them
    # before being told which rows to worry about.
    for concern in concerns:
        report.warn(concern)

    report.paragraph(
        "Train R2 only ever rises -- it cannot do otherwise, since each degree\n"
        "adds freedom to a model that minimises exactly that number. Test R2 is\n"
        "the one with an opinion. The gap between the two columns is the whole\n"
        "argument for holding data back."
    )


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
