"""Putting columns on a common scale, and the discipline that comes with it.

Least squares does not need this. Rescale a column and the closed form rescales
its coefficient to match, leaving every prediction identical -- ordinary
regression is scale equivariant, so ``Standardizer`` would be a no-op in front
of it.

Two things break that equivariance, and both appear in this library:

* **A penalty**, which prices coefficients against each other. A weight measured
  per square metre and a weight measured per bathroom are not comparable
  quantities, so a single penalty on both is arbitrary.
* **A single learning rate** in gradient descent, which has to serve every
  direction at once.

Standardizing is therefore about the *other* parts of the pipeline. And it
introduces the rule that governs everything learned from data: the mean and the
spread are learned parameters, so they must come from the training rows alone.
"""

from __future__ import annotations

import logging

from examples.datasets import mixed_units
from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import RidgeRegression, Standardizer, TrainTestSplitter

logger = logging.getLogger(__name__)

PENALTIES = [0.0, 1.0, 10.0, 100.0]


def main() -> None:
    report = Report(logger)
    data = mixed_units()

    report.heading("Two columns, two very different scales")

    standardizer = Standardizer()
    standardizer.fit(data.input_features)

    report.table(
        ["feature", "mean", "sd"],
        [
            [scaling.name, f"{scaling.mean:.3f}", f"{scaling.standard_deviation:.3f}"]
            for scaling in standardizer.scalings_
        ],
    )

    report.paragraph(
        "A one-unit change in floor_area_sqm is a rounding error; a one-unit\n"
        "change in bathrooms is a quarter of the whole range. The coefficients\n"
        "inherit that asymmetry."
    )

    report.heading("Why a penalty cares")

    scaled_features = standardizer.transform(data.input_features)

    rows = []
    for penalty in PENALTIES:
        raw_model = RidgeRegression(penalty=penalty)
        raw_model.fit(data.input_features, data.target_feature)
        scaled_model = RidgeRegression(penalty=penalty)
        scaled_model.fit(scaled_features, data.target_feature)

        rows.append(
            [
                f"{penalty:g}",
                f"{raw_model.coefficients_['floor_area_sqm']:.4f}",
                f"{raw_model.coefficients_['bathrooms']:.4f}",
                f"{scaled_model.coefficients_['floor_area_sqm']:.4f}",
                f"{scaled_model.coefficients_['bathrooms']:.4f}",
            ]
        )

    report.table(["penalty", "raw area", "raw baths", "std area", "std baths"], rows)

    report.paragraph(
        "On the raw columns the penalty barely touches floor_area_sqm -- its\n"
        "coefficient is already tiny, because the column is large -- while it\n"
        "leans hard on bathrooms. Standardized, both are penalised on the same\n"
        "terms, which is the only reading of 'penalty=1.0' that means anything."
    )

    report.heading("Fit on train, transform on test -- and why that order")

    split = TrainTestSplitter(test_fraction=0.3, random_seed=4).split(data.dataset)

    honest = Standardizer()
    honest.fit(split.training.input_features)
    honest_model = RidgeRegression(penalty=1.0)
    honest_model.fit(
        honest.transform(split.training.input_features), split.training.target_feature
    )
    honest_score = honest_model.score(
        honest.transform(split.testing.input_features), split.testing.target_feature
    )

    # The leaky version: mean and sd computed over every row, test rows
    # included, before the split is respected.
    leaky = Standardizer()
    leaky.fit(data.input_features)
    leaky_model = RidgeRegression(penalty=1.0)
    leaky_model.fit(
        leaky.transform(split.training.input_features), split.training.target_feature
    )
    leaky_score = leaky_model.score(
        leaky.transform(split.testing.input_features), split.testing.target_feature
    )

    report.line(f"fitted on training rows only : test R2 = {honest_score:.6f}")
    report.line(f"fitted on every row (leaky)  : test R2 = {leaky_score:.6f}")

    report.warn(
        "the second number is not reportable: its standardizer saw the test "
        "rows' mean and spread before the model was fitted, so the held-out "
        "score is no longer held out"
    )

    report.paragraph(
        "The gap here is small, and saying so is more useful than pretending\n"
        "otherwise: a mean and a standard deviation leak very little. What\n"
        "matters is the mechanism, because the same mistake made with anything\n"
        "that touches the target -- feature selection, target encoding, imputing\n"
        "from the response -- is not small at all.\n\n"
        "Note also what the honest version needed: the standardizer had to be\n"
        "refitted inside the split by hand. Nothing in the type system stopped\n"
        "the leaky version. That is the hole a Pipeline object closes."
    )


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
