"""Models that learn nothing, and the three things that decide whether they work.

Every other estimator in this library ends up holding a small set of numbers
that stand in for the data. A neighbour model holds the data. ``fit`` validates
its inputs and remembers them, and every decision waits for ``predict``.

That buys one thing and costs three. What it buys is shape: no assumption is
made about the form of the answer, so a boundary that encloses a region, or a
curve nobody guessed, is read straight off the rows. What it costs is a choice
of metric, a choice of ``k``, and a hard dependence on the units its inputs
happen to be measured in. This script walks all four.

    python -m examples.nearest_neighbours
"""

from __future__ import annotations

import logging

import numpy as np

from examples.datasets import concentric_rings, temperature_by_hour
from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import (
    DistanceMetric,
    Feature,
    KNearestNeighboursClassifier,
    KNearestNeighboursRegressor,
    LogisticRegression,
    Standardizer,
    TrainTestSplitter,
)

logger = logging.getLogger(__name__)


def _shape_no_line_can_cut(report: Report) -> None:
    """A circular boundary, against a model that can only draw straight ones."""
    report.heading("A boundary that is not a hyperplane")

    data = concentric_rings()
    split = TrainTestSplitter(test_fraction=0.3, random_seed=1).split(data)

    report.paragraph(
        "One class sits in a disc, the other in a ring around it. The rings "
        "overlap, so perfect separation is not on offer to anything -- but the "
        "structure is obvious to the eye, and no straight line can exploit it."
    )

    linear = LogisticRegression(learning_rate=0.5, max_epochs=20_000)
    linear.fit(split.training.input_features, split.training.target_feature)
    linear_accuracy = linear.score(
        split.testing.input_features, split.testing.target_feature
    )

    neighbours = KNearestNeighboursClassifier(n_neighbours=5)
    neighbours.fit(split.training.input_features, split.training.target_feature)
    neighbour_accuracy = neighbours.score(
        split.testing.input_features, split.testing.target_feature
    )

    report.table(
        ["model", "test accuracy"],
        [
            ["LogisticRegression", f"{linear_accuracy:.4f}"],
            ["KNearestNeighboursClassifier", f"{neighbour_accuracy:.4f}"],
        ],
    )

    report.warn(
        f"the linear model converged={linear.converged} and scored "
        f"{linear_accuracy:.4f} -- it did not fail, it fitted the best line "
        "available and the best line available is worthless here"
    )


def _the_dial(report: Report) -> None:
    """``k`` is the whole of the bias-variance trade for this family."""
    report.heading("k, and why training accuracy is not evidence")

    data = concentric_rings()
    split = TrainTestSplitter(test_fraction=0.3, random_seed=1).split(data)

    rows = []
    for n_neighbours in (1, 3, 5, 15, 51, split.training.n_samples):
        model = KNearestNeighboursClassifier(n_neighbours=n_neighbours)
        model.fit(split.training.input_features, split.training.target_feature)

        on_training = model.score(
            split.training.input_features, split.training.target_feature
        )
        on_testing = model.score(
            split.testing.input_features, split.testing.target_feature
        )

        rows.append([str(n_neighbours), f"{on_training:.4f}", f"{on_testing:.4f}"])

    report.table(["k", "train accuracy", "test accuracy"], rows)

    report.paragraph(
        "Read the two columns against each other. Training accuracy falls the "
        "whole way down the table while test accuracy rises, peaks, and only "
        "then falls -- so the two disagree about the best model over most of "
        "the range, and the training column is wrong every time they do."
    )
    report.paragraph(
        "At k=1 every training row is its own nearest neighbour, so training "
        "accuracy is exactly 1.0 by construction and carries no information "
        "whatsoever. At the other end every query sees the whole training set "
        "and gets the global majority, which is the most biased model there "
        "is. The useful range is in between, and only the test column finds it."
    )


def _units_decide_the_answer(report: Report) -> None:
    """Distance sums over features, so the widest column wins by default."""
    report.heading("Standardising is part of being correct here")

    data = temperature_by_hour()
    split = TrainTestSplitter(test_fraction=0.3, random_seed=2).split(data)

    raw = KNearestNeighboursRegressor(n_neighbours=5)
    raw.fit(split.training.input_features, split.training.target_feature)
    raw_score = raw.score(split.testing.input_features, split.testing.target_feature)

    standardizer = Standardizer()
    scaled_training = standardizer.fit_transform(split.training.input_features)
    scaled_testing = standardizer.transform(split.testing.input_features)

    scaled = KNearestNeighboursRegressor(n_neighbours=5)
    scaled.fit(scaled_training, split.training.target_feature)
    scaled_score = scaled.score(scaled_testing, split.testing.target_feature)

    report.table(
        ["inputs", "test R^2"],
        [
            ["as supplied", f"{raw_score:.4f}"],
            ["standardised", f"{scaled_score:.4f}"],
        ],
    )

    report.paragraph(
        "The signal is entirely in 'hour', which runs 0 to 24. The noise column "
        "is a pressure in pascals near 101325. Squared distance is a sum over "
        "the columns, so before standardising the pressure column contributes "
        "essentially all of it, and 'nearest' means 'closest in pressure' -- "
        "which is to say, arbitrary."
    )

    horizontal = np.column_stack(
        [feature.values for feature in split.training.input_features]
    )
    gaps = horizontal.std(axis=0) ** 2
    report.detail(
        f"share of the raw squared distance owned by pressure: "
        f"{gaps[1] / gaps.sum():.6f}"
    )


def _the_metric_is_the_model(report: Report) -> None:
    """Six notions of near, and they do not agree."""
    report.heading("Choosing what near means")

    data = temperature_by_hour()
    split = TrainTestSplitter(test_fraction=0.3, random_seed=2).split(data)

    standardizer = Standardizer()
    scaled_training = standardizer.fit_transform(split.training.input_features)
    scaled_testing = standardizer.transform(split.testing.input_features)

    rows = []
    for metric in DistanceMetric:
        model = KNearestNeighboursRegressor(n_neighbours=5, metric=metric)
        model.fit(scaled_training, split.training.target_feature)

        rows.append(
            [
                str(metric),
                f"{model.score(scaled_testing, split.testing.target_feature):.4f}",
            ]
        )

    report.table(["metric", "test R^2"], rows)

    report.paragraph(
        "These are not small differences, and none of them is a bug. Cosine "
        "ignores magnitude, which throws away most of what an hour means. "
        "Hamming asks only whether two values are equal, which on continuous "
        "columns is almost never. The metric is not a tuning knob bolted onto "
        "the model -- for this family it is the model."
    )


def _it_cannot_extrapolate(report: Report) -> None:
    """Past the edge of the data the surface goes flat."""
    report.heading("Beyond the training range")

    data = temperature_by_hour()

    model = KNearestNeighboursRegressor(n_neighbours=5)
    model.fit([data.input_features[0]], data.target_feature)

    beyond = [0.0, 12.0, 24.0, 48.0, 240.0, 10_000.0]
    predictions = model.predict([Feature("hour", beyond)])

    report.table(
        ["hour", "predicted temperature"],
        [
            [f"{hour:.0f}", f"{prediction:.4f}"]
            for hour, prediction in zip(beyond, predictions, strict=True)
        ],
    )

    report.paragraph(
        "The last three rows are identical. Once every neighbour lies on the "
        "same side of the query, the answer is their mean and stays their mean "
        "however far out you go. A linear model would extend its line instead, "
        "confidently and just as unfoundedly. Neither is right; they are "
        "different ways of being wrong, and the flat one at least cannot run "
        "off to negative infinity."
    )


def main() -> None:
    report = Report(logger)

    report.heading("Nearest neighbours")
    report.line("a model that remembers instead of learning")

    _shape_no_line_can_cut(report)
    _the_dial(report)
    _units_decide_the_answer(report)
    _the_metric_is_the_model(report)
    _it_cannot_extrapolate(report)


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
