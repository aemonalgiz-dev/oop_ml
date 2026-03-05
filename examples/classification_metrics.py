"""Why a classifier needs four numbers rather than one.

Accuracy is the number everybody reaches for first, and on an unbalanced target
it is close to useless. The data here is a rare event: about one row in eleven
is positive. A model that answers "no" to everything scores 92% and finds
nothing, which is the whole argument for the confusion matrix in one line.

The second half is about the threshold. The model produces a probability, and
turning that into a label needs a cut. Where the cut goes is a decision about
which mistake you would rather make, and the sweep here shows precision and
recall moving in opposite directions as it slides.
"""

from __future__ import annotations

import logging

import numpy as np

from examples.datasets import rare_event
from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import (
    ClassificationEvaluation,
    LogisticRegression,
    TrainTestSplitter,
    UndefinedMetricError,
)

logger = logging.getLogger(__name__)

THRESHOLDS = [0.10, 0.20, 0.30, 0.50, 0.70, 0.90]


def main() -> None:
    report = Report(logger)
    data = rare_event()

    report.detail(
        f"{data.dataset.n_samples} rows, {data.positive_rate:.1%} of them positive"
    )

    split = TrainTestSplitter(test_fraction=0.3, random_seed=7).split(data.dataset)
    model = LogisticRegression(learning_rate=0.5, max_epochs=50_000, tolerance=1e-10)
    model.fit(split.training.input_features, split.training.target_feature)

    held_out = model.evaluate(
        split.testing.input_features, split.testing.target_feature
    )

    report.heading("Accuracy alone would have told you nothing")

    report.confusion("fitted model", held_out)

    always_negative = ClassificationEvaluation(
        split.testing.target_feature,
        np.zeros(split.testing.n_samples),
    )
    report.confusion("answering no every time", always_negative)

    report.paragraph(
        "Two models, and the one that never fires is three points behind on\n"
        "accuracy. The confusion matrices are not three points apart. Accuracy\n"
        "divides by the whole table, so on a target that is 92% negative it\n"
        "mostly measures how big the negative class is."
    )

    try:
        report.line(f"precision of that second model: {always_negative.precision:.4f}")
    except UndefinedMetricError as error:
        report.caught(error)

    report.paragraph(
        "That guard is the reason the metrics raise rather than return nan. A\n"
        "model that never predicted positive has not earned a precision of zero;\n"
        "it has no precision at all, and reporting 0.0 would be answering a\n"
        "question that was never asked."
    )

    report.heading("The threshold is a dial, not a constant")

    probabilities = model.predict_probability(split.testing.input_features)
    actual = split.testing.target_feature

    rows = []
    for threshold in THRESHOLDS:
        evaluation = ClassificationEvaluation(
            actual, (probabilities >= threshold).astype(np.float64)
        )
        matrix = evaluation.confusion_matrix

        rows.append(
            [
                f"{threshold:.2f}",
                str(matrix.true_positives),
                str(matrix.false_positives),
                str(matrix.false_negatives),
                report_metric(evaluation, "precision"),
                report_metric(evaluation, "recall"),
                report_metric(evaluation, "f1_score"),
            ]
        )

    report.table(
        ["threshold", "tp", "fp", "fn", "precision", "recall", "f1"],
        rows,
    )

    report.paragraph(
        "Recall falls as the threshold rises and precision generally climbs,\n"
        "because a stricter cut fires less often: fewer false alarms, more\n"
        "misses. The coefficients never moved. Every row of that table is the\n"
        "same fitted model, read at a different cut."
    )

    report.heading("Pick the cut from the cost of the mistake")

    default_boundary = model.decision_boundary_at("risk_score")
    report.line(f"default threshold : {model.threshold:.2f}")
    report.line(f"  boundary on risk_score : {default_boundary:.4f}")

    cautious = LogisticRegression(
        learning_rate=0.5, max_epochs=50_000, tolerance=1e-10, threshold=0.2
    )
    cautious.fit(split.training.input_features, split.training.target_feature)

    cautious_boundary = cautious.decision_boundary_at("risk_score")
    report.line(f"cautious threshold: {cautious.threshold:.2f}")
    report.line(f"  boundary on risk_score : {cautious_boundary:.4f}")

    report.confusion(
        "cautious model",
        cautious.evaluate(split.testing.input_features, split.testing.target_feature),
    )

    report.paragraph(
        "Accuracy went down and recall went up, which is the trade being made\n"
        "rather than a fault in the fit. Nineteen extra false alarms bought four\n"
        "more of the fifteen real positives, and whether that is a good deal is a\n"
        "question about the application rather than about the model.\n"
        "\n"
        "The threshold is a field, so the cautious model is a different object\n"
        "rather than a different call. It moves the reported boundary too, since\n"
        "the cut is a line drawn on the log-odds scale and lowering it slides\n"
        "that line down the risk_score axis."
    )


def report_metric(evaluation: ClassificationEvaluation, metric_name: str) -> str:
    """One metric formatted, or ``undefined`` where its denominator was empty."""
    try:
        return f"{getattr(evaluation, metric_name):.4f}"
    except UndefinedMetricError:
        return "undefined"


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
