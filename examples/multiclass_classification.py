"""Three classes, two routes to them, and why one number is not a score.

Binary classification asks which side of a boundary a row falls on. With more
than two classes the question changes shape: the model has to produce a
probability per class, and those probabilities have to be a distribution.

Two ways to get there. Softmax gives every class its own weight vector and
normalises across them, so the answer is a distribution by construction.
One-vs-rest fits an independent binary model per class and compares their
answers, which needs no new mathematics and offers no such guarantee. This
script fits both on the same data and reports where they differ.

The second half is about scoring. On an unbalanced target a single accuracy
figure is close to useless, and macro and micro averaging are two honest
answers to the same question that can be twenty points apart.
"""

from __future__ import annotations

import logging

import numpy as np

from examples.datasets import iris_like_species
from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import (
    LogisticRegression,
    MultinomialLogisticRegression,
    OneVsRestClassifier,
    Standardizer,
)

logger = logging.getLogger(__name__)


def main() -> None:
    report = Report(logger)
    data = iris_like_species()
    counts = np.bincount(data.target_feature.values.astype(int))

    report.detail(
        f"{data.dataset.n_samples} rows over {len(counts)} classes, "
        f"counts {counts.tolist()}"
    )

    standardizer = Standardizer()
    scaled = standardizer.fit_transform(data.input_features)

    report.heading("Softmax: one model, one weight vector per class")

    softmax_model = MultinomialLogisticRegression(
        learning_rate=1.0, max_epochs=200_000, tolerance=1e-9
    )
    softmax_model.fit(scaled, data.target_feature)

    report.line(
        f"converged : {softmax_model.converged} after {softmax_model.epochs_run} epochs"
    )
    report.table(
        ["class", "intercept", "sepal_length", "sepal_width"],
        [
            [
                str(index),
                f"{softmax_model.intercepts[index]:+.4f}",
                f"{softmax_model.coefficients_for(index)['sepal_length']:+.4f}",
                f"{softmax_model.coefficients_for(index)['sepal_width']:+.4f}",
            ]
            for index in range(softmax_model.n_classes)
        ],
    )
    report.paragraph(
        "Class 0 is all zeros, and not because the fit failed. Adding the same\n"
        "constant to every class's weight for a feature leaves every\n"
        "probability unchanged, so the likelihood has a flat ridge running\n"
        "through it and no unique maximum. Holding one class at zero pins that\n"
        "ridge down. Every other class's weights then read as against class 0."
    )

    report.heading("One-vs-rest: three models that were never introduced")

    wrapper = OneVsRestClassifier(
        binary_model=LogisticRegression(
            learning_rate=0.5, max_epochs=200_000, tolerance=1e-9
        )
    )
    wrapper.fit(scaled, data.target_feature)

    report.table(
        ["fitted model", "sepal_length", "sepal_width", "intercept"],
        [
            [
                f"class {index} vs rest",
                f"{wrapper.model_for(index).coefficients['sepal_length']:+.4f}",
                f"{wrapper.model_for(index).coefficients['sepal_width']:+.4f}",
                f"{wrapper.model_for(index).intercept:+.4f}",
            ]
            for index in range(wrapper.n_classes)
        ],
    )
    report.paragraph(
        "One model per class, each fitted on every feature. The features never\n"
        "split up; only the target is recoded to is-it-this-class. Each fit\n"
        "also sees a deliberately unbalanced problem, and the rarest class\n"
        "draws the hardest imbalance of the three."
    )

    softmax_probabilities = softmax_model.predict_probabilities(scaled)
    wrapper_probabilities = wrapper.predict_probabilities(scaled)
    softmax_totals = softmax_probabilities.sum(axis=1)
    wrapper_totals = wrapper_probabilities.sum(axis=1)
    worst = int(np.argmax(np.abs(wrapper_totals - 1.0)))

    report.table(
        ["model", "row sum: min", "median", "max"],
        [
            [
                "softmax",
                f"{softmax_totals.min():.4f}",
                f"{np.median(softmax_totals):.4f}",
                f"{softmax_totals.max():.4f}",
            ],
            [
                "one-vs-rest",
                f"{wrapper_totals.min():.4f}",
                f"{np.median(wrapper_totals):.4f}",
                f"{wrapper_totals.max():.4f}",
            ],
        ],
    )
    report.line(
        f"worst row {worst}: one-vs-rest "
        f"{np.round(wrapper_probabilities[worst], 3).tolist()} "
        f"sums to {wrapper_totals[worst]:.4f}"
    )
    report.line(
        f"             softmax     "
        f"{np.round(softmax_probabilities[worst], 3).tolist()} "
        f"sums to {softmax_totals[worst]:.4f}"
    )
    report.paragraph(
        "The one-vs-rest rows do not sum to one, and normalising them would\n"
        "not fix it. Three unrelated opinions divided by their total add up to\n"
        "one without being the probability of anything. The library reports\n"
        "them raw and says so, because a row summing to 1.3 is two models both\n"
        "confident, which is a fact about the fit worth seeing."
    )

    disagreements = int(
        (softmax_model.predict(scaled) != wrapper.predict(scaled)).sum()
    )
    report.line(
        f"the two disagree on the predicted class for {disagreements} of "
        f"{data.dataset.n_samples} rows"
    )

    report.heading("One accuracy figure is not a score")

    evaluation = softmax_model.evaluate(scaled, data.target_feature)
    report.class_table("softmax", evaluation)

    report.table(
        ["measure", "value"],
        [
            ["accuracy", f"{evaluation.accuracy:.4f}"],
            ["micro precision", f"{evaluation.micro_precision:.4f}"],
            ["micro recall", f"{evaluation.micro_recall:.4f}"],
            ["macro precision", f"{evaluation.macro_precision:.4f}"],
            ["macro recall", f"{evaluation.macro_recall:.4f}"],
            ["macro F1", f"{evaluation.macro_f1_score:.4f}"],
        ],
    )
    report.paragraph(
        "Accuracy, micro precision and micro recall are the same number, and\n"
        "always will be. Every row gets exactly one prediction, so the pooled\n"
        "numerator is the diagonal and both pooled denominators are every row.\n"
        "Reporting all three is reporting one number three times."
    )
    report.paragraph(
        "Macro is the one that carries information the others do not. It\n"
        "averages the per-class scores, so the class holding 15% of the rows\n"
        "moves it exactly as much as the class holding 55%. Where macro recall\n"
        "sits well below micro, the model is doing badly on a class that is\n"
        "too small to dent the overall figure -- which is usually the class\n"
        "somebody cared about."
    )


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
