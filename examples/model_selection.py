"""The whole arc: hold out, cross-validate, choose, refit, report once.

This is the example the rest build toward, because it is the only one that
answers the question a user actually has -- *how well will this do on data I have
not collected yet* -- without cheating.

Four stages, and each one throws away what the last one built:

1. **Hold out** a test set and do not look at it again until stage 4.
2. **Cross-validate** every candidate on the training part. Produces one score
   per candidate; all the fitted models and all the folds are discarded.
3. **Choose** a penalty. Now even the scores are discarded -- the only thing
   carried forward is a single number.
4. **Refit once** on all the training rows at that penalty and score on the
   held-out rows exactly once. That number is the claim.

Why stage 4 needs its own untouched data: by stage 3 the cross-validation mean
has been used to *select*, so it is no longer an unbiased estimate of anything.
It ranks candidates well and flatters whichever one it crowned.

The data is deliberately the shape where a penalty earns its keep -- thirty
correlated predictors over eighty-odd training rows. On easy, well-conditioned
data the honest answer is that penalty=0 wins and the whole exercise only
confirms it.
"""

from __future__ import annotations

import logging
import statistics

from examples.datasets import wide_correlated_design
from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import (
    CrossValidation,
    CrossValidationResult,
    KFold,
    RidgeRegression,
    TrainTestSplitter,
)

logger = logging.getLogger(__name__)

CANDIDATE_PENALTIES = [0.0, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0]


def standard_error_of_mean_r2(result: CrossValidationResult) -> float:
    """Standard error of the fold R^2 scores.

    ``CrossValidationResult`` exposes the mean and the max-minus-min spread; the
    standard error is derived here by iterating the folds, which the result
    object supports directly. The spread answers "should I trust this mean at
    all"; the standard error answers "is this mean distinguishable from that
    one", and those are different questions with different scales.
    """
    fold_scores = [evaluation.r2_score for evaluation in result]

    return statistics.stdev(fold_scores) / len(fold_scores) ** 0.5


def main() -> None:
    report = Report(logger)
    data = wide_correlated_design()

    report.heading("Stage 1: hold out a test set, then forget it exists")

    split = TrainTestSplitter(test_fraction=0.25, random_seed=0).split(data.dataset)
    report.line(f"features      : {split.training.n_features}")
    report.line(f"training rows : {split.training.n_samples}")
    report.line(f"testing rows  : {split.testing.n_samples}  (untouched until stage 4)")

    report.heading("Stage 2: cross-validate each candidate on the training rows")

    cross_validation = CrossValidation(folds=KFold(n_folds=5, random_seed=0))

    results = {}
    rows = []
    for penalty in CANDIDATE_PENALTIES:
        result = cross_validation.evaluate(
            RidgeRegression(penalty=penalty), split.training
        )
        results[penalty] = result
        report.cross_validation(f"penalty={penalty:<6g}", result)

        rows.append(
            [
                f"{penalty:g}",
                f"{result.mean_r2_score:.4f}",
                f"{standard_error_of_mean_r2(result):.4f}",
                f"{result.r2_score_spread:.4f}",
            ]
        )

    report.table(["penalty", "mean R2", "std error", "spread"], rows)

    report.paragraph(
        "Thirty-five fits, and every model from them is now discarded. What\n"
        "survives each candidate is one mean. Note that the mean peaks *above*\n"
        "penalty 0 -- on this data least squares is genuinely beaten, which is\n"
        "the only situation where choosing a penalty is a real decision."
    )

    report.heading("Stage 3: choose, using the spread to say how sure you are")

    best_penalty = max(results, key=lambda penalty: results[penalty].mean_r2_score)
    best_result = results[best_penalty]
    best_standard_error = standard_error_of_mean_r2(best_result)

    report.line(
        f"best mean R2 : {best_result.mean_r2_score:.4f} at penalty={best_penalty:g}"
    )
    report.line(f"std error    : {best_standard_error:.4f}")

    # A difference smaller than the noise in the measurement is not a
    # difference. Among candidates the experiment cannot separate, the most
    # regularised is the safer bet on data nobody has seen.
    threshold = best_result.mean_r2_score - best_standard_error
    indistinguishable = [
        penalty
        for penalty, result in results.items()
        if result.mean_r2_score >= threshold
    ]
    chosen_penalty = max(indistinguishable)

    report.line(
        f"within one std error: {[f'{penalty:g}' for penalty in indistinguishable]}"
    )
    report.line(f"choosing penalty={chosen_penalty:g} -- the most regularised of those")

    # The spread is a trust diagnostic, not a selection threshold. Used as one it
    # degenerates: it is wide enough here to admit every candidate on the list,
    # including the ones the means clearly rank last.
    spread_threshold = best_result.mean_r2_score - best_result.r2_score_spread
    admitted_by_spread = [
        penalty
        for penalty, result in results.items()
        if result.mean_r2_score >= spread_threshold
    ]
    if len(admitted_by_spread) == len(results):
        report.warn(
            f"selecting on the max-minus-min spread instead would have admitted "
            f"all {len(results)} candidates and picked "
            f"penalty={max(admitted_by_spread):g}, the worst of them. The spread "
            f"says how much the folds disagree, not how precisely the mean is "
            f"pinned down -- select on the standard error."
        )

    report.heading("Stage 4: refit once, and score once")

    final_model = RidgeRegression(penalty=chosen_penalty)
    final_model.fit(split.training.input_features, split.training.target_feature)

    report.evaluation(
        f"held-out (penalty={chosen_penalty:g})",
        final_model.evaluate(
            split.testing.input_features, split.testing.target_feature
        ),
    )

    # For contrast only -- in real work you would not fit this, because each
    # extra look at the test set spends a little more of its independence.
    unpenalised = RidgeRegression(penalty=0.0)
    unpenalised.fit(split.training.input_features, split.training.target_feature)
    report.evaluation(
        "held-out (penalty=0)",
        unpenalised.evaluate(
            split.testing.input_features, split.testing.target_feature
        ),
    )

    in_sample_score = final_model.score(
        split.training.input_features, split.training.target_feature
    )
    report.line(f"\nin-sample R2 for comparison: {in_sample_score:.4f}")
    report.paragraph(
        "The held-out number is the one to quote. The in-sample number is what\n"
        "the model was built to maximise, so it cannot be evidence of anything.\n"
        "The cross-validation mean sits between them: honest about each fold,\n"
        "but spent the moment it was used to pick a winner."
    )


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
