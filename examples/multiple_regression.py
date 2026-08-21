"""Several predictors, and why the library insists they carry names.

``MultipleLinearRegression`` takes ``Sequence[Feature]`` rather than a matrix.
That is the central API decision in the library, and this example is where it
pays: the learned weights come back as
:class:`~oop_ml.core.coefficients.Coefficients`, keyed by the name of the column that
produced them, and ``predict`` matches by name too.

So reordering the features between fit and predict is harmless here, where in an
array-in/array-out library it silently produces confident nonsense. The last
section demonstrates exactly that, and also the limit of the guarantee.
"""

from __future__ import annotations

import logging

import numpy as np

from examples.datasets import independent_predictors
from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import MLLibError, MultipleLinearRegression

logger = logging.getLogger(__name__)


def main() -> None:
    report = Report(logger)
    data = independent_predictors()

    report.detail(
        f"{data.dataset.n_samples} rows, {data.dataset.n_features} predictors: "
        f"{[feature.name for feature in data.input_features]}"
    )

    report.heading("Fitting three predictors at once")

    model = MultipleLinearRegression()
    model.fit(data.input_features, data.target_feature)

    report.coefficients(
        model.coefficients_,
        data.true_coefficients,
        model.intercept_,
        data.true_intercept,
    )
    report.evaluation(
        "in-sample", model.evaluate(data.input_features, data.target_feature)
    )

    report.heading("Coefficients are addressed by name, never by position")

    report.line(f"rooms       : {model.coefficients_['rooms']:.3f} per room")
    report.line(
        f"age_years   : {model.coefficients_.value_for('age_years'):.3f} per year"
    )
    report.line(f"'rooms' known to the model : {'rooms' in model.coefficients_}")
    report.line(f"'garden' known to the model: {'garden' in model.coefficients_}")

    report.paragraph(
        "Each weight is a *partial* effect: the change in y per unit of that\n"
        "feature with the others held fixed. That is why they are not the same\n"
        "numbers you would get from three separate simple regressions."
    )

    report.heading("Order does not matter, presence does")

    shuffled = list(reversed(data.input_features))
    report.line(f"fitted order  : {[feature.name for feature in data.input_features]}")
    report.line(f"predict order : {[feature.name for feature in shuffled]}")

    as_fitted = model.predict(data.input_features)
    as_shuffled = model.predict(shuffled)
    largest_difference = float(np.max(np.abs(as_fitted - as_shuffled)))

    report.line(f"agree to floating point : {np.allclose(as_fitted, as_shuffled)}")
    report.line(f"bitwise identical       : {np.array_equal(as_fitted, as_shuffled)}")
    report.line(f"largest difference      : {largest_difference:.2e}")

    report.paragraph(
        "Every weight was matched to its own column by name, so the arithmetic\n"
        "is the same arithmetic. What changed is the order the terms were added\n"
        "up, and floating point addition is not associative -- hence a few bits\n"
        "of difference and no more. Correctness here is `allclose`, not `==`."
    )

    # A missing column, by contrast, makes the hyperplane unevaluable -- so it is
    # an error rather than a silently different answer.
    report.paragraph("Dropping a feature is refused rather than approximated:")
    try:
        model.predict(data.input_features[:2])
    except MLLibError as error:
        report.caught(error)


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
