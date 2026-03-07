"""Report output for the examples, addressed to a logger rather than stdout.

The examples print a lot, but they should not call ``print``. A script that
writes to stdout directly cannot be quietened, cannot be redirected, and cannot
say whether a given line is the answer you asked for or a complaint about the
fit. Logging gives all three, and the levels carry real meaning here:

``INFO``
    The report itself -- headings, tables, and the prose saying what to notice.
    This is what the script is *for*, so it is the default level.
``WARNING``
    The modelling is telling you something: a fit that did not converge, a
    held-out score gone negative, a deliberately leaky pipeline.
``ERROR``
    An exception was raised and caught on purpose, to show a guard firing.
``DEBUG``
    Mechanical detail -- shapes, seeds, per-fold scores. Off by default; pass
    ``--verbose`` to a script to see it.

Configuration belongs to the application and never to a module, so nothing here
touches global logging state except :func:`configure_logging`, which the scripts
call from their ``__main__`` block. Importing an example is therefore free of
side effects, which is what lets the test suite run every ``main()`` without
each one fighting over the root logger.

:class:`Report` is an object rather than a module of functions for the same
reason :class:`~oop_ml.evaluation.regression.RegressionEvaluation` is: the
logger is state the formatting needs, and passing it to every call would be a
parameter every caller has to remember.
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser
from collections.abc import Callable, Iterable, Sequence

from oop_ml import (
    ClassificationEvaluation,
    Coefficients,
    CrossValidationResult,
    MLLibError,
    MultiClassEvaluation,
    RegressionEvaluation,
    UndefinedMetricError,
)


class LevelPrefixFormatter(logging.Formatter):
    """Decorate a record only when it is more urgent than ``INFO``.

    The reports are read as text and contain aligned tables, so an ``INFO`` line
    is emitted exactly as written. Anything above it has to interrupt that flow
    to be noticed, and a level prefix on every line of the record is what does
    that -- including for the multi-line messages, where prefixing only the
    first line would let the rest read as ordinary report text.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()

        if record.levelno <= logging.INFO:
            return message

        prefix = f"[{record.levelname.lower()}] "

        return "\n".join(
            prefix + line if line else line for line in message.split("\n")
        )


def configure_logging(level: int = logging.INFO) -> None:
    """Send records to stderr through :class:`LevelPrefixFormatter`.

    Called by the scripts, never on import. Replaces any handlers already on the
    root logger so that running two examples in one process does not double
    every line.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(LevelPrefixFormatter())

    logging.basicConfig(level=level, handlers=[handler], force=True)


def configure_logging_from_command_line() -> None:
    """Configure logging, with ``--verbose`` lowering the level to ``DEBUG``."""
    parser = ArgumentParser(description="Run one of the library's examples.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="include DEBUG detail: shapes, seeds, per-fold scores",
    )

    configure_logging(logging.DEBUG if parser.parse_args().verbose else logging.INFO)


class Report:
    """Formatted output for one example, written to one logger.

    Parameters
    ----------
    logger:
        Where the records go. Each script passes ``logging.getLogger(__name__)``
        so its output is attributable and can be silenced independently.
    """

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def heading(self, title: str) -> None:
        """A titled rule, so a script's sections separate at a glance."""
        self._logger.info("\n%s\n%s", title, "-" * len(title))

    def line(self, message: str) -> None:
        """One fact, one line."""
        self._logger.info("%s", message)

    def paragraph(self, message: str) -> None:
        """Prose saying what to notice, set off by a blank line."""
        self._logger.info("\n%s", message)

    def detail(self, message: str) -> None:
        """Mechanical detail, shown only under ``--verbose``."""
        self._logger.debug("%s", message)

    def warn(self, message: str) -> None:
        """Something about the fit or the data that deserves attention."""
        self._logger.warning("%s", message)

    def caught(self, error: MLLibError) -> None:
        """Report a guard that fired, having been provoked on purpose."""
        self._logger.error("%s: %s", type(error).__name__, error)

    def table(self, column_names: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
        """Log rows under right-aligned headers, sized to the widest cell.

        Emitted as a single record so that nothing can interleave between the
        header and its rows.
        """
        materialised = [list(row) for row in rows]
        widths = [
            max(len(str(name)), *(len(str(row[position])) for row in materialised))
            if materialised
            else len(str(name))
            for position, name in enumerate(column_names)
        ]

        lines = [self._justified(column_names, widths)]
        lines.extend(self._justified(row, widths) for row in materialised)

        self._logger.info("%s", "\n".join(lines))

    @staticmethod
    def _justified(cells: Sequence[str], widths: Sequence[int]) -> str:
        return "  ".join(
            str(cell).rjust(width) for cell, width in zip(cells, widths, strict=True)
        )

    def coefficients(
        self,
        learned: Coefficients,
        truth: Coefficients,
        learned_intercept: float,
        true_intercept: float,
    ) -> None:
        """Show each learned weight beside the value that generated the data."""
        rows = [["intercept", f"{true_intercept:.3f}", f"{learned_intercept:.3f}"]]
        rows.extend(
            [
                coefficient.name,
                f"{truth.value_for(coefficient.name):.3f}",
                f"{coefficient.value:.3f}",
            ]
            for coefficient in learned
        )

        self.table(["term", "true", "fitted"], rows)

    def evaluation(self, label: str, evaluation: RegressionEvaluation) -> None:
        """One line of performance: the three numbers worth reading together."""
        self.line(
            f"{label}: R2={evaluation.r2_score:.4f}  "
            f"MSE={evaluation.mean_squared_error:.4f}  "
            f"n={evaluation.n_samples}"
        )

        if evaluation.r2_score < 0.0:
            self.warn(
                f"{label} R2 is negative: this model is worse on these rows than "
                f"predicting their mean and ignoring every feature."
            )

    def confusion(self, label: str, evaluation: ClassificationEvaluation) -> None:
        """The four counts as a table, then the metrics derived from them.

        Any metric whose denominator is empty is reported as ``undefined``
        rather than being skipped, since a model that never fired having no
        precision is itself worth seeing.
        """
        matrix = evaluation.confusion_matrix

        self.line(f"{label}: {matrix.n_samples} rows")
        self.table(
            ["", "predicted 1", "predicted 0"],
            [
                ["actual 1", str(matrix.true_positives), str(matrix.false_negatives)],
                ["actual 0", str(matrix.false_positives), str(matrix.true_negatives)],
            ],
        )
        self.line(
            f"accuracy={evaluation.accuracy:.4f}  "
            f"precision={self._or_undefined(lambda: evaluation.precision)}  "
            f"recall={self._or_undefined(lambda: evaluation.recall)}  "
            f"f1={self._or_undefined(lambda: evaluation.f1_score)}"
        )

    @staticmethod
    def _or_undefined(metric: Callable[[], float]) -> str:
        """Format a metric, or say so when its denominator was empty."""
        try:
            return f"{metric():.4f}"
        except UndefinedMetricError:
            return "undefined"

    def class_table(self, label: str, evaluation: MultiClassEvaluation) -> None:
        """The K x K table, then each class's own precision and recall.

        Per class rather than pooled, because that is the fact a single number
        hides: a model can be excellent on the common class and useless on the
        rare one and still look respectable overall.
        """
        matrix = evaluation.confusion_matrix
        classes = range(evaluation.n_classes)

        self.line(f"{label}: {matrix.n_samples} rows, {evaluation.n_classes} classes")
        self.table(
            ["actual vs predicted"] + [f"as {index}" for index in classes],
            [
                [f"class {row}"]
                + [str(matrix.counts[row][column]) for column in classes]
                for row in classes
            ],
        )
        self.table(
            ["class", "rows", "precision", "recall", "f1"],
            [
                [
                    str(index),
                    str(matrix.actually_are(index)),
                    self._or_undefined(
                        lambda index=index: evaluation.precision_for(index)
                    ),
                    self._or_undefined(
                        lambda index=index: evaluation.recall_for(index)
                    ),
                    self._or_undefined(lambda index=index: evaluation.f1_for(index)),
                ]
                for index in classes
            ],
        )

    def cross_validation(self, label: str, result: CrossValidationResult) -> None:
        """Mean and spread together -- the mean alone is half the story."""
        self.line(
            f"{label}: mean R2={result.mean_r2_score:.4f}  "
            f"spread={result.r2_score_spread:.4f}  "
            f"folds={result.n_folds}"
        )
        self.detail(
            "  fold scores: " + ", ".join(f"{fold.r2_score:.4f}" for fold in result)
        )
