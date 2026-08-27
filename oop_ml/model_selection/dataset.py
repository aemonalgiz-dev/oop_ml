"""One partition of a dataset, named rather than positional.

Separate from :class:`~oop_ml.core.data.dataset.Dataset` because the pairing is
vocabulary the whole library speaks and a *partition* is a model_selection
concern -- nothing in ``core`` has any use for the idea of a held-out half.
"""

from __future__ import annotations

from oop_ml.core.data.dataset import Dataset


class DataSplit:
    """One partition of a dataset into a part to learn from and a part to score on.

    Parameters
    ----------
    training:
        The rows the model is allowed to see.
    testing:
        The rows held back, used only to measure how well it generalises.

    Raises
    ------
    EmptyValuesError
        If either half has no rows.
    """

    __slots__ = ("_testing", "_training")

    def __init__(self, training: Dataset, testing: Dataset) -> None:
        self._training = training
        self._testing = testing

    @property
    def training(self) -> Dataset:
        """The rows the model learns from."""
        return self._training

    @property
    def testing(self) -> Dataset:
        """The rows held back for scoring."""
        return self._testing

    @property
    def n_samples(self) -> int:
        """Total rows across both halves."""
        return self._training.n_samples + self._testing.n_samples

    def __repr__(self) -> str:
        return (
            f"DataSplit(training={self._training.n_samples}, "
            f"testing={self._testing.n_samples})"
        )
