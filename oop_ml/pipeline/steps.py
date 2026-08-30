"""The transformers a pipeline runs, in order, each one named.

A pipeline's steps are a sequence of transformers applied one after another,
and the order is part of the meaning: standardizing and then expanding to
polynomial terms is not the same model as expanding and then standardizing. A
bare ``list[Transformer]`` carries the order and nothing else -- no way to
address a step, and nothing stopping the same transformer appearing twice with
no way to tell the two apart.

:class:`PipelineStep` is one transformer bound to a name.
:class:`PipelineSteps` is the ordered group, and it owns the two rules that make
a list of transformers a *pipeline*: the names are unique, and the order is
preserved exactly as given.

Why names, when the order already identifies a step
----------------------------------------------------
Because a report reads better and a replacement reads better. ``steps["scaler"]``
says what was asked for where ``steps[0]`` says where it happened to sit, and
:meth:`PipelineSteps.replacing` is what lets a hyperparameter search vary one
step while leaving the rest alone.

What the names are deliberately *not* used for
-----------------------------------------------
scikit-learn addresses a nested hyperparameter with a magic string:
``scaler__with_mean``, a step name and a field name joined by a double
underscore and parsed at runtime. A typo there is a runtime failure at best and
a silently ignored setting at worst, which is the same class of problem
``extra="forbid"`` and :class:`~oop_ml.model_selection.search.ParameterRange`
exist to prevent here.

So this library does not do that. A search over a pipeline varies whole
*objects* -- a list of configured ``PolynomialFeatures`` instances, or a list of
configured models -- and each one was validated by its own constructor when it
was built. The names here are for reading and replacing, never for addressing a
field.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from oop_ml.core.base.estimator import Transformer
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
)


class PipelineStep:
    """One named transformer in a pipeline.

    Parameters
    ----------
    name:
        What this step is called. Used for reading a fitted pipeline and for
        replacing a step, never for addressing a field inside it.
    transformer:
        The transformer to run. Unfitted when the pipeline is built; the
        pipeline fits it, and fits it again from scratch on every ``fit``.

    Raises
    ------
    InvalidValuesError
        If ``name`` is not a non-empty string.
    """

    __slots__ = ("_name", "_transformer")

    def __init__(self, name: str, transformer: Transformer[Sequence[Feature]]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise InvalidValuesError("a pipeline step's name must be non-empty")

        self._name = name.strip()
        self._transformer = transformer

    @property
    def name(self) -> str:
        """What this step is called."""
        return self._name

    @property
    def transformer(self) -> Transformer[Sequence[Feature]]:
        """The transformer this step runs."""
        return self._transformer

    def __repr__(self) -> str:
        return f"PipelineStep({self._name!r}, {type(self._transformer).__name__})"


class PipelineSteps:
    """The transformers to run, in order, with unique names.

    Parameters
    ----------
    steps:
        In the order they should run. May be empty: a pipeline with no
        preprocessing is still a pipeline, and allowing it means the pipeline
        can be the default wrapper rather than a special case a caller has to
        decide about.

    Raises
    ------
    NonUniqueFeaturesError
        If two steps share a name, which would make ``replacing`` ambiguous.
    """

    __slots__ = ("_steps",)

    def __init__(self, steps: Sequence[PipelineStep] = ()) -> None:
        names = [step.name for step in steps]

        if len(set(names)) != len(names):
            raise NonUniqueFeaturesError(
                f"pipeline step names must be unique; got {names}"
            )

        self._steps = tuple(steps)

    @classmethod
    def of(cls, **transformers: Transformer[Sequence[Feature]]) -> PipelineSteps:
        """Build from keyword arguments, which is how it usually reads.

        ``PipelineSteps.of(scaler=Standardizer(), terms=PolynomialFeatures())``
        rather than assembling ``PipelineStep`` objects by hand. Keyword order
        is preserved in Python, and that order is the order the steps run in.
        """
        return cls(
            [
                PipelineStep(name, transformer)
                for name, transformer in transformers.items()
            ]
        )

    @property
    def names(self) -> tuple[str, ...]:
        """What the steps are called, in running order."""
        return tuple(step.name for step in self._steps)

    def value_for(self, name: str) -> PipelineStep:
        """The step called ``name``.

        Raises
        ------
        InvalidValuesError
            If no step has that name.
        """
        for step in self._steps:
            if step.name == name:
                return step

        raise InvalidValuesError(
            f"unknown step {name!r}; this pipeline runs {list(self.names)}"
        )

    def replacing(
        self, name: str, transformer: Transformer[Sequence[Feature]]
    ) -> PipelineSteps:
        """The same steps in the same order, with one transformer swapped.

        Returns a new group rather than mutating, so a search can vary one step
        across candidates without the earlier candidates' transformers leaking
        into the later ones.

        Raises
        ------
        InvalidValuesError
            If no step has that name. A search that misnames a step should stop
            rather than silently leave the pipeline unchanged.
        """
        if name not in self.names:
            raise InvalidValuesError(
                f"unknown step {name!r}; this pipeline runs {list(self.names)}"
            )

        return PipelineSteps(
            [
                PipelineStep(step.name, transformer) if step.name == name else step
                for step in self._steps
            ]
        )

    def check_not_empty(self) -> None:
        """Raise if there is nothing to run.

        Not a constructor invariant, because an empty pipeline is legitimate --
        this is for the callers that need at least one step to mean anything.

        Raises
        ------
        EmptyValuesError
            If there are no steps.
        """
        if not self._steps:
            raise EmptyValuesError("these steps are empty, so nothing would run")

    def __getitem__(self, name: str) -> PipelineStep:
        return self.value_for(name)

    def __contains__(self, name: object) -> bool:
        return any(step.name == name for step in self._steps)

    def __iter__(self) -> Iterator[PipelineStep]:
        return iter(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"PipelineSteps({list(self.names)})"
