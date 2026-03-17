"""What a walk looks like, for the models that arrive rather than jump.

:mod:`~oop_ml.core.solving.path` records the trajectory an
:class:`~oop_ml.core.base.iterative_solver.IterativeSolver` took -- every pass,
its step, and which of the two exits ended it.

Separate from ``core.base`` because none of it is inherited. A path is a fact
about one run, the way a ``Split`` is a fact about one column, and the four
models that share the walk all produce the same kind of record.
"""
