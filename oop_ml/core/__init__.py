"""The shared frame: everything that is not a model.

Four concerns, and the split is by what a thing *is* rather than what it is
for.

:mod:`~oop_ml.core.data` holds the vocabulary that crosses every boundary -- a
validated column, a column that knows its name, a group of them, and the names
a fit binds its weights to. :mod:`~oop_ml.core.evaluation` holds the three ways
of pairing a prediction with the truth. :mod:`~oop_ml.core.base` holds what a
model inherits from, as opposed to what it is. The remaining three modules are
plumbing every package needs: the type aliases, the exception hierarchy, and
the coercion guards.

The models themselves live one level up, under the task they serve, because a
user looks for "how do I classify things" and not "what is linear in its
coefficients".
"""
