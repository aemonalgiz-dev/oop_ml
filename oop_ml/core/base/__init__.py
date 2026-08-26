"""What a model inherits from, as opposed to what it is.

Four layers, and they answer different questions.

:mod:`~oop_ml.core.base.estimator` is the contract: Fittable tracks whether a fit
has happened, Estimator learns against a target, Transformer learns from the
inputs alone, and Regressor, Classifier and MultiClassClassifier each say what
answering looks like for one task.

:mod:`~oop_ml.core.base.linear_model` is the machinery every model that is linear in
its coefficients shares -- the design matrix, the intercept split, pairing
weights with the names they came from -- and it deliberately spans two tasks,
because "linear" describes the coefficients and not the question.

:mod:`~oop_ml.core.base.iterative_solver` is the walk shared by every model that
arrives at an answer rather than jumping to it, where the only thing a subclass
supplies is the step.

:mod:`~oop_ml.core.base.neighbour_model` is the frame for the models that learn
nothing at all. It sits beside ``linear_model`` rather than under it because it
shares none of that machinery -- no design matrix, no intercept, no
coefficients to pair with names -- and like ``linear_model`` it spans two tasks,
since finding the nearest rows is the same work whether the target is a label or
a quantity. What differs is one line, and that line is all a subclass supplies.

Nothing here is a model. The models live under the task they serve.
"""
