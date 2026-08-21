"""Models that predict which class a row belongs to.

The directory names the task, not the model family, which is the whole reason
this sits beside ``oop_ml.regression`` rather than under it. Logistic regression
is linear in the coefficients and reuses every piece of
:class:`~oop_ml.core.linear_model.LinearModel` that the regressors use, while
not being regression at all.
"""
