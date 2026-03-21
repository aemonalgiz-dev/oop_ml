"""What an ensemble is built from, none of it a model.

:mod:`~oop_ml.core.ensemble.bootstrap` is the resample that makes members
differ, and the out-of-bag rows that fall out of it for free.
:mod:`~oop_ml.core.ensemble.fits` is what a fitted ensemble looks like from the
inside -- the members and their disagreement for the averaging family, the
ordered rounds and their shrinking residuals for the boosting one.

Two records rather than one, because the families disagree about what a member
is. Averaging members are peers and could have been fitted in any order;
boosting members are a sequence where order is the entire structure.
"""
