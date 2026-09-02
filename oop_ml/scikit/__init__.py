"""The scikit-learn backend: the same encapsulations, a production engine underneath.

Every class exported here has a namesake in :mod:`oop_ml.numpy` with the same
constructor, the same ``fit`` and ``predict``, and the same learned-parameter
properties, so the two interchange at the call site. What differs is only what
does the arithmetic. The numpy backend implements every model by hand, which is
what makes it worth reading; this one wraps scikit-learn, which is what makes it
worth running.

Absence is declared, never silent
----------------------------------
Not every model exists in every backend. scikit-learn has no Hopfield network
and no self-organising map, and this library's teaching-only models have no
production twin worth wrapping. Rather than let such a model be quietly missing
here, every numpy export that this backend does not provide is listed in
:data:`NOT_PROVIDED` with the reason, and a contract test asserts that the two
lists together cover every model exactly once. A model can be present or it can
be declined; it cannot be forgotten.

The backend starts empty. Each family is added behind the same contract suite
the numpy backend already passes, and its entry leaves :data:`NOT_PROVIDED` as
it lands.
"""

from __future__ import annotations

#: Every model the numpy backend exports that this one deliberately does not,
#: with the reason. Read by the contract suite, which refuses a model that is
#: neither provided here nor declined here.
NOT_PROVIDED: dict[str, str] = {
    "SimpleLinearRegression": "not yet wrapped",
    "MultipleLinearRegression": "not yet wrapped",
    "GradientDescentRegression": "not yet wrapped",
    "RidgeRegression": "not yet wrapped",
    "LassoRegression": "not yet wrapped",
    "KernelRidgeRegression": "not yet wrapped",
    "KNearestNeighboursRegressor": "not yet wrapped",
    "DecisionTreeRegressor": "not yet wrapped",
    "BaggingRegressor": "not yet wrapped",
    "RandomForestRegressor": "not yet wrapped",
    "GradientBoostingRegressor": "not yet wrapped",
    "LogisticRegression": "not yet wrapped",
    "NewtonLogisticRegression": "not yet wrapped",
    "MultinomialLogisticRegression": "not yet wrapped",
    "OneVsRestClassifier": "not yet wrapped",
    "KNearestNeighboursClassifier": "not yet wrapped",
    "DecisionTreeClassifier": "not yet wrapped",
    "BaggingClassifier": "not yet wrapped",
    "RandomForestClassifier": "not yet wrapped",
    "SupportVectorClassifier": "not yet wrapped",
    "KMeans": "not yet wrapped",
    "PrincipalComponentAnalysis": "not yet wrapped",
    "KernelPrincipalComponentAnalysis": "not yet wrapped",
    "Standardizer": "not yet wrapped",
    "PolynomialFeatures": "not yet wrapped",
    "MinMaxScaler": "not yet wrapped",
    "MaxAbsScaler": "not yet wrapped",
    "RobustScaler": "not yet wrapped",
    "RestrictedBoltzmannMachine": (
        "not yet wrapped; scikit-learn's BernoulliRBM is the candidate"
    ),
    "SelfOrganisingMap": "scikit-learn has no self-organising map",
    "HopfieldNetwork": "scikit-learn has no associative memory",
    "HebbianPrincipalComponents": (
        "a teaching route to PCA; the production answer is PrincipalComponentAnalysis"
    ),
    "RootMeanSquareScaler": "scikit-learn has no root-mean-square scaler",
}

__all__: list[str] = []
