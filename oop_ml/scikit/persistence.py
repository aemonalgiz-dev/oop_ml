"""Which wrapped models a document may name, and why the rest cannot.

The contract is the plain one: a document written by this backend loads as
this backend's model. Nothing is scaffolded into a wrapper to make it resemble
its from-scratch namesake, and nothing here reads the other backend's learned
state.

What decides whether a wrapper can be saved
--------------------------------------------
Whether it still needs its engine to answer. Nineteen of these read what they
need off the fitted engine and let it go, so what they hold afterwards is the
whole of the fitted model and restoring it restores the model. The other
eighteen keep the engine and predict through it, and an engine cannot travel in
this format: a fitted decision tree is a Cython structure with no public way
back in, and the way that does exist is pickle, which this format was built to
avoid. Restoring those from their learned state would hand back something that
looks fitted and raises on the first prediction, which is worse than refusing.

So they decline by name and say why, the same way an absent model does in
:data:`~oop_ml.scikit.NOT_PROVIDED`. A backend can decline; it cannot forget,
and a test holds this list against the exported one.

Where that leaves a caller who wants both speed and a saved model
------------------------------------------------------------------
Fit with the from-scratch namesake, which persists all forty of its
models, or persist the engine with the tool scikit-learn ships for it. The
refusal says so rather than leaving it to be worked out.
"""

from __future__ import annotations

from pydantic import BaseModel

from oop_ml.core.persistence.registry import register_backend
from oop_ml.core.schedule import (
    ConstantSchedule,
    ExponentialDecaySchedule,
    LinearDecaySchedule,
)
from oop_ml.scikit.classification import (
    GaussianNaiveBayes,
    LinearDiscriminantAnalysis,
    LogisticRegression,
    MultinomialLogisticRegression,
    NewtonLogisticRegression,
    QuadraticDiscriminantAnalysis,
)
from oop_ml.scikit.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    Standardizer,
)
from oop_ml.scikit.regression import (
    ElasticNetRegression,
    HuberRegression,
    LassoRegression,
    MultipleLinearRegression,
    RidgeRegression,
)
from oop_ml.scikit.unsupervised import (
    DBSCAN,
    AgglomerativeClustering,
    GaussianMixture,
    RestrictedBoltzmannMachine,
)

SCIKIT_BACKEND = "scikit"
"""What a document written by this backend records as its origin."""

PERSISTABLE_TYPES: dict[str, type[BaseModel]] = {
    persistable.__name__: persistable
    for persistable in (
        # linear, which read their coefficients off the engine and release it
        MultipleLinearRegression,
        RidgeRegression,
        LassoRegression,
        ElasticNetRegression,
        HuberRegression,
        LogisticRegression,
        NewtonLogisticRegression,
        MultinomialLogisticRegression,
        # generative, whose fitted self is three summaries per class
        GaussianNaiveBayes,
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
        # preprocessing, which learn a pair of numbers per column
        Standardizer,
        MinMaxScaler,
        MaxAbsScaler,
        RobustScaler,
        # clustering, whose fitted self is the rows it saw and their labels
        DBSCAN,
        AgglomerativeClustering,
        GaussianMixture,
        # generative, whose fitted self is a weight block and two biases
        RestrictedBoltzmannMachine,
        # configuration a model may hold, encoded inside its document
        ConstantSchedule,
        LinearDecaySchedule,
        ExponentialDecaySchedule,
    )
}
"""Every wrapper whose own state is the whole of its fitted self."""

NEEDS_ITS_ENGINE = (
    "predicts through the fitted scikit-learn estimator, which cannot travel "
    "in this format. Fit the oop_ml.numpy namesake to save a model, or persist "
    "the estimator with the tool scikit-learn ships for it"
)

NOT_PERSISTABLE: dict[str, str] = {
    "SimpleLinearRegression": NEEDS_ITS_ENGINE,
    "KNearestNeighboursRegressor": NEEDS_ITS_ENGINE,
    "AdaBoostClassifier": NEEDS_ITS_ENGINE,
    "KNearestNeighboursClassifier": NEEDS_ITS_ENGINE,
    "DecisionTreeRegressor": NEEDS_ITS_ENGINE,
    "DecisionTreeClassifier": NEEDS_ITS_ENGINE,
    "BaggingRegressor": NEEDS_ITS_ENGINE,
    "BaggingClassifier": NEEDS_ITS_ENGINE,
    "RandomForestRegressor": NEEDS_ITS_ENGINE,
    "RandomForestClassifier": NEEDS_ITS_ENGINE,
    "GradientBoostingRegressor": NEEDS_ITS_ENGINE,
    "KernelRidgeRegression": NEEDS_ITS_ENGINE,
    "SupportVectorClassifier": NEEDS_ITS_ENGINE,
    "KMeans": NEEDS_ITS_ENGINE,
    "PrincipalComponentAnalysis": NEEDS_ITS_ENGINE,
    "KernelPrincipalComponentAnalysis": NEEDS_ITS_ENGINE,
    "PolynomialFeatures": NEEDS_ITS_ENGINE,
    "OneVsRestClassifier": (
        "holds one fitted member per class, and a member is itself a wrapper "
        "around an engine. " + NEEDS_ITS_ENGINE
    ),
}
"""Every wrapper this backend has and cannot bring back, with the reason."""


register_backend(SCIKIT_BACKEND, PERSISTABLE_TYPES, NOT_PERSISTABLE)
