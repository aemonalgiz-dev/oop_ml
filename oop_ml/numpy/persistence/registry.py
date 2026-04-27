"""Which from-scratch models a document may name.

A dict rather than an import-by-path precisely so the set is closed: a
document naming anything else is refused with the list of what exists, and no
string in a file can ever cause an import. That is the pickle problem this
format was built to avoid, in miniature.

Registered here rather than in ``core`` so that the shared machinery never
imports a backend, and imported for its side effect by this package's
``__init__``.
"""

from __future__ import annotations

from pydantic import BaseModel

from oop_ml.core.kernel.functions import (
    LinearKernel,
    PolynomialKernel,
    RadialBasisKernel,
    SigmoidKernel,
)
from oop_ml.core.persistence.registry import NUMPY_BACKEND, register_backend
from oop_ml.core.pipeline.pipelines import (
    ClassificationPipeline,
    RegressionPipeline,
)
from oop_ml.core.schedule import (
    ConstantSchedule,
    ExponentialDecaySchedule,
    LinearDecaySchedule,
)
from oop_ml.numpy.classification.binary.logistic_regression import LogisticRegression
from oop_ml.numpy.classification.binary.newton_logistic_regression import (
    NewtonLogisticRegression,
)
from oop_ml.numpy.classification.ensembles.bagging_classifier import BaggingClassifier
from oop_ml.numpy.classification.ensembles.random_forest_classifier import (
    RandomForestClassifier,
)
from oop_ml.numpy.classification.kernels.support_vector_classifier import (
    SupportVectorClassifier,
)
from oop_ml.numpy.classification.multiclass.multinomial_logistic_regression import (
    MultinomialLogisticRegression,
)
from oop_ml.numpy.classification.multiclass.one_vs_rest import OneVsRestClassifier
from oop_ml.numpy.classification.neighbours.k_nearest_classifier import (
    KNearestNeighboursClassifier,
)
from oop_ml.numpy.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from oop_ml.numpy.clustering.k_means import KMeans
from oop_ml.numpy.decomposition.kernel_principal_component_analysis import (
    KernelPrincipalComponentAnalysis,
)
from oop_ml.numpy.decomposition.principal_component_analysis import (
    PrincipalComponentAnalysis,
)
from oop_ml.numpy.generative.restricted_boltzmann_machine import (
    RestrictedBoltzmannMachine,
)
from oop_ml.numpy.preprocessing.polynomial.features import PolynomialFeatures
from oop_ml.numpy.preprocessing.rescaling.affine import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    RootMeanSquareScaler,
)
from oop_ml.numpy.preprocessing.standardization.standardizer import Standardizer
from oop_ml.numpy.regression.ensembles.bagging_regressor import BaggingRegressor
from oop_ml.numpy.regression.ensembles.gradient_boosting_regressor import (
    GradientBoostingRegressor,
)
from oop_ml.numpy.regression.ensembles.random_forest_regressor import (
    RandomForestRegressor,
)
from oop_ml.numpy.regression.kernels.kernel_ridge_regression import (
    KernelRidgeRegression,
)
from oop_ml.numpy.regression.least_squares.gradient_descent_regression import (
    GradientDescentRegression,
)
from oop_ml.numpy.regression.least_squares.multiple_feature_regression import (
    MultipleLinearRegression,
)
from oop_ml.numpy.regression.least_squares.simple_linear_regression import (
    SimpleLinearRegression,
)
from oop_ml.numpy.regression.neighbours.k_nearest_regressor import (
    KNearestNeighboursRegressor,
)
from oop_ml.numpy.regression.penalised.lasso_regression import LassoRegression
from oop_ml.numpy.regression.penalised.ridge_regression import RidgeRegression
from oop_ml.numpy.regression.trees.decision_tree_regressor import DecisionTreeRegressor

PERSISTABLE_TYPES: dict[str, type[BaseModel]] = {
    persistable.__name__: persistable
    for persistable in (
        # regression
        SimpleLinearRegression,
        MultipleLinearRegression,
        GradientDescentRegression,
        RidgeRegression,
        LassoRegression,
        KNearestNeighboursRegressor,
        DecisionTreeRegressor,
        BaggingRegressor,
        RandomForestRegressor,
        GradientBoostingRegressor,
        KernelRidgeRegression,
        # classification
        LogisticRegression,
        NewtonLogisticRegression,
        MultinomialLogisticRegression,
        OneVsRestClassifier,
        KNearestNeighboursClassifier,
        DecisionTreeClassifier,
        BaggingClassifier,
        RandomForestClassifier,
        SupportVectorClassifier,
        # unsupervised
        KMeans,
        PrincipalComponentAnalysis,
        KernelPrincipalComponentAnalysis,
        # preprocessing and composition
        RestrictedBoltzmannMachine,
        Standardizer,
        MinMaxScaler,
        MaxAbsScaler,
        RobustScaler,
        RootMeanSquareScaler,
        PolynomialFeatures,
        RegressionPipeline,
        ClassificationPipeline,
        # kernels appear inside hyperparameters, never as top-level models
        LinearKernel,
        ConstantSchedule,
        LinearDecaySchedule,
        ExponentialDecaySchedule,
        PolynomialKernel,
        RadialBasisKernel,
        SigmoidKernel,
    )
}
"""Every class a document may name, and the only ones it may.

A dict rather than an import-by-path precisely so the set is closed: a
document naming anything else is refused with the list of what exists, and no
string in a file can ever cause an import.
"""


register_backend(NUMPY_BACKEND, PERSISTABLE_TYPES)
