"""The from-scratch backend, in numpy, which is the one the teaching website runs.

Every model here is implemented by hand, and every class name it exports is
promised to exist under the same name and signature in the other backends, so

    from oop_ml.numpy import RidgeRegression

and

    from oop_ml.scikit import RidgeRegression

are meant to be interchangeable at the call site. The shared contract, the data
vocabulary, the base classes, the exceptions and the evaluations, lives in
``oop_ml.core`` and is not repeated here.
"""

from oop_ml.core.base.linear_classifier import LinearClassifier
from oop_ml.core.decomposition.kernel_components import (
    KernelComponent,
    KernelComponents,
)
from oop_ml.core.generative.boltzmann import (
    BoltzmannParameters,
    ContrastiveDivergenceUpdate,
)
from oop_ml.core.kernel.support_vectors import (
    SupportVector,
    SupportVectors,
)
from oop_ml.core.preprocessing.affine_scalings import (
    AffineScaling,
    AffineScalings,
)
from oop_ml.core.preprocessing.feature_scalings import (
    FeatureScaling,
    FeatureScalings,
)
from oop_ml.core.preprocessing.polynomial_terms import PolynomialTerm, PolynomialTerms
from oop_ml.core.sequences.transitions import (
    StateDistribution,
    TransitionCounts,
    TransitionMatrix,
)
from oop_ml.numpy.associative_memory.hopfield_network import (
    BipolarPattern,
    HebbianWeights,
    HopfieldNetwork,
    RecallPass,
    RecallStop,
    RecallWalk,
    UpdateRule,
)
from oop_ml.numpy.classification.binary.logistic_regression import LogisticRegression
from oop_ml.numpy.classification.binary.newton_logistic_regression import (
    NewtonLogisticRegression,
)
from oop_ml.numpy.classification.ensembles.adaboost_classifier import (
    AdaBoostClassifier,
)
from oop_ml.numpy.classification.ensembles.bagging_classifier import BaggingClassifier
from oop_ml.numpy.classification.ensembles.random_forest_classifier import (
    RandomForestClassifier,
)
from oop_ml.numpy.classification.generative.gaussian_naive_bayes import (
    GaussianNaiveBayes,
)
from oop_ml.numpy.classification.generative.linear_discriminant_analysis import (
    LinearDiscriminantAnalysis,
)
from oop_ml.numpy.classification.generative.quadratic_discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
)
from oop_ml.numpy.classification.kernels.support_vector_classifier import (
    SupportVectorClassifier,
)
from oop_ml.numpy.classification.multiclass.multinomial_logistic_regression import (
    MultinomialLogisticRegression,
)
from oop_ml.numpy.classification.multiclass.one_vs_rest import OneVsRestClassifier
from oop_ml.numpy.classification.multiclass.one_vs_rest_fits import (
    ClassFit,
    OneVsRestFits,
)
from oop_ml.numpy.classification.neighbours.k_nearest_classifier import (
    KNearestNeighboursClassifier,
)
from oop_ml.numpy.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from oop_ml.numpy.clustering.agglomerative import AgglomerativeClustering
from oop_ml.numpy.clustering.dbscan import DBSCAN
from oop_ml.numpy.clustering.gaussian_mixture import GaussianMixture
from oop_ml.numpy.clustering.k_means import KMeans
from oop_ml.numpy.clustering.self_organising_map import (
    GridPosition,
    MapUnit,
    SelfOrganisingMap,
    UnitGrid,
)
from oop_ml.numpy.decomposition.hebbian_principal_components import (
    HebbianDirection,
    HebbianDirections,
    HebbianPrincipalComponents,
)
from oop_ml.numpy.decomposition.kernel_principal_component_analysis import (
    KernelPrincipalComponentAnalysis,
)
from oop_ml.numpy.decomposition.principal_component_analysis import (
    PrincipalComponentAnalysis,
)
from oop_ml.numpy.generative.restricted_boltzmann_machine import (
    GibbsState,
    RestrictedBoltzmannMachine,
)
from oop_ml.numpy.persistence import (
    ModelDocument,
    build_model,
    load_model,
    model_document,
    save_model,
)
from oop_ml.numpy.preprocessing.polynomial.features import PolynomialFeatures
from oop_ml.numpy.preprocessing.rescaling.affine import (
    FeatureScaler,
    MaxAbsScaler,
    MeanCentrer,
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
from oop_ml.numpy.regression.linear_feature_regressor import LinearFeatureRegressor
from oop_ml.numpy.regression.neighbours.k_nearest_regressor import (
    KNearestNeighboursRegressor,
)
from oop_ml.numpy.regression.penalised.elastic_net_regression import (
    ElasticNetRegression,
)
from oop_ml.numpy.regression.penalised.lasso_regression import LassoRegression
from oop_ml.numpy.regression.penalised.ridge_regression import RidgeRegression
from oop_ml.numpy.regression.robust.huber_regression import HuberRegression
from oop_ml.numpy.regression.trees.decision_tree_regressor import DecisionTreeRegressor
from oop_ml.numpy.sequences.markov_chain import MarkovChain

__all__ = [
    "BipolarPattern",
    "HebbianWeights",
    "HopfieldNetwork",
    "RecallPass",
    "RecallStop",
    "RecallWalk",
    "UpdateRule",
    "GaussianNaiveBayes",
    "LinearDiscriminantAnalysis",
    "QuadraticDiscriminantAnalysis",
    "LogisticRegression",
    "NewtonLogisticRegression",
    "AdaBoostClassifier",
    "BaggingClassifier",
    "RandomForestClassifier",
    "SupportVector",
    "SupportVectorClassifier",
    "SupportVectors",
    "LinearClassifier",
    "MultinomialLogisticRegression",
    "OneVsRestClassifier",
    "ClassFit",
    "OneVsRestFits",
    "KNearestNeighboursClassifier",
    "DecisionTreeClassifier",
    "KMeans",
    "DBSCAN",
    "GaussianMixture",
    "AgglomerativeClustering",
    "GridPosition",
    "MapUnit",
    "SelfOrganisingMap",
    "UnitGrid",
    "HebbianDirection",
    "HebbianDirections",
    "HebbianPrincipalComponents",
    "KernelComponent",
    "KernelComponents",
    "KernelPrincipalComponentAnalysis",
    "PrincipalComponentAnalysis",
    "BoltzmannParameters",
    "ContrastiveDivergenceUpdate",
    "GibbsState",
    "RestrictedBoltzmannMachine",
    "ModelDocument",
    "build_model",
    "load_model",
    "model_document",
    "save_model",
    "PolynomialFeatures",
    "PolynomialTerm",
    "PolynomialTerms",
    "AffineScaling",
    "AffineScalings",
    "FeatureScaler",
    "MaxAbsScaler",
    "MinMaxScaler",
    "RobustScaler",
    "RootMeanSquareScaler",
    "MeanCentrer",
    "FeatureScaling",
    "FeatureScalings",
    "Standardizer",
    "MarkovChain",
    "StateDistribution",
    "TransitionCounts",
    "TransitionMatrix",
    "BaggingRegressor",
    "GradientBoostingRegressor",
    "RandomForestRegressor",
    "KernelRidgeRegression",
    "GradientDescentRegression",
    "MultipleLinearRegression",
    "SimpleLinearRegression",
    "LinearFeatureRegressor",
    "KNearestNeighboursRegressor",
    "LassoRegression",
    "ElasticNetRegression",
    "RidgeRegression",
    "HuberRegression",
    "DecisionTreeRegressor",
]
