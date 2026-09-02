"""An object-oriented machine learning library for Python.

Everything most applications need is re-exported here, so ordinary use is one
import::

    from oop_ml import Feature, RidgeRegression

    model = RidgeRegression(penalty=1.0)
    model.fit(
        [Feature("area", areas), Feature("baths", baths)],
        Feature("price", prices),
    )
    model.coefficients["area"]

The full paths still work and are what the library itself uses internally, so
reach for ``from oop_ml.regression.penalised.ridge_regression import RidgeRegression``
whenever you want to be explicit about where something lives, or when you need a
name that this surface does not re-export.
"""

from oop_ml.classification.binary.logistic_regression import LogisticRegression
from oop_ml.classification.binary.newton_logistic_regression import (
    NewtonLogisticRegression,
)
from oop_ml.classification.ensembles.bagging_classifier import BaggingClassifier
from oop_ml.classification.ensembles.random_forest_classifier import (
    RandomForestClassifier,
)
from oop_ml.classification.kernels.support_vector_classifier import (
    SupportVector,
    SupportVectorClassifier,
    SupportVectors,
)
from oop_ml.classification.linear_classifier import LinearClassifier
from oop_ml.classification.multiclass.multinomial_logistic_regression import (
    MultinomialLogisticRegression,
)
from oop_ml.classification.multiclass.one_vs_rest import OneVsRestClassifier
from oop_ml.classification.multiclass.one_vs_rest_fits import (
    ClassFit,
    OneVsRestFits,
)
from oop_ml.classification.neighbours.k_nearest_classifier import (
    KNearestNeighboursClassifier,
)
from oop_ml.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from oop_ml.clustering.k_means import KMeans
from oop_ml.core.base.ensemble import AveragingEnsemble, BoostingEnsemble
from oop_ml.core.base.estimator import (
    Classifier,
    Clusterer,
    Estimator,
    Fittable,
    MultiClassClassifier,
    Regressor,
    Transformer,
)
from oop_ml.core.base.linear_model import LinearModel
from oop_ml.core.base.neighbour_model import NeighbourModel
from oop_ml.core.base.tree_model import TreeModel
from oop_ml.core.clustering.centroids import Centroid, Centroids
from oop_ml.core.clustering.clustering import Clustering, InitialisationAttempt
from oop_ml.core.data.coefficients import Coefficient, Coefficients
from oop_ml.core.data.column import Column
from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.design_matrix import DesignMatrix
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import (
    ClassScores,
    Probabilities,
    ProbabilityMatrix,
)
from oop_ml.core.data.row_block import RowBlock
from oop_ml.core.decomposition.components import (
    PrincipalComponent,
    PrincipalComponents,
)
from oop_ml.core.distance.calculations import (
    BroadcastDistance,
    CanberraDistance,
    CosineDistance,
    Distance,
    EuclideanDistance,
    HammingDistance,
    MinkowskiDistance,
)
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.ensemble.bootstrap import BootstrapSample
from oop_ml.core.ensemble.fits import (
    BoostingRound,
    BoostingRounds,
    EnsembleFits,
    MemberFit,
)
from oop_ml.core.ensemble.out_of_bag import OutOfBagEstimate
from oop_ml.core.evaluation.classification import (
    ClassificationEvaluation,
    ConfusionMatrix,
)
from oop_ml.core.evaluation.multiclass import (
    MultiClassConfusionMatrix,
    MultiClassEvaluation,
)
from oop_ml.core.evaluation.regression import RegressionEvaluation
from oop_ml.core.exceptions import (
    AllSameValuesError,
    CollinearFeaturesError,
    DivergenceError,
    EmptyValuesError,
    InvalidDocumentError,
    InvalidValuesError,
    MLLibError,
    NonBinaryLabelsError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    NotFittedError,
    ShapeMismatchError,
    SingleClassError,
    SingularHessianError,
    TooFewValuesError,
    UndefinedMetricError,
)
from oop_ml.core.importance.importances import (
    FeatureImportance,
    FeatureImportances,
)
from oop_ml.core.importance.permutation import PermutationImportance
from oop_ml.core.kernel.functions import (
    Kernel,
    LinearKernel,
    PolynomialKernel,
    RadialBasisKernel,
    SigmoidKernel,
)
from oop_ml.core.kernel.matrix import KernelMatrix
from oop_ml.core.neighbours.search import NeighbourQuery, NeighbourSearch
from oop_ml.core.network.activation import (
    Activation,
    HyperbolicTangent,
    Identity,
    RectifiedLinear,
    Sigmoid,
)
from oop_ml.core.network.convolution import Conv2d
from oop_ml.core.network.flatten import Flatten
from oop_ml.core.network.gradient import BackwardPass, LayerCorrection, LayerGradient
from oop_ml.core.network.layer import DenseLayer, Layer, LayerResponse
from oop_ml.core.network.loss import (
    AbsoluteError,
    BinaryCrossEntropy,
    HuberError,
    Loss,
    LossMeasurement,
    SoftmaxCrossEntropy,
    SquaredError,
)
from oop_ml.core.network.neuron import Neuron, NeuronResponse
from oop_ml.core.network.pooling import AveragePool2d, MaxPool2d, Pool2d
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.network.stack import LayerStack, StackResponse
from oop_ml.core.observation import Observation, Stage
from oop_ml.core.solving.normal_equations import (
    LeastSquaresLine,
    NormalEquations,
)
from oop_ml.core.solving.path import SolverPath, SolverStep, SolverStop
from oop_ml.core.tree.criterion import ClassificationCriterion, RegressionCriterion
from oop_ml.core.tree.impurity import (
    EntropyImpurity,
    GiniImpurity,
    Impurity,
    VarianceImpurity,
)
from oop_ml.core.tree.node import (
    ClassificationLeaf,
    DecisionNode,
    LeafNode,
    TreeNode,
)
from oop_ml.core.tree.search import (
    SplitCandidate,
    SplitRejection,
    SplitSearch,
)
from oop_ml.core.tree.split import Split
from oop_ml.core.types import (
    FloatArray,
    IndexArray,
    Numeric,
    NumericInput,
    NumericValues,
)
from oop_ml.decomposition.kernel_principal_component_analysis import (
    KernelComponent,
    KernelComponents,
    KernelPrincipalComponentAnalysis,
)
from oop_ml.decomposition.principal_component_analysis import (
    PrincipalComponentAnalysis,
)
from oop_ml.model_selection.cross_validation import (
    ClassificationCrossValidationResult,
    CrossValidation,
    RegressionCrossValidationResult,
)
from oop_ml.model_selection.dataset import DataSplit
from oop_ml.model_selection.search import (
    Candidate,
    GridSearch,
    ParameterRange,
    ScoredCandidate,
    SearchResult,
    SearchSpace,
)
from oop_ml.model_selection.splitting import (
    KFold,
    RowShuffler,
    Splits,
    TrainTestSplitter,
)
from oop_ml.persistence.document import ModelDocument
from oop_ml.persistence.store import (
    build_model,
    load_model,
    model_document,
    save_model,
)
from oop_ml.pipeline.pipelines import (
    ClassificationPipeline,
    Pipeline,
    RegressionPipeline,
)
from oop_ml.pipeline.steps import PipelineStep, PipelineSteps
from oop_ml.preprocessing.polynomial.features import PolynomialFeatures
from oop_ml.preprocessing.polynomial.terms import PolynomialTerm, PolynomialTerms
from oop_ml.preprocessing.standardization.scaling import FeatureScaling, FeatureScalings
from oop_ml.preprocessing.standardization.standardizer import Standardizer
from oop_ml.regression.ensembles.bagging_regressor import BaggingRegressor
from oop_ml.regression.ensembles.gradient_boosting_regressor import (
    GradientBoostingRegressor,
)
from oop_ml.regression.ensembles.random_forest_regressor import (
    RandomForestRegressor,
)
from oop_ml.regression.kernels.kernel_ridge_regression import (
    KernelRidgeRegression,
)
from oop_ml.regression.least_squares.gradient_descent_regression import (
    GradientDescentRegression,
)
from oop_ml.regression.least_squares.multiple_feature_regression import (
    MultipleLinearRegression,
)
from oop_ml.regression.least_squares.simple_linear_regression import (
    SimpleLinearRegression,
)
from oop_ml.regression.linear_feature_regressor import LinearFeatureRegressor
from oop_ml.regression.neighbours.k_nearest_regressor import (
    KNearestNeighboursRegressor,
)
from oop_ml.regression.penalised.lasso_regression import LassoRegression
from oop_ml.regression.penalised.ridge_regression import RidgeRegression
from oop_ml.regression.trees.decision_tree_regressor import DecisionTreeRegressor

__all__ = [
    # Type aliases, for annotating your own code
    "Numeric",
    "NumericValues",
    "NumericInput",
    "FloatArray",
    "IndexArray",
    # Data
    "Column",
    "Feature",
    "FeatureSet",
    "Dataset",
    "DesignMatrix",
    "ClassScores",
    "Predictions",
    "Probabilities",
    "ProbabilityMatrix",
    "RowBlock",
    "DataSplit",
    # Results
    "Coefficient",
    "Coefficients",
    "RegressionEvaluation",
    "ClassificationEvaluation",
    "ConfusionMatrix",
    "MultiClassEvaluation",
    "MultiClassConfusionMatrix",
    # Base classes, for writing your own
    "Fittable",
    "Estimator",
    "Regressor",
    "Classifier",
    "MultiClassClassifier",
    "Transformer",
    "LinearModel",
    "LinearFeatureRegressor",
    "NeighbourModel",
    "TreeModel",
    "Impurity",
    "GiniImpurity",
    "EntropyImpurity",
    "VarianceImpurity",
    "ClassificationCriterion",
    "RegressionCriterion",
    "Split",
    "Observation",
    "Stage",
    "NormalEquations",
    "LeastSquaresLine",
    "OneVsRestFits",
    "ClassFit",
    "SplitSearch",
    "SplitCandidate",
    "SplitRejection",
    "SolverPath",
    "SolverStep",
    "SolverStop",
    "NeighbourSearch",
    "NeighbourQuery",
    "TreeNode",
    "DecisionNode",
    "LeafNode",
    "ClassificationLeaf",
    "DistanceMetric",
    "Distance",
    "BroadcastDistance",
    "MinkowskiDistance",
    "EuclideanDistance",
    "CosineDistance",
    "HammingDistance",
    "CanberraDistance",
    "LinearClassifier",
    # Regression
    "SimpleLinearRegression",
    "MultipleLinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "GradientDescentRegression",
    "KNearestNeighboursRegressor",
    "DecisionTreeRegressor",
    "BaggingRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    # Classification
    "LogisticRegression",
    "NewtonLogisticRegression",
    "MultinomialLogisticRegression",
    "OneVsRestClassifier",
    "KNearestNeighboursClassifier",
    "DecisionTreeClassifier",
    "BaggingClassifier",
    "RandomForestClassifier",
    "AveragingEnsemble",
    "BoostingEnsemble",
    "BootstrapSample",
    "OutOfBagEstimate",
    "FeatureImportance",
    "FeatureImportances",
    "PermutationImportance",
    "MemberFit",
    "EnsembleFits",
    "BoostingRound",
    "BoostingRounds",
    # Preprocessing
    "Standardizer",
    "PolynomialFeatures",
    "PolynomialTerm",
    "PolynomialTerms",
    "FeatureScaling",
    "FeatureScalings",
    # Model selection
    "RowShuffler",
    "TrainTestSplitter",
    "KFold",
    "Splits",
    "CrossValidation",
    "ClassificationCrossValidationResult",
    "Candidate",
    "CollinearFeaturesError",
    "DivergenceError",
    "Centroid",
    "ClassificationPipeline",
    "Clusterer",
    "Centroids",
    "Clustering",
    "InitialisationAttempt",
    "GridSearch",
    "KMeans",
    "Kernel",
    "KernelComponent",
    "KernelComponents",
    "KernelMatrix",
    "KernelPrincipalComponentAnalysis",
    "KernelRidgeRegression",
    "LinearKernel",
    "PolynomialKernel",
    "ParameterRange",
    "Pipeline",
    "PipelineStep",
    "PipelineSteps",
    "PrincipalComponent",
    "PrincipalComponentAnalysis",
    "PrincipalComponents",
    "RadialBasisKernel",
    "RegressionCrossValidationResult",
    "RegressionPipeline",
    "ScoredCandidate",
    "SearchResult",
    "SearchSpace",
    "SigmoidKernel",
    "SupportVector",
    "SupportVectorClassifier",
    "SupportVectors",
    "build_model",
    "load_model",
    "model_document",
    "save_model",
    # Networks, the vocabulary a neural model is built from
    "Activation",
    "Identity",
    "RectifiedLinear",
    "Sigmoid",
    "HyperbolicTangent",
    "Neuron",
    "NeuronResponse",
    "DenseLayer",
    "LayerResponse",
    "LayerShape",
    "Conv2d",
    "Flatten",
    "Layer",
    "LayerCorrection",
    "Pool2d",
    "MaxPool2d",
    "AveragePool2d",
    "PassPurpose",
    "AbsoluteError",
    "BackwardPass",
    "BinaryCrossEntropy",
    "HuberError",
    "LayerGradient",
    "Loss",
    "LossMeasurement",
    "SoftmaxCrossEntropy",
    "SquaredError",
    "LayerStack",
    "StackResponse",
    # Errors, all of which derive from MLLibError
    "ShapeMismatchError",
    "MLLibError",
    "ModelDocument",
    "NotFittedError",
    "EmptyValuesError",
    "TooFewValuesError",
    "NonEqualArrayLengthError",
    "InvalidDocumentError",
    "InvalidValuesError",
    "AllSameValuesError",
    "UndefinedMetricError",
    "NonUniqueFeaturesError",
    "NonBinaryLabelsError",
    "SingleClassError",
    "SingularHessianError",
]
