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
from oop_ml.classification.linear_classifier import LinearClassifier
from oop_ml.classification.multiclass.multinomial_logistic_regression import (
    MultinomialLogisticRegression,
)
from oop_ml.classification.multiclass.one_vs_rest import OneVsRestClassifier
from oop_ml.classification.neighbours.k_nearest_classifier import (
    KNearestNeighboursClassifier,
)
from oop_ml.core.base.estimator import (
    Classifier,
    Estimator,
    Fittable,
    MultiClassClassifier,
    Regressor,
    Transformer,
)
from oop_ml.core.base.linear_model import LinearModel
from oop_ml.core.base.neighbour_model import NeighbourModel
from oop_ml.core.data.coefficients import Coefficient, Coefficients
from oop_ml.core.data.column import Column
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
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
    EmptyValuesError,
    InvalidValuesError,
    MLLibError,
    NonBinaryLabelsError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    NotFittedError,
    SingleClassError,
    SingularHessianError,
    TooFewValuesError,
    UndefinedMetricError,
)
from oop_ml.core.types import (
    FloatArray,
    IndexArray,
    Numeric,
    NumericInput,
    NumericValues,
)
from oop_ml.model_selection.cross_validation import (
    CrossValidation,
    CrossValidationResult,
)
from oop_ml.model_selection.dataset import Dataset, DataSplit
from oop_ml.model_selection.splitting import (
    KFold,
    RowShuffler,
    Splits,
    TrainTestSplitter,
)
from oop_ml.preprocessing.polynomial.features import PolynomialFeatures
from oop_ml.preprocessing.polynomial.terms import PolynomialTerm, PolynomialTerms
from oop_ml.preprocessing.standardization.scaling import FeatureScaling, FeatureScalings
from oop_ml.preprocessing.standardization.standardizer import Standardizer
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
    # Classification
    "LogisticRegression",
    "NewtonLogisticRegression",
    "MultinomialLogisticRegression",
    "OneVsRestClassifier",
    "KNearestNeighboursClassifier",
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
    "CrossValidationResult",
    # Errors, all of which derive from MLLibError
    "MLLibError",
    "NotFittedError",
    "EmptyValuesError",
    "TooFewValuesError",
    "NonEqualArrayLengthError",
    "InvalidValuesError",
    "AllSameValuesError",
    "UndefinedMetricError",
    "NonUniqueFeaturesError",
    "NonBinaryLabelsError",
    "SingleClassError",
    "SingularHessianError",
]
