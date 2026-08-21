"""An object-oriented machine learning library for Python.

Everything most applications need is re-exported here, so ordinary use is one
import::

    from oop_ml import Feature, RidgeRegression

    model = RidgeRegression(penalty=1.0)
    model.fit(
        [Feature("area", areas), Feature("baths", baths)],
        Feature("price", prices),
    )
    model.coefficients_["area"]

The full paths still work and are what the library itself uses internally, so
reach for ``from oop_ml.regression.ridge_regression import RidgeRegression``
whenever you want to be explicit about where something lives, or when you need a
name that this surface does not re-export.
"""

from oop_ml.core.base import Estimator, Fittable, Regressor, Transformer
from oop_ml.core.coefficients import Coefficient, Coefficients
from oop_ml.core.column import Column
from oop_ml.core.evaluation import RegressionEvaluation
from oop_ml.core.exceptions import (
    AllSameValuesError,
    EmptyValuesError,
    InvalidValuesError,
    MLLibError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    NotFittedError,
    TooFewValuesError,
    UndefinedMetricError,
)
from oop_ml.core.feature import Feature
from oop_ml.core.feature_set import FeatureSet
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
from oop_ml.preprocessing.polynomial import PolynomialTerm, PolynomialTerms
from oop_ml.preprocessing.polynomial_features import PolynomialFeatures
from oop_ml.preprocessing.scaling import FeatureScaling, FeatureScalings
from oop_ml.preprocessing.standardizer import Standardizer
from oop_ml.regression.gradient_descent_regression import GradientDescentRegression
from oop_ml.regression.lasso_regression import LassoRegression
from oop_ml.regression.linear_feature_regressor import LinearFeatureRegressor
from oop_ml.regression.multiple_feature_regression import MultipleLinearRegression
from oop_ml.regression.ridge_regression import RidgeRegression
from oop_ml.regression.simple_linear_regression import SimpleLinearRegression

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
    # Base classes, for writing your own
    "Fittable",
    "Estimator",
    "Regressor",
    "Transformer",
    "LinearFeatureRegressor",
    # Regression
    "SimpleLinearRegression",
    "MultipleLinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "GradientDescentRegression",
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
]
