# oop_ml

An object-oriented machine learning library for Python, written from scratch.
Models here are objects that take named inputs, expose the parameters they
learned, and hand back an evaluation you can read metrics off, instead of
taking arrays in and giving arrays back.

```python
from oop_ml import Feature, RidgeRegression

model = RidgeRegression(penalty=1.0).fit(
    [
        Feature("floor_area_sqm", [72, 140, 96, 210, 55, 118, 165, 88]),
        Feature("bathrooms", [1, 2, 1, 3, 1, 2, 2, 1]),
    ],
    Feature("price_thousands", [310, 505, 372, 690, 240, 448, 560, 350]),
)

model.intercept  # 99.55
model.coefficients["floor_area_sqm"]  # 2.81
model.score(features, target)  # R²
```

All of it is built on numpy alone, with no scikit-learn and no pandas in the
core, since I wanted an API of a different shape and an implementation I could
read end to end.

## Installation

```bash
pip install -e ".[dev]"
```

Python 3.11 or later.

## What Is Included

| | |
|---|---|
| **Regression** | simple, multiple, gradient descent, ridge, lasso |
| **Classification** | logistic (gradient ascent and IRLS), softmax, one-vs-rest |
| **Neighbours** | k-nearest for both tasks, six distance metrics |
| **Trees** | decision tree for both tasks, Gini / entropy / variance |
| **Ensembles** | bagging, random forest, gradient boosting, out-of-bag scoring |
| **Interpretation** | feature importance, by impurity and by permutation |
| **Preprocessing** | standardization, polynomial features |
| **Model selection** | train/test split, k-fold, stratified folds, cross-validation, grid search |
| **Pipelines** | preprocessing and a model as one estimator, safe inside a fold |
| **Persistence** | fitted models as readable JSON, revalidated on load |
| **Performance** | benchmarked against scikit-learn; ties or wins on the linear-algebra-bound families |
| **Decomposition** | principal component analysis, kernel PCA, explained variance |
| **Clustering** | k-means with k-means++ seeding, inertia, named centroids |
| **Kernels** | linear, polynomial, radial basis, sigmoid; kernel ridge, SVM |
| **Evaluation** | regression, binary and multi-class, each on its own object |

## Four Conventions I Held To

**Construction configures and `fit` learns.** Hyperparameters are Pydantic
fields validated at construction, so `RidgeRegression(penalty=-1)` fails
straight away rather than somewhere deep inside a solve, and data is only ever
passed to `fit`.

**Columns carry their own names.** A model takes `Feature` objects, matches
them by name when you call `predict`, and hands back coefficients indexed by
name, so the order the columns arrive in does not matter and a mismatched set
raises instead of quietly answering the wrong question. That holds all the way
down: a tree's split routes rows by looking its feature up by name, not by the
column position the search happened to record.

```python
model.coefficients["bathrooms"]  # not coefficients[1]
```

**Metrics live on an object.** `evaluate` predicts once and returns an
evaluation, and you read as many metrics off it as you want.

```python
evaluation = model.evaluate(features, target)
evaluation.r_squared, evaluation.mean_squared_error, evaluation.residuals
```

**Every failure is a typed error.** `NotFittedError`,
`NonEqualArrayLengthError`, `SingleClassError` and the rest all derive from
`MLLibError`, so you can catch the category you actually care about rather than
matching on a message.

## Every Signature Says What It Takes

`FloatArray` says "floats". It does not say whether an array is one row per
observation or one per feature, whether column zero is an intercept, or whether
anything is bounded. Each method here takes a type that does.

```python
def _solve(self, design_matrix: DesignMatrix, target_column: Column) -> Weights:
def _leaf(self, target_values: Column) -> LeafNode:
def _combine(self, member_predictions: MemberPredictions) -> FloatArray:
def predict_probabilities(self, input_values) -> ClassScores:
```

Three of those types earn their place by making a past bug unwriteable.

`Column` cannot be empty, so `Impurity.of` no longer has to define what an
empty node means -- and it used to need thirty lines to do it, because Gini
returns 1.0 on nothing while variance returns `nan`, and one `nan` makes every
comparison false so a split search silently never picks that candidate.

`DesignMatrix` knows whether its first column is the intercept, so
`penalty_diagonal` is the only thing deciding what ridge exempts. That rule
used to be written twice and one copy was wrong.

`ClassScores` and `ProbabilityMatrix` are two types because a one-vs-rest
wrapper's rows genuinely do not sum to one -- its K models were never asked to
agree. It returns the weaker type and says so.

Wrapping the outputs costs nothing to use, because they implement `__array__`:

```python
np.allclose(model.predict(rows), expected)  # works, it is array-like
model.predict(rows).n_rows  # and it is also an object
```

## A Second Route Through The Calculation

Every model whose intermediates are worth seeing has two methods: the efficient
one, which returns the answer, and an observed one, which keeps the working.

```python
search = model.split_search(rows, targets)

len(search)  # 22 candidates considered
search.best  # Split(slept < 6.25, gain=0.2133)
```

The same shape gives you `solver_path` on the iterative models and
`neighbour_search` on the neighbour ones. Every pair carries a test asserting
that the two routes agree, since a fast path and a slow path with nothing
between them are really two implementations of the same thing.

## Package Layout

```
oop_ml/
  core/            everything that is not a model
    data/          Column, Feature, FeatureSet, Coefficients, Dataset
    base/          the Estimator hierarchy and one frame per model family
    kernel/        four kernels and the Gram matrix they produce
    clustering/    centroids and what a grouping is
    distance/      six metrics behind one closed enum
    tree/          impurity, splits, nodes
    ensemble/      bootstrap samples and the records a fit leaves behind
    evaluation/    one evaluation class per task
  regression/      least_squares, penalised, neighbours, trees, ensembles
  classification/  binary, multiclass, neighbours, trees, ensembles
  preprocessing/   standardization, polynomial
  model_selection/ splitting, cross-validation
```

The directory names the task rather than the model family, since a user tends
to look for "how do I classify things"; the family is carried by a base class
instead. All of it is re-exported from the top level, so `from oop_ml import
Feature` is all most code needs.

## Examples

Eleven runnable scripts in [examples/](examples/), each written against the
installed package rather than the library's internals.

```bash
python -m examples.model_selection
```

That one walks the whole arc: hold out a test set, cross-validate candidate
penalties against the remainder, choose one, refit, and report a single held-out
number that has not been spent on anything else.

## Development

```bash
pytest                  # 1806 tests
ruff check .
ruff format .
pyright oop_ml test
```

## Documentation

- [Design notes](docs/design.md), for why the API looks like this
- [The models](docs/models.md), for each family and what measuring it showed
- [On scikit-learn](docs/comparison.md), for what differs and when to use theirs

## Status

Every supervised family is implemented and green: **1806 passing tests**, `ruff`
and `pyright` clean, no stubs.

Not built yet, roughly in the order it will matter if you are putting this into
an application:

1. **Tree pruning.** Growth stops on rules chosen up front, rather than growing
   first and cutting back on what a subtree turned out to be worth.
2. **More unsupervised models.** k-means is in and Gaussian mixtures are not;
   expectation-maximisation is what makes k-means make sense retroactively, as
   EM with the guesses hardened.
