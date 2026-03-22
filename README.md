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
| **Preprocessing** | standardization, polynomial features |
| **Model selection** | train/test split, k-fold, cross-validation |
| **Evaluation** | regression, binary and multi-class, each on its own object |

## Four Conventions I Held To

**Construction configures and `fit` learns.** Hyperparameters are Pydantic
fields validated at construction, so `RidgeRegression(penalty=-1)` fails
straight away rather than somewhere deep inside a solve, and data is only ever
passed to `fit`.

**Columns carry their own names.** A model takes `Feature` objects, matches
them by name when you call `predict`, and hands back coefficients indexed by
name, so the order the columns arrive in does not matter and a mismatched set
raises instead of quietly answering the wrong question.

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
pytest                  # 1326 tests
ruff check .
ruff format .
pyright oop_ml test
```

## Documentation

- [Design notes](docs/design.md), for why the API looks like this
- [The models](docs/models.md), for each family and what measuring it showed
- [On scikit-learn](docs/comparison.md), for what differs and when to use theirs

## Status

Every supervised family is implemented and green: **1326 passing tests**, `ruff`
and `pyright` clean, no stubs.

Not built yet, roughly in the order it will matter if you are putting this into
an application:

1. **Persistence.** A fitted model cannot be serialised yet, so a process has
   to fit at start-up rather than load a trained artifact.
2. **`Pipeline`.** A transformer fitted outside a cross-validation loop can
   leak across the split, and the serving path has to re-apply it by hand.
3. **Cross-validated classification.** `CrossValidation` scores with R², so it
   only speaks to regressors, and fixing that needs stratified folds as well.
4. **Tree pruning.** Growth stops on rules chosen up front, rather than growing
   first and cutting back on what a subtree turned out to be worth.
5. **Hyperparameter search.** `CrossValidation` scores one candidate; nothing
   searches a space of them.
6. **Unsupervised learning and kernels.** Both absent entirely. Every base
   class here takes a target, so k-means and PCA need new frames rather than
   new models.
