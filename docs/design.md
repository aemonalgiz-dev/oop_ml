# Design notes

Why the API looks the way it does. The [README](../README.md) is the short
version; this is the reasoning behind it.

## Why I Felt it Matters

The distance between a model that works in a notebook and a model that works in
an application has always seemed larger to me than it needs to be, so let us
look at what that gap actually costs us. Most Python machine learning
libraries take arrays in and give arrays back, which is
perfectly reasonable for exploratory work, although once that model is sitting
behind a request handler, the meaning of some column lives somewhere outside of
the model entirely, usually in a constant that you are now responsible for
keeping in sync.

```python
FEATURE_ORDER = ["floor_area_sqm", "bathrooms"]  # must match training


def price(record: dict[str, float]) -> float:
    row = np.array([[record[name] for name in FEATURE_ORDER]])
    return float(model.predict(row)[0])
```

Consider what happens six months from now, when someone reorders the columns in
the training script. This function still returns a number; it just won't be the
right one. The convention of named columns protects implementations from this.

```python
def price(record: dict[str, float]) -> float:
    features = [Feature(name, [value]) for name, value in record.items()]
    return float(model.predict(features)[0])
```

There is no ordering left to get wrong, and a missing or misspelled feature
raises `InvalidValuesError` instead of quietly producing something plausible.

## Names Instead Of Positions

```python
model.coefficients["floor_area_sqm"]  # 2.81
"garden" in model.coefficients  # False
```

`predict` matches by name as well, so you may hand it features in whatever order
happens to be convenient. You do have to hand it all of them, though. A model
that is missing a column cannot evaluate its own hyperplane, and guessing at
what you meant would be considerably worse than refusing.

## Errors You Can Actually Route

```python
InvalidValuesError: expected features bathrooms, floor_area_sqm; got floor_area_sqm
NotFittedError: RidgeRegression must be fit before this is available
```

There are ten exception types, all of them deriving from `MLLibError`, so a
handler can return a 400 for `InvalidValuesError` and a 503 for
`NotFittedError` without anyone having to match on message strings.

One distinction worth knowing before you write the `except` clause: bad data
raises `MLLibError`, while bad hyperparameters raise Pydantic's
`ValidationError` at construction time, before any data is involved at all.
Those belong in two different places in your application.

## Metrics That Live On An Object

```python
evaluation = model.evaluate(features, target)

evaluation.r2_score  # 0.9959
evaluation.mean_squared_error  # 76.41
evaluation.residual_sum_of_squares  # 611.29
```

I did not want a module of free functions here. `evaluate` predicts once and
aligns the predictions with the truth once, so reading three metrics does not
quietly cost you three passes over your data, and comparing two vectors that do
not line up is not something you can do by accident.

## Keeping Data Together

```python
dataset = Dataset([area, bathrooms], price)
split = TrainTestSplitter(test_fraction=0.25, random_seed=0).split(dataset)

split.training.n_samples  # 6
split.testing.n_samples  # 2
```

`Dataset.select_rows` subsets every feature and the target using the same
indices, so a shuffle cannot decouple your inputs from your outputs. It is a
small guarantee, though it is the kind of bug that is genuinely painful to find
after the fact, because nothing about the code looks wrong and the model
still trains.

## Cross-Validation And Its Uncertainty

```python
result = CrossValidation(folds=KFold(n_folds=4, random_seed=0)).evaluate(
    RidgeRegression(penalty=1.0), dataset
)

result.mean_r2_score  # 0.9054
result.r2_score_spread  # 0.2294
```

The spread comes back alongside the mean deliberately. A mean R² of 0.8
assembled from folds at 0.79, 0.81, and 0.80 is a very different claim than the
same mean assembled out of 0.2, 0.95, and 1.0, and no single number is going to
tell you which of the two you are looking at.

## Watching A Model Work

Every model here has a second route through its own calculation. The efficient
one returns the answer and discards the working; the observed one keeps all of
it.

```python
search = model.split_search(rows, targets)

len(search)  # 22 candidates considered
search.best  # Split(slept < 6.25, gain=0.2133)
```

```
Split(slept < 6.25, gain=0.2133)     6/9
Split(studied < 7.25, gain=0.1800)   12/3
Split(studied < 6.25, gain=0.1600)   10/5
```

The winner is rarely the interesting part; the field it beat is. Rejected
candidates are kept too, with their gain and the reason they were excluded,
because *"excluded despite scoring 0.24"* is a fact about the model that a list
of survivors cannot express.

Same shape everywhere:

| model | efficient | observed |
|---|---|---|
| trees | `_best_split` | `split_search()` |
| gradient descent, logistic, Newton, softmax, lasso | `_solve` | `solver_path()` |
| k-NN | `_neighbour_indices` | `neighbour_search()` |
| multiple, ridge | `_solve` | `normal_equations()` |
| simple linear | `fit` | `least_squares_line()` |
| one-vs-rest | `fit` | `one_vs_rest_fits()` |

What it makes visible is hard to say any other way. The same objective, two
solvers:

```
ascent    4495 passes   movement 4.06e-02 -> 9.98e-09
newton       6 passes   movement 1.98e+00 -> 2.28e-16
```

`normal_equations` exposes `condition_number`, which is the only place
collinearity shows, since the coefficients simply come back large and cancelling.
`NeighbourQuery.first_rejected_distance` answers whether `k` was a real choice
or an arbitrary one. `OneVsRestFits` makes *"the probabilities do not sum to
one"* something you can check rather than something a docstring claims.

**Two routes, one definition.** Each pair carries a test asserting they agree,
because a fast path and a slow path with nothing between them are two
implementations rather than one calculation seen two ways. That test earned its
keep immediately, catching the two tree routes disagreeing on an exact tie.

The observed route is free to be slow and allocates freely; the efficient one is
untouched by any of it.

## The Package Layout

| Package | Contents |
|---------|----------|
| `oop_ml.core.data` | `Column`, `Feature`, `FeatureSet`, `Coefficients`, `Dataset`, the vocabulary every other package speaks |
| `oop_ml.core.evaluation` | `RegressionEvaluation`, `ClassificationEvaluation`, `MultiClassEvaluation` |
| `oop_ml.core.base` | The generic `Estimator[InputT, TargetT]` hierarchy, plus the `LinearModel`, `IterativeSolver`, `NeighbourModel`, `TreeModel`, `AveragingEnsemble` and `BoostingEnsemble` frames |
| `oop_ml.core.distance` | `DistanceMetric` and the six `Distance` calculations behind it |
| `oop_ml.core.tree` | Impurity measures, criteria, `Split`, and the nodes a fitted tree is made of |
| `oop_ml.core.ensemble` | `BootstrapSample` and the records a fit leaves behind |
| `oop_ml.core.observation` | The `Observation` protocol, which is the second route through a calculation |
| `oop_ml.core.solving` | `SolverPath`, `NormalEquations`, `LeastSquaresLine` |
| `oop_ml.core.neighbours` | `NeighbourSearch`, holding every distance behind a prediction |
| `oop_ml.regression.least_squares` | Simple, multiple, gradient descent; squared error with nothing added to it |
| `oop_ml.regression.penalised` | Ridge and lasso, where the *shape* of the penalty is the whole difference |
| `oop_ml.regression.neighbours` | `KNearestNeighboursRegressor`, with no assumed shape and no coefficients |
| `oop_ml.regression.trees` | `DecisionTreeRegressor`, a piecewise-constant surface you can read |
| `oop_ml.regression.ensembles` | Bagging, random forest, gradient boosting |
| `oop_ml.classification.binary` | `LogisticRegression` and `NewtonLogisticRegression`, one objective and two solvers |
| `oop_ml.classification.multiclass` | `MultinomialLogisticRegression` and `OneVsRestClassifier` |
| `oop_ml.classification.neighbours` | `KNearestNeighboursClassifier`, where any number of classes is the same code |
| `oop_ml.classification.trees` | `DecisionTreeClassifier`, whose boundary is a union of boxes |
| `oop_ml.classification.ensembles` | Bagging and random forest, averaging probabilities rather than votes |
| `oop_ml.preprocessing.standardization` | `Standardizer` and the `FeatureScalings` it learns |
| `oop_ml.preprocessing.polynomial` | `PolynomialFeatures` and the `PolynomialTerms` it builds |
| `oop_ml.model_selection` | `DataSplit`, train/test and k-fold splitters, `CrossValidation` |

`oop_ml.core` is everything that is not a model, split by what a thing is
rather than what it is for, with the type aliases, exception hierarchy and
coercion guards sitting beside those three as the plumbing they all need. The
packages listed above it are the tasks, because a user looks for "how do I
classify things" rather than "what is linear in its coefficients".

All of it is re-exported from the top level, so `from oop_ml import Feature` is
all that most code needs, though the full path is there whenever you want to be
explicit about where something lives.

Adding a linear model means writing exactly one method:

```python
class RidgeRegression(LinearFeatureRegressor):
    penalty: float = Field(default=1.0, ge=0.0)

    def _solve(self, design_matrix, target_column): ...
```

If it arrives at the answer rather than jumping to it, the one method is the
step instead, and the walk around it is inherited too:

```python
class NewtonLogisticRegression(IterativeSolver, LinearClassifier):
    max_iterations: int = Field(default=100, gt=0)

    def _step(self, design_matrix, target_column, weights): ...
```

Start at zero, cap the passes, apply the step, count it, stop when it falls
under the tolerance, record which of the two exits happened. That is the same
in every iterative solver, and writing it out per model is how two of them
ended up reporting `converged = True` beside zero passes run.

Validation, the design matrix, the intercept split, coefficient pairing by name,
and `predict` are all inherited. Note that the directory names the task and not
the model family; "linear" means linear in the coefficients, which is why
`PolynomialFeatures` can fit curves without the estimator changing at all, and
why `LogisticRegression` sits in `oop_ml.classification` and still reuses
this same machinery, right down to the design matrix and the intercept split.

## The Rules I Held To

- Raw input becomes a `Column` once, at the boundary, and nothing downstream
  ever validates it again.
- A rule that spans several values belongs in a value object that enforces it in
  its constructor, rather than in a `check_*` function that every caller has to
  remember to call.
- Nothing returns a tuple of several different things. If I found myself
  reaching for `return first, second`, that pairing was a class I had not
  written yet, and the caller should never have to know the positional order.
- Learned parameters are read-only and raise `NotFittedError` if you read them
  before `fit`. They carry no trailing underscore: `model.coefficients`, not
  `model.coefficients_`. The suffix is a scikit-learn habit that exists to mark
  fitted state, and `NotFittedError` already does that job at the moment it
  matters, with a message instead of a naming convention you have to know.
- A hyperparameter you misspell raises rather than being ignored. Pydantic's
  default is to drop an unrecognised keyword, which for hyperparameters means
  silently keeping the default, and a default is by construction a plausible
  number, so `RidgeRegression(alpha=2.0)` would fit an unpenalised model and
  report perfectly reasonable-looking coefficients. This is not hypothetical: an
  example in this repository was written against the wrong field name and ran
  quietly on the wrong train/test split until a type checker pointed at it.
- Public methods hand back Python floats rather than numpy scalars.
- Everything is typed, and the package ships a `py.typed` marker so that your
  type checker sees the annotations instead of treating all of this as untyped.
