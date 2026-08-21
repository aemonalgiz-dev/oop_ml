# oop_ml

An object oriented machine learning library for Python, with cleaner
abstractions for modeling, testing, and validating models.

## Where The Trap Is Set

The distance between a model that works in a notebook and a model that works in
an application has always seemed larger to me than it needs to be. Most Python
machine learning libraries take arrays in and give arrays back, which is
perfectly reasonable for exploratory work, although once that model is sitting
behind a request handler, the meaning of column three lives somewhere outside of
the model entirely, usually in a constant that you are now responsible for
keeping in sync.

```python
FEATURE_ORDER = ["floor_area_sqm", "bathrooms"]  # must match training


def price(record: dict[str, float]) -> float:
    row = np.array([[record[name] for name in FEATURE_ORDER]])
    return float(model.predict(row)[0])
```

Consider what happens six months from now, when someone reorders the columns in
the training script. This function still returns a number; it simply returns the
wrong one. Nothing raises, nothing logs, and the failure sits there quietly
until somebody happens to check.

The fix is not clever. It is only that a feature should know its own name.

```python
def price(record: dict[str, float]) -> float:
    features = [Feature(name, [value]) for name, value in record.items()]
    return float(model.predict(features)[0])
```

There is no ordering left to get wrong, and a missing or misspelled feature
raises `InvalidValuesError` instead of quietly producing something plausible.

## Installation

```bash
pip install -e ".[dev]"
```

## Fitting Your First Model

```python
from oop_ml import Feature, RidgeRegression

model = RidgeRegression(penalty=1.0)
model.fit(
    [
        Feature("floor_area_sqm", [72, 140, 96, 210, 55, 118, 165, 88]),
        Feature("bathrooms", [1, 2, 1, 3, 1, 2, 2, 1]),
    ],
    Feature("price_thousands", [310, 505, 372, 690, 240, 448, 560, 350]),
)

model.intercept_  # 99.55
model.coefficients_["floor_area_sqm"]  # 2.81
model.coefficients_["bathrooms"]  # 2.33
```

The constructor takes hyperparameters and `fit` takes data, and the library
holds to that distinction everywhere. Hyperparameters are validated Pydantic
fields, so `RidgeRegression(penalty=-1)` fails immediately at construction
rather than somewhere deep inside a solve, where the traceback would tell you
very little about what you actually did wrong.

## Names Instead Of Positions

```python
model.coefficients_["floor_area_sqm"]  # 2.81
"garden" in model.coefficients_  # False
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

There are nine exception types, all of them deriving from `MLLibError`, so a
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

## The Package Layout

| Package | Contents |
|---------|----------|
| `oop_ml.core` | `Column`, `Feature`, `FeatureSet`, `Coefficients`, `RegressionEvaluation`, and the generic `Estimator[InputT, TargetT]` hierarchy |
| `oop_ml.regression` | Simple, multiple, ridge, gradient descent, lasso |
| `oop_ml.preprocessing` | `Standardizer`, `PolynomialFeatures` |
| `oop_ml.model_selection` | `Dataset`, train/test and k-fold splitters, `CrossValidation` |

All of it is re-exported from the top level, so `from oop_ml import Feature` is
all that most code needs, though the full path is there whenever you want to be
explicit about where something lives.

Adding a linear model means writing exactly one method:

```python
class RidgeRegression(LinearFeatureRegressor):
    penalty: float = Field(default=1.0, ge=0.0)

    def _solve(self, design_matrix, target_column): ...
```

Validation, the design matrix, the intercept split, coefficient pairing by name,
and `predict` are all inherited. Note that the directory names the task and not
the model family; "linear" means linear in the coefficients, which is why
`PolynomialFeatures` can fit curves without the estimator changing at all, and
why logistic regression will end up in `oop_ml.classification` while still
reusing this same machinery.

## The Rules I Held To

- Raw input becomes a `Column` once, at the boundary, and nothing downstream
  ever validates it again.
- A rule that spans several values belongs in a value object that enforces it in
  its constructor, rather than in a `check_*` function that every caller has to
  remember to call.
- Nothing returns a tuple of several different things. If I found myself
  reaching for `return first, second`, that pairing was a class I had not
  written yet, and the caller should never have to know the positional order.
- Learned parameters are read-only, end in a trailing underscore, and raise
  `NotFittedError` if you read them before `fit`.
- Public methods hand back Python floats rather than numpy scalars.
- Everything is typed, and the package ships a `py.typed` marker so that your
  type checker sees the annotations instead of treating all of this as untyped.

## Examples

There are seven runnable scripts in [examples/](examples/), and each one is
written the way you would write your own code against the installed package
rather than against the library's internals.

```bash
python -m examples.model_selection
```

That one walks the whole arc: hold out a test set, cross-validate a list of
candidate penalties against the remainder, choose one, refit, and finally report
a single held-out number that has not been spent on anything else along the way.

## On scikit-learn

Nothing here wraps it. The algorithms are implemented directly against numpy,
which means every line is readable and auditable, and each module carries its
derivation and the trade-offs behind it in the docstring.

That usually raises a fair question about performance, so rather than assert an
answer I shipped the benchmark:

```bash
pip install -e ".[benchmark]"
python -m benchmarks.against_scikit_learn
```

```
                 task      size  oop_ml (s)  sklearn (s)  ratio       answers
                  OLS   1000x20      0.0002       0.0012   0.2x       matches
                Ridge   1000x20      0.0002       0.0005   0.4x       matches
                Lasso   1000x20      0.0011       0.0006   1.9x       matches
     Gradient descent   1000x20      0.0108       0.0385   0.3x  not compared
         Standardizer   1000x20      0.0006       0.0008   0.8x       matches
                  OLS  20000x50      0.0111       0.0231   0.5x       matches
                Ridge  20000x50      0.0110       0.0077   1.4x       matches
                Lasso  20000x50      0.0894       0.0707   1.3x       matches
     Gradient descent  20000x50      1.7636       1.4983   1.2x  not compared
         Standardizer  20000x50      0.0120       0.0129   0.9x       matches
PolynomialFeatures d3    2000x8      0.0028       0.0018   1.6x       matches
PolynomialFeatures d3   2000x12      0.0064       0.0037   1.7x       matches
```

Ratios below 1.0 mean this library was faster, which surprised me the first time
I ran it. The reason is that numpy already hands the heavy linear algebra to
BLAS either way, so scikit-learn's Cython has very little left to win, while the
input validation it performs on every call is enough to lose it the smaller
comparisons outright.

The iterative solvers are where scikit-learn's Cython genuinely should pull
ahead, and the first version of this table said so, with lasso trailing by
nearly five times. Profiling turned up something more embarrassing than a
language gap. Coordinate descent was rebuilding the residual from scratch for
every column of every sweep, which made a sweep cost O(n p^2) where it should
cost O(n p). Carrying the residual and repairing it in place after each
coefficient moves, with the column norms computed once, brought that to 1.3x
without changing a single coefficient it produces. The polynomial expansion was
recomputing the same powers for every term that needed them, and memoising those
took it from 4.1x to 1.7x.

Note the last column, because a benchmark that only reported timings would be
close to worthless. Every task that can be compared agrees with scikit-learn's
coefficients to within 1e-6, exact zeros in the lasso solution included, once
the penalty parameterisations are converted. Gradient descent is marked as not
compared on purpose; batch descent and stochastic descent reach the same
objective by different routes and stop on different rules, so matching
coefficients was never the expectation. Those numbers came off my machine and
my BLAS, so run it on yours.

Where scikit-learn is genuinely ahead is breadth, the surrounding ecosystem, and
robustness on degenerate input; a perfectly collinear design matrix currently
raises `LinAlgError` here rather than returning a minimum-norm solution. If you
need any of those, use it. Reach for this library when the shape of the API is
what is costing you.

## Where This Stands

Regression is complete: five models, two preprocessors, and cross-validation,
sitting at 445 tests with `ruff` and `pyright` clean.

Rather than leave you to discover these three the hard way, here is what is not
built yet, in the order it will matter if you are putting this into an
application:

1. **Persistence.** A fitted model cannot be serialised yet, so a process has to
   fit at start-up instead of loading a trained artifact.
2. **`Pipeline`.** A transformer fitted outside of a cross-validation loop can
   leak across the split, and your serving path currently has to re-apply it by
   hand.
3. **Classification.** This is next, and the generic base is already shaped for
   it.
