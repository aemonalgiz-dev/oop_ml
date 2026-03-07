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

model.intercept  # 99.55
model.coefficients["floor_area_sqm"]  # 2.81
model.coefficients["bathrooms"]  # 2.33
```

The constructor takes hyperparameters and `fit` takes data, and the library
holds to that distinction everywhere. Hyperparameters are validated Pydantic
fields, so `RidgeRegression(penalty=-1)` fails immediately at construction
rather than somewhere deep inside a solve, where the traceback would tell you
very little about what you actually did wrong.

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

## Classifying Instead Of Predicting

```python
from oop_ml import Feature, LogisticRegression

model = LogisticRegression(learning_rate=0.05, max_epochs=200_000)
model.fit([hours_studied, hours_slept], passed)

model.predict_probability(features)  # 0.0094 ... 0.9903
model.predict(features)  # 0.0 ... 1.0
model.odds_multiplier_for("hours_studied")  # 2.55
```

Almost nothing about the API moves. Features go in by name, coefficients come
back by name, and `evaluate` still hands you an object. What changes is the
question being asked, and two consequences follow from that.

The first is that there is no closed form to jump to. Setting the gradient of
the logistic likelihood to zero leaves the coefficients trapped inside a
sigmoid, so gradient ascent here is not the slow way of doing what `solve`
does; it is the only way. That makes `converged` worth reading rather than
decorative. On perfectly separable classes the maximum likelihood estimate does
not exist at all, the coefficients grow without bound, and the only thing that
ends the walk is your epoch cap.

The second is that a coefficient no longer means what it meant in regression.
It is a multiplier on the odds and not an amount added to the answer, so
`odds_multiplier_for` returns the number you actually want instead of leaving
you to remember which direction to exponentiate. One more hour of study
multiplies the odds of passing by 2.55, and it does so no matter where on the
curve you started, which is not something the change in probability does.

That learning rate is small and the epoch count is large for a reason worth
knowing about. Both predictors here are raw hours, and a single step size has
to serve every direction at once, so `Standardizer` earns its keep before a
gradient method rather than after it.

```python
evaluation = model.evaluate(features, target)

evaluation.confusion_matrix.false_negatives  # 17
evaluation.recall  # 0.8350
evaluation.precision  # 0.7963
```

`RegressionEvaluation` has a sibling rather than a subclass here, because the
two of them share no metric worth the name. R-squared has nothing useful to say
about a column of zeroes and ones.

One decision in there is worth flagging before it surprises you. Ask a model
that never predicts positive for its precision and you get an
`UndefinedMetricError`, not a zero and not a nan. A model that has never fired
has not made a wrong positive call, and it has not made a right one either, so
zero would be an answer to a question nobody asked.

The threshold is a field on the model rather than an argument to `predict`,
which matters more than it looks. On an unbalanced target, where the cut falls
is the decision that determines whether the thing is useful, and a model built
to miss as little as possible should be a different object that you can pass
around and log, not a keyword somebody has to remember at every call site.

There are two logistic solvers, and picking between them is a real decision
rather than a default worth hiding. `LogisticRegression` walks uphill and takes
a `learning_rate`. `NewtonLogisticRegression` uses the second derivative to work
out its own step length, so it has no learning rate at all and converges in
single-digit iterations rather than hundreds of epochs:

```python
walked = LogisticRegression(learning_rate=0.5).fit(features, passed)
jumped = NewtonLogisticRegression().fit(features, passed)

walked.epochs_run  # 749
jumped.iterations_run  # 7
```

Both land on `+2.20430913` for the same coefficient. The objective is concave,
so there is one maximum and no room to disagree about where it is. Newton is
the one to reach for unless the problem is wide, since building the curvature
matrix costs a factor of the predictor count per pass; the benchmark below
shows where that trade turns over.

## More Than Two Classes

```python
from oop_ml import MultinomialLogisticRegression

model = MultinomialLogisticRegression(learning_rate=1.0).fit(features, species)

model.n_classes  # 3
model.predict_probabilities(features)  # (300, 3), every row summing to 1
model.coefficients_for(2)["sepal_length"]  # class 2's weight, against class 0
```

Softmax gives each class its own weight vector and normalises across them, so
the answer is a distribution by construction rather than by tidying up
afterwards. The gradient turns out to be the binary one with the 0/1 label
swapped for a 0/1 indicator of "is this row class k", and at two classes the
whole thing collapses exactly onto `LogisticRegression` — checked in the tests
to 1e-6, because it is an identity rather than an approximation.

Class 0 is held at zero and not fitted. That is not a shortcut: add the same
constant to every class's weight for a feature and every probability comes back
unchanged, so the likelihood has a flat ridge through it and no unique maximum.
Pinning one class down is what makes the answer reproducible, and it is why the
remaining coefficients read as "against class 0".

`OneVsRestClassifier` is the other route, wrapping any binary classifier:

```python
wrapper = OneVsRestClassifier(binary_model=LogisticRegression())
wrapper.fit(features, species)
wrapper.model_for(2).coefficients["sepal_length"]  # that class's own model
```

It needs no new mathematics and it makes no promises the maths does not
support. The K models were fitted independently, so their probabilities do not
sum to one — measured on 300 rows, the row totals ran from 0.4137 to 1.7790
against exactly 1.0000 for softmax, and the two disagreed on the predicted class
for 13 of them. The library reports those totals raw. Normalising them would
produce something that adds to one without being the probability of anything.

### Scoring it

```python
evaluation = model.evaluate(features, species)

evaluation.accuracy  # 0.9033
evaluation.micro_recall  # 0.9033
evaluation.macro_recall  # 0.8731
evaluation.per_class_recall  # [0.9636, 0.8333, 0.8222]
```

Accuracy and both micro figures are the same number, always. Every row gets
exactly one prediction, so the pooled numerator is the diagonal and both pooled
denominators are every row — reporting all three is reporting one number three
times.

Macro is the one that says something else. It averages the per-class scores, so
the class holding 15% of the rows moves it exactly as much as the class holding
55%. On a target split 165 / 90 / 45 that is a three-point gap here and can
easily be twenty; wherever macro sits well below micro, the model is failing a
class too small to dent the overall figure, which is usually the class somebody
cared about.

## The Package Layout

| Package | Contents |
|---------|----------|
| `oop_ml.core.data` | `Column`, `Feature`, `FeatureSet`, `Coefficients` — the vocabulary every other package speaks |
| `oop_ml.core.evaluation` | `RegressionEvaluation`, `ClassificationEvaluation`, `MultiClassEvaluation` |
| `oop_ml.core.base` | The generic `Estimator[InputT, TargetT]` hierarchy, plus the `LinearModel` and `IterativeSolver` frames |
| `oop_ml.regression` | Simple, multiple, ridge, gradient descent, lasso |
| `oop_ml.classification` | `LogisticRegression`, `NewtonLogisticRegression`, `MultinomialLogisticRegression`, `OneVsRestClassifier` |
| `oop_ml.preprocessing` | `Standardizer`, `PolynomialFeatures` |
| `oop_ml.model_selection` | `Dataset`, train/test and k-fold splitters, `CrossValidation` |

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
- Public methods hand back Python floats rather than numpy scalars.
- Everything is typed, and the package ships a `py.typed` marker so that your
  type checker sees the annotations instead of treating all of this as untyped.

## Examples

There are ten runnable scripts in [examples/](examples/), and each one is
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
                  OLS   1000x20      0.0001       0.0009   0.1x       matches
                Ridge   1000x20      0.0001       0.0005   0.3x       matches
                Lasso   1000x20      0.0008       0.0005   1.8x       matches
     Gradient descent   1000x20      0.0068       0.0297   0.2x  not compared
      Logistic ascent   1000x20      0.0023       0.0034   0.7x       matches
      Logistic Newton   1000x20      0.0010       0.0027   0.4x       matches
         Standardizer   1000x20      0.0004       0.0004   1.0x       matches
                  OLS  20000x50      0.0066       0.0222   0.3x       matches
                Ridge  20000x50      0.0064       0.0073   0.9x       matches
                Lasso  20000x50      0.0751       0.0598   1.3x       matches
     Gradient descent  20000x50      1.5596       1.0438   1.5x  not compared
      Logistic ascent  20000x50      0.1813       0.0232   7.8x       matches
      Logistic Newton  20000x50      0.0584       0.0300   1.9x       matches
         Standardizer  20000x50      0.0073       0.0104   0.7x       matches
PolynomialFeatures d3    2000x8      0.0021       0.0014   1.5x       matches
PolynomialFeatures d3   2000x12      0.0058       0.0033   1.8x       matches
```

Ratios below 1.0 mean this library was faster, which surprised me the first time
I ran it. The reason is that numpy already hands the heavy linear algebra to
BLAS either way, so scikit-learn's Cython has very little left to win, while the
input validation it performs on every call is enough to lose it the smaller
comparisons outright.

The iterative solvers are where Cython should genuinely pull ahead, and the
first version of this table said exactly that, with lasso trailing by nearly
five times. Then I profiled it, and what turned up was more embarrassing than a
language gap. My coordinate descent was rebuilding the residual from scratch for
every column of every sweep, so a sweep cost O(n p^2) when it had no business
costing more than O(n p). It now carries the residual and repairs it in place
after each coefficient moves, with the column norms computed once up front. That
alone got it to around 1.2x, and not one coefficient changed. The polynomial
expansion had the same disease, recomputing identical powers for every term that
asked for them, and memoising took it from 4.1x to under 2x.

The one I would not have guessed was the memory layout. Every linear model here
reaches for X.T @ v, and an iterative one wants it once an epoch. Stored
row-major, that product walks down a column with an entire row's stride between
consecutive elements, so on a tall matrix it misses cache on very nearly every
access; at 20000x51 that costs 4.4x what the same product costs on column-major
storage. A feature set is assembling itself a column at a time regardless, so
storing it that way round costs nothing and there was no trade to weigh. That
one change took gradient descent from 1.2x to 0.9x, ridge to parity,
standardisation from 1.2x to 0.7x, and logistic regression from 19.4x to 9.2x,
without altering a line of arithmetic anywhere.

The two logistic rows are one objective solved two ways, and the distance
between them is the most instructive thing in the table. Both land on the same
coefficients as scikit-learn, to within 1e-8. What differs is how many passes
that takes: 394 epochs of gradient ascent, 8 Newton iterations, 13 iterations of
lbfgs.

Gradient ascent knows only which way is uphill and has to be told how far to
walk, which is all a learning rate ever was. Newton reads the curvature too, and
the curvature is exactly the information a step length needs, so it computes the
distance instead of guessing it. That is not a faster implementation of the same
algorithm, it is a different one, and no amount of array tuning takes the first
to where the second begins.

For a while this was the one row scikit-learn won outright, at 19.4x. Memory
layout took it to 9.2x and writing the Newton solver took it to 1.9x, beating
lbfgs on the smaller problem at 0.4x. What is left is a real trade rather than a
defect: lbfgs does O(n p) work per iteration where IRLS does O(n p^2) to build
the curvature matrix, so it needs more iterations and each one is cheaper. At
fifty predictors that still favours it, narrowly. At twenty it does not.

The learning rate is worth one more line, because it is why the ascent row moves
around between runs. Raising it from 0.5 to 2.0 took that row from 135x to 20x
without shifting a coefficient. A number with that much leverage over the
result is not a footnote to the benchmark; it is the thing the second solver
exists so that you never have to pick.

Note the last column, because a benchmark that only reported timings would be
close to worthless. Every task that can be compared agrees with scikit-learn's
coefficients to within 1e-6, exact zeros in the lasso solution included, once
the penalty parameterisations are converted. Gradient descent is marked as not
compared on purpose; batch descent and stochastic descent reach the same
objective by different routes and stop on different rules, so matching
coefficients was never the expectation. Those numbers came off my machine and
my BLAS, so run it on yours.

Where scikit-learn is genuinely ahead is breadth, second-order optimisation,
the surrounding ecosystem, and robustness on degenerate input; a perfectly
collinear design matrix currently raises `LinAlgError` here rather than
returning a minimum-norm solution. If you
need any of those, use it. Reach for this library when the shape of the API is
what is costing you.

## Where This Stands

Regression is complete: five models, two preprocessors, and cross-validation.
Classification has landed alongside it: two binary logistic solvers, softmax and
one-vs-rest for more than two classes, and the confusion-matrix metrics for
both. That is 688 tests, `ruff` and `pyright` clean.

Rather than leave you to discover these the hard way, here is what is not built
yet, in the order it will matter if you are putting this into an application:

1. **Persistence.** A fitted model cannot be serialised yet, so a process has to
   fit at start-up instead of loading a trained artifact.
2. **`Pipeline`.** A transformer fitted outside of a cross-validation loop can
   leak across the split, and your serving path currently has to re-apply it by
   hand. This is next.
3. **Cross-validated classification.** `CrossValidation` scores with R², so it
   only speaks to regressors. Fixing that needs stratified folds as well: plain
   k-fold on a rare class produces folds with no positives at all by k=10, and
   recall on such a fold is undefined rather than zero.
4. **A second-order multi-class solver.** The softmax model walks uphill where
   the binary one can jump. Its Hessian has cross-class blocks and is
   `(K-1)p` square, so Newton there is a judgement call rather than the clear
   win it was at two classes.
