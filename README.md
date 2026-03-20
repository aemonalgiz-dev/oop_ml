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

## Models That Learn Nothing

Every model above ends up holding a handful of numbers that stand in for the
data. `fit` does the work, the training rows are discarded, and `predict` is
arithmetic on coefficients. Nearest neighbours inverts all three. `fit`
validates its inputs and remembers them, and every decision waits until you
ask a question.

```python
from oop_ml import KNearestNeighboursClassifier, DistanceMetric

model = KNearestNeighboursClassifier(n_neighbours=5)
model.fit(features, species)

model.predict(new_features)  # majority vote among the 5 nearest rows
model.predict_probabilities(new_features)  # each class's share of those 5
model.n_remembered  # 300 — the model's size, which no other model here has
```

What that buys is shape. Nothing constrains the decision surface, so a class
sitting in a ring around another one is read straight off the rows. On exactly
that data, [examples/nearest_neighbours.py](examples/nearest_neighbours.py)
reports:

```
                       model  test accuracy
          LogisticRegression         0.5100
KNearestNeighboursClassifier         0.9100
```

The linear model did not error, and it did not fail to converge. It fitted the
best straight line available and the best straight line available is worth
nothing when the boundary is a circle.

Multi-class comes free, which is worth pausing on given how much machinery the
previous section needed. There is no reference class here, no flat ridge in a
likelihood, no one-vs-rest wrapper and no softmax — a vote does not care how
many candidates are on the ballot. Two classes and twenty are the same code,
because all that machinery existed to make *parameters* identifiable, and there
are no parameters.

### What it costs

Three things, and none of them is optional.

**`k` is the entire bias-variance dial.** At `k=1` every training row is its own
nearest neighbour, so training accuracy is 1.0 by construction and means
precisely nothing:

```
  k  train accuracy  test accuracy
  1          1.0000         0.8500
  3          0.9267         0.9000
  5          0.9100         0.9100
 15          0.8933         0.9000
 51          0.8833         0.8700
300          0.5167         0.4500
```

Training accuracy falls the whole way down that table while test accuracy
rises, peaks and only then falls. The two columns disagree about the best model
over most of the range, and wherever they disagree the training column is the
wrong one.

**Standardising stops being a convenience.** Distance is a sum over the
features, so a column measured in larger numbers drowns one measured in
smaller. Predicting temperature from the hour of day, with a junk pressure
column in pascals sitting beside it:

```
      inputs  test R^2
 as supplied   -0.0051
standardised    0.8898
```

Every other model here would have shrunk the useless column toward zero on the
evidence. This one has no coefficients to shrink and no way to notice.

**The metric is not a tuning knob, it is the model.** Six are built in, and
they disagree by a lot:

```
   metric  test R^2
euclidean    0.8898
manhattan    0.9037
chebyshev    0.8713
   cosine    0.6126
  hamming   -2.5184
 canberra    0.8156
```

`EUCLIDEAN`, `MANHATTAN` and `CHEBYSHEV` are one formula at `p = 2`, `1` and
infinity, so raising `p` shifts weight onto whichever single feature disagrees
most. The other three are genuinely different questions. `COSINE` compares
direction and ignores magnitude, which is why `(1, 1, 1)` and `(3, 3, 3)` are
at distance 0. `HAMMING` asks only whether two values are *equal*, making it
the only one of the six that means anything on categorical codes — and, on the
continuous columns above, the only one that scores worse than predicting the
mean. `CANBERRA` divides each gap by the size of the values involved, which
makes it the one option that tolerates unstandardised input.

The enum names six; the calculation behind it takes any `p`, and any object
with a `between` method is accepted:

```python
from oop_ml import MinkowskiDistance

KNearestNeighboursRegressor(metric=MinkowskiDistance(3))
```

There is also a fourth cost that no amount of care removes: the model cannot
extrapolate. Query past the edge of the training data and every neighbour is on
the same side, so the answer is their mean and stays their mean forever. A
linear model extends its line instead — confidently, and just as unfoundedly.

## Trees, Which Read Instead Of Weigh

Every model above assembles its answer from weights. A tree asks questions.

```python
from oop_ml import DecisionTreeClassifier

model = DecisionTreeClassifier(min_samples_split=5)
model.fit(features, passed)

print(model.describe())
```

```
slept < 6.25 ?  [n=15, impurity=0.4800, gain=0.2133]
  predict 0  [n=6, impurity=0.0000]
  studied < 4.5 ?  [n=9, impurity=0.4444, gain=0.2778]
    predict 0  [n=4, impurity=0.3750]
    predict 1  [n=5, impurity=0.0000]
```

That is the entire model, and you can read it. Sleep is the first gate; study is
tested only inside the region where sleep already cleared its threshold. That
conditional — *studying helps only if you slept* — is an interaction, and a tree
expresses it by nesting one question inside another. A linear model needs the
product term handed to it as a column before it can say the same thing.

It costs nothing for units, either. A split compares one column against one
threshold, so nothing is summed across features and no column can drown another
by being measured in thousands. Standardising a tree's inputs changes precisely
nothing — which is the opposite of the neighbour models above, where it was part
of being correct.

### What it pays for that

**Stopping rules are the whole of the regularisation.** Left alone the tree
carves out a box per row:

```
min_samples_split=5   depth 2   leaves 3   training accuracy 0.9333
defaults              depth 4   leaves 5   training accuracy 1.0000
```

The second is not the better model, it is the memorised one — the same failure
`k=1` produced for a neighbour model, wearing a different hat. Watch `n_leaves`
against your row count.

**The search is greedy, and cannot be otherwise.** Finding the optimal tree is
NP-hard, so each node takes the best split available now and never reconsiders.
Usually fine, occasionally fatal: on a parity target every single first split
scores exactly zero, and the recursion stops at the root even though two levels
would separate the classes perfectly.

**The boundary is axis-aligned.** A diagonal is only reachable as a staircase.

## Watching A Model Work

Every model here has a second route through its own calculation. The efficient
one returns the answer and discards the working; the observed one keeps all of
it.

```python
search = model.split_search(rows, targets)

len(search)        # 22 candidates considered
search.best        # Split(slept < 6.25, gain=0.2133)
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
collinearity shows — the coefficients just come back large and cancelling.
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
| `oop_ml.core.data` | `Column`, `Feature`, `FeatureSet`, `Coefficients` — the vocabulary every other package speaks |
| `oop_ml.core.evaluation` | `RegressionEvaluation`, `ClassificationEvaluation`, `MultiClassEvaluation` |
| `oop_ml.core.base` | The generic `Estimator[InputT, TargetT]` hierarchy, plus the `LinearModel`, `IterativeSolver`, `NeighbourModel` and `TreeModel` frames |
| `oop_ml.core.distance` | `DistanceMetric` and the six `Distance` calculations behind it |
| `oop_ml.core.tree` | Impurity measures, criteria, `Split`, and the nodes a fitted tree is made of |
| `oop_ml.core.observation` | The `Observation` protocol — the second route through a calculation |
| `oop_ml.core.solving` | `SolverPath`, `NormalEquations`, `LeastSquaresLine` |
| `oop_ml.core.neighbours` | `NeighbourSearch` — every distance behind a prediction |
| `oop_ml.regression.least_squares` | Simple, multiple, gradient descent — squared error and nothing added to it |
| `oop_ml.regression.penalised` | Ridge and lasso, where the *shape* of the penalty is the whole difference |
| `oop_ml.regression.neighbours` | `KNearestNeighboursRegressor` — no assumed shape, no coefficients |
| `oop_ml.regression.trees` | `DecisionTreeRegressor` — a piecewise-constant surface you can read |
| `oop_ml.classification.binary` | `LogisticRegression` and `NewtonLogisticRegression`, one objective and two solvers |
| `oop_ml.classification.multiclass` | `MultinomialLogisticRegression` and `OneVsRestClassifier` |
| `oop_ml.classification.neighbours` | `KNearestNeighboursClassifier`, where any number of classes is the same code |
| `oop_ml.classification.trees` | `DecisionTreeClassifier`, whose boundary is a union of boxes |
| `oop_ml.preprocessing.standardization` | `Standardizer` and the `FeatureScalings` it learns |
| `oop_ml.preprocessing.polynomial` | `PolynomialFeatures` and the `PolynomialTerms` it builds |
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
- A hyperparameter you misspell raises rather than being ignored. Pydantic's
  default is to drop an unrecognised keyword, which for hyperparameters means
  silently keeping the default — and a default is by construction a plausible
  number, so `RidgeRegression(alpha=2.0)` would fit an unpenalised model and
  report perfectly reasonable-looking coefficients. This is not hypothetical: an
  example in this repository was written against the wrong field name and ran
  quietly on the wrong train/test split until a type checker pointed at it.
- Public methods hand back Python floats rather than numpy scalars.
- Everything is typed, and the package ships a `py.typed` marker so that your
  type checker sees the annotations instead of treating all of this as untyped.

## Examples

There are eleven runnable scripts in [examples/](examples/), and each one is
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
                 task           size  oop_ml (s)  sklearn (s)  ratio       answers
                  OLS        1000x20      0.0003       0.0018   0.2x       matches
                Ridge        1000x20      0.0003       0.0011   0.3x       matches
                Lasso        1000x20      0.0022       0.0012   1.9x       matches
     Gradient descent        1000x20      0.0166       0.0648   0.3x  not compared
      Logistic ascent        1000x20      0.0048       0.0062   0.8x       matches
      Logistic Newton        1000x20      0.0020       0.0053   0.4x       matches
         Standardizer        1000x20      0.0009       0.0009   1.0x       matches
                  OLS       20000x50      0.0109       0.0333   0.3x       matches
                Ridge       20000x50      0.0113       0.0123   0.9x       matches
                Lasso       20000x50      0.1258       0.1035   1.2x       matches
     Gradient descent       20000x50      1.8879       2.5219   0.7x  not compared
      Logistic ascent       20000x50      0.3805       0.0363  10.5x       matches
      Logistic Newton       20000x50      0.0865       0.0642   1.3x       matches
         Standardizer       20000x50      0.0259       0.0181   1.4x       matches
PolynomialFeatures d3         2000x8      0.0038       0.0028   1.4x       matches
PolynomialFeatures d3        2000x12      0.0095       0.0056   1.7x       matches
       k-NN regressor   5000x20 q500      0.0264       0.0096   2.8x       matches
      k-NN classifier   5000x20 q500      0.0261       0.0095   2.7x       matches
       k-NN regressor  20000x20 q500      0.0686       0.0157   4.4x       matches
      k-NN classifier  20000x20 q500      0.0657       0.0161   4.1x       matches
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

The four neighbour rows are the ones this library loses, and they are worth
reading because the reason is not the one I expected.

They started far worse. The first implementation computed distance by pairing
every query against every remembered row and reducing over the features, which
is the honest translation of the definition and builds an
`(n_queries, n_remembered, n_features)` array to do it — 1.6 GB, for an answer
occupying 80 MB. Expanding the square into `||a||^2 - 2 a.b + ||b||^2` turns
the expensive part into one matrix multiply and never builds that array at all,
which measured 12x quicker. Counting the votes was a second, smaller find:
`np.add.at` is the correct way to accumulate into repeated indices and also a
slow one, and flattening the two-dimensional tally into a single `bincount` ran
5.7x faster. Together those took prediction from 3.56s to 0.32s at
20000 remembered rows.

The expansion has a well-known catch, which is why it is not simply the obvious
thing to do. Recovering a small number by subtracting two large nearly-equal
ones loses it: on two points 1e-06 apart with coordinates near 1e06 the naive
form returns exactly `0.0`, a 100% error, and can go slightly negative so that
the square root yields `nan` for a point measured against itself. Both symptoms
come from the coordinates being far from the origin rather than from the points
being close together, so subtracting the remembered rows' mean from both inputs
first removes them — it shifts every point equally, leaving all distances
unchanged. On that same pathological pair the centred form is exact to the last
bit, and across random data the largest relative error against the definition
was 3.5e-16.

Those two findings left the row at 15.7x, and I wrote at that point that what
remained was a fused-kernel gap numpy could not close. That was half right, and
the half I had wrong was the more useful half. Timing the two halves separately
showed the work splitting almost evenly — 0.064s building the distance matrix
against 0.059s selecting from it — and only the first half was parallel, because
BLAS threads the matrix multiply on its own while `argpartition` runs on one
core. Roughly half the calculation was using one core of twenty-four.

Queries are independent of each other, so the fix is to hand blocks of them to
threads. Threads rather than processes, because numpy releases the interpreter
lock for exactly these operations, which makes it real parallelism with no
pickling and no copies — every worker reads the same remembered rows. That is
another 2.5x to 3.4x, and it takes the largest row from 15.7x to 4.4x.

It needs a floor. Starting a pool costs a millisecond or two, which is nothing
against a hundred and ruinous against one: at 500 queries by 500 remembered rows
the threaded route measured nine times *slower*, so below half a million
(query, remembered) pairs it stays on one thread. The threshold counts pairs
rather than either dimension alone, because the work is the product of the two.

What is left now really is a fused-kernel gap. scikit-learn never materialises
the distance matrix at all; it keeps a per-query heap of the k smallest while
streaming the distances past it, in Cython, across threads. numpy has no way to
express that — every intermediate is a real array — so 160 MB gets written and
read back before the selection even starts. Blocking over the remembered rows to
keep those intermediates in cache is the obvious numpy answer to that, and it
measured *worse*, between 0.4x and 0.7x, because maintaining a running top-k
across blocks costs more than the cache locality returns.

The interesting part is what does *not* close it. The textbook answer to a slow
brute-force sweep is a spatial index, and at twenty features it is dramatically
worse:

```
 dims     brute   kd_tree   verdict
    2    0.0158    0.0038   tree wins (0.2x)
    4    0.0141    0.0073   tree wins (0.5x)
    8    0.0147    0.0906   brute wins (6.2x)
   12    0.0155    0.2979   brute wins (19.3x)
   16    0.0149    0.4995   brute wins (33.4x)
   20    0.0153    0.6246   brute wins (40.8x)
   30    0.0173    0.8681   brute wins (50.1x)
```

Those are both scikit-learn, the same library against itself, and its `auto`
setting picks brute force from eight dimensions upward for exactly this reason.
A KD-tree earns its keep by pruning branches that cannot contain a nearer
point, and that pruning needs some points to be meaningfully closer than
others. Sample a thousand uniform points and the gap between the nearest and
the farthest, relative to the nearest, falls from 4390.95 in one dimension to
2.61 in ten and 0.28 in two hundred. By the time everything is nearly
equidistant there is nothing left to prune, and the tree is a slower way to
visit every point anyway. The crossover here sits between four and eight
features, which is a good deal lower than most descriptions of KD-trees
suggest.

So the remaining gap is an implementation gap and an honest one, unlike the
logistic row above it, where the fix was a different algorithm. Both answers
agree exactly with scikit-learn's, the classifier's labels included.

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

Every model family that predicts from a single fitted thing is here. Regression:
simple, multiple, gradient descent, ridge, lasso. Classification: two binary
logistic solvers, softmax, one-vs-rest. Both non-parametric families, k-nearest
neighbours and decision trees, across both tasks. Plus preprocessing, splitting,
cross-validation and three kinds of evaluation — and a second, observed route
through every calculation with intermediates worth seeing.

That is **1170 passing tests**, `ruff` and `pyright` clean, with no stubs left.

What is missing is not a gap in that list but three areas beside it: **ensembles**
(bagging, forests, boosting — trees are their prerequisite, which is why they are
next), **unsupervised learning** entirely, since every base class here takes a
target, and **kernels**. Naive Bayes and discriminant analysis are smaller and
would slot in anywhere.

Rather than leave you to discover these the hard way, here is what is not built
yet, in the order it will matter if you are putting this into an application:

1. **Persistence.** A fitted model cannot be serialised yet, so a process has to
   fit at start-up instead of loading a trained artifact.
2. **`Pipeline`.** A transformer fitted outside of a cross-validation loop can
   leak across the split, and your serving path currently has to re-apply it by
   hand.
3. **Cross-validated classification.** `CrossValidation` scores with R², so it
   only speaks to regressors. Fixing that needs stratified folds as well: plain
   k-fold on a rare class produces folds with no positives at all by k=10, and
   recall on such a fold is undefined rather than zero.
4. **A second-order multi-class solver.** The softmax model walks uphill where
   the binary one can jump. Its Hessian has cross-class blocks and is
   `(K-1)p` square, so Newton there is a judgement call rather than the clear
   win it was at two classes.
5. **Distance-weighted neighbours.** Every neighbour currently counts the same
   regardless of how near it is. Weighting by inverse distance is a genuine
   improvement and also a different model — it has no `k` at which it stops
   caring, and it needs a rule for a query sitting exactly on a training row,
   where the weight is infinite.
6. **Tree pruning.** Growth stops on rules chosen up front. Cost-complexity
   pruning grows first and cuts back afterwards, judging a subtree by what it
   was worth rather than by a limit set before anything was seen.
7. **Hyperparameter search.** `CrossValidation` scores a candidate; nothing
   searches a space of them, so choosing `max_depth` is currently a loop you
   write yourself.
