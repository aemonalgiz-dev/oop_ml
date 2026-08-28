# The models, and what measuring them showed

A walk through each family, with the numbers that came out of building it --
including the several places where a measurement contradicted what I had
written down first.

The [README](../README.md) is the short version.

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
whole thing collapses exactly onto `LogisticRegression`, which is checked in the tests
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
sum to one; measured on 300 rows, the row totals ran from 0.4137 to 1.7790
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
denominators are every row, so reporting all three is reporting one number three
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
model.n_remembered  # 300, the model's size, which no other model here has
```

What that buys is shape. Nothing constrains the decision surface, so a class
sitting in a ring around another one is read straight off the rows. On exactly
that data, [examples/nearest_neighbours.py](../examples/nearest_neighbours.py)
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
likelihood, no one-vs-rest wrapper and no softmax, since a vote does not care how
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
the only one of the six that means anything on categorical codes, and on the
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
linear model extends its line instead, confidently and just as unfoundedly.

## Trees, Which Read Instead Of Weigh

Every model up to this point assembles its answer out of weights, whereas a
tree instead asks a sequence of questions and reads the answer off wherever it
lands.

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

That is the entire model, and we can read it straight off the page. Sleep is
the first gate; study is tested only inside the region where sleep has already
cleared its threshold. That conditional, where studying helps only if you slept, is an
interaction, and a tree expresses it simply by nesting one question inside
another, whereas a linear model needs the product term handed to it as a column
before it can say the same thing.

Units cost nothing here either. A split compares one column against one
threshold, so nothing is ever summed across features and no column can drown
another out by being measured in thousands. Standardising a tree's inputs
changes nothing at all, which is the reverse of the neighbour models above,
where standardising was part of being correct.

### What it pays for that

**Stopping rules are the whole of the regularisation.** Left alone the tree
carves out a box per row:

```
min_samples_split=5   depth 2   leaves 3   training accuracy 0.9333
defaults              depth 4   leaves 5   training accuracy 1.0000
```

The second of those is not the better model, it is the memorised one, and it is
the same failure that `k=1` produces for a neighbour model. Watch `n_leaves`
against the row count.

**The search is greedy, and cannot really be otherwise.** Finding the optimal
tree is NP-hard, so each node takes the best split available to it at the time
and never goes back to reconsider. That is usually fine, although it is
occasionally fatal. Consider a parity target, where the class is 1 when exactly
one of two features is 1. If we split on either feature, both sides come back
half and half, which is the impurity the node already had, so neither question
looks to the search like it helps.

I originally wrote here that the recursion therefore stops at the root, and
then I measured it and found otherwise. On three hundred rows the real features
score a gain of 0.0037 rather than 0.0000, since a finite sample is never
perfectly balanced, and the tree splits on that and goes on to recover the
target completely. What actually sinks it is competition from an irrelevant
column: adding one column of pure noise gives that column a gain of 0.0084, it
takes the root on the strength of nothing at all, and a depth-3 tree then lands
at 0.537 against the 0.5 a coin would manage. The search is not being unlucky;
it is correctly reporting that no single question helps, and a single question
is all it is able to see.

**The boundary is axis-aligned.** A diagonal is only reachable as a staircase.

## Why I Reached For More Than One Model

A decision tree is unstable, which is to say that if we change a handful of
training rows the root split can change, and everything underneath it changes
along with it. On its own that is a defect, although in bulk it turns out to be
the resource the whole family is built on, since models that disagree with each
other can be averaged and the disagreement is what pays for the averaging.

```python
from oop_ml import RandomForestRegressor

forest = RandomForestRegressor(n_members=100, max_features=3, random_seed=0)
forest.fit(features, target)
forest.score(held_out_features, held_out_target)
```

There are two families here and they share very little beyond the word
"ensemble". Consider `B` members, each of variance `s²`, with pairwise
correlation `r` between them:

```
Var(average)  =  r·s²  +  (1 - r)·s²/B
```

Only the second term shrinks as `B` grows, so `r` sets a floor that no number
of members will get underneath.

| | Averaging | Boosting |
|---|---|---|
| Members fitted | independently, on resamples | in sequence, on what is left |
| Attacks | variance | bias |
| Ideal member | deep, unpruned | a stump |
| Order | meaningless | load-bearing |
| More members | stops helping | starts hurting |

Let us dwell on the third row, because it inverts everything the previous
section said about stopping rules. Bagging wants the member with the *most*
variance and the least bias, which is precisely the unpruned tree that
`max_depth` and `min_samples_split` exist to prevent; in a lone tree those
rules are the only defence it has, whereas here the averaging has taken that
job over and the rules mostly get in the way. Boosting wants the opposite,
since nothing in it is averaging a deep member's noise away and the noise is
instead being added up.

### What The Measurements Said

Bagging goes after the second term by adding members. A random forest goes
after `r` itself, by restricting which features each *node* may consider, so
that the members stop all finding the same strong split first. On a fixture
with one deliberately dominant feature:

```
max_features=None   20 of 20 members root on `dominant`
max_features=3      5 or 6 different features appear there
```

So the mechanism does what it claims. The held-out R² on that same fixture,
however, tells a rather different story:

```
one tree              0.6019
bagged                0.7364
forest, 3 of 6        0.7343
forest, 2 of 6        0.7117
forest, 1 of 6        0.5877
```

**The forest does not beat bagging here.** Holding each node to three of the six
features makes every individual tree worse at the same time as it makes the
trees less alike, and on two hundred rows where one predictor carries five
times the coefficient of the others, those two effects came out within 0.002 of
each other. I could have kept adjusting the fixture until the numbers flattered
the library. What the tests assert instead is that both of them beat a single
tree, and the figures above are the ones I actually measured.

The clearest win belongs to bagging on its own. On the parity target from the
previous section, where a lone tree sits at 0.537:

```
lone tree, depth 3           0.52 - 0.59
forest, unrestricted         0.95 - 0.97
forest, 1 feature per node   0.99
```

I had the mechanism wrong here too. My intention was to demonstrate that
restricting the features is what lets a forest escape a bad first move, and
then I measured the *unrestricted* forest and found that resampling alone
already does most of the work, simply by varying which spurious split happens
to win each member's root. Restricting features sharpens the result, though it
is doing less of the work than I expected it to.

### The Held-Out Score That Was Already There

Every member missed about 36.8% of the training set, so the ensemble is already
carrying a held-out set and nobody has to set one aside. For each training row,
find the members whose resample never drew it, average only those, and compare
to the truth.

```python
forest.out_of_bag_score()  # R^2 against rows each member never saw
forest.out_of_bag_evaluate()  # the full evaluation, if you want more than one metric
```

Measured on the same fixture:

```
                       training    out-of-bag    held out
BaggingClassifier        0.9900        0.8350      0.8500
RandomForestRegressor    0.9615        0.6992      0.7545
```

The out-of-bag number sits beside the held-out number and nowhere near the
training one, which is the whole claim. It also lands slightly *below* the
held-out number in both rows, and that is not noise. Each row is judged by
roughly `0.368 * B` members rather than by all `B`, so what we are measuring is
a smaller ensemble than the one we fitted, and a smaller averaging ensemble is
a worse one. The estimate is conservative by construction.

There is a failure mode worth knowing before leaning on it. A row is in-bag for
*every* member with probability `(1 - 1/e)^B`, so at three members:

```
3 members:   152 covered, 48 uncovered, 1.5 judges per row
```

A quarter of the rows had nobody entitled to judge them, against the 25.2% the
formula predicts. At a hundred members the same quantity is around 1e-20 and
the concern evaporates, but `OutOfBagEstimate` reports `n_uncovered` either way
rather than quietly scoring on whatever was left.

### Boosting, Which Fits The Mistakes

Where averaging fits members that never meet each other, boosting fits one
member, looks at what it got wrong, and fits the next one to *that*.

```python
from oop_ml import GradientBoostingRegressor

model = GradientBoostingRegressor(n_rounds=200, learning_rate=0.05, max_depth=3)
```

Everyone learns this as "fit the residuals", meaning `target - prediction`,
which is correct although it is only a special case. The derivative of squared
error with respect to the prediction is `-(target - prediction)`, so for this
loss the residual happens to coincide with the negative gradient; stating it as
the gradient is what allows the same machinery to carry losses whose residual
is not a subtraction, such as log loss, where it comes out as
`target - probability`. Each round is therefore one step of gradient descent
taken in the space of functions rather than the space of parameters, with the
member supplying the direction and the learning rate supplying the step size.

That is also why the rate and the round count have to be chosen together rather
than separately. Both of the following travel a nominal distance of 5.0:

```
100 rounds at 0.05    better on rows it never saw
  5 rounds at 1.00    worse
```

Committing less on each step leaves less room to commit to noise, which is the
same shrinkage a ridge penalty buys, arrived at from a rather different
direction.

It is also considerably cheaper than we might expect, because a boosted member
is shallow by design:

```
40 boosted rounds     0.055s
20 bagged members     0.522s
```

That is nine times less work for twice as many trees, which is the
bias/variance reversal from the table showing up as wall-clock time, since an
unpruned tree on two hundred rows keeps recursing until every leaf is pure.

### Averaging Probabilities Rather Than Counting Votes

A bagged classifier here averages its members' probability matrices instead of
counting how they voted. Consider six members that put a class at 0.51 against
four that put the other class at 0.99: the six win the vote, six to four,
though they lose the average 0.30 to 0.70, and the averaged answer is the one
that has used how confident each member actually was.

There is a second benefit that I did not anticipate. A lone unpruned tree
reports a probability of 1.0 from every leaf it grows, since every leaf it
grows is pure, so its probabilities are useless however accurate its
predictions are. Averaging a hundred such certainties across members that
disagree with each other produces a graded number, and it is the first
probability the trees in this library have produced that I would be willing to
act on.
