"""The vocabulary a network is built from, separate from any model that trains one.

:mod:`~oop_ml.core.network.activation` is the bend between layers, and the only
reason stacking them buys anything. :mod:`~oop_ml.core.network.neuron` is the
unit itself, which is the logistic model this library already had, plus the
observation that one of them can only ever draw one hyperplane.
:mod:`~oop_ml.core.network.layer` is many of those neurons reading one shared
row, and :mod:`~oop_ml.core.network.shape` is the pair of widths that makes
joining two layers a question with an answer.

This sits in ``core`` rather than beside the models for the reason
:mod:`~oop_ml.core.tree` gives: none of it is a model. A neuron is a fact about
a weight vector, and an activation is a fact about a number, and the regressor
and the classifier that will eventually stack them both want the same pieces
while being different questions. Following the library's layout rule, those two
models belong under ``regression`` and ``classification`` respectively, where a
reader looks for the task rather than the family.

Shapes are facts here, not checks
---------------------------------
The point of putting the weights inside the neuron is that its input width
stops being a number someone declares and starts being a property of the
object: a neuron reading ``n`` inputs has ``n`` weights, because the dot
product is undefined otherwise. A layer of ``m`` such neurons answers with
``m`` numbers whatever ``n`` was.

Stack them and the only constraint in the whole network is a chain of integer
equalities, each layer's output count matching the next layer's input count.
Every one of those numbers is known at construction. The data supplies exactly
two facts, the width at the front and the width at the back, and neither takes
part in any interior agreement.

So whether a network fits together is decidable before a single row arrives,
which is what makes a shape error something to refuse at construction rather
than discover part-way through a training run.
"""
