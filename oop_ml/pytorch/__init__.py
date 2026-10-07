"""The torch backend: the same network vocabulary, autograd underneath.

What is here, and what it is for
---------------------------------
:class:`LayerStack`, with the numpy stack's constructor and interface, built
on this library's own layers and run by torch. Its reason to exist is the
check the numpy stack already makes and torch does not: a chain of layers that
cannot work is refused at construction, in integer comparisons naming the
seam, before a tensor has been allocated. torch answers the same question at
the first forward pass, from inside a matrix multiply. Knowing that a network
is wrong before training it, rather than during, is the whole of what this
package adds, and :mod:`oop_ml.pytorch.stack` says how.

Around it, the translation that makes it possible. Every layer this library
ships is either turned into the torch module that does what it describes,
weights carried across so the two agree to the last bits, or refused by name
with the reason in :data:`~oop_ml.pytorch.layers.NOT_TRANSLATED`. Every loss
is turned into a criterion that measures the same number. Both are the third
instance of a shape this library already has twice, in the kernel and metric
translations of the scikit backend.

What is not here yet, and why that is declared rather than silent
------------------------------------------------------------------
Every model the numpy backend exports is declined in :data:`NOT_PROVIDED`
with a reason, so the contract suite that holds every backend to the
reference runs against this one and skips with a sentence a person wrote,
exactly as the scikit backend began. The reasons are of two kinds. Sixteen
models have an honest torch twin, since their fit is a differentiable
objective over a parameter block or a factorisation ``torch.linalg``
performs, and say they are not yet wrapped. Twenty-eight have none, because
what torch contributes is autograd, modules and an optimiser, and a tree, a
neighbour vote, a scaler's single reduction, a clustering loop or a local
learning rule uses none of those; a wrapper would be the numpy backend's own
arithmetic with a tensor substituted for an array, which is the thing the
scikit backend's own declines name as the reason not to.

The next piece of work is the estimator that wraps a stack, in both backends
at once.

Precision and placement
------------------------
Everything here is ``float64`` on the CPU. The numpy backend is ``float64``,
and the agreement between the two is asserted to the last bits rather than to
a tolerance, which single precision cannot support. A caller who wants either
the speed of ``float32`` or a device has the module itself.
"""

from __future__ import annotations

from oop_ml.pytorch.engine_version import check_engine_available

check_engine_available()

from oop_ml.core.network.stack import StackResponse
from oop_ml.pytorch.layers import (
    NOT_TRANSLATED,
    Translation,
    torch_activation_for,
    torch_module_for,
    translation_of,
)
from oop_ml.pytorch.losses import torch_loss_for
from oop_ml.pytorch.stack import LayerStack

_NOT_YET_WRAPPED = (
    "has an honest torch twin, since its fit is a differentiable objective over "
    "a parameter block or a factorisation torch.linalg performs, and is not yet "
    "wrapped. This backend's first increment is the network vocabulary, which "
    "is what the estimator that would wrap this model needs to exist first. "
    "Fit with the numpy namesake meanwhile"
)

_NO_TREES = (
    "torch has no decision tree and no estimator object for one. A tree is "
    "grown by a discrete split search over sorted columns, which is not a "
    "differentiable objective, so there is nothing for autograd to descend and "
    "nothing in torch.nn to wrap; the whole fit would be the numpy backend's "
    "own recursion with a tensor substituted for an array"
)

_NO_NEIGHBOURS = (
    "torch has no neighbour estimator. torch.cdist and topk compute the "
    "distances and the selection, but that is the arithmetic the numpy backend "
    "already does with numpy, and a wrapper would own the memory of the "
    "training rows, the vote or the mean, and every refusal, with no engine "
    "object underneath, which is the reason the scikit backend declines "
    "RootMeanSquareScaler"
)

_NO_PREPROCESSING = (
    "torch has no fit-and-transform preprocessing object. The statistic is one "
    "tensor reduction and the transform is one broadcast, so a wrapper would be "
    "the numpy backend's arithmetic spelled in tensors with nothing of torch's, "
    "no module, no autograd and no optimiser, doing any of the work"
)

_NO_GENERATIVE_CLASSIFIER = (
    "torch has no generative classifier. The fit is closed-form sums per class "
    "and the score is a quadratic form, which torch.linalg can evaluate but "
    "which nothing in torch models; the wrapper would be the numpy backend's "
    "arithmetic in a tensor coat"
)

_NO_CLUSTERER = (
    "torch has no clusterer. The fit is an assignment loop or a merge loop in "
    "which no gradient is taken and no module exists, so autograd and "
    "torch.nn have nothing to contribute and a wrapper would own the whole "
    "algorithm itself"
)

_NO_LOCAL_RULE = (
    "learns by a local rule rather than by a gradient, which is the one thing "
    "autograd does not do. The update reads only the two ends of a weight, "
    "nothing is propagated from a loss, and a wrapper would spell the numpy "
    "backend's rule in tensors with torch contributing the array type and "
    "nothing else"
)

#: Every model the numpy backend exports that this one does not yet provide,
#: with the reason. Read by the contract suite, which refuses a model that is
#: neither provided here nor declined here.
NOT_PROVIDED: dict[str, str] = {
    # Sixteen with a twin, awaiting the estimator that would wrap them.
    "SimpleLinearRegression": _NOT_YET_WRAPPED,
    "MultipleLinearRegression": _NOT_YET_WRAPPED,
    "RidgeRegression": _NOT_YET_WRAPPED,
    "LassoRegression": _NOT_YET_WRAPPED,
    "ElasticNetRegression": _NOT_YET_WRAPPED,
    "GradientDescentRegression": _NOT_YET_WRAPPED,
    "HuberRegression": _NOT_YET_WRAPPED,
    "LogisticRegression": _NOT_YET_WRAPPED,
    "NewtonLogisticRegression": _NOT_YET_WRAPPED,
    "MultinomialLogisticRegression": _NOT_YET_WRAPPED,
    "OneVsRestClassifier": _NOT_YET_WRAPPED,
    "SupportVectorClassifier": _NOT_YET_WRAPPED,
    "RestrictedBoltzmannMachine": _NOT_YET_WRAPPED,
    "PrincipalComponentAnalysis": _NOT_YET_WRAPPED,
    "KernelPrincipalComponentAnalysis": _NOT_YET_WRAPPED,
    "KernelRidgeRegression": _NOT_YET_WRAPPED,
    # Twenty-eight with none.
    "DecisionTreeRegressor": _NO_TREES,
    "DecisionTreeClassifier": _NO_TREES,
    "RandomForestRegressor": _NO_TREES,
    "RandomForestClassifier": _NO_TREES,
    "BaggingRegressor": _NO_TREES,
    "BaggingClassifier": _NO_TREES,
    "GradientBoostingRegressor": _NO_TREES,
    "AdaBoostClassifier": _NO_TREES,
    "KNearestNeighboursRegressor": _NO_NEIGHBOURS,
    "KNearestNeighboursClassifier": _NO_NEIGHBOURS,
    "Standardizer": _NO_PREPROCESSING,
    "MinMaxScaler": _NO_PREPROCESSING,
    "MaxAbsScaler": _NO_PREPROCESSING,
    "RobustScaler": _NO_PREPROCESSING,
    "RootMeanSquareScaler": _NO_PREPROCESSING,
    "MeanCentrer": _NO_PREPROCESSING,
    "PolynomialFeatures": _NO_PREPROCESSING,
    "GaussianNaiveBayes": _NO_GENERATIVE_CLASSIFIER,
    "LinearDiscriminantAnalysis": _NO_GENERATIVE_CLASSIFIER,
    "QuadraticDiscriminantAnalysis": _NO_GENERATIVE_CLASSIFIER,
    "KMeans": _NO_CLUSTERER,
    "DBSCAN": _NO_CLUSTERER,
    "AgglomerativeClustering": _NO_CLUSTERER,
    "GaussianMixture": _NO_CLUSTERER,
    "SelfOrganisingMap": _NO_LOCAL_RULE,
    "HopfieldNetwork": _NO_LOCAL_RULE,
    "HebbianPrincipalComponents": _NO_LOCAL_RULE,
    "MarkovChain": (
        "torch has no model of a sequence of states. Counting transitions is "
        "the whole fit and a matrix power is the whole prediction, neither of "
        "which takes a gradient or lives in a module, so a wrapper would be the "
        "numpy backend's arithmetic with no engine underneath it at all"
    ),
}

__all__: list[str] = [
    "NOT_PROVIDED",
    "NOT_TRANSLATED",
    "LayerStack",
    "StackResponse",
    "Translation",
    "torch_activation_for",
    "torch_loss_for",
    "torch_module_for",
    "translation_of",
]
