"""The scikit-learn backend: the same encapsulations, a production engine underneath.

Every class exported here has a namesake in :mod:`oop_ml.numpy` with the same
constructor, the same ``fit`` and ``predict``, and the same learned-parameter
properties, so the two interchange at the call site. What differs is only what
does the arithmetic. The numpy backend implements every model by hand, which is
what makes it worth reading; this one wraps scikit-learn, which is what makes it
worth running.

Absence is declared, never silent
----------------------------------
Not every model exists in every backend. scikit-learn has no Hopfield network
and no self-organising map, and for three more of this library's models its
nearest estimator either answers a different question or would leave configured
fields read by nothing. Rather than let such a model be quietly missing here,
every numpy export that this backend does not provide is listed in
:data:`NOT_PROVIDED` with the reason, and a contract test asserts that the two
lists together cover every model exactly once. A model can be present or it can
be declined; it cannot be forgotten.

Twenty-eight models are wrapped here and five are declined. Each family was
built behind the same contract suite the numpy backend already passes, and each
left :data:`NOT_PROVIDED` honest as it landed. The five that remain are
permanent absences rather than work still to do, so each reason names the
mechanism that breaks and the measurement behind it. Naming a missing class
would be the weaker refusal, since the engine ships a near miss for every one of
the five, and a reader told only that a name is absent will reach for it.

What does not interchange
--------------------------
Saving. ``save_model`` and ``load_model`` hold a closed registry of the numpy
backend's models keyed by bare class name, so a model fitted here is refused
with ``Standardizer is not a registered persistable type`` where its numpy
namesake saves. The refusal is right and only its wording is short of the fact,
since a ``Standardizer`` plainly is registered and it is the other one. It is
recorded here because the promise at the top of this module covers the
constructor, ``fit``, ``predict`` and the learned properties, and a reader has
nowhere else to learn that it stops there.
"""

from __future__ import annotations

from oop_ml.scikit.classification import (
    BaggingClassifier,
    DecisionTreeClassifier,
    KNearestNeighboursClassifier,
    LogisticRegression,
    MultinomialLogisticRegression,
    NewtonLogisticRegression,
    OneVsRestClassifier,
    RandomForestClassifier,
    SupportVectorClassifier,
)
from oop_ml.scikit.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PolynomialFeatures,
    RobustScaler,
    Standardizer,
)
from oop_ml.scikit.regression import (
    BaggingRegressor,
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    KernelRidgeRegression,
    KNearestNeighboursRegressor,
    LassoRegression,
    MultipleLinearRegression,
    RandomForestRegressor,
    RidgeRegression,
    SimpleLinearRegression,
)
from oop_ml.scikit.unsupervised import (
    KernelPrincipalComponentAnalysis,
    KMeans,
    PrincipalComponentAnalysis,
    RestrictedBoltzmannMachine,
)

#: Every model the numpy backend exports that this one deliberately does not,
#: with the reason. Read by the contract suite, which refuses a model that is
#: neither provided here nor declined here.
NOT_PROVIDED: dict[str, str] = {
    "GradientDescentRegression": (
        "scikit-learn has two estimators that step towards a least-squares fit "
        "at a step size a caller names, and neither is a batch gradient descent "
        "that keeps this model's fields. SGDRegressor updates after every row "
        "rather than once per epoch, so scaled such that an epoch of it sums to "
        "one batch step it reaches the same minimum by a different path, missing "
        "the numpy docstring's own worked example by 1.9e-03 at a thousand "
        "epochs, and the rate at which it diverges sits between 1.2 and 25 times "
        "the closed-form threshold that docstring gives. MLPRegressor with no "
        "hidden layer and one batch per epoch does take the exact step, "
        "agreeing to 8.9e-16 over a thousand of them, though only at twice the "
        "rate, since the engine's squared loss carries a half and this model's "
        "gradient does not. It also takes only one step per call, so a wrapper "
        "around it would own the start, the loop, the counter, the convergence "
        "test, the rate translation and the refusal while writing weights into "
        "the engine's fitted attributes before every epoch. Measured over two "
        "hundred epochs, that costs 46x the numpy backend's own time on "
        "twenty-five rows of two columns and 5x on five thousand rows of "
        "twenty, since the engine's per-call bookkeeping is what dominates a "
        "small step. The closed-form answer to the same objective is "
        "MultipleLinearRegression"
    ),
    "SelfOrganisingMap": (
        "scikit-learn has no self-organising map, and nothing else it ships can "
        "stand in for one. The prototypes are not what is missing, since KMeans "
        "finds those; the neighbourhood is. Here a unit moves because a grid "
        "neighbour won, so the grid decides what the prototypes become, while "
        "every scikit-learn clusterer finds its centres without a grid and "
        "leaves an arrangement that can only be imposed afterwards. Imposed "
        "afterwards it survives setting neighbourhood_radius to zero, and a "
        "1 x 8 and a 2 x 4 fit of the same rows come back bit-identical where "
        "the numpy model puts 0.80 to 0.92 between them over four seeds. "
        "learning_rate, neighbourhood_radius, final_epoch_movement and "
        "converged would each have nothing to read them"
    ),
    "HopfieldNetwork": (
        "scikit-learn has no associative memory. Swept across all 208 "
        "estimators, BernoulliRBM.gibbs is the only method that takes a state "
        "and answers with a state, and it belongs to a different model. An RBM "
        "is restricted, meaning it has no visible to visible weight, so there is "
        "no symmetric matrix over the units to hand back and no diagonal to "
        "zero, and marginalising its Bernoulli hidden units leaves a sum of "
        "softplus terms rather than a quadratic form. That is what breaks the "
        "theorem this model states about itself, that a state and its negation "
        "sit at exactly the same energy; measured on the numpy spec's three "
        "orthogonal patterns a fitted engine puts each pattern and its negation "
        "32.4, 183.6 and 52.6 apart and holds none of the three negations as a "
        "fixed point. It also learns by persistent contrastive divergence where "
        "this stores in one shot, recovering none of the 48 one-flip probes at "
        "100 epochs and a rate of 0.01, at most 0.625 of them at 100 epochs and "
        "a rate of 0.5, and all 48 only at 2000 epochs, an epoch count and a "
        "learning rate this model has no field for. EmpiricalCovariance does "
        "reproduce the Hebbian matrix to the last bit, 29x slower on that "
        "fixture than the outer product it replaces, and it leaves both update "
        "rules, the energy, the fixed point test and the recall walk here"
    ),
    "HebbianPrincipalComponents": (
        "scikit-learn has no local learning rule. Every route to the principal "
        "directions there is a matrix factorisation rather than a walk, so none "
        "of them reads one row at a time under a decaying rate. IncrementalPCA "
        "runs an exact SVD per batch and takes no rate, no epoch count and no "
        "seed at all, MiniBatchSparsePCA optimises an L1 objective whose "
        "components stop being the principal ones as soon as alpha leaves zero, "
        "and PCA under its randomized solver is a batch solver whose seed moves "
        "its directions by 2.2e-16 over eight of them, which is rounding rather "
        "than a walk. Wrapping any of them leaves learning_rate, max_epochs, "
        "tolerance and random_seed accepted at construction and read nowhere, "
        "which is the fault TreeModel.classification_criterion was deleted for. "
        "The two diagnostics this model exists to report go the same way. "
        "Measured on the numpy spec's tilted-grid fixture, a factorisation "
        "hands back directions of length exactly 1 and an orthogonality of "
        "exactly zero where the finite walk leaves 1.0003 and 2.4e-04. For a "
        "decomposition too wide to form its covariance matrix, "
        "reach for IncrementalPCA directly; the production answer to the same "
        "question is PrincipalComponentAnalysis"
    ),
    "RootMeanSquareScaler": (
        "scikit-learn has no scaler that measures spread about zero. "
        "StandardScaler(with_mean=False) is the near miss and it is not one, "
        "because the engine computes the mean whatever that switch says, its "
        "own source saying the incremental variance needs one, so scale_ stays "
        "the standard deviation about the mean and only the subtraction is "
        "skipped. Measured on the column [100, 101, 102, 103, 104] it answers "
        "1.4142 where the root mean square is 102.0098, and over 300 random "
        "offset columns it agreed with the numpy backend on none of them. "
        "Normalizer divides by a row's norm and not a column's, MaxAbsScaler by "
        "the largest magnitude, RobustScaler by the interquartile range, and the "
        "only root mean square in the engine is a regression metric taken "
        "between two vectors. The number is reachable exactly by reflecting the "
        "training rows through zero and reading the engine's variance, at 50x "
        "the cost of computing it directly on twenty-five rows of one column "
        "and 7x on twenty thousand rows of twenty, since the reflected block is "
        "twice the size and the engine's own bookkeeping is what dominates a "
        "small one, but the engine is then fitted to a dataset the wrapper "
        "invented, which is arithmetic wearing an engine's coat rather than a "
        "wrap"
    ),
}

__all__: list[str] = [
    "BaggingClassifier",
    "BaggingRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "GradientBoostingRegressor",
    "KNearestNeighboursClassifier",
    "KMeans",
    "KNearestNeighboursRegressor",
    "KernelPrincipalComponentAnalysis",
    "KernelRidgeRegression",
    "LassoRegression",
    "LogisticRegression",
    "MaxAbsScaler",
    "MinMaxScaler",
    "MultinomialLogisticRegression",
    "MultipleLinearRegression",
    "NewtonLogisticRegression",
    "OneVsRestClassifier",
    "PolynomialFeatures",
    "PrincipalComponentAnalysis",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "RestrictedBoltzmannMachine",
    "RidgeRegression",
    "RobustScaler",
    "SimpleLinearRegression",
    "Standardizer",
    "SupportVectorClassifier",
]
