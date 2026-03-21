# On scikit-learn

What this library does differently, and where you should reach for the
established one instead.

The [README](../README.md) is the short version.

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
`(n_queries, n_remembered, n_features)` array to do it, at 1.6 GB, for an answer
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
first removes them, since it shifts every point equally and leaves all distances
unchanged. On that same pathological pair the centred form is exact to the last
bit, and across random data the largest relative error against the definition
was 3.5e-16.

Those two findings left the row at 15.7x, and I wrote at that point that what
remained was a fused-kernel gap numpy could not close. That was half right, and
the half I had wrong was the more useful half. Timing the two halves separately
showed the work splitting almost evenly, at 0.064s building the distance matrix
against 0.059s selecting from it, and only the first half was parallel, because
BLAS threads the matrix multiply on its own while `argpartition` runs on one
core. Roughly half the calculation was using one core of twenty-four.

Queries are independent of each other, so the fix is to hand blocks of them to
threads. Threads rather than processes, because numpy releases the interpreter
lock for exactly these operations, which makes it real parallelism with no
pickling and no copies, since every worker reads the same remembered rows. That is
another 2.5x to 3.4x, and it takes the largest row from 15.7x to 4.4x.

It needs a floor. Starting a pool costs a millisecond or two, which is nothing
against a hundred and ruinous against one: at 500 queries by 500 remembered rows
the threaded route measured nine times *slower*, so below half a million
(query, remembered) pairs it stays on one thread. The threshold counts pairs
rather than either dimension alone, because the work is the product of the two.

What is left now really is a fused-kernel gap. scikit-learn never materialises
the distance matrix at all; it keeps a per-query heap of the k smallest while
streaming the distances past it, in Cython, across threads. numpy has no way to
express that, since every intermediate is a real array, so 160 MB gets written and
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
