# Examples

Seven runnable scripts, simplest first. Each one is written the way a user of
the installed package writes code — everything comes from the top-level
`oop_ml` import, and no example reaches into the library's internal module
paths. Reading one tells you what your own code should look like.

Each carries its reasoning in the module docstring: the reported numbers are the
point, and the prose says what to notice in them.

## Install and run

```bash
git clone https://github.com/aemonalgiz-dev/oop_ml.git
cd oop_ml
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Then, from the repository root:

```bash
python -m examples.simple_regression
```

The scripts import `oop_ml` as an installed package, so once the editable
install is in place they behave exactly as your own project would. What they
additionally import from `examples.datasets` and `examples.reporting` are the
two local helpers described below, which is why they run from the repository
root rather than from anywhere.

## Reading order

| # | Script | What it shows |
|---|--------|---------------|
| 1 | `simple_regression` | The three-call shape: construct, `fit`, ask. Why `evaluate` returns an object rather than a number. |
| 2 | `multiple_regression` | Features carry names, so coefficients are addressed by name and `predict` cannot be fooled by column order. |
| 3 | `gradient_descent` | The same objective as the closed form, walked to instead of jumped to — and the two hyperparameters that arrive with it. |
| 4 | `regularization` | Ridge shrinks, lasso selects. Why that difference falls out of the shape of the penalty. |
| 5 | `polynomial_curves` | Fitting a curve without changing the model at all, and what rising degree costs on held-out data. |
| 6 | `standardization` | Why a penalty and a learning rate care about units when least squares does not — and where leakage gets in. |
| 7 | `model_selection` | The capstone: hold out, cross-validate, choose, refit, report once. |

## Supporting modules

- `datasets.py` — synthetic data, each generator returning a `SyntheticRegression`
  that pairs the data with the coefficients that produced it. The truth is
  carried as `Coefficients`, the same type the models learn, so every example
  can report the estimate beside the answer.
- `reporting.py` — the `Report` object every script writes through, plus the
  logging setup.

## Output and log levels

Nothing here calls `print`. Each script writes through a `Report` bound to its
own module logger, so output can be filtered, redirected, or silenced per
example — and so a result is distinguishable from a complaint. The levels mean
something specific:

| Level | Carries |
|-------|---------|
| `INFO` | The report itself — headings, tables, and the prose saying what to notice. The default. |
| `WARNING` | The modelling telling you something: a fit that did not converge, a held-out R² gone negative, a deliberately leaky pipeline. |
| `ERROR` | An exception raised and caught on purpose, to show a guard firing. |
| `DEBUG` | Mechanical detail — shapes, seeds, per-fold scores. Off by default. |

Warnings and errors are prefixed (`[warning] …`) so they break out of the report
text; `INFO` lines are emitted exactly as written, which is what keeps the
tables aligned.

Add `--verbose` for the `DEBUG` detail:

```bash
python -m examples.model_selection --verbose
```

That turns the cross-validation summary into per-fold scores, which is where you
can see the disagreement the spread column is summarising.

Configuration happens only in each script's `__main__` block, never on import —
so `test/test_examples.py` can call every `main()` without seven modules
fighting over the root logger.

## Why synthetic data

Every dataset here is generated, and that is deliberate. With real data the true
coefficients are unknown, so a fitted model can only be compared against other
fitted models and each example becomes a plausibility argument. Generating the
data means the answer is available to report in the next column.

The trade-off is honest to state: synthetic data is *well behaved*. There are no
missing values, no measurement error in the predictors, no drift between
training and deployment. The examples show the mathematics working, not the data
cleaning that dominates real work.

## A note on the numbers

Every seed is fixed, so these scripts print the same values on every run and
across machines. Two of the tables are worth pausing on:

- `polynomial_curves` — train R² rises monotonically to 0.98 while test R² falls
  to **−133**. A model that fits its training data almost perfectly and is worse
  than useless on new data, in six lines of output.
- `model_selection` — held-out R² of 0.70 with a penalty against 0.61 without
  one, on data chosen so that a penalty genuinely helps. Alongside an in-sample
  score of 0.91, which is what the same model claims about itself.

The examples are covered by `test/test_examples.py`, which runs each one and
fails if it raises — so they cannot quietly rot as the library changes.
