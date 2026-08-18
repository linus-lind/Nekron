# `nekron.cv` — model-agnostic cross-validation

A sweep is a schedule, a model fitted once per fold, and one pooled number at the
end. None of those three steps needs to know what the model is, and all three are
easy to get subtly wrong in a fresh copy, so they are written once here and shared.

Two models use it today — the conditional autoencoder and the LSTM-Transformer —
and they agree on almost nothing:

| | conditional autoencoder | LSTM-Transformer |
|---|---|---|
| a "period" | one date's cross-section | one window end-date × entity |
| a fold's score | `RSquared(total, predictive)` | a scalar loss |
| pooling | `1 − Σsse / Σsst` | `Σ(loss·n) / Σn` |
| selection | maximise R² | minimise loss |
| where its `SplitConfig` lives | `data_pipeline.split` | `data.split`, or inherited from another model's run |

That is the whole design constraint: **the driver never constructs a model, never
names a metric and never interprets a column.**

---

## What a model supplies

Two callables and, optionally, some prose.

```python
from nekron.cv import FoldOutcome, SweepReport, plan_schedule, run_sweep, sweep_tags

def body(bounds, window, run, num_folds) -> FoldOutcome:
    """Fit and score one fold, inside the nested run already opened for it."""
    ...
    return FoldOutcome(
        scores={"test_loss": ...},          # -> the summary table's columns
        metrics={"test/loss": ...},         # -> MLflow, which is a different namespace
        rows=per_period_frame,              # -> concatenated across folds, then pooled
        best_epoch=fit.best_epoch,
        best_metric=fit.best_metric,        # in whatever direction the model selects on
    )

def pool(rows) -> Mapping[str, float]:
    """The headline scalars, over every scored row of every fold."""
    ...

result = run_sweep(
    bounds, dates, body,
    policy=cfg.cv, tracker=tracker, seed=cfg.train.seed,
    pool=pool, spread_columns=("test_loss",),
    report=SweepReport(headline="test loss", spread_column="test_loss", spread_label="loss"),
    tags=sweep_tags(split, len(bounds)),
)
```

`body` is normally a closure over the model's own panel and configuration.

### Why a callback and not a base class

* The two models share **no state** worth inheriting — their panel, config and
  splits types are entirely different, so a base class ends up generic in three
  parameters or typed `Any`, and under `mypy --strict` the first is unreadable and
  the second is a lie.
* `dataclasses.replace` is typed against a concrete dataclass, so the per-fold
  `replace(cfg, train=replace(cfg.train, ...))` **cannot** live in a
  config-agnostic base class. That is a hard blocker, not a preference — which is
  why `fold_seed` and `fold_checkpoint_dir` are shared while the two-line
  `replace` stays in each model.
* A protected method is an invitation to override exactly the invariants this
  extraction exists to freeze.
* It matches the rest of the codebase: `ScoreObjective`, the featurizer,
  preprocessor and aligner registries are all this shape.

---

## What the driver owns

`run_sweep` owns the parts that are the same everywhere and go wrong quietly:

**Seeding order.** Each fold is reseeded *before* `body` runs. Weight
initialization draws from the global RNG and a trainer reseeds only once its model
exists, so a fold that relied on that would be initialized from whatever state the
fold before it left behind. Under `seed_mode="fixed"` every fold therefore starts
from identical weights, and the spread across folds is a property of the *data*.

**Pooled, never averaged.** Each fold's rows are kept and concatenated, and the
headline number is computed from the concatenation. Folds differ in how many rows
they carry and in how much variation those rows hold, so the mean of their scores
weights a thin fold like a thick one — and a ratio-of-sums statistic cannot be
recovered from a set of per-fold ratios at all. This is why `FoldOutcome.rows`
carries *sums* rather than a ratio. The per-fold spread is reported alongside, as
a second and different summary.

**Nested runs.** Each fold is opened with `tracker.child(...)` while the parent is
inside its own `with`. MLflow's fluent API targets whichever run is innermost, so
the parent's aggregate is written only after every child has exited — a
"log running aggregates as we go" convenience would silently write them onto the
fold's run instead.

**Failure policy.** `on_fold_error="skip"` records the failure, leaves the fold's
run marked FAILED and *without* the `fold_status` tag (which is how a later reader
tells completed folds from failed ones), and carries on. `"raise"` propagates the
original exception unwrapped. A sweep in which *every* fold failed still raises.

**The summary table**, the per-fold spread statistics, and the artifact names
(`folds.parquet`, `test_periods.parquet`, `columns.json`).

---

## `plan_schedule`

Every model discards some of the panel's dates before it can fit anything — the
autoencoder drops a period whose cross-section is too small to price, a sequence
model drops one that ends no complete window. So the sequence a fold indexes is
**shorter** than the panel's date index, and both halves of the fix matter:

* folds are cut over the **surviving** sequence, or every boundary after the first
  discarded date is off by the number discarded;
* the split is **rebased against the panel first**, so a fraction-based single
  split lands on the same calendar date whichever sequence it is then applied to.

Half-implementing either produces a boundary that is wrong by a handful of dates,
which no test of the model itself would notice.

---

## Layout

```
cv/
├── config.py     # CrossValidationConfig — free of torch, pandas and MLflow
├── schedule.py   # plan_schedule: cut folds over the periods a model can use
└── sweep.py      # run_sweep + the FoldBody/Pooler seam + the records
```

`config.py` is deliberately dependency-light so that composing or validating a
configuration does not pay for a multi-second torch import; the driver in
`sweep.py` needs torch, pandas and MLflow and does not pretend otherwise.

## Related shared code

* `nekron.nn.config` — `OptimConfig` and `TrainConfig`, embedded by each model's
  root config. A model needing an extra field subclasses rather than redeclares;
  anything that names a *statistic* (which metric selects a model, and in which
  direction) stays in the model package, because that is a property of what the
  model computes.
* `nekron.nn.training` — `EarlyStopping`, `FitResult`, `build_optimization`,
  `should_log_epoch`, `save_checkpoint`: the four operations every fit does the
  same way, each with the one detail that is easy to lose in a fresh copy.
* `nekron.data_adapter.splitting` — `FoldBounds`, `FoldWindow`, `plan_folds`,
  `rebase_split`. Pure position arithmetic; reads no data, imports no torch.
