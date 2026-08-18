# Conditional Autoencoder — asset-pricing model

A PyTorch + MLflow implementation of

> Shihao Gu, Bryan Kelly, Dacheng Xiu. *Autoencoder Asset Pricing Models.*
> Journal of Econometrics, 2021 (SSRN 3335536).

A conditional latent-factor model

```
r_{i,t} = beta(z_{i,t-1})' f_t + u_{i,t}
```

whose factor **loadings** are a neural-network function of per-stock input columns
(firm characteristics in the paper; the *beta network*) and whose latent **factors**
are a neural-network function of the characteristic-managed portfolios
`x_t = (Z_{t-1}' Z_{t-1})^{-1} Z_{t-1}' r_t` (the *factor network*). Their inner
product reconstructs the cross-section of returns — an autoencoder in which the
encoder is informed by the input columns and the decoder is the loadings.

---

## Architecture → paper mapping

| Component | File | Paper |
|-----------|------|-------|
| Beta (factor-loading) network `z -> beta`, per stock | `modules/beta_network.py` | §2.2 |
| Factor network `x -> f`, per period | `modules/factor_network.py` | §2.2 |
| Configurable MLP block (linear / batch-norm / activation / dropout) | `nekron.nn.MLP` (shared) | §2.2 |
| Managed portfolios `x = (Z^T Z)^{-1} Z^T r` + reconstruction `beta . f` | `model.py` | §2.2, eq. (10) |
| Mean squared pricing error + L1 (LASSO) penalty | `losses.py` | §2.3 |
| Total and predictive R² | `metrics.py` | §2.4 |
| Factor forecast `λ_{t-1}` (expanding mean / EWMA) | `metrics.py` | §2.4 |
| Adam trainer, early stopping, checkpointing | `engine/trainer.py` + `nekron.nn.training` (shared) | §2.3 |
| Walk-forward sweep: schedule, nested runs, pooling | `nekron.cv` (shared) + `engine/cv.py` | — |
| Walk-forward refitting, one model per fold | `engine/cv.py` | §3.1 |
| Characteristic importance by zeroing a characteristic | `analysis.py` | §4.3 |
| Cross-sectional standardization + per-period cross-sections | `data/dataset.py` | §2.1 |
| Ingestion → preprocessing → features → temporal split | `nekron.data_adapter` (shared) | §2.1 |

Every hyperparameter lives in the Hydra config
(`configs/conditional_autoencoder.yaml`), typed by the dataclasses in `config.py`;
nothing that governs model behavior is hard-coded inside the modules. Model-agnostic
MLflow settings are composed from the shared `configs/mlflow/default.yaml` group.

```
conditional_autoencoder/
├── config.py            # Hydra structured-config schema (dataclasses)
├── model.py             # ConditionalAutoencoder nn.Module + CaeOutput
├── losses.py            # MSE pricing error + L1 penalty
├── metrics.py           # total & predictive R^2, the factor forecast, factor diagnostics
├── analysis.py          # reload a finished sweep; characteristic importance
├── data/                # feature panel -> per-period cross-sections
├── modules/             # beta network, factor network (MLP from nekron.nn)
└── engine/              # trainer, walk-forward runner, MLflow tracking
```

Shared, model-agnostic building blocks live outside this package and are used by
every model here:

* **`nekron.nn`** — `set_seed`, `resolve_device`, `build_activation`,
  `count_parameters`, the configurable `MLP`, and the training-loop pieces every
  fit repeats: `EarlyStopping`, `FitResult`, `build_optimization`,
  `should_log_epoch`, `save_checkpoint`, plus the shared `OptimConfig` /
  `TrainConfig` (this package's `TrainConfig` subclasses the shared one to add
  `selection_metric`, which names a statistic only this model computes).
* **`nekron.cv`** — the cross-validation driver. `engine/cv.py` now holds only
  what the driver cannot know: how this model's periods are built, how one fold is
  fitted and scored, and what its numbers are called. See `nekron/cv/README.md`.

---

## Data

Data is provided by the shared **`nekron.data_adapter`**, which runs the
config-driven pipeline **ingestion → preprocessing → feature creation** and splits
the resulting `(date, entity)` feature panel into train/val/test by date. The model
composes the three stage configs as Hydra groups under `data_pipeline` and sets the
split in its own config:

```yaml
defaults:
  - base_conditional_autoencoder
  - mlflow: default
  - data@data_pipeline.ingestion: crsp          # nekron.data
  - preprocessing@data_pipeline.preprocessing: crsp
  - features@data_pipeline.features: crsp        # nekron.featurers
  - _self_

data_pipeline:
  split: { train_end: null, val_end: null, date_level: date }  # null -> 60/80% quantiles
```

From the produced feature panel the model needs only two settings — every other
column is used automatically:

* **`data.return_column`** — the target, a forward return produced by feature
  creation (e.g. `fwd_ret_1m`).
* **`data.portfolio_columns`** — the (typically smaller) subset of characteristics
  used to form the managed portfolios; empty defaults to all beta columns.

The **beta network uses every other feature column** (size `P_beta`, inferred — no
list to maintain); make sure the feature config emits no forward-looking column
other than the target, or it would leak into the predictors. Because every column
of the panel becomes a predictor, the feature config is also where the input set is
pruned: `keep_inputs` chooses which raw panel columns (if any) survive alongside
the features, while `keep_features` — and the per-spec `intermediate` marking —
drops the scaffolding columns that exist only to feed a later featurizer, so they
are computed but never reach `Z_beta`. The factor network's input is the
`P_port = len(portfolio_columns)` managed portfolios. Both input widths are
inferred from the data, so nothing about them is declared in config. Each period's
cross-section is one training example: the beta-input matrix `Z_beta` (`[N, P_beta]`),
the portfolio-characteristic matrix `Z_port` (`[N, P_port]`) and the returns `r`
(`[N]`). The model supplies the two transforms it needs on top of the features:

* **standardization** — `data/dataset.py` rank-transforms each feature column
  cross-sectionally to `[-1, 1]` per period (missing → the cross-sectional median,
  `0`) and z-scores the return column cross-sectionally per period (zero mean, unit
  std). Toggle with `data.standardize`. It is per-date and stateless, so applying
  it per split is identical to standardizing the full panel.
* **managed portfolios** — the model forms `x = (Z_port^T Z_port)^{-1} Z_port^T r`
  (the OLS coefficient of returns on the portfolio characteristics) inside its
  forward pass; the factor network never sees pre-aggregated inputs.

Features are computed on the full panel before the split; every featurizer is
backward-looking or per-date cross-sectional, so no test information leaks into
earlier splits (the only forward-looking columns are the `fwd_ret_*` targets).

---

## Cross-validation

`data_pipeline.split.scheme` chooses how the panel is cut. `single` cuts it once at
two dates — the historical behaviour, still the default. `walk_forward` sweeps a
window across the sample and refits at every position, so the model is scored on
several disjoint out-of-sample windows rather than one:

```yaml
data_pipeline:
  split:
    scheme: walk_forward
    burn_in: 252            # drop the featurizers' warm-up before cutting anything
    walk_forward:
      mode: rolling         # rolling | expanding
      train_size: 1260      # periods (usable trading dates), so five years
      val_size: 252
      test_size: 252
      step: null            # null -> test_size: test windows tile edge to edge
      purge: 0              # periods dropped at each inner boundary; see below
      max_folds: null
```

**`rolling` vs `expanding`.** Rolling slides the train window forward with the rest
of the fold, so every fold trains on the same five years and the model is never
shown the distant past — the honest test of whether it still works once the regime
it was fitted in has passed. Expanding anchors the window at the start of the
sample and lets it grow. Both place the validation and test windows identically, so
the two are directly comparable.

**`purge` is set from the target horizon.** The target at date `t` is realized over
`(t, t+H]`, so the last `H-1` dates of a segment carry labels reaching into the
segment scored next. The rule is `purge >= H - 1`: with `fwd_ret_1d` (`H = 1`) zero
is correct, and switching the target to a monthly forward return on this daily
panel makes it `20`. Nothing derives it automatically — the horizon lives in the
featurizer that produced the target column, not in the split config.

**`burn_in` is set from the longest trailing featurizer window.** Features are
computed over the whole panel before it is split, so the first ~252 rows sit inside
the 12-month statistics and carry columns that are entirely missing, which
standardization then maps to a constant. Under a single split those dates are
diluted across a long train set; under walk-forward they can be a large fraction of
the first fold, which is where model selection happens first.

**Fold count.** `floor((N - burn_in - span) / step) + 1`, where
`span = train_size + val_size + test_size + 2*purge`. Only whole folds are
produced; a partial tail is left unused rather than scored on fewer periods than
the others. The schedule is printed at startup.

**Cost.** One fit per fold. Rolling is constant per fold — roughly
`K * train_size / (single-split train size)` times a single run. Expanding grows,
so it is about half that again. Use `max_folds: 2` and a short epoch budget to
shake out a configuration before committing to the full sweep.

### What a fold shares, and what it does not

Cross-sections are built **once** over the whole panel and a fold is a range of
positions into that shared sequence. Every transform in `data/dataset.py` is within
a single period — the rank, the z-score and the managed-portfolio least-squares
solve all run per date — so slicing a shared sequence is bit-identical to building
each fold's periods from its own frame, and overlapping folds reuse the same tensor
objects rather than re-ranking and re-solving the dates they share.

Folds are cut over the periods that **survive** `data.min_cross_section`, not over
the panel's date index. A date whose cross-section is too small to price produces
no training example, so cutting over the raw dates would shift every boundary after
the first dropped date.

Models, however, share nothing. Each fold constructs a new model and a new
`Trainer` (which owns the optimizer, the schedule and the diagnostics probes) and
reseeds *before* the model is built, since weight initialization draws from the
global RNG. Under the default `cv.seed_mode: fixed` every fold starts from
identical weights, so the spread across folds is a property of the data rather than
of the initialization.

```yaml
cv:
  seed_mode: fixed        # fixed | per_fold
  on_fold_error: skip     # skip | raise
```

---

## Quickstart

### Library

```python
from nekron.asset_pricing import conditional_autoencoder as cae

cfg = cae.ConditionalAutoencoderConfig()  # or compose the Hydra config
# cfg.data.return_column is the target; cfg.data.portfolio_columns the portfolio subset;
# cfg.data_pipeline holds the ingestion / preprocessing / feature / split configs.

# Assemble the panel, build every usable period once, cut the configured schedule.
panel, bounds = cae.build_schedule(cfg)
with cae.MlflowTracker(cfg.mlflow) as tracker:
    result = cae.run_folds(panel, bounds, cfg, tracker)

result.pooled       # RSquared over every scored period of every fold
result.summary      # one row per fold: bounds, best epoch, val and test scores
result.periods      # one row per scored date: sse/sst, ready for a distribution
```

One fold or twenty, this is the same call: a single split is a schedule of length
one. The lower-level pieces are still there when a single fit is all that is
wanted:

```python
from nekron.data_adapter import build_datasets

splits = build_datasets(cfg.data_pipeline)               # raises under walk_forward
panel_splits = cae.build_panel_splits(splits, cfg.data)
model = cae.ConditionalAutoencoder.from_config(          # widths inferred from the data
    cfg,
    num_beta_columns=panel_splits.num_beta_columns,
    num_portfolios=panel_splits.num_portfolios,
)
with cae.MlflowTracker(cfg.mlflow) as tracker:
    cae.Trainer(model, cfg, panel_splits, tracker).fit()
```

### CLI (Hydra)

```bash
# trains on the configured pipeline (CRSP ingestion -> preprocessing -> features)
python -m nekron.asset_pricing.conditional_autoencoder \
    data_pipeline.split.train_end=2016-12-31 data_pipeline.split.val_end=2018-12-31 \
    model.num_factors=6 optim.lr=1e-3

# print the resolved config (including the composed data_pipeline), don't train
python -m nekron.asset_pricing.conditional_autoencoder --cfg job
# sweep the factor count
python -m nekron.asset_pricing.conditional_autoencoder --multirun model.num_factors=1,3,5

# walk-forward: five-year rolling window, one-year validation and test, one-year step
python -m nekron.asset_pricing.conditional_autoencoder \
    data_pipeline.split.scheme=walk_forward \
    data_pipeline.split.burn_in=252 \
    data_pipeline.split.walk_forward.train_size=1260

# two folds and a short budget first, to check the schedule before paying for it
python -m nekron.asset_pricing.conditional_autoencoder \
    data_pipeline.split.scheme=walk_forward \
    data_pipeline.split.walk_forward.max_folds=2 train.epochs=20
```

---

## Configuration

Both networks are fully configurable through Hydra. Their **input** widths are
inferred from the data — the beta network from every non-return feature column, the
factor network from `len(data.portfolio_columns)` — while their **output** width
(`K = model.num_factors`) is shared, since both emit `K` values for the `beta . f`
inner product. Everything else is per network:

| Field | Beta network | Factor network |
|-------|--------------|----------------|
| `hidden_dims` | `[32, 16, 8]` (CA3) | `[]` (linear) |
| `activation` | `relu` | `relu` |
| `batch_norm` | `true` | `false` (unsupported: one vector per period) |
| `dropout` | `0.0` | `0.0` |
| `bias` | `true` | `false` |

Set `beta_network.hidden_dims` to `[]`, `[32]`, `[32,16]` or `[32,16,8]` for the
paper's CA0–CA3. Loss/optimization defaults: `l1_lambda=1e-4`, Adam `lr=1e-3`,
`weight_decay=0` (the paper regularizes with L1 + early stopping, not L2),
early-stopping patience `5`, seed `42`, selection on validation `total_r2`.

---

## Outputs & interpretation

`Trainer.evaluate` returns `RSquared(total, predictive)`, both benchmarked against a
zero forecast (uncentered denominator `Σ r²`, per the paper):

* **total R²** `= 1 − Σ(r − beta . f_t)² / Σ r²` — cross-sectional fit using the
  period's contemporaneous fitted factors.
* **predictive R²** `= 1 − Σ(r − beta . λ_{t-1})² / Σ r²` — where `λ_{t-1}` forecasts
  the period's factors from the factors of earlier periods only. Which forecast is
  configured under `factor_forecast`:

  | `mode` | `λ_{t-1}` | knob |
  | --- | --- | --- |
  | `expanding_mean` | prevailing mean of every factor observed so far | — |
  | `ewma` | `(1−α) λ_{t-2} + α f_{t-1}`, seeded at the first factor | `ewma_alpha` |

  `ewma_alpha` is the weight on the newest factor (pandas' `ewm(alpha=…,
  adjust=False)`); a factor `k` periods old carries `(1−α)^k`, so the memory reads
  as a halflife of `ln(0.5)/ln(1−α)` periods — 69 at the default `0.01`.

  The forecast is **one object carried through a fold in calendar order**: warmed on
  the training window by `Trainer.advance_forecast`, carried into validation and
  then into test by `Trainer.score(..., forecast=)`, which returns the rows *and*
  the state the next window opens with. So every validation and test period is
  scored, not just from the second onward, and the per-epoch `val/predictive_r2`
  is the same statistic the fold reports as `val/final_predictive_r2`.

  Advancing the forecast over a window that is not itself scored — the training
  window — costs one batched pass through the factor network alone, since the beta
  network is not needed to produce a factor. That is what keeps the train-seeded
  epoch metric from adding a forward pass over the cross-sections; it computes the
  same function as the per-period call, to float32 rounding.

`Trainer.predict` returns per-period factors `[W, K]`, loadings and fitted returns.

> **Not comparable across this change.** Validation predictive R²
> (`val_predictive_r2`, `val/final_predictive_r2`, per-epoch `val/predictive_r2`) used
> to start cold on the validation window and throw its first period away; it now
> opens carrying the training window. The metric *keys* are deliberately unchanged —
> renaming them would fork the MLflow history and silently drop the old series from
> every query — so a run recorded before this change is a different statistic under
> the same name. Test predictive R² keeps its meaning under `expanding_mean` — the
> chain train → val → test accumulates the same factors the old
> `train_sum + val_sum` seed did — but not the same floats. Two effects, both far
> below anything the statistic resolves: summing one window and then the next is
> not the same float as adding two subtotals (~1e-14 relative), and the training
> window's factors now come from a batched factor-network pass rather than `W`
> per-period ones, which on some backends rounds differently at float32 precision
> (~1e-7 relative on `λ`).

### What a sweep records

A run is one MLflow run tree. The parent carries the configuration and the
aggregate; each fold is a nested run underneath it carrying its own curve, model
and errors:

```
run: <run_name>                       parent
  params    the full flattened config
  tags      cv_scheme, cv_mode, cv_folds, cv_purge, cv_burn_in
  metrics   test/pooled_total_r2, test/pooled_predictive_r2
            test/total_r2_{mean,median,std,min,max,iqr}
            test/total_r2 at step=fold_index   -> the sweep as a curve
            folds/{planned,completed,failed,test_periods}
  artifacts folds.parquet          one row per fold
            test_periods.parquet   every scored date of every fold
            columns.json           beta/portfolio names, in input order
  └─ run: fold-00                   nested
       params    fold_index, seed, the config as that fold saw it
       tags      train_start/end/size, val_*, test_* — searchable
       metrics   train/*, val/*, factors/*, grad/* by epoch; test/* at step 0
       artifacts model/                 mlflow.pytorch — load_model()
                 checkpoint             state, config, both column tuples
                 test_periods.parquet
                 columns.json
```

**Squared errors are recorded, not ratios.** An R² is a ratio of sums, so a set of
per-fold R² values cannot be combined into the R² of their union — each has its own
denominator. `test_periods.parquet` therefore carries `sse_total`, `sst_total`,
`sse_pred` and `sst_pred` per date, from which any later pooling (by fold, by year,
by regime) is a groupby. The schema is unchanged by the configurable forecast — only
what a `sse_pred` value was measured against is, and `factor_forecast` is recorded in
the run's parameters. `total_r2` and `predictive_r2` are included per row for
convenience but are derived. `pooled_r_squared` is the aggregation that is correct;
the mean of per-fold scores is not, and will differ whenever folds price different
numbers of stocks or differently volatile windows.

Two distributions come out of a sweep, and they answer different questions:
`result.summary` has one row per fold — the sample behind a box plot *across folds*
— and `result.periods` has one row per scored date, which is a much larger sample
and supports per-fold or per-year boxes, regime splits, and forecast-comparison
tests.

### Reading the models back

```python
from nekron.asset_pricing import conditional_autoencoder as cae

folds = cae.load_folds(parent_run_id, cfg.mlflow)   # models + column order + bounds
panel, bounds = cae.build_schedule(cfg)             # cached; rebuilds the windows

importances = cae.fold_importances(
    folds, [panel.view(b.test) for b in bounds], method="zero_out"
)
cae.importance_summary(importances)   # median, spread and sign consistency per characteristic
```

`zero_out` follows the paper: set a characteristic to zero everywhere and record
the drop in total R². Zero is the right neutral value rather than an arbitrary one
— the features are rank-mapped to `[-1, 1]` with the median at zero, so zeroing a
column *is* setting every stock to that characteristic's median. When the
characteristic is also a portfolio column it is neutralized on both sides and the
period's managed portfolios are re-solved, because a characteristic the model sees
twice is not held out by removing it once. `sensitivity` instead reports the mean
absolute gradient of the loadings with respect to the characteristic — how sharply
the loadings respond, rather than how much the fit needs it.

Because each fold answers separately, every characteristic gets one value per fold:
a stable characteristic is a tight box, a regime-dependent one is a wide box with an
outlier in a particular year, and `folds_positive` separates a characteristic the
model relies on from one whose large median rests on a single fold. That is a claim
a single split structurally cannot make.

**Column order is load-bearing.** The beta network's input width is inferred from
the data, so dimension `p` means "the p-th name in `beta_columns`" and nothing else.
Both column tuples are written next to every model (`columns.json`, and inside the
checkpoint) precisely so a model reloaded after the feature config has moved on
fails rather than answering confidently and wrongly.

**What does not pool.** The importances above are computed from fitted returns and
are therefore invariant to the model's rotation indeterminacy (`beta f = (beta R)(R⁻¹ f)`
reconstructs identically), so they compare fold to fold directly. Raw loadings and
the factor series are **not**: separate fits land in different bases, and "factor 3"
in one fold has no reason to be "factor 3" in the next. Stitching a factor series
across folds produces a plot that looks meaningful and is not; anything
factor-specific needs an explicit alignment step first.

---

## Documented interpretation decisions

Faithful to the paper, and where the reference reproductions diverge, the paper is
followed (all configurable):

* **No per-stock alpha** — the restricted, no-arbitrage model outputs exactly `K`
  loadings (`r_hat = beta . f`), not an intercept-augmented `K+1`.
* **Factor network is linear** — a single `P_port -> K` map with no hidden layer and
  no bias by default; the depth asymmetry (only the beta side is deep) is the paper's.
* **L1, not L2** — the objective is MSE plus an L1 penalty on the linear weight
  matrices; `optim.weight_decay` defaults to `0`.
* **Standardization** — features: cross-sectional rank to `[-1, 1]`, missing →
  median (`0`); returns: cross-sectional z-score (zero mean, unit std) per period.
* **Managed portfolios** — `x = (Z^T Z)^{-1} Z^T r`, the OLS coefficient of the
  cross-section of returns on the portfolio characteristics `Z_port` (least-squares
  solve). The factor input width `P_port` is independent of the beta width `P_beta`.
* **Learning-rate schedule** — the paper's "shrinkage" is Adam's built-in
  adaptivity; the default `optim.lr_decay_gamma=1.0` is plain Adam. An optional
  exponential decay is exposed for tuning.
* **Predictive-R² forecast** — the paper prices the predictive statistic against the
  prevailing (expanding) mean of the estimated factors, which is the default
  (`factor_forecast.mode=expanding_mean`). The exponentially weighted alternative is
  a deviation from the paper rather than a correction of it, exposed because a mean
  over a five-year window cannot track a premium that moves; only the predictive
  statistic changes under it, and by construction both estimators score the same set
  of periods, so the two numbers are comparable.

Ingestion, preprocessing and feature creation are wired through the shared
`nekron.data_adapter` (the `data_pipeline` config). Deferred: the paper's
**10-network ensemble** (train multiple seeds and average forecasts) — the trainer
here fits a single network.

---

## Notes

* One period is one forward pass, so the beta network's batch normalization
  normalizes over that period's cross-section. Training with `batch_norm=true`
  therefore needs at least two stocks per period (raise `data.min_cross_section`
  or disable batch norm for pathologically small cross-sections).
* `train.device = "auto"` selects CUDA → Apple-silicon MPS → CPU.
* `data.batch_periods` accumulates gradients over that many periods per optimizer
  step; each period's cross-section is processed independently.
