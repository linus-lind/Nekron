# LSTM-Transformer — a sequence model over conditional-autoencoder residuals

A fitted conditional autoencoder prices each date's cross-section and leaves a
residual behind:

```
u_{i,t} = r_{i,t} - beta(z_{i,t})' f_t
```

one series per entity, one value per date it was priced on. This package models
those series. It recovers the frozen autoencoder from its MLflow run, replays it
to build the residual panel, cuts each entity's series into fixed-length windows
wherever the whole span is present, and scores each window with

```
bidirectional LSTM -> transformer stack -> temporal pooling -> MLP score head
```

**The objective is deliberately absent.** Everything around it is finished, so a
run today builds the residuals, the windows, the model and the trainer, records
them, and stops short of fitting. See [Implementing the objective](#implementing-the-objective).

---

## Architecture

| Component | File | Configured by |
|-----------|------|---------------|
| Frozen autoencoder, recovered from MLflow | `data/residuals.py` | `cae` |
| Residual panel `[periods, entities]` | `data/residuals.py` | (from the autoencoder's own config) |
| Gap-free windows, split by their last date | `data/windows.py` | `data` |
| Bidirectional LSTM over the window | `nekron.nn.LSTMEncoder` (shared) | `model.lstm` |
| Transformer stack (RoPE / sinusoidal / none, pre- or post-LN, causal or not) | `nekron.nn.TransformerEncoder` (shared, wraps `torch.nn.TransformerEncoder`) | `model.transformer` |
| Temporal aggregation (mean / last / max / learned attention / concatenations) | `nekron.nn.TemporalPooling` (shared) | `model.pooling` |
| MLP score head | `nekron.nn.MLP` (shared) | `model.head` |
| Training objective | `losses.py` | `loss` — **stub** |
| Adam trainer, early stopping, diagnostics, checkpointing | `engine/trainer.py` + `nekron.nn.training` (shared) | `optim`, `train` |
| Walk-forward sweep: schedule, nested runs, pooling | `nekron.cv` (shared) + `engine/cv.py` | `cv`, `data.split` |

```
lstm_transformer/
├── config.py            # Hydra structured-config schema (dataclasses)
├── model.py             # LstmTransformer nn.Module + ScoreOutput
├── losses.py            # objective contract -- NOT YET IMPLEMENTED
├── data/
│   ├── residuals.py     # MLflow run -> frozen CAE -> residual panel (cached)
│   └── windows.py       # residual panel -> windows -> train/val/test
├── modules/
│   ├── sequence_encoder.py  # LSTM -> transformer -> pooling
│   └── score_head.py        # pooled vector -> scores
└── engine/
    ├── trainer.py       # Adam, early stopping, MLflow, checkpoint
    └── cv.py            # this model's fold body + pooler for nekron.cv
```

Every hyperparameter lives in `configs/lstm_transformer.yaml`, typed by the
dataclasses in `config.py`; nothing that governs model behaviour is hard-coded
inside the modules. The four building blocks are model-agnostic and live in the
reusable `nekron.nn` package.

---

## Data

### Where the autoencoder comes from

One setting: `cae.run_id`. The run's checkpoint artifact carries the weights, the
column order they were fitted against *and* the entire data-pipeline
configuration that produced that panel, so the residuals are reproducible from a
run id alone.

**This package's config therefore has no `data_pipeline` block.** A second copy
of the autoencoder's pipeline could disagree with the one it was actually fitted
under, and residuals computed against a different feature panel are not residuals.
The one thing that is *checked* rather than assumed is the column order: the
networks' input widths are inferred from the data, so input dimension `p` means
"the p-th name in `beta_columns`" and nothing else. A feature configuration that
has moved on since the fit is refused by name, loudly, rather than silently
mispriced.

The checkpoint artifact is preferred over the run's logged *model*: MLflow pickles
the whole `nn.Module`, tying it to the module path and the device it was pickled
on, and it carries neither the column order nor the data configuration.

### Units

The autoencoder is fitted on cross-sectionally z-scored returns, so `r`, `r_hat`
and therefore `u` are all in **standardized** units — a residual of 0.5 is half a
cross-sectional standard deviation of that period's returns. Nothing is rescaled.
That keeps the series comparable across periods of very different volatility,
which is what a fixed-length sequence model needs. Recovering raw return units is
a multiplication by the period's cross-sectional return standard deviation, which
the panel would have to be re-read to obtain.

The residuals are **not mean-zero within a period**: the model carries no
intercept, so nothing forces the cross-sectional mean of `u` to vanish.

### Windows

A window is `data.seq_len` (default 252, one trading year) consecutive periods of
**one** entity's residual series, and it exists only where that entity has a
residual on *every* one of those periods. A gap ends the run and the count starts
again after it — an entity that leaves the investable universe for a week
contributes no window spanning that week, rather than one with a hole interpolated
into it.

Windows are an **index, not a copy**. A daily panel of a thousand names over
fifteen years carries a few million valid windows; at 252 steps each, storing them
would cost hundreds of gigabytes, while the residual matrix they are all cut from
is a few dozen megabytes. The matrix is stored once, on the training device, and a
window is two integers. A batch is one gather against that shared matrix, which
never crosses the host boundary during training.

### Splits

A window belongs to the split containing its **last** date; the preceding
`seq_len - 1` periods may reach back across a boundary.

```
train |............................|
val                    |xxxxxxxxxxxxxxxxxxxx|
                        ^--- window ends here -> VAL
                        lookback reaches into train (allowed)
```

That is deliberate. A window's lookback lies entirely in the past of the date it
is assigned to, so nothing later than that date is ever read; demanding full
containment would cost `seq_len - 1` windows at every boundary, which at the
default 252 is most of a one-year validation window.

The schedule itself is inherited from the autoencoder's own run
(`data.inherit_cae_split`, default true). That is what keeps the frozen
autoencoder out of sample on this model's test window: the dates it was fitted on
stay this model's *training* dates.

### Caching

Replaying the autoencoder is seconds; rebuilding the per-period cross-sections it
consumes is minutes, because every feature column is ranked within every date. The
residual panel is therefore cached like any other pipeline stage, in the same
`.nekron_cache` directory, keyed on the **fitted weights** as well as on the
configuration and the identity of every source file — two runs of the same
architecture produce different residuals, and a key that ignored the weights would
hand the second run the first one's.

---

## Quickstart

### CLI (Hydra)

```bash
# build residuals, windows and the model from a fitted autoencoder run
python -m nekron.time_series_modelling.lstm_transformer cae.run_id=<run id>

# print the resolved config, don't build anything
python -m nekron.time_series_modelling.lstm_transformer cae.run_id=<id> --cfg job

# a shorter window, a causal stack, sinusoidal positions
python -m nekron.time_series_modelling.lstm_transformer cae.run_id=<id> \
    data.seq_len=126 model.transformer.causal=true \
    model.transformer.positional_encoding=sinusoidal

# sweep the aggregation
python -m nekron.time_series_modelling.lstm_transformer --multirun cae.run_id=<id> \
    model.pooling.kind=last,mean,attention,last_mean
```

### Library

```python
from nekron.time_series_modelling import lstm_transformer as lt

cfg = lt.to_config(composed_config)
cae = lt.load_pretrained_cae(cfg.cae)           # MLflow run -> frozen model + its config
panel = lt.build_residual_panel(cae)            # replay -> [periods, entities], cached
device = lt.resolve_device(cfg.train.device)
splits = lt.build_window_splits(panel, cfg.data, device=device)
model = lt.LstmTransformer.from_config(cfg, in_dim=splits.num_channels, seq_len=splits.seq_len)
with lt.MlflowTracker(cfg.mlflow) as tracker:
    trainer = lt.Trainer(model, cfg, splits, tracker)
    scores = trainer.predict(splits.test)       # works with no objective
    result = trainer.fit()                      # raises until one exists
```

---

## Configuration

Only the **input** width is inferred from the data (the number of channels a
residual window carries — one, the residual itself). Every width after it is
determined by the stage before it: the transformer is told the LSTM's output width
and projects it to `d_model` only if they differ, the pooling is told the
transformer's, and the head is told the pooling's. No setting can claim a width
that disagrees with the one the layer before it actually produces.

| Section | Notable settings |
|---------|------------------|
| `model.lstm` | `hidden_dim` 128, `num_layers` 2, `bidirectional` true, `dropout` (between layers only), `output_dropout` |
| `model.transformer` | `d_model` 256, `num_layers` 2, `num_heads` 8, `ff_dim` 1024, `norm_position` pre/post, `positional_encoding` none/sinusoidal/rope, `causal`, `dropout` |
| `model.pooling` | `kind` last / mean / max / attention / last_mean / mean_max, `attention_dim` |
| `model.head` | `hidden_dims` `[64]`, `out_dim` 1, `activation`, `batch_norm`, `dropout` |
| `data` | `seq_len` 252, `stride` 1, `target_horizon` 0, `batch_size` 256, `inherit_cae_split` |
| `optim` / `train` | Adam `lr=1e-4`, `grad_clip_norm=1.0`, early stopping patience 10, per-epoch diagnostics. These three subclass `nekron.nn.OptimConfig` / `TrainConfig` purely to restate defaults this model was tuned at, which differ from the shared ones. |
| `cv` | `seed_mode`, `on_fold_error` — the shared sweep policy from `nekron.cv` |

Constraints are validated at config time, not at forward time: `d_model` must be
divisible by `num_heads`, RoPE additionally needs an even head width, sinusoidal
needs an even `d_model`.

### Built on torch's own encoder

`nekron.nn.TransformerEncoder` configures
`torch.nn.TransformerEncoderLayer` / `torch.nn.TransformerEncoder` rather than
reimplementing attention. The blocks, the attention, the feed-forward sublayer,
the residual connections, the normalization placement and the parameter
initialization are all torch's; this package adds the input projection, the
positional encoding, the causal mask and the configuration surface. `norm_position`
maps onto torch's `norm_first`, `ff_dim` onto `dim_feedforward`, and `dropout` is
torch's single rate (attention weights, feed-forward net, and each sublayer's
output). `activation` is handed over as a *module* rather than by name, which keeps
the whole `nekron.nn` activation registry available where torch's string form
accepts only `relu` and `gelu` — and keeps the activation visible to
`ActivationProbe`.

The one exception is RoPE. `torch.nn.MultiheadAttention` projects, splits and
attends in a single call, and rotary embeddings have to be applied to the queries
and keys *between* the projection and the attention; there is no hook there. So
`RotaryEncoderLayer` subclasses torch's encoder layer and replaces its attention
block, inheriting the layer's parameters, initialization, feed-forward sublayer,
norms and dropouts unchanged — a rotary stack and an ordinary one differ in exactly
one operation, and their state dicts are interchangeable.

That subclass also overrides `forward`, and must. Torch's own `forward` dispatches
to a fused kernel in evaluation mode under `no_grad` — precisely how a model is
scored — and that kernel never calls `_sa_block`. A layer that replaced only
`_sa_block` would rotate while training and silently stop while predicting. The
override is torch's own slow path, and a regression test asserts the rotary
attention still runs once per layer under `eval()` + `no_grad()`.

### Causality

`model.transformer.causal` restricts the **attention** so a step may not attend to
later steps of the window. It does *not* by itself make the model causal step by
step, because a bidirectional LSTM upstream has already mixed the window in both
directions. For a step-wise causal model set `model.lstm.bidirectional=false` as
well.

The default (`causal: false`, `bidirectional: true`) is the right one here: the
window lies entirely in the past of the date being scored, so mixing *within* the
window reads nothing from the future. Only mixing *across the window's leading
edge* would, and nothing does.

### Targets

`data.target_horizon` is `0` by default, which carries no target at all — the
honest setting while no objective exists. A positive value reads the residual that
many periods after the window and enforces two rules: the target's date must fall
inside the **same split** as the window (so a window at the end of training is
never labelled from the first day of validation), and the target cell must itself
carry a residual.

---

## Implementing the objective

`losses.py` defines the contract and nothing else. Two edits, no rewiring:

1. write a class satisfying `ScoreObjective` in `losses.py`;
2. add its name to `config.OBJECTIVES` and to `build_objective`.

An objective is handed `(output, batch, model_parameters)`:

* **`output`** — both the scores and the pooled representation behind them, so a
  penalty on the representation itself needs no second forward pass;
* **`batch`** — not just a target vector. A cross-sectional objective (rank the
  entities within each date, say) needs `date_indices` and `entity_indices` to know
  which scores belong together, and a batch of windows does not arrive grouped;
* **`model_parameters`** — so a weight penalty is part of the objective rather than
  a second thing the trainer has to know about.

Until then `build_objective` returns `None`, `Trainer.fit()` refuses with a message
naming what is missing, and `Trainer.predict()` still runs.

---

## Notes

* `train.device = "auto"` selects CUDA → Apple-silicon MPS → CPU. The residual
  matrix is moved there once and stays; the frozen autoencoder is replayed on the
  CPU by default (`cae.device`) because that pass is cached and reproducibility is
  worth more there than the seconds an accelerator saves.
* Absent cells in the residual matrix are `NaN`, not zero. They are never indexed —
  a window exists only where the whole span is present — so a `NaN` reaching a
  batch means a window index is wrong, which is exactly the failure that should be
  loud.
* `data.stride` is the first knob to reach for if an epoch is too long: at
  `stride: 1` a daily panel of a thousand names yields a few million windows per
  epoch.
* **Cross-validation** runs through the shared `nekron.cv` driver, exactly as the
  conditional autoencoder's does. `scheme: single` is a schedule with one fold and
  fits one model; `walk_forward` fits one per position, each in its own nested
  MLflow run, and the headline loss is pooled over every scored window of every
  fold rather than averaged over the folds' means. The residual matrix and the
  validity scan are built once by `build_window_panel` and each fold takes a
  constant-time `.splits(bounds)` view, which is what keeps a seven-fold sweep from
  holding seven copies of the largest object in the process.

  With no objective configured the sweep would only record one identical refusal
  per fold, so the entry point builds the first fold and reports it instead.

  Each fold contributes **one row per scored date**, carrying that date's window
  count and summed loss rather than their ratio — the same shape the autoencoder's
  per-period table has, and for the same reason: a mean over dates cannot be
  recovered from a set of per-date means, so pooling across folds, years or regimes
  after the fact is only possible from the numerator and denominator kept apart.
