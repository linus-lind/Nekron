"""Executable entry point.

Recovers the frozen conditional autoencoder from its MLflow run, replays it into a
residual panel, cuts the panel into windowed folds, and fits one model per fold.
One code path for both schemes: a single split is a schedule with one fold, so
``scheme="single"`` fits one model exactly as before and ``"walk_forward"`` fits
one per position of the sweep, each in its own nested MLflow run.

With no objective configured — the default — the run stops after recording what it
built, which is what makes the pipeline runnable while :mod:`.losses` is still a
stub.
"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from . import (
    LstmTransformer,
    MlflowTracker,
    Trainer,
    WindowSplits,
    build_residual_panel,
    build_schedule,
    load_pretrained_cae,
    register_configs,
    resolve_device,
    run_folds,
    set_seed,
    to_config,
)

register_configs()
_CONFIG_PATH = "../../../../configs"


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="lstm_transformer")
def main(dict_cfg: DictConfig) -> None:
    cfg = to_config(dict_cfg)
    set_seed(cfg.train.seed)

    cae = load_pretrained_cae(cfg.cae)
    print(cae.describe())

    panel = build_residual_panel(cae, device=cfg.cae.device)
    print(panel.describe())

    device = resolve_device(cfg.train.device)
    windows, bounds = build_schedule(panel, cfg, device=device)
    split = panel.split if cfg.data.inherit_cae_split else cfg.data.split
    scheme = (
        f"{split.scheme}/{split.walk_forward.mode}"
        if split.scheme == "walk_forward"
        else split.scheme
    )
    print(f"Schedule -> {scheme} folds={len(bounds)}")
    for planned in bounds:
        print(f"  {planned.window(windows.dates).describe()}")

    provenance = {
        "cae_run_id": cae.run_id,
        "cae_weights": cae.weights_digest[:16],
        "seq_len": str(cfg.data.seq_len),
    }

    if cfg.loss.objective == "none":
        # Nothing to optimize yet, so the sweep would only record three identical
        # refusals. Build the first fold instead and report what it produced.
        splits = windows.splits(bounds[0])
        print(splits.describe())
        model = LstmTransformer.from_config(cfg, in_dim=splits.num_channels, seq_len=splits.seq_len)
        print(f"Model parameters: {model.num_parameters():,} on {device}")
        with MlflowTracker(cfg.mlflow) as tracker:
            Trainer(model, cfg, splits, tracker, provenance=provenance).log_summary()
            _forward_check(model, splits, device)
        print(
            "\nloss.objective='none': the residuals, the windows, the model and the trainer "
            "are built and recorded, and there is nothing to optimize yet.\n"
            "Implement an objective in "
            "nekron.time_series_modelling.lstm_transformer.losses, add its name to "
            "config.OBJECTIVES, then set loss.objective to it."
        )
        return

    with MlflowTracker(cfg.mlflow) as tracker:
        tracker.set_tags(provenance)
        result = run_folds(windows, bounds, cfg, tracker)

    print(result.describe())
    for fold in result.folds:
        if not fold.completed:
            print(f"  fold {fold.index} FAILED: {fold.error}")


@torch.no_grad()
def _forward_check(model: LstmTransformer, splits: WindowSplits, device: torch.device) -> None:
    """Score one real batch, so a build-only run still proves the model runs.

    The shapes a stack of this kind gets wrong are the ones that only appear on
    real data — the window length against the positional tables, the encoder's
    output width against the head's input — and none of them is visible from a
    parameter count.
    """
    batch = next(splits.train.iter_batches(8, shuffle=False), None)
    if batch is None:
        return
    was_training = model.training
    model.eval()
    output = model(batch.sequences.to(device))
    model.train(was_training)
    print(
        f"Forward check -> windows={len(batch)} "
        f"sequences={tuple(batch.sequences.shape)} -> scores={tuple(output.scores.shape)} "
        f"(finite={bool(torch.isfinite(output.scores).all())})"
    )


if __name__ == "__main__":
    main()
