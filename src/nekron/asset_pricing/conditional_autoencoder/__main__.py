"""Executable entry point.

One code path for both schemes. ``data_pipeline.split.scheme="single"`` cuts one
fold and fits one model, exactly as before; ``"walk_forward"`` cuts the configured
sweep and fits one model per fold, each in its own nested MLflow run.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from . import (
    MlflowTracker,
    build_schedule,
    register_configs,
    run_folds,
    set_seed,
    to_config,
)

register_configs()
_CONFIG_PATH = "../../../../configs"


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="conditional_autoencoder")
def main(dict_cfg: DictConfig) -> None:
    cfg = to_config(dict_cfg)
    set_seed(cfg.train.seed)

    panel, bounds = build_schedule(cfg)
    split = cfg.data_pipeline.split
    print(
        f"Data -> periods={len(panel)} "
        f"({panel.dates[0].date()}..{panel.dates[-1].date()}) "
        f"beta_columns={panel.num_beta_columns} portfolios={panel.num_portfolios} "
        f"factors={cfg.model.num_factors}"
    )
    scheme = (
        f"{split.scheme}/{split.walk_forward.mode}"
        if split.scheme == "walk_forward"
        else split.scheme
    )
    print(f"Schedule -> {scheme} folds={len(bounds)} burn_in={split.burn_in}")
    for fold in bounds:
        print(f"  {fold.window(panel.dates).describe()}")

    with MlflowTracker(cfg.mlflow) as tracker:
        result = run_folds(panel, bounds, cfg, tracker)

    print(result.describe())
    for outcome in result.folds:
        if not outcome.completed:
            print(f"  fold {outcome.index} FAILED: {outcome.error}")


if __name__ == "__main__":
    main()
