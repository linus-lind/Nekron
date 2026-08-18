"""Tests for cross-sectional and time-series normalization featurizers."""

from __future__ import annotations

import numpy as np

from nekron.features import (
    CrossSectionalDemean,
    CrossSectionalRank,
    CrossSectionalWinsorize,
    CrossSectionalZScore,
    FeaturePipeline,
    GroupNeutralize,
    RollingZScore,
)

from .conftest import make_panel


def _run(feat, panel):
    return FeaturePipeline(featurizers=(feat,)).apply(panel)


def test_signed_rank_hits_bounds_per_date() -> None:
    panel = make_panel(
        ["2020-01-01", "2020-01-02"],
        ["A", "B", "C", "D"],
        {"x": [4.0, 3.0, 2.0, 1.0, 10.0, 20.0, 30.0, 40.0]},
    )
    out = _run(
        CrossSectionalRank(("x",), ("r",), method="average", ascending=True, scaling="signed"),
        panel,
    )
    for _, grp in out.groupby(level="date"):
        vals = np.sort(grp["r"].to_numpy())
        assert np.isclose(vals[0], -1.0) and np.isclose(vals[-1], 1.0)


def test_signed_rank_single_name_is_zero() -> None:
    panel = make_panel(["2020-01-01"], ["A"], {"x": [5.0]})
    out = _run(
        CrossSectionalRank(("x",), ("r",), method="average", ascending=True, scaling="signed"),
        panel,
    )
    assert np.isclose(out["r"].to_numpy()[0], 0.0)


def test_zscore_and_demean_per_date() -> None:
    panel = make_panel(["2020-01-01"], ["A", "B", "C"], {"x": [1.0, 2.0, 3.0]})
    z = _run(CrossSectionalZScore(("x",), ("z",), ddof=1), panel)["z"].to_numpy()
    dm = _run(CrossSectionalDemean(("x",), ("d",)), panel)["d"].to_numpy()
    assert np.isclose(np.mean(z), 0.0, atol=1e-12)
    assert np.allclose(dm, [-1.0, 0.0, 1.0])


def test_winsorize_clips_outlier() -> None:
    entities = [str(i) for i in range(11)]
    panel = make_panel(["2020-01-01"], entities, {"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000.0]})
    out = _run(CrossSectionalWinsorize(("x",), ("w",), lower=0.0, upper=0.9), panel)
    assert out["w"].to_numpy().max() < 1000.0


def test_group_neutralize_demean_within_group() -> None:
    panel = make_panel(
        ["2020-01-01"],
        ["A", "B", "C", "D"],
        {"x": [1.0, 3.0, 10.0, 20.0], "sector": ["fin", "fin", "tech", "tech"]},
    )
    out = _run(GroupNeutralize(("x",), ("n",), group_column="sector", mode="demean", ddof=1), panel)
    vals = out["n"].to_numpy()
    # fin mean 2 -> [-1, 1]; tech mean 15 -> [-5, 5]
    assert np.allclose(np.sort(vals), [-5.0, -1.0, 1.0, 5.0])


def test_rolling_zscore_per_entity() -> None:
    dates = [f"2020-01-{i:02d}" for i in range(1, 11)]
    panel = make_panel(dates, ["A"], {"x": list(map(float, range(10)))})
    out = _run(RollingZScore("x", (5,), ("z",), min_periods=5, ddof=1), panel)["z"].to_numpy()
    valid = out[~np.isnan(out)]
    assert np.isfinite(valid).all()
    # last window [5,6,7,8,9], mean 7, std ~1.5811 -> z of 9 ~1.2649
    assert np.isclose(out[-1], (9 - 7) / np.std([5, 6, 7, 8, 9], ddof=1))
