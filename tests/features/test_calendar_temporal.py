"""Tests for calendar and temporal (lag/lead/difference) featurizers."""

from __future__ import annotations

import numpy as np

from nekron.features import CalendarFeatures, Difference, FeaturePipeline, TemporalShift

from .conftest import make_panel, single_entity_panel


def _run(feat, panel):
    return FeaturePipeline(featurizers=(feat,)).apply(panel)


def test_calendar_attributes() -> None:
    # 2020-01-31 is a Friday and a month end
    panel = make_panel(["2020-01-30", "2020-01-31"], ["A"], {"x": [1.0, 2.0]})
    out = _run(
        CalendarFeatures(("day_of_week", "month", "is_month_end"), ("dow", "month", "eom")),
        panel,
    )
    assert out["month"].unique().tolist() == [1.0]
    eom = out.xs("A", level="entity")["eom"].to_numpy()
    assert eom[0] == 0.0 and eom[1] == 1.0
    assert out.xs("A", level="entity")["dow"].to_numpy()[1] == 4.0  # Friday


def test_temporal_shift_lag_and_lead() -> None:
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    panel = single_entity_panel(dates, {"x": [1.0, 2.0, 3.0]})
    out = _run(TemporalShift("x", (1, -1), ("lag1", "lead1")), panel)
    lag = out["lag1"].to_numpy()
    lead = out["lead1"].to_numpy()
    assert np.isnan(lag[0]) and lag[1] == 1.0 and lag[2] == 2.0
    assert lead[0] == 2.0 and lead[1] == 3.0 and np.isnan(lead[2])


def test_difference() -> None:
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    panel = single_entity_panel(dates, {"x": [1.0, 4.0, 9.0]})
    out = _run(Difference("x", (1,), ("d1",)), panel)["d1"].to_numpy()
    assert np.isnan(out[0]) and out[1] == 3.0 and out[2] == 5.0


def test_lead_is_a_target_no_bleed_across_entities() -> None:
    panel = make_panel(["2020-01-01", "2020-01-02"], ["A", "B"], {"x": [1.0, 10.0, 2.0, 20.0]})
    out = _run(TemporalShift("x", (-1,), ("target",)), panel)
    a = out.xs("A", level="entity")["target"].to_numpy()
    b = out.xs("B", level="entity")["target"].to_numpy()
    assert a[0] == 2.0 and np.isnan(a[1])
    assert b[0] == 20.0 and np.isnan(b[1])  # B's future is B's, not A's
