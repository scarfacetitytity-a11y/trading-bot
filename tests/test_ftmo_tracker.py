"""FTMOTracker unit tests — DD gate, profit target, best day rule."""
import pytest
from unittest.mock import patch

from execution.ftmo_tracker import FTMOTracker


@pytest.fixture()
def tracker(tmp_path):
    state_path = tmp_path / "ftmo_tracker_state.json"
    with patch("execution.ftmo_tracker.STATE_FILE", state_path):
        yield FTMOTracker(initial_equity=10_000, challenge="2step-p1")


# ── DD gate — orchestrator uses check()["total_dd_pct"] >= ["total_dd_limit"] ─

def test_dd_within_limit(tracker):
    p = tracker.check(9_500)  # 5% DD from 10k peak → safe
    assert p["total_dd_pct"] < p["total_dd_limit"]


def test_dd_at_limit(tracker):
    p = tracker.check(9_000)  # 10% DD = exactly at limit
    assert round(p["total_dd_pct"], 4) >= p["total_dd_limit"]


def test_dd_above_limit(tracker):
    p = tracker.check(8_950)  # 10.5% → breach
    assert p["total_dd_pct"] >= p["total_dd_limit"]


# ── Profit target ─────────────────────────────────────────────────────────────

def test_not_passed_below_target(tracker):
    p = tracker.check(10_900)  # 9% → not yet
    assert not p["passed"]


def test_passed_at_target(tracker):
    from datetime import date, timedelta
    base = date(2026, 7, 1)
    # record_trade_day only records today once — inject 4 distinct dates directly
    tracker.state.trade_days = [str(base + timedelta(days=i)) for i in range(4)]
    p = tracker.check(11_000)  # exactly 10%
    assert p["passed"]


# ── Trading days count ────────────────────────────────────────────────────────

def test_trading_days_not_met(tracker):
    tracker.record_trade_day()
    p = tracker.check(11_000)
    assert not p["trading_days_met"]


def test_trading_days_met_after_four(tracker):
    from datetime import date, timedelta
    base = date(2026, 7, 1)
    # Inject 4 unique dates directly
    tracker.state.trade_days = [
        str(base + timedelta(days=i)) for i in range(4)
    ]
    p = tracker.check(11_000)
    assert p["trading_days_met"]


# ── Progress pct ──────────────────────────────────────────────────────────────

def test_progress_pct_halfway(tracker):
    p = tracker.check(10_500)  # 5% profit = 50% of 10% target
    assert abs(p["progress_pct"] - 50.0) < 0.1


# ── Peak equity advances ──────────────────────────────────────────────────────

def test_peak_equity_advances(tracker):
    tracker.check(10_500)
    assert tracker.state.peak_equity == 10_500
    tracker.check(10_300)  # falls back — peak should not drop
    assert tracker.state.peak_equity == 10_500
    # DD should be computed from 10_500, not 10_000
    p = tracker.check(10_300)
    expected_dd = (10_500 - 10_300) / 10_500 * 100
    assert abs(p["total_dd_pct"] - expected_dd) < 0.01


# ── 1-step best day rule ──────────────────────────────────────────────────────

def test_best_day_rule_violation(tmp_path):
    state_path = tmp_path / "ftmo.json"
    with patch("execution.ftmo_tracker.STATE_FILE", state_path):
        t = FTMOTracker(initial_equity=10_000, challenge="1step")
        # Day 1: +600, Day 2: +300, Day 3: +100 → total +1000, max=600 > 50%
        t.state.daily_pnl = {"2026-07-01": 600, "2026-07-02": 300, "2026-07-03": 100}
        p = t.check(11_000)
        assert p["best_day_violation"]


def test_best_day_rule_ok(tmp_path):
    state_path = tmp_path / "ftmo.json"
    with patch("execution.ftmo_tracker.STATE_FILE", state_path):
        t = FTMOTracker(initial_equity=10_000, challenge="1step")
        t.state.daily_pnl = {"2026-07-01": 400, "2026-07-02": 350, "2026-07-03": 250}
        p = t.check(11_000)
        assert not p["best_day_violation"]
