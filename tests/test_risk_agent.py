"""RiskAgent unit tests — gate logic, size scaling, consecutive-loss pause."""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from execution.risk_agent import RiskAgent, RiskConfig, RiskState


@pytest.fixture()
def agent(tmp_path):
    """Fresh RiskAgent with no state file side-effects."""
    state_path = tmp_path / "risk_agent_state.json"
    with patch("execution.risk_agent.STATE_FILE", state_path):
        yield RiskAgent(initial_equity=10_000)


def _set_state(agent, **kwargs):
    for k, v in kwargs.items():
        setattr(agent.state, k, v)


# ── Normal pass ───────────────────────────────────────────────────────────────

def test_normal_can_trade(agent):
    can, mult, _ = agent.pre_trade_check(10_000)
    assert can
    assert mult == 1.0


# ── Concurrent trade cap ──────────────────────────────────────────────────────

def test_concurrent_trades_cap(agent):
    # Risk is governed by risk_pct per trade; the count cap is intentionally high
    # (default 20) so valid setups are never blocked by position count alone.
    cap = agent.config.max_concurrent_trades
    can, _, _ = agent.pre_trade_check(10_000, open_trade_count=cap - 1)
    assert can  # below cap → allowed

    can, mult, reason = agent.pre_trade_check(10_000, open_trade_count=cap)
    assert not can
    assert mult == 0.0
    assert "concurrent" in reason.lower()


# ── Daily loss limit ──────────────────────────────────────────────────────────

def test_daily_loss_blocks(agent):
    _set_state(agent, daily_start_equity=10_000)
    can, _, reason = agent.pre_trade_check(9_790)  # -2.1% → over 2% limit
    assert not can
    assert "daily" in reason.lower()


def test_daily_loss_within_limit(agent):
    _set_state(agent, daily_start_equity=10_000)
    can, _, _ = agent.pre_trade_check(9_810)  # -1.9% → under limit
    assert can


# ── Weekly drawdown ───────────────────────────────────────────────────────────

def test_weekly_dd_blocks(agent):
    # Gate order: #3 account-DD, #4 daily-loss, #5 weekly-DD.
    # Set peak and daily_start close to current so earlier gates don't fire first.
    _set_state(agent, weekly_start_equity=10_000,
               daily_start_equity=9_290,       # daily-loss = 0.1% < 2% limit
               peak_equity=9_350)              # account-DD = (9350-9280)/9350 = 0.75% < 7%
    can, _, reason = agent.pre_trade_check(9_280)  # weekly-DD = 7.2% ≥ 7%
    assert not can
    assert "weekly" in reason.lower()


# ── Account peak DD ───────────────────────────────────────────────────────────

def test_account_dd_blocks(agent):
    _set_state(agent, peak_equity=10_000,
               daily_start_equity=10_000,
               weekly_start_equity=10_000)
    can, _, reason = agent.pre_trade_check(9_280)  # 7.2% from peak
    assert not can
    assert "DD" in reason or "dd" in reason.lower()


# ── Rolling win rate ──────────────────────────────────────────────────────────

def test_low_wr_halves_size(agent):
    # 10 trades, 3 wins (30%) — below 35% scale threshold
    _set_state(agent, recent_trades=[-1, -1, 1, -1, -1, -1, 1, -1, 1, -1])
    can, mult, reason = agent.pre_trade_check(10_000)
    assert can
    assert mult == 0.5
    assert "half" in reason.lower() or "0.5" in reason or "30" in reason


def test_very_low_wr_quarter_size(agent):
    # Spec (risk_agent.py step 8): WR <= wr_halt_threshold scales to 0.25x rather
    # than halting. Old test asserted a halt. Whether to restore a hard pause is
    # an open human decision (docs/RISK_INVARIANTS.md, D3).
    _set_state(agent, recent_trades=[-1, -1, -1, -1, -1, -1, -1, -1, -1, 1])
    can, mult, reason = agent.pre_trade_check(10_000)
    assert can and mult == 0.25
    assert "wr" in reason.lower()


# ── Consecutive losses → pause ────────────────────────────────────────────────

def test_consecutive_losses_trigger_pause(agent):
    with patch("execution.risk_agent.STATE_FILE", agent.state):
        pass  # state already fresh
    agent.state.peak_equity       = 10_000
    agent.state.daily_start_equity  = 10_000
    agent.state.weekly_start_equity = 10_000
    agent.state.consecutive_losses = 0

    # Spec: graduated size reduction (3 losses -> 0.5x), pause at
    # max_consecutive_losses (7). Size never increases after a loss.
    for _ in range(3):
        agent.record_trade(-1.0, 9_900)
    can, mult, _ = agent.pre_trade_check(9_900)
    assert can and mult == 0.5

    for _ in range(4):
        agent.record_trade(-1.0, 9_900)
    can, _, reason = agent.pre_trade_check(9_900)
    assert not can
    assert "consecutive" in reason.lower() or "paused" in reason.lower()


def test_size_never_increases_with_losses(agent):
    agent.state.peak_equity = agent.state.daily_start_equity = agent.state.weekly_start_equity = 10_000
    prev = 1.0
    for _ in range(6):
        agent.record_trade(-1.0, 9_990)
        can, mult, _ = agent.pre_trade_check(9_990)
        assert mult <= prev
        prev = mult


def test_risk_config_clamped_to_invariants():
    from core.system_invariants import INVARIANTS
    cfg = RiskConfig(max_daily_loss_pct=0.5, max_concurrent_trades=99, max_account_dd_pct=0.5)
    assert cfg.max_daily_loss_pct == INVARIANTS.max_daily_loss_pct / 100
    assert cfg.max_concurrent_trades == INVARIANTS.max_concurrent_trades
    assert cfg.max_account_dd_pct == INVARIANTS.max_total_kill_pct / 100


def test_pause_expires(agent):
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _set_state(agent, pause_until=past,
               peak_equity=10_000,
               daily_start_equity=10_000,
               weekly_start_equity=10_000)
    can, _, _ = agent.pre_trade_check(10_000)
    assert can
    assert agent.state.pause_until is None


# ── Record trade updates peak equity ─────────────────────────────────────────

def test_record_trade_updates_peak(agent):
    agent.state.peak_equity = 10_000
    agent.record_trade(2.5, 10_200)
    assert agent.state.peak_equity == 10_200


def test_record_win_resets_consecutive_losses(agent):
    agent.state.consecutive_losses = 2
    agent.record_trade(1.5, 10_100)
    assert agent.state.consecutive_losses == 0
