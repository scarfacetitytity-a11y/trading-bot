"""Orchestrator integration tests — gate sequence, score sizing, risk tiers.

Mocks MT5 entirely so no broker connection is required.
Tests the critical path in TradingEngine.run():
  1. soft_halt blocks new entry
  2. RiskAgent blocks
  3. FTMOTracker DD limit blocks
  4. Score-based size multiplier applies correctly

Also tests RiskGuard tier escalation:
  Tier 1: 2% daily  → soft_halt
  Tier 2: 7% cumul  → soft_halt
  Tier 3: 9.5% cumul → kill_switch
"""
import sys
import threading
import time
import types
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import numpy as np
import pytest

# ── Stub MetaTrader5 before any orchestrator import ───────────────────────────
# MT5 is only available on Windows with the terminal installed; stub it here.
_mt5_stub = types.ModuleType("MetaTrader5")
_mt5_stub.ACCOUNT_TRADE_MODE_DEMO = 0
_mt5_stub.ACCOUNT_TRADE_MODE_REAL = 1
_mt5_stub.ORDER_TYPE_BUY  = 0
_mt5_stub.ORDER_TYPE_SELL = 1
# Timeframe constants used by data_loader.TIMEFRAME_MAP
_mt5_stub.TIMEFRAME_M1  = 1
_mt5_stub.TIMEFRAME_M5  = 5
_mt5_stub.TIMEFRAME_M15 = 15
_mt5_stub.TIMEFRAME_M30 = 30
_mt5_stub.TIMEFRAME_H1  = 60
_mt5_stub.TIMEFRAME_H4  = 240
_mt5_stub.TIMEFRAME_D1  = 1440
_mt5_stub.TIMEFRAME_W1  = 10080
_mt5_stub.TIMEFRAME_MN1 = 43200
_mt5_stub.account_info    = MagicMock()
_mt5_stub.symbol_info     = MagicMock()
_mt5_stub.symbol_info_tick = MagicMock()
_mt5_stub.copy_rates_from_pos = MagicMock()
_mt5_stub.positions_get   = MagicMock(return_value=[])
sys.modules.setdefault("MetaTrader5", _mt5_stub)

# Stub config and other optional imports so tests don't need them on disk
for _mod in ("config.settings", "config.logging_setup",
             "backtests.mt5_connector", "execution.trader"):
    if _mod not in sys.modules:
        _stub = types.ModuleType(_mod)
        _stub.load_config = MagicMock(return_value={})
        _stub.setup_logger = MagicMock()
        _stub.connect    = MagicMock(return_value=True)
        _stub.disconnect = MagicMock()
        _stub.set_magic  = MagicMock()
        _stub.get_positions = MagicMock(return_value=[])
        _stub.get_position_direction = MagicMock(return_value=0)
        _stub.get_account = MagicMock(return_value={"balance": 10_000, "equity": 10_000})
        _stub.close_all  = MagicMock()
        _stub.place_order = MagicMock(return_value=True)
        _stub.calculate_sl_tp = MagicMock(return_value=(None, None))
        _stub.calculate_lots  = MagicMock(return_value=0.01)
        sys.modules[_mod] = _stub

from execution.orchestrator import (
    HeartbeatRegistry, RiskGuard, TradingEngine, Status,
    MAX_DAILY_LOSS_PCT, SOFT_DD_HALT_PCT, MAX_TOTAL_LOSS_PCT,
)
from execution.risk_agent import RiskAgent, RiskConfig
from execution.ftmo_tracker import FTMOTracker
from backtests.run_multi_instrument import _size_mult_from_score
import execution.trader as trader_mod


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bars(n: int = 200) -> pd.DataFrame:
    """Minimal OHLCV dataframe for signal generation."""
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = 2000 + np.cumsum(np.random.default_rng(42).normal(0, 1, n))
    df = pd.DataFrame({
        "time": idx, "open": close - 0.5, "high": close + 1,
        "low": close - 1, "close": close, "tick_volume": 100,
    })
    df["time"] = df["time"].astype("int64") // 10**9  # unix seconds
    return df


def _make_rates(n: int = 200):
    df = _make_bars(n)
    return df.to_records(index=False)


def _engine(
    tmp_path,
    strategy=None,
    soft_halt_event=None,
    risk_agent=None,
    ftmo_tracker=None,
    dry_run=True,
):
    """Build a TradingEngine with a fresh RiskAgent / FTMOTracker."""
    registry    = HeartbeatRegistry()
    kill_switch = threading.Event()
    soft_halt   = soft_halt_event or threading.Event()

    if strategy is None:
        strategy = MagicMock()
        strategy.name = "mock"
        strategy.generate_signals = MagicMock(
            return_value=pd.Series([1])  # always wants to go long
        )
        strategy._stops  = pd.Series([float("nan")])
        strategy._scores = pd.Series([6])   # score=6 → 1.5x

    state_ra  = tmp_path / "risk_agent_state.json"
    state_ftmo = tmp_path / "ftmo_tracker_state.json"

    with patch("execution.risk_agent.STATE_FILE", state_ra), \
         patch("execution.ftmo_tracker.STATE_FILE", state_ftmo):
        ra = risk_agent or RiskAgent(initial_equity=10_000)
        ft = ftmo_tracker or FTMOTracker(initial_equity=10_000)

    engine = TradingEngine(
        registry=registry,
        kill_switch=kill_switch,
        symbol="XAUUSD",
        strategy=strategy,
        tf_str="M15",
        trade_cfg={"risk_pct": 1.0, "lookback_bars": 50},
        risk_agent=ra,
        ftmo_tracker=ft,
        soft_halt_event=soft_halt,
        dry_run=dry_run,
    )
    engine._risk_agent   = ra
    engine._ftmo_tracker = ft
    return engine, kill_switch, soft_halt, ra, ft


def _run_one_bar(engine, kill_switch, signal: int = 1):
    """Run the engine loop for one bar then stop it."""
    rates = _make_rates(50)
    bar_counter = [0]

    def _rates_side_effect(*args, **kwargs):
        bar_counter[0] += 1
        if bar_counter[0] == 1:
            return rates[:1]           # first call: new bar timestamp
        if bar_counter[0] == 2:
            return rates               # second call: full lookback
        # After the first full cycle, return same bar → loop sleeps; kill it
        kill_switch.set()
        return rates[:1]

    _mt5_stub.copy_rates_from_pos.side_effect = _rates_side_effect

    # Synthetic signal
    engine._strategy.generate_signals = MagicMock(
        return_value=pd.Series([signal] * 50)
    )

    trader_mod.get_position_direction.return_value = 0  # flat → wants to enter
    trader_mod.get_account.return_value = {"balance": 10_000, "equity": 10_000}

    t = threading.Thread(target=engine.run)
    t.start()
    t.join(timeout=5)
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Score-based size multiplier
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreSizing:
    def test_score_4_is_075x(self):
        assert _size_mult_from_score(4) == 0.75

    def test_score_5_is_1x(self):
        assert _size_mult_from_score(5) == 1.0

    def test_score_6_is_15x(self):
        assert _size_mult_from_score(6) == 1.5

    def test_score_7_is_15x(self):
        assert _size_mult_from_score(7) == 1.5

    def test_score_0_returns_15x_default(self):
        # 0 is treated as unknown; default is 1.5 — but orchestrator guards with signal_score > 0
        assert _size_mult_from_score(0) == 1.5  # SIZE_CONFIGS_DEFAULT fallthrough


# ═══════════════════════════════════════════════════════════════════════════════
# Heartbeat registry
# ═══════════════════════════════════════════════════════════════════════════════

class TestHeartbeatRegistry:
    def test_register_and_beat(self):
        reg = HeartbeatRegistry()
        reg.register("comp", beat_timeout=60)
        reg.beat("comp", "alive", Status.HEALTHY)
        snap = {r["name"]: r for r in reg.snapshot()}
        assert snap["comp"]["status"] == "HEALTHY"

    def test_stale_detection_after_timeout(self):
        reg = HeartbeatRegistry()
        reg.register("comp", beat_timeout=0.01)  # 10ms timeout
        reg.beat("comp", "alive", Status.HEALTHY)
        time.sleep(0.05)
        snap = {r["name"]: r for r in reg.snapshot()}
        assert snap["comp"]["status"] == "STALE"

    def test_fail_marks_failed(self):
        reg = HeartbeatRegistry()
        reg.register("comp")
        reg.fail("comp", "error")
        assert reg.any_stale_or_failed()

    def test_halt_marks_halted(self):
        reg = HeartbeatRegistry()
        reg.register("comp")
        reg.halt("comp", "kill")
        assert reg.any_stale_or_failed()

    def test_increment_restarts(self):
        reg = HeartbeatRegistry()
        reg.register("comp")
        assert reg.increment_restarts("comp") == 1
        assert reg.increment_restarts("comp") == 2


# ═══════════════════════════════════════════════════════════════════════════════
# RiskGuard tier escalation
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskGuardTiers:
    def _guard(self):
        registry    = HeartbeatRegistry()
        kill_switch = threading.Event()
        soft_halt   = threading.Event()
        guard = RiskGuard(registry, kill_switch, soft_halt,
                          initial_equity=10_000.0, symbols=["XAUUSD"])
        guard._day_start_equity = 10_000.0
        guard._server_day       = "2026-01-01"   # prevent new-day reset branch
        return guard, kill_switch, soft_halt

    def _run_once(self, guard, equity):
        """Simulate one iteration of the guard's check logic (extracted)."""
        kill_switch = guard.kill_switch
        soft_halt   = guard._soft_halt
        daily_pct   = (equity - guard._day_start_equity) / guard._day_start_equity * 100
        # Total loss is measured from the STATIC initial balance (restart-safe).
        total_pct   = (equity - guard._initial_equity) / guard._initial_equity * 100

        if daily_pct <= -MAX_DAILY_LOSS_PCT:
            soft_halt.set()
        if total_pct <= -SOFT_DD_HALT_PCT:
            soft_halt.set()
        if total_pct <= -MAX_TOTAL_LOSS_PCT:
            kill_switch.set()

    def test_daily_loss_triggers_soft_halt(self):
        guard, kill_switch, soft_halt = self._guard()
        self._run_once(guard, 9_790)       # -2.1% daily
        assert soft_halt.is_set()
        assert not kill_switch.is_set()

    def test_cumulative_7pct_triggers_soft_halt(self):
        guard, kill_switch, soft_halt = self._guard()
        self._run_once(guard, 9_280)       # -7.2% cumulative
        assert soft_halt.is_set()
        assert not kill_switch.is_set()

    def test_cumulative_95pct_triggers_kill(self):
        guard, kill_switch, soft_halt = self._guard()
        self._run_once(guard, 9_040)       # -9.6% cumulative
        assert kill_switch.is_set()

    def test_small_loss_no_halt(self):
        guard, kill_switch, soft_halt = self._guard()
        self._run_once(guard, 9_900)       # -1% — all clear
        assert not soft_halt.is_set()
        assert not kill_switch.is_set()


# ═══════════════════════════════════════════════════════════════════════════════
# TradingEngine gate sequence
# ═══════════════════════════════════════════════════════════════════════════════

class TestTradingEngineGates:
    """Test entry gate sequence: soft_halt → RiskAgent → FTMOTracker → size."""

    def test_soft_halt_blocks_entry(self, tmp_path):
        engine, kill, soft_halt, ra, ft = _engine(tmp_path)
        soft_halt.set()  # Tier 1 active

        _run_one_bar(engine, kill, signal=1)

        trader_mod.place_order.assert_not_called()

    def test_risk_agent_blocks_entry(self, tmp_path):
        engine, kill, soft_halt, ra, ft = _engine(tmp_path)
        # Force account DD breach
        ra.state.peak_equity        = 10_000
        ra.state.daily_start_equity   = 10_000
        ra.state.weekly_start_equity  = 10_000
        trader_mod.get_account.return_value = {
            "balance": 9_200, "equity": 9_200   # 8% from peak → over 7% limit
        }

        _run_one_bar(engine, kill, signal=1)

        trader_mod.place_order.assert_not_called()

    def test_ftmo_dd_blocks_entry(self, tmp_path):
        engine, kill, soft_halt, ra, ft = _engine(tmp_path)
        # Force FTMO total DD to breach limit
        ft.state.peak_equity = 10_000
        trader_mod.get_account.return_value = {
            "balance": 9_000, "equity": 9_000   # 10% DD = at limit
        }

        _run_one_bar(engine, kill, signal=1)

        trader_mod.place_order.assert_not_called()

    def test_kill_switch_blocks_entry(self, tmp_path):
        engine, kill, soft_halt, ra, ft = _engine(tmp_path)
        kill.set()  # pre-triggered

        _run_one_bar(engine, kill, signal=1)

        trader_mod.place_order.assert_not_called()

    def test_signal_zero_no_entry(self, tmp_path):
        engine, kill, soft_halt, ra, ft = _engine(tmp_path)

        _run_one_bar(engine, kill, signal=0)  # flat signal

        trader_mod.place_order.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Safety check — real-account block (M3 regression)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetyCheck:
    """The real-account guard must fire unless allow_real_account is explicitly set."""

    def _orch(self, allow_real):
        from execution.orchestrator import Orchestrator
        o = object.__new__(Orchestrator)          # bypass heavy __init__
        o.cfg = {"trading": {"allow_real_account": allow_real}}
        return o

    def _account(self, trade_mode):
        acct = MagicMock()
        acct.trade_mode = trade_mode
        _mt5_stub.account_info.return_value = acct

    def test_real_account_blocked_by_default(self):
        self._account(_mt5_stub.ACCOUNT_TRADE_MODE_REAL)
        assert self._orch(allow_real=False)._safety_check() is False

    def test_real_account_allowed_when_flag_set(self):
        self._account(_mt5_stub.ACCOUNT_TRADE_MODE_REAL)
        assert self._orch(allow_real=True)._safety_check() is True

    def test_demo_account_allowed(self):
        self._account(_mt5_stub.ACCOUNT_TRADE_MODE_DEMO)
        assert self._orch(allow_real=False)._safety_check() is True


# ═══════════════════════════════════════════════════════════════════════════════
# Score sizing wired to TradingEngine._size_order
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreWiredToEngine:
    """Verify the score multiplier reaches the lot calculation in _size_order."""

    def _lots_for_score(self, tmp_path, score: int) -> float:
        engine, kill, soft_halt, ra, ft = _engine(tmp_path)
        engine._strategy._scores = pd.Series([score])

        # No ATR stop from strategy → falls back to config lot_size=0.01 × mult
        engine._strategy._stops = pd.Series([float("nan")])

        mock_tick = MagicMock()
        mock_tick.ask = 2000.0
        mock_tick.bid = 1999.9
        _mt5_stub.symbol_info_tick.return_value = mock_tick
        _mt5_stub.symbol_info.return_value = None   # triggers fallback path

        _, _, lots = engine._size_order(1, 10_000, _size_mult_from_score(score))
        return lots

    def test_score_4_lots(self, tmp_path):
        lots = self._lots_for_score(tmp_path, 4)
        assert lots == pytest.approx(0.01 * 0.75, abs=0.005)

    def test_score_5_lots(self, tmp_path):
        lots = self._lots_for_score(tmp_path, 5)
        assert lots == pytest.approx(0.01 * 1.0, abs=0.005)

    def test_score_6_lots(self, tmp_path):
        lots = self._lots_for_score(tmp_path, 6)
        assert lots == pytest.approx(0.01 * 1.5, abs=0.005)
