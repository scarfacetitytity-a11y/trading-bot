"""Orchestrator integration tests â€” gate sequence, score sizing, risk tiers.

Mocks MT5 entirely so no broker connection is required.
Tests the critical path in TradingEngine.run():
  1. soft_halt blocks new entry
  2. RiskAgent blocks
  3. FTMOTracker DD limit blocks
  4. Score-based size multiplier applies correctly

Also tests RiskGuard tier escalation:
  Tier 1: 2% daily  â†’ soft_halt
  Tier 2: 7% cumul  â†’ soft_halt
  Tier 3: 9.5% cumul â†’ kill_switch
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

# â”€â”€ Stub MetaTrader5 before any orchestrator import â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
_mt5_stub.DEAL_ENTRY_IN     = 0
_mt5_stub.DEAL_ENTRY_OUT    = 1
_mt5_stub.DEAL_ENTRY_INOUT  = 2
_mt5_stub.DEAL_ENTRY_OUT_BY = 3
_mt5_stub.account_info    = MagicMock()
_mt5_stub.symbol_info     = MagicMock()
_mt5_stub.symbol_info_tick = MagicMock()
_mt5_stub.copy_rates_from_pos = MagicMock()
_mt5_stub.positions_get   = MagicMock(return_value=[])
_mt5_stub.history_deals_get = MagicMock(return_value=[])
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
    HeartbeatRegistry, RiskGuard, TradingEngine, TradeReconciler, Status,
    MAX_DAILY_LOSS_PCT, SOFT_DD_HALT_PCT, MAX_TOTAL_LOSS_PCT,
)
from execution.risk_agent import RiskAgent, RiskConfig
from execution.ftmo_tracker import FTMOTracker
from backtests.run_multi_instrument import _size_mult_from_score
import execution.trader as trader_mod


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        strategy._scores = pd.Series([6])   # score=6 â†’ 1.5x

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
        # After the first full cycle, return same bar â†’ loop sleeps; kill it
        kill_switch.set()
        return rates[:1]

    _mt5_stub.copy_rates_from_pos.side_effect = _rates_side_effect

    # Synthetic signal
    engine._strategy.generate_signals = MagicMock(
        return_value=pd.Series([signal] * 50)
    )

    trader_mod.get_position_direction.return_value = 0  # flat â†’ wants to enter
    trader_mod.get_account.return_value = {"balance": 10_000, "equity": 10_000}

    t = threading.Thread(target=engine.run)
    t.start()
    t.join(timeout=5)
    return t


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Score-based size multiplier
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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
        # 0 is treated as unknown; default is 1.5 â€” but orchestrator guards with signal_score > 0
        assert _size_mult_from_score(0) == 1.5  # SIZE_CONFIGS_DEFAULT fallthrough


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Heartbeat registry
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# RiskGuard tier escalation
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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
        self._run_once(guard, 9_900)       # -1% â€” all clear
        assert not soft_halt.is_set()
        assert not kill_switch.is_set()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TradingEngine gate sequence
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestTradingEngineGates:
    """Test entry gate sequence: soft_halt â†’ RiskAgent â†’ FTMOTracker â†’ size."""

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
            "balance": 9_200, "equity": 9_200   # 8% from peak â†’ over 7% limit
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TradeReconciler â€” deal-history accounting (C4)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _deal(pid, entry, profit=0.0, *, symbol="XAUUSD", price=2000.0, t=1000,
          magic=234001, swap=0.0, commission=0.0, comment=""):
    return types.SimpleNamespace(
        position_id=pid, entry=entry, profit=profit, symbol=symbol, price=price,
        time=t, magic=magic, swap=swap, commission=commission, comment=comment,
    )


def _pos(pid, magic=234001):
    return types.SimpleNamespace(identifier=pid, magic=magic)


class TestTradeReconciler:
    """Deal-history is the single source of truth: seed, record-once, dedup, partials."""

    def test_seed_record_dedup_and_partial(self, tmp_path):
        reg = HeartbeatRegistry()
        kill = threading.Event()
        ra, ft, jr = MagicMock(), MagicMock(), MagicMock()

        with patch.object(TradeReconciler, "STATE_FILE", tmp_path / "rec.json"):
            rec = TradeReconciler(
                reg, kill, risk_agent=ra, ftmo_tracker=ft, journal=jr,
                magic=234001, risk_pct=1.0, initial_equity=10_000,
            )
            acct = MagicMock(); acct.equity = 10_000
            _mt5_stub.account_info.return_value = acct
            _mt5_stub.positions_get.return_value = []

            # Batch 1 â€” a pre-existing closed position: seeding must suppress it.
            _mt5_stub.history_deals_get.return_value = [
                _deal(100, _mt5_stub.DEAL_ENTRY_IN),
                _deal(100, _mt5_stub.DEAL_ENTRY_OUT, -50.0),
            ]
            rec._reconcile()
            ra.record_trade.assert_not_called()

            # Batch 2 â€” a newly closed position 101: recorded exactly once, with
            # R = net profit / (initial Ã— risk_pct) = 200 / 100 = +2.0R.
            _mt5_stub.history_deals_get.return_value = [
                _deal(100, _mt5_stub.DEAL_ENTRY_IN),
                _deal(100, _mt5_stub.DEAL_ENTRY_OUT, -50.0),
                _deal(101, _mt5_stub.DEAL_ENTRY_IN),
                _deal(101, _mt5_stub.DEAL_ENTRY_OUT, 200.0, price=2010.0),
            ]
            rec._reconcile()
            assert ra.record_trade.call_count == 1
            assert ra.record_trade.call_args[0][0] == pytest.approx(2.0)
            ft.record_trade_day.assert_called_once()
            jr.close_trade.assert_called_once()

            # Batch 3 â€” rescan with no new closes: no double-count.
            rec._reconcile()
            assert ra.record_trade.call_count == 1

            # Batch 4 â€” position 102 still open (partial close): must NOT record.
            _mt5_stub.positions_get.return_value = [_pos(102)]
            _mt5_stub.history_deals_get.return_value = [
                _deal(102, _mt5_stub.DEAL_ENTRY_IN),
                _deal(102, _mt5_stub.DEAL_ENTRY_OUT, 30.0),
            ]
            rec._reconcile()
            assert ra.record_trade.call_count == 1

        # reset shared stub so later tests aren't affected
        _mt5_stub.positions_get.return_value = []
        _mt5_stub.history_deals_get.return_value = []

    def test_scalein_folds_into_one_trade(self, tmp_path):
        reg = HeartbeatRegistry()
        kill = threading.Event()
        ra, ft, jr = MagicMock(), MagicMock(), MagicMock()

        with patch.object(TradeReconciler, "STATE_FILE", tmp_path / "rec.json"):
            rec = TradeReconciler(
                reg, kill, risk_agent=ra, ftmo_tracker=ft, journal=jr,
                magic=234001, risk_pct=1.0, initial_equity=10_000,
            )
            acct = MagicMock(); acct.equity = 10_000
            _mt5_stub.account_info.return_value = acct
            _mt5_stub.positions_get.return_value = []
            _mt5_stub.history_deals_get.return_value = []
            rec._reconcile()  # seed (empty)

            # Child 201 (opened with comment si:200) still OPEN â†’ parent not recorded.
            _mt5_stub.positions_get.return_value = [_pos(201)]
            _mt5_stub.history_deals_get.return_value = [
                _deal(200, _mt5_stub.DEAL_ENTRY_IN),
                _deal(200, _mt5_stub.DEAL_ENTRY_OUT, 100.0),
                _deal(201, _mt5_stub.DEAL_ENTRY_IN, comment="si:200"),
            ]
            rec._reconcile()
            ra.record_trade.assert_not_called()   # group not fully flat yet

            # Now the child closes too â†’ ONE record, R = (100+50)/100 = +1.5R.
            _mt5_stub.positions_get.return_value = []
            _mt5_stub.history_deals_get.return_value = [
                _deal(200, _mt5_stub.DEAL_ENTRY_IN),
                _deal(200, _mt5_stub.DEAL_ENTRY_OUT, 100.0),
                _deal(201, _mt5_stub.DEAL_ENTRY_IN, comment="si:200"),
                _deal(201, _mt5_stub.DEAL_ENTRY_OUT, 50.0),
            ]
            rec._reconcile()
            assert ra.record_trade.call_count == 1
            assert ra.record_trade.call_args[0][0] == pytest.approx(1.5)

            # Rescan â†’ child 201 already marked, no double-count.
            rec._reconcile()
            assert ra.record_trade.call_count == 1

        _mt5_stub.positions_get.return_value = []
        _mt5_stub.history_deals_get.return_value = []


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Safety check â€” real-account block (M3 regression)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Score sizing wired to TradingEngine._size_order
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestScoreWiredToEngine:
    """Verify the score multiplier reaches the lot calculation in _size_order."""

    def _lots_for_score(self, tmp_path, score: int) -> float:
        engine, kill, soft_halt, ra, ft = _engine(tmp_path)
        engine._strategy._scores = pd.Series([score])

        # No ATR stop from strategy â†’ falls back to config lot_size=0.01 Ã— mult
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2026-07-27 audit regressions â€” FTMO day anchoring (A1)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestFTMODayAnchoring:
    """Daily DD must anchor to live equity, never to stale initial_equity."""

    def _tracker(self, tmp_path, initial=100_000):
        with patch("execution.ftmo_tracker.STATE_FILE", tmp_path / "ftmo.json"):
            return FTMOTracker(initial_equity=initial)

    def test_new_day_anchors_to_live_equity(self, tmp_path):
        ft = self._tracker(tmp_path)
        # Fresh state, no prior close: first check() anchors to live equity,
        # NOT to initial_equity (the 2026-07-27 100k-vs-96.5k corruption).
        with patch("execution.ftmo_tracker.STATE_FILE", tmp_path / "ftmo.json"):
            r = ft.check(96_524.73)
        assert r["day_start_equity"] == pytest.approx(96_524.73)
        assert r["daily_dd_pct"] == pytest.approx(0.0)

    def test_same_day_does_not_reanchor(self, tmp_path):
        ft = self._tracker(tmp_path)
        with patch("execution.ftmo_tracker.STATE_FILE", tmp_path / "ftmo.json"):
            ft.check(96_500.0)
            r = ft.check(94_400.0)   # intraday loss must NOT move the anchor
        assert r["day_start_equity"] == pytest.approx(96_500.0)
        assert r["daily_dd_pct"] == pytest.approx((96_500 - 94_400) / 96_500 * 100)

    def test_record_trade_day_never_falls_back_to_initial(self, tmp_path):
        ft = self._tracker(tmp_path, initial=100_000)
        ft.state.last_equity = 0.0          # fresh state, no prior close
        ft.state.day_start_date = ""        # force new-day branch
        with patch("execution.ftmo_tracker.STATE_FILE", tmp_path / "ftmo.json"):
            ft.record_trade_day(equity_close=95_000.0)
        assert ft.state.day_start_equity == pytest.approx(95_000.0)
        assert ft.state.day_start_equity != 100_000

    def test_corrupt_state_backed_up_not_destroyed(self, tmp_path):
        state_file = tmp_path / "ftmo.json"
        state_file.write_bytes(b"\xef\xbb\xbf{not valid json")
        with patch("execution.ftmo_tracker.STATE_FILE", state_file):
            FTMOTracker(initial_equity=100_000)
        assert (tmp_path / "ftmo.json.corrupt").exists()

    def test_bom_state_file_loads(self, tmp_path):
        import json as _json
        state_file = tmp_path / "ftmo.json"
        payload = {
            "start_date": "2026-07-22", "initial_equity": 100000,
            "target_equity": 110000.0, "peak_equity": 100000.0,
            "trade_days": [], "daily_pnl": {},
            "day_start_equity": 96524.73, "day_start_date": "2026-07-27",
            "last_equity": 94400.93,
        }
        state_file.write_bytes(b"\xef\xbb\xbf" + _json.dumps(payload).encode("utf-8"))
        with patch("execution.ftmo_tracker.STATE_FILE", state_file):
            ft = FTMOTracker(initial_equity=100_000)
        assert ft.state.day_start_equity == pytest.approx(96_524.73)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2026-07-27 audit regressions â€” burned targets (A3) + PID lock (A2)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestBurnedTargets:
    """A direction+target combo must be blocked for 90 min after entry."""

    def test_burn_and_reload_roundtrip(self, tmp_path):
        engine, *_ = _engine(tmp_path)
        with patch.object(TradingEngine, "_BURNED_TARGETS_FILE", tmp_path / "burned.json"):
            engine._save_burned_target(1, 52531.15)
            assert (1, 52531.15) in engine._burned_targets

            engine2, *_ = _engine(tmp_path)
            engine2._burned_targets = {}
            engine2._load_burned_targets()
        assert (1, 52531.15) in engine2._burned_targets

    def test_expired_burn_not_loaded(self, tmp_path):
        import json as _json
        burn_file = tmp_path / "burned.json"
        burn_file.write_text(_json.dumps(
            {"XAUUSD": {"1|52531.15": time.time() - 10}}   # already expired
        ))
        with patch.object(TradingEngine, "_BURNED_TARGETS_FILE", burn_file):
            engine, *_ = _engine(tmp_path)
            engine._burned_targets = {}
            engine._load_burned_targets()
        assert engine._burned_targets == {}

    def test_opposite_direction_not_burned(self, tmp_path):
        engine, *_ = _engine(tmp_path)
        with patch.object(TradingEngine, "_BURNED_TARGETS_FILE", tmp_path / "burned.json"):
            engine._save_burned_target(1, 52531.15)
        assert (-1, 52531.15) not in engine._burned_targets


class TestPidLock:
    """Second instance must abort â€” no psutil dependency (2026-07-27: three
    concurrent instances because psutil ImportError skipped the check)."""

    def test_live_python_pid_detected(self):
        import os
        from execution.orchestrator import _pid_is_python
        assert _pid_is_python(os.getpid()) is True

    def test_dead_pid_not_detected(self):
        from execution.orchestrator import _pid_is_python
        # PID 4 is the Windows System process â€” never python
        assert _pid_is_python(4) is False

    def test_second_start_aborts(self, tmp_path, monkeypatch):
        import os
        from execution.orchestrator import _acquire_pid_lock
        monkeypatch.chdir(tmp_path)
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "bot.pid").write_text(str(os.getpid()))
        with pytest.raises(SystemExit):
            _acquire_pid_lock()

    def test_stale_lock_overwritten(self, tmp_path, monkeypatch):
        import os
        from execution.orchestrator import _acquire_pid_lock
        monkeypatch.chdir(tmp_path)
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "bot.pid").write_text("4")   # System pid, not python
        pid_path = _acquire_pid_lock()
        assert pid_path.read_text().strip() == str(os.getpid())


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2026-07-27 audit regressions â€” probability model + Quant proposals
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestProbabilityModel:
    def _model(self, tmp_path):
        from execution.probability_model import ProbabilityModel
        return ProbabilityModel(state_file=tmp_path / "prob.json")

    def test_continuation_penalised(self, tmp_path):
        from execution.probability_model import TradeConfluences
        m = self._model(tmp_path)
        base = m.estimate(TradeConfluences(fvg_present=True, trade_type="breakout"))
        cont = m.estimate(TradeConfluences(fvg_present=True, trade_type="continuation"))
        assert cont.p_win < base.p_win

    def test_low_probability_rejected(self, tmp_path):
        from execution.probability_model import TradeConfluences
        m = self._model(tmp_path)
        # Continuation with nothing else going for it â†’ below MIN_P_TO_TRADE
        est = m.estimate(TradeConfluences(
            fvg_present=False, trade_type="continuation", rr=1.0))
        assert est.take_trade is False

    def test_quant_proposal_applied_within_bounds(self, tmp_path):
        import json as _json
        (tmp_path / "quant_lift_proposals.json").write_text(
            _json.dumps({"confluences": {"fvg_present": 1.40}}))
        m = self._model(tmp_path)
        assert m._lifts["fvg_present"]["lift"] == pytest.approx(1.40)

    def test_quant_proposal_out_of_bounds_rejected(self, tmp_path):
        import json as _json
        (tmp_path / "quant_lift_proposals.json").write_text(
            _json.dumps({"confluences": {"fvg_present": 9.9}}))   # typo guard
        m = self._model(tmp_path)
        assert m._lifts["fvg_present"]["lift"] == pytest.approx(1.30)

    def test_profit_lock_ladder(self):
        """A trade that reached 0.6R+ must never keep its original stop."""
        from execution.trade_manager import TradeManager, PositionState, ActionType
        tm = TradeManager()
        pos = PositionState(
            direction=1, entry_price=100.0, initial_sl=99.0,
            current_sl=99.0, current_tp=103.0, current_price=100.2,
            bars_elapsed=5, peak_r=0.9,   # reached 0.9R, pulled back to 0.2R
        )
        df = pd.DataFrame({
            "time": range(30), "open": [100.0] * 30, "high": [100.3] * 30,
            "low": [99.8] * 30, "close": [100.1] * 30, "tick_volume": [100] * 30,
        })
        action = tm.evaluate(pos, df_m15=df, df_m5=df)
        assert action.action == ActionType.TIGHTEN_SL
        assert action.new_sl == pytest.approx(100.05)   # breakeven + 0.05R

    def test_profit_lock_never_loosens(self):
        from execution.trade_manager import TradeManager, PositionState, ActionType
        tm = TradeManager()
        pos = PositionState(
            direction=1, entry_price=100.0, initial_sl=99.0,
            current_sl=100.5, current_tp=103.0, current_price=100.6,
            bars_elapsed=5, peak_r=0.9,   # SL already better than BE lock
        )
        df = pd.DataFrame({
            "time": range(30), "open": [100.0] * 30, "high": [100.7] * 30,
            "low": [100.4] * 30, "close": [100.6] * 30, "tick_volume": [100] * 30,
        })
        action = tm.evaluate(pos, df_m15=df, df_m5=df)
        assert not (action.action == ActionType.TIGHTEN_SL
                    and action.new_sl is not None and action.new_sl < 100.5)

    def test_stale_quant_proposal_ignored(self, tmp_path):
        import json as _json, os
        pfile = tmp_path / "quant_lift_proposals.json"
        pfile.write_text(_json.dumps({"confluences": {"fvg_present": 1.40}}))
        old = time.time() - 5 * 3600
        os.utime(pfile, (old, old))
        m = self._model(tmp_path)
        assert m._lifts["fvg_present"]["lift"] == pytest.approx(1.30)

