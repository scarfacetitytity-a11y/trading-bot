"""Orchestrator — the floor manager of the trading bot.

Single governing process that owns, monitors, and supervises every component:

    Orchestrator
    ├── MT5Monitor      — heartbeat on the broker connection
    ├── DataWatcher     — confirms each symbol receives bars on schedule
    ├── SignalEngine     — runs strategy, produces desired positions
    ├── ExecutionEngine  — executes orders when signal changes
    └── RiskGuard       — enforces FTMO loss limits; triggers kill switch

All components register with the HeartbeatRegistry. The Orchestrator's
monitor loop checks every component every 60 s. If a component hasn't
beaten within its timeout it is flagged STALE → FAILED, restarted with
exponential back-off, and the issue is logged.

A global kill_switch Event halts all trading immediately when:
  - Daily loss exceeds MAX_DAILY_LOSS_PCT of starting equity
  - Total drawdown exceeds MAX_TOTAL_LOSS_PCT of starting equity
  - MT5 connection cannot be recovered after MAX_RECONNECT_TRIES attempts
  - Any component exceeds MAX_RESTARTS before recovering

Usage:
    python -m execution.orchestrator
    python -m execution.orchestrator --dry-run
    python -m execution.orchestrator --symbol XAUUSD --dry-run
"""
import argparse
import json
import logging
import math
import sys
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import MetaTrader5 as mt5
import pandas as pd

from config.settings import load_config
from config.logging_setup import setup_logger
from backtests.mt5_connector import connect, disconnect
from backtests.data_loader import TIMEFRAME_MAP
from execution import trader, risk
from execution import news_gate as ng
from execution.trade_manager import TradeManager, PositionState, ActionType
from execution.signal_detectors import (
    detect_m5_entry_trigger, detect_accumulation, detect_liquidity_draw,
)
from execution.trade_analyzer import analyze_entry, manage_trade as analyze_manage
from execution.trade_agent import TradeAgent
from strategies.sniper import SniperStrategy
from strategies.london_breakout import LondonBreakoutStrategy
from strategies.fvg_ob import FVGOrderBlockStrategy
from strategies.aiden_index import AiDENIndexStrategy
from execution.risk_agent import RiskAgent, RiskConfig
from execution.ftmo_tracker import FTMOTracker
from execution.trade_journal import TradeJournal
from execution.portfolio_manager import PortfolioAllocator, PortfolioBook, OpenPos, Trim
from execution.level_monitor import LevelMonitor
from execution import telegram_notify as tg
from backtests.run_multi_instrument import (
    INSTRUMENTS, OPTIMISED_PARAMS, TRAIL_CONFIGS, BIDIRECTIONAL, M15_PARAMS,
    _size_mult_from_score,
)

logger = logging.getLogger(__name__)

# ── Risk limits (Council of 12 — tighter than FTMO stated limits) ────────────
MAX_DAILY_LOSS_PCT  = 2.0   # circuit breaker: halt all new entries (FTMO limit 5%)
SOFT_DD_HALT_PCT    = 7.0   # soft halt: no new entries, let open trades run (FTMO limit 10%)
MAX_TOTAL_LOSS_PCT  = 9.5   # hard kill switch: emergency close all (FTMO 10% hard floor)
MAX_RECONNECT_TRIES = 5
MAX_RESTARTS        = 10
MONITOR_INTERVAL    = 60    # seconds between orchestrator health checks
# ─────────────────────────────────────────────────────────────────────────────

_TF_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}

# Instruments that move together — used for divergence gate.
# If any member has an active position in OPPOSITE direction, block new entry.
_CORR_GROUPS: list[set] = [
    {"US30.cash", "US100.cash", "US500.cash", "US2000.cash"},  # US indices
    {"UK100.cash", "GER40.cash"},                               # EU indices
    {"XAUUSD", "XAGUSD"},                                       # metals
    {"GBPUSD", "EURUSD"},                                       # DXY-driven forex pairs (inverse)
    {"USDJPY"},                                                  # JPY pair — BOJ-sensitive, standalone
]

# Max concurrent SAME-direction positions within a correlation group.
# Monte Carlo autopsy: correlated same-dir clustering is the #1 blow-up cause
# (worst day: 17 stacked losers = -17R). Cap of 3 strips clustered losers while
# keeping winners — lowers blow rate AND slightly raises pass rate.
MAX_SAME_DIR_CLUSTER = 3

STRATEGY_MAP = {
    "sniper":          SniperStrategy,
    "london_breakout": LondonBreakoutStrategy,
    "fvg_ob":          FVGOrderBlockStrategy,
    "aiden_index":     AiDENIndexStrategy,
}


def _build_aiden_strategy(symbol: str) -> AiDENIndexStrategy:
    """Build a per-symbol AiDENIndexStrategy with correct live params."""
    cfg   = INSTRUMENTS.get(symbol, {})
    tf_p  = M15_PARAMS.copy()
    v2    = dict(
        long_only=(symbol not in BIDIRECTIONAL),
        rr_model3_bonus=0.5, rr_trend_bonus=0.5, rr_trend_threshold=0.003, rr_max=5.0,
        session_prime_start=13, session_prime_end=15,
        use_rsi=True, rsi_period=56,
        rsi_long_lo=25.0, rsi_long_hi=55.0, rsi_short_lo=45.0, rsi_short_hi=75.0,
        trail_to_be=symbol in TRAIL_CONFIGS,
        trail_be_r=TRAIL_CONFIGS[symbol].get("trail_be_r", 1.0) if symbol in TRAIL_CONFIGS else 1.0,
        trail_lock_r=TRAIL_CONFIGS[symbol].get("trail_lock_r", 2.0) if symbol in TRAIL_CONFIGS else 2.0,
        t1_r=TRAIL_CONFIGS[symbol].get("t1_r", 0.0) if symbol in TRAIL_CONFIGS else 0.0,
        t1_partial_pct=TRAIL_CONFIGS[symbol].get("t1_partial_pct", 0.5) if symbol in TRAIL_CONFIGS else 0.5,
        time_stop_bars=TRAIL_CONFIGS[symbol].get("time_stop_bars", 0) if symbol in TRAIL_CONFIGS else 0,
    )
    if symbol in OPTIMISED_PARAMS:
        opt  = OPTIMISED_PARAMS[symbol].copy()
        bias = opt.pop("h4_bias_method", tf_p.pop("h4_bias_method", "ema"))
        stop = opt.pop("atr_stop_buffer", tf_p.pop("atr_stop_buffer", 0.5))
        return AiDENIndexStrategy(
            min_score=opt.get("min_score", 4),
            min_fvg_atr=opt.get("min_fvg_atr", 0.10),
            rr_target=opt.get("rr_target", 2.5),
            session_start=opt.get("session_start", cfg.get("session_start", 7)),
            session_end=opt.get("session_end", cfg.get("session_end", 21)),
            h4_bias_method=bias, atr_stop_buffer=stop,
            **{k: v for k, v in tf_p.items() if k not in ("h4_bias_method", "atr_stop_buffer")},
            **v2,
        )
    bias = tf_p.pop("h4_bias_method", "ema")
    stop = tf_p.pop("atr_stop_buffer", 0.5)
    return AiDENIndexStrategy(
        min_score=4, min_fvg_atr=0.10, rr_target=2.5,
        session_start=cfg.get("session_start", 7),
        session_end=cfg.get("session_end", 21),
        h4_bias_method=bias, atr_stop_buffer=stop,
        **tf_p, **v2,
    )


# ── Status enum ──────────────────────────────────────────────────────────────

class Status(Enum):
    STARTING = "STARTING"
    HEALTHY  = "HEALTHY"
    STALE    = "STALE"
    FAILED   = "FAILED"
    HALTED   = "HALTED"


# ── Heartbeat registry ───────────────────────────────────────────────────────

@dataclass
class ComponentRecord:
    name:       str
    status:     Status   = Status.STARTING
    last_beat:  float    = field(default_factory=time.time)
    message:    str      = ""
    restarts:   int      = 0
    error_count: int     = 0
    beat_timeout: float  = 120.0  # seconds before flagged STALE


class HeartbeatRegistry:
    """Thread-safe registry of all component heartbeats."""

    def __init__(self):
        self._lock    = threading.Lock()
        self._records: Dict[str, ComponentRecord] = {}

    def register(self, name: str, beat_timeout: float = 120.0) -> None:
        with self._lock:
            self._records[name] = ComponentRecord(name=name, beat_timeout=beat_timeout)

    def beat(self, name: str, message: str = "", status: Status = Status.HEALTHY) -> None:
        with self._lock:
            if name in self._records:
                r = self._records[name]
                r.last_beat = time.time()
                r.status    = status
                r.message   = message

    def fail(self, name: str, message: str = "") -> None:
        with self._lock:
            if name in self._records:
                r = self._records[name]
                r.status      = Status.FAILED
                r.message     = message
                r.error_count += 1

    def halt(self, name: str, message: str = "") -> None:
        with self._lock:
            if name in self._records:
                r = self._records[name]
                r.status  = Status.HALTED
                r.message = message

    def increment_restarts(self, name: str) -> int:
        with self._lock:
            if name in self._records:
                self._records[name].restarts += 1
                return self._records[name].restarts
        return 0

    def snapshot(self) -> list:
        with self._lock:
            now = time.time()
            rows = []
            for r in self._records.values():
                age = now - r.last_beat
                effective_status = r.status
                if r.status == Status.HEALTHY and age > r.beat_timeout:
                    effective_status = Status.STALE
                rows.append({
                    "name":     r.name,
                    "status":   effective_status.value,
                    "last_beat_secs": int(age),
                    "message":  r.message,
                    "restarts": r.restarts,
                    "errors":   r.error_count,
                })
            return rows

    def any_stale_or_failed(self) -> bool:
        now = time.time()
        with self._lock:
            for r in self._records.values():
                if r.status in (Status.FAILED, Status.HALTED):
                    return True
                if r.status == Status.HEALTHY and (now - r.last_beat) > r.beat_timeout:
                    return True
        return False


# ── Base component ────────────────────────────────────────────────────────────

class Component(ABC):
    def __init__(
        self,
        name: str,
        registry: HeartbeatRegistry,
        kill_switch: threading.Event,
        beat_timeout: float = 120.0,
    ):
        self.name        = name
        self.registry    = registry
        self.kill_switch = kill_switch
        self._stop       = threading.Event()
        registry.register(name, beat_timeout=beat_timeout)

    def beat(self, message: str = "") -> None:
        self.registry.beat(self.name, message)

    def stop(self) -> None:
        self._stop.set()

    @abstractmethod
    def run(self) -> None:
        """Main loop. Must call self.beat() regularly and respect self._stop."""


# ── MT5 Monitor ──────────────────────────────────────────────────────────────

class MT5Monitor(Component):
    """Pings the MT5 terminal every 30 s and attempts reconnect on failure."""

    PING_INTERVAL = 30

    def __init__(self, registry, kill_switch, terminal_path: Optional[str] = None):
        super().__init__("MT5Monitor", registry, kill_switch, beat_timeout=90)
        self._path    = terminal_path
        self._retries = 0

    def run(self) -> None:
        while not self._stop.is_set() and not self.kill_switch.is_set():
            info = mt5.account_info()
            if info is not None:
                self._retries = 0
                self.beat(f"connected | balance={info.balance:.2f}")
            else:
                self._retries += 1
                logger.warning("[MT5Monitor] Lost connection (attempt %d/%d), reconnecting...",
                               self._retries, MAX_RECONNECT_TRIES)
                self.registry.beat(self.name, f"reconnecting ({self._retries})", Status.STALE)

                if self._retries >= MAX_RECONNECT_TRIES:
                    msg = "MT5 connection lost — could not recover. Triggering kill switch."
                    logger.critical("[MT5Monitor] %s", msg)
                    self.registry.halt(self.name, msg)
                    self.kill_switch.set()
                    return

                connected = connect(self._path)
                if not connected:
                    time.sleep(10 * self._retries)  # back-off

            time.sleep(self.PING_INTERVAL)


# ── Data Watcher ─────────────────────────────────────────────────────────────

class DataWatcher(Component):
    """Monitors that each symbol is receiving new bars within expected intervals."""

    def __init__(self, registry, kill_switch, symbols: list, tf_str: str):
        super().__init__("DataWatcher", registry, kill_switch, beat_timeout=300)
        self._symbols   = symbols
        self._tf_str    = tf_str
        self._tf_secs   = _TF_SECONDS.get(tf_str, 3600)
        self._last_bar  = {s: None for s in symbols}

    def run(self) -> None:
        tf_code = TIMEFRAME_MAP.get(self._tf_str)
        while not self._stop.is_set() and not self.kill_switch.is_set():
            issues = []
            for symbol in self._symbols:
                bars = mt5.copy_rates_from_pos(symbol, tf_code, 0, 1)
                if bars is None or len(bars) == 0:
                    issues.append(f"{symbol}:no_data")
                    continue
                bar_time = int(bars[0]["time"])
                last     = self._last_bar[symbol]
                if last is not None and (time.time() - bar_time) > self._tf_secs * 3:
                    issues.append(f"{symbol}:stale_bars")
                self._last_bar[symbol] = bar_time

            if issues:
                self.registry.beat(self.name, f"issues={','.join(issues)}", Status.STALE)
                logger.warning("[DataWatcher] %s", issues)
            else:
                self.beat(f"all {len(self._symbols)} symbols feeding")

            time.sleep(self._tf_secs // 4)


# ── Risk Guard ───────────────────────────────────────────────────────────────

class RiskGuard(Component):
    """Enforces Council limits: 2% daily halt, 7% soft halt, 9.5% hard kill.

    Drawdown references are FTMO-correct and restart-safe:
      - TOTAL loss is measured from a STATIC initial balance (persisted), never
        from session-start equity. A restart mid-drawdown cannot move the floor,
        so the hard kill always fires at the true FTMO distance.
      - DAILY loss is measured from the equity at the start of the current
        broker-SERVER trading day (FTMO resets at server midnight, not local
        midnight), persisted so a restart reloads the baseline rather than
        re-anchoring it to a mid-day, possibly already-drawn-down equity.

    A daily-only halt is cleared at the next server day; a cumulative halt is
    sticky until the cumulative drawdown recovers above the soft threshold.
    """

    CHECK_INTERVAL = 60
    STATE_FILE = Path("logs") / "risk_guard_state.json"

    def __init__(
        self,
        registry,
        kill_switch,
        soft_halt_event: threading.Event,
        initial_equity: float,
        symbols: list,
        daily_halt_pct: float = MAX_DAILY_LOSS_PCT,
        soft_dd_pct: float = SOFT_DD_HALT_PCT,
        total_kill_pct: float = MAX_TOTAL_LOSS_PCT,
    ):
        super().__init__("RiskGuard", registry, kill_switch, beat_timeout=180)
        self._soft_halt        = soft_halt_event
        self._initial_equity   = float(initial_equity)   # STATIC FTMO floor reference
        self._symbols          = symbols
        # Config-driven thresholds (were hardcoded — soft_dd_halt_pct in config
        # was silently ignored). Defaults fall back to the module constants.
        self._daily_halt_pct   = float(daily_halt_pct)
        self._soft_dd_pct      = float(soft_dd_pct)
        self._total_kill_pct   = float(total_kill_pct)
        self._day_start_equity: Optional[float] = None
        self._server_day:       Optional[str]   = None
        self._halt_cause:       Optional[str]   = None   # "daily" | "cumulative"
        self._load_state()

    def _load_state(self) -> None:
        try:
            if self.STATE_FILE.exists():
                d = json.loads(self.STATE_FILE.read_text())
                self._day_start_equity = d.get("day_start_equity")
                self._server_day       = d.get("server_day")
                self._halt_cause       = d.get("halt_cause")
        except Exception:
            pass

    def _save_state(self) -> None:
        try:
            self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.STATE_FILE.write_text(json.dumps({
                "day_start_equity": self._day_start_equity,
                "server_day":       self._server_day,
                "halt_cause":       self._halt_cause,
                "initial_equity":   self._initial_equity,
            }))
        except Exception:
            pass

    def _current_server_day(self) -> str:
        """Broker-server calendar day. FTMO's daily loss resets at server midnight;
        MT5 tick.time is server time expressed as a UTC epoch."""
        for sym in self._symbols:
            tick = mt5.symbol_info_tick(sym)
            if tick is not None and tick.time:
                return datetime.fromtimestamp(tick.time, timezone.utc).strftime("%Y-%m-%d")
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def run(self) -> None:
        while not self._stop.is_set() and not self.kill_switch.is_set():
            info = mt5.account_info()
            if info is None:
                self.registry.beat(self.name, "waiting for MT5", Status.STALE)
                time.sleep(self.CHECK_INTERVAL)
                continue

            equity     = info.equity
            server_day = self._current_server_day()

            # Total loss from the STATIC initial balance — never trails, never resets.
            total_pct = (
                (equity - self._initial_equity) / self._initial_equity * 100
                if self._initial_equity else 0.0
            )

            # New broker-server day → re-anchor the daily baseline. Clear a
            # DAILY-only halt if cumulative drawdown is healthy; keep cumulative
            # halts sticky.
            if server_day != self._server_day:
                self._server_day       = server_day
                self._day_start_equity = equity
                if (self._soft_halt.is_set() and self._halt_cause == "daily"
                        and total_pct > -self._soft_dd_pct):
                    self._soft_halt.clear()
                    self._halt_cause = None
                    logger.info("[RiskGuard] New server day %s — daily halt cleared, baseline=%.2f",
                                server_day, equity)
                else:
                    logger.info("[RiskGuard] New server day %s — daily baseline=%.2f",
                                server_day, equity)
                self._save_state()

            if self._day_start_equity is None:
                self._day_start_equity = equity
                self._save_state()

            daily_pct = (
                (equity - self._day_start_equity) / self._day_start_equity * 100
                if self._day_start_equity else 0.0
            )

            msg = (f"eq={equity:.2f} daily={daily_pct:+.2f}% total={total_pct:+.2f}%"
                   f"{' [SOFT-HALT]' if self._soft_halt.is_set() else ''}")

            # Tier 1: daily circuit breaker — halt new entries for rest of server day
            if daily_pct <= -self._daily_halt_pct and not self._soft_halt.is_set():
                logger.critical("[RiskGuard] DAILY CIRCUIT BREAKER: %.2f%% — no new entries today", daily_pct)
                self._soft_halt.set()
                self._halt_cause = "daily"
                self._save_state()

            # Tier 2: soft halt — cumulative soft_dd_pct from initial, halt new entries
            if total_pct <= -self._soft_dd_pct:
                if not self._soft_halt.is_set():
                    logger.critical("[RiskGuard] SOFT HALT: cumulative %.2f%% from initial — no new entries", total_pct)
                    self._soft_halt.set()
                self._halt_cause = "cumulative"   # promote: survives the day boundary
                self._save_state()

            # Tier 3: hard kill — total_kill_pct cumulative from initial, emergency stop
            if total_pct <= -self._total_kill_pct:
                kill_msg = f"HARD KILL: cumulative {total_pct:.2f}% from initial — FTMO breach imminent"
                logger.critical("[RiskGuard] %s", kill_msg)
                self.registry.halt(self.name, kill_msg)
                self.kill_switch.set()
                return

            self.beat(msg)
            time.sleep(self.CHECK_INTERVAL)


# ── Trade Reconciler ─────────────────────────────────────────────────────────

class TradeReconciler(Component):
    """Single source of truth for trade-outcome accounting.

    Polls MT5 deal history for CLOSED positions tagged with our magic — whether
    closed by broker-side SL/TP, an engine close, or manually — and records each
    exactly once into RiskAgent, FTMOTracker, and the TradeJournal.

    This replaces the previous inline recording in the execution loop, which:
      1. never saw broker-side SL/TP closes (the most common outcome), because
         those close server-side and the signal loop only recorded on its own
         close path; and
      2. fed RiskAgent a bogus '(equity - balance) / balance' value that was
         neither an R multiple nor sign-correct (other symbols' floating P&L
         contaminated it).

    Realized R is computed from the position's actual net profit (profit + swap
    + commission across all its deals) divided by the intended per-trade risk
    (initial_balance × risk_pct). Deduplicated by position_id and persisted so a
    restart never double-counts.
    """

    CHECK_INTERVAL = 30
    STATE_FILE = Path("logs") / "reconciler_state.json"

    def __init__(
        self,
        registry,
        kill_switch,
        risk_agent: RiskAgent,
        ftmo_tracker,
        journal,
        magic: int,
        risk_pct: float,
        initial_equity: float,
    ):
        super().__init__("TradeReconciler", registry, kill_switch, beat_timeout=120)
        self._risk_agent     = risk_agent
        self._ftmo           = ftmo_tracker
        self._journal        = journal
        self._magic          = int(magic)
        self._risk_pct       = float(risk_pct) / 100
        self._initial_equity = float(initial_equity)
        self._recorded: set  = set()
        self._child_to_parent: dict = {}   # scale-in child pos_id → parent pos_id
        self._lock           = threading.Lock()
        self._seeded         = self.STATE_FILE.exists()
        self._load_state()

    def _load_state(self) -> None:
        try:
            if self.STATE_FILE.exists():
                self._recorded = set(json.loads(self.STATE_FILE.read_text()).get("recorded", []))
        except Exception:
            pass

    def _save_state(self) -> None:
        try:
            self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            keep = sorted(self._recorded)[-2000:]   # cap growth
            self._recorded = set(keep)
            self.STATE_FILE.write_text(json.dumps({"recorded": keep}))
        except Exception:
            pass

    @staticmethod
    def _scalein_parent(deal) -> Optional[int]:
        """If this is a scale-in add-on's opening deal, return its parent pos_id.

        Scale-in orders are tagged with comment 'si:<parent_position_id>' so a
        logical trade that was added to records as ONE trade, not two."""
        if deal.entry != mt5.DEAL_ENTRY_IN:
            return None
        c = (getattr(deal, "comment", "") or "")
        if c.startswith("si:"):
            try:
                return int(c[3:])
            except ValueError:
                return None
        return None

    def _closed_position_deals(self) -> dict:
        """Return {parent_position_id: [deals]} for fully-closed logical trades.

        Scale-in add-ons are folded into their parent so one idea = one record.
        A group is 'closed' only when the parent AND every child is flat."""
        now   = datetime.now()
        # Upper bound is generous: MT5 filters on SERVER time, which can lead the
        # local clock (FTMO is EET). A tight bound would exclude a just-closed
        # deal until the local clock caught up — a multi-hour delay that defeats
        # prompt SL-hit reaction. Dedup by position_id makes the wide window safe.
        deals = mt5.history_deals_get(now - timedelta(days=3), now + timedelta(days=1))
        if not deals:
            return {}
        bot_deals = [d for d in deals if d.magic == self._magic]

        # Build scale-in child → parent map from IN-deal comments.
        self._child_to_parent = {}
        for d in bot_deals:
            parent = self._scalein_parent(d)
            if parent is not None:
                self._child_to_parent[d.position_id] = parent

        live = {p.identifier for p in (mt5.positions_get() or []) if p.magic == self._magic}

        # Group deals under the parent (children fold in).
        by_pos: dict = {}
        for d in bot_deals:
            pid = self._child_to_parent.get(d.position_id, d.position_id)
            by_pos.setdefault(pid, []).append(d)

        def group_live(parent_pid: int) -> bool:
            if parent_pid in live:
                return True
            return any(c in live for c, p in self._child_to_parent.items() if p == parent_pid)

        return {
            pid: dl for pid, dl in by_pos.items()
            if not group_live(pid)
            and any(d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY) for d in dl)
        }

    def _mark_recorded(self, parent_pid: int) -> None:
        """Record the parent and all its scale-in children as done."""
        self._recorded.add(parent_pid)
        for child, parent in self._child_to_parent.items():
            if parent == parent_pid:
                self._recorded.add(child)

    def _reconcile(self) -> None:
        closed = self._closed_position_deals()

        # First run with no prior state: baseline whatever is already closed
        # (possibly nothing) so only closes AFTER startup are counted. This must
        # run even when `closed` is empty — otherwise the first real close on a
        # fresh deploy would be seeded (suppressed) instead of recorded.
        if not self._seeded:
            for pid in closed:
                self._mark_recorded(pid)
            self._seeded = True
            self._save_state()
            logger.info("[TradeReconciler] Seeded %d pre-existing closed positions (baseline)",
                        len(self._recorded))
            return

        if not closed:
            return

        info        = mt5.account_info()
        equity      = info.equity if info else self._initial_equity
        risk_amount = self._initial_equity * self._risk_pct
        new = False

        for pos_id, dlist in closed.items():
            if pos_id in self._recorded:
                continue
            realized = sum(d.profit + d.swap + d.commission for d in dlist)
            r_mult   = realized / risk_amount if risk_amount > 0 else 0.0
            last_out = max(
                (d for d in dlist if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY)),
                key=lambda d: d.time,
            )
            symbol = last_out.symbol

            self._risk_agent.record_trade(r_mult, equity)
            if self._ftmo is not None:
                self._ftmo.record_trade_day(equity)
            if self._journal is not None:
                try:
                    self._journal.close_trade(symbol, last_out.price, equity)
                except Exception:
                    logger.exception("[TradeReconciler] journal close failed for %s", symbol)

            self._mark_recorded(pos_id)
            new = True
            n_children = sum(1 for p in self._child_to_parent.values() if p == pos_id)
            logger.info("[TradeReconciler] Recorded close pos=%s %s R=%.2f pnl=%.2f%s",
                        pos_id, symbol, r_mult, realized,
                        f" (+{n_children} scale-in)" if n_children else "")

        if new:
            self._save_state()

    def reconcile_now(self) -> None:
        """Record any just-closed positions synchronously (called from the
        execution loop on close, so a following opposite entry sees the result).
        Deduped by position_id, so the background loop never re-records."""
        with self._lock:
            try:
                self._reconcile()
            except Exception:
                logger.exception("[TradeReconciler] reconcile_now failed")

    def run(self) -> None:
        while not self._stop.is_set() and not self.kill_switch.is_set():
            with self._lock:
                try:
                    self._reconcile()
                    self.beat(f"tracked={len(self._recorded)}")
                except Exception as exc:
                    self.registry.fail(self.name, str(exc))
                    logger.exception("[TradeReconciler] Error: %s", exc)
            time.sleep(self.CHECK_INTERVAL)


# ── Signal + Execution Engine ────────────────────────────────────────────────

class TradingEngine(Component):
    """Per-symbol component: fetches bars → generates signal → executes orders."""

    def __init__(
        self,
        registry,
        kill_switch,
        symbol: str,
        strategy,
        tf_str: str,
        trade_cfg: dict,
        risk_agent: RiskAgent,
        ftmo_tracker=None,
        soft_halt_event: Optional[threading.Event] = None,
        dry_run: bool = False,
        journal: Optional["TradeJournal"] = None,
        reconciler: Optional["TradeReconciler"] = None,
        risk_guard: Optional["RiskGuard"] = None,
        allocator: Optional["PortfolioAllocator"] = None,
        book: Optional["PortfolioBook"] = None,
    ):
        name = f"TradingEngine[{symbol}]"
        tf_secs = _TF_SECONDS.get(tf_str, 3600)
        super().__init__(name, registry, kill_switch, beat_timeout=tf_secs * 4)
        self._symbol           = symbol
        self._strategy         = strategy
        self._tf_str           = tf_str
        self._tf_code          = TIMEFRAME_MAP.get(tf_str)
        self._trade_cfg        = trade_cfg
        self._risk_agent       = risk_agent
        self._ftmo_tracker     = ftmo_tracker
        self._soft_halt        = soft_halt_event
        self._dry_run          = dry_run
        self._journal          = journal
        self._reconciler       = reconciler
        self._risk_guard       = risk_guard
        self._allocator        = allocator
        self._book             = book
        self._adaptive_port    = bool(trade_cfg.get("adaptive_portfolio", False))
        self._lookback         = trade_cfg.get("lookback_bars", 500)
        self._level_monitor:   Optional[LevelMonitor] = None
        self._news_gate:        Optional[ng.NewsGate] = None
        self._event_dir_cache: dict[str, int] = {}
        self._trade_manager:   TradeManager = TradeManager()
        self._last_bar         = None
        self._open_entry_price: Optional[float] = None
        self._open_sl:          Optional[float] = None   # initial SL at entry — never moved
        self._open_tp:          Optional[float] = None
        self._last_plan               = None    # last TradePlan from analyze_entry
        self._agent            = TradeAgent()   # per-trade adjudication + tickets
        self._open_score:       int = 0
        self._t1_hit:           bool = False
        self._bars_since_entry: int  = 0
        self._block_entry:      bool = False
        self._scaled_in:        bool = False
        self._pending_signal:   int  = 0    # M15 setup waiting for M5 trigger
        self._pending_bars:     int  = 0    # bars elapsed since pending set
        self._ltf_trigger_bars: int  = 6    # max M15 bars to wait for M5 trigger (90 min)
        self._state_file       = Path("logs") / f"pos_state_{symbol.replace('.','_')}.json"
        self._load_position_state()

    def _save_position_state(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w") as f:
                json.dump({
                    "entry_price": self._open_entry_price,
                    "initial_sl":  self._open_sl,
                    "tp":          self._open_tp,
                    "score":       self._open_score,
                    "t1_hit":      self._t1_hit,
                    "bars":        self._bars_since_entry,
                }, f)
        except Exception:
            pass

    def _load_position_state(self) -> None:
        try:
            if not self._state_file.exists():
                return
            with open(self._state_file) as f:
                d = json.load(f)
            # Only restore if MT5 actually has an open position for this symbol
            positions = mt5.positions_get(symbol=self._symbol)
            if not positions:
                self._state_file.unlink(missing_ok=True)
                return
            self._open_entry_price = d.get("entry_price")
            self._open_sl          = d.get("initial_sl")
            self._open_tp          = d.get("tp")
            self._open_score       = d.get("score", 0)
            self._t1_hit           = d.get("t1_hit", False)
            self._bars_since_entry = d.get("bars", 0)
            logger.info("[%s] Restored position state: entry=%.5f sl=%.5f t1_hit=%s",
                        self.name,
                        self._open_entry_price or 0,
                        self._open_sl or 0,
                        self._t1_hit)
        except Exception:
            pass

    def _size_order(
        self,
        direction: int,
        balance: float,
        size_mult: float,
    ) -> tuple[Optional[float], Optional[float], float]:
        """Return (sl_price, tp_price, lots) for the next order.

        Prefers strategy-computed ATR stops over config fixed points.
        Falls back to config if the strategy provides no stop.
        """
        sl: Optional[float] = None
        tp: Optional[float] = None
        atr_sized = False

        # Try to get price-level SL from strategy (e.g. FVG_OB stores ATR stops)
        stops = getattr(self._strategy, "_stops", None)
        if stops is not None and len(stops) > 0:
            raw_sl = float(stops.iloc[-1])
            if not math.isnan(raw_sl) and raw_sl > 0:
                sl = raw_sl

        if sl is not None:
            # Compute entry from live tick
            tick = mt5.symbol_info_tick(self._symbol)
            entry = tick.ask if direction == 1 else tick.bid
            info = mt5.symbol_info(self._symbol)

            # ── Structural stop + liquidity target (replaces ATR stop & arithmetic TP) ──
            # Every trade gets its own stop AND target from real structure — swing
            # levels, sweep wicks, liquidity pools — not an ATR distance and not
            # entry +/- rr*stop. The strategy's ATR stop is only a fallback reference.
            rr_fb   = getattr(self._strategy, "rr_target", 2.0)
            atr_ser = getattr(self._strategy, "_atr_cache", None)
            atr_val = 0.0
            if atr_ser is not None and len(atr_ser) > 0:
                _a = float(atr_ser.iloc[-1])
                atr_val = _a if not math.isnan(_a) else 0.0
            df_m15_a = self._fetch_ltf_bars("M15", count=120)
            df_m5_a  = self._fetch_ltf_bars("M5",  count=120)
            h4b = (self._strategy.current_h4_bias(df_m15_a)
                   if df_m15_a is not None and hasattr(self._strategy, "current_h4_bias")
                   else 0)
            plan = analyze_entry(
                df_m15=df_m15_a, df_m5=df_m5_a, direction=direction,
                entry=entry, stop=sl, atr=atr_val, h4_bias=h4b, rr_fallback=rr_fb,
            )
            sl = plan.stop          # structural stop replaces the ATR stop
            tp = plan.tp
            self._last_plan = plan
            logger.info("[%s] PLAN grade=%s type=%s size=%.2fx RR=%.2f stop@%s | %s",
                        self.name, plan.grade, plan.trade_type, plan.size_mult,
                        plan.rr, plan.stop_src, plan.thesis)

            # Size lots so that the STRUCTURAL SL hit = risk_pct of balance
            dist = abs(entry - sl)

            # ── Minimum stop distance (fix #1) ────────────────────────────────
            # A suicidally tight stop (e.g. 6pts on US500) gets noise-stopped in
            # minutes AND makes the risk formula spit out a runaway position. The
            # stop must clear a floor of both ATR and % of price, or we don't trade.
            min_atr  = atr_val * float(self._trade_cfg.get("min_stop_atr_mult", 0.5))
            min_pct  = entry * float(self._trade_cfg.get("min_stop_pct", 0.05)) / 100.0
            min_dist = max(min_atr, min_pct)
            if dist < min_dist:
                logger.warning("[%s] STOP TOO TIGHT: dist=%.5f < floor=%.5f "
                               "(atr=%.5f) — REJECTING entry", self.name, dist, min_dist, atr_val)
                return sl, tp, 0.0

            risk_pct = float(self._trade_cfg.get("risk_pct", 1.0)) / 100
            # Automatic risk reduction during news-heavy/volatile days.
            # JP mentor video 4: "Be risk off — half a percent. Take what you're given."
            # If >= 2 high-impact events are upcoming within 4 hours, cap risk at 50%.
            if self._news_gate is not None:
                try:
                    _nctx = self._news_gate.get_context()
                    _high_soon = [
                        e for e in _nctx.upcoming_high
                        if e.minutes_until <= 240
                    ]
                    if len(_high_soon) >= 2:
                        risk_pct *= 0.5
                        logger.info("[%s] News-heavy period (%d events ≤4h) → risk halved to %.3f%%",
                                    self.name, len(_high_soon), risk_pct * 100)
                except Exception:
                    pass
            if info and info.trade_tick_size > 0 and dist > 0:
                point_value_per_lot = (
                    info.trade_tick_value / info.trade_tick_size * info.point
                )
                sl_points = dist / info.point
                sl_value_per_lot = sl_points * point_value_per_lot
                raw_lots = (balance * risk_pct) / sl_value_per_lot if sl_value_per_lot > 0 else 0.01
            else:
                raw_lots = float(self._trade_cfg.get("lot_size", 0.01))
            raw_lots *= plan.size_mult

            if info:
                tp = round(tp, info.digits)
                sl = round(sl, info.digits)
            atr_sized = True
        else:
            # Fallback: config fixed points
            raw_lots = float(self._trade_cfg.get("lot_size", 0.01))
            sl, tp   = risk.calculate_sl_tp(self._symbol, direction, self._trade_cfg)

        lots = raw_lots if atr_sized else risk.calculate_lots(self._symbol, self._trade_cfg, balance)
        lots = lots * size_mult
        # Clamp to broker min/max/step on BOTH paths (H2 — the ATR path previously
        # bypassed this and could exceed volume_max or violate volume_step).
        lots = risk._clamp_lots(self._symbol, lots)

        # ── Max notional cap (fix #2) ─────────────────────────────────────────
        # Backstop so a tight-ish stop can never produce an absurd position size.
        # Big lots stay allowed on high-conviction trades — just not runaway.
        tick_c = mt5.symbol_info_tick(self._symbol)
        info_c = mt5.symbol_info(self._symbol)
        px_c   = (tick_c.ask if direction == 1 else tick_c.bid) if tick_c else 0.0
        if info_c is not None and px_c > 0 and lots > 0:
            contract = getattr(info_c, "trade_contract_size", 1.0) or 1.0
            notional = lots * contract * px_c
            max_notional = balance * float(self._trade_cfg.get("max_notional_x", 30))
            if max_notional > 0 and notional > max_notional:
                capped = risk._clamp_lots(self._symbol, lots * max_notional / notional)
                logger.warning("[%s] NOTIONAL CAP: %.2f -> %.2f lots (notional %.0f > cap %.0f)",
                               self.name, lots, capped, notional, max_notional)
                lots = capped

        lots = max(lots, 0.01)
        return sl, tp, lots

    def _reset_position_state(self) -> None:
        self._open_entry_price  = None
        self._open_sl           = None
        self._open_tp           = None
        self._open_score        = 0
        self._t1_hit            = False
        self._bars_since_entry  = 0
        self._scaled_in         = False
        self._pending_signal    = 0
        self._pending_bars      = 0
        self._state_file.unlink(missing_ok=True)

    def _record_close_now(self) -> None:
        """Record a just-closed position immediately via the reconciler so a
        following opposite entry's risk gate sees the outcome (no 30s lag)."""
        if self._reconciler is not None and not self._dry_run:
            self._reconciler.reconcile_now()

    def _detect_post_event_direction(
        self, event_time: datetime, window_min: float = 15.0, threshold_pct: float = 0.10
    ) -> int:
        """Measure price direction in window_min after an event fires using M1 bars.

        Returns +1 if price moved up by threshold_pct, -1 if down, 0 if unclear or insufficient data.
        This is instrument-specific — it tells us how THIS symbol reacted to the event,
        not the theoretical macro mapping. Market price IS the actual.
        """
        tf_m1 = TIMEFRAME_MAP.get("M1")
        if tf_m1 is None:
            return 0
        bars = mt5.copy_rates_range(
            self._symbol, tf_m1,
            event_time,
            event_time + timedelta(minutes=window_min),
        )
        if bars is None or len(bars) < 3:
            return 0
        open_px  = float(bars[0]["open"])
        close_px = float(bars[-1]["close"])
        if open_px <= 0:
            return 0
        move_pct = (close_px - open_px) / open_px * 100
        if move_pct > threshold_pct:
            return 1
        if move_pct < -threshold_pct:
            return -1
        return 0

    def _price_confirmed_event_direction(self, news_ctx) -> int:
        """Return the price-action-confirmed direction from recent HIGH events.

        Checks fired events that are settled (>=15 min ago, <=2h ago). Results are
        cached per event so MT5 is only queried once per event per session.
        """
        for event in news_ctx.fired_high:
            mins = event.minutes_since
            if mins < 15:
                continue   # market still reacting, wait for settlement
            cache_key = event.event_time.isoformat()
            if cache_key not in self._event_dir_cache:
                self._event_dir_cache[cache_key] = self._detect_post_event_direction(
                    event.event_time
                )
            d = self._event_dir_cache[cache_key]
            if d != 0:
                return d
        return 0

    def _fetch_ltf_bars(self, tf_str: str, count: int) -> Optional[pd.DataFrame]:
        """Fetch recent LTF bars; returns None on failure."""
        tf_code = TIMEFRAME_MAP.get(tf_str)
        if tf_code is None:
            return None
        rates = mt5.copy_rates_from_pos(self._symbol, tf_code, 0, count)
        if rates is None or len(rates) < 10:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df[["time", "open", "high", "low", "close", "tick_volume"]]

    def _execute_trade_action(self, action, pos, pos_dir: int, equity: float) -> None:
        """Execute a TradeAction from the TradeManager."""
        if action.action == ActionType.HOLD:
            if action.counter_score > 0 or action.cont_score > 0:
                logger.debug("[%s] TM: %s", self.name, action.reason)
            return

        logger.info("[%s] TM → %s | %s", self.name, action.action.value, action.reason)

        if action.council_notes:
            for note in action.council_notes:
                logger.info("[%s] Council: %s", self.name, note)

        if action.action == ActionType.EXIT:
            if not self._dry_run:
                trader.close_all(self._symbol)
                self._record_close_now()   # prompt, deduped accounting
            self._reset_position_state()
            self._block_entry = True

        elif action.action == ActionType.WAIT:
            logger.info("[%s] TM WAIT — sweep_risk=%.2f, holding this bar",
                        self.name, action.sweep_risk)

        elif action.action == ActionType.PARTIAL_CLOSE:
            if not self._dry_run:
                trader.partial_close(pos, action.close_pct or 0.30)
                if action.new_sl is not None:
                    trader.modify_sl_tp(self._symbol, pos.ticket,
                                        new_sl=action.new_sl, new_tp=pos.tp)

        elif action.action == ActionType.TIGHTEN_SL:
            if action.new_sl is not None and not self._dry_run:
                trader.modify_sl_tp(self._symbol, pos.ticket,
                                    new_sl=action.new_sl, new_tp=pos.tp)

        elif action.action == ActionType.EXTEND_TP:
            if action.new_tp is not None and not self._dry_run:
                new_sl = action.new_sl if action.new_sl is not None else pos.sl
                trader.modify_sl_tp(self._symbol, pos.ticket,
                                    new_sl=new_sl, new_tp=action.new_tp)
                self._open_tp = action.new_tp
                logger.info("[%s] TP extended -> %.5f | SL tightened -> %.5f",
                            self.name, action.new_tp, new_sl)

        elif action.action == ActionType.HOLD_RUNNER:
            if action.new_sl is not None and not self._dry_run:
                trader.modify_sl_tp(self._symbol, pos.ticket,
                                    new_sl=action.new_sl, new_tp=pos.tp)
                logger.info("[%s] HOLD_RUNNER: SL tightened to structure -> %.5f",
                            self.name, action.new_sl)

    def _correlation_divergence(self, signal_dir: int) -> bool:
        """Return True if entering signal_dir contradicts the dominant group direction.

        Checks all instruments in the same correlation group. If any group member
        has an active position in the OPPOSITE direction, this entry is divergent
        against the correlated move — block it.
        """
        group = next((g for g in _CORR_GROUPS if self._symbol in g), None)
        if group is None:
            return False
        for peer in group:
            if peer == self._symbol:
                continue
            peer_positions = [p for p in (mt5.positions_get(symbol=peer) or [])
                              if p.magic == trader._MAGIC]
            if not peer_positions:
                continue
            for pos in peer_positions:
                peer_dir = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                if peer_dir != signal_dir:
                    logger.info(
                        "[%s] CORR DIVERGENCE: %s is %s but signal=%+d — blocked",
                        self.name, peer, "LONG" if peer_dir == 1 else "SHORT", signal_dir,
                    )
                    return True
        return False

    def _correlation_cluster_cap(self, signal_dir: int) -> bool:
        """Return True if the correlation group already holds the max SAME-direction
        concurrent positions — block this entry to prevent loser clustering.

        Monte Carlo autopsy: the #1 blow-up cause is correlated same-direction
        trades losing together on one day (worst historical day: 17 stacked losers
        = -17R). Capping concurrent same-dir positions per group attacks that
        directly without giving up edge (it strips clustered losers, keeps winners).
        """
        group = next((g for g in _CORR_GROUPS if self._symbol in g), None)
        if group is None:
            return False
        same_dir = 0
        for peer in group:
            if peer == self._symbol:
                continue
            for pos in (mt5.positions_get(symbol=peer) or []):
                peer_dir = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                if peer_dir == signal_dir:
                    same_dir += 1
        if same_dir >= MAX_SAME_DIR_CLUSTER:
            logger.info(
                "[%s] CORR CLUSTER CAP: %d peers already %s in group (max %d) — blocked",
                self.name, same_dir, "LONG" if signal_dir == 1 else "SHORT",
                MAX_SAME_DIR_CLUSTER,
            )
            return True
        return False

    def _dd_room(self, equity: float) -> tuple[float, float]:
        """Return (daily_room_pct, total_room_pct) — how much DD budget is left before
        the Council/FTMO limits. Used by the per-trade agent's Compliance voice."""
        try:
            acct = trader.get_account()
            bal  = acct.get("balance", equity) or equity
            # daily: room to the -2% soft halt from today's start (approx via balance)
            start = getattr(self, "_day_start_equity", bal) or bal
            daily_loss = max(0.0, (start - equity) / start * 100) if start else 0.0
            daily_room = MAX_DAILY_LOSS_PCT - daily_loss
            # total: room to the soft DD halt from peak
            peak = max(getattr(self, "_peak_equity", bal) or bal, equity)
            self._peak_equity = peak
            total_loss = max(0.0, (peak - equity) / peak * 100) if peak else 0.0
            total_room = SOFT_DD_HALT_PCT - total_loss
            return daily_room, total_room
        except Exception:
            return MAX_DAILY_LOSS_PCT, SOFT_DD_HALT_PCT

    def _fetch_dxy_bias(self) -> int:
        """H4 EMA bias of DXY index. Returns +1 (USD strong), -1 (USD weak), 0 (unavailable).

        Tries common broker symbol names for USD index. Gold has a robust inverse
        relationship with DXY — DXY bullish = gold bearish and vice versa.
        """
        _dxy_symbols = ("DXY", "USDX", "USDINDEX", "DX.f", "USDOLLAR")
        for _sym in _dxy_symbols:
            try:
                _rates = mt5.copy_rates_from_pos(_sym, mt5.TIMEFRAME_M15, 0, 200)
                if _rates is None or len(_rates) < 50:
                    continue
                _ddf = pd.DataFrame(_rates)
                _ddf.columns = [c.lower() for c in _ddf.columns]
                _times = pd.to_datetime(_ddf["time"], unit="s", utc=True)
                _tmp = _ddf[["close"]].copy()
                _tmp.index = _times
                _h4c = _tmp["close"].resample("4h").last().dropna()
                if len(_h4c) < 10:
                    continue
                _fast = _h4c.ewm(span=5, adjust=False).mean()
                _slow = _h4c.ewm(span=20, adjust=False).mean()
                return 1 if float(_fast.iloc[-1]) > float(_slow.iloc[-1]) else -1
            except Exception:
                continue
        return 0  # DXY not available on this broker — no penalty

    def _portfolio_pnl_r(self) -> float:
        """Return total portfolio floating P&L in R units (risk_pct of equity per trade)."""
        try:
            all_pos = trader.get_all_positions()   # magic-filtered — bot positions only
            total_pnl = sum(p.profit for p in all_pos)
            account   = trader.get_account()
            equity    = account.get("equity", account.get("balance", 0))
            risk_pct  = float(self._trade_cfg.get("risk_pct", 0.5)) / 100
            risk_per_trade = equity * risk_pct
            return total_pnl / risk_per_trade if risk_per_trade > 0 else 0.0
        except Exception:
            return 0.0

    def _open_portfolio_risk_pct(self, equity: float) -> float:
        """Aggregate at-risk amount across all open bot positions, as % of equity.

        Risk per position = distance to its current SL × point value × volume.
        A position trailed to break-even or better contributes ~0. Positions with
        no stop are charged the configured per-trade risk as a conservative proxy.
        """
        if equity <= 0:
            return 0.0
        fallback_risk = equity * (float(self._trade_cfg.get("risk_pct", 1.0)) / 100)
        total_risk = 0.0
        for p in trader.get_all_positions():
            if not p.sl or p.sl <= 0:
                total_risk += fallback_risk
                continue
            info = mt5.symbol_info(p.symbol)
            if info is None or info.trade_tick_size <= 0 or info.point <= 0:
                total_risk += fallback_risk
                continue
            point_value_per_lot = info.trade_tick_value / info.trade_tick_size * info.point
            direction = 1 if p.type == mt5.ORDER_TYPE_BUY else -1
            # Only the adverse distance counts; SL beyond entry (locked profit) = 0 risk
            adverse = (p.price_open - p.sl) if direction == 1 else (p.sl - p.price_open)
            if adverse <= 0:
                continue
            sl_points = adverse / info.point
            total_risk += sl_points * point_value_per_lot * p.volume
        return total_risk / equity * 100

    def _portfolio_snapshot(self, equity: float):
        """Whole-book view for the allocator: (list[OpenPos], {ticket: (risk_pct, pos)}).

        Per-position risk is distance-to-current-SL in % of equity (0 if at BE+);
        pnl_r is floating P&L in R; score comes from the shared PortfolioBook."""
        positions = trader.get_all_positions()
        if self._book is not None:
            self._book.prune([p.ticket for p in positions])
        fallback_pct = float(self._trade_cfg.get("risk_pct", 1.0))
        book_pos: list = []
        risk_map: dict = {}
        for p in positions:
            direction = 1 if p.type == mt5.ORDER_TYPE_BUY else -1
            info = mt5.symbol_info(p.symbol)
            risk_amt = 0.0
            if p.sl and p.sl > 0 and info and info.trade_tick_size > 0 and info.point > 0:
                adverse = (p.price_open - p.sl) if direction == 1 else (p.sl - p.price_open)
                if adverse > 0:
                    pvpl = info.trade_tick_value / info.trade_tick_size * info.point
                    risk_amt = (adverse / info.point) * pvpl * p.volume
                risk_pct = risk_amt / equity * 100 if equity > 0 else 0.0
            else:
                risk_pct = fallback_pct
                risk_amt = equity * fallback_pct / 100
            pnl_r = (p.profit / risk_amt) if risk_amt > 0 else 0.0
            score = self._book.score_for(p.ticket) if self._book is not None else 5
            book_pos.append(OpenPos(symbol=p.symbol, score=score, risk_pct=risk_pct,
                                    direction=direction, pnl_r=pnl_r, ticket=p.ticket))
            risk_map[p.ticket] = (risk_pct, p)
        return book_pos, risk_map

    def _portfolio_daily_guard(self) -> None:
        """Behaviour 3: actively trim the weakest exposure so open positions can
        never run the book past the FTMO daily loss limit. Gated off by default."""
        if not (self._adaptive_port and self._allocator is not None):
            return
        try:
            account = trader.get_account()
            equity  = account.get("equity", account.get("balance", 0))
            day_start = getattr(self._risk_guard, "_day_start_equity", None) if self._risk_guard else None
            if not day_start or equity <= 0:
                return
            day_loss_pct = max(0.0, (day_start - equity) / day_start * 100)
            limit = float(self._trade_cfg.get("ftmo_daily_limit_pct", 5.0))
            buf   = float(self._trade_cfg.get("daily_guard_buffer_pct", 1.0))
            book, risk_map = self._portfolio_snapshot(equity)
            trims = self._allocator.daily_guard(day_loss_pct, book, limit_pct=limit, buffer_pct=buf)
            if trims:
                logger.warning("[%s] DAILY GUARD: day loss %.2f%% — shedding %.2f%% open risk",
                               self.name, day_loss_pct, sum(t.reduce_pct for t in trims))
                self._execute_trims(trims, risk_map)
        except Exception as exc:
            logger.exception("[%s] daily guard error: %s", self.name, exc)

    def _execute_trims(self, trims, risk_map) -> None:
        """Reduce risk on the given tickets by partial-closing the matching lots."""
        for t in trims:
            entry = risk_map.get(t.ticket)
            if entry is None:
                continue
            risk_pct, pos = entry
            if risk_pct <= 0:
                continue
            frac = min(1.0, t.reduce_pct / risk_pct)   # portion of the position to shed
            if frac <= 0:
                continue
            logger.info("[%s] REALLOCATE: trim %s ticket=%s by %.2f%% risk (%.0f%% of lots)",
                        self.name, t.symbol, t.ticket, t.reduce_pct, frac * 100)
            if not self._dry_run:
                trader.partial_close(pos, frac)

    def _tighten_all_sl(self, new_sl: float, direction: int) -> None:
        """Move SL to new_sl on every bot position for this symbol, only where it
        tightens (protects scaled-in add-ons that share this symbol — H4)."""
        for p in trader.get_positions(self._symbol):
            cur = p.sl
            tighter = (
                (direction == 1  and (cur <= 0 or new_sl > cur + 1e-8)) or
                (direction == -1 and (cur <= 0 or new_sl < cur - 1e-8))
            )
            if tighter and not self._dry_run:
                trader.modify_sl_tp(self._symbol, p.ticket, new_sl=new_sl, new_tp=p.tp)

    def _manage_open_position(self) -> None:
        """Bar-by-bar T1 partial, trail stop, and time stop for open positions."""
        positions = trader.get_positions(self._symbol)
        if not positions:
            return

        pos = positions[0]

        # Recover entry state from MT5 position data if internal state was lost (e.g. restart)
        if self._open_entry_price is None:
            self._open_entry_price = pos.price_open
            self._open_sl          = pos.sl if pos.sl > 0 else None
            self._open_tp          = pos.tp if pos.tp > 0 else None
            if self._open_sl is not None:
                logger.info("[%s] Position state recovered from MT5: entry=%.5f sl=%.5f",
                            self.name, self._open_entry_price, self._open_sl)
                self._save_position_state()

        if self._open_sl is None:
            return

        entry     = self._open_entry_price
        init_sl   = self._open_sl        # never changed — used for all R calculations
        risk_dist = abs(entry - init_sl)
        if risk_dist <= 0:
            return

        tick = mt5.symbol_info_tick(self._symbol)
        if tick is None:
            return
        mid_price  = (tick.bid + tick.ask) / 2.0
        current_sl = pos.sl

        # Track MFE/MAE for the post-trade review
        if self._journal is not None:
            self._journal.update_path(self._symbol, tick.ask, tick.bid)

        t1_r   = getattr(self._strategy, "t1_r", 0.0)
        t1_pct = getattr(self._strategy, "t1_partial_pct", 0.5)
        ts_bars = getattr(self._strategy, "time_stop_bars", 0)

        # T1 partial close — fires once
        if t1_r > 0 and not self._t1_hit:
            t1_price = entry + t1_r * risk_dist if pos.type == mt5.ORDER_TYPE_BUY else entry - t1_r * risk_dist
            t1_hit   = (pos.type == mt5.ORDER_TYPE_BUY and mid_price >= t1_price) or \
                       (pos.type == mt5.ORDER_TYPE_SELL and mid_price <= t1_price)
            if t1_hit:
                logger.info("[%s] T1 hit @ %.5f — partial close %.0f%%", self.name, mid_price, t1_pct * 100)
                if not self._dry_run:
                    trader.partial_close(pos, t1_pct)
                    self._tighten_all_sl(entry, 1 if pos.type == mt5.ORDER_TYPE_BUY else -1)
                self._t1_hit = True
                current_sl   = entry  # reflect BE move for trail logic below

        # Trail stop (symbol must be in TRAIL_CONFIGS)
        if self._symbol in TRAIL_CONFIGS:
            trail_cfg = TRAIL_CONFIGS[self._symbol]
            be_r      = trail_cfg.get("trail_be_r", 1.0)
            lock_r    = trail_cfg.get("trail_lock_r", 2.0)

            if pos.type == mt5.ORDER_TYPE_BUY:
                if mid_price >= entry + lock_r * risk_dist:
                    new_sl = max(current_sl, entry + risk_dist)
                elif mid_price >= entry + be_r * risk_dist and not self._t1_hit:
                    new_sl = max(current_sl, entry)
                else:
                    new_sl = current_sl
                if new_sl > current_sl + 1e-8:
                    self._tighten_all_sl(new_sl, 1)   # protects scaled-in add-ons too
                    logger.info("[%s] Trail SL: %.5f -> %.5f (long)", self.name, current_sl, new_sl)

            elif pos.type == mt5.ORDER_TYPE_SELL:
                if mid_price <= entry - lock_r * risk_dist:
                    new_sl = min(current_sl, entry - risk_dist)
                elif mid_price <= entry - be_r * risk_dist and not self._t1_hit:
                    new_sl = min(current_sl, entry)
                else:
                    new_sl = current_sl
                if new_sl < current_sl - 1e-8:
                    self._tighten_all_sl(new_sl, -1)   # protects scaled-in add-ons too
                    logger.info("[%s] Trail SL: %.5f -> %.5f (short)", self.name, current_sl, new_sl)

        # ── News protection (JP mentor): protect profits before high-impact events ──
        # "If I'm up four grand before that time, I'll protect my four grand." — don't let
        # a news spike wipe an already-won position. Move SL to BE if in profit and event
        # is within 30 minutes. Only fires once per trade (SL already at or past BE).
        if self._news_gate is not None and risk_dist > 0:
            _news_ctx = self._news_gate.get_context()
            _instr_ccy = ng._INSTRUMENT_CURRENCY.get(self._symbol, "")
            _relevant_high = [
                e for e in _news_ctx.upcoming_high
                if e.currency == _instr_ccy and e.minutes_until <= 30
            ]
            if _relevant_high:
                _cur_r = (
                    (mid_price - entry) / risk_dist if pos.type == mt5.ORDER_TYPE_BUY
                    else (entry - mid_price) / risk_dist
                )
                if _cur_r >= 0.5 and current_sl != entry:
                    _event_names = ", ".join(e.name for e in _relevant_high)
                    logger.info(
                        "[%s] NEWS PROTECT: High event in %.0fmin (%s) — "
                        "trade at %.2fR, moving SL to BE %.5f",
                        self.name, min(e.minutes_until for e in _relevant_high),
                        _event_names, _cur_r, entry,
                    )
                    _protect_sl = entry  # breakeven
                    if pos.type == mt5.ORDER_TYPE_BUY:
                        _protect_sl = max(current_sl, entry)
                    else:
                        _protect_sl = min(current_sl, entry)
                    if abs(_protect_sl - current_sl) > 1e-8:
                        self._tighten_all_sl(_protect_sl, 1 if pos.type == mt5.ORDER_TYPE_BUY else -1)

        # ── Adaptive structural management (per trade type) ──
        # Ratchet the stop to new structure and extend the target to the next draw
        # as the trade develops. Tighten-only on SL; only extends TP.
        try:
            pos_dir = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            cur_r   = ((mid_price - entry) if pos_dir == 1 else (entry - mid_price)) / risk_dist
            ttype   = self._last_plan.trade_type if self._last_plan else "continuation"
            df_mgmt = self._fetch_ltf_bars("M5", count=120)
            atr_m   = 0.0
            if df_mgmt is not None and len(df_mgmt) > 14:
                atr_m = float((df_mgmt["high"] - df_mgmt["low"]).rolling(14).mean().iloc[-1])
            live_sl = pos.sl if pos.sl > 0 else current_sl
            live_tp = pos.tp if pos.tp > 0 else (self._open_tp or 0.0)
            bank_r = float(self._trade_cfg.get("bank_tp_r", 0.0)) if self._adaptive_port else 0.0
            mdec = analyze_manage(
                df=df_mgmt, direction=pos_dir, entry=entry, initial_sl=init_sl,
                current_sl=live_sl, current_tp=live_tp, price=mid_price,
                atr=atr_m, trade_type=ttype, cur_r=cur_r, bank_min_r=bank_r,
            )
            apply_sl = mdec.new_sl if (mdec.new_sl is not None and (
                (pos_dir == 1 and mdec.new_sl > live_sl + 1e-8) or
                (pos_dir == -1 and mdec.new_sl < live_sl - 1e-8))) else None
            # Extend-out always allowed; bank-IN allowed only when gated and the
            # nearer TP still sits in profit beyond current price (never a loss).
            _extend_ok = mdec.new_tp is not None and (
                (pos_dir == 1 and mdec.new_tp > live_tp) or
                (pos_dir == -1 and mdec.new_tp < live_tp))
            _bank_ok = mdec.new_tp is not None and "bank_tp" in (mdec.reason or "") and (
                (pos_dir == 1 and mid_price < mdec.new_tp) or
                (pos_dir == -1 and mid_price > mdec.new_tp))
            apply_tp = mdec.new_tp if (_extend_ok or _bank_ok) else None
            if (apply_sl is not None or apply_tp is not None) and not self._dry_run:
                trader.modify_sl_tp(self._symbol, pos.ticket,
                                    new_sl=apply_sl if apply_sl is not None else live_sl,
                                    new_tp=apply_tp if apply_tp is not None else live_tp)
                if apply_tp is not None:
                    self._open_tp = apply_tp
                logger.info("[%s] ADAPTIVE(%s) sl=%s tp=%s | %s",
                            self.name, ttype, apply_sl, apply_tp, mdec.reason)
        except Exception as exc:
            logger.debug("[%s] adaptive mgmt skipped: %s", self.name, exc)

        # Time stop — close if no exit after N bars
        if ts_bars > 0:
            self._bars_since_entry += 1
            if self._bars_since_entry >= ts_bars:
                logger.warning("[%s] Time stop: %d bars elapsed — closing", self.name, ts_bars)
                if not self._dry_run:
                    trader.close_all(self._symbol)
                    self._record_close_now()   # prompt, deduped accounting
                self._reset_position_state()
                self._block_entry = True   # skip new entry on this bar
                return

        # Post-event invalidation: if price confirms opposing direction after HIGH news, close
        if self._news_gate is not None:
            news_ctx = self._news_gate.get_context()
            upcoming = news_ctx.upcoming_summary()
            if upcoming:
                logger.info("[%s] Upcoming news: %s", self.name, upcoming)
            confirmed_dir = self._price_confirmed_event_direction(news_ctx)
            if confirmed_dir != 0:
                pos_dir = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                event_names = [e.name for e in news_ctx.fired_high]
                if confirmed_dir == -pos_dir:
                    logger.warning(
                        "[%s] Post-event price invalidates %s (market moved %+d after news) | %s",
                        self.name, "LONG" if pos_dir == 1 else "SHORT",
                        confirmed_dir, event_names,
                    )
                    if not self._dry_run:
                        trader.close_all(self._symbol)
                        self._record_close_now()   # prompt, deduped accounting
                    self._reset_position_state()
                    self._block_entry = True
                else:
                    logger.info(
                        "[%s] Post-event price confirms position direction (%+d) | %s",
                        self.name, confirmed_dir, event_names,
                    )

        # If already exited by news invalidation, don't also run TradeManager
        if self._block_entry:
            return

        # ── TradeManager: adaptive multi-timeframe decisions ──────────────────
        # Refresh positions (may have changed after partial close / news close)
        positions = trader.get_positions(self._symbol)
        if not positions:
            return

        pos           = positions[0]
        df_m15_struct = self._fetch_ltf_bars(self._tf_str, count=80)
        if df_m15_struct is None:
            return
        df_m5 = self._fetch_ltf_bars("M5", count=60)
        df_m1 = self._fetch_ltf_bars("M1", count=30)

        h4_bias = (
            self._strategy.current_h4_bias(df_m15_struct)
            if hasattr(self._strategy, "current_h4_bias")
            else getattr(self._strategy, "_last_h4_bias", 0)
        )

        news_dir = 0
        if self._news_gate is not None:
            news_dir = self._price_confirmed_event_direction(
                self._news_gate.get_context()
            )

        tm_pos_dir = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
        pos_state = PositionState(
            direction     = tm_pos_dir,
            entry_price   = self._open_entry_price,
            initial_sl    = self._open_sl,
            current_sl    = pos.sl,
            current_tp    = pos.tp,
            current_price = mid_price,
            bars_elapsed  = self._bars_since_entry,
            t1_hit        = self._t1_hit,
            h4_bias       = h4_bias,
        )

        # Portfolio P&L in R units: sum of all open positions' floating P&L
        # divided by the per-trade risk. Protects gains by tightening losers.
        portfolio_pnl_r = self._portfolio_pnl_r()

        action = self._trade_manager.evaluate(
            position           = pos_state,
            df_m15             = df_m15_struct,
            df_m5              = df_m5,
            df_m1              = df_m1,
            news_confirmed_dir = news_dir,
            portfolio_pnl_r    = portfolio_pnl_r,
        )

        account = trader.get_account()
        equity  = account.get("equity", account.get("balance", 0))
        self._execute_trade_action(action, pos, tm_pos_dir, equity)

    def run(self) -> None:
        tf_secs      = _TF_SECONDS.get(self._tf_str, 3600)
        poll_interval = min(30, max(5, tf_secs // 30))

        logger.info("[%s] Started | strategy=%s | tf=%s | dry_run=%s",
                    self.name, self._strategy.name, self._tf_str, self._dry_run)

        while not self._stop.is_set() and not self.kill_switch.is_set():
            try:
                bars = mt5.copy_rates_from_pos(self._symbol, self._tf_code, 0, 1)
                if bars is None or len(bars) == 0:
                    self.registry.beat(self.name, "waiting for bar", Status.STALE)
                    time.sleep(poll_interval)
                    continue

                bar_time = int(bars[0]["time"])
                if bar_time == self._last_bar:
                    time.sleep(poll_interval)
                    continue

                self._last_bar = bar_time
                bar_dt = datetime.fromtimestamp(bar_time, tz=timezone.utc)

                # Fetch lookback and generate signal
                rates = mt5.copy_rates_from_pos(self._symbol, self._tf_code, 0, self._lookback)
                if rates is None or len(rates) == 0:
                    self.registry.beat(self.name, "no lookback data", Status.STALE)
                    continue

                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                df = df[["time", "open", "high", "low", "close", "tick_volume"]]

                signals = self._strategy.generate_signals(df)
                desired = int(signals.iloc[-1])
                current = trader.get_position_direction(self._symbol)

                self.beat(
                    f"bar={bar_dt.strftime('%H:%M')} | "
                    f"signal={desired:+d} | pos={current:+d}"
                )

                logger.info("[%s] bar=%s signal=%+d pos=%+d",
                            self.name, bar_dt.strftime("%Y-%m-%d %H:%M UTC"), desired, current)

                # Position management — T1 partial, trail, time stop
                self._block_entry = False
                self._portfolio_daily_guard()
                self._manage_open_position()

                if self._block_entry:
                    self._pending_signal = 0
                    continue

                # ── Pending M5 trigger: age out expired setups ────────────────
                if self._pending_signal != 0 and current == 0:
                    self._pending_bars += 1
                    df_m5_check = self._fetch_ltf_bars("M5", count=40)
                    atr_check   = 0.0
                    if df_m5_check is not None and len(df_m5_check) > 14:
                        atr_check = float(
                            (df_m5_check["high"] - df_m5_check["low"])
                            .rolling(14).mean().iloc[-1]
                        )
                    triggered = detect_m5_entry_trigger(
                        df_m5_check, self._pending_signal, atr=atr_check
                    ) if df_m5_check is not None else False

                    if triggered:
                        logger.info("[%s] M5 trigger CONFIRMED for pending signal=%+d (bar %d)",
                                    self.name, self._pending_signal, self._pending_bars)
                        desired = self._pending_signal
                        self._pending_signal = 0
                        self._pending_bars   = 0
                        # fall through to entry logic below
                    elif self._pending_bars >= self._ltf_trigger_bars:
                        logger.info("[%s] M5 trigger EXPIRED (signal=%+d, %d bars)",
                                    self.name, self._pending_signal, self._pending_bars)
                        self._pending_signal = 0
                        self._pending_bars   = 0
                        continue
                    else:
                        continue  # still waiting

                # Scale-in: when signal confirms a profitable position, add a unit
                if desired == current and current != 0 and not getattr(self, "_scaled_in", False):
                    if self._open_entry_price is not None and self._open_sl is not None:
                        risk_dist = abs(self._open_entry_price - self._open_sl)
                        tick_now  = mt5.symbol_info_tick(self._symbol)
                        cur_price = (tick_now.bid + tick_now.ask) / 2.0 if tick_now else 0.0
                        if risk_dist > 0 and cur_price > 0:
                            cur_r = (
                                (self._open_entry_price - cur_price) / risk_dist if current == -1
                                else (cur_price - self._open_entry_price) / risk_dist
                            )
                            if cur_r >= 0.5:  # position at 0.5R+ profit — add to winner
                                account = trader.get_account()
                                equity  = account["equity"] if "equity" in account else account["balance"]
                                # Gate scale-in on remaining portfolio risk budget (C3)
                                max_port   = float(self._trade_cfg.get("max_portfolio_risk_pct", 4.0))
                                open_risk  = self._open_portfolio_risk_pct(equity)
                                add_risk   = float(self._trade_cfg.get("risk_pct", 1.0)) * 0.5
                                if open_risk + add_risk > max_port:
                                    logger.info("[%s] SCALE-IN skipped: portfolio risk %.2f%% + %.2f%% > cap %.2f%%",
                                                self.name, open_risk, add_risk, max_port)
                                    continue
                                _, _, lots = self._size_order(current, equity, 0.5)  # 50% size add-on
                                sl_add = self._open_sl   # same SL as original
                                # Tag with parent position id so the reconciler folds
                                # this add-on into the original trade (one idea = one
                                # recorded trade, not two).
                                parent_positions = trader.get_positions(self._symbol)
                                parent_pid = parent_positions[0].identifier if parent_positions else 0
                                if not self._dry_run:
                                    ok = trader.place_order(self._symbol, current, lots,
                                                            sl=sl_add, comment=f"si:{parent_pid}")
                                    if ok:
                                        self._scaled_in = True
                                        logger.info("[%s] SCALE-IN: +%.2f lots at %.5f (cur_r=%.2f)",
                                                    self.name, lots, cur_price, cur_r)
                                else:
                                    logger.info("[%s] DRY RUN SCALE-IN: %.2f lots (cur_r=%.2f)",
                                                self.name, lots, cur_r)
                    continue

                if desired == current:
                    continue

                account = trader.get_account()
                equity  = account["equity"] if "equity" in account else account["balance"]

                # Close existing position and record trade outcome
                if current != 0:
                    if self._dry_run:
                        logger.info("[%s] DRY RUN: close %s", self.name,
                                    "LONG" if current == 1 else "SHORT")
                    else:
                        trader.close_all(self._symbol)
                        # Record the close NOW (deduped) so the opposite entry below
                        # sees the outcome in its risk gate — no async lag.
                        self._record_close_now()
                        self._reset_position_state()

                # Open new position — gate through soft halt, RiskAgent, FTMOTracker
                if desired != 0:
                    # Soft halt check (2% daily or 7% cumulative DD)
                    if self._soft_halt is not None and self._soft_halt.is_set():
                        logger.warning("[%s] SOFT HALT active — blocking new entry", self.name)
                        continue

                    # ── M5 entry trigger gate ─────────────────────────────────
                    df_m5_entry = self._fetch_ltf_bars("M5", count=40)
                    atr_entry   = 0.0
                    if df_m5_entry is not None and len(df_m5_entry) > 14:
                        atr_entry = float(
                            (df_m5_entry["high"] - df_m5_entry["low"])
                            .rolling(14).mean().iloc[-1]
                        )
                    if not detect_m5_entry_trigger(df_m5_entry, desired, atr=atr_entry):
                        # M5 hasn't broken structure yet — set pending, wait
                        if self._pending_signal != desired:
                            logger.info("[%s] M5 not confirmed — pending signal=%+d",
                                        self.name, desired)
                            self._pending_signal = desired
                            self._pending_bars   = 0
                        continue

                    # ── Accumulation / distribution gate ─────────────────────
                    accum_risk = detect_accumulation(
                        df_m5_entry, desired, atr=atr_entry
                    )
                    if accum_risk >= 0.66:
                        logger.warning(
                            "[%s] ACCUM GATE: %.0f%% risk of opposing accumulation — blocked signal=%+d",
                            self.name, accum_risk * 100, desired,
                        )
                        continue

                    # ── Liquidity draw gate ───────────────────────────────────
                    liq = detect_liquidity_draw(df_m5_entry, desired, atr=atr_entry)
                    if liq["block_entry"]:
                        logger.warning(
                            "[%s] LIQUIDITY GATE: opposing pool %.1f ATR away vs aligned %.1f ATR — blocked",
                            self.name, liq["opposing_pool"], liq["aligned_pool"],
                        )
                        continue

                    # ── Correlation divergence gate ───────────────────────────
                    if self._correlation_divergence(desired):
                        continue

                    # ── Correlation cluster cap (same-direction) ──────────────
                    if self._correlation_cluster_cap(desired):
                        continue

                    open_count = len(trader.get_all_positions())   # magic-filtered (H3)
                    can_trade, size_mult, reason = self._risk_agent.pre_trade_check(
                        equity, open_count
                    )
                    if not can_trade:
                        logger.warning("[%s] RiskAgent blocked: %s", self.name, reason)
                        continue

                    # FTMO compliance pre-trade gate (#05) — both total AND daily DD
                    if self._ftmo_tracker is not None:
                        ftmo_status = self._ftmo_tracker.check(equity)
                        if ftmo_status["total_dd_pct"] >= ftmo_status["total_dd_limit"]:
                            logger.critical("[%s] FTMO total DD limit — blocking entry", self.name)
                            continue
                        # Daily DD: measured from RiskGuard's SERVER-day baseline —
                        # the single source of truth for the daily reference, so this
                        # gate and the RiskGuard breaker never disagree at rollover.
                        day_start = getattr(self._risk_guard, "_day_start_equity", None) if self._risk_guard else None
                        if day_start:
                            daily_dd = (day_start - equity) / day_start * 100
                            if daily_dd >= ftmo_status["daily_dd_limit"]:
                                logger.critical("[%s] FTMO daily DD %.2f%% >= %.1f%% — blocking entry",
                                                self.name, daily_dd, ftmo_status["daily_dd_limit"])
                                continue

                    # Level monitor — update key levels, log any approaches
                    _approaching_levels = []
                    if self._level_monitor is not None:
                        try:
                            _atr_now = float(getattr(self._strategy, "_atr_cache", pd.Series()).iloc[-1]) \
                                       if hasattr(self._strategy, "_atr_cache") else 0.0
                            _approaching_levels = self._level_monitor.update(
                                self._symbol, df, _atr_now or (equity * 0.002)
                            )
                        except Exception:
                            pass

                    # Score-based sizing: psychology_mult * score_mult * concentration_mult
                    _sc          = getattr(self._strategy, "_scores", None)
                    signal_score = int(_sc.iloc[-1]) if _sc is not None else 0

                    # News gate: price-confirmed direction preferred; consensus as fallback
                    # Amplifier only — never penalises
                    if self._news_gate is not None:
                        news_ctx      = self._news_gate.get_context()
                        confirmed_dir = self._price_confirmed_event_direction(news_ctx)
                        if confirmed_dir != 0:
                            if confirmed_dir == desired:
                                logger.info("[%s] Post-event price confirms signal: score +1 | events=%s",
                                            self.name, [e.name for e in news_ctx.fired_high])
                                signal_score += 1
                        else:
                            news_mod = news_ctx.score_modifier(self._symbol, desired)
                            if news_mod > 0:
                                logger.info("[%s] Macro consensus amplifies signal: +%d | %s",
                                            self.name, news_mod,
                                            news_ctx.fired_summary(self._symbol))
                                signal_score += news_mod

                    # Level confluence: +1 when entry fires at a pre-marked key level
                    if _approaching_levels:
                        signal_score += 1
                        lvl_names = ", ".join(l.label for l in _approaching_levels)
                        logger.info("[%s] Key level confluence +1: %s", self.name, lvl_names)

                    # DXY alignment check — metals and forex pairs with known DXY correlation.
                    # JP mentor: "Dixie is the driving force behind GU and EU — if Dixie gains
                    # strength, GU and EU come down because the dollar side gets heavier."
                    # XAGUSD, GBPUSD, EURUSD: inverse DXY relationship — DXY bullish = pair bearish.
                    # XAUUSD excluded: JP mentor v6 — gold is "its own entity / safe haven",
                    # "the correlation isn't that tight" — DXY filter overfits on gold.
                    # → DXY is a FILTER (opposition = -1), not a bonus (alignment = no change).
                    _DXY_INVERSE = {"XAGUSD", "GBPUSD", "EURUSD"}
                    if self._symbol in _DXY_INVERSE:
                        _dxy_bias = self._fetch_dxy_bias()
                        if _dxy_bias != 0:
                            # DXY bullish + going long (metals or USD pair) = DXY opposing
                            # DXY bearish + going short (metals or USD pair) = DXY opposing
                            if (_dxy_bias == 1 and desired == 1) or (_dxy_bias == -1 and desired == -1):
                                signal_score -= 1
                                logger.info("[%s] DXY opposing: DXY bias=%+d, trade=%+d → score %d",
                                            self.name, _dxy_bias, desired, signal_score)

                    # ── Trend-aware quality bar (fix #4) ──────────────────────
                    # Bidirectional stays. The bar is NOT a flat score-6 on every
                    # trade — that would miss too much. With-trend (aligned to the
                    # H4 bias) clears a normal bar; a COUNTER-trend trade (e.g. an
                    # index short into an up-bias) must be genuinely high-conviction.
                    base_min    = int(self._trade_cfg.get("min_entry_score", 5))
                    counter_min = int(self._trade_cfg.get("min_entry_score_counter", 6))
                    try:
                        h4_bias = (self._strategy.current_h4_bias(df)
                                   if hasattr(self._strategy, "current_h4_bias")
                                   else getattr(self._strategy, "_last_h4_bias", 0))
                    except Exception:
                        h4_bias = 0
                    against_trend = (h4_bias != 0 and desired != h4_bias)
                    needed = counter_min if against_trend else base_min

                    # ── DD-aware score boost ───────────────────────────────────
                    # When equity is deep in drawdown (within dd_score_boost_threshold %
                    # of the soft halt), raise the bar. This gates marginal setups in
                    # correlated instruments while still allowing genuinely high-conviction
                    # trades to fire naturally (score high enough to clear the boosted floor).
                    if self._risk_guard is not None:
                        _init_eq   = getattr(self._risk_guard, "_initial_equity", None)
                        _soft_pct  = float(self._trade_cfg.get("soft_dd_halt_pct", 7.0))
                        _boost_thr = float(self._trade_cfg.get("dd_score_boost_threshold", 1.5))
                        _boost_n   = int(self._trade_cfg.get("dd_score_boost", 2))
                        if _init_eq and _init_eq > 0:
                            _total_dd_pct = (_init_eq - equity) / _init_eq * 100
                            if _total_dd_pct >= (_soft_pct - _boost_thr):
                                old_needed = needed
                                needed = needed + _boost_n
                                logger.info(
                                    "[%s] DD-boost active (DD=%.2f%%, within %.1f%% of soft halt): "
                                    "score floor %d→%d",
                                    self.name, _total_dd_pct, _boost_thr, old_needed, needed)

                    if signal_score < needed:
                        logger.info("[%s] LOW CONVICTION skip: %s score=%d < %d (%s)",
                                    self.name, "BUY" if desired == 1 else "SELL",
                                    signal_score, needed,
                                    "counter-trend" if against_trend else "with-trend")
                        continue

                    # Quality-scaled risk (JP mentor): size scales DOWN for borderline setups.
                    # A soldier executes on strategy; he doesn't double-down on a shaky setup.
                    # 99% of prop account blowups = oversized trades that weren't that great.
                    _score_over_floor = signal_score - needed
                    if _score_over_floor <= 0:
                        score_mult = 0.25   # floor entry — borderline, reduce exposure hard
                    elif _score_over_floor == 1:
                        score_mult = 0.50   # decent setup — half risk
                    else:
                        score_mult = 1.00   # well above floor — full risk; never over-size

                    # Concentration mult: fewer concurrent positions = more size per trade
                    # 0-1 open → 2x  |  2-3 open → 1.5x  |  4+ open → 1x
                    n_open = len(trader.get_all_positions())   # magic-filtered (H3)
                    concentration_mult = 2.0 if n_open <= 1 else (1.5 if n_open <= 3 else 1.0)
                    combined_mult = size_mult * score_mult * concentration_mult

                    # ── Portfolio risk management (Council #03/#05) ────────────
                    base_risk = float(self._trade_cfg.get("risk_pct", 1.0))
                    max_port  = float(self._trade_cfg.get("max_portfolio_risk_pct", 4.0))
                    new_risk  = base_risk * combined_mult

                    if self._adaptive_port and self._allocator is not None:
                        # ACTIVE allocation (behaviours 1+2): size to remaining
                        # budget AND quality; if a stronger setup is starved, trim
                        # weaker/losing open trades to fund it rather than skip.
                        book, risk_map = self._portfolio_snapshot(equity)
                        alloc = self._allocator.allocate(signal_score, new_risk, book)
                        if not alloc.taken:
                            logger.warning("[%s] ALLOCATOR skip: %s", self.name, alloc.reason)
                            continue
                        if alloc.trims:
                            self._execute_trims(alloc.trims, risk_map)
                        logger.info("[%s] ALLOCATOR grant %.2f%% (intended %.2f%%) — %s",
                                    self.name, alloc.granted_pct, new_risk, alloc.reason)
                        combined_mult = alloc.granted_pct / base_risk if base_risk > 0 else combined_mult
                    else:
                        # PASSIVE cap (default): bound the entry to remaining budget.
                        open_risk = self._open_portfolio_risk_pct(equity)
                        room      = max_port - open_risk
                        if room <= 0.1:
                            logger.warning("[%s] PORTFOLIO RISK CAP: open=%.2f%% >= cap %.2f%% — entry blocked",
                                           self.name, open_risk, max_port)
                            continue
                        if new_risk > room:
                            scale = room / new_risk
                            combined_mult *= scale
                            logger.info("[%s] Portfolio cap: new-trade risk %.2f%% > room %.2f%% — scaled x%.3f",
                                        self.name, new_risk, room, scale)

                    sl, tp, lots = self._size_order(desired, account["balance"], combined_mult)

                    # ── Enforce a valid stop + honour size_order rejections (fix #1/#3) ──
                    # lots==0 means the stop was too tight (rejected). And we NEVER
                    # place an order without a real stop — no naked positions.
                    if lots <= 0:
                        continue   # already logged (stop too tight)
                    if sl is None or sl <= 0:
                        logger.warning("[%s] NO STOP — refusing to place a naked order", self.name)
                        continue

                    # ── Liquidity thesis gate: no clean draw within reach = no trade ──
                    # A trade with no real liquidity target is not a trade — it's
                    # arithmetic. Skip it rather than place a token lot at a fake TP.
                    direction_str = "BUY" if desired == 1 else "SELL"
                    if self._last_plan is not None and not self._last_plan.tradeable:
                        logger.info("[%s] NO-DRAW skip: %s (grade %s) — %s",
                                    self.name, direction_str,
                                    self._last_plan.grade, self._last_plan.thesis)
                        continue

                    # ── Per-trade agent: Council adjudication + durable ticket ──
                    # Uses the plan already computed in _size_order (no recompute).
                    tick_e    = mt5.symbol_info_tick(self._symbol)
                    entry_px  = (tick_e.ask if desired == 1 else tick_e.bid) if tick_e else 0.0
                    dd_daily_room, dd_total_room = self._dd_room(equity)
                    ticket = self._agent.evaluate(
                        symbol=self._symbol, direction=desired,
                        df_m15=None, df_m5=None, entry=entry_px, ref_stop=sl or 0.0,
                        atr=0.0, h4_bias=0, plan=self._last_plan,
                        gates={"m5_trigger": True},   # gates already passed above
                        dd_room_daily=dd_daily_room, dd_room_total=dd_total_room,
                    )
                    if ticket.verdict != "GO":
                        logger.info("[%s] AGENT NO_GO: %s | dissent: %s",
                                    self.name, direction_str, "; ".join(ticket.dissent))
                        continue

                    # ── Council gate: Telegram approval before order fires ──
                    require_approval = self._trade_cfg.get("require_approval", False)
                    if require_approval and not self._dry_run:
                        tick_now  = mt5.symbol_info_tick(self._symbol)
                        entry_est = tick_now.ask if desired == 1 else tick_now.bid
                        approved  = tg.request_approval(
                            symbol        = self._symbol,
                            direction     = desired,
                            score         = signal_score,
                            entry         = entry_est,
                            sl            = sl or 0.0,
                            tp            = tp,
                            lots          = lots,
                            equity        = equity,
                            council_notes = reason,
                            timeout_sec   = self._trade_cfg.get("approval_timeout_sec", 90),
                            auto_approve_score = self._trade_cfg.get("auto_approve_score", 6),
                        )
                        if not approved:
                            logger.info("[%s] Trade vetoed via Telegram", self.name)
                            continue

                    if self._dry_run:
                        logger.info("[%s] DRY RUN: %s %.2f lots SL=%s TP=%s | score=%d x%.2f | %s",
                                    self.name, direction_str, lots, sl, tp,
                                    signal_score, combined_mult, reason)
                    else:
                        ok = trader.place_order(self._symbol, desired, lots, sl=sl, tp=tp)
                        if ok:
                            tick = mt5.symbol_info_tick(self._symbol)
                            self._open_entry_price  = tick.ask if desired == 1 else tick.bid
                            self._open_sl           = sl
                            self._open_tp           = tp
                            self._open_score        = signal_score
                            self._t1_hit            = False
                            self._bars_since_entry  = 0
                            self._save_position_state()
                            # Register this trade's quality in the shared book so
                            # the allocator can weigh it against future setups.
                            if self._book is not None:
                                for _p in trader.get_positions(self._symbol):
                                    self._book.register(_p.ticket, signal_score)
                            if self._journal is not None:
                                import numpy as np
                                _atr_series = getattr(self._strategy, "_atr_cache", None)
                                _atr_val = float(_atr_series.iloc[-1]) if _atr_series is not None and not np.isnan(float(_atr_series.iloc[-1])) else None
                                _p = self._last_plan
                                _sr = getattr(self._strategy, "_score_reasons", None)
                                _reasons = list(_sr.iloc[-1]) if _sr is not None and len(_sr) else []
                                self._journal.open_trade(
                                    symbol=self._symbol,
                                    direction=desired,
                                    score=signal_score,
                                    entry_price=self._open_entry_price,
                                    sl_price=sl or 0.0,
                                    tp_price=tp,
                                    lots=lots,
                                    equity=equity,
                                    atr=_atr_val,
                                    df=df,
                                    trade_type=_p.trade_type if _p else None,
                                    grade=_p.grade if _p else None,
                                    target_price=_p.tp if _p else None,
                                    thesis=_p.thesis if _p else None,
                                    reasons=_reasons,
                                )

            except Exception as exc:
                self.registry.fail(self.name, str(exc))
                logger.exception("[%s] Error: %s", self.name, exc)
                time.sleep(30)

        logger.info("[%s] Stopped.", self.name)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """Floor manager — owns every component and monitors their heartbeats."""

    def __init__(self, cfg: dict, dry_run: bool = False, symbol_override: Optional[str] = None):
        self.cfg          = cfg
        self.dry_run      = dry_run
        self.kill_switch  = threading.Event()
        self.soft_halt    = threading.Event()   # 2% daily or 7% cumulative — no new entries
        self.registry     = HeartbeatRegistry()
        self._threads: list   = []
        self._components: list = []

        mt5_cfg   = cfg.get("mt5", {})
        data_cfg  = cfg.get("data", {})
        trade_cfg = cfg.get("trading", {})

        self._terminal_path = mt5_cfg.get("terminal_path") or None
        self._tf_str        = data_cfg.get("timeframe", "M15").upper()
        self._symbols       = (
            [symbol_override] if symbol_override
            else data_cfg.get("symbols", ["XAUUSD"])
        )
        self._trade_cfg = trade_cfg
        trader.set_magic(trade_cfg.get("magic", 234001))

        # Build per-symbol strategy instances
        strategy_name = trade_cfg.get("strategy", "aiden_index")
        if strategy_name == "aiden_index":
            self._strategies = {s: _build_aiden_strategy(s) for s in self._symbols}
        else:
            strategy_cls = STRATEGY_MAP.get(strategy_name, FVGOrderBlockStrategy)
            self._strategies = {s: strategy_cls() for s in self._symbols}

        # Shared RiskAgent + FTMO tracker — both persist state to disk
        initial_equity = trade_cfg.get("initial_equity", 10_000)
        challenge_type = trade_cfg.get("ftmo_challenge", "2step-p1")
        self._risk_agent   = RiskAgent(
            initial_equity=initial_equity,
            config=RiskConfig(
                max_account_dd_pct=float(trade_cfg.get("total_kill_pct", 9.5)) / 100,
                max_daily_loss_pct=float(trade_cfg.get("daily_halt_pct", 2.0)) / 100,
                max_weekly_dd_pct=float(trade_cfg.get("soft_dd_halt_pct", 9.0)) / 100,
            ),
        )
        self._ftmo_tracker = FTMOTracker(initial_equity=initial_equity, challenge=challenge_type)
        log_cfg            = cfg.get("logging", {})
        self._journal      = TradeJournal(log_dir=log_cfg.get("log_dir", "logs"))

    # ── Startup ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info("=" * 60)
        logger.info("  ORCHESTRATOR STARTING")
        logger.info("  symbols=%s | tf=%s | dry_run=%s", self._symbols, self._tf_str, self.dry_run)
        logger.info("=" * 60)

        if not connect(self._terminal_path):
            logger.critical("Cannot connect to MT5. Aborting.")
            tg.notify_council_flag("01 Principal", "MT5 connection FAILED on startup. Bot aborted.")
            sys.exit(1)

        self._news_gate = ng.NewsGate()
        self._news_gate.start()

        # Log FTMO challenge status before anything trades
        try:
            account = mt5.account_info()
            if account:
                for line in self._ftmo_tracker.report(account.equity).splitlines():
                    logger.info(line)
        except Exception:
            pass

        if not self._safety_check():
            disconnect()
            sys.exit(1)

        account = mt5.account_info()
        tg.notify_startup(self._symbols, self.dry_run, account.equity if account else 0.0)

        # Build the shared risk components first so the engines can reference them:
        #  - RiskGuard owns the single server-day daily baseline (engines read it)
        #  - TradeReconciler is the single outcome recorder (engines trigger it
        #    synchronously on close)
        risk_guard = RiskGuard(
            self.registry, self.kill_switch,
            soft_halt_event=self.soft_halt,
            initial_equity=self._ftmo_tracker.state.initial_equity,
            symbols=self._symbols,
            daily_halt_pct=float(self._trade_cfg.get("daily_halt_pct", MAX_DAILY_LOSS_PCT)),
            soft_dd_pct=float(self._trade_cfg.get("soft_dd_halt_pct", SOFT_DD_HALT_PCT)),
            total_kill_pct=float(self._trade_cfg.get("total_kill_pct", MAX_TOTAL_LOSS_PCT)),
        )
        reconciler = TradeReconciler(
            self.registry, self.kill_switch,
            risk_agent=self._risk_agent,
            ftmo_tracker=self._ftmo_tracker,
            journal=self._journal,
            magic=self._trade_cfg.get("magic", 234001),
            risk_pct=self._trade_cfg.get("risk_pct", 1.0),
            initial_equity=self._ftmo_tracker.state.initial_equity,
        )
        self._risk_guard = risk_guard
        self._reconciler = reconciler

        # Shared adaptive allocator + cross-engine quality book (behaviours 1-3).
        # Gated by trade_cfg['adaptive_portfolio'] inside each engine.
        self._book  = PortfolioBook()
        self._alloc = PortfolioAllocator(
            daily_budget_pct=float(self._trade_cfg.get("max_portfolio_risk_pct", 4.0)),
            score_edge=int(self._trade_cfg.get("realloc_score_edge", 1)),
            min_trade_pct=float(self._trade_cfg.get("min_trade_risk_pct", 0.25)),
        )

        components: list[Component] = [
            MT5Monitor(self.registry, self.kill_switch, self._terminal_path),
            DataWatcher(self.registry, self.kill_switch, self._symbols, self._tf_str),
            risk_guard,
            reconciler,
        ]

        level_monitor = LevelMonitor(
            proximity_atr=float(self._trade_cfg.get("level_proximity_atr", 1.0))
        )

        for symbol in self._symbols:
            engine = TradingEngine(
                self.registry,
                self.kill_switch,
                symbol,
                self._strategies[symbol],
                self._tf_str,
                self._trade_cfg,
                risk_agent=self._risk_agent,
                ftmo_tracker=self._ftmo_tracker,
                soft_halt_event=self.soft_halt,
                dry_run=self.dry_run,
                journal=self._journal,
                reconciler=reconciler,
                risk_guard=risk_guard,
                allocator=self._alloc,
                book=self._book,
            )
            engine._news_gate    = self._news_gate
            engine._level_monitor = level_monitor
            components.append(engine)

        self._components = components

        for comp in components:
            t = threading.Thread(target=self._run_component, args=(comp,),
                                 name=comp.name, daemon=True)
            t.start()
            self._threads.append(t)

        self._monitor_loop()

    # ── Monitor loop ─────────────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        try:
            while not self.kill_switch.is_set():
                self._print_dashboard()
                time.sleep(MONITOR_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user.")
            self.kill_switch.set()
        finally:
            self._shutdown()

    def _print_dashboard(self) -> None:
        rows = self.registry.snapshot()
        now  = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = [
            "",
            f"  ╔══ ORCHESTRATOR DASHBOARD  {now} ══╗",
            f"  {'Component':<30} {'Status':<10} {'Last Beat':>10}  {'Message'}",
            f"  {'─'*30} {'─'*10} {'─'*10}  {'─'*30}",
        ]
        for r in rows:
            age_str = f"{r['last_beat_secs']}s ago"
            lines.append(
                f"  {r['name']:<30} {r['status']:<10} {age_str:>10}  {r['message'][:60]}"
            )
        lines.append(f"  {'─'*82}")

        try:
            account = mt5.account_info()
            if account:
                lines.append(f"  {self._ftmo_tracker.status_line(account.equity)}")
        except Exception:
            pass

        lines.append("")
        for line in lines:
            logger.info(line)

    # ── Component runner with auto-restart ───────────────────────────────────

    def _run_component(self, comp: Component) -> None:
        backoff = 5
        while not self.kill_switch.is_set():
            try:
                comp.run()
            except Exception as exc:
                logger.exception("[%s] Crashed: %s", comp.name, exc)
                self.registry.fail(comp.name, str(exc))

            if self.kill_switch.is_set():
                break

            restarts = self.registry.increment_restarts(comp.name)
            if restarts >= MAX_RESTARTS:
                msg = f"{comp.name} exceeded MAX_RESTARTS ({MAX_RESTARTS}). Triggering kill switch."
                logger.critical(msg)
                self.registry.halt(comp.name, msg)
                self.kill_switch.set()
                break

            logger.warning("[%s] Restarting in %ds (restart #%d)...", comp.name, backoff, restarts)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)  # cap at 5 min

    # ── Safety checks ─────────────────────────────────────────────────────────

    def _safety_check(self) -> bool:
        account = mt5.account_info()
        if account is None:
            logger.error("No account info — is MT5 open and logged in?")
            return False

        is_demo    = account.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
        allow_real = self.cfg.get("trading", {}).get("allow_real_account", False)

        if not is_demo and not allow_real:
            logger.error(
                "BLOCKED: real account detected and 'allow_real_account' is not set."
            )
            return False

        if not is_demo:
            logger.warning("WARNING: trading on a REAL account.")

        return True

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        logger.info("Orchestrator shutting down — stopping all components...")
        for comp in self._components:
            comp.stop()
        for t in self._threads:
            t.join(timeout=15)
        if hasattr(self, "_news_gate"):
            self._news_gate.stop()
        disconnect()
        self._journal.push_to_github()
        account = mt5.account_info()
        equity  = account.equity if account else 0.0
        session_pnl = equity - (self._risk_agent.initial_equity if hasattr(self._risk_agent, "initial_equity") else equity)
        tg.notify_shutdown(equity, session_pnl)
        logger.info("Orchestrator stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trading bot orchestrator.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log signals only — no real orders placed")
    parser.add_argument("--symbol", default=None,
                        help="Single symbol (default: all in config)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg     = load_config(*([args.config] if args.config else []))
    log_cfg = cfg.get("logging", {})
    setup_logger(
        "",
        log_dir=log_cfg.get("log_dir", "logs"),
        log_file="orchestrator.log",
        level=log_cfg.get("level", "INFO"),
    )

    orch = Orchestrator(cfg, dry_run=args.dry_run, symbol_override=args.symbol)
    orch.start()


if __name__ == "__main__":
    main()
