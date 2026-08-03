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
import os
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
from execution.regime_classifier import classify_regime, entry_gate as regime_entry_gate, size_mult as regime_size_mult
from execution.portfolio_optimizer import CorrelationMatrix, var_cvar
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
from core.amd_detector import AMDDetector, SweepEvent
from execution.market_context_agent import MarketContextAgent, MarketContext
from execution.probability_model import ProbabilityModel, TradeConfluences
from execution.order_flow import analyse_order_flow, order_flow_score_modifier, get_dom_key_levels
from execution import telegram_notify as tg
from execution.telegram_commands import TelegramCommandHandler
from backtests.run_multi_instrument import (
    INSTRUMENTS, OPTIMISED_PARAMS, TRAIL_CONFIGS, BIDIRECTIONAL, M15_PARAMS,
    _size_mult_from_score,
)
from core.instrument_profile import PROFILES, build_strategy_kwargs
from core.probability_stack import StackInput, shadow_logger as _shadow_logger
from core.analyzer_engine import AnalyzerEngine
from news.news_intelligence import NewsIntelligence, NewsSignal

logger = logging.getLogger(__name__)

# ── Risk limits (Council of 12 — tighter than FTMO stated limits) ────────────
# Daily DD: tiered gating, not a flat halt. Score floor rises as loss grows.
# Tiers: 1.5% → +1 score, 2.5% → +2 score, 3.5% → +3 score, 4.0% → hard halt.
# Preserves 1% buffer to FTMO's 5% daily limit.
MAX_DAILY_LOSS_PCT  = 4.0   # hard daily halt (FTMO limit 5%) — was 2.0 flat block
DAILY_DD_TIER1_PCT  = 1.5   # score floor +1
DAILY_DD_TIER2_PCT  = 2.5   # score floor +2
DAILY_DD_TIER3_PCT  = 3.5   # score floor +3
SOFT_DD_HALT_PCT    = 7.0   # soft halt: no new entries, let open trades run (FTMO limit 10%)
MAX_TOTAL_LOSS_PCT  = 9.5   # hard kill switch: emergency close all (FTMO 10% hard floor)
MAX_RECONNECT_TRIES = 5
MAX_RESTARTS        = 10
MONITOR_INTERVAL    = 60    # seconds between orchestrator health checks
# ─────────────────────────────────────────────────────────────────────────────

# ── ICT / JP methodology helpers ─────────────────────────────────────────────

def _in_ict_macro(dt: "pd.Timestamp") -> bool:
    """True if the bar falls inside an ICT algorithm delivery macro window (UTC).

    ICT Macro windows are 15-20 minute periods where the market maker algorithm
    delivers price efficiently. A bar inside these windows has higher probability
    of following through on the setup direction.

    Times are UTC. Provisional — sourced from ICT community teaching; not yet
    verified against JP transcripts or confirmed on live data.
    """
    t = dt.hour * 60 + dt.minute
    _WINDOWS = [
        (2*60+33, 3*60),        # 02:33-03:00 — Pre-London CBDR macro
        (4*60+3,  4*60+30),     # 04:03-04:30 — Early London build
        (8*60+50, 9*60+10),     # 08:50-09:10 — London open kill zone
        (9*60+50, 10*60+10),    # 09:50-10:10 — Mid-London
        (10*60+50, 11*60+10),   # 10:50-11:10 — Late London
        (11*60+50, 12*60+10),   # 11:50-12:10 — Pre-NY
        (13*60+10, 13*60+40),   # 13:10-13:40 — NY open kill zone
        (14*60+50, 15*60+10),   # 14:50-15:10 — PM session open
        (15*60+15, 15*60+45),   # 15:15-15:45 — PM afternoon macro
    ]
    return any(lo <= t < hi for lo, hi in _WINDOWS)


def _ipda_aligned(df: "pd.DataFrame", direction: int, price: float) -> bool:
    """True if trade direction aligns with IPDA 20/40/60-day delivery.

    IPDA (Interbank Price Delivery Algorithm): price tends to deliver from one
    range extreme to the other over 20/40/60 trading days. If price is at the
    low of the 20-day range (discount), the IPDA target is the 20-day high →
    favor longs. If at the high (premium), favor shorts.

    Uses 20-day range as primary signal; falls back neutral if insufficient data.
    """
    try:
        times = pd.to_datetime(df["time"])
        tmp   = df[["high", "low", "close"]].copy()
        tmp.index = times
        d1 = tmp.resample("1D", closed="left", label="left").agg(
            {"high": "max", "low": "min", "close": "last"}
        ).dropna()
        if len(d1) < 20:
            return False
        recent = d1.tail(20)
        hi20 = float(recent["high"].max())
        lo20 = float(recent["low"].min())
        rng  = hi20 - lo20
        if rng <= 0:
            return False
        pos = (price - lo20) / rng  # 0=at low, 1=at high
        return (pos < 0.30 and direction == 1) or (pos > 0.70 and direction == -1)
    except Exception:
        return False


def _inside_day(df: "pd.DataFrame") -> bool:
    """True if today's daily range is inside yesterday's — JP: 'no daily sweep expected'.

    Resamples M15 bars to D1 and checks whether the current day's high/low
    is contained within the prior day's high/low (inside bar pattern).
    """
    try:
        times = pd.to_datetime(df["time"])
        tmp   = df[["high", "low"]].copy()
        tmp.index = times
        d1 = tmp.resample("1D", closed="left", label="left").agg(
            {"high": "max", "low": "min"}
        ).dropna()
        if len(d1) < 2:
            return False
        prev = d1.iloc[-2]
        curr = d1.iloc[-1]
        return bool(curr["high"] < prev["high"] and curr["low"] > prev["low"])
    except Exception:
        return False


# ── SMT divergence (JP mentor: "cousin pairs") ───────────────────────────────
# When EURUSD sweeps its Asian low but GBPUSD does NOT make a new low simultaneously,
# the EU move is a fake trap. The divergence signals the real direction.
# Cache stores last 20-bar swing for each symbol, updated per bar.

_smt_cache: dict[str, dict] = {}   # {symbol: {"lo": float, "hi": float}}

_SMT_COUSINS: dict[str, str] = {
    "EURUSD": "GBPUSD",   "GBPUSD": "EURUSD",
    "XAUUSD": "XAGUSD",   "XAGUSD": "XAUUSD",
    "US30":   "US100",    "US100":  "US30",
    "US500":  "US100",
}

_SMT_WINDOW = 20   # bars to look back for swing extreme comparison


def _smt_divergence_present(symbol: str, df: "pd.DataFrame", direction: int) -> bool:
    """True when our symbol swept a new swing extreme but the cousin pair did NOT.

    JP mentor (TR3,4,5,8): "GU is EU's ugly cousin. When EU takes Asian lows
    and GU hasn't, that divergence is your signal."
    """
    sym_key = symbol.replace(".cash", "").replace(".fx", "").upper()
    cousin  = _SMT_COUSINS.get(sym_key)
    if cousin is None or cousin not in _smt_cache:
        return False
    try:
        window = min(_SMT_WINDOW, len(df))
        if window < 5:
            return False
        our_lo = float(df["low"].tail(window).min())
        our_hi = float(df["high"].tail(window).max())
        cousin_data = _smt_cache[cousin]
        if direction == 1:
            # Long: our symbol swept to new 20-bar low; cousin did NOT also make new low
            prior_lo = float(df["low"].tail(window + 5).min())
            swept_new_low = our_lo <= prior_lo * 1.001
            cousin_also_low = cousin_data["lo"] <= cousin_data.get("prior_lo", cousin_data["lo"]) * 1.001
            return swept_new_low and not cousin_also_low
        else:
            # Short: our symbol swept to new high; cousin did NOT
            prior_hi = float(df["high"].tail(window + 5).max())
            swept_new_hi = our_hi >= prior_hi * 0.999
            cousin_also_hi = cousin_data["hi"] >= cousin_data.get("prior_hi", cousin_data["hi"]) * 0.999
            return swept_new_hi and not cousin_also_hi
    except Exception:
        return False


def _update_smt_cache(symbol: str, df: "pd.DataFrame") -> None:
    """Update swing cache for this symbol. Called once per bar per engine."""
    try:
        w = min(_SMT_WINDOW, len(df))
        pw = min(_SMT_WINDOW + 5, len(df))
        _smt_cache[symbol.replace(".cash","").replace(".fx","").upper()] = {
            "lo":       float(df["low"].tail(w).min()),
            "hi":       float(df["high"].tail(w).max()),
            "prior_lo": float(df["low"].tail(pw).min()),
            "prior_hi": float(df["high"].tail(pw).max()),
        }
    except Exception:
        pass


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

# Cousin pairs: when both qualify in same direction on the same bar, only the
# higher-scoring one enters (JP mentor: "GU and EU are cousins — pick the better one").
_COUSIN_PAIRS: list[frozenset] = [
    frozenset({"GBPUSD", "EURUSD"}),
    # US equity indices: highly correlated — only the highest-scoring fires per bar
    frozenset({"US30.cash", "US500.cash", "US100.cash", "US2000.cash"}),
]


class CousinRouter:
    """Thread-safe quality router for correlated cousin pairs.

    Engine calls try_claim(bar_time, symbol, score, direction) before placing
    an order.  Returns True if cleared to enter, False if a better cousin
    already claimed this bar+direction slot.
    """

    def __init__(self):
        self._lock  = threading.Lock()
        # key: (bar_time, frozenset_pair, direction) → (symbol, score)
        self._slots: dict = {}

    def _pair_for(self, symbol: str) -> frozenset | None:
        for pair in _COUSIN_PAIRS:
            if symbol in pair:
                return pair
        return None

    def try_claim(
        self, bar_time: int, symbol: str, score: int, direction: int
    ) -> bool:
        pair = self._pair_for(symbol)
        if pair is None:
            return True   # not a cousin pair — always cleared
        key = (bar_time, pair, direction)
        with self._lock:
            existing = self._slots.get(key)
            if existing is None:
                self._slots[key] = (symbol, score)
                return True
            ex_sym, ex_score = existing
            if ex_sym == symbol:
                return True   # same engine re-checking
            if score > ex_score:
                # this engine wins — take the slot, block the previous claimant
                self._slots[key] = (symbol, score)
                return True
            return False   # a better-scoring cousin already claimed it

    def cleanup(self, current_bar_time: int):
        """Drop slots older than 2 bars to prevent unbounded growth."""
        with self._lock:
            stale = [k for k in self._slots if k[0] < current_bar_time - 2]
            for k in stale:
                del self._slots[k]


STRATEGY_MAP = {
    "sniper":          SniperStrategy,
    "london_breakout": LondonBreakoutStrategy,
    "fvg_ob":          FVGOrderBlockStrategy,
    "aiden_index":     AiDENIndexStrategy,
}


def _build_aiden_strategy(symbol: str) -> AiDENIndexStrategy:
    """Build a per-symbol AiDENIndexStrategy with correct live params.

    Phase 0: params now come from core.instrument_profile.PROFILES, whose
    strategy_kwargs replicate the previous inline assembly exactly. Unknown
    symbols fall back to build_strategy_kwargs (same defaults as before).
    """
    if symbol in PROFILES:
        return AiDENIndexStrategy(**PROFILES[symbol].strategy_kwargs)
    return AiDENIndexStrategy(**build_strategy_kwargs(symbol))


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
    STATE_FILE = Path(__file__).parent.parent / "logs" / "risk_guard_state.json"

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

    def _reconstruct_day_start_equity(self, server_day: str) -> float:
        """Compute true day-start equity from MT5 deal history.

        Called whenever the baseline can't be trusted (fresh start or restart
        after trades already closed today). Avoids the re-anchoring bug where
        a post-restart baseline silently masks prior realized losses.
        """
        try:
            info = mt5.account_info()
            if info is None:
                return self._initial_equity
            day_start_dt = datetime.strptime(server_day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            deals = mt5.history_deals_get(day_start_dt, datetime.now(timezone.utc))
            if not deals:
                return float(info.balance)
            today_pnl = sum(d.profit + d.commission + d.swap for d in deals)
            reconstructed = float(info.balance) - today_pnl
            logger.warning(
                "[RiskGuard] Reconstructed day_start=%.2f from %d deals (today_pnl=%.2f). "
                "Bot restarted mid-day — prior losses now visible to circuit breaker.",
                reconstructed, len(deals), today_pnl,
            )
            return reconstructed
        except Exception as exc:
            logger.error("[RiskGuard] day_start reconstruction failed: %s — falling back to initial_equity", exc)
            return self._initial_equity

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
                self._day_start_equity = self._reconstruct_day_start_equity(server_day)
                if (self._soft_halt.is_set() and self._halt_cause == "daily"
                        and total_pct > -self._soft_dd_pct):
                    self._soft_halt.clear()
                    self._halt_cause = None
                    logger.info("[RiskGuard] New server day %s — daily halt cleared, baseline=%.2f",
                                server_day, self._day_start_equity)
                else:
                    logger.info("[RiskGuard] New server day %s — daily baseline=%.2f",
                                server_day, self._day_start_equity)
                self._save_state()

            if self._day_start_equity is None:
                self._day_start_equity = self._reconstruct_day_start_equity(server_day)
                self._save_state()

            daily_pct = (
                (equity - self._day_start_equity) / self._day_start_equity * 100
                if self._day_start_equity else 0.0
            )

            msg = (f"eq={equity:.2f} daily={daily_pct:+.2f}% total={total_pct:+.2f}%"
                   f"{' [SOFT-HALT]' if self._soft_halt.is_set() else ''}")

            # Tier 1: daily hard halt at 4% — score tiers (1.5/2.5/3.5%) gate entries before this
            if daily_pct <= -self._daily_halt_pct and not self._soft_halt.is_set():
                logger.critical("[RiskGuard] DAILY HARD HALT: %.2f%% — no new entries today", daily_pct)
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
    STATE_FILE = Path(__file__).parent.parent / "logs" / "reconciler_state.json"

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
                    self._journal.close_trade(symbol, last_out.price, equity, realized_pnl=realized)
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
        self._cousin_router:   Optional["CousinRouter"] = None
        self._adaptive_port    = bool(trade_cfg.get("adaptive_portfolio", False))
        self._lookback         = trade_cfg.get("lookback_bars", 500)
        self._level_monitor:   Optional[LevelMonitor] = None
        self._amd_detector:    Optional[AMDDetector]  = None
        self._mc_agent:        Optional[MarketContextAgent] = None
        self._corr_matrix:     Optional[CorrelationMatrix] = None
        self._prob_model:      ProbabilityModel = ProbabilityModel()
        self._open_confluences: Optional[TradeConfluences] = None  # confluences at entry time
        self._scout_regime:    dict = {}   # last cadre_regime_state.json payload
        self._open_peak_r:     float = 0.0  # best R this trade — profit-lock ladder input
        # Burned targets: (direction, tp) → expiry ts. Blocks re-entering the
        # same idea for 90 min after it traded (audit A3: 4 US30 longs at the
        # same target in 30 min, all losers). Persisted across restarts.
        self._burned_targets:  dict = {}
        self._load_burned_targets()
        self._news_gate:        Optional[ng.NewsGate] = None
        self._news_intel:       Optional[NewsIntelligence] = None
        self._consec_losses:    int   = 0       # anti-tilt: consecutive losing trades
        self._entry_equity:     float = 0.0     # equity at entry — used to detect win/loss at close
        self._event_dir_cache: dict[str, int] = {}
        self._trade_manager:   TradeManager = TradeManager()
        self._last_bar         = None
        self._open_entry_price: Optional[float] = None
        self._open_sl:          Optional[float] = None   # initial SL at entry — never moved
        self._open_tp:          Optional[float] = None
        self._last_plan               = None    # last TradePlan from analyze_entry
        self._agent            = TradeAgent()   # per-trade adjudication + tickets
        self._open_score:       int = 0
        self._t1_hit:              bool = False
        self._bars_since_entry:    int  = 0
        self._consecutive_waits:   int  = 0
        self._entry_cooldown_until: float = 0.0  # epoch time; TM exits blocked until then
        self._last_partial_bar:    int   = 0     # bar timestamp of last PARTIAL_CLOSE
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

            # If MT5 has an open position but the journal has no entry for it,
            # create a recovery entry so the trade is always journaled.
            if self._journal is not None and self._symbol not in self._journal._open:
                try:
                    pos = positions[0]
                    pos_dir = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                    self._journal.open_trade(
                        symbol=self._symbol, direction=pos_dir,
                        score=self._open_score, entry_price=pos.price_open,
                        sl_price=pos.sl or self._open_sl or 0.0,
                        tp_price=pos.tp or self._open_tp,
                        lots=pos.volume, equity=pos.price_open,
                        trade_type="recovered", grade="B",
                        thesis=f"[RECOVERED ON RESTART] entry={pos.price_open:.5f}",
                        reasons=["bot_restart_recovery"],
                    )
                    logger.info("[%s] Journal recovery entry created for existing position", self.name)
                except Exception as _re:
                    logger.warning("[%s] Journal recovery failed: %s", self.name, _re)
        except Exception:
            pass

    _BURNED_TARGETS_FILE = Path(__file__).resolve().parent.parent / "logs" / "burned_targets.json"
    _BURN_TTL_SECS = 90 * 60

    def _load_burned_targets(self) -> None:
        """Load unexpired burned targets for this symbol (survives restarts —
        the 2026-07-27 revenge loop spanned bot restarts)."""
        try:
            if not self._BURNED_TARGETS_FILE.exists():
                return
            data = json.loads(self._BURNED_TARGETS_FILE.read_text(encoding="utf-8"))
            now = time.time()
            for k, expiry in data.get(self._symbol, {}).items():
                if expiry > now:
                    d_str, tp_str = k.split("|")
                    self._burned_targets[(int(d_str), float(tp_str))] = expiry
        except Exception:
            pass

    def _save_burned_target(self, direction: int, tp: float) -> None:
        expiry = time.time() + self._BURN_TTL_SECS
        self._burned_targets[(direction, round(tp, 5))] = expiry
        try:
            data = {}
            if self._BURNED_TARGETS_FILE.exists():
                data = json.loads(self._BURNED_TARGETS_FILE.read_text(encoding="utf-8"))
            sym = data.setdefault(self._symbol, {})
            sym[f"{direction}|{round(tp, 5)}"] = expiry
            now = time.time()
            for s in list(data.keys()):
                data[s] = {k: v for k, v in data[s].items() if v > now}
            self._BURNED_TARGETS_FILE.write_text(json.dumps(data), encoding="utf-8")
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
            # Floor raised 0.5→0.75 ATR after 2026-07-27: two 0.68-ATR stops on
            # US30 (54 lots each) noise-stopped within 15s — audit finding A4.
            min_atr  = atr_val * float(self._trade_cfg.get("min_stop_atr_mult", 0.75))
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

        # ── Max notional cap ──────────────────────────────────────────────────
        # Notional must be in account currency (USD). For pairs where USD is the
        # BASE (e.g. USDJPY), notional = lots * contract (already USD — don't
        # multiply by price, which would give JPY notional and massively over-cap).
        tick_c = mt5.symbol_info_tick(self._symbol)
        info_c = mt5.symbol_info(self._symbol)
        px_c   = (tick_c.ask if direction == 1 else tick_c.bid) if tick_c else 0.0
        if info_c is not None and px_c > 0 and lots > 0:
            contract      = getattr(info_c, "trade_contract_size", 1.0) or 1.0
            currency_base = getattr(info_c, "currency_base", "")
            acc_currency  = "USD"
            if currency_base == acc_currency:
                notional_usd = lots * contract
            else:
                notional_usd = lots * contract * px_c
            max_notional = balance * float(self._trade_cfg.get("max_notional_x", 30))
            if max_notional > 0 and notional_usd > max_notional:
                capped = risk._clamp_lots(self._symbol, lots * max_notional / notional_usd)
                logger.warning("[%s] NOTIONAL CAP: %.2f -> %.2f lots (notional_usd %.0f > cap %.0f)",
                               self.name, lots, capped, notional_usd, max_notional)
                lots = capped

        lots = max(lots, 0.01)
        return sl, tp, lots

    def _reset_position_state(self) -> None:
        self._open_entry_price  = None
        self._open_sl           = None
        self._open_tp           = None
        self._open_score        = 0
        self._t1_hit            = False
        self._bars_since_entry       = 0
        self._consecutive_waits      = 0
        self._entry_cooldown_until   = 0.0
        self._last_partial_bar       = 0
        self._open_confluences       = None
        self._open_peak_r            = 0.0
        self._scaled_in              = False
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
            # Bayesian update: did this trade win? Feed outcome back to probability model.
            _exit_r = action.counter_score   # counter_score used as proxy; reconciler computes real R
            _won = pos.profit > 0 if hasattr(pos, "profit") else False
            if self._open_confluences is not None:
                try:
                    _confl_dict = {
                        "fvg_present":        self._open_confluences.fvg_present,
                        "ob_present":         self._open_confluences.ob_present,
                        "m5_confirmed":       self._open_confluences.m5_confirmed,
                        "h4_aligned":         self._open_confluences.h4_aligned,
                        "at_htf_level":       self._open_confluences.at_htf_level,
                        "order_flow_aligned": self._open_confluences.order_flow_aligned,
                        "dom_aligned":        self._open_confluences.dom_aligned,
                        "news_aligned":       self._open_confluences.news_aligned,
                        "continuation_type":  self._open_confluences.trade_type == "continuation",
                        "in_ict_macro":       self._open_confluences.in_ict_macro,
                        "ipda_aligned":       self._open_confluences.ipda_aligned,
                        "smt_divergence":     self._open_confluences.smt_divergence,
                        "eq_liq_cluster":     self._open_confluences.eq_liq_cluster,
                        "early_leakage":      self._open_confluences.early_leakage,
                        "inside_day":         self._open_confluences.inside_day,
                    }
                    self._prob_model.update_from_outcome(_confl_dict, _won)
                except Exception:
                    pass
            if not self._dry_run:
                trader.close_all(self._symbol)
                self._record_close_now()   # prompt, deduped accounting
            self._reset_position_state()
            self._block_entry = True

        elif action.action == ActionType.WAIT:
            self._consecutive_waits += 1
            logger.info("[%s] TM WAIT — sweep_risk=%.2f, holding this bar (consecutive=%d)",
                        self.name, action.sweep_risk, self._consecutive_waits)

        elif action.action == ActionType.PARTIAL_CLOSE:
            self._consecutive_waits = 0
            self._last_partial_bar  = self._last_bar   # dedup: one PARTIAL_CLOSE per bar
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

    @staticmethod
    def _read_scout_regime() -> dict:
        """Load Scout's latest regime assessment from cadre_regime_state.json.

        Scout writes this file every 30 min. We treat it as stale if > 35 min old
        so we never block on a missing Scout run.
        Returns empty dict if unavailable or stale.
        """
        regime_file = Path(__file__).parent.parent / "logs" / "cadre_regime_state.json"
        if not regime_file.exists():
            return {}
        try:
            age = time.time() - regime_file.stat().st_mtime
            if age > 35 * 60:
                return {}
            return json.loads(regime_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # DXY directional impact per instrument:
    # +1 DXY bullish (USD strong): EURUSD -1, GBPUSD -1, XAUUSD -1, US30 +1, US100 +1
    _DXY_INSTRUMENT_BIAS: dict[str, int] = {
        "EURUSD": -1, "GBPUSD": -1, "USDJPY": +1, "USDCHF": +1,
        "XAUUSD": -1, "XAGUSD": -1,
        "US30": +1, "US100": +1, "US500": +1, "US2000": +1,
        "JP225": -1,
    }

    def _scout_regime_aligned(self, symbol: str, direction: int) -> bool:
        """True if Scout's DXY bias aligns with the proposed trade direction."""
        regime = self._scout_regime
        dxy_bias = regime.get("dxy_bias", 0)
        if dxy_bias == 0:
            return False
        # Find the instrument's DXY relationship
        sym_base = symbol.upper().replace("_SB", "").replace(".", "")
        instr_dxy_dir = self._DXY_INSTRUMENT_BIAS.get(sym_base, 0)
        if instr_dxy_dir == 0:
            return False
        # Aligned if: DXY bias * instrument_direction == proposed_direction
        expected_dir = dxy_bias * instr_dxy_dir
        return expected_dir == direction

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

        # ── Hard profit-lock floor (Anton rule: NEVER go from profit to loss) ──
        # Independent of TRAIL_CONFIGS. If trade is up ≥ 0.5R and SL is still
        # below entry (for longs) or above entry (for shorts), force SL to BE.
        # This fires every poll cycle so an intrabar spike to profit can't reverse
        # all the way to the original stop.
        cur_r_now = (
            (mid_price - entry) / risk_dist if pos.type == mt5.ORDER_TYPE_BUY
            else (entry - mid_price) / risk_dist
        )
        if cur_r_now >= 0.5:
            be_floor = entry
            if pos.type == mt5.ORDER_TYPE_BUY and current_sl < be_floor - 1e-8:
                logger.info("[%s] PROFIT LOCK: %.2fR → forcing SL to BE %.5f",
                            self.name, cur_r_now, be_floor)
                self._tighten_all_sl(be_floor, 1)
                current_sl = be_floor
            elif pos.type == mt5.ORDER_TYPE_SELL and current_sl > be_floor + 1e-8:
                logger.info("[%s] PROFIT LOCK: %.2fR → forcing SL to BE %.5f",
                            self.name, cur_r_now, be_floor)
                self._tighten_all_sl(be_floor, -1)
                current_sl = be_floor

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
            # Asian session 50% midpoint — structural BE trigger (JP mentor v9)
            _asian_50: Optional[float] = None
            if self._level_monitor is not None:
                for _lv in self._level_monitor.get_levels(self._symbol):
                    if _lv.label == "Asian 50% Mid":
                        _asian_50 = _lv.price
                        break
            _t1_price: Optional[float] = (
                self._last_plan.t1_price if self._last_plan is not None else None
            )
            mdec = analyze_manage(
                df=df_mgmt, direction=pos_dir, entry=entry, initial_sl=init_sl,
                current_sl=live_sl, current_tp=live_tp, price=mid_price,
                atr=atr_m, trade_type=ttype, cur_r=cur_r, bank_min_r=bank_r,
                asian_50=_asian_50, t1_price=_t1_price,
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
            direction          = tm_pos_dir,
            entry_price        = self._open_entry_price,
            initial_sl         = self._open_sl,
            current_sl         = pos.sl,
            current_tp         = pos.tp,
            current_price      = mid_price,
            bars_elapsed       = self._bars_since_entry,
            t1_hit             = self._t1_hit,
            h4_bias            = h4_bias,
            consecutive_waits  = self._consecutive_waits,
            peak_r             = self._open_peak_r,
        )
        # Ratchet the peak — feeds the TM profit-lock ladder
        self._open_peak_r = max(self._open_peak_r, pos_state.current_r)

        # Portfolio P&L in R units: sum of all open positions' floating P&L
        # divided by the per-trade risk. Protects gains by tightening losers.
        portfolio_pnl_r = self._portfolio_pnl_r()

        # Entry cooldown: block TM exits for 60s after entry to absorb bar-open noise
        _in_cooldown = time.time() < self._entry_cooldown_until
        if _in_cooldown:
            logger.debug("[%s] TM skip — entry cooldown (%ds left)",
                         self.name, int(self._entry_cooldown_until - time.time()))
            return

        action = self._trade_manager.evaluate(
            position           = pos_state,
            df_m15             = df_m15_struct,
            df_m5              = df_m5,
            df_m1              = df_m1,
            news_confirmed_dir = news_dir,
            portfolio_pnl_r    = portfolio_pnl_r,
        )

        # Dedup PARTIAL_CLOSE: only once per bar (counter signals don't reset intrabar)
        if action.action == ActionType.PARTIAL_CLOSE and self._last_bar == self._last_partial_bar:
            logger.debug("[%s] PARTIAL_CLOSE suppressed — already fired this bar", self.name)
            return

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
                    # Intrabar management: run on every poll so trail/TM can react
                    # within the bar, not only at bar close.
                    self._portfolio_daily_guard()
                    self._manage_open_position()
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

                # Gap 7 — USDJPY macro filter: set H4 bias on strategy before signal gen.
                # Only applies to DXY-inverse instruments (EUR/GBP/metals).
                # USDJPY bullish = USD strong = headwind for DXY-inverse longs.
                _DXY_INVERSE = {"EURUSD", "GBPUSD", "XAUUSD", "XAGUSD"}
                if self._symbol in _DXY_INVERSE and hasattr(self._strategy, "_uj_h4_bias"):
                    try:
                        _uj_rates = mt5.copy_rates_from_pos(
                            "USDJPY", TIMEFRAME_MAP.get("H1", mt5.TIMEFRAME_H1), 0, 200
                        )
                        if _uj_rates is not None and len(_uj_rates) >= 20:
                            _uj_df = pd.DataFrame(_uj_rates)
                            _uj_df["time"] = pd.to_datetime(_uj_df["time"], unit="s", utc=True)
                            from strategies.aiden_index import _resample_h4 as _uj_resample_h4
                            from strategies.aiden_index import _compute_h4_bias_ema as _uj_bias_fn
                            _uj_h4 = _uj_resample_h4(_uj_df)
                            if len(_uj_h4) >= 10:
                                _uj_bias, _ = _uj_bias_fn(_uj_h4, 10, 20)
                                self._strategy._uj_h4_bias = int(_uj_bias.iloc[-1]) if len(_uj_bias) > 0 else 0
                            else:
                                self._strategy._uj_h4_bias = 0
                        else:
                            self._strategy._uj_h4_bias = 0
                    except Exception:
                        self._strategy._uj_h4_bias = 0

                if hasattr(self._strategy, "_symbol"):
                    self._strategy._symbol = self._symbol
                _az      = getattr(self, "_analyzer", None)
                _v3_mode = getattr(self, "_v3_mode", False)

                if _v3_mode and _az is not None:
                    # Phase 4 live: AnalyzerEngine drives the signal
                    try:
                        _df_m5_az = self._fetch_ltf_bars("M5", count=40)
                        _az.update(df, _df_m5_az)
                        desired = _az.get_armed_direction()
                        # Use stack score as signal_score proxy for downstream sizing
                        _az_score = _az.get_armed_score()
                        # Inject into strategy's _scores so downstream sizing picks it up
                        if hasattr(self._strategy, "_scores") and self._strategy._scores is not None and len(self._strategy._scores) > 0:
                            self._strategy._scores.iloc[-1] = int(_az_score / 10)  # normalize 0-100 → 0-10 scale
                    except Exception as _v3e:
                        logger.warning("[%s] V3 mode error — falling back to v2 signal: %s", self.name, _v3e)
                        signals = self._strategy.generate_signals(df)
                        desired = int(signals.iloc[-1])
                else:
                    # Phase 0-3: old gate cascade generates signal
                    signals = self._strategy.generate_signals(df)
                    desired = int(signals.iloc[-1])
                    # Shadow AnalyzerEngine update (dry-run only)
                    if _az is not None:
                        try:
                            _df_m5_az = self._fetch_ltf_bars("M5", count=40)
                            _az.update(df, _df_m5_az)
                        except Exception:
                            pass

                current = trader.get_position_direction(self._symbol)

                # Update shared correlation matrix with this bar's closes
                if self._corr_matrix is not None and df is not None and len(df) >= 2:
                    import numpy as _np_corr
                    self._corr_matrix.update(self._symbol, _np_corr.array(df["close"].values))

                # ── White-blood-cell: reconcile internal state vs MT5 reality ──
                # If we think we're in a trade but MT5 shows flat, someone closed
                # it externally (manual close, broker action, SL/TP hit outside loop).
                # Trust MT5, close the journal entry, reset state, notify once.
                _internal_in_trade = self._open_entry_price is not None
                if _internal_in_trade and current == 0:
                    logger.warning(
                        "[%s] STATE MISMATCH: internal=open, MT5=flat — external close detected. "
                        "Reconciling.", self.name
                    )
                    try:
                        tick = mt5.symbol_info_tick(self._symbol)
                        _recon_price = (tick.bid + tick.ask) / 2.0 if tick else (self._open_entry_price or 0.0)
                        if self._journal is not None:
                            self._journal.close_trade(self._symbol, _recon_price, equity)
                        tg.notify_council_flag(
                            "SRE #07",
                            f"{self._symbol} closed externally (manual/broker). "
                            f"Bot reconciled at {_recon_price:.5g}. State reset."
                        )
                    except Exception:
                        logger.exception("[%s] Reconcile failed — forcing state reset", self.name)
                    self._reset_position_state()
                    self._block_entry = True   # don't re-enter on the same bar after external close
                    if self._journal is not None:
                        self._journal._save_open()

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
                        # Anti-tilt: track win/loss streak by comparing equity to entry equity
                        if self._entry_equity > 0:
                            _close_eq = account.get("equity", self._entry_equity) if "account" in dir() else self._entry_equity
                            if _close_eq < self._entry_equity:
                                self._consec_losses += 1
                                logger.info("[%s] Anti-tilt: loss #%d in streak (entry_eq=%.2f close_eq=%.2f)",
                                            self.name, self._consec_losses, self._entry_equity, _close_eq)
                            else:
                                if self._consec_losses > 0:
                                    logger.info("[%s] Anti-tilt: win resets streak (was %d losses)",
                                                self.name, self._consec_losses)
                                self._consec_losses = 0
                            self._entry_equity = 0.0
                        # Record the close NOW (deduped) so the opposite entry below
                        # sees the outcome in its risk gate — no async lag.
                        self._record_close_now()
                        self._reset_position_state()

                # Open new position — gate through soft halt, RiskAgent, FTMOTracker
                if desired != 0:
                    _profile = PROFILES.get(self._symbol)
                    _shad = StackInput(
                        symbol=self._symbol,
                        signal_dir=desired,
                        archetype=_profile.archetype if _profile else "liquidity",
                        archetype_threshold=_profile.entry_threshold if _profile else 65,
                        bar_time=bar_dt.isoformat() if bar_dt else None,
                    )
                    # Council Watch hard-halt file check (external daemon backup).
                    _council_halt = Path(__file__).parent.parent / "logs" / "council_halt.flag"
                    if _council_halt.exists():
                        try:
                            _ch = json.loads(_council_halt.read_text())
                            if _ch.get("day") == datetime.now(timezone.utc).strftime("%Y-%m-%d"):
                                logger.critical(
                                    "[%s] COUNCIL HALT active (%s) — blocking new entry",
                                    self.name, _ch.get("reason", "?"),
                                )
                                if self._soft_halt is not None:
                                    self._soft_halt.set()
                                _shad.gates["council_halt"] = {"blocked": True}; _shad.blocking_gate = "council_halt"; _shadow_logger.record(_shad)
                                continue
                        except Exception:
                            pass

                    # Hard halt check (4% daily or 7% cumulative DD).
                    # RiskGuard sets soft_halt at those limits; engines also check inline
                    # since the background thread ticks every 60s.
                    if self._soft_halt is not None and self._soft_halt.is_set():
                        logger.warning("[%s] SOFT HALT active — blocking new entry", self.name)
                        _shad.gates["soft_halt"] = {"blocked": True}; _shad.blocking_gate = "soft_halt"; _shadow_logger.record(_shad)
                        continue
                    _daily_boost = 0
                    if self._risk_guard is not None:
                        _rg_day_eq  = getattr(self._risk_guard, "_day_start_equity", None)
                        _rg_init_eq = getattr(self._risk_guard, "_initial_equity", None)
                        _rg_daily_limit = getattr(self._risk_guard, "_daily_halt_pct", MAX_DAILY_LOSS_PCT)
                        _rg_soft_limit  = getattr(self._risk_guard, "_soft_dd_pct", SOFT_DD_HALT_PCT)
                        if _rg_day_eq and _rg_day_eq > 0:
                            _inline_daily_dd = (_rg_day_eq - equity) / _rg_day_eq * 100
                            if _inline_daily_dd >= _rg_daily_limit:
                                logger.critical(
                                    "[%s] INLINE DAILY DD GATE: %.2f%% >= %.2f%% hard limit — blocking entry, "
                                    "setting soft halt", self.name, _inline_daily_dd, _rg_daily_limit
                                )
                                if self._soft_halt is not None:
                                    self._soft_halt.set()
                                _shad.gates["daily_dd"] = {"blocked": True, "value": round(_inline_daily_dd, 3)}; _shad.blocking_gate = "daily_dd"; _shadow_logger.record(_shad)
                                continue
                            # Compute tiered daily DD boost — applied to score floor after needed is set
                            if _inline_daily_dd >= DAILY_DD_TIER3_PCT:
                                _daily_boost = 3
                            elif _inline_daily_dd >= DAILY_DD_TIER2_PCT:
                                _daily_boost = 2
                            elif _inline_daily_dd >= DAILY_DD_TIER1_PCT:
                                _daily_boost = 1
                        if _rg_init_eq and _rg_init_eq > 0:
                            _inline_total_dd = (_rg_init_eq - equity) / _rg_init_eq * 100
                            if _inline_total_dd >= _rg_soft_limit:
                                logger.critical(
                                    "[%s] INLINE TOTAL DD GATE: %.2f%% >= %.2f%% soft limit — blocking entry",
                                    self.name, _inline_total_dd, _rg_soft_limit
                                )
                                if self._soft_halt is not None:
                                    self._soft_halt.set()
                                _shad.gates["total_dd"] = {"blocked": True, "value": round(_inline_total_dd, 3)}; _shad.blocking_gate = "total_dd"; _shadow_logger.record(_shad)
                                continue

                    # ── M5 confluence (not a gate — confluence only) ──────────
                    _extra_reasons: list[str] = []
                    df_m5_entry = self._fetch_ltf_bars("M5", count=40)
                    atr_entry   = 0.0
                    if df_m5_entry is not None and len(df_m5_entry) > 14:
                        atr_entry = float(
                            (df_m5_entry["high"] - df_m5_entry["low"])
                            .rolling(14).mean().iloc[-1]
                        )
                    m5_confirmed = detect_m5_entry_trigger(df_m5_entry, desired, atr=atr_entry)
                    _shad.m5_confirmed = m5_confirmed
                    if not m5_confirmed:
                        # Route to pending M5 trigger — never enter without LTF confirmation.
                        # "entering anyway" produced 0 wins across live testing.
                        logger.info("[%s] M5 not confirmed — queuing pending trigger (not entering blind)", self.name)
                        if self._pending_signal == 0:
                            self._pending_signal = desired
                            self._pending_bars   = 0
                        _shad.gates["m5_pending"] = {"blocked": True}; _shad.blocking_gate = "m5_pending"; _shadow_logger.record(_shad)
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
                        _shad.gates["accumulation"] = {"blocked": True, "value": round(accum_risk, 2)}; _shad.blocking_gate = "accumulation"; _shadow_logger.record(_shad)
                        continue
                    _shad.gates["accumulation"] = {"blocked": False, "value": round(accum_risk, 2)}

                    # ── Liquidity draw gate ───────────────────────────────────
                    liq = detect_liquidity_draw(df_m5_entry, desired, atr=atr_entry)
                    if liq["block_entry"]:
                        logger.warning(
                            "[%s] LIQUIDITY GATE: opposing pool %.1f ATR away vs aligned %.1f ATR — blocked",
                            self.name, liq["opposing_pool"], liq["aligned_pool"],
                        )
                        _shad.gates["liquidity_draw"] = {"blocked": True, "opposing": liq["opposing_pool"], "aligned": liq["aligned_pool"]}; _shad.blocking_gate = "liquidity_draw"; _shadow_logger.record(_shad)
                        continue
                    _shad.gates["liquidity_draw"] = {"blocked": False}

                    # ── Correlation divergence gate ───────────────────────────
                    if self._correlation_divergence(desired):
                        _shad.gates["corr_divergence"] = {"blocked": True}; _shad.blocking_gate = "corr_divergence"; _shadow_logger.record(_shad)
                        continue

                    # ── Correlation cluster cap (same-direction) ──────────────
                    if self._correlation_cluster_cap(desired):
                        _shad.gates["corr_cluster_cap"] = {"blocked": True}; _shad.blocking_gate = "corr_cluster_cap"; _shadow_logger.record(_shad)
                        continue

                    # ── Weekend new-entry guard (FTMO compliance) ─────────────
                    # Standard (non-swing) accounts must not open trades that
                    # carry over the weekend gap. Disabled by default; enable
                    # when account type is confirmed as standard.
                    if self._trade_cfg.get("prevent_weekend_entries", False):
                        _now_utc = datetime.utcnow()
                        _wd = _now_utc.weekday()  # 0=Mon … 4=Fri, 5=Sat, 6=Sun
                        _weekend_blocked = (
                            (_wd == 4 and _now_utc.hour >= 20)  # Fri after 20:00 UTC
                            or _wd >= 5                          # Sat or Sun
                        )
                        if _weekend_blocked:
                            logger.info(
                                "[%s] Weekend gate: no new entries Fri≥20:00 UTC or Sat/Sun",
                                self.name,
                            )
                            _shad.gates["weekend"] = {"blocked": True}; _shad.blocking_gate = "weekend"; _shadow_logger.record(_shad)
                            continue

                    open_count = len(trader.get_all_positions())   # magic-filtered (H3)
                    can_trade, size_mult, reason = self._risk_agent.pre_trade_check(
                        equity, open_count
                    )
                    if not can_trade:
                        logger.warning("[%s] RiskAgent blocked: %s", self.name, reason)
                        _shad.gates["risk_agent"] = {"blocked": True, "reason": str(reason)}; _shad.blocking_gate = "risk_agent"; _shadow_logger.record(_shad)
                        continue

                    # FTMO compliance pre-trade gate (#05) — both total AND daily DD
                    if self._ftmo_tracker is not None:
                        ftmo_status = self._ftmo_tracker.check(equity)
                        if ftmo_status["total_dd_pct"] >= ftmo_status["total_dd_limit"]:
                            logger.critical("[%s] FTMO total DD limit — blocking entry", self.name)
                            _shad.gates["ftmo_total_dd"] = {"blocked": True, "value": round(ftmo_status["total_dd_pct"], 3)}; _shad.blocking_gate = "ftmo_total_dd"; _shadow_logger.record(_shad)
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
                                _shad.gates["ftmo_daily_dd"] = {"blocked": True, "value": round(daily_dd, 3)}; _shad.blocking_gate = "ftmo_daily_dd"; _shadow_logger.record(_shad)
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

                    # AMD sweep detection — runs before score, blocks first.
                    # If price swept a key level and closed back past it, the expected
                    # displacement is in the sweep direction. Trading AGAINST that is
                    # the wrong side of the manipulation. Strong sweeps hard-block;
                    # weak sweeps require floor+2 extra if counter-direction.
                    _amd_sweep: Optional[SweepEvent] = None
                    if self._amd_detector is not None and self._level_monitor is not None:
                        _ls_amd = self._level_monitor._level_sets.get(self._symbol)
                        if _ls_amd is not None:
                            _atr_amd = float(getattr(self._strategy, "_atr_cache", pd.Series()).iloc[-1]) \
                                       if hasattr(self._strategy, "_atr_cache") else 0.0
                            _all_levels_amd = list(_ls_amd.levels) + list(_ls_amd.zones)
                            try:
                                _amd_phase = self._amd_detector.assess_phase(
                                    self._symbol, df, _all_levels_amd,
                                    _atr_amd or (equity * 0.002),
                                )
                                _amd_sweep = _amd_phase.sweep if _amd_phase.phase.value in (
                                    "manipulation", "displacement"
                                ) else None
                                if _amd_phase.phase.value != "unknown":
                                    logger.info("[%s] AMD phase: %s", self.name, _amd_phase)
                                    _shad.regime = _amd_phase.phase.value
                            except Exception:
                                _amd_sweep = None
                            if _amd_sweep is not None:
                                logger.info(
                                    "[%s] AMD sweep: %s", self.name, _amd_sweep
                                )
                                _shad.sweep_present  = (_amd_sweep.direction == desired)
                                _shad.eq_liq_cluster = _amd_sweep.eq_liq
                                # Block if signal goes AGAINST a confirmed sweep
                                if _amd_sweep.direction != desired:
                                    if _amd_sweep.strong:
                                        logger.info(
                                            "[%s] AMD SWEEP HARD BLOCK: STRONG %s sweep @ %s but signal=%+d",
                                            self.name,
                                            "BULL" if _amd_sweep.direction == 1 else "BEAR",
                                            _amd_sweep.level_label, desired,
                                        )
                                        _shad.gates["amd_sweep"] = {
                                            "blocked": True, "sweep_dir": _amd_sweep.direction,
                                            "strong": True, "level": _amd_sweep.level_label,
                                        }
                                        _shad.blocking_gate = "amd_sweep"
                                        _shadow_logger.record(_shad)
                                        continue
                                    # Weak counter-sweep: deferred to score gate (needs floor+2)
                                    logger.info(
                                        "[%s] AMD SWEEP WEAK counter-signal: %s sweep @ %s "
                                        "vs signal=%+d — will need floor+2",
                                        self.name,
                                        "BULL" if _amd_sweep.direction == 1 else "BEAR",
                                        _amd_sweep.level_label, desired,
                                    )

                    # Score-based sizing: psychology_mult * score_mult * concentration_mult
                    _sc          = getattr(self._strategy, "_scores", None)
                    signal_score = int(_sc.iloc[-1]) if _sc is not None else 0

                    # AMD aligned sweep boost: displacement entry = strongest confluence.
                    # Strong sweep aligned with signal = +2 (equals HTF zone boost).
                    # Weak sweep aligned = +1.
                    if _amd_sweep is not None and _amd_sweep.direction == desired:
                        _amd_boost = 2 if _amd_sweep.strong else 1
                        signal_score += _amd_boost
                        logger.info(
                            "[%s] AMD aligned %s sweep @ %s → +%d score",
                            self.name,
                            "BULL" if _amd_sweep.direction == 1 else "BEAR",
                            _amd_sweep.level_label, _amd_boost,
                        )
                        _extra_reasons.append(f"AMD sweep +{_amd_boost}")

                    # M5 confluence bonus applied here after base score is loaded
                    if m5_confirmed:
                        signal_score += 1
                        _extra_reasons.append("M5 structure confirmed")
                        logger.info("[%s] M5 confirmed — +1 score → %d", self.name, signal_score)

                    # Order flow confluence: checks delta + imbalance + DOM alignment
                    try:
                        _df_m5 = self._strategy._m5_df if hasattr(self._strategy, "_m5_df") else None
                        _of_snap = analyse_order_flow(self._symbol, _df_m5, mt5=mt5)
                        _of_mod  = order_flow_score_modifier(_of_snap, desired)
                        if _of_mod != 0 and _of_snap is not None:
                            signal_score += _of_mod
                            _extra_reasons.append(f"OrderFlow {'+' if _of_mod>0 else ''}{_of_mod}: {_of_snap.summary}")
                            logger.info("[%s] OrderFlow modifier %+d | %s", self.name, _of_mod, _of_snap.summary)
                    except Exception:
                        pass

                    # DOM key levels — bookmap equivalent: where large orders cluster
                    _dom_supporting = False   # wall stacked on OUR side of the trade
                    try:
                        _dom_levels = get_dom_key_levels(self._symbol, mt5)
                        if _dom_levels:
                            _tick = mt5.symbol_info_tick(self._symbol)
                            _mid  = (_tick.bid + _tick.ask) / 2 if _tick else 0.0
                            for _dl in _dom_levels:
                                _dist_pts = abs(_dl.price - _mid)
                                logger.info(
                                    "[%s] DOM wall: %s @ %.5f | vol=%.0f (%.1fx avg) | dist=%.1f pts",
                                    self.name, _dl.side.upper(), _dl.price, _dl.volume, _dl.strength, _dist_pts,
                                )
                            # Closest DOM wall opposing the trade direction acts as a filter
                            _opposing_walls = [
                                d for d in _dom_levels
                                if (desired == 1 and d.side == "ask") or (desired == -1 and d.side == "bid")
                            ]
                            if _opposing_walls:
                                closest = min(_opposing_walls, key=lambda x: abs(x.price - _mid))
                                _extra_reasons.append(f"DOM wall {closest.side} @ {closest.price:.1f} ({closest.strength:.1f}x)")
                            # Supporting wall: bid stack under a long / ask stack over
                            # a short — institutions parked with us. Feeds dom_aligned.
                            _dom_supporting = any(
                                (desired == 1 and d.side == "bid") or (desired == -1 and d.side == "ask")
                                for d in _dom_levels
                            )
                    except Exception:
                        pass

                    # News gate: price-confirmed direction preferred; NewsIntelligence (decayed) as fallback
                    # Amplifier only — never penalises
                    _news_sig: Optional[NewsSignal] = None
                    if self._news_gate is not None:
                        news_ctx      = self._news_gate.get_context()
                        confirmed_dir = self._price_confirmed_event_direction(news_ctx)
                        if confirmed_dir != 0:
                            if confirmed_dir == desired:
                                logger.info("[%s] Post-event price confirms signal: score +1 | events=%s",
                                            self.name, [e.name for e in news_ctx.fired_high])
                                signal_score += 1
                                _extra_reasons.append("News confirm +1")
                                _shad.news_mod = 1
                        else:
                            # Use NewsIntelligence (decayed+tier-aware) when available; fall back to flat +1
                            if self._news_intel is not None:
                                try:
                                    _news_sig = self._news_intel.get_signal(self._symbol, desired)
                                    if _news_sig.score_mod > 0:
                                        signal_score += _news_sig.score_mod
                                        _extra_reasons.append(f"NewsIntel +{_news_sig.score_mod}: {_news_sig.note}")
                                        logger.info("[%s] NewsIntel: +%d | %s", self.name, _news_sig.score_mod, _news_sig.note)
                                        _shad.news_mod = _news_sig.score_mod
                                    if _news_sig.upcoming_caution:
                                        logger.info("[%s] NewsIntel caution: %s in %.0fmin",
                                                    self.name, _news_sig.upcoming_name, _news_sig.upcoming_min)
                                except Exception as _ne:
                                    logger.debug("[%s] NewsIntel error: %s", self.name, _ne)
                            else:
                                news_mod = news_ctx.score_modifier(self._symbol, desired)
                                if news_mod > 0:
                                    logger.info("[%s] Macro consensus amplifies signal: +%d | %s",
                                                self.name, news_mod,
                                                news_ctx.fired_summary(self._symbol))
                                    signal_score += news_mod
                                    _extra_reasons.append(f"Macro consensus +{news_mod}")
                                    _shad.news_mod = news_mod

                    # Level confluence: HTF zones carry explicit directional bias.
                    # Selling into a bullish zone (EQL sweep, bullish OB/FVG) = wrong location.
                    # Block those trades; only boost score when zone aligns with signal direction.
                    if _approaching_levels:
                        _htf_zones  = [l for l in _approaching_levels if hasattr(l, "zone_type")]
                        _key_levels = [l for l in _approaching_levels if not hasattr(l, "zone_type")]

                        # Hard block: zone direction conflicts with trade direction
                        _counter_zones = [z for z in _htf_zones if z.direction != 0 and z.direction != desired]
                        if _counter_zones:
                            _cz_labels = ", ".join(z.label for z in _counter_zones)
                            logger.info(
                                "[%s] HTF ZONE DIRECTION BLOCK: signal=%+d conflicts with %s",
                                self.name, desired, _cz_labels,
                            )
                            _shad.gates["htf_zone_direction"] = {"blocked": True, "zones": _cz_labels}
                            _shad.blocking_gate = "htf_zone_direction"
                            _shadow_logger.record(_shad)
                            continue

                        # Boost only for direction-aligned zones
                        _aligned = [z for z in _htf_zones if z.direction == 0 or z.direction == desired]
                        _all_aligned = _key_levels + _aligned
                        if _all_aligned:
                            _lvl_boost = max(getattr(l, "score_boost", 1) for l in _all_aligned)
                            signal_score += _lvl_boost
                            lvl_names = ", ".join(l.label for l in _all_aligned)
                            logger.info("[%s] Level confluence +%d: %s", self.name, _lvl_boost, lvl_names)
                            _extra_reasons.append(f"Level +{_lvl_boost} ({lvl_names})")

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
                                _extra_reasons.append(f"-DXY opposing (bias={_dxy_bias:+d})")
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
                    if _daily_boost:
                        old_needed = needed
                        needed += _daily_boost
                        logger.info(
                            "[%s] Daily DD tier (%.2f%%) — score floor %d→%d (+%d)",
                            self.name, _inline_daily_dd, old_needed, needed, _daily_boost,
                        )

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

                    _shad.base_score = signal_score; _shad.final_score = signal_score; _shad.score_floor = needed; _shad.floor_reason = "counter_trend" if against_trend else "with_trend"; _shad.h4_aligned = (getattr(self._strategy, "_last_h4_bias", 0) == desired)
                    if signal_score < needed:
                        logger.info("[%s] LOW CONVICTION skip: %s score=%d < %d (%s)",
                                    self.name, "BUY" if desired == 1 else "SELL",
                                    signal_score, needed,
                                    "counter-trend" if against_trend else "with-trend")
                        _shad.gates["score_floor"] = {"blocked": True, "score": signal_score, "needed": needed}; _shad.blocking_gate = "score_floor"; _shadow_logger.record(_shad)
                        continue
                    _shad.gates["score_floor"] = {"blocked": False, "score": signal_score, "needed": needed}

                    # ── Weak AMD counter-sweep gate (deferred to here for floor+2 check) ──
                    if (_amd_sweep is not None
                            and _amd_sweep.direction != desired
                            and not _amd_sweep.strong):
                        _sweep_sof = signal_score - needed
                        if _sweep_sof < 2:
                            logger.info(
                                "[%s] AMD SWEEP WEAK BLOCK: counter-%s sweep @ %s, "
                                "need floor+2 (have +%d)",
                                self.name,
                                "BULL" if _amd_sweep.direction == 1 else "BEAR",
                                _amd_sweep.level_label, _sweep_sof,
                            )
                            _shad.gates["amd_sweep"] = {
                                "blocked": True, "sweep_dir": _amd_sweep.direction,
                                "strong": False, "over_floor": _sweep_sof,
                            }
                            _shad.blocking_gate = "amd_sweep"
                            _shadow_logger.record(_shad)
                            continue
                        logger.info(
                            "[%s] AMD SWEEP WEAK: counter-sweep but floor+%d >= 2 — allowed",
                            self.name, signal_score - needed,
                        )

                    # ── HTF premium/discount range gate ────────────────────
                    # If price is in the upper 25% of H4 range (premium), only take
                    # sells; if in lower 25% (discount), only take buys. Counter-range
                    # entries need floor+2 extra conviction — these are low-probability
                    # trades. Mid-range (50%) = no constraint.
                    if self._level_monitor is not None:
                        _ls_rb = self._level_monitor._level_sets.get(self._symbol)
                        if _ls_rb is not None:
                            _cur_px   = float(df["close"].iloc[-1])
                            _rng_bias = _ls_rb.range_bias(_cur_px)
                            if _rng_bias != 0 and _rng_bias != desired:
                                _range_label = "premium" if _rng_bias == -1 else "discount"
                                _score_over_floor_rb = signal_score - needed
                                if _score_over_floor_rb < 2:
                                    logger.info(
                                        "[%s] RANGE BIAS GATE: signal=%+d but price in H4 %s — need floor+2 "
                                        "(have +%d)",
                                        self.name, desired, _range_label, _score_over_floor_rb,
                                    )
                                    _shad.gates["htf_range_bias"] = {
                                        "blocked": True, "bias": _range_label, "over_floor": _score_over_floor_rb,
                                    }
                                    _shad.blocking_gate = "htf_range_bias"
                                    _shadow_logger.record(_shad)
                                    continue
                                logger.info(
                                    "[%s] RANGE BIAS: counter-range %s in %s — allowed (floor+%d >= 2)",
                                    self.name, "BUY" if desired == 1 else "SELL",
                                    _range_label, _score_over_floor_rb,
                                )

                    # ── M1 structural confirmation ─────────────────────────────
                    # Require the last completed M1 candle to break above/below the
                    # prior candle's high/low in the trade direction. Filters entries
                    # where price hasn't shown any intent — the "never worked" class.
                    _df_m1c = self._fetch_ltf_bars("M1", count=10)
                    if _df_m1c is not None and len(_df_m1c) >= 3:
                        _m1_close = float(_df_m1c["close"].iloc[-2])   # last completed bar
                        _m1_open  = float(_df_m1c["open"].iloc[-2])
                        _m1_phigh = float(_df_m1c["high"].iloc[-3])    # prior bar
                        _m1_plow  = float(_df_m1c["low"].iloc[-3])
                        _m1_bull  = _m1_close > _m1_open
                        _m1_bear  = _m1_close < _m1_open
                        _m1_ok    = _m1_bull if desired == 1 else _m1_bear
                        if not _m1_ok:
                            logger.info(
                                "[%s] M1 CONFIRM GATE: no %s structural break — skip "
                                "(close=%.5f phigh=%.5f plow=%.5f)",
                                self.name, "bull" if desired == 1 else "bear",
                                _m1_close, _m1_phigh, _m1_plow,
                            )
                            _shad.gates["m1_confirm"] = {"blocked": True}; _shad.blocking_gate = "m1_confirm"; _shadow_logger.record(_shad)
                            continue

                    # ── Z-score regime gate (ruflo/neural-trader classifier) ──
                    # Classifies the symbol's statistical volatility regime from
                    # M15 closes. Flatline = no edge (skip). Oscillation +
                    # continuation = ranging context mismatch (skip). Spike = half
                    # size (extreme event, unpredictable direction).
                    _plan_type_pre = self._last_plan.trade_type if self._last_plan else "breakout"
                    _regime_result = None
                    if df is not None and len(df) >= 20:
                        import numpy as _np
                        _regime_result = classify_regime(_np.array(df["close"].values, dtype=float))
                        _rg_allow, _rg_reason = regime_entry_gate(_regime_result, desired, _plan_type_pre)
                        if not _rg_allow:
                            logger.info("[%s] REGIME GATE: %s", self.name, _rg_reason)
                            _shad.regime = _regime_result.regime if _regime_result else "unknown"; _shad.gates["regime"] = {"blocked": True, "regime": _shad.regime, "reason": str(_rg_reason)}; _shad.blocking_gate = "regime"; _shadow_logger.record(_shad)
                            continue
                        if _regime_result:
                            _shad.regime = _regime_result.regime
                            logger.info(
                                "[%s] Regime: %s (maxZ=%.2f lastZ=%.2f highPct=%.0f%%)",
                                self.name, _regime_result.regime, _regime_result.max_z,
                                _regime_result.last_z, _regime_result.high_count_pct * 100,
                            )

                    # ── MarketContextAgent — structural environment check ──────
                    # Consult the active level-intelligence agent before sizing.
                    # It tells us: are we AT a level? Fighting one? What direction?
                    _mc_ctx: Optional[MarketContext] = None
                    if self._mc_agent is not None and df is not None:
                        _mc_ctx = self._mc_agent.assess(self._symbol, df, atr_entry or 1.0, desired)
                        if _mc_ctx.council_notes:
                            for _cn in _mc_ctx.council_notes:
                                logger.info("[%s] %s", self.name, _cn)
                        if _mc_ctx.entry_block:
                            logger.info("[%s] MCAgent BLOCK: %s", self.name, _mc_ctx.narrative)
                            _shad.gates["mc_agent"] = {"blocked": True, "narrative": str(_mc_ctx.narrative)}; _shad.blocking_gate = "mc_agent"; _shadow_logger.record(_shad)
                            continue

                    # Continuation trades: require floor+2 minimum.
                    _plan_type = self._last_plan.trade_type if self._last_plan else "breakout"
                    _score_over_floor = signal_score - needed
                    if _plan_type == "continuation" and _score_over_floor < 2:
                        logger.info("[%s] CONTINUATION GATE: score only %d over floor — skipping (need +2)",
                                    self.name, _score_over_floor)
                        _shad.gates["continuation_score"] = {"blocked": True, "score_over_floor": _score_over_floor}; _shad.blocking_gate = "continuation_score"; _shadow_logger.record(_shad)
                        continue
                    # Off-level continuation: require floor+3 (one more point than at-level).
                    # Softened from hard ban — score=floor+3 is high enough conviction.
                    if _plan_type == "continuation" and not (_mc_ctx and _mc_ctx.at_level):
                        if _score_over_floor < 3:
                            logger.info("[%s] CONTINUATION GATE: off-level score %d over floor — need +3 off-level",
                                        self.name, _score_over_floor)
                            _shad.gates["continuation_level"] = {"blocked": True, "score_over_floor": _score_over_floor}; _shad.blocking_gate = "continuation_level"; _shadow_logger.record(_shad)
                            continue
                        logger.info("[%s] CONTINUATION GATE: off-level override — score %d >= floor+3",
                                    self.name, _score_over_floor)

                    # ── Bayesian probability model — replaces flat score_mult ──
                    # Build confluence inputs from what's confirmed above.
                    _plan_rr = self._last_plan.rr if self._last_plan else 2.5

                    # Scout regime: load every entry (cached file, negligible I/O)
                    self._scout_regime = self._read_scout_regime()
                    _scout_aligned = self._scout_regime_aligned(self._symbol, desired)
                    if self._scout_regime:
                        logger.info(
                            "[%s] Scout regime: %s | dxy_bias=%s | scout_aligned=%s",
                            self.name,
                            self._scout_regime.get("narrative", "?"),
                            self._scout_regime.get("dxy_bias", "?"),
                            _scout_aligned,
                        )

                    _of_aligned = bool(_of_mod > 0) if "_of_mod" in dir() else False
                    _bar_time   = pd.to_datetime(df["time"].iloc[-1]) if df is not None and len(df) > 0 else None
                    _bar_price  = float(df["close"].iloc[-1]) if df is not None and len(df) > 0 else 0.0
                    if df is not None:
                        _update_smt_cache(self._symbol, df)
                    _smt_div    = _smt_divergence_present(self._symbol, df, desired) if df is not None else False
                    _sr_now     = getattr(self._strategy, "_score_reasons", None)
                    _cur_rsns   = list(_sr_now.iloc[-1]) if _sr_now is not None and len(_sr_now) > 0 else []
                    _eq_liq     = any("liq cluster" in r for r in _cur_rsns)
                    # Early leakage: Asian session boundary broken before London open
                    # JP: "When I see price leak out of Asia early, London violates it harder"
                    _early_leak = False
                    if _bar_time is not None and pd.to_datetime(_bar_time, utc=True).hour < 9:
                        try:
                            _asian_lvls = self._mc_agent._lm.get_levels(self._symbol)
                            for _lv in _asian_lvls:
                                if _lv.source_tf != "Asian":
                                    continue
                                if desired == 1 and _lv.direction == 1 and _bar_price > _lv.price:
                                    _early_leak = True; break
                                if desired == -1 and _lv.direction == -1 and _bar_price < _lv.price:
                                    _early_leak = True; break
                        except Exception:
                            pass
                    _in_day = _inside_day(df) if df is not None else False
                    news_dir = 0
                    if self._news_gate is not None:
                        try:
                            news_dir = self._price_confirmed_event_direction(
                                self._news_gate.get_context()
                            )
                        except Exception:
                            pass
                    _confl = TradeConfluences(
                        fvg_present        = True,        # strategy fires on FVG detection
                        ob_present         = _score_over_floor >= 1,
                        m5_confirmed       = m5_confirmed,
                        h4_aligned         = getattr(self._strategy, "_last_h4_bias", 0) == desired,
                        at_htf_level       = bool(_mc_ctx and _mc_ctx.at_level),
                        level_strength     = _mc_ctx.level_strength if _mc_ctx else 0.0,
                        order_flow_aligned = _of_aligned or _scout_aligned,
                        dom_aligned        = _dom_supporting,
                        news_aligned       = (
                            (_news_sig.score_mod > 0) if _news_sig is not None
                            else (news_dir == desired if news_dir != 0 else False)
                        ),
                        trade_type         = _plan_type,
                        rr                 = _plan_rr,
                        in_ict_macro       = _in_ict_macro(_bar_time) if _bar_time is not None else False,
                        ipda_aligned       = _ipda_aligned(df, desired, _bar_price) if df is not None else False,
                        smt_divergence     = _smt_div,
                        eq_liq_cluster     = _eq_liq,
                        early_leakage      = _early_leak,
                        inside_day         = _in_day,
                    )
                    _prob = self._prob_model.estimate(_confl)

                    # Populate all confluence booleans on shadow capture now that _confl is built
                    _shad.at_htf_level       = _confl.at_htf_level
                    _shad.level_strength     = _confl.level_strength
                    _shad.order_flow_aligned = _confl.order_flow_aligned
                    _shad.dom_aligned        = _confl.dom_aligned
                    _shad.news_aligned       = _confl.news_aligned
                    _shad.in_ict_macro       = _confl.in_ict_macro
                    _shad.ipda_aligned       = _confl.ipda_aligned
                    _shad.smt_divergence     = _confl.smt_divergence
                    _shad.eq_liq_cluster     = _confl.eq_liq_cluster
                    _shad.early_leakage      = _confl.early_leakage
                    _shad.ob_present         = _confl.ob_present
                    _shad.prob_model_prob    = getattr(_prob, "p_win", 0.0)
                    _shad.sweep_present      = any("sweep" in str(r).lower() for r in _cur_rsns)
                    _shad.scout_aligned      = _scout_aligned

                    if not _prob.take_trade:
                        logger.info("[%s] PROB MODEL skip: %s", self.name, _prob.note)
                        _shad.gates["prob_model"] = {"blocked": True, "note": str(_prob.note)}; _shad.blocking_gate = "prob_model"; _shadow_logger.record(_shad)
                        continue
                    _shad.gates["prob_model"] = {"blocked": False, "p_win": getattr(_prob, "p_win", 0.0)}

                    # score_mult driven by probability model output
                    score_mult = _prob.size_mult

                    # Apply MarketContextAgent probability lift on top
                    if _mc_ctx is not None:
                        score_mult *= _mc_ctx.probability_lift

                    # Concentration mult: fewer concurrent positions = more size per trade
                    # 0-1 open → 2x  |  2-3 open → 1.5x  |  4+ open → 1x
                    n_open = len(trader.get_all_positions())   # magic-filtered (H3)
                    concentration_mult = 2.0 if n_open <= 1 else (1.5 if n_open <= 3 else 1.0)
                    combined_mult = size_mult * score_mult * concentration_mult

                    # ── Anti-tilt: geometric risk decay after consecutive losses ──
                    # Each unbroken loss shrinks the next trade's size by decay_factor.
                    # A win resets the counter. Config: anti_tilt_decay (default 0.85).
                    _tilt_decay = float(self._trade_cfg.get("anti_tilt_decay", 0.85))
                    _tilt_max   = int(self._trade_cfg.get("anti_tilt_max_losses", 4))
                    if self._consec_losses > 0:
                        _tilt_n    = min(self._consec_losses, _tilt_max)
                        _tilt_mult = _tilt_decay ** _tilt_n
                        combined_mult *= _tilt_mult
                        logger.info("[%s] Anti-tilt: %d consecutive losses → x%.3f size",
                                    self.name, self._consec_losses, _tilt_mult)

                    # ── ATR-percentile: scale size inversely with volatility regime ──
                    # High-vol (ATR in top quartile over 100 bars): size 0.75x — outsized moves,
                    # unpredictable slippage. Low-vol / compression (bottom quartile): size 1.15x —
                    # breakout expected, higher conviction per unit of stop distance.
                    if df is not None and len(df) >= 20:
                        try:
                            _atr_series = (df["high"] - df["low"]).rolling(14).mean()
                            _lookback   = min(100, len(_atr_series.dropna()))
                            _atr_now    = float(_atr_series.iloc[-1])
                            _atr_hist   = _atr_series.dropna().iloc[-_lookback:]
                            _atr_pct    = float((_atr_hist < _atr_now).mean())  # percentile rank 0-1
                            if _atr_pct >= 0.75:
                                combined_mult *= 0.75
                                logger.info("[%s] ATR-pct %.0f%% → high vol, size 0.75x", self.name, _atr_pct * 100)
                            elif _atr_pct <= 0.25:
                                combined_mult *= 1.15
                                logger.info("[%s] ATR-pct %.0f%% → low vol, size 1.15x", self.name, _atr_pct * 100)
                        except Exception:
                            pass

                    # Monday size reduction — JP: "Markets play catchup to weekend events,
                    # fundamentals can throw technical analysis out the window."
                    if _bar_time is not None and pd.to_datetime(_bar_time, utc=True).weekday() == 0:
                        combined_mult *= 0.75
                        logger.info("[%s] MONDAY 0.75x size — catchup volatility risk", self.name)

                    # Spike regime: halve size — extreme Z-score event, direction uncertain
                    _rg_mult = regime_size_mult(_regime_result)
                    if _rg_mult < 1.0:
                        combined_mult *= _rg_mult
                        logger.info("[%s] REGIME spike — size halved (maxZ=%.2f)",
                                    self.name, _regime_result.max_z if _regime_result else 0)

                    # Correlation-aware sizing (ruflo RiskDecision portfolioCorrelation)
                    # Reduce size when adding correlated exposure to open positions.
                    if self._corr_matrix is not None:
                        _open_pos = trader.get_all_positions()
                        _open_syms = [(p.symbol, (1 if p.type == 0 else -1))
                                      for p in _open_pos if p.symbol != self._symbol]
                        _corr_mult, _corr_reason = self._corr_matrix.correlated_size_mult(
                            self._symbol, _open_syms
                        )
                        if _corr_mult == 0.0:
                            logger.info("[%s] CORRELATION BLOCK: %s", self.name, _corr_reason)
                            continue
                        if _corr_mult < 1.0:
                            combined_mult *= _corr_mult
                            logger.info("[%s] CORRELATION size adj %.2fx: %s",
                                        self.name, _corr_mult, _corr_reason)

                    # VaR/CVaR logging — risk metrics for this entry
                    _p_win   = getattr(_prob, "p_win", 0.4) if "_prob" in dir() else 0.4
                    _risk_pc = float(self._trade_cfg.get("risk_pct", 0.75)) / 100
                    _vm      = var_cvar(_p_win, _risk_pc * combined_mult)
                    logger.info(
                        "[%s] VaR(95)=%.3f%% CVaR(95)=%.3f%% EV=%.4f (p_win=%.0f%%)",
                        self.name, _vm.var_95 * 100, _vm.cvar_95 * 100,
                        _vm.ev, _p_win * 100,
                    )

                    # NY-open whipsaw guard: 14:00 UTC hour is 0/5 live (-0.40R avg).
                    # Half size until a 10+ trade sample says otherwise — not a
                    # blacklist (C12: sample too small to ban an hour outright).
                    if datetime.now(timezone.utc).hour == 14:
                        combined_mult *= 0.5
                        logger.info("[%s] NY-OPEN GUARD: 14:00 UTC hour — size halved", self.name)

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
                            _shad.gates["allocator"] = {"blocked": True, "reason": str(alloc.reason)}; _shad.blocking_gate = "allocator"; _shadow_logger.record(_shad)
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
                            _shad.gates["portfolio_cap"] = {"blocked": True, "open_risk": round(open_risk, 2)}; _shad.blocking_gate = "portfolio_cap"; _shadow_logger.record(_shad)
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

                    # ── Burned-target guard (audit A3) — revenge-loop killer ──
                    # If this direction+target combo traded in the last 90 min,
                    # the idea already had its shot. New idea = new target.
                    _now_ts = time.time()
                    self._burned_targets = {k: v for k, v in self._burned_targets.items() if v > _now_ts}
                    if tp and (desired, round(tp, 5)) in self._burned_targets:
                        logger.warning(
                            "[%s] BURNED TARGET: dir=%+d tp=%.5f already traded within 90min — skip",
                            self.name, desired, tp,
                        )
                        continue

                    # ── Duplicate guard (audit A2) — never stack a second
                    # position on a symbol from the entry path.
                    if not self._dry_run and trader.get_positions(self._symbol):
                        logger.warning(
                            "[%s] DUPLICATE GUARD: position already open on %s — skip entry",
                            self.name, self._symbol,
                        )
                        continue

                    # ── Liquidity thesis gate: no clean draw = min size only, not skip ──
                    # Grade A/B (real draw): full plan size. Grade C (no draw): allow at
                    # plan's 0.25x min size if score >= floor+2. Below floor+2 = skip.
                    direction_str = "BUY" if desired == 1 else "SELL"
                    if self._last_plan is not None and not self._last_plan.tradeable:
                        if _score_over_floor < 2:
                            logger.info("[%s] NO-DRAW skip: %s (grade %s, score only %d over floor) — %s",
                                        self.name, direction_str, self._last_plan.grade,
                                        _score_over_floor, self._last_plan.thesis)
                            _shad.gates["no_draw"] = {"blocked": True, "grade": str(self._last_plan.grade)}; _shad.blocking_gate = "no_draw"; _shadow_logger.record(_shad)
                            continue
                        logger.info("[%s] NO-DRAW allow: %s grade C, score %d over floor — proceeding at 0.25x",
                                    self.name, direction_str, _score_over_floor)
                        _shad.gates["no_draw"] = {"blocked": False, "grade": "C_override", "score_over_floor": _score_over_floor}

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
                        _shad.gates["trade_agent"] = {"blocked": True, "dissent": list(ticket.dissent)}; _shad.blocking_gate = "trade_agent"; _shadow_logger.record(_shad)
                        continue

                    # ── OmniRoute CouncilRouter governance check ──────────────
                    # Selects top-fit Council personas via composite scoring,
                    # runs rule-based verdicts, emits COUNCIL_VERDICT to event bus.
                    try:
                        from council import CouncilRouter
                        if not hasattr(self, "_council_router"):
                            self._council_router = CouncilRouter()
                        _council_verdict = self._council_router.governance_check(
                            symbol       = self._symbol,
                            direction    = desired,
                            signal_score = signal_score,
                            daily_dd_pct = _inline_daily_dd,
                            lots         = lots,
                            equity       = equity,
                            sl           = sl or 0.0,
                            tp           = tp,
                            plan_type    = _plan_type,
                            atr_ratio    = (atr_entry / 1.0) if atr_entry else 1.0,
                            news_window  = bool(ctx.get("news_window") if (ctx := getattr(self, "_news_gate", None)) and hasattr(ctx, "news_window") else False),
                        )
                        for _cn in _council_verdict.notes:
                            logger.info("[%s] Council: %s", self.name, _cn)
                        if not _council_verdict.approved:
                            _shad.gates["council_router"] = {
                                "blocked": True,
                                "vetoed_by": _council_verdict.vetoed_by,
                                "notes": _council_verdict.notes,
                            }
                            _shad.blocking_gate = "council_router"
                            _shadow_logger.record(_shad)
                            continue
                        _shad.gates["council_router"] = {
                            "blocked": False,
                            "selected": _council_verdict.selected,
                            "profile": _council_verdict.strategy_profile,
                        }
                    except Exception as _cr_exc:
                        logger.debug("[%s] CouncilRouter error (non-blocking): %s", self.name, _cr_exc)

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

                    # ── Cousin quality router (JP mentor): GU/EU — only the
                    # higher-scoring cousin enters when both qualify same bar+direction.
                    if self._cousin_router is not None:
                        if not self._cousin_router.try_claim(bar_time, self._symbol, signal_score, desired):
                            logger.info(
                                "[%s] COUSIN SKIP: %s dir=%+d score=%d — better cousin claimed this bar",
                                self.name, self._symbol, desired, signal_score,
                            )
                            self._cousin_router.cleanup(bar_time)
                            _shad.gates["cousin_router"] = {"blocked": True}; _shad.blocking_gate = "cousin_router"; _shadow_logger.record(_shad)
                            continue
                        self._cousin_router.cleanup(bar_time)

                    _shad.fired = True; _shadow_logger.record(_shad)
                    # V3: consume the armed candidate so lifecycle advances to FIRED
                    if _v3_mode and _az is not None:
                        try:
                            _az.consume_armed(desired)
                        except Exception:
                            pass
                    # Anti-tilt: record equity at entry so close can compare
                    self._entry_equity = equity
                    if self._dry_run:
                        logger.info("[%s] DRY RUN: %s %.2f lots SL=%s TP=%s | score=%d x%.2f | %s",
                                    self.name, direction_str, lots, sl, tp,
                                    signal_score, combined_mult, reason)
                    else:
                        # Pre-order price snapshot for slippage measurement
                        _pre_tick = mt5.symbol_info_tick(self._symbol)
                        _intended_px = (
                            _pre_tick.ask if desired == 1 else _pre_tick.bid
                        ) if _pre_tick else None
                        _pre_spread = (
                            _pre_tick.ask - _pre_tick.bid
                        ) if _pre_tick else None
                        ok = trader.place_order(self._symbol, desired, lots, sl=sl, tp=tp)
                        if ok:
                            # Read confirmed fill from MT5 — price_open/sl/tp reflect the
                            # actual broker values after slippage, not the pre-order estimate.
                            _live_pos = trader.get_positions(self._symbol)
                            if _live_pos:
                                _lp = _live_pos[0]
                                self._open_entry_price = _lp.price_open
                                self._open_sl          = _lp.sl if _lp.sl > 0 else sl
                                self._open_tp          = _lp.tp if _lp.tp > 0 else tp
                                sl = self._open_sl    # use confirmed values downstream
                                tp = self._open_tp
                            else:
                                _pre_tick2 = mt5.symbol_info_tick(self._symbol)
                                self._open_entry_price = (
                                    _pre_tick2.ask if desired == 1 else _pre_tick2.bid
                                ) if _pre_tick2 else (_intended_px or 0.0)
                                self._open_sl          = sl
                                self._open_tp          = tp
                            # Execution quality log — slippage/spread per fill
                            try:
                                _eq_log = Path("logs") / "execution_quality.jsonl"
                                _eq_log.parent.mkdir(parents=True, exist_ok=True)
                                _slip = (
                                    (self._open_entry_price - _intended_px) * (1 if desired == 1 else -1)
                                ) if _intended_px else None
                                _eq_row = {
                                    "ts":         datetime.now(timezone.utc).isoformat(),
                                    "symbol":     self._symbol,
                                    "direction":  desired,
                                    "lots":       lots,
                                    "score":      signal_score,
                                    "intended":   _intended_px,
                                    "fill":       self._open_entry_price,
                                    "slippage":   round(_slip, 6) if _slip is not None else None,
                                    "spread":     round(_pre_spread, 6) if _pre_spread is not None else None,
                                    "sl":         self._open_sl,
                                    "tp":         self._open_tp,
                                }
                                with _eq_log.open("a", encoding="utf-8") as _ef:
                                    _ef.write(json.dumps(_eq_row) + "\n")
                            except Exception:
                                pass
                            self._open_score             = signal_score
                            self._t1_hit                 = False
                            self._bars_since_entry        = 0
                            self._entry_cooldown_until    = time.time() + 60  # 60s grace: block TM exits at bar-open
                            self._last_partial_bar        = 0
                            self._open_confluences        = _confl  # saved for Bayesian update at close
                            self._open_peak_r             = 0.0
                            if tp:
                                self._save_burned_target(desired, tp)
                            self._save_position_state()
                            if self._risk_agent is not None:
                                self._risk_agent.record_entry()
                            # Register this trade's quality in the shared book so
                            # the allocator can weigh it against future setups.
                            if self._book is not None:
                                for _p in trader.get_positions(self._symbol):
                                    self._book.register(_p.ticket, signal_score)
                            import numpy as np
                            _atr_series = getattr(self._strategy, "_atr_cache", None)
                            _atr_val = float(_atr_series.iloc[-1]) if _atr_series is not None and not np.isnan(float(_atr_series.iloc[-1])) else None
                            _p = self._last_plan
                            _sr = getattr(self._strategy, "_score_reasons", None)
                            _reasons = list(_sr.iloc[-1]) if _sr is not None and len(_sr) else []
                            _reasons = _reasons + _extra_reasons
                            try:
                                if self._journal is not None:
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
                            except Exception as _je:
                                logger.exception("[%s] Journal open_trade failed (trade live): %s", self.name, _je)
                            tg.notify_trade_open(
                                symbol=self._symbol,
                                direction=desired,
                                score=signal_score,
                                entry=self._open_entry_price,
                                sl=sl or 0.0,
                                tp=tp,
                                lots=lots,
                                equity=equity,
                                atr=_atr_val,
                                df=df,
                                reasons=_reasons,
                            )
                            try:
                                from execution.aiden_event_bus import append_event
                                append_event("TRADE_EXECUTED",
                                    symbol=self._symbol, direction=desired,
                                    score=signal_score, lots=lots, equity=equity)
                            except Exception:
                                pass

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
                daily_profit_target_pct=float(trade_cfg.get("daily_profit_target_pct", 0.0)) / 100,
                daily_profit_floor_pct=float(trade_cfg.get("daily_profit_floor_pct", 0.0)) / 100,
            ),
        )
        self._ftmo_tracker  = FTMOTracker(initial_equity=initial_equity, challenge=challenge_type)
        log_cfg             = cfg.get("logging", {})
        self._journal       = TradeJournal(log_dir=log_cfg.get("log_dir", "logs"))
        # Shared correlation matrix — updated each bar, read by all engines at entry
        self._corr_matrix   = CorrelationMatrix(lookback=100)

    # ── Startup ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info("=" * 60)
        logger.info("  ORCHESTRATOR STARTING")
        logger.info("  symbols=%s | tf=%s | dry_run=%s", self._symbols, self._tf_str, self.dry_run)
        logger.info("=" * 60)

        _mt5_retries = 0
        while not connect(self._terminal_path):
            _mt5_retries += 1
            if _mt5_retries >= 10:
                logger.critical("Cannot connect to MT5 after 10 attempts. Aborting.")
                tg.notify_council_flag("01 Principal", "MT5 connection FAILED after 10 retries.")
                sys.exit(1)
            _wait = min(30 * _mt5_retries, 300)  # 30s, 60s, ... max 5min
            logger.warning("MT5 connect failed (attempt %d/10) — retrying in %ds", _mt5_retries, _wait)
            time.sleep(_wait)

        self._news_gate = ng.NewsGate()
        self._news_gate.start()
        self._news_intel = NewsIntelligence(self._news_gate)

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

        # Sync system state to Obsidian vault on startup
        try:
            import threading
            from execution import obsidian_sync as _ob
            _preday = None
            try:
                import json as _json
                _p = Path(__file__).resolve().parent.parent / "logs" / "preday_brief.json"
                if _p.exists():
                    _preday = _json.loads(_p.read_text(encoding="utf-8"))
            except Exception:
                pass
            threading.Thread(
                target=_ob.write_system_state,
                args=(list(self._symbols), account.equity if account else 0.0, _preday),
                daemon=True,
            ).start()
        except Exception:
            pass

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
        amd_detector  = AMDDetector()
        cousin_router = CousinRouter()

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
            engine._news_gate     = self._news_gate
            engine._news_intel    = self._news_intel
            engine._level_monitor = level_monitor
            engine._amd_detector  = amd_detector
            engine._mc_agent      = MarketContextAgent(level_monitor)
            engine._cousin_router = cousin_router
            engine._corr_matrix   = self._corr_matrix
            components.append(engine)

            # Phase 3/4: AnalyzerEngine runs alongside every live engine.
            # Shadow mode (Phase 3): just logs. Live mode (Phase 4): drives signal.
            _v3_cfg    = self.cfg.get("v3_cutover", {})
            _v3_live   = set(_v3_cfg.get("live", []))
            _v3_min_samples = int(_v3_cfg.get("go_no_go_min_samples", 50))
            _born_thr  = float(_v3_cfg.get("born_threshold", 35.0))
            try:
                _az = AnalyzerEngine(symbol, self._strategies[symbol])
                _az._BORN_THRESHOLD = _born_thr
                if symbol in _v3_live:
                    _shadow_path = Path(__file__).parent.parent / "logs" / "shadow_stack.jsonl"
                    _ok, _msg = AnalyzerEngine.validate_cutover_ready(symbol, _shadow_path, _v3_min_samples)
                    if _ok:
                        engine._v3_mode = True
                        logger.info("[V3 CUTOVER] %s → LIVE on v3 stack. %s", symbol, _msg)
                    else:
                        engine._v3_mode = False
                        logger.warning("[V3 CUTOVER] %s blocked: %s", symbol, _msg)
                else:
                    engine._v3_mode = False
                engine._analyzer = _az
            except Exception as _ae:
                logger.warning("AnalyzerEngine init failed for %s: %s", symbol, _ae)
                engine._v3_mode = False

        self._components = components

        for comp in components:
            t = threading.Thread(target=self._run_component, args=(comp,),
                                 name=comp.name, daemon=True)
            t.start()
            self._threads.append(t)

        # Two-way Telegram command handler (Phase 5 upgrade)
        self._tg_cmd = TelegramCommandHandler(
            status_fn   = self._tg_status,
            pause_fn    = self._tg_pause,
            resume_fn   = self._tg_resume,
            flatten_fn  = self._tg_flatten,
            why_fn      = self._tg_why,
            set_risk_fn = self._tg_set_risk,
        )
        self._tg_cmd.start()

        self._monitor_loop()

    # ── Monitor loop ─────────────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        _preday_sent_date: Optional[str] = None   # track which UTC date brief was sent
        try:
            while not self.kill_switch.is_set():
                # _print_dashboard calls mt5.account_info() which can hang
                # indefinitely on weekend disconnects — run in a guarded thread.
                _dash_done = threading.Event()
                def _dash_target():
                    try:
                        self._print_dashboard()
                    finally:
                        _dash_done.set()
                _dash_thread = threading.Thread(target=_dash_target, name="Dashboard", daemon=True)
                _dash_thread.start()
                if not _dash_done.wait(timeout=30):
                    logger.warning("[Monitor] dashboard timed out (MT5 call hung >30s) — continuing")

                # Pre-day brief — send once per day at London open (07:00 UTC)
                _now = datetime.now(timezone.utc)
                _today_str = _now.strftime("%Y-%m-%d")
                # >= 7 with noon cutoff: a morning restart still sends the
                # brief; past noon it's stale, skip until tomorrow
                if (7 <= _now.hour < 12
                        and _preday_sent_date != _today_str):
                    try:
                        from execution.preday_analysis import run_preday_brief
                        import MetaTrader5 as _mt5_pd
                        _acct = _mt5_pd.account_info()
                        _eq   = _acct.equity if _acct else 0.0
                        run_preday_brief(self._symbols, _eq)
                        _preday_sent_date = _today_str
                    except Exception:
                        logger.exception("[PreDay] brief failed")

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

    # ── Telegram command callbacks ────────────────────────────────────────────

    def _tg_status(self) -> str:
        lines = []
        try:
            account = mt5.account_info()
            if account:
                lines.append(f"Equity: ${account.equity:,.2f}  Balance: ${account.balance:,.2f}")
                lines.append(self._ftmo_tracker.status_line(account.equity))
        except Exception:
            pass
        halt_state = "HALTED" if self.soft_halt.is_set() else "active"
        lines.append(f"Bot: {halt_state}")
        rows = self.registry.snapshot()
        stale = [r["name"] for r in rows if r["status"] not in ("HEALTHY",)]
        if stale:
            lines.append(f"Stale/failed: {', '.join(stale)}")
        else:
            lines.append(f"Components: {len(rows)} all healthy")
        try:
            positions = mt5.positions_get()
            if positions:
                for p in positions:
                    lines.append(
                        f"  {p.symbol} {'LONG' if p.type==0 else 'SHORT'} "
                        f"lots={p.volume:.2f} P&L=${p.profit:+.2f}"
                    )
            else:
                lines.append("No open positions")
        except Exception:
            pass
        return "\n".join(lines)

    def _tg_pause(self) -> None:
        self.soft_halt.set()
        logger.info("[TgCmd] Soft halt SET via Telegram command")

    def _tg_resume(self) -> None:
        self.soft_halt.clear()
        logger.info("[TgCmd] Soft halt CLEARED via Telegram command")

    def _tg_flatten(self) -> str:
        closed = 0
        errors = 0
        for sym in self._symbols:
            try:
                n = trader.close_all(sym)
                closed += n
            except Exception as exc:
                logger.warning("[TgCmd] flatten error for %s: %s", sym, exc)
                errors += 1
        msg = f"Closed {closed} position(s) across {len(self._symbols)} symbol(s)."
        if errors:
            msg += f" {errors} error(s) — check logs."
        logger.info("[TgCmd] %s", msg)
        return msg

    def _tg_why(self) -> str:
        shadow_path = Path(__file__).parent.parent / "logs" / "shadow_stack.jsonl"
        if not shadow_path.exists():
            return "No shadow log yet — gates haven't run since restart."
        try:
            last_line = None
            with shadow_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        last_line = line.strip()
            if not last_line:
                return "Shadow log is empty."
            row = json.loads(last_line)
            symbol  = row.get("symbol", "?")
            gate    = row.get("blocking_gate", "none")
            bar     = row.get("bar_time", "?")
            score   = row.get("stack_score", row.get("score", "?"))
            direc   = "+1 (LONG)" if row.get("signal_dir", 0) > 0 else "-1 (SHORT)"
            msg = f"Symbol: {symbol}  Direction: {direc}\nBar: {bar}\nBlocked by: <b>{gate}</b>\nScore: {score}"
            gates = row.get("gates", {})
            if gates:
                gate_lines = []
                for g, v in list(gates.items())[-5:]:
                    if isinstance(v, dict) and v.get("blocked"):
                        gate_lines.append(f"  ✗ {g}")
                    elif isinstance(v, dict):
                        gate_lines.append(f"  ✓ {g}")
                if gate_lines:
                    msg += "\n" + "\n".join(gate_lines)
            return msg
        except Exception as exc:
            return f"Error reading shadow log: {exc}"

    def _tg_set_risk(self, pct: float) -> str:
        self._trade_cfg["risk_pct"] = pct
        for comp in self._components:
            if hasattr(comp, "_trade_cfg"):
                comp._trade_cfg["risk_pct"] = pct
        logger.info("[TgCmd] risk_pct set to %.2f via Telegram", pct)
        return f"risk_pct={pct:.2f}%"

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
        if hasattr(self, "_tg_cmd"):
            self._tg_cmd.stop()
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

def _pid_is_python(pid: int) -> bool:
    """True if pid is a live python process (Windows tasklist, no deps)."""
    try:
        import subprocess
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=10,
        )
        return "python" in out.stdout.lower()
    except Exception:
        return True   # can't verify — fail SAFE: assume it's running


def _acquire_pid_lock() -> Path:
    """Ensure only one instance of the bot runs at a time.

    Writes our PID to logs/bot.pid. If a prior PID file exists and the
    process is still alive, aborts immediately. This prevents the double-entry
    bug where two instances trade the same instruments and fight each other.
    """
    pid_path = Path("logs/bot.pid")
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    if pid_path.exists():
        try:
            existing_pid = int(pid_path.read_text().strip())
        except ValueError:
            existing_pid = None   # corrupt file — overwrite
        # No psutil dependency: the previous psutil-based check silently
        # passed on ImportError, which let a second instance start
        # (2026-07-27 — three instances found trading concurrently).
        if existing_pid is not None and _pid_is_python(existing_pid):
            raise SystemExit(
                f"Bot already running (PID {existing_pid}). "
                f"Stop the existing instance before starting a new one. "
                f"If it crashed, delete logs/bot.pid manually."
            )
    pid_path.write_text(str(os.getpid()))
    return pid_path


def main():
    parser = argparse.ArgumentParser(description="Trading bot orchestrator.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log signals only — no real orders placed")
    parser.add_argument("--symbol", default=None,
                        help="Single symbol (default: all in config)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    pid_path = _acquire_pid_lock()
    try:
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
    finally:
        pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
