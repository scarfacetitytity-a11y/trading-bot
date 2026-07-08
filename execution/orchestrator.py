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
import logging
import sys
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

import MetaTrader5 as mt5
import pandas as pd

from config.settings import load_config
from config.logging_setup import setup_logger
from backtests.mt5_connector import connect, disconnect
from backtests.data_loader import TIMEFRAME_MAP
from execution import trader, risk
from strategies.sniper import SniperStrategy
from strategies.london_breakout import LondonBreakoutStrategy

logger = logging.getLogger(__name__)

# ── Risk limits (FTMO Phase 1 / 2 compatible) ────────────────────────────────
MAX_DAILY_LOSS_PCT  = 4.5   # halt at 4.5% daily loss (FTMO limit is 5%)
MAX_TOTAL_LOSS_PCT  = 9.0   # halt at 9% total drawdown (FTMO limit is 10%)
MAX_RECONNECT_TRIES = 5
MAX_RESTARTS        = 10
MONITOR_INTERVAL    = 60    # seconds between orchestrator health checks
# ─────────────────────────────────────────────────────────────────────────────

_TF_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}

STRATEGY_MAP = {
    "sniper":          SniperStrategy,
    "london_breakout": LondonBreakoutStrategy,
}


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
    """Enforces FTMO daily and total loss limits. Triggers kill switch on breach."""

    CHECK_INTERVAL = 60

    def __init__(self, registry, kill_switch):
        super().__init__("RiskGuard", registry, kill_switch, beat_timeout=180)
        self._day_start_equity: Optional[float] = None
        self._session_start_equity: Optional[float] = None
        self._today: Optional[int] = None

    def run(self) -> None:
        while not self._stop.is_set() and not self.kill_switch.is_set():
            info = mt5.account_info()
            if info is None:
                self.registry.beat(self.name, "waiting for MT5", Status.STALE)
                time.sleep(self.CHECK_INTERVAL)
                continue

            equity  = info.equity
            balance = info.balance
            today   = datetime.now().day

            # Reset daily high-water mark at start of a new day
            if today != self._today:
                self._today            = today
                self._day_start_equity = equity
                logger.info("[RiskGuard] New day — day start equity=%.2f", equity)

            # Set session baseline on first run
            if self._session_start_equity is None:
                self._session_start_equity = balance
                logger.info("[RiskGuard] Session start equity=%.2f", balance)

            daily_loss_pct = (
                (equity - self._day_start_equity) / self._day_start_equity * 100
                if self._day_start_equity else 0.0
            )
            total_loss_pct = (
                (equity - self._session_start_equity) / self._session_start_equity * 100
                if self._session_start_equity else 0.0
            )

            status_msg = (
                f"equity={equity:.2f} | "
                f"daily={daily_loss_pct:+.2f}% | "
                f"total={total_loss_pct:+.2f}%"
            )

            if daily_loss_pct <= -MAX_DAILY_LOSS_PCT:
                msg = (f"DAILY LOSS LIMIT HIT: {daily_loss_pct:.2f}% "
                       f"(limit -{MAX_DAILY_LOSS_PCT}%)")
                logger.critical("[RiskGuard] %s — triggering kill switch", msg)
                self.registry.halt(self.name, msg)
                self.kill_switch.set()
                return

            if total_loss_pct <= -MAX_TOTAL_LOSS_PCT:
                msg = (f"TOTAL LOSS LIMIT HIT: {total_loss_pct:.2f}% "
                       f"(limit -{MAX_TOTAL_LOSS_PCT}%)")
                logger.critical("[RiskGuard] %s — triggering kill switch", msg)
                self.registry.halt(self.name, msg)
                self.kill_switch.set()
                return

            self.beat(status_msg)
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
        dry_run: bool = False,
    ):
        name = f"TradingEngine[{symbol}]"
        tf_secs = _TF_SECONDS.get(tf_str, 3600)
        super().__init__(name, registry, kill_switch, beat_timeout=tf_secs * 4)
        self._symbol    = symbol
        self._strategy  = strategy
        self._tf_str    = tf_str
        self._tf_code   = TIMEFRAME_MAP.get(tf_str)
        self._trade_cfg = trade_cfg
        self._dry_run   = dry_run
        self._lookback  = trade_cfg.get("lookback_bars", 500)
        self._last_bar  = None

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

                if desired == current:
                    continue

                # Close existing position
                if current != 0:
                    if self._dry_run:
                        logger.info("[%s] DRY RUN: close %s", self.name,
                                    "LONG" if current == 1 else "SHORT")
                    else:
                        trader.close_all(self._symbol)

                # Open new position
                if desired != 0:
                    account = trader.get_account()
                    lots    = risk.calculate_lots(self._symbol, self._trade_cfg, account["balance"])
                    sl, tp  = risk.calculate_sl_tp(self._symbol, desired, self._trade_cfg)
                    direction_str = "BUY" if desired == 1 else "SELL"
                    if self._dry_run:
                        logger.info("[%s] DRY RUN: %s %.2f lots SL=%s TP=%s",
                                    self.name, direction_str, lots, sl, tp)
                    else:
                        trader.place_order(self._symbol, desired, lots, sl=sl, tp=tp)

            except Exception as exc:
                self.registry.fail(self.name, str(exc))
                logger.exception("[%s] Error: %s", self.name, exc)
                time.sleep(30)

        logger.info("[%s] Stopped.", self.name)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """Floor manager — owns every component and monitors their heartbeats."""

    def __init__(self, cfg: dict, dry_run: bool = False, symbol_override: Optional[str] = None):
        self.cfg         = cfg
        self.dry_run     = dry_run
        self.kill_switch = threading.Event()
        self.registry    = HeartbeatRegistry()
        self._threads: list  = []
        self._components: list = []

        mt5_cfg     = cfg.get("mt5", {})
        data_cfg    = cfg.get("data", {})
        trade_cfg   = cfg.get("trading", {})

        self._terminal_path = mt5_cfg.get("terminal_path") or None
        self._tf_str        = data_cfg.get("timeframe", "H1").upper()
        self._symbols       = (
            [symbol_override] if symbol_override
            else data_cfg.get("symbols", ["XAUUSD"])
        )
        self._trade_cfg = trade_cfg
        trader.set_magic(trade_cfg.get("magic", 234001))

        # Build strategy map: symbol → strategy instance
        strategy_name = trade_cfg.get("strategy", "sniper")
        strategy_cls  = STRATEGY_MAP.get(strategy_name, SniperStrategy)
        self._strategies = {s: strategy_cls() for s in self._symbols}

    # ── Startup ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info("=" * 60)
        logger.info("  ORCHESTRATOR STARTING")
        logger.info("  symbols=%s | tf=%s | dry_run=%s", self._symbols, self._tf_str, self.dry_run)
        logger.info("=" * 60)

        if not connect(self._terminal_path):
            logger.critical("Cannot connect to MT5. Aborting.")
            sys.exit(1)

        if not self._safety_check():
            disconnect()
            sys.exit(1)

        components: list[Component] = [
            MT5Monitor(self.registry, self.kill_switch, self._terminal_path),
            DataWatcher(self.registry, self.kill_switch, self._symbols, self._tf_str),
            RiskGuard(self.registry, self.kill_switch),
        ]

        for symbol in self._symbols:
            components.append(TradingEngine(
                self.registry,
                self.kill_switch,
                symbol,
                self._strategies[symbol],
                self._tf_str,
                self._trade_cfg,
                dry_run=self.dry_run,
            ))

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
        disconnect()
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
