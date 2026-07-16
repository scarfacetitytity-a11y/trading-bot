"""Risk Psychology Agent.

Acts as a gate before every live trade. Tracks consecutive losses,
daily drawdown, weekly drawdown, and recent win rate, then decides:
  - Can we trade at all right now?
  - What size multiplier should we apply (1.0 = full, 0.5 = half, 0.0 = pause)?

Psychology rules applied:
  1. Consecutive losses  — after N losses in a row, pause for cooldown period
  2. Daily loss limit    — if today's loss exceeds X% of account, stop for day
  3. Weekly drawdown     — if week's DD exceeds Y%, pause until Monday
  4. Rolling WR check    — if last N trades WR < threshold, scale down size
  5. Account drawdown    — if overall DD > Z%, halt all trading immediately

State is persisted to logs/risk_agent_state.json so it survives restarts.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional

STATE_FILE = Path(__file__).resolve().parent.parent / "logs" / "risk_agent_state.json"


@dataclass
class RiskConfig:
    # Consecutive loss circuit breaker
    max_consecutive_losses: int   = 3       # halt after this many losses in a row
    consecutive_loss_cooldown_h: int = 24   # hours to pause after hitting limit

    # Daily loss limit (% of account balance at day start)
    max_daily_loss_pct: float     = 0.02    # 2% — Council circuit breaker (FTMO limit is 5%)

    # Weekly drawdown limit (% of week-start balance)
    max_weekly_dd_pct: float      = 0.07    # 7%

    # Account-level circuit breaker (% from peak equity)
    max_account_dd_pct: float     = 0.07    # 7% — Council soft halt (FTMO limit is 10%)

    # Rolling win rate — scale down if recent WR is bad
    wr_lookback_trades: int       = 10      # last N trades
    wr_scale_threshold: float     = 0.35    # WR below this → half size
    wr_halt_threshold: float      = 0.20    # WR below this → pause

    # Max open trades at once — risk is governed by risk_pct per trade, not a count cap.
    # Set high so valid setups never get blocked by position count alone.
    max_concurrent_trades: int    = 20


@dataclass
class RiskState:
    consecutive_losses: int       = 0
    pause_until: Optional[str]    = None    # ISO datetime string
    daily_start_equity: float     = 0.0
    daily_date: str               = ""      # YYYY-MM-DD
    weekly_start_equity: float    = 0.0
    weekly_start_date: str        = ""      # ISO date of Monday
    peak_equity: float            = 0.0
    current_equity: float         = 0.0
    recent_trades: list           = field(default_factory=list)  # list of R multiples
    trade_log: list               = field(default_factory=list)  # full history

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RiskAgent:
    """Psychology-aware risk manager for live trading.

    Usage:
        agent = RiskAgent(initial_equity=10_000)

        # Before placing a trade:
        can_trade, size_mult, reason = agent.pre_trade_check(current_equity)
        if can_trade:
            lots = base_lots * size_mult
            place_trade(lots)

        # After a trade closes:
        agent.record_trade(r_multiple=1.8, equity_after=10_180)
    """

    def __init__(
        self,
        initial_equity: float = 10_000,
        config: RiskConfig | None = None,
    ):
        self.config = config or RiskConfig()
        self.state  = self._load_state()

        # Initialise equity tracking on first run
        if self.state.current_equity == 0.0:
            self.state.current_equity    = initial_equity
            self.state.peak_equity       = initial_equity
            self.state.daily_start_equity  = initial_equity
            self.state.weekly_start_equity = initial_equity
            self.state.daily_date        = str(date.today())
            self.state.weekly_start_date = str(_monday())
            self._save_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def pre_trade_check(
        self,
        current_equity: float,
        open_trade_count: int = 0,
    ) -> tuple[bool, float, str]:
        """Return (can_trade, size_multiplier, reason_string).

        size_multiplier: 1.0 = full size, 0.5 = reduced, 0.0 = no trade.
        """
        self._refresh_daily_weekly(current_equity)
        cfg = self.config
        s   = self.state

        # 1. Active pause (consecutive losses or manual pause)
        if s.pause_until:
            pause_dt = datetime.fromisoformat(s.pause_until).replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) < pause_dt:
                return False, 0.0, f"Paused until {s.pause_until} (consecutive losses)"
            else:
                s.pause_until = None

        # 2. Concurrent trade cap
        if open_trade_count >= cfg.max_concurrent_trades:
            return False, 0.0, f"Max concurrent trades ({cfg.max_concurrent_trades}) reached"

        # 3. Account DD circuit breaker
        if s.peak_equity > 0:
            account_dd = (s.peak_equity - current_equity) / s.peak_equity
            if account_dd >= cfg.max_account_dd_pct:
                return False, 0.0, (
                    f"Account DD {account_dd*100:.1f}% >= {cfg.max_account_dd_pct*100:.0f}% limit — HALT"
                )

        # 4. Daily loss limit
        if s.daily_start_equity > 0:
            daily_loss = (s.daily_start_equity - current_equity) / s.daily_start_equity
            if daily_loss >= cfg.max_daily_loss_pct:
                return False, 0.0, (
                    f"Daily loss {daily_loss*100:.1f}% >= {cfg.max_daily_loss_pct*100:.0f}% — done for today"
                )

        # 5. Weekly DD limit
        if s.weekly_start_equity > 0:
            weekly_dd = (s.weekly_start_equity - current_equity) / s.weekly_start_equity
            if weekly_dd >= cfg.max_weekly_dd_pct:
                return False, 0.0, (
                    f"Weekly DD {weekly_dd*100:.1f}% >= {cfg.max_weekly_dd_pct*100:.0f}% — wait until Monday"
                )

        # 6. Rolling win rate check
        size_mult = 1.0
        reason    = "OK"
        if len(s.recent_trades) >= cfg.wr_lookback_trades:
            recent = s.recent_trades[-cfg.wr_lookback_trades:]
            wr = sum(1 for r in recent if r > 0) / len(recent)
            if wr <= cfg.wr_halt_threshold:
                return False, 0.0, (
                    f"Rolling WR {wr*100:.0f}% <= {cfg.wr_halt_threshold*100:.0f}% — pause"
                )
            if wr <= cfg.wr_scale_threshold:
                size_mult = 0.5
                reason = f"Rolling WR {wr*100:.0f}% — trading at half size"

        self._save_state()
        return True, size_mult, reason

    def record_trade(self, r_multiple: float, equity_after: float):
        """Call after every trade closes."""
        s = self.state
        s.current_equity = equity_after
        s.peak_equity    = max(s.peak_equity, equity_after)

        # Update consecutive losses
        if r_multiple < 0:
            s.consecutive_losses += 1
        else:
            s.consecutive_losses = 0

        # Trigger pause if consecutive loss limit hit
        if s.consecutive_losses >= self.config.max_consecutive_losses:
            cooldown = timedelta(hours=self.config.consecutive_loss_cooldown_h)
            s.pause_until = (datetime.now(timezone.utc) + cooldown).isoformat()
            print(
                f"  [RiskAgent] {s.consecutive_losses} consecutive losses — "
                f"pausing until {s.pause_until}"
            )

        # Log trade
        s.recent_trades.append(r_multiple)
        if len(s.recent_trades) > 100:
            s.recent_trades = s.recent_trades[-100:]

        s.trade_log.append({
            "time":     datetime.now(timezone.utc).isoformat(),
            "r":        r_multiple,
            "equity":   equity_after,
        })
        if len(s.trade_log) > 1000:
            s.trade_log = s.trade_log[-1000:]

        self._save_state()

    def status_report(self, current_equity: float) -> str:
        """Human-readable status string for dashboard / logs."""
        s   = self.state
        cfg = self.config
        self._refresh_daily_weekly(current_equity)

        daily_pnl  = current_equity - s.daily_start_equity
        weekly_pnl = current_equity - s.weekly_start_equity
        account_dd = (s.peak_equity - current_equity) / s.peak_equity * 100 if s.peak_equity else 0
        recent     = s.recent_trades[-cfg.wr_lookback_trades:] if s.recent_trades else []
        wr         = sum(1 for r in recent if r > 0) / len(recent) * 100 if recent else 0

        lines = [
            f"  Equity        : ${current_equity:,.2f}  (peak ${s.peak_equity:,.2f})",
            f"  Account DD    : {account_dd:.1f}% (limit {cfg.max_account_dd_pct*100:.0f}%)",
            f"  Daily P&L     : ${daily_pnl:+,.2f} ({daily_pnl/s.daily_start_equity*100:+.1f}%)",
            f"  Weekly P&L    : ${weekly_pnl:+,.2f} ({weekly_pnl/s.weekly_start_equity*100:+.1f}%)",
            f"  Consec losses : {s.consecutive_losses}/{cfg.max_consecutive_losses}",
            f"  Rolling WR    : {wr:.0f}% (last {len(recent)} trades)",
            f"  Paused until  : {s.pause_until or 'N/A'}",
        ]
        return "\n".join(lines)

    def reset_daily(self, current_equity: float):
        """Call at start of each trading day."""
        self.state.daily_start_equity = current_equity
        self.state.daily_date         = str(date.today())
        self._save_state()

    def reset_weekly(self, current_equity: float):
        """Call every Monday."""
        self.state.weekly_start_equity = current_equity
        self.state.weekly_start_date   = str(_monday())
        self._save_state()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _refresh_daily_weekly(self, current_equity: float):
        today = str(date.today())
        mon   = str(_monday())

        if self.state.daily_date != today:
            self.state.daily_start_equity = current_equity
            self.state.daily_date         = today

        if self.state.weekly_start_date != mon:
            self.state.weekly_start_equity = current_equity
            self.state.weekly_start_date   = mon

    def _load_state(self) -> RiskState:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    return RiskState.from_dict(json.load(f))
            except Exception:
                pass
        return RiskState()

    def _save_state(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2, default=str)


def _monday() -> date:
    today = date.today()
    return today - timedelta(days=today.weekday())
