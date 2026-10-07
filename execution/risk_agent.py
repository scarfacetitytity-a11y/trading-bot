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
    # Consecutive loss size scaling — graduated, not a halt
    # 3 losses → 0.5x size, 5 losses → 0.25x size, 7+ → 0.1x (minimum)
    # Agent manages risk, it doesn't stop trading.
    max_consecutive_losses: int   = 7       # only halt at extreme (7 in a row)
    consecutive_loss_cooldown_h: int = 4    # short cooldown if 7-loss extreme hit

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

    # Max open trades at once. Was 20 ("risk is governed by risk_pct, not a count"),
    # which was no cap; now bounded by the invariant envelope (8 = 4% portfolio
    # cap / 0.5% min trade risk, so non-binding for current config).
    max_concurrent_trades: int    = 8

    # Max entries per UTC day across all symbols — JP mentor v7: "I do more when I do less.
    # Decision fatigue kicks in; every trade after the first few is lower quality."
    max_daily_entries: int        = 6

    # Daily profit floor — JP mentor: "I am not going to leave the financial markets today
    # without $1,500 clear profit… that means I've got $500 to play with."
    # Once daily gain >= daily_profit_target_pct, protect gains down to daily_profit_floor_pct.
    # Any new trade whose worst case (1× risk loss) would push P&L below the floor is blocked.
    # Set to 0.0 to disable (default: disabled until live trading proves the system).
    daily_profit_target_pct: float = 0.0   # % of day-start equity; 0 = feature off
    daily_profit_floor_pct:  float = 0.0   # % of day-start equity to protect

    def __post_init__(self):
        # Envelope clamp: whatever the caller passes, RiskAgent can never be
        # configured looser than core/system_invariants.
        from core.system_invariants import INVARIANTS as I
        self.max_daily_loss_pct    = min(self.max_daily_loss_pct, I.max_daily_loss_pct / 100)
        self.max_weekly_dd_pct     = min(self.max_weekly_dd_pct, I.max_weekly_dd_pct / 100)
        self.max_account_dd_pct    = min(self.max_account_dd_pct, I.max_total_kill_pct / 100)
        self.max_concurrent_trades = min(self.max_concurrent_trades, I.max_concurrent_trades)
        self.max_daily_entries     = min(self.max_daily_entries, I.max_daily_entries)
        self.max_consecutive_losses = min(self.max_consecutive_losses,
                                          I.max_consecutive_losses_before_pause)


@dataclass
class RiskState:
    consecutive_losses: int       = 0
    pause_until: Optional[str]    = None    # ISO datetime string
    daily_start_equity: float     = 0.0
    daily_date: str               = ""      # YYYY-MM-DD
    daily_entries: int            = 0       # entries placed today (UTC day)
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
            self.state.daily_date        = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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

        # 6. Daily entry cap — JP mentor v7: "I do more when I do less."
        # Decision fatigue degrades trade quality; hard cap per UTC day.
        if s.daily_entries >= cfg.max_daily_entries:
            return False, 0.0, (
                f"Daily entry cap ({cfg.max_daily_entries}) reached — done for today"
            )

        # 6b. Daily profit floor — JP mentor: "I've got $500 to play with"
        # Once the day's gain hits the target, only risk funds above the floor.
        # If a new loss (1× risk) would push today's P&L below the floor, block it.
        if cfg.daily_profit_target_pct > 0 and s.daily_start_equity > 0:
            daily_gain_pct = (current_equity - s.daily_start_equity) / s.daily_start_equity
            if daily_gain_pct >= cfg.daily_profit_target_pct:
                floor_equity = s.daily_start_equity * (1 + cfg.daily_profit_floor_pct)
                risk_loss    = current_equity * (cfg.max_daily_loss_pct / cfg.max_daily_entries)
                if current_equity - risk_loss < floor_equity:
                    return False, 0.0, (
                        f"Profit floor active: +{daily_gain_pct*100:.1f}% today — "
                        f"protecting floor {cfg.daily_profit_floor_pct*100:.1f}%"
                    )

        # 7. Graduated consecutive-loss size scaling
        # Agent manages risk by shrinking size — never shuts down except at extreme.
        size_mult = 1.0
        reason    = "OK"
        cl = s.consecutive_losses
        if cl >= 5:
            size_mult = 0.25
            reason = f"{cl} consecutive losses — 0.25x size"
        elif cl >= 3:
            size_mult = 0.5
            reason = f"{cl} consecutive losses — 0.5x size"
        elif cl >= 1:
            size_mult = 0.75
            reason = f"{cl} consecutive loss — 0.75x size"

        # 8. Rolling win rate — additional scale-down on sustained poor WR
        if len(s.recent_trades) >= cfg.wr_lookback_trades:
            recent = s.recent_trades[-cfg.wr_lookback_trades:]
            wr = sum(1 for r in recent if r > 0) / len(recent)
            if wr <= cfg.wr_halt_threshold:
                size_mult = min(size_mult, 0.25)
                reason = f"Rolling WR {wr*100:.0f}% — 0.25x size (sustained poor form)"
            elif wr <= cfg.wr_scale_threshold:
                size_mult = min(size_mult, 0.5)
                reason = f"Rolling WR {wr*100:.0f}% — 0.5x size"

        self._save_state()
        return True, size_mult, reason

    def record_entry(self):
        """Call when a new trade is placed (before it closes) to count daily entries."""
        self.state.daily_entries += 1
        self._save_state()

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

    def status_report(self, current_equity: float, p_win: float = 0.4,
                      risk_pct: float = 0.01) -> str:
        """Human-readable status string for dashboard / logs."""
        from execution.portfolio_optimizer import var_cvar as _var_cvar
        s   = self.state
        cfg = self.config
        self._refresh_daily_weekly(current_equity)

        daily_pnl  = current_equity - s.daily_start_equity
        weekly_pnl = current_equity - s.weekly_start_equity
        account_dd = (s.peak_equity - current_equity) / s.peak_equity * 100 if s.peak_equity else 0
        recent     = s.recent_trades[-cfg.wr_lookback_trades:] if s.recent_trades else []
        wr         = sum(1 for r in recent if r > 0) / len(recent) * 100 if recent else 0

        # VaR/CVaR for next potential trade at current risk settings
        vm = _var_cvar(p_win, risk_pct)

        lines = [
            f"  Equity        : ${current_equity:,.2f}  (peak ${s.peak_equity:,.2f})",
            f"  Account DD    : {account_dd:.1f}% (limit {cfg.max_account_dd_pct*100:.0f}%)",
            f"  Daily P&L     : ${daily_pnl:+,.2f} ({daily_pnl/s.daily_start_equity*100:+.1f}%)",
            f"  Weekly P&L    : ${weekly_pnl:+,.2f} ({weekly_pnl/s.weekly_start_equity*100:+.1f}%)",
            f"  Consec losses : {s.consecutive_losses}/{cfg.max_consecutive_losses}",
            f"  Rolling WR    : {wr:.0f}% (last {len(recent)} trades)",
            f"  VaR(95)/trade : {vm.var_95*100:+.3f}%  CVaR(95): {vm.cvar_95*100:+.3f}%  EV: {vm.ev*100:+.4f}%",
            f"  Paused until  : {s.pause_until or 'N/A'}",
        ]
        return "\n".join(lines)

    def reset_daily(self, current_equity: float):
        """Call at start of each trading day."""
        self.state.daily_start_equity = current_equity
        self.state.daily_date         = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._save_state()

    def reset_weekly(self, current_equity: float):
        """Call every Monday."""
        self.state.weekly_start_equity = current_equity
        self.state.weekly_start_date   = str(_monday())
        self._save_state()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _refresh_daily_weekly(self, current_equity: float):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        mon   = str(_monday())

        if self.state.daily_date != today:
            self.state.daily_start_equity = current_equity
            self.state.daily_date         = today
            self.state.daily_entries      = 0   # reset entry count each UTC day

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
