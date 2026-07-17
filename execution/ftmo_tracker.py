"""FTMO challenge progress tracker.

Tracks profit progress, days remaining, minimum trading days met, and current
drawdown vs FTMO limits. Supports both 2-Step and 1-Step challenge types.

FTMO 2-Step rules (default):
  Phase 1: 10% profit target | 5% daily DD | 10% total DD | 30-day window | 4 min trade days
  Phase 2: 5%  profit target | 5% daily DD | 10% total DD | 60-day window | 4 min trade days

FTMO 1-Step rules:
  10% profit target | 3% daily DD | 10% total DD | no min trade days
  Best Day Rule: largest winning day ≤ 50% of sum of all positive days
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from datetime import date
from pathlib import Path
from typing import Literal

STATE_FILE = Path(__file__).resolve().parent.parent / "logs" / "ftmo_tracker_state.json"

# 2-Step Phase 1 defaults (most common challenge type)
_RULES = {
    "2step-p1": dict(profit_target=10.0, daily_dd=5.0, total_dd=10.0, window=30, min_days=4),
    "2step-p2": dict(profit_target=5.0,  daily_dd=5.0, total_dd=10.0, window=60, min_days=4),
    "1step":    dict(profit_target=10.0, daily_dd=3.0, total_dd=10.0, window=None, min_days=0),
}


@dataclass
class _FTMOState:
    start_date:       str
    initial_equity:   float
    target_equity:    float
    peak_equity:      float
    trade_days:       list = field(default_factory=list)   # ISO dates with at least 1 trade
    daily_pnl:        dict = field(default_factory=dict)   # date → pnl (for Best Day Rule)
    day_start_equity: float = 0.0                          # equity at start of current day
    day_start_date:   str   = ""                           # date the above was anchored
    last_equity:      float = 0.0                          # most recent recorded close

    @classmethod
    def new(cls, equity: float, profit_target_pct: float) -> "_FTMOState":
        return cls(
            start_date=str(date.today()),
            initial_equity=equity,
            target_equity=round(equity * (1 + profit_target_pct / 100), 2),
            peak_equity=equity,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "_FTMOState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


ChallengeType = Literal["2step-p1", "2step-p2", "1step"]


class FTMOTracker:
    """Stateful FTMO challenge progress monitor.

    Usage:
        tracker = FTMOTracker(initial_equity=10_000, challenge="2step-p1")
        logger.info(tracker.report(account_equity))   # on startup
        logger.info(tracker.status_line(equity))       # on each dashboard tick
        tracker.record_trade_day()                     # call once per calendar day traded
    """

    def __init__(
        self,
        initial_equity: float = 10_000,
        challenge: ChallengeType = "2step-p1",
    ):
        self.rules = _RULES[challenge]
        self.challenge = challenge
        self.state = self._load_or_init(initial_equity)

    # ── Public API ────────────────────────────────────────────────────────────

    def record_trade_day(self, equity_close: float | None = None) -> None:
        """Call once per day on which at least one trade was placed."""
        today = str(date.today())
        s = self.state
        if today not in s.trade_days:
            s.trade_days.append(today)
        if equity_close is not None:
            # Anchor today's baseline to the equity carried from the previous
            # recorded day (its close), NOT to peak equity — peak overstates the
            # daily loss and corrupts the 1-Step Best Day Rule.
            if s.day_start_date != today:
                s.day_start_equity = s.last_equity if s.last_equity else s.initial_equity
                s.day_start_date   = today
            s.daily_pnl[today] = equity_close - s.day_start_equity
            s.last_equity      = equity_close
        self._save()

    def check(self, current_equity: float) -> dict:
        """Return structured progress dict. Advances peak equity and saves state."""
        s = self.state
        if current_equity > s.peak_equity:
            s.peak_equity = current_equity
            self._save()

        rules     = self.rules
        start     = date.fromisoformat(s.start_date)
        elapsed   = (date.today() - start).days
        window    = rules["window"]
        remaining = max(0, window - elapsed) if window else None

        profit_made      = current_equity - s.initial_equity
        profit_pct       = profit_made / s.initial_equity * 100
        profit_needed    = max(0.0, s.target_equity - current_equity)
        progress_pct     = min(100.0, profit_pct / rules["profit_target"] * 100)
        total_dd_pct     = (s.peak_equity - current_equity) / s.peak_equity * 100 if s.peak_equity else 0.0

        trading_days_met = len(s.trade_days) >= rules["min_days"]
        expired = (remaining == 0 and current_equity < s.target_equity) if window else False
        passed  = current_equity >= s.target_equity and trading_days_met

        # Best Day Rule (1-Step only)
        best_day_violation = False
        if self.challenge == "1step" and s.daily_pnl:
            positive = [v for v in s.daily_pnl.values() if v > 0]
            if positive:
                total_positive = sum(positive)
                max_day = max(positive)
                best_day_violation = max_day > 0.5 * total_positive

        return {
            "passed":              passed,
            "expired":             expired,
            "elapsed_days":        elapsed,
            "remaining_days":      remaining,
            "window_days":         window,
            "current_equity":      current_equity,
            "initial_equity":      s.initial_equity,
            "target_equity":       s.target_equity,
            "profit_made":         profit_made,
            "profit_pct":          profit_pct,
            "profit_needed":       profit_needed,
            "progress_pct":        progress_pct,
            "total_dd_pct":        total_dd_pct,
            "peak_equity":         s.peak_equity,
            "trade_days_count":    len(s.trade_days),
            "min_days_required":   rules["min_days"],
            "trading_days_met":    trading_days_met,
            "best_day_violation":  best_day_violation,
            "daily_dd_limit":      rules["daily_dd"],
            "total_dd_limit":      rules["total_dd"],
        }

    def status_line(self, current_equity: float) -> str:
        p = self.check(current_equity)
        if p["passed"]:
            tag = "PASSED"
        elif p["expired"]:
            tag = "EXPIRED"
        elif p["best_day_violation"]:
            tag = "BEST-DAY-WARN"
        else:
            tag = "ACTIVE"

        day_str = (
            f"Day {p['elapsed_days']}/{p['window_days']} ({p['remaining_days']}d left)"
            if p["window_days"] else
            f"Day {p['elapsed_days']} (no window limit)"
        )
        min_days_str = (
            f" | Trade days {p['trade_days_count']}/{p['min_days_required']}"
            if p["min_days_required"] else ""
        )
        return (
            f"[FTMO {tag}] {day_str}{min_days_str} | "
            f"P&L ${p['profit_made']:+,.2f} ({p['profit_pct']:+.2f}%) "
            f"→ ${p['target_equity']:,.2f} | "
            f"{_bar(p['progress_pct'])} {p['progress_pct']:.0f}% | "
            f"DD {p['total_dd_pct']:.2f}%/{p['total_dd_limit']}%"
        )

    def report(self, current_equity: float) -> str:
        p = self.check(current_equity)
        if p["passed"]:
            tag = "PASSED"
        elif p["expired"]:
            tag = "FAILED - EXPIRED"
        elif p["best_day_violation"]:
            tag = "WARNING - BEST DAY RULE"
        else:
            tag = "ACTIVE"

        bar = _bar(p["progress_pct"], width=20)
        window_line = (
            f"Day {p['elapsed_days']} of {p['window_days']} ({p['remaining_days']} days left)"
            if p["window_days"] else
            f"Day {p['elapsed_days']} (no maximum window)"
        )
        days_line = (
            f"{p['trade_days_count']} of {p['min_days_required']} required"
            if p["min_days_required"] else "no minimum"
        )
        lines = [
            "╔══════════════════ FTMO CHALLENGE PROGRESS ══════════════════╗",
            f"  Challenge     : {self.challenge.upper()}",
            f"  Status        : {tag}",
            f"  Window        : {window_line}",
            f"  Trading days  : {days_line}",
            f"  Equity        : ${p['current_equity']:,.2f}  (peak ${p['peak_equity']:,.2f})",
            f"  Profit        : ${p['profit_made']:+,.2f}  ({p['profit_pct']:+.2f}% / {self.rules['profit_target']}% target)",
            f"  Still need    : ${p['profit_needed']:,.2f}",
            f"  Progress      : {bar} {p['progress_pct']:.1f}%",
            f"  Total DD      : {p['total_dd_pct']:.2f}%  (limit {p['total_dd_limit']}%)",
            f"  Daily DD lim  : {p['daily_dd_limit']}%",
        ]
        if self.challenge == "1step":
            lines.append(f"  Best Day Rule : {'VIOLATION' if p['best_day_violation'] else 'OK'}")
        lines.append("╚═════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_or_init(self, initial_equity: float) -> _FTMOState:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    return _FTMOState.from_dict(json.load(f))
            except Exception:
                pass
        state = _FTMOState.new(initial_equity, self.rules["profit_target"])
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        return state

    def _save(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)


def _bar(pct: float, width: int = 15) -> str:
    filled = int(width * min(pct, 100.0) / 100.0)
    return "[" + "#" * filled + "." * (width - filled) + "]"
