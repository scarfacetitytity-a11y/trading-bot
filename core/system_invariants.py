"""AiDEN system invariants — the immutable safety envelope.

PROTECTED FILE. Changes here must go through a human-reviewed pull request.
No agent, autoresearch loop, Telegram command or self-upgrade may modify this
file or loosen any value it defines (see docs/RISK_INVARIANTS.md).

Model
-----
Invariants are an *envelope*, not operating values. Operating values live in
config (config.yaml / config_vps.yaml) and may be equal to or TIGHTER than the
envelope. Anything looser is clamped to the envelope at startup and logged as
a violation — clamping, not refusing to start, because a refused start would
leave open positions unmanaged.

Why each number is what it is (full audit in docs/RISK_INVARIANTS.md):

  max_daily_loss_pct   4.0  FTMO daily limit is 5%. 4.0 keeps a 1% buffer and is
                            the strictest value every existing component already
                            honours (orchestrator RiskGuard, config_vps). OS docs
                            disagree (2.0 in live-trading-roles, 3.75 in
                            daily_limits) — tightening is a human decision.
  max_soft_dd_pct      9.0  OS daily_limits: at 90% of the 10% max DD trade
                            minimum size only. No new entries past this.
  max_total_kill_pct   9.5  Emergency flatten before FTMO's 10% breach.
  max_weekly_dd_pct    7.0  RiskAgent's documented weekly limit. The orchestrator
                            used to feed it soft_dd_halt_pct (9%), which silently
                            disabled the weekly check.
  max_risk_per_trade   1.0  OS position_sizing: "1–2% likely"; preferred 0.5%.
                            1.0 is the existing code default; config runs 0.75.
  max_portfolio_risk   4.0  Existing portfolio cap.
  max_concurrent       8    = max_portfolio_risk / min_trade_risk (4.0 / 0.5).
                            Non-binding for the current config; RiskAgent's 20
                            was no cap at all. Tightening to 3 (autoresearch
                            baseline CAP_CONCURRENT) is a pending human decision.
  max_daily_entries    6    Existing RiskAgent value (decision-fatigue cap).
  min_reward_risk      1.5  OS risk docs: "Min R:R 1.5:1 — hard floor".
  min_cutover_samples  50   orchestrator default for v3 go/no-go; config_vps had
                            0, which bypassed cutover validation entirely.
  min_lift_trades      30   Devil's Advocate minimum sample (learning_loop).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

CEILING = "ceiling"   # config value must be <= envelope
FLOOR   = "floor"     # config value must be >= envelope


@dataclass(frozen=True)
class SystemInvariants:
    max_daily_loss_pct: float = 4.0
    max_weekly_dd_pct: float = 7.0
    max_soft_dd_pct: float = 9.0
    max_total_kill_pct: float = 9.5
    max_risk_per_trade_pct: float = 1.0
    max_portfolio_risk_pct: float = 4.0
    max_concurrent_trades: int = 8
    max_daily_entries: int = 6
    max_consecutive_losses_before_pause: int = 7
    min_reward_risk: float = 1.5
    min_cutover_samples: int = 50
    min_lift_proposal_trades: int = 30
    require_hard_stop: bool = True
    require_human_approval_live: bool = True


INVARIANTS = SystemInvariants()

# trading.<key> in config  ->  (invariant field, direction)
TRADE_CONFIG_BOUNDS: dict[str, tuple[str, str]] = {
    "daily_halt_pct":         ("max_daily_loss_pct", CEILING),
    "weekly_dd_halt_pct":     ("max_weekly_dd_pct", CEILING),
    "soft_dd_halt_pct":       ("max_soft_dd_pct", CEILING),
    "total_kill_pct":         ("max_total_kill_pct", CEILING),
    "risk_pct":               ("max_risk_per_trade_pct", CEILING),
    "max_portfolio_risk_pct": ("max_portfolio_risk_pct", CEILING),
    "max_concurrent_trades":  ("max_concurrent_trades", CEILING),
    "max_daily_entries":      ("max_daily_entries", CEILING),
    "min_rr":                 ("min_reward_risk", FLOOR),
}

# Every parameter whose *direction of change* matters for safety, across
# live config, RiskConfig and research/params.py. "lower" = lower is safer.
RISK_PARAM_SAFE_DIRECTION: dict[str, str] = {
    "risk_pct": "lower", "BASE_RISK": "lower",
    "daily_halt_pct": "lower", "DAILY_DD_LIMIT": "lower", "max_daily_loss_pct": "lower",
    "weekly_dd_halt_pct": "lower", "max_weekly_dd_pct": "lower",
    "soft_dd_halt_pct": "lower", "total_kill_pct": "lower", "max_account_dd_pct": "lower",
    "max_portfolio_risk_pct": "lower",
    "max_concurrent_trades": "lower", "CAP_CONCURRENT": "lower",
    "max_daily_entries": "lower",
    "max_consecutive_losses": "lower",
    "min_rr": "higher", "MIN_RR": "higher",
    "go_no_go_min_samples": "higher",
    "require_approval": "higher",            # False -> True is safer
}

# Research/RiskConfig names that map onto an envelope field.
_PARAM_TO_ENVELOPE: dict[str, tuple[str, str]] = {
    **TRADE_CONFIG_BOUNDS,
    "BASE_RISK":              ("max_risk_per_trade_pct", CEILING),
    "DAILY_DD_LIMIT":         ("max_daily_loss_pct", CEILING),
    "CAP_CONCURRENT":         ("max_concurrent_trades", CEILING),
    "MIN_RR":                 ("min_reward_risk", FLOOR),
    "max_consecutive_losses": ("max_consecutive_losses_before_pause", CEILING),
    "go_no_go_min_samples":   ("min_cutover_samples", FLOOR),
}


@dataclass
class Violation:
    key: str
    value: Any
    limit: Any
    rule: str

    def __str__(self) -> str:
        return f"{self.key}={self.value!r} violates {self.rule} (limit {self.limit!r})"


# ── Config envelope ──────────────────────────────────────────────────────────

def _breaches(value: float, limit: float, direction: str) -> bool:
    return value > limit if direction == CEILING else value < limit


def validate_trade_config(trade_cfg: dict, inv: SystemInvariants = INVARIANTS) -> list[Violation]:
    out: list[Violation] = []
    for key, (field_name, direction) in TRADE_CONFIG_BOUNDS.items():
        if key not in trade_cfg or trade_cfg[key] is None:
            continue
        try:
            value = float(trade_cfg[key])
        except (TypeError, ValueError):
            out.append(Violation(key, trade_cfg[key], "number", "type"))
            continue
        limit = getattr(inv, field_name)
        if _breaches(value, limit, direction):
            out.append(Violation(key, value, limit, f"{direction} {field_name}"))
    soft = trade_cfg.get("soft_dd_halt_pct")
    kill = trade_cfg.get("total_kill_pct")
    if soft is not None and kill is not None and float(soft) >= float(kill):
        out.append(Violation("soft_dd_halt_pct", soft, kill, "soft halt must precede kill"))
    return out


def enforce_trade_config(trade_cfg: dict, inv: SystemInvariants = INVARIANTS) -> tuple[dict, list[Violation]]:
    """Return (clamped copy, violations). Never loosens; only clamps to envelope."""
    violations = validate_trade_config(trade_cfg, inv)
    cfg = dict(trade_cfg)
    for v in violations:
        if v.key in TRADE_CONFIG_BOUNDS and v.rule != "type":
            field_name, _ = TRADE_CONFIG_BOUNDS[v.key]
            limit = getattr(inv, field_name)
            cfg[v.key] = int(limit) if isinstance(limit, int) else float(limit)
    soft, kill = cfg.get("soft_dd_halt_pct"), cfg.get("total_kill_pct")
    if soft is not None and kill is not None and float(soft) >= float(kill):
        cfg["soft_dd_halt_pct"] = round(float(kill) - 0.5, 2)
    return cfg, violations


def clamp_risk_pct(pct: float, inv: SystemInvariants = INVARIANTS) -> float:
    return max(0.0, min(float(pct), inv.max_risk_per_trade_pct))


# ── Human approval ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ApprovalPolicy:
    required: bool
    fail_closed: bool          # no Telegram / timeout / error => VETO
    allow_auto_approve: bool   # score-based approval on timeout
    mode: str


def execution_mode(trade_cfg: dict) -> str:
    """'live' (default) or 'demo_autonomous'. Unknown values are treated as live."""
    mode = str(trade_cfg.get("execution_mode", "live")).strip().lower()
    return mode if mode in ("live", "demo_autonomous") else "live"


def approval_policy(trade_cfg: dict, is_real_account: bool,
                    inv: SystemInvariants = INVARIANTS) -> ApprovalPolicy:
    """Human approval is mandatory, fail-closed, for anything that is live.

    'demo_autonomous' must be chosen explicitly in config and is ignored on a
    real account. FTMO challenge/funded accounts report as MT5 *demo* accounts,
    so account type alone can't tell us whether capital is at stake — hence the
    default is 'live'.
    """
    mode = execution_mode(trade_cfg)
    if is_real_account or mode == "live":
        return ApprovalPolicy(required=inv.require_human_approval_live,
                              fail_closed=True, allow_auto_approve=False,
                              mode="live")
    return ApprovalPolicy(required=bool(trade_cfg.get("require_approval", False)),
                          fail_closed=False, allow_auto_approve=True,
                          mode="demo_autonomous")


# ── Emergency halt ───────────────────────────────────────────────────────────
# Unlike council_halt.flag (auto-cleared each new day), the emergency halt is
# persistent: only a human clears it, via `python -m execution.upgrade_gate --clear-halt`.

EMERGENCY_HALT_FILE = "emergency_halt.flag"
EMERGENCY_HALT_ENV  = "AIDEN_EMERGENCY_HALT"


def _logs_dir(logs_dir: Optional[Path]) -> Path:
    if logs_dir is not None:
        return Path(logs_dir)
    from core.paths import logs_dir as _ld
    return _ld()


def is_emergency_halted(logs_dir: Optional[Path] = None) -> tuple[bool, str]:
    if os.environ.get(EMERGENCY_HALT_ENV, "").strip().lower() in ("1", "true", "yes"):
        return True, f"{EMERGENCY_HALT_ENV} set in environment"
    f = _logs_dir(logs_dir) / EMERGENCY_HALT_FILE
    if f.exists():
        try:
            return True, json.loads(f.read_text(encoding="utf-8")).get("reason", "emergency halt")
        except Exception:
            return True, "emergency halt (unreadable flag — treated as active)"
    return False, ""


def set_emergency_halt(reason: str, logs_dir: Optional[Path] = None) -> Path:
    d = _logs_dir(logs_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / EMERGENCY_HALT_FILE
    f.write_text(json.dumps({"reason": reason,
                             "set_at": datetime.now(timezone.utc).isoformat()}),
                 encoding="utf-8")
    return f


def clear_emergency_halt(logs_dir: Optional[Path] = None) -> bool:
    f = _logs_dir(logs_dir) / EMERGENCY_HALT_FILE
    if f.exists():
        f.unlink()
        return True
    return False


# ── Pre-order check (last gate before order_send) ────────────────────────────

@dataclass
class OrderCheck:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def check_pre_order(direction: int, entry: float, sl: Optional[float], tp: Optional[float],
                    risk_pct: float, logs_dir: Optional[Path] = None,
                    inv: SystemInvariants = INVARIANTS) -> OrderCheck:
    reasons: list[str] = []
    halted, why = is_emergency_halted(logs_dir)
    if halted:
        reasons.append(f"EMERGENCY_HALT: {why}")
    if inv.require_hard_stop:
        if not sl:
            reasons.append("NO_HARD_STOP: every order needs a stop at placement")
        elif (direction == 1 and sl >= entry) or (direction == -1 and sl <= entry):
            reasons.append(f"STOP_WRONG_SIDE: sl={sl} entry={entry} dir={direction}")
    if sl and tp and entry and abs(entry - sl) > 0:
        rr = abs(tp - entry) / abs(entry - sl)
        if rr + 1e-9 < inv.min_reward_risk:
            reasons.append(f"RR_BELOW_FLOOR: {rr:.2f} < {inv.min_reward_risk}")
    if risk_pct > inv.max_risk_per_trade_pct + 1e-9:
        reasons.append(f"RISK_ABOVE_CEILING: {risk_pct}% > {inv.max_risk_per_trade_pct}%")
    return OrderCheck(ok=not reasons, reasons=reasons)


# ── Change classification (used by the self-upgrade lifecycle) ──────────────

def classify_param_change(name: str, old: Any, new: Any) -> str:
    """'tightens' | 'loosens' | 'neutral' for safety-relevant params."""
    direction = RISK_PARAM_SAFE_DIRECTION.get(name)
    if direction is None or old == new:
        return "neutral"
    try:
        o, n = float(old), float(new)
    except (TypeError, ValueError):
        return "loosens"   # unparseable change to a risk param: assume unsafe
    if direction == "lower":
        return "tightens" if n < o else "loosens"
    return "tightens" if n > o else "loosens"


def risk_change_violations(changes: dict[str, tuple[Any, Any]],
                           inv: SystemInvariants = INVARIANTS) -> list[Violation]:
    """Violations for an automatic change set. Any loosening of a risk param is
    a violation, even inside the envelope: risk limits may only be loosened by a
    human outside the automatic upgrade path."""
    out: list[Violation] = []
    for name, (old, new) in changes.items():
        if classify_param_change(name, old, new) == "loosens":
            out.append(Violation(name, new, old, "automatic loosening of a risk limit"))
        if name in _PARAM_TO_ENVELOPE:
            field_name, direction = _PARAM_TO_ENVELOPE[name]
            try:
                if _breaches(float(new), getattr(inv, field_name), direction):
                    out.append(Violation(name, new, getattr(inv, field_name),
                                         f"{direction} {field_name}"))
            except (TypeError, ValueError):
                pass
    return out


def fingerprint() -> str:
    """SHA-256 of this file — recorded with every promotion so a change to the
    envelope is visible in the upgrade ledger."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
