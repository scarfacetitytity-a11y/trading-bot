"""Council Watch — always-on enforcement daemon.

Runs beside the trading bot as a separate process. Enforces hard limits that
survive bot restarts and has no dependency on the orchestrator being alive.

Roles enforced (Council of 12 mapping):
  - #05 Compliance Officer : FTMO daily DD, challenge-day count, no-trade windows
  - #07 SRE               : state file staleness, heartbeat monitoring
  - #03 Data Engineer     : baseline integrity, equity drift detection
  - #06 App Engineer      : open trade audit, zombie detection

Cadre event bridge:
  Writes structured events to logs/council_events.jsonl. On key events,
  creates firm.db unit tasks so Cadre agents (Sage/Quant/Builder/Scout)
  pick them up at next Claude Code session.

All significant events are routed to the AiDEN Obsidian vault via
council_obsidian.py.

Usage:
    python -m execution.council_watch          # production
    python -m execution.council_watch --once   # single-pass health check
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ── Paths ────────────────────────────────────────────────────────────────────
_BOT_ROOT   = Path(__file__).resolve().parent.parent
_LOGS       = _BOT_ROOT / "logs"
_FIRM_DB    = Path(r"C:\Users\anton\.firm\firm.db")
_EVENTS_LOG = _LOGS / "council_events.jsonl"

# State files written by the bot
_RG_STATE    = _LOGS / "risk_guard_state.json"
_RA_STATE    = _LOGS / "risk_agent_state.json"
_FTMO_STATE  = _LOGS / "ftmo_tracker_state.json"
_OPEN_TRADES = _LOGS / "open_trades.json"
_HALT_FILE   = _LOGS / "council_halt.flag"  # council writes, orchestrator reads

# ── Thresholds ───────────────────────────────────────────────────────────────
DAILY_DD_WARN_PCT    = 3.5   # warn before RiskGuard fires at 2% (may lag on restart)
DAILY_DD_HARD_PCT    = 4.5   # hard halt — 0.5% buffer before FTMO's 5%
TOTAL_DD_WARN_PCT    = 7.0   # warn
TOTAL_DD_HARD_PCT    = 9.0   # hard halt — 1% buffer before FTMO's 10%
STATE_STALE_SECS     = 300   # 5 min — RiskGuard beats every 60s; 5 min = bot likely dead
ZOMBIE_TRADE_HOURS   = 24    # open trade older than this = zombie, flag for review
POLL_INTERVAL        = 30    # seconds between checks

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOGS / "council_watch.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("council_watch")

# ── Deferred Obsidian import ─────────────────────────────────────────────────
try:
    from execution.council_obsidian import write_brain_note, write_alert, update_session_note
    _OBSIDIAN_OK = True
except ImportError:
    _OBSIDIAN_OK = False
    logger.warning("council_obsidian not importable — Obsidian routing disabled")

# ── Telegram import ──────────────────────────────────────────────────────────
try:
    from execution import telegram_notify as tg
    _TG_OK = True
except ImportError:
    _TG_OK = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _today_str() -> str:
    return _utcnow().strftime("%Y-%m-%d")


# ── Event log ────────────────────────────────────────────────────────────────

def _log_event(event_type: str, data: dict[str, Any]) -> None:
    _LOGS.mkdir(parents=True, exist_ok=True)
    record = {"ts": _utcnow().isoformat(), "type": event_type, **data}
    with _EVENTS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Halt file ────────────────────────────────────────────────────────────────

def _write_halt(reason: str) -> None:
    """Write council_halt.flag. Orchestrator checks this file before new entries."""
    _HALT_FILE.write_text(json.dumps({
        "halted_at": _utcnow().isoformat(),
        "reason": reason,
        "day": _today_str(),
    }), encoding="utf-8")
    logger.critical("[COUNCIL HALT] %s — halt file written", reason)


def _clear_halt_if_new_day() -> None:
    if not _HALT_FILE.exists():
        return
    try:
        d = json.loads(_HALT_FILE.read_text(encoding="utf-8"))
        if d.get("day") != _today_str():
            _HALT_FILE.unlink()
            logger.info("[Council] New day — halt file cleared")
    except Exception:
        pass


# ── Firm DB — Cadre event bridge ─────────────────────────────────────────────

def _cadre_task(assignee_id: str, name: str, description: str, project_id: str = "PROJ-001") -> None:
    """Write a unit task to firm.db for a Cadre agent to pick up."""
    if not _FIRM_DB.exists():
        return
    try:
        import uuid
        unit_id = f"UNIT-CW-{uuid.uuid4().hex[:6].upper()}"
        now     = _utcnow().isoformat()
        with sqlite3.connect(str(_FIRM_DB)) as conn:
            conn.execute("""
                INSERT INTO unit
                  (id, firm_id, project_id, name, description, assignee_member_id,
                   status, priority, rank, created_at, updated_at)
                VALUES (?, 'aiden', ?, ?, ?, ?, 'pending', 'high', 0, ?, ?)
            """, (unit_id, project_id, name, description, assignee_id, now, now))
        logger.info("[Council] Cadre task queued for %s: %s", assignee_id, name)
    except Exception as exc:
        logger.warning("[Council] firm.db write failed: %s", exc)


# ── Checks ────────────────────────────────────────────────────────────────────

class _AlertTracker:
    """Prevents the same alert firing every 30 seconds."""
    def __init__(self):
        self._fired: dict[str, str] = {}  # key → day fired

    def should_fire(self, key: str) -> bool:
        today = _today_str()
        if self._fired.get(key) != today:
            self._fired[key] = today
            return True
        return False

    def reset(self, key: str) -> None:
        self._fired.pop(key, None)


_alerts = _AlertTracker()


def _check_risk_guard() -> dict:
    issues = {}
    if not _RG_STATE.exists():
        issues["rg_missing"] = "risk_guard_state.json missing"
        return issues

    try:
        age_secs = (_utcnow() - datetime.fromtimestamp(_RG_STATE.stat().st_mtime, timezone.utc)).total_seconds()
        if age_secs > STATE_STALE_SECS:
            issues["rg_stale"] = f"risk_guard_state.json not updated for {age_secs:.0f}s — bot may be down"
    except Exception:
        pass

    try:
        d = json.loads(_RG_STATE.read_text(encoding="utf-8"))
        day_start  = d.get("day_start_equity") or 0.0
        server_day = d.get("server_day", "")
    except Exception:
        return issues

    # Cross-validate against ftmo_tracker
    if _FTMO_STATE.exists():
        try:
            ftmo = json.loads(_FTMO_STATE.read_text(encoding="utf-8"))
            ftmo_start = ftmo.get("day_start_equity") or 0.0
            if ftmo_start > 0 and day_start > 0:
                drift_pct = abs(day_start - ftmo_start) / ftmo_start * 100
                if drift_pct > 1.0:
                    issues["baseline_drift"] = (
                        f"RiskGuard baseline {day_start:.2f} vs FTMOTracker {ftmo_start:.2f} "
                        f"({drift_pct:.1f}% drift) — possible restart re-anchor"
                    )
        except Exception:
            pass

    return issues


def _check_equity_limits() -> dict:
    issues = {}
    if not _FTMO_STATE.exists():
        return issues

    try:
        d        = json.loads(_FTMO_STATE.read_text(encoding="utf-8"))
        initial  = float(d.get("initial_equity") or 100000)
        last_eq  = float(d.get("last_equity") or initial)
        day_start = float(d.get("day_start_equity") or initial)
        today    = _today_str()

        if d.get("day_start_date") == today and day_start > 0:
            daily_pct = (last_eq - day_start) / day_start * 100
        else:
            daily_pct = 0.0

        total_pct = (last_eq - initial) / initial * 100

        if daily_pct <= -DAILY_DD_HARD_PCT:
            issues["daily_hard"] = f"DAILY DD BREACH: {daily_pct:.2f}% (limit {DAILY_DD_HARD_PCT}%)"
        elif daily_pct <= -DAILY_DD_WARN_PCT:
            issues["daily_warn"] = f"Daily DD warning: {daily_pct:.2f}%"

        if total_pct <= -TOTAL_DD_HARD_PCT:
            issues["total_hard"] = f"TOTAL DD BREACH: {total_pct:.2f}% (limit {TOTAL_DD_HARD_PCT}%)"
        elif total_pct <= -TOTAL_DD_WARN_PCT:
            issues["total_warn"] = f"Total DD warning: {total_pct:.2f}%"

    except Exception as exc:
        logger.debug("equity check error: %s", exc)

    return issues


def _check_open_trades() -> dict:
    issues = {}
    if not _OPEN_TRADES.exists():
        return issues

    try:
        trades = json.loads(_OPEN_TRADES.read_text(encoding="utf-8"))
        if not isinstance(trades, dict):
            return issues

        now = _utcnow()
        zombies = []
        for ticket, trade in trades.items():
            opened_raw = trade.get("opened_at") or trade.get("open_time")
            if not opened_raw:
                continue
            try:
                opened = datetime.fromisoformat(str(opened_raw).replace("Z", "+00:00"))
                if (now - opened).total_seconds() > ZOMBIE_TRADE_HOURS * 3600:
                    zombies.append(f"{ticket} ({trade.get('symbol','?')} open {opened_raw})")
            except Exception:
                pass

        if zombies:
            issues["zombies"] = f"Zombie trades (>{ZOMBIE_TRADE_HOURS}h): {', '.join(zombies)}"

    except Exception as exc:
        logger.debug("open trades check error: %s", exc)

    return issues


def _check_ftmo_challenge() -> dict:
    issues = {}
    if not _FTMO_STATE.exists():
        return issues

    try:
        d          = json.loads(_FTMO_STATE.read_text(encoding="utf-8"))
        initial    = float(d.get("initial_equity") or 100000)
        target     = float(d.get("target_equity") or 110000)
        last_eq    = float(d.get("last_equity") or initial)
        trade_days = d.get("trade_days") or []

        # Profit target proximity
        progress_pct = (last_eq - initial) / (target - initial) * 100 if target > initial else 0
        if progress_pct >= 80:
            issues["target_close"] = f"Challenge {progress_pct:.0f}% to profit target ({last_eq:.0f} / {target:.0f})"

        # Minimum trading days (FTMO requires at least 4 profitable trading days)
        # Flag if approaching end without enough days
        start_raw = d.get("start_date")
        if start_raw:
            start    = datetime.strptime(start_raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            elapsed  = (_utcnow() - start).days
            if elapsed >= 25 and len(trade_days) < 4:
                issues["min_days"] = f"Only {len(trade_days)} trade days with {elapsed} elapsed — FTMO min-day rule at risk"

    except Exception as exc:
        logger.debug("ftmo check error: %s", exc)

    return issues


# ── Alert dispatch ────────────────────────────────────────────────────────────

def _dispatch(key: str, message: str, severity: str = "warning") -> None:
    if not _alerts.should_fire(key):
        return

    logger.log(
        logging.CRITICAL if severity == "critical" else logging.WARNING,
        "[Council/%s] %s", severity.upper(), message,
    )
    _log_event("council_alert", {"key": key, "severity": severity, "message": message})

    if _TG_OK:
        try:
            tg.send(f"[Council/{severity.upper()}] {message}")
        except Exception:
            pass

    if _OBSIDIAN_OK:
        try:
            write_alert(
                alert_id=key,
                title=f"Council Alert — {key}",
                severity=severity,
                body=message,
                tags=["alert", "aiden", "council", severity],
            )
            update_session_note(f"Council/{key}", f"{severity.upper()}: {message[:120]}")
        except Exception as exc:
            logger.debug("Obsidian write failed: %s", exc)


def _handle_critical(key: str, message: str) -> None:
    _dispatch(key, message, severity="critical")
    _write_halt(message)

    # Queue Quant for post-incident review (firm.db — picked up at next session)
    _cadre_task(
        assignee_id="MEM-002",  # Quant
        name=f"Post-incident review — {key}",
        description=f"Council Watch flagged critical: {message}. Review ftmo_tracker_state.json and orchestrator.log. Identify root cause and recommend rule changes.",
    )

    # Also invoke Quant asynchronously now via cadre_invoke
    try:
        import threading
        from execution import cadre_invoke
        t = threading.Thread(
            target=cadre_invoke.invoke,
            args=("MEM-002", "post_incident_review", message),
            daemon=True,
        )
        t.start()
    except Exception as exc:
        logger.debug("Cadre async invoke failed: %s", exc)


# ── Cadre schedule state ──────────────────────────────────────────────────────
# Tracks last-run timestamp and trade count so interval-based tasks don't fire
# every 30 seconds. Persisted to disk so it survives council_watch restarts.

_CADRE_SCHED_STATE = _LOGS / "cadre_schedule_state.json"

# Intervals (seconds)
_SCOUT_INTERVAL   = 30 * 60    # 30 min
_QUANT_INTERVAL   = 2 * 60 * 60  # 2 hours (routine)
_BUILDER_INTERVAL = 60 * 60    # 1 hour (error scan)


def _sched_load() -> dict:
    try:
        if _CADRE_SCHED_STATE.exists():
            return json.loads(_CADRE_SCHED_STATE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _sched_save(state: dict) -> None:
    try:
        _LOGS.mkdir(parents=True, exist_ok=True)
        _CADRE_SCHED_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("Cadre sched state save failed: %s", exc)


def _sched_due(state: dict, key: str, interval_secs: float) -> bool:
    last = state.get(key, 0)
    return (time.time() - last) >= interval_secs


def _sched_mark(state: dict, key: str) -> None:
    state[key] = time.time()


def _fire_cadre(member_id: str, event: str, extra: str = "") -> None:
    """Launch a Cadre invocation on a daemon thread."""
    import threading
    from execution import cadre_invoke
    threading.Thread(
        target=cadre_invoke.invoke,
        args=(member_id, event, extra),
        daemon=True,
    ).start()


# ── Context builders for scheduled tasks ─────────────────────────────────────

def _ctx_scout() -> str:
    """Pull session + FTMO state for Scout's regime research task."""
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    parts = []
    if _FTMO_STATE.exists():
        try:
            ftmo = json.loads(_FTMO_STATE.read_text(encoding="utf-8"))
            parts.append(f"FTMO state: day_pnl={ftmo.get('daily_pnl_pct', 0):.2f}% "
                         f"total_dd={ftmo.get('total_dd_pct', 0):.2f}% "
                         f"day={ftmo.get('challenge_day', '?')}/30")
        except Exception:
            pass
    parts.append(
        f"Task: Research the current market regime. Assess DXY direction (H4 EMA trend: +1 bullish "
        f"USD, -1 bearish USD, 0 neutral), active session (Asian/London/NY), and any macro risk "
        f"events in the next 4 hours (NFP, FOMC, CPI etc.). "
        f"Score overall regime: trending_bull / trending_bear / ranging / high_vol.\n\n"
        f"REQUIRED OUTPUT (two parts, both mandatory):\n"
        f"1. Write a JSON file to EXACTLY this path: "
        f"C:\\Users\\anton\\Documents\\trading-bot\\logs\\cadre_regime_state.json\n"
        f"   Format: {{\"regime\": \"ranging\", \"dxy_bias\": 0, \"session\": \"London\", "
        f"\"risk_events\": [], \"narrative\": \"one sentence\", \"ts_utc\": \"ISO timestamp\"}}\n"
        f"   dxy_bias must be +1, -1, or 0. This file is machine-read by the trading bot.\n\n"
        f"2. Write a markdown note to Brain vault: "
        f"C:\\Users\\anton\\OneDrive\\Desktop\\Aiden\\AiDEN\\Brain\\Intelligence\\regime_{now_str}.md\n"
        f"   Use MOP frontmatter (type: note, status: active, tags: [regime, scout])."
    )
    return "\n\n".join(parts)


def _ctx_quant_routine() -> str:
    """Pull trades.jsonl summary for Quant's periodic performance check."""
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    parts = []
    trades_file = _LOGS / "trades.jsonl"
    if trades_file.exists():
        try:
            lines = trades_file.read_text(encoding="utf-8").strip().splitlines()
            recent = lines[-20:]
            parts.append(f"Last 20 trades (of {len(lines)} total):\n" + "\n".join(recent))
        except Exception:
            pass
    prob_file = _LOGS / "prob_model_state.json"
    if prob_file.exists():
        try:
            parts.append(f"Current Bayesian lift table:\n{prob_file.read_text(encoding='utf-8')}")
        except Exception:
            pass
    parts.append(
        f"Task: Analyse the recent trades. For each confluence key (fvg_present, ob_present, "
        f"m5_confirmed, h4_aligned, at_htf_level, order_flow_aligned, continuation_type): "
        f"compute observed win rate from the trade data above, compare to the current lift in "
        f"prob_model_state.json, and recommend an adjusted lift where the data justifies it.\n\n"
        f"REQUIRED OUTPUT (two parts, both mandatory):\n"
        f"1. Write a JSON file to EXACTLY this path: "
        f"C:\\Users\\anton\\Documents\\trading-bot\\logs\\quant_lift_proposals.json\n"
        f"   Format: {{\"confluences\": {{\"fvg_present\": 1.28, \"ob_present\": 1.20, ...}}, "
        f"\"ts\": \"ISO timestamp\", \"n_trades_analysed\": N}}\n"
        f"   Only include keys where observed WR meaningfully differs from current lift. "
        f"This file is machine-read by the trading bot.\n\n"
        f"2. Write a markdown analysis to Brain vault: "
        f"C:\\Users\\anton\\OneDrive\\Desktop\\Aiden\\AiDEN\\Brain\\Quant\\performance_{now_str}.md\n"
        f"   Include: WR table per confluence, lift recommendations, risk findings."
    )
    return "\n\n".join(parts)


def _ctx_quant_post_trade(trade_count: int) -> str:
    """Quant context when a new trade just closed."""
    trades_file = _LOGS / "trades.jsonl"
    last_trade  = ""
    if trades_file.exists():
        try:
            lines = trades_file.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                last_trade = lines[-1]
        except Exception:
            pass
    return (
        f"A new trade just closed (total trades: {trade_count}).\n"
        f"Last trade record: {last_trade}\n\n"
        "Task: Analyse this trade specifically. Was the entry thesis validated? "
        "Did the Bayesian model's P(win) estimate match outcome? Which confluences "
        "were present and which fired correctly? Write a brief post-trade note to "
        "Brain vault as Quant/post_trade_YYYYMMDD_HHMM.md. If the lift table needs "
        "updating, state the recommended changes."
    )


def _ctx_builder_error_scan() -> str:
    """Builder context for routine error log scan."""
    log_file = _LOGS / "orchestrator.log"
    errors   = []
    if log_file.exists():
        try:
            lines = log_file.read_text(encoding="utf-8").splitlines()
            errors = [l for l in lines if "ERROR" in l or "CRITICAL" in l or "Exception" in l][-20:]
        except Exception:
            pass
    if not errors:
        return ""
    return (
        f"Recent errors from orchestrator.log:\n" + "\n".join(errors) + "\n\n"
        "Task: Review these errors. For each: identify the root cause, determine if "
        "there is a code fix needed, and if yes — apply the minimal targeted fix to "
        "C:\\Users\\anton\\Documents\\trading-bot\\. Log what changed to Brain vault "
        "as Builder/fix_YYYYMMDD.md."
    )


# ── Main loop ─────────────────────────────────────────────────────────────────

def _check_cadre_scheduled() -> None:
    """Trigger time-based Cadre reviews on rotating schedules."""
    try:
        state   = _sched_load()
        now     = _utcnow()
        ra      = {}
        if _RA_STATE.exists():
            try:
                ra = json.loads(_RA_STATE.read_text(encoding="utf-8"))
            except Exception:
                pass

        # ── Scout: market regime every 30 minutes ─────────────────────────────
        if _sched_due(state, "scout_regime", _SCOUT_INTERVAL):
            logger.info("[Council] Scout — market regime research (30-min tick)")
            _fire_cadre("MEM-004", "market_regime_research", _ctx_scout())
            _sched_mark(state, "scout_regime")

        # ── Quant: routine performance review every 2 hours ───────────────────
        if _sched_due(state, "quant_routine", _QUANT_INTERVAL):
            logger.info("[Council] Quant — routine performance review (2-hour tick)")
            # Log current model confidence before invoking Quant
            try:
                from execution.probability_model import ProbabilityModel
                _pm = ProbabilityModel()
                logger.info("[Council] %s", _pm.confidence_report())
            except Exception:
                pass
            _fire_cadre("MEM-002", "daily_loss_review", _ctx_quant_routine())
            _sched_mark(state, "quant_routine")

        # ── Quant: post-trade analysis when trade count increases ─────────────
        current_trade_count = 0
        trades_file = _LOGS / "trades.jsonl"
        if trades_file.exists():
            try:
                current_trade_count = trades_file.read_text(encoding="utf-8").count("\n")
            except Exception:
                pass
        last_trade_count = int(state.get("last_trade_count", 0))
        if current_trade_count > last_trade_count:
            logger.info("[Council] Quant — new trade closed (%d total), post-trade analysis",
                        current_trade_count)
            _fire_cadre("MEM-002", "post_incident_review",
                        _ctx_quant_post_trade(current_trade_count))
            state["last_trade_count"] = current_trade_count

        # ── Quant: loss streak ≥ 5 ────────────────────────────────────────────
        consec = ra.get("consecutive_losses", 0)
        if consec >= 5 and _alerts.should_fire("quant_loss_streak"):
            logger.warning("[Council] Loss streak %d — invoking Quant emergency review", consec)
            _fire_cadre("MEM-002", "loss_streak_5",
                        f"Consecutive losses: {consec}\n\n{_ctx_quant_routine()}")

        # ── Builder: error log scan every hour ───────────────────────────────
        if _sched_due(state, "builder_error_scan", _BUILDER_INTERVAL):
            ctx = _ctx_builder_error_scan()
            if ctx:
                logger.info("[Council] Builder — new errors detected, invoking fix scan")
                _fire_cadre("MEM-003", "bot_error_detected", ctx)
            _sched_mark(state, "builder_error_scan")

        # ── Sage: Monday 07:00 UTC weekly strategy review ─────────────────────
        if now.weekday() == 0 and now.hour == 7 and _alerts.should_fire("sage_weekly"):
            logger.info("[Council] Sage — Monday 07:00 UTC weekly strategy review")
            _fire_cadre("MEM-001", "weekly_strategy_review",
                        f"Week starting {now.strftime('%Y-%m-%d')}. "
                        f"FTMO day: {ra.get('challenge_day', '?')}/30.")

        _sched_save(state)

    except Exception as exc:
        logger.debug("Cadre scheduled check error: %s", exc)


def run_once() -> dict[str, str]:
    _clear_halt_if_new_day()

    all_issues: dict[str, str] = {}
    all_issues.update(_check_risk_guard())
    all_issues.update(_check_equity_limits())
    all_issues.update(_check_open_trades())
    all_issues.update(_check_ftmo_challenge())
    _check_cadre_scheduled()

    for key, message in all_issues.items():
        if "hard" in key or "breach" in key.lower():
            _handle_critical(key, message)
        else:
            _dispatch(key, message, severity="warning")

    # Clear stale alerts for issues that resolved
    known_keys = {"rg_missing", "rg_stale", "baseline_drift", "daily_warn", "daily_hard",
                  "total_warn", "total_hard", "zombies", "target_close", "min_days"}
    for k in known_keys - set(all_issues.keys()):
        _alerts.reset(k)

    if not all_issues:
        logger.info("[Council] All checks HEALTHY")

    return all_issues


def run_loop() -> None:
    logger.info("[Council Watch] Starting — poll every %ds", POLL_INTERVAL)
    _log_event("daemon_start", {"poll_interval": POLL_INTERVAL})

    if _OBSIDIAN_OK:
        try:
            update_session_note("CouncilWatch", "STARTED")
        except Exception:
            pass

    while True:
        try:
            run_once()
        except KeyboardInterrupt:
            logger.info("[Council Watch] Interrupted — stopping")
            _log_event("daemon_stop", {"reason": "keyboard_interrupt"})
            break
        except Exception as exc:
            logger.error("[Council Watch] Unhandled error: %s", exc)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AiDEN Council Watch daemon")
    parser.add_argument("--once", action="store_true", help="Run a single health check and exit")
    args = parser.parse_args()

    if args.once:
        issues = run_once()
        sys.exit(1 if any("hard" in k for k in issues) else 0)
    else:
        run_loop()
