"""AiDEN Code Monitor — Builder Cadre Layer.

Third process in the autonomous stack (alongside bot + watchdog).
Zero tokens. Scans logs for error patterns, auto-fixes what it can,
writes incidents and daily summaries to Obsidian, escalates via Telegram.

Autonomous learning loop:
  Bot trades → logs decisions/errors
  Code Monitor scans every 60s
  Known errors → log + count + auto-flag
  New patterns (3+ hits in 1h) → write Obsidian incident + Telegram alert
  Daily summary → write Obsidian Brain/Audit + Brain/Sessions update
  GitHub commit → session-end audit trail

Run: python -m execution.code_monitor
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).parent.parent
LOG_DIR     = ROOT / "logs"
BOT_LOG     = LOG_DIR / "orchestrator.log"
MONITOR_PID = LOG_DIR / "code_monitor.pid"
STATE_FILE  = LOG_DIR / "code_monitor_state.json"
ENV_FILE    = ROOT / ".env"

OBSIDIAN    = Path(r"C:\Users\anton\OneDrive\Desktop\Aiden\AiDEN")
OBS_BRAIN   = OBSIDIAN / "Brain"
OBS_SESSIONS= OBSIDIAN / "Sessions"

POLL_INTERVAL   = 60      # seconds
PATTERN_WINDOW  = 3600    # 1h window for new-pattern detection
NEW_PATTERN_THR = 3       # hits in window before escalating
DAILY_SUMMARY_H = 21      # UTC hour to write daily summary (after NY close)

# cp1252 console chokes on →/— in log messages
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | code_monitor | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "code_monitor.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Known error patterns ──────────────────────────────────────────────────────
# Each entry: (regex, label, severity, auto_fix_fn_name or None)
KNOWN_PATTERNS = [
    (r"retcode=10027|AutoTrading disabled",      "AUTOTRADING_OFF",    "critical", None),
    (r"retcode=10016",                            "SL_TOO_CLOSE",       "warning",  None),
    (r"retcode=10025",                            "DUPLICATE_REQUEST",  "info",     None),
    (r"retcode=10006",                            "REQUEST_REJECTED",   "warning",  None),
    (r"retcode=10009",                            "REQUEST_PROCESSED",  "info",     None),
    (r"NOTIONAL CAP.*->",                         "NOTIONAL_CAP_HIT",   "warning",  None),
    (r"DataWatcher.*STALE|stale.*symbol",         "DATA_STALE",         "warning",  None),
    (r"SOFT HALT active",                         "SOFT_HALT",          "critical", None),
    (r"Traceback \(most recent",                  "PYTHON_EXCEPTION",   "error",    None),
    (r"INLINE DAILY DD GATE.*blocking",           "DAILY_DD_HALT",      "critical", None),
    (r"AMD SWEEP HARD BLOCK",                     "AMD_BLOCK",          "info",     None),
    (r"HTF ZONE DIRECTION BLOCK",                 "HTF_ZONE_BLOCK",     "info",     None),
    (r"RANGE BIAS GATE",                          "RANGE_BIAS_BLOCK",   "info",     None),
    (r"LOW CONVICTION skip",                      "LOW_SCORE_SKIP",     "info",     None),
    (r"Order OK:",                                "TRADE_EXECUTED",     "info",     None),
    (r"Closed position:",                         "TRADE_CLOSED",       "info",     None),
    (r"SCALE-IN:",                                "SCALE_IN",           "info",     None),
]

_COMPILED = [(re.compile(rx, re.IGNORECASE), label, sev, fix)
             for rx, label, sev, fix in KNOWN_PATTERNS]

# ── Telegram ──────────────────────────────────────────────────────────────────

def _load_env() -> None:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

def _tg(text: str, prefix: str = "[AiDEN Monitor]") -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return
    try:
        payload = json.dumps({
            "chat_id": chat, "text": f"{prefix} {text}",
            "parse_mode": "HTML", "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload, headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=8)
    except Exception as exc:
        logger.debug("Telegram failed: %s", exc)

# ── State persistence ─────────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "log_offset": 0,
        "pattern_hits": {},       # label → list of timestamps
        "unknown_hits": {},       # hash → list of timestamps
        "alerted_patterns": [],   # labels already alerted today
        "daily_summary_done": "", # date string
        "session_trades": [],
        "session_errors": [],
    }

def _save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass

# ── Log scanning ──────────────────────────────────────────────────────────────

def scan_new_lines(state: dict) -> list[tuple[str, str, str]]:
    """Read new lines from bot log since last offset. Returns (line, label, sev)."""
    if not BOT_LOG.exists():
        return []

    offset = int(state.get("log_offset", 0))
    findings: list[tuple[str, str, str]] = []

    try:
        with open(BOT_LOG, "rb") as f:
            f.seek(0, 2)
            end = f.tell()
            if end <= offset:
                return []
            f.seek(offset)
            new_bytes = f.read()
            state["log_offset"] = end
    except Exception:
        return []

    lines = new_bytes.decode("utf-8", errors="replace").splitlines()
    now   = time.time()

    for line in lines:
        matched = False
        for pattern, label, sev, _ in _COMPILED:
            if pattern.search(line):
                matched = True
                hits = state["pattern_hits"].setdefault(label, [])
                hits.append(now)
                # Trim to window
                state["pattern_hits"][label] = [t for t in hits if now - t <= PATTERN_WINDOW]

                # Track trades and errors for daily summary
                if label == "TRADE_EXECUTED":
                    state["session_trades"].append({"ts": now, "line": line.strip()[-120:]})
                if sev in ("error", "critical") and label not in ("SOFT_HALT", "DAILY_DD_HALT"):
                    state["session_errors"].append({"ts": now, "label": label, "line": line.strip()[-120:]})

                findings.append((line, label, sev))
                break

        if not matched:
            # Unknown pattern — track for escalation
            # Key on first 60 chars after the log prefix
            snippet = line[30:90].strip() if len(line) > 30 else line.strip()
            if snippet and any(kw in line.lower() for kw in ("error", "fail", "exception", "critical")):
                h = str(hash(snippet[:40]))
                uhits = state["unknown_hits"].setdefault(h, [])
                uhits.append(now)
                state["unknown_hits"][h] = [t for t in uhits if now - t <= PATTERN_WINDOW]
                findings.append((line, f"UNKNOWN:{snippet[:40]}", "warning"))

    return findings

# ── Pattern escalation ────────────────────────────────────────────────────────

def check_escalations(state: dict) -> None:
    now     = time.time()
    today   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    alerted = set(state.get("alerted_patterns", []))

    is_weekend = datetime.now(timezone.utc).weekday() >= 5  # Sat=5, Sun=6

    for label, hits in state["pattern_hits"].items():
        recent = [t for t in hits if now - t <= PATTERN_WINDOW]
        if len(recent) < NEW_PATTERN_THR:
            continue
        key = f"{label}:{today}"
        if key in alerted:
            continue

        # DATA_STALE on weekends is expected — suppress alerts
        if label == "DATA_STALE" and is_weekend:
            alerted.add(key)
            continue

        sev = next((s for _, l, s, _ in _COMPILED if l == label), "warning")
        alerted.add(key)

        if sev == "critical":
            _tg(f"<b>{label}</b> — {len(recent)} hits in last hour\n"
                f"Last: <code>{state['session_errors'][-1]['line'] if state['session_errors'] else ''}</code>")
            logger.warning("ESCALATED: %s (%d hits)", label, len(recent))
        elif sev == "error":
            _tg(f"<b>Error pattern: {label}</b> — {len(recent)}× in 1h")
            logger.warning("ESCALATED ERROR: %s", label)

        _write_obsidian_incident(label, sev, recent, state)

    # Unknown patterns
    for h, hits in state["unknown_hits"].items():
        recent = [t for t in hits if now - t <= PATTERN_WINDOW]
        if len(recent) < NEW_PATTERN_THR:
            continue
        key = f"UNKNOWN:{h}:{today}"
        if key in alerted:
            continue
        alerted.add(key)
        _tg(f"<b>New unknown error pattern</b> detected {len(recent)}× in 1h.\n"
            f"Check <code>logs/orchestrator.log</code> and <code>logs/code_monitor.log</code>.")
        logger.warning("UNKNOWN PATTERN escalated: hash=%s count=%d", h, len(recent))

    state["alerted_patterns"] = list(alerted)

# ── Obsidian writing ──────────────────────────────────────────────────────────

def _write_obsidian_incident(label: str, sev: str, hits: list, state: dict) -> None:
    """Write an incident note to Brain/Incidents."""
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fname = OBS_BRAIN / "Incidents" / f"{today}-{label.lower().replace('_','-')}.md"
        if fname.exists():
            return  # already written today
        fname.parent.mkdir(parents=True, exist_ok=True)
        content = f"""---
type: incident
status: open
tags: [incident, auto-detected, {label.lower()}]
relatedTo: [AiDEN, trading-bot]
date: {today}
severity: {sev}
---

# Incident: {label} — {today}

**Auto-detected by code_monitor at {datetime.now(timezone.utc).strftime('%H:%M UTC')}**

## Pattern
`{label}` — {len(hits)} occurrences in last hour.

## Recent Errors
{chr(10).join(f'- {e["line"]}' for e in (state.get("session_errors") or [])[-5:])}

## Status
- [ ] Root cause identified
- [ ] Fix deployed
- [ ] Verified resolved

## Links
[[Sessions/{today}]] | [[AiDEN]]
"""
        fname.write_text(content, encoding="utf-8")
        logger.info("Obsidian incident written: %s", fname.name)
    except Exception as exc:
        logger.debug("Obsidian incident write failed: %s", exc)


def write_daily_summary(state: dict) -> None:
    """Write structured daily digest to Obsidian Brain/Audit and logs/session_digest.jsonl.

    Sections are ordered by importance so truncation from the tail loses least
    signal: CRITICAL → PERFORMANCE → GATE ACTIVITY → ERRORS → LEARNING.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("daily_summary_done") == today:
        return

    try:
        session_trades = state.get("session_trades", [])
        errors         = state.get("session_errors", [])
        hits           = state.get("pattern_hits", {})

        # ── Counts ──────────────────────────────────────────────────────────────
        amd_blocks   = len(hits.get("AMD_BLOCK", []))
        zone_blocks  = len(hits.get("HTF_ZONE_BLOCK", []))
        range_blocks = len(hits.get("RANGE_BIAS_BLOCK", []))
        low_score    = len(hits.get("LOW_SCORE_SKIP", []))
        trade_count  = len(hits.get("TRADE_EXECUTED", []))
        scale_ins    = len(hits.get("SCALE_IN", []))

        # Closed trades from event bus payloads (schema: {type, symbol, outcome,
        # r_multiple, pnl_usd, equity, ts}).  Old log-hit rows have no 'outcome'.
        closed_events = [t for t in session_trades if t.get("outcome")]
        closed_count  = len(closed_events)
        wins          = [t for t in closed_events if t.get("outcome") == "win"]
        losses        = [t for t in closed_events if t.get("outcome") == "loss"]
        win_rate      = len(wins) / closed_count if closed_count else 0.0
        r_vals        = [float(t.get("r_multiple") or 0) for t in closed_events]
        avg_r         = sum(r_vals) / len(r_vals) if r_vals else 0.0

        # ── Risk state ───────────────────────────────────────────────────────────
        risk_data = {}
        risk_path = LOG_DIR / "risk_agent_state.json"
        if risk_path.exists():
            try:
                risk_data = json.loads(risk_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        eq_now   = float(risk_data.get("current_equity") or 0)
        eq_start = float(risk_data.get("daily_start_equity") or 0)
        pnl      = eq_now - eq_start if eq_now and eq_start else 0
        daily_dd_pct = ((eq_start - eq_now) / eq_start * 100) if eq_start > 0 else 0.0

        # ── Run learning loop BEFORE composing content so output lands in digest ─
        _applied: list[str] = []
        try:
            from execution import learning_loop
            _ll      = learning_loop.analyze()
            _applied = learning_loop.auto_apply(_ll)
            learning_loop.write_report(_ll, _applied)
            logger.info("Learning loop: %d obs from %d trades, %d auto-applied",
                        len(_ll["proposals"]), _ll["trades"], len(_applied))
        except Exception as exc:
            logger.warning("Learning-loop report failed: %s", exc)

        # ── CRITICAL flags ───────────────────────────────────────────────────────
        critical_flags: list[str] = []
        if daily_dd_pct > 3.5:
            critical_flags.append(
                f"DAILY DD {daily_dd_pct:.2f}% > 3.5% threshold (FTMO limit 5%)"
            )
        if hits.get("DAILY_DD_HALT"):
            critical_flags.append("DAILY_DD_HALT triggered — trading halted")
        if hits.get("PYTHON_EXCEPTION"):
            critical_flags.append(f"PYTHON_EXCEPTION: {len(hits['PYTHON_EXCEPTION'])} traceback(s)")
        # unknown_hits is keyed by hash (not label), stored in its own dict
        active_unknown = [h for h, ts in state.get("unknown_hits", {}).items() if ts]
        if active_unknown:
            critical_flags.append(
                f"Unknown error pattern(s) active: {len(active_unknown)}"
            )

        gen_ts = datetime.now(timezone.utc).strftime("%H:%M UTC")

        # ── Markdown audit — importance-ordered sections ─────────────────────────
        critical_section = (
            "\n".join(f"- **{f}**" for f in critical_flags) if critical_flags
            else "- None"
        )
        error_section = (
            "\n".join(f'- **{e["label"]}**: {e["line"][:80]}' for e in errors[-10:])
            if errors else "- None"
        )
        learning_section = (
            "\n".join(f"- {a}" for a in _applied) if _applied else "- No changes auto-applied"
        )

        fname = OBS_BRAIN / "Audit" / f"{today}-daily-audit.md"
        fname.parent.mkdir(parents=True, exist_ok=True)
        content = f"""---
type: audit
status: complete
tags: [audit, daily, auto-generated]
relatedTo: [AiDEN, trading-bot, Sessions/{today}]
date: {today}
---

# Daily Audit — {today}

*Auto-generated by code_monitor at {gen_ts}*

## 1. CRITICAL

{critical_section}

## 2. PERFORMANCE

| Metric | Value |
|--------|-------|
| Trades executed | {trade_count} |
| Positions closed | {closed_count} |
| Wins / Losses | {len(wins)} / {len(losses)} |
| Win rate | {win_rate*100:.1f}% |
| Avg R | {avg_r:+.3f} |
| Scale-ins | {scale_ins} |
| Daily P&L | ${pnl:+.2f} |
| Equity | ${eq_now:,.2f} |
| Daily DD | {daily_dd_pct:.2f}% |

## 3. GATE ACTIVITY

| Gate | Blocks |
|------|--------|
| AMD sweep blocks | {amd_blocks} |
| HTF zone direction blocks | {zone_blocks} |
| H4 range bias blocks | {range_blocks} |
| Low score skips | {low_score} |

## 4. ERRORS

{error_section}

## 5. LEARNING

{learning_section}

## Links
[[Sessions/{today}]] | [[AiDEN]]
"""
        fname.write_text(content, encoding="utf-8")
        logger.info("Daily audit written: %s", fname.name)

        # ── Machine-readable digest — one JSON line per day ──────────────────────
        digest_record = {
            "date": today, "generated_ts": gen_ts,
            "critical": critical_flags,
            "performance": {
                "trade_count": trade_count, "closed_count": closed_count,
                "wins": len(wins), "losses": len(losses),
                "win_rate": round(win_rate, 4), "avg_r": round(avg_r, 4),
                "pnl": round(pnl, 2), "equity": round(eq_now, 2),
                "daily_dd_pct": round(daily_dd_pct, 2),
            },
            "gate_activity": {
                "amd_blocks": amd_blocks, "zone_blocks": zone_blocks,
                "range_blocks": range_blocks, "low_score_skips": low_score,
                "scale_ins": scale_ins,
            },
            "error_count": len(errors),
            "learning_applied": _applied,
        }
        digest_path = LOG_DIR / "session_digest.jsonl"
        try:
            with open(digest_path, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(digest_record) + "\n")
            logger.info("Session digest appended: %s", digest_path.name)
        except Exception as exc:
            logger.warning("Session digest write failed: %s", exc)

        # Commit audit to GitHub
        _git_commit_audit(fname, today)

        state["daily_summary_done"] = today
        state["session_trades"]     = []
        state["session_errors"]     = []

    except Exception as exc:
        logger.error("Daily summary failed: %s", exc)


def _git_commit_audit(path: Path, date: str) -> None:
    """Commit the daily audit into the vault's git repo (not trading-bot's —
    the audit file lives in the Obsidian vault, outside ROOT)."""
    vault_repo = OBSIDIAN.parent
    if not (vault_repo / ".git").exists():
        logger.debug("Vault repo not initialised — skipping audit commit")
        return
    try:
        r = subprocess.run(
            ["git", "add", str(path)],
            cwd=str(vault_repo), capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            logger.warning("Audit git add failed: %s", r.stderr.strip())
            return
        r = subprocess.run(
            ["git", "commit", "-m",
             f"Auto: daily audit {date}\n\nCo-Authored-By: AiDEN Code Monitor <noreply@aiden>"],
            cwd=str(vault_repo), capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            logger.info("Audit commit skipped: %s", (r.stdout + r.stderr).strip()[:120])
            return
        remotes = subprocess.run(
            ["git", "remote"], cwd=str(vault_repo),
            capture_output=True, text=True, timeout=10,
        )
        if remotes.stdout.strip():
            subprocess.run(
                ["git", "push", "origin", "HEAD"],
                cwd=str(vault_repo), capture_output=True, timeout=30,
            )
            logger.info("Daily audit committed and pushed")
        else:
            logger.info("Daily audit committed (no remote — local only)")
    except Exception as exc:
        logger.warning("Audit git commit failed: %s", exc)

# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    _load_env()
    LOG_DIR.mkdir(exist_ok=True)

    # Prevent duplicates
    my_pid = os.getpid()
    try:
        existing = int(MONITOR_PID.read_text().strip()) if MONITOR_PID.exists() else 0
        if existing and existing != my_pid:
            try:
                os.kill(existing, 0)
                logger.error("Code monitor already running (PID %d) — exiting", existing)
                sys.exit(1)
            except OSError:
                pass
    except Exception:
        pass
    MONITOR_PID.write_text(str(my_pid))

    logger.info("Code monitor started (PID %d)", my_pid)
    _tg(f"Code monitor online (PID {my_pid})")

    state = _load_state()
    state.setdefault("session_trades", [])

    try:
        from execution.aiden_event_bus import EventBusReader
        _bus = EventBusReader("code_monitor")
    except Exception as _bus_exc:
        logger.warning("Event bus unavailable: %s — code_monitor running without bus", _bus_exc)
        _bus = None

    try:
        while True:
            # Consume event bus — append trade/learning events to session state.
            if _bus is not None:
                try:
                    for event in _bus.iter_new():
                        etype = event.get("type", "")
                        if etype == "TRADE_CLOSED":
                            state["session_trades"].append(event)
                        elif etype == "LEARNING_APPLIED":
                            logger.info(
                                "[EventBus] LEARNING_APPLIED: %s", event.get("applied", [])
                            )
                except Exception as exc:
                    logger.debug("Event bus consume error: %s", exc)

            # Scan new log lines
            findings = scan_new_lines(state)
            if findings:
                critical = [f for _, l, s in findings if s == "critical"]
                if critical:
                    logger.warning("%d critical events in this scan", len(critical))

            # Check if any patterns need escalation
            check_escalations(state)

            # Daily summary: any poll at or after DAILY_SUMMARY_H UTC triggers it.
            # write_daily_summary() has an internal date-guard so it only runs once.
            now_utc = datetime.now(timezone.utc)
            if now_utc.hour >= DAILY_SUMMARY_H:
                write_daily_summary(state)

            _save_state(state)
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Code monitor stopped")
        write_daily_summary(state)
        _save_state(state)
    finally:
        MONITOR_PID.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
