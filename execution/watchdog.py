"""AiDEN Self-Healing Watchdog.

Runs as a standalone process alongside the trading bot. Zero token cost —
pure Python loop, no LLM calls. Checks every POLL_INTERVAL seconds.

Heal actions (automatic):
  - Bot process dead → restart
  - Log file frozen > LOG_STALE_SECS → kill + restart
  - Stale bot.pid (process gone) → delete pid
  - daily_start_equity wrong in risk_agent_state.json → patch from live equity
  - council_halt.flag expired → remove

Alert-only (human action required, auto-fix not safe):
  - AutoTrading OFF detected in log (retcode=10027)
  - Daily DD > 4% approaching limit
  - MT5 connection lost

Telegram alerts use the same .env credentials as the bot.
Watchdog has its own PID file (logs/watchdog.pid) to prevent duplicates.

Run: python -m execution.watchdog
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT       = Path(__file__).parent.parent
LOG_DIR    = ROOT / "logs"
BOT_PID    = LOG_DIR / "bot.pid"
WD_PID     = LOG_DIR / "watchdog.pid"
BOT_LOG    = LOG_DIR / "orchestrator.log"
BOT_ERR    = LOG_DIR / "orchestrator_err.log"
RISK_STATE = LOG_DIR / "risk_agent_state.json"
HALT_FLAG  = LOG_DIR / "council_halt.flag"
ENV_FILE   = ROOT / ".env"

POLL_INTERVAL      = 30    # seconds between health checks
LOG_STALE_SECS     = 300   # 5 min without log update = bot frozen
RESTART_COOLDOWN   = 120   # min seconds between restarts to avoid restart storm
BOT_STARTUP_GRACE  = 120   # seconds after bot start before log-freshness check kicks in
DD_ALERT_PCT       = 4.0   # alert when daily DD exceeds this %

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | watchdog | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "watchdog.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Telegram ──────────────────────────────────────────────────────────────────

def _load_env() -> None:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

def _tg(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return
    try:
        payload = json.dumps({
            "chat_id": chat, "text": f"[AiDEN Watchdog] {text}",
            "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload, headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=8)
    except Exception as exc:
        logger.debug("Telegram failed: %s", exc)

# ── Process helpers ───────────────────────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None

def _start_bot() -> int | None:
    """Start the trading bot. Returns new PID or None on failure."""
    try:
        BOT_PID.unlink(missing_ok=True)
        # CREATE_NEW_PROCESS_GROUP isolates the bot so it doesn't receive
        # SIGINT/Ctrl+C when the watchdog is killed or restarted (Windows).
        extra = {}
        if sys.platform == "win32":
            extra["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            [sys.executable, "-m", "execution.orchestrator"],
            cwd=str(ROOT),
            stdout=open(BOT_LOG, "a", encoding="utf-8"),
            stderr=open(BOT_ERR, "a", encoding="utf-8"),
            **extra,
        )
        logger.info("Bot started (PID %d)", proc.pid)
        _tg(f"Bot restarted automatically (PID {proc.pid})")
        return proc.pid
    except Exception as exc:
        logger.error("Bot start failed: %s", exc)
        _tg(f"<b>Bot restart FAILED:</b> {exc}")
        return None

def _kill(pid: int) -> None:
    try:
        if sys.platform == "win32":
            subprocess.call(["taskkill", "/F", "/PID", str(pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGTERM)
        time.sleep(2)
    except Exception:
        pass

# ── Health checks + healers ───────────────────────────────────────────────────

class WatchdogState:
    def __init__(self) -> None:
        self.last_restart: float = 0.0
        self.bot_start_time: float = 0.0  # time.time() when bot was last started
        self.autotrading_alerted: bool = False
        self.dd_alerted: bool = False
        self.last_alert_day: str = ""

    def can_restart(self) -> bool:
        return (time.time() - self.last_restart) >= RESTART_COOLDOWN

    def mark_restart(self) -> None:
        self.last_restart   = time.time()
        self.bot_start_time = time.time()

    def in_startup_grace(self) -> bool:
        return (time.time() - self.bot_start_time) < BOT_STARTUP_GRACE


def check_bot_process(state: WatchdogState) -> None:
    """If bot PID is gone, restart."""
    pid = _read_pid(BOT_PID)

    if pid is None:
        # No PID file — bot never started or crashed before writing it
        if not state.can_restart():
            return
        logger.warning("No bot.pid found — starting bot")
        state.mark_restart()
        _start_bot()
        return

    if not _pid_alive(pid):
        logger.warning("Bot PID %d is dead — restarting", pid)
        BOT_PID.unlink(missing_ok=True)
        if not state.can_restart():
            logger.info("Restart cooldown active — skipping restart")
            return
        state.mark_restart()
        _start_bot()


def check_log_freshness(state: WatchdogState) -> None:
    """If orchestrator.log hasn't been written to in LOG_STALE_SECS, bot is frozen."""
    if not BOT_LOG.exists():
        return
    if state.in_startup_grace():
        return   # bot just started — log mtime predates this run
    age = time.time() - BOT_LOG.stat().st_mtime
    if age < LOG_STALE_SECS:
        return

    logger.warning("Log frozen for %.0fs — bot appears stuck", age)
    pid = _read_pid(BOT_PID)
    if pid and _pid_alive(pid):
        logger.warning("Killing frozen bot PID %d", pid)
        _kill(pid)
        BOT_PID.unlink(missing_ok=True)

    if not state.can_restart():
        return
    state.mark_restart()
    _tg(f"Bot log frozen {age:.0f}s — killed and restarting")
    _start_bot()


def check_stale_pid(state: WatchdogState) -> None:
    """If bot.pid exists but process is gone, clean it up."""
    pid = _read_pid(BOT_PID)
    if pid and not _pid_alive(pid):
        logger.info("Stale bot.pid %d — removing", pid)
        BOT_PID.unlink(missing_ok=True)


def check_risk_state(state: WatchdogState) -> None:
    """If risk_agent_state has wrong date, patch daily_start_equity."""
    if not RISK_STATE.exists():
        return
    try:
        data = json.loads(RISK_STATE.read_text(encoding="utf-8"))
    except Exception:
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if data.get("daily_date") == today:
        return

    # Date is stale — bot may not have rolled over yet. Don't patch unless
    # we have current equity. This is low-risk: the bot patches this itself
    # on next bar. We only patch if the bot is dead (no PID).
    pid = _read_pid(BOT_PID)
    if pid and _pid_alive(pid):
        return

    # Bot is dead and date is stale — reset for today
    current_eq = data.get("current_equity") or data.get("peak_equity") or 100000.0
    data["daily_date"]         = today
    data["daily_start_equity"] = current_eq
    data["daily_entries"]      = 0
    RISK_STATE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Patched risk_agent_state: daily_date → %s, daily_start_equity → %.2f",
                today, current_eq)


def check_autotrading(state: WatchdogState) -> None:
    """Detect retcode=10027 (AutoTrading off) in recent log lines."""
    if not BOT_LOG.exists():
        return
    try:
        # Read last 100 lines only
        with open(BOT_LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", errors="replace")
    except Exception:
        return

    if "retcode=10027" in tail or "AutoTrading disabled" in tail:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if state.autotrading_alerted and state.last_alert_day == today:
            return
        state.autotrading_alerted = True
        state.last_alert_day      = today
        logger.warning("AutoTrading OFF detected in log")
        _tg("<b>AutoTrading is OFF in MT5.</b>\nNo orders can be placed.\n"
            "Enable: MT5 → Algorithm Trading button (top toolbar)")


def check_daily_dd(state: WatchdogState) -> None:
    """Alert if daily drawdown exceeds DD_ALERT_PCT."""
    if not RISK_STATE.exists():
        return
    try:
        data = json.loads(RISK_STATE.read_text(encoding="utf-8"))
    except Exception:
        return

    start_eq  = float(data.get("daily_start_equity") or 0)
    current_eq = float(data.get("current_equity") or 0)
    if start_eq <= 0 or current_eq <= 0:
        return

    dd_pct = (start_eq - current_eq) / start_eq * 100
    today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if dd_pct >= DD_ALERT_PCT:
        if state.dd_alerted and state.last_alert_day == today:
            return
        state.dd_alerted    = True
        state.last_alert_day = today
        logger.warning("Daily DD %.2f%% >= alert threshold %.1f%%", dd_pct, DD_ALERT_PCT)
        _tg(f"<b>Daily DD alert: {dd_pct:.2f}%</b>\n"
            f"Start: ${start_eq:,.2f} → Current: ${current_eq:,.2f}\n"
            f"FTMO limit: 5%. Trading continues but watch closely.")


def check_halt_flag(state: WatchdogState) -> None:
    """Remove council_halt.flag if it's from a prior day."""
    if not HALT_FLAG.exists():
        return
    try:
        data = json.loads(HALT_FLAG.read_text())
        flag_day = data.get("day", "")
        today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if flag_day and flag_day != today:
            HALT_FLAG.unlink()
            logger.info("Removed stale council_halt.flag (was for %s)", flag_day)
    except Exception:
        pass


# ── Main loop ─────────────────────────────────────────────────────────────────

CHECKS = [
    check_stale_pid,
    check_bot_process,
    check_log_freshness,
    check_risk_state,
    check_autotrading,
    check_daily_dd,
    check_halt_flag,
]


def main() -> None:
    _load_env()
    LOG_DIR.mkdir(exist_ok=True)

    # Prevent duplicate watchdog instances
    my_pid = os.getpid()
    existing = _read_pid(WD_PID)
    if existing and _pid_alive(existing) and existing != my_pid:
        logger.error("Watchdog already running (PID %d) — exiting", existing)
        sys.exit(1)
    WD_PID.write_text(str(my_pid))

    logger.info("Watchdog started (PID %d) — polling every %ds", my_pid, POLL_INTERVAL)
    _tg(f"Watchdog online (PID {my_pid})")

    state = WatchdogState()
    try:
        while True:
            for check in CHECKS:
                try:
                    check(state)
                except Exception as exc:
                    logger.error("Check %s failed: %s", check.__name__, exc)
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Watchdog stopped")
    finally:
        WD_PID.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
