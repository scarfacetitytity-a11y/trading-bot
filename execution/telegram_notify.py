"""Telegram trade alerts for AiDEN bot.

Sends a message to the operator on every:
  - Bot startup / shutdown
  - Trade open  (with rationale: score, SL, TP, estimated hold)
  - Trade close (with outcome: R-multiple, P&L, running account)
  - Council flag (drift / consecutive losses)

Token and chat ID loaded from .env — never hardcoded.
"""
from __future__ import annotations

import logging
import os
import urllib.request
import urllib.parse
import json
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Load from .env if not already in environment
def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def _send(text: str) -> None:
    if not _TOKEN or not _CHAT_ID:
        logger.warning("[Telegram] No token/chat_id configured — skipping alert.")
        return
    try:
        url     = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"
        payload = json.dumps({"chat_id": _CHAT_ID, "text": text, "parse_mode": "Markdown"}).encode()
        req     = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        logger.warning("[Telegram] Send failed: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def notify_startup(symbols: list[str], dry_run: bool, equity: float) -> None:
    mode = "DRY RUN" if dry_run else "LIVE"
    _send(
        f"*AiDEN Bot Started* ({mode})\n"
        f"Account equity: *${equity:,.2f}*\n"
        f"Instruments: {', '.join(symbols)}\n"
        f"Time: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


def notify_shutdown(equity: float, pnl: float) -> None:
    sign = "+" if pnl >= 0 else ""
    _send(
        f"*AiDEN Bot Stopped*\n"
        f"Equity: *${equity:,.2f}*  ({sign}${pnl:,.2f} this session)\n"
        f"Time: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


def notify_trade_open(
    symbol: str,
    direction: int,
    score: int,
    entry: float,
    sl: float,
    tp: float | None,
    lots: float,
    equity: float,
    atr: float | None = None,
) -> None:
    dir_str  = "LONG" if direction == 1 else "SHORT"
    risk_usd = abs(equity * 0.005)  # approx at 0.5% risk

    # Estimate hold time from ATR
    hold_est = ""
    if atr and tp and atr > 0:
        tp_dist   = abs(tp - entry)
        bars_est  = tp_dist / atr
        mins_est  = bars_est * 15
        if mins_est < 90:
            hold_est = f"~{int(mins_est)}min"
        else:
            hold_est = f"~{mins_est/60:.1f}hrs"

    # Score rationale
    score_bar = "█" * score + "░" * (10 - score)
    score_label = "Strong" if score >= 7 else "Good" if score >= 5 else "Min threshold"

    msg = (
        f"*TRADE OPEN — {symbol}*\n"
        f"Direction: *{dir_str}*\n"
        f"Entry: `{entry:.5g}`  |  Lots: `{lots:.2f}`\n"
        f"SL: `{sl:.5g}`  |  TP: `{tp:.5g if tp else 'none'}`\n"
        f"Risk: ~${risk_usd:,.0f}\n"
        f"Score: *{score}/10* [{score_bar}] {score_label}\n"
    )
    if hold_est:
        msg += f"Est. hold: {hold_est}\n"
    msg += f"Equity: ${equity:,.2f}"

    _send(msg)


def notify_trade_close(
    symbol: str,
    direction: int,
    outcome: str,
    r_multiple: float,
    pnl_usd: float,
    equity: float,
    session_pnl: float,
) -> None:
    dir_str  = "LONG" if direction == 1 else "SHORT"
    emoji    = "WIN" if outcome == "win" else ("LOSS" if outcome == "loss" else "BE")
    sign     = "+" if pnl_usd >= 0 else ""
    sess_sign = "+" if session_pnl >= 0 else ""

    _send(
        f"*TRADE CLOSE — {symbol}* [{emoji}]\n"
        f"{dir_str}  |  R: *{r_multiple:+.2f}R*\n"
        f"P&L: *{sign}${pnl_usd:,.2f}*\n"
        f"Equity: ${equity:,.2f}  (session: {sess_sign}${session_pnl:,.2f})\n"
        f"Time: {datetime.now(tz=timezone.utc).strftime('%H:%M UTC')}"
    )


def notify_council_flag(member: str, message: str) -> None:
    _send(f"*Council #{member} FLAG*\n{message}")
