"""Telegram trade alerts for AiDEN bot — rich formatting + chart images.

Messages use HTML mode (cleaner than MarkdownV2).
Charts generated with matplotlib and sent as photos.

Token and chat ID loaded from .env — never hardcoded.
"""
from __future__ import annotations

import io
import json
import logging
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

logger = logging.getLogger(__name__)


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


# ── Transport ─────────────────────────────────────────────────────────────────

def _send_text(html: str) -> None:
    if not _TOKEN or not _CHAT_ID:
        return
    try:
        url     = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"
        payload = json.dumps({
            "chat_id":    _CHAT_ID,
            "text":       html,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=8)
    except Exception as exc:
        logger.warning("[Telegram] send_text failed: %s", exc)


def _send_photo(img_bytes: bytes, caption: str) -> None:
    if not _TOKEN or not _CHAT_ID:
        return
    try:
        boundary = "AiDENBotBoundary"
        body  = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="chat_id"\r\n\r\n{_CHAT_ID}\r\n'
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="parse_mode"\r\n\r\nHTML\r\n'
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="caption"\r\n\r\n{caption}\r\n'
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="photo"; filename="chart.png"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

        url = f"https://api.telegram.org/bot{_TOKEN}/sendPhoto"
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        urllib.request.urlopen(req, timeout=15)
    except Exception as exc:
        logger.warning("[Telegram] send_photo failed: %s", exc)


# ── Chart generation ──────────────────────────────────────────────────────────

def _make_trade_chart(
    df: pd.DataFrame,
    entry: float,
    sl: float,
    tp: Optional[float],
    symbol: str,
    direction: int,
) -> bytes:
    """Generate a price chart with entry / SL / TP levels. Returns PNG bytes."""
    tail = df.tail(60).copy()
    times  = range(len(tail))
    closes = tail["close"].values

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    # Price line
    ax.plot(times, closes, color="#e0e0e0", linewidth=1.2, zorder=3)
    ax.fill_between(times, closes, closes.min() * 0.998, alpha=0.08, color="#4fc3f7")

    # Entry / SL / TP lines
    ax.axhline(entry, color="#4fc3f7", linewidth=1.4, linestyle="--", label=f"Entry {entry:.5g}", zorder=4)
    ax.axhline(sl,    color="#ef5350", linewidth=1.4, linestyle=":",  label=f"SL {sl:.5g}", zorder=4)
    if tp:
        ax.axhline(tp, color="#66bb6a", linewidth=1.4, linestyle=":", label=f"TP {tp:.5g}", zorder=4)

    # Shade SL-to-TP zone
    if tp:
        lo, hi = (sl, tp) if direction == 1 else (tp, sl)
        ax.axhspan(lo, entry, alpha=0.08, color="#ef5350")
        ax.axhspan(entry, hi, alpha=0.10, color="#66bb6a")

    # Mark entry bar
    ax.axvline(len(times) - 1, color="#4fc3f7", linewidth=0.8, linestyle="-", alpha=0.5)

    # Styling
    dir_str = "LONG" if direction == 1 else "SHORT"
    ax.set_title(f"{symbol}  |  {dir_str}  |  M15  —  AiDEN",
                 color="#e0e0e0", fontsize=11, pad=8)
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    legend = ax.legend(fontsize=8, loc="upper left",
                       facecolor="#1a1a2e", edgecolor="#444",
                       labelcolor="#e0e0e0")

    ax.yaxis.tick_right()
    ax.set_xticks([])

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


# ── Score bar ─────────────────────────────────────────────────────────────────

def _score_bar(score: int, max_score: int = 10) -> str:
    filled = round(score / max_score * 10)
    return "█" * filled + "░" * (10 - filled)


def _score_label(score: int) -> str:
    if score >= 8: return "Elite"
    if score >= 6: return "Strong"
    if score >= 4: return "Qualified"
    return "Marginal"


# ── Public API ────────────────────────────────────────────────────────────────

def notify_startup(symbols: list[str], dry_run: bool, equity: float) -> None:
    mode = "🔵 DRY RUN" if dry_run else "🟢 LIVE"
    pairs = " | ".join(symbols)
    _send_text(
        f"{mode} <b>AiDEN Bot Online</b>\n\n"
        f"💰 <b>Equity:</b> ${equity:,.2f}\n"
        f"🎯 <b>Target:</b> ${equity * 1.10:,.2f} (+10% FTMO)\n"
        f"📊 <b>Instruments:</b> {len(symbols)}\n"
        f"<code>{pairs}</code>\n\n"
        f"🕐 {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"<i>SL/TP are broker-managed — safe if connection drops.</i>"
    )


def notify_shutdown(equity: float, pnl: float) -> None:
    icon = "🟢" if pnl >= 0 else "🔴"
    sign = "+" if pnl >= 0 else ""
    _send_text(
        f"🔴 <b>AiDEN Bot Stopped</b>\n\n"
        f"💰 <b>Equity:</b> ${equity:,.2f}\n"
        f"{icon} <b>Session P&L:</b> {sign}${pnl:,.2f}\n"
        f"<i>Open positions remain protected by broker SL/TP.</i>\n\n"
        f"🕐 {datetime.now(tz=timezone.utc).strftime('%H:%M UTC')}"
    )


def notify_trade_open(
    symbol: str,
    direction: int,
    score: int,
    entry: float,
    sl: float,
    tp: Optional[float],
    lots: float,
    equity: float,
    atr: Optional[float] = None,
    df: Optional[pd.DataFrame] = None,
) -> None:
    dir_str  = "LONG  📈" if direction == 1 else "SHORT 📉"
    dir_icon = "🟢" if direction == 1 else "🔴"
    risk_usd = equity * float(os.environ.get("RISK_PCT", "0.5")) / 100

    # RR
    if tp and sl:
        dist_sl = abs(entry - sl)
        dist_tp = abs(tp - entry)
        rr = dist_tp / dist_sl if dist_sl > 0 else 0
        rr_str = f"1:{rr:.1f}"
    else:
        rr_str = "—"

    # Est. hold time
    hold_str = ""
    if atr and tp and atr > 0:
        bars = abs(tp - entry) / atr
        mins = bars * 15
        hold_str = f"~{int(mins)}min" if mins < 90 else f"~{mins/60:.1f}hrs"

    caption = (
        f"{dir_icon} <b>TRADE OPEN — {symbol}</b>\n\n"
        f"<b>Direction:</b> {dir_str}\n"
        f"<b>Entry:</b> <code>{entry:.5g}</code>  |  <b>Lots:</b> <code>{lots:.2f}</code>\n\n"
        f"🛡 <b>SL:</b> <code>{sl:.5g}</code>\n"
        f"🎯 <b>TP:</b> <code>{f'{tp:.5g}' if tp else '—'}</code>\n"
        f"⚖️ <b>RR:</b> {rr_str}\n"
        f"💸 <b>Risk:</b> ~${risk_usd:,.0f}\n\n"
        f"📊 <b>Score:</b> {score}/10  <code>{_score_bar(score)}</code>  {_score_label(score)}\n"
    )
    if hold_str:
        caption += f"⏱ <b>Est. hold:</b> {hold_str}\n"
    caption += f"\n🕐 {datetime.now(tz=timezone.utc).strftime('%H:%M UTC')}"

    if df is not None and len(df) >= 10:
        try:
            img = _make_trade_chart(df, entry, sl, tp, symbol, direction)
            _send_photo(img, caption)
            return
        except Exception as exc:
            logger.warning("[Telegram] Chart generation failed: %s", exc)

    _send_text(caption)


def notify_trade_close(
    symbol: str,
    direction: int,
    outcome: str,
    r_multiple: float,
    pnl_usd: float,
    equity: float,
    session_pnl: float,
) -> None:
    icons = {"win": "✅", "loss": "❌", "breakeven": "➖", "unknown": "❓"}
    icon  = icons.get(outcome, "❓")
    sign  = "+" if pnl_usd >= 0 else ""
    s_sign = "+" if session_pnl >= 0 else ""
    dir_str = "LONG" if direction == 1 else "SHORT"

    _send_text(
        f"{icon} <b>TRADE CLOSE — {symbol}</b>\n\n"
        f"<b>Direction:</b> {dir_str}  |  <b>Result:</b> {outcome.upper()}\n"
        f"<b>R-Multiple:</b> <code>{r_multiple:+.2f}R</code>\n"
        f"<b>P&L:</b> <code>{sign}${pnl_usd:,.2f}</code>\n\n"
        f"💰 <b>Equity:</b> ${equity:,.2f}\n"
        f"📈 <b>Session:</b> {s_sign}${session_pnl:,.2f}\n\n"
        f"🕐 {datetime.now(tz=timezone.utc).strftime('%H:%M UTC')}"
    )


def notify_council_flag(member: str, message: str) -> None:
    _send_text(
        f"⚠️ <b>Council #{member}</b>\n\n"
        f"{message}"
    )
