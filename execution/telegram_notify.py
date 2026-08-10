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
import time
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


def _mirror_obsidian(event_type: str, title: str, body: str, tags: list) -> None:
    """Mirror significant bot events to Obsidian Brain — fire-and-forget."""
    import threading
    def _bg():
        try:
            from execution.council_obsidian import write_alert
            write_alert(
                alert_id=f"{event_type}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
                title=title,
                severity="info",
                body=body,
                tags=tags + ["telegram-mirror", "aiden"],
            )
        except Exception:
            pass
    threading.Thread(target=_bg, name="TgObsidianMirror", daemon=True).start()


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


def _score_bar_blocks(score: int, max_score: int = 10) -> str:
    filled = max(0, min(10, round(score / max_score * 10)))
    return "▰" * filled + "▱" * (10 - filled)


def _conviction(score: int) -> str:
    if score >= 8: return "ELITE"
    if score >= 6: return "HIGH CONVICTION"
    if score == 5: return "QUALIFIED"
    if score == 4: return "ESTIMATED"
    return "SPECULATIVE"


def _rr_fmt(rr: float) -> str:
    """1:5 when clean, 1:6.3 otherwise — never a third number."""
    return f"1:{rr:.1f}".rstrip("0").rstrip(".")


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
    _mirror_obsidian(
        "startup",
        f"Bot Online — ${equity:,.0f} — {'DRY RUN' if dry_run else 'LIVE'}",
        f"Equity: ${equity:,.2f}\nTarget: ${equity * 1.10:,.2f}\nInstruments: {', '.join(symbols)}",
        ["startup", "bot-event"],
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
    _mirror_obsidian(
        "shutdown",
        f"Bot Stopped — Session P&L {sign}${pnl:,.2f}",
        f"Equity: ${equity:,.2f}\nSession P&L: {sign}${pnl:,.2f}",
        ["shutdown", "bot-event"],
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
    reasons: Optional[list] = None,
) -> None:
    dir_icon = "🟢" if direction == 1 else "🔴"
    dir_word = "LONG" if direction == 1 else "SHORT"

    # Actual risk from lot size × SL distance (pip value approximation)
    # Falls back to equity × risk_pct only if we can't compute from position
    if sl and abs(entry - sl) > 0 and lots > 0:
        try:
            import MetaTrader5 as mt5
            info = mt5.symbol_info(symbol)
            tick_val = info.trade_tick_value if info else 1.0
            tick_size = info.trade_tick_size if info else 0.00001
            sl_points = abs(entry - sl) / tick_size
            risk_usd = lots * sl_points * tick_val
        except Exception:
            risk_usd = equity * float(os.environ.get("RISK_PCT", "0.5")) / 100
    else:
        risk_usd = equity * float(os.environ.get("RISK_PCT", "0.5")) / 100

    # R:R + potential profit (the money)
    if tp and sl and abs(entry - sl) > 0:
        rr = abs(tp - entry) / abs(entry - sl)
        rr_str = _rr_fmt(rr)
        profit_usd = rr * risk_usd
    else:
        rr, rr_str, profit_usd = 0.0, "—", 0.0

    # Est. hold time
    hold_str = ""
    if atr and tp and atr > 0:
        mins = (abs(tp - entry) / atr) * 15
        hold_str = f"~{int(mins)}min" if mins < 90 else f"~{mins/60:.1f}h"

    tp_str = f"{tp:.5g}" if tp else "—"
    caption = (
        f"{dir_icon} <b>{dir_word}  {symbol}</b>\n\n"
        f"💰 <b>+${profit_usd:,.0f}</b>   🛡 −${risk_usd:,.0f}   ⚖️ {rr_str}\n\n"
        f"<b>Entry</b> <code>{entry:.5g}</code> · {lots:.2f} lots\n"
        f"🎯 <code>{tp_str}</code>  ·  🛡 <code>{sl:.5g}</code>\n\n"
        f"📊 {score} {_score_bar_blocks(score, max_score=max(10, score))} <b>{_conviction(score)}</b>\n"
    )
    # Score breakdown — the confluences that built the number
    if reasons:
        caption += "".join(f"   ✓ {r}\n" for r in reasons)
    caption += f"🕐 {datetime.now(tz=timezone.utc).strftime('%H:%M UTC')}"
    if hold_str:
        caption += f" · {hold_str}"

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


def notify_trade_review(
    symbol:       str,
    direction:    int,
    score:        int,
    entry:        float,
    sl:           float,
    tp:           Optional[float],
    close_price:  float,
    r_multiple:   float,
    pnl_usd:      float,
    equity:       float,
    session_pnl:  float,
    outcome:      str,
    thesis:       Optional[str]  = None,
    reasons:      Optional[list] = None,
    hit_target:   Optional[bool] = None,
    mfe_r:        Optional[float] = None,
    mae_r:        Optional[float] = None,
    lesson:       Optional[str]  = None,
    review_notes: Optional[list] = None,
) -> None:
    """Full post-trade analysis message — entry, TP, thesis, convictions, result, lesson, balance."""
    icons   = {"win": "✅", "loss": "❌", "breakeven": "➖", "unknown": "❓"}
    icon    = icons.get(outcome, "❓")
    dir_str = "LONG" if direction == 1 else "SHORT"
    s_sign  = "+" if session_pnl >= 0 else ""
    r_sign  = "+" if r_multiple >= 0 else ""
    usd_sign = "+" if pnl_usd >= 0 else ""

    tp_str = f"{tp:.5g}" if tp else "—"

    msg = (
        f"{icon} <b>{dir_str}  {symbol}  {r_sign}{r_multiple:.2f}R</b>\n\n"
        f"<b>Entry</b> <code>{entry:.5g}</code>  ·  "
        f"<b>TP</b> <code>{tp_str}</code>  ·  "
        f"<b>SL</b> <code>{sl:.5g}</code>\n"
        f"<b>Close</b> <code>{close_price:.5g}</code>  ·  "
        f"<b>P&L</b> <code>{usd_sign}${pnl_usd:,.2f}</code>\n\n"
    )

    if thesis:
        msg += f"<b>Thesis:</b> {thesis}\n\n"

    msg += f"<b>Score:</b> {score} {_score_bar_blocks(score, max_score=max(10, score))} <b>{_conviction(score)}</b>\n"
    if reasons:
        msg += "".join(f"   ✓ {r}\n" for r in reasons)

    msg += "\n"

    # Path analysis
    if mfe_r is not None and mae_r is not None:
        msg += (
            f"<b>MFE</b> {'+' if mfe_r >= 0 else ''}{mfe_r:.2f}R  ·  "
            f"<b>MAE</b> {mae_r:.2f}R"
        )
        if hit_target is not None:
            msg += f"  ·  <b>Target hit:</b> {'YES' if hit_target else 'NO'}"
        msg += "\n"

    if lesson:
        msg += f"\n<b>Lesson:</b> <i>{lesson}</i>\n"
    if review_notes:
        for n in review_notes:
            msg += f"   • {n}\n"

    msg += (
        f"\n💰 <b>Balance:</b> ${equity:,.2f}  "
        f"| <b>Session:</b> {s_sign}${session_pnl:,.2f}\n"
        f"🕐 {datetime.now(tz=timezone.utc).strftime('%H:%M UTC')}"
    )

    _send_text(msg)


def notify_council_flag(member: str, message: str) -> None:
    _send_text(
        f"⚠️ <b>Council #{member}</b>\n\n"
        f"{message}"
    )
    _mirror_obsidian(
        "council-flag",
        f"Council #{member} — {message[:60]}",
        f"Member: {member}\n\n{message}",
        ["council", "alert", "bot-event"],
    )


def notify_preday_brief(brief: dict) -> None:
    """Morning pre-day analysis brief.

    brief keys:
      date          : str  e.g. "Mon 21 Jul"
      levels        : list of dict {symbol, label, price, note}
      watchlist     : list of str  e.g. ["XAUUSD watching Asian sweep"]
      news          : list of str  e.g. ["09:30 USD CPI (high)"]
      equity        : float
    """
    lines = [
        f"<b>AiDEN Pre-Day Brief — {brief.get('date', '')}</b>",
        f"Equity: <b>${brief.get('equity', 0):,.0f}</b>",
        "",
    ]

    news = brief.get("news", [])
    if news:
        lines.append("<b>Key News</b>")
        for n in news[:6]:
            lines.append(f"  • {n}")
        lines.append("")

    levels = brief.get("levels", [])
    if levels:
        lines.append("<b>Key Levels</b>")
        sym_groups: dict = {}
        for lv in levels:
            sym_groups.setdefault(lv["symbol"], []).append(lv)
        for sym, lvs in sym_groups.items():
            lines.append(f"  <b>{sym}</b>")
            for lv in lvs:
                note = f" — {lv['note']}" if lv.get("note") else ""
                lines.append(f"    {lv['label']}: {lv['price']:.5g}{note}")
        lines.append("")

    watchlist = brief.get("watchlist", [])
    if watchlist:
        lines.append("<b>Watching</b>")
        for w in watchlist[:6]:
            lines.append(f"  • {w}")

    _send_text("\n".join(lines))


# ── Telegram approval flow ────────────────────────────────────────────────────
# Sends a trade proposal with APPROVE / VETO inline buttons.
# Polls getUpdates for up to timeout_sec. Auto-decides on timeout based on score.

_last_update_id: int = 0


def _send_with_keyboard(html: str, reply_markup: str) -> Optional[int]:
    """Send message with inline keyboard. Returns message_id or None."""
    if not _TOKEN or not _CHAT_ID:
        return None
    try:
        url     = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"
        payload = json.dumps({
            "chat_id":      _CHAT_ID,
            "text":         html,
            "parse_mode":   "HTML",
            "reply_markup": json.loads(reply_markup),
            "disable_web_page_preview": True,
        }).encode()
        req  = urllib.request.Request(url, data=payload,
                                      headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=8)
        data = json.loads(resp.read())
        return data["result"]["message_id"] if data.get("ok") else None
    except Exception as exc:
        logger.warning("[Telegram] send_with_keyboard failed: %s", exc)
        return None


def _get_updates(offset: int, timeout: int = 5) -> list:
    if not _TOKEN:
        return []
    try:
        url  = (f"https://api.telegram.org/bot{_TOKEN}/getUpdates"
                f"?offset={offset}&timeout={max(1, timeout)}&allowed_updates=callback_query")
        req  = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=timeout + 3)
        data = json.loads(resp.read())
        return data.get("result", []) if data.get("ok") else []
    except Exception:
        return []


def _answer_callback(callback_id: str) -> None:
    if not _TOKEN:
        return
    try:
        url     = f"https://api.telegram.org/bot{_TOKEN}/answerCallbackQuery"
        payload = json.dumps({"callback_query_id": callback_id}).encode()
        req     = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def _edit_message(msg_id: int, html: str) -> None:
    if not _TOKEN or not _CHAT_ID or not msg_id:
        return
    try:
        url     = f"https://api.telegram.org/bot{_TOKEN}/editMessageText"
        payload = json.dumps({
            "chat_id":    _CHAT_ID,
            "message_id": msg_id,
            "text":       html,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=8)
    except Exception:
        pass


def request_approval(
    symbol: str,
    direction: int,
    score: int,
    entry: float,
    sl: float,
    tp: Optional[float],
    lots: float,
    equity: float,
    council_notes: str = "",
    timeout_sec: int   = 90,
    auto_approve_score: int = 6,
) -> bool:
    """Send trade proposal with APPROVE/VETO buttons. Returns True = go ahead.

    If no Telegram configured, always returns True (non-blocking).
    On timeout: auto-approves if score >= auto_approve_score, else auto-vetoes.
    """
    global _last_update_id

    if not _TOKEN or not _CHAT_ID:
        return True

    dir_str  = "LONG 📈" if direction == 1 else "SHORT 📉"
    dir_icon = "🟢" if direction == 1 else "🔴"
    rr_str   = ""
    if tp and sl:
        dist_sl = abs(entry - sl)
        dist_tp = abs(tp - entry)
        rr_str  = f"1:{dist_tp/dist_sl:.1f}" if dist_sl > 0 else "—"

    auto_action = "AUTO-APPROVE" if score >= auto_approve_score else "AUTO-VETO"
    base_text = (
        f"{dir_icon} <b>TRADE SIGNAL — {symbol}</b>\n\n"
        f"<b>Direction:</b>  {dir_str}\n"
        f"<b>Entry:</b>     <code>{entry:.5g}</code>\n"
        f"<b>SL:</b>        <code>{sl:.5g}</code>\n"
        f"<b>TP:</b>        <code>{f'{tp:.5g}' if tp else '—'}</code>\n"
        f"<b>RR:</b>        {rr_str}\n"
        f"<b>Lots:</b>      {lots:.2f}  |  <b>Equity:</b> ${equity:,.0f}\n\n"
        f"📊 <b>Score:</b> {score}/10  <code>{_score_bar(score)}</code>  {_score_label(score)}\n"
    )
    if council_notes:
        base_text += f"\n🏛 <b>Council:</b> {council_notes}\n"
    base_text += f"\n⏳ <i>{auto_action} in {timeout_sec}s if no response</i>"

    approval_id  = f"{symbol.replace('.','_')}_{int(time.time())}"
    reply_markup = json.dumps({"inline_keyboard": [[
        {"text": "✅  APPROVE", "callback_data": f"approve_{approval_id}"},
        {"text": "❌  VETO",    "callback_data": f"veto_{approval_id}"},
    ]]})

    msg_id = _send_with_keyboard(base_text, reply_markup)

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        poll_timeout = min(5, max(1, int(deadline - time.time())))
        updates = _get_updates(_last_update_id + 1, timeout=poll_timeout)

        for update in updates:
            uid = update.get("update_id", 0)
            if uid > _last_update_id:
                _last_update_id = uid

            cb = update.get("callback_query")
            if cb and approval_id in cb.get("data", ""):
                _answer_callback(cb["id"])
                approved   = cb["data"].startswith("approve_")
                result_str = "✅ APPROVED" if approved else "❌ VETOED"
                _edit_message(msg_id, base_text + f"\n\n<b>{result_str}</b>")
                logger.info("[Telegram] Trade %s %s by user", symbol, result_str)
                return approved

    auto  = score >= auto_approve_score
    label = "✅ AUTO-APPROVED (timeout)" if auto else "❌ AUTO-VETOED (timeout)"
    _edit_message(msg_id, base_text + f"\n\n<b>{label}</b>")
    logger.info("[Telegram] Trade %s %s", symbol, label)
    return auto
