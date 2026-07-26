"""Pre-day analysis — morning brief sent to Telegram before London open.

JP mentor v10: "my alerts work as my little AI workers — they keep an eye on
things and let me know when liquidity is taken."

Runs at a configurable UTC time (default 06:45 — 15 min before London).
For each symbol, identifies:
  - Asian session H/L (prior Asia consolidation range — resting liquidity)
  - Prior day H/L (daily draws on price)
  - Weekly H/L (highest-TF institutional draw)
  - Active FVG zones from the strategy (if available)
  - Any key structural swing levels within 3×ATR of current price

Output goes to Telegram via notify_preday_brief().
Can also be called manually: python -m execution.preday_analysis
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Asian session UTC hours (approximate — Tokyo/Sydney overlap)
_ASIA_START_H = 22   # 22:00 UTC prior day
_ASIA_END_H   = 7    # 07:00 UTC

# Symbols to brief (override via config)
_DEFAULT_SYMBOLS = ["XAUUSD", "US30.cash", "US100.cash", "GBPUSD", "EURUSD",
                    "USDJPY", "JP225.cash", "US2000.cash"]


def _asia_range(bars_h1, current_utc: datetime) -> tuple[Optional[float], Optional[float]]:
    """Return (asian_high, asian_low) from H1 bars in the most recent Asia session."""
    import numpy as np
    import pandas as pd

    if bars_h1 is None or len(bars_h1) == 0:
        return None, None

    df = pd.DataFrame(bars_h1)
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # Most recent Asia session: yesterday 22:00 → today 07:00 UTC
    today = current_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    asia_end   = today.replace(hour=_ASIA_END_H)
    asia_start = (today - timedelta(days=1)).replace(hour=_ASIA_START_H)

    mask = (df["dt"] >= asia_start) & (df["dt"] < asia_end)
    asia = df[mask]
    if len(asia) == 0:
        return None, None
    return float(asia["high"].max()), float(asia["low"].min())


def _prior_day_hl(bars_h1, current_utc: datetime) -> tuple[Optional[float], Optional[float]]:
    """Return (prior_day_high, prior_day_low) from H1 bars."""
    import pandas as pd

    if bars_h1 is None or len(bars_h1) == 0:
        return None, None

    df = pd.DataFrame(bars_h1)
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)

    today = current_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    pd_start = today - timedelta(days=1)
    pd_end   = today

    mask = (df["dt"] >= pd_start) & (df["dt"] < pd_end)
    prev = df[mask]
    if len(prev) == 0:
        return None, None
    return float(prev["high"].max()), float(prev["low"].min())


def _weekly_hl(bars_h1, current_utc: datetime) -> tuple[Optional[float], Optional[float]]:
    """Return (weekly_high, weekly_low) from the current week's H1 bars."""
    import pandas as pd

    if bars_h1 is None or len(bars_h1) == 0:
        return None, None

    df = pd.DataFrame(bars_h1)
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # Monday 00:00 UTC of current week
    days_since_mon = current_utc.weekday()
    week_start = (current_utc - timedelta(days=days_since_mon)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    mask = df["dt"] >= week_start
    week = df[mask]
    if len(week) == 0:
        return None, None
    return float(week["high"].max()), float(week["low"].min())


def _current_atr(bars_h1) -> float:
    """14-period ATR from H1 bars."""
    import numpy as np

    if bars_h1 is None or len(bars_h1) < 15:
        return 0.0
    highs  = [float(b["high"])  for b in bars_h1[-15:]]
    lows   = [float(b["low"])   for b in bars_h1[-15:]]
    closes = [float(b["close"]) for b in bars_h1[-15:]]
    trs = [max(highs[i] - lows[i],
               abs(highs[i] - closes[i-1]),
               abs(lows[i]  - closes[i-1])) for i in range(1, 15)]
    return float(np.mean(trs))


def build_brief(symbols: list[str], equity: float) -> dict:
    """Build the pre-day brief dict. Requires MT5 to be connected."""
    import MetaTrader5 as mt5

    now = datetime.now(timezone.utc)
    levels = []
    watchlist = []

    for sym in symbols:
        # Fetch 5 days of H1 bars
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 120)
        if rates is None or len(rates) == 0:
            continue

        tick = mt5.symbol_info_tick(sym)
        price = ((tick.bid + tick.ask) / 2.0) if tick else 0.0
        atr   = _current_atr(list(rates))

        # Asian session H/L
        a_hi, a_lo = _asia_range(list(rates), now)
        if a_hi is not None:
            levels.append({"symbol": sym, "label": "Asia H", "price": a_hi,
                           "note": "above price" if a_hi > price else "below price"})
        if a_lo is not None:
            levels.append({"symbol": sym, "label": "Asia L", "price": a_lo,
                           "note": "above price" if a_lo > price else "below price"})

        # Prior day H/L
        pd_hi, pd_lo = _prior_day_hl(list(rates), now)
        if pd_hi is not None:
            levels.append({"symbol": sym, "label": "PDH", "price": pd_hi,
                           "note": "above" if pd_hi > price else "below"})
        if pd_lo is not None:
            levels.append({"symbol": sym, "label": "PDL", "price": pd_lo,
                           "note": "above" if pd_lo > price else "below"})

        # Weekly H/L
        w_hi, w_lo = _weekly_hl(list(rates), now)
        if w_hi is not None and w_hi != pd_hi:
            levels.append({"symbol": sym, "label": "WeekH", "price": w_hi,
                           "note": "above" if w_hi > price else "below"})
        if w_lo is not None and w_lo != pd_lo:
            levels.append({"symbol": sym, "label": "WeekL", "price": w_lo,
                           "note": "above" if w_lo > price else "below"})

        # Watchlist entry: levels within 1.5×ATR of current price
        if atr > 0:
            nearby = [lv for lv in levels if lv["symbol"] == sym
                      and abs(lv["price"] - price) <= 1.5 * atr]
            for nb in nearby[:2]:
                watchlist.append(
                    f"{sym} — {nb['label']} {nb['price']:.5g} nearby ({abs(nb['price']-price):.5g} away)"
                )

    # Pull upcoming high-impact news for the day
    news_items: list[str] = []
    try:
        from execution import news_gate as ng
        gate = ng.NewsGate()
        ctx  = gate.get_context(upcoming_window_min=60 * 16)   # next 16 hours
        for ev in sorted(ctx.upcoming_high, key=lambda e: e.minutes_until)[:8]:
            eta_h = ev.minutes_until // 60
            eta_m = ev.minutes_until % 60
            news_items.append(f"+{eta_h}h{eta_m:02d}m {ev.currency} {ev.title}")
    except Exception:
        pass

    return {
        "date":      now.strftime("%a %d %b %H:%M UTC"),
        "equity":    equity,
        "levels":    levels,
        "watchlist": watchlist,
        "news":      news_items,
    }


def run_preday_brief(symbols: list[str] | None = None, equity: float = 0.0) -> None:
    """Build and send the pre-day brief. Requires MT5 connected."""
    import json as _json
    from execution import telegram_notify as tg

    syms = symbols or _DEFAULT_SYMBOLS
    try:
        brief = build_brief(syms, equity)
        tg.notify_preday_brief(brief)
        logger.info("[PreDay] Brief sent — %d levels, %d news",
                    len(brief["levels"]), len(brief["news"]))
        # Cache to disk so obsidian-session-start.py can inject it into Claude sessions
        _cache = Path(__file__).resolve().parent.parent / "logs" / "preday_brief.json"
        _cache.write_text(_json.dumps(brief, default=str), encoding="utf-8")
    except Exception:
        logger.exception("[PreDay] Failed to build/send brief")


def main() -> None:
    """Standalone: connect MT5, send brief, disconnect."""
    import sys
    from config.settings import load_config
    from config.logging_setup import setup_logger
    from backtests.mt5_connector import connect, disconnect

    setup_logger()
    cfg = load_config()
    syms = cfg.get("trading", {}).get("symbols", _DEFAULT_SYMBOLS)
    terminal = cfg.get("mt5", {}).get("terminal_path", "")

    if not connect(terminal):
        print("MT5 connect failed")
        sys.exit(1)

    import MetaTrader5 as mt5
    acct = mt5.account_info()
    equity = acct.equity if acct else 0.0

    run_preday_brief(syms, equity)
    disconnect()


if __name__ == "__main__":
    main()
