"""Level Monitor — pre-marks key liquidity levels and watches for price approach.

The bot scans every M15 bar reactively. This module adds a proactive layer:
pre-compute the key structural levels (D1 H/L, Asian H/L, weekly H/L, active
FVG zones) at the start of each UTC day, then alert when price enters proximity
of any level. When price is near a level the strategy switches to higher attention
— the orchestrator can tighten its scoring gate or log an explicit approach warning.

Levels refreshed: on UTC day rollover (00:00) and on demand.
Alert: when price within proximity_atr * ATR of a level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KeyLevel:
    label: str          # e.g. "Prior D1 High", "Asian Session Low"
    price: float
    direction: int      # +1 = resistance above (sell zone), -1 = support below (buy zone), 0 = both
    source_tf: str      # "D1", "H4", "Asian", "Weekly"
    formed_at: datetime
    hit: bool = False   # True once price has touched within proximity


@dataclass
class LevelSet:
    symbol: str
    levels: list[KeyLevel] = field(default_factory=list)
    computed_at: Optional[datetime] = None

    def nearest(self, price: float, direction: int) -> Optional[KeyLevel]:
        """Return the closest unhit level in the trade direction."""
        candidates = [
            l for l in self.levels
            if not l.hit and (l.direction == 0 or l.direction == direction)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda l: abs(l.price - price))

    def approaching(self, price: float, atr: float, proximity_atr: float = 1.0) -> list[KeyLevel]:
        """Return all levels price is within proximity_atr * ATR of."""
        threshold = proximity_atr * atr
        return [l for l in self.levels if not l.hit and abs(l.price - price) <= threshold]

    def mark_hit(self, price: float, atr: float) -> None:
        """Mark levels that price has passed through."""
        for l in self.levels:
            if not l.hit and abs(l.price - price) <= 0.1 * atr:
                l.hit = True
                logger.info("[LevelMonitor] Level hit: %s %.5f", l.label, l.price)


class LevelMonitor:
    """Computes and tracks key structural levels for all symbols.

    Called by TradingEngine at the start of each bar. When price approaches
    a key level, logs an alert and returns the level for scoring amplification.
    """

    def __init__(self, proximity_atr: float = 1.0):
        self.proximity_atr = proximity_atr
        self._level_sets: dict[str, LevelSet] = {}
        self._last_day: dict[str, int] = {}

    def update(self, symbol: str, df: pd.DataFrame, atr: float) -> list[KeyLevel]:
        """Refresh levels if new UTC day; return any levels price is approaching.

        df: M15 dataframe with columns time, open, high, low, close.
        atr: current ATR value for proximity calculation.
        Returns list of KeyLevel objects price is near (empty = nothing notable).
        """
        now = datetime.now(timezone.utc)
        today = now.day

        if symbol not in self._last_day or self._last_day[symbol] != today:
            self._compute_levels(symbol, df, now)
            self._last_day[symbol] = today

        ls = self._level_sets.get(symbol)
        if ls is None or not ls.levels:
            return []

        current_price = float(df["close"].iloc[-1])
        ls.mark_hit(current_price, atr)

        approaching = ls.approaching(current_price, atr, self.proximity_atr)
        for lvl in approaching:
            logger.info(
                "[LevelMonitor] %s approaching %s @ %.5f (current=%.5f dist=%.5f atr=%.5f)",
                symbol, lvl.label, lvl.price, current_price,
                abs(lvl.price - current_price), atr,
            )
        return approaching

    def get_levels(self, symbol: str) -> list[KeyLevel]:
        ls = self._level_sets.get(symbol)
        return ls.levels if ls else []

    def _compute_levels(self, symbol: str, df: pd.DataFrame, now: datetime) -> None:
        """Recompute all key levels from current data."""
        times  = pd.to_datetime(df["time"])
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        close  = df["close"].astype(float)

        levels: list[KeyLevel] = []
        today_utc = now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

        # ── Prior D1 High / Low ───────────────────────────────────────────────
        yesterday = today_utc - pd.Timedelta(days=1)
        d1_mask   = (times >= yesterday) & (times < today_utc)
        if d1_mask.any():
            d1_high = float(high[d1_mask].max())
            d1_low  = float(low[d1_mask].min())
            levels.append(KeyLevel("Prior D1 High", d1_high, +1, "D1", now))
            levels.append(KeyLevel("Prior D1 Low",  d1_low,  -1, "D1", now))

        # ── Asian Session H/L (00:00–07:00 UTC of today) ─────────────────────
        asian_start = today_utc
        asian_end   = today_utc.replace(hour=7)
        asian_mask  = (times >= asian_start) & (times < asian_end)
        if asian_mask.any():
            a_high = float(high[asian_mask].max())
            a_low  = float(low[asian_mask].min())
            levels.append(KeyLevel("Asian Session High", a_high, +1, "Asian", now))
            levels.append(KeyLevel("Asian Session Low",  a_low,  -1, "Asian", now))

        # ── Weekly H/L (Mon 00:00 UTC of current week) ───────────────────────
        days_since_mon = now.weekday()  # Monday=0
        week_start = today_utc - pd.Timedelta(days=days_since_mon)
        weekly_mask = times >= week_start
        if weekly_mask.any():
            w_high = float(high[weekly_mask].max())
            w_low  = float(low[weekly_mask].min())
            levels.append(KeyLevel("Weekly High", w_high, +1, "Weekly", now))
            levels.append(KeyLevel("Weekly Low",  w_low,  -1, "Weekly", now))

        # ── Prior NY Session H/L (12:00–21:00 UTC yesterday) ────────────────────
        # JP mentor places alerts at "New York session highs" as key POIs.
        ny_start_h, ny_end_h = 12, 21
        ny_mask = (
            (times >= yesterday) & (times < today_utc)
            & (times.dt.hour >= ny_start_h) & (times.dt.hour < ny_end_h)
        )
        if ny_mask.any():
            ny_high = float(high[ny_mask].max())
            ny_low  = float(low[ny_mask].min())
            levels.append(KeyLevel("Prior NY High", ny_high, +1, "NY", now))
            levels.append(KeyLevel("Prior NY Low",  ny_low,  -1, "NY", now))

        # ── London Session 50% Midpoint (07:00–12:00 UTC today) ────────────────
        # JP mentor video 4: "NY pushed up to 50% of the London session and had
        # huge wick rejection, then continued London's bearish energy."
        lon_start_h, lon_end_h = 7, 12
        lon_mask = (times >= today_utc) & (times.dt.hour >= lon_start_h) & (times.dt.hour < lon_end_h)
        if lon_mask.any():
            lon_high = float(high[lon_mask].max())
            lon_low  = float(low[lon_mask].min())
            lon_mid  = (lon_high + lon_low) / 2.0
            levels.append(KeyLevel("London 50% Mid", lon_mid, 0, "London", now))

        # ── H4 Swing Highs / Lows (last 5 H4 bars) ───────────────────────────
        h4_start = today_utc - pd.Timedelta(hours=20)
        h4_mask  = times >= h4_start
        if h4_mask.any():
            h4_df = df[h4_mask].copy()
            h4_df.index = pd.to_datetime(h4_df["time"])
            h4_rs = h4_df[["high", "low", "close"]].resample("4h").agg(
                {"high": "max", "low": "min", "close": "last"}
            ).dropna()
            if len(h4_rs) >= 2:
                # Most recent completed H4 swing points
                h4_hi = float(h4_rs["high"].iloc[-2])
                h4_lo = float(h4_rs["low"].iloc[-2])
                levels.append(KeyLevel("H4 Swing High", h4_hi, +1, "H4", now))
                levels.append(KeyLevel("H4 Swing Low",  h4_lo, -1, "H4", now))

        # Remove exact duplicates (within 0.01% of each other)
        deduped: list[KeyLevel] = []
        for lvl in levels:
            if not any(abs(l.price - lvl.price) / max(lvl.price, 1) < 0.0001 for l in deduped):
                deduped.append(lvl)

        self._level_sets[symbol] = LevelSet(
            symbol=symbol, levels=deduped, computed_at=now
        )
        logger.info(
            "[LevelMonitor] %s: %d levels computed — %s",
            symbol, len(deduped),
            ", ".join(f"{l.label}={l.price:.2f}" for l in deduped),
        )
