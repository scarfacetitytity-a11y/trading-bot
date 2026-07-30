"""Level Monitor — pre-marks key liquidity levels and watches for price approach.

The bot scans every M15 bar reactively. This module adds a proactive layer:
pre-compute the key structural levels (D1 H/L, Asian H/L, weekly H/L, active
FVG zones) at the start of each UTC day, then alert when price enters proximity
of any level. When price is near a level the strategy switches to higher attention
— the orchestrator can tighten its scoring gate or log an explicit approach warning.

JP mentor methodology (pre-session level-setting):
- Mark H4/D1 FVGs and OBs before London open
- Identify equal highs/lows as liquidity pools (expect sweep + reverse)
- Know buy/sell bias at each zone BEFORE price arrives
- Fire only when price taps the pre-marked zone

Levels refreshed: on UTC day rollover (00:00) and on demand.
Alert: when price within proximity_atr * ATR of a level, or when bar enters HTF zone.
Pre-London brief sent via Telegram at 07:00-07:30 UTC daily.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ENV_PATH = Path(__file__).parent.parent / ".env"


def _load_tg_credentials() -> tuple[str, str]:
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    return (
        os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        os.environ.get("TELEGRAM_CHAT_ID", ""),
    )


_TG_TOKEN, _TG_CHAT = _load_tg_credentials()


def _tg_send(text: str) -> None:
    if not _TG_TOKEN or not _TG_CHAT:
        return
    try:
        url     = f"https://api.telegram.org/bot{_TG_TOKEN}/sendMessage"
        payload = json.dumps({
            "chat_id":   _TG_CHAT,
            "text":      text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=8)
    except Exception as exc:
        logger.debug("[LevelMonitor] Telegram send failed: %s", exc)


@dataclass
class KeyLevel:
    label: str          # e.g. "Prior D1 High", "Asian Session Low"
    price: float
    direction: int      # +1 = resistance above (sell zone), -1 = support below (buy zone), 0 = both
    source_tf: str      # "D1", "H4", "Asian", "Weekly"
    formed_at: datetime
    hit: bool = False   # True once price has touched within proximity
    score_boost: int = 1  # confluence points added to entry score


@dataclass
class HTFZone:
    """Higher-timeframe FVG, order block, or equal H/L liquidity zone.

    Unlike KeyLevel (single price), HTFZone has a price range (zone_lo, zone_hi).
    direction: +1 = bullish zone (expect buy when price enters), -1 = bearish zone.
    score_boost: extra points added when bar enters zone (FVG=2, OB=2, EQ liquidity=1).
    """
    label: str
    zone_lo: float
    zone_hi: float
    direction: int      # +1 = bullish zone, -1 = bearish zone
    zone_type: str      # "fvg", "ob", "eqh", "eql"
    source_tf: str      # "H4", "D1"
    formed_at: datetime
    touched: bool = False   # True once price has traded into zone
    score_boost: int = 2    # extra confluence points

    @property
    def price(self) -> float:
        """Midpoint — allows generic l.price access in orchestrator."""
        return (self.zone_lo + self.zone_hi) / 2.0

    @property
    def hit(self) -> bool:
        return self.touched

    @hit.setter
    def hit(self, v: bool) -> None:
        self.touched = v


@dataclass
class LevelSet:
    symbol: str
    levels: list[KeyLevel] = field(default_factory=list)
    zones:  list[HTFZone]  = field(default_factory=list)
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

    def in_zones(self, bar_high: float, bar_low: float) -> list[HTFZone]:
        """Return zones the current bar has entered (intra-bar containment check)."""
        entered = []
        for z in self.zones:
            if z.touched:
                continue
            # Bar enters zone if any part of bar overlaps the zone range
            if bar_low <= z.zone_hi and bar_high >= z.zone_lo:
                entered.append(z)
        return entered

    def mark_hit(self, price: float, atr: float) -> None:
        """Mark levels that price has passed through."""
        for l in self.levels:
            if not l.hit and abs(l.price - price) <= 0.1 * atr:
                l.hit = True
                logger.info("[LevelMonitor] Level hit: %s %.5f", l.label, l.price)

    def mark_zones_touched(self, bar_high: float, bar_low: float) -> None:
        """Mark zones price has entered as touched."""
        for z in self.zones:
            if not z.touched and bar_low <= z.zone_hi and bar_high >= z.zone_lo:
                z.touched = True
                logger.info("[LevelMonitor] Zone touched: %s [%.5f-%.5f]",
                            z.label, z.zone_lo, z.zone_hi)


class LevelMonitor:
    """Computes and tracks key structural levels for all symbols.

    Called by TradingEngine at the start of each bar. When price approaches
    a key level, logs an alert and returns the level for scoring amplification.

    Pre-London brief (07:00-07:30 UTC): sends Telegram summary of H4/D1 zones
    for each symbol so the trader knows what to watch before London opens.
    """

    # UTC hour window for pre-London daily brief
    _BRIEF_HOUR_START = 7
    _BRIEF_HOUR_END   = 8

    def __init__(self, proximity_atr: float = 1.0):
        self.proximity_atr = proximity_atr
        self._level_sets: dict[str, LevelSet] = {}
        self._last_day:   dict[str, int] = {}
        self._brief_sent_day: dict[str, int] = {}  # track per-symbol brief day

    def update(self, symbol: str, df: pd.DataFrame, atr: float) -> list:
        """Refresh levels if new UTC day; return any levels/zones price is near or entering.

        df: M15 dataframe with columns time, open, high, low, close.
        atr: current ATR value for proximity calculation.
        Returns list of KeyLevel | HTFZone objects (all have .label, .price, .score_boost).
        """
        now   = datetime.now(timezone.utc)
        today = now.day

        if symbol not in self._last_day or self._last_day[symbol] != today:
            self._compute_levels(symbol, df, now)
            self._last_day[symbol] = today

        ls = self._level_sets.get(symbol)
        if ls is None:
            return []

        current_price = float(df["close"].iloc[-1])
        bar_high      = float(df["high"].iloc[-1])
        bar_low       = float(df["low"].iloc[-1])

        ls.mark_hit(current_price, atr)
        ls.mark_zones_touched(bar_high, bar_low)

        # Key level proximity (existing scoring)
        approaching: list = ls.approaching(current_price, atr, self.proximity_atr)
        for lvl in approaching:
            logger.info(
                "[LevelMonitor] %s approaching %s @ %.5f (current=%.5f dist=%.5f atr=%.5f)",
                symbol, lvl.label, lvl.price, current_price,
                abs(lvl.price - current_price), atr,
            )

        # HTF zone entries — intra-bar containment (new)
        zone_hits: list = ls.in_zones(bar_high, bar_low)
        for z in zone_hits:
            logger.info(
                "[LevelMonitor] %s BAR ENTERED ZONE %s [%.5f-%.5f] dir=%+d boost=%d",
                symbol, z.label, z.zone_lo, z.zone_hi, z.direction, z.score_boost,
            )

        # Pre-London daily brief
        self._maybe_send_brief(symbol, ls, now)

        return approaching + zone_hits

    def get_levels(self, symbol: str) -> list[KeyLevel]:
        ls = self._level_sets.get(symbol)
        return ls.levels if ls else []

    def get_zones(self, symbol: str) -> list[HTFZone]:
        ls = self._level_sets.get(symbol)
        return ls.zones if ls else []

    # ── Level computation ─────────────────────────────────────────────────────

    def _compute_levels(self, symbol: str, df: pd.DataFrame, now: datetime) -> None:
        """Recompute all key levels and HTF zones from current data."""
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

        # ── Previous Battlefields (PB) — D1 swing extremes from past 2–7 days ─
        for _pb_day in range(2, 8):
            _pb_start = today_utc - pd.Timedelta(days=_pb_day)
            _pb_end   = today_utc - pd.Timedelta(days=_pb_day - 1)
            _pb_mask  = (times >= _pb_start) & (times < _pb_end)
            if _pb_mask.any():
                _pb_h = float(high[_pb_mask].max())
                _pb_l = float(low[_pb_mask].min())
                levels.append(KeyLevel(f"PB High D-{_pb_day}", _pb_h, +1, "D1", now))
                levels.append(KeyLevel(f"PB Low D-{_pb_day}",  _pb_l, -1, "D1", now))

        # ── Asian Session H/L + 50% midpoint (00:00–07:00 UTC of today) ────────
        asian_start = today_utc
        asian_end   = today_utc.replace(hour=7)
        asian_mask  = (times >= asian_start) & (times < asian_end)
        if asian_mask.any():
            a_high = float(high[asian_mask].max())
            a_low  = float(low[asian_mask].min())
            a_mid  = (a_high + a_low) / 2.0
            levels.append(KeyLevel("Asian Session High", a_high, +1, "Asian", now))
            levels.append(KeyLevel("Asian Session Low",  a_low,  -1, "Asian", now))
            levels.append(KeyLevel("Asian 50% Mid",      a_mid,   0, "Asian", now))

        # ── Weekly H/L (Mon 00:00 UTC of current week) ───────────────────────
        days_since_mon = now.weekday()
        week_start = today_utc - pd.Timedelta(days=days_since_mon)
        weekly_mask = times >= week_start
        if weekly_mask.any():
            w_high = float(high[weekly_mask].max())
            w_low  = float(low[weekly_mask].min())
            levels.append(KeyLevel("Weekly High", w_high, +1, "Weekly", now))
            levels.append(KeyLevel("Weekly Low",  w_low,  -1, "Weekly", now))

        # ── Prior NY Session H/L (12:00–21:00 UTC yesterday) ────────────────────
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
                h4_hi = float(h4_rs["high"].iloc[-2])
                h4_lo = float(h4_rs["low"].iloc[-2])
                levels.append(KeyLevel("H4 Swing High", h4_hi, +1, "H4", now))
                levels.append(KeyLevel("H4 Swing Low",  h4_lo, -1, "H4", now))

        # Remove exact duplicates (within 0.01% of each other)
        deduped: list[KeyLevel] = []
        for lvl in levels:
            if not any(abs(l.price - lvl.price) / max(lvl.price, 1) < 0.0001 for l in deduped):
                deduped.append(lvl)

        # ── HTF FVG / OB / Equal H/L zones (JP pre-session methodology) ──────
        current_price = float(close.iloc[-1])
        zones = self._scan_htf_zones(df, current_price, now)

        self._level_sets[symbol] = LevelSet(
            symbol=symbol, levels=deduped, zones=zones, computed_at=now
        )
        logger.info(
            "[LevelMonitor] %s: %d levels, %d HTF zones computed",
            symbol, len(deduped), len(zones),
        )

    def _scan_htf_zones(self, df: pd.DataFrame, current_price: float, now: datetime) -> list[HTFZone]:
        """Detect H4 FVGs, H4 order blocks, and D1 equal highs/lows.

        Resamples M15 data to H4 (48 bars = 8 days) and D1 (10 bars).
        Only returns FRESH zones (not yet entered by price since formation).
        """
        zones: list[HTFZone] = []

        try:
            df_idx = df.copy()
            df_idx.index = pd.to_datetime(df_idx["time"])
            df_idx = df_idx.sort_index()

            # ── H4 resample ───────────────────────────────────────────────────
            h4 = df_idx[["open", "high", "low", "close"]].resample("4h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna().iloc[-48:]  # last 8 days of H4

            # ── H4 FVGs ───────────────────────────────────────────────────────
            # Bullish FVG: bar[i].high < bar[i+2].low → imbalance above (buy zone)
            # Bearish FVG: bar[i].low > bar[i+2].high → imbalance below (sell zone)
            for i in range(len(h4) - 2):
                b0 = h4.iloc[i]
                b1 = h4.iloc[i + 1]
                b2 = h4.iloc[i + 2]
                ts = pd.Timestamp(h4.index[i + 1]).to_pydatetime()

                # Bullish FVG: unfilled gap between b0.high and b2.low
                if b2["low"] > b0["high"]:
                    z_lo, z_hi = float(b0["high"]), float(b2["low"])
                    # Fresh: price hasn't yet traded back into the zone
                    later_lows = h4["low"].iloc[i + 3:]
                    if later_lows.empty or later_lows.min() > z_lo:
                        zones.append(HTFZone(
                            label=f"H4 Bullish FVG {ts.strftime('%m/%d %Hh')}",
                            zone_lo=z_lo, zone_hi=z_hi,
                            direction=+1, zone_type="fvg", source_tf="H4",
                            formed_at=ts, score_boost=2,
                        ))

                # Bearish FVG: unfilled gap between b2.high and b0.low
                if b2["high"] < b0["low"]:
                    z_lo, z_hi = float(b2["high"]), float(b0["low"])
                    later_highs = h4["high"].iloc[i + 3:]
                    if later_highs.empty or later_highs.max() < z_hi:
                        zones.append(HTFZone(
                            label=f"H4 Bearish FVG {ts.strftime('%m/%d %Hh')}",
                            zone_lo=z_lo, zone_hi=z_hi,
                            direction=-1, zone_type="fvg", source_tf="H4",
                            formed_at=ts, score_boost=2,
                        ))

            # ── H4 Order Blocks ───────────────────────────────────────────────
            # Bullish OB: last bearish candle before 3-bar bullish impulse
            # Bearish OB: last bullish candle before 3-bar bearish impulse
            for i in range(len(h4) - 4):
                b   = h4.iloc[i]
                nxt = h4.iloc[i + 1 : i + 4]
                ts  = pd.Timestamp(h4.index[i]).to_pydatetime()

                # Bullish OB: bearish base candle followed by 3 bullish candles
                if b["close"] < b["open"]:  # bearish base
                    if all(nxt["close"].values > nxt["open"].values):
                        z_lo = float(b["low"])
                        z_hi = float(b["open"])
                        if z_hi > z_lo:
                            # Fresh: price hasn't closed below z_lo since
                            later = h4["low"].iloc[i + 4:]
                            if later.empty or later.min() > z_lo:
                                zones.append(HTFZone(
                                    label=f"H4 Bullish OB {ts.strftime('%m/%d %Hh')}",
                                    zone_lo=z_lo, zone_hi=z_hi,
                                    direction=+1, zone_type="ob", source_tf="H4",
                                    formed_at=ts, score_boost=2,
                                ))

                # Bearish OB: bullish base candle followed by 3 bearish candles
                if b["close"] > b["open"]:  # bullish base
                    if all(nxt["close"].values < nxt["open"].values):
                        z_lo = float(b["close"])
                        z_hi = float(b["high"])
                        if z_hi > z_lo:
                            later = h4["high"].iloc[i + 4:]
                            if later.empty or later.max() < z_hi:
                                zones.append(HTFZone(
                                    label=f"H4 Bearish OB {ts.strftime('%m/%d %Hh')}",
                                    zone_lo=z_lo, zone_hi=z_hi,
                                    direction=-1, zone_type="ob", source_tf="H4",
                                    formed_at=ts, score_boost=2,
                                ))

            # ── D1 Equal Highs / Equal Lows (liquidity pools) ─────────────────
            # Equal highs: 2+ daily highs within 0.02% → liquidity above (expect sweep + reverse short)
            # Equal lows: 2+ daily lows within 0.02% → liquidity below (expect sweep + reverse long)
            d1 = df_idx[["open", "high", "low", "close"]].resample("1D").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna().iloc[-10:]

            d1_highs = d1["high"].values
            d1_lows  = d1["low"].values
            d1_times = [pd.Timestamp(t).to_pydatetime() for t in d1.index]

            tol = 0.0002  # 0.02% tolerance for "equal"

            for i in range(len(d1) - 1):
                for j in range(i + 1, len(d1)):
                    # Equal highs
                    h_i, h_j = d1_highs[i], d1_highs[j]
                    if abs(h_i - h_j) / max(h_i, 1) < tol:
                        eq_price = (h_i + h_j) / 2.0
                        spread   = max(h_i, h_j) * tol
                        # Only include if price hasn't yet traded through these highs
                        if current_price < eq_price - spread:
                            zones.append(HTFZone(
                                label=f"D1 Equal Highs {d1_times[i].strftime('%m/%d')}/{d1_times[j].strftime('%m/%d')}",
                                zone_lo=eq_price - spread,
                                zone_hi=eq_price + spread,
                                direction=-1,  # sweep above highs → expect reversal short
                                zone_type="eqh", source_tf="D1",
                                formed_at=d1_times[j], score_boost=1,
                            ))

                    # Equal lows
                    l_i, l_j = d1_lows[i], d1_lows[j]
                    if abs(l_i - l_j) / max(l_i, 1) < tol:
                        eq_price = (l_i + l_j) / 2.0
                        spread   = max(l_i, l_j) * tol
                        if current_price > eq_price + spread:
                            zones.append(HTFZone(
                                label=f"D1 Equal Lows {d1_times[i].strftime('%m/%d')}/{d1_times[j].strftime('%m/%d')}",
                                zone_lo=eq_price - spread,
                                zone_hi=eq_price + spread,
                                direction=+1,  # sweep below lows → expect reversal long
                                zone_type="eql", source_tf="D1",
                                formed_at=d1_times[j], score_boost=1,
                            ))

        except Exception as exc:
            logger.warning("[LevelMonitor] HTF zone scan error: %s", exc)

        # Sort by proximity to current price, keep top 12 most relevant
        zones.sort(key=lambda z: abs(z.price - current_price))
        return zones[:12]

    # ── Pre-London daily brief ────────────────────────────────────────────────

    def _maybe_send_brief(self, symbol: str, ls: LevelSet, now: datetime) -> None:
        """Send pre-London Telegram brief once per symbol per day, 07:00–08:00 UTC."""
        if not _TG_TOKEN or not _TG_CHAT:
            return
        today = now.day
        if self._brief_sent_day.get(symbol) == today:
            return
        if not (self._BRIEF_HOUR_START <= now.hour < self._BRIEF_HOUR_END):
            return

        self._brief_sent_day[symbol] = today
        self._send_level_brief(symbol, ls)

    def _send_level_brief(self, symbol: str, ls: LevelSet) -> None:
        """Format and send pre-session level brief to Telegram."""
        lines = [f"<b>AiDEN Pre-Session Levels — {symbol}</b> {datetime.now(timezone.utc).strftime('%H:%M UTC')}"]

        # Fresh HTF zones (most important)
        fresh_zones = [z for z in ls.zones if not z.touched]
        if fresh_zones:
            lines.append("\n<b>HTF Zones (watch for tap):</b>")
            for z in fresh_zones[:6]:
                dir_str = "BUY" if z.direction == +1 else "SELL"
                lines.append(
                    f"  {z.zone_type.upper()} {dir_str} [{z.zone_lo:.5g}–{z.zone_hi:.5g}] "
                    f"<i>{z.label}</i>"
                )

        # Key structural levels nearby (top 5)
        nearby = [l for l in ls.levels if not l.hit][:5]
        if nearby:
            lines.append("\n<b>Key Levels:</b>")
            for l in nearby:
                dir_str = "↑" if l.direction == -1 else "↓" if l.direction == +1 else "↕"
                lines.append(f"  {dir_str} {l.price:.5g} — {l.label}")

        _tg_send("\n".join(lines))
        logger.info("[LevelMonitor] Pre-session brief sent for %s (%d zones, %d levels)",
                    symbol, len(fresh_zones), len(nearby))
