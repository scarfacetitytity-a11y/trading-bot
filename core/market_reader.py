"""Phase 3: MarketReader — stateful zone tracker with lifecycle management.

Each FVG, OB, or liquidity sweep zone moves through states:
  FRESH → TOUCHED → VIOLATED or CONSUMED

IntraBarEvents are raised when price action changes zone status mid-bar,
enabling the AnalyzerEngine to react before bar close.

Phase 3: Zones updated on M15 bar close (batch mode).
Phase 4 upgrade: background thread polling M5 every 30s for intra-bar events.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ZoneState(str, Enum):
    FRESH     = "fresh"      # Zone formed, not yet tested
    TOUCHED   = "touched"    # Price entered zone, did not violate
    VIOLATED  = "violated"   # Price passed through — zone failed
    CONSUMED  = "consumed"   # Price swept through and rejected back — full trade signal


class ZoneType(str, Enum):
    FVG        = "fvg"
    OB         = "ob"
    SWEEP      = "sweep"
    MANIPULATION = "manipulation"


class IntraBarEvent(str, Enum):
    SWEEP_REJECTED    = "sweep_rejected"     # Price swept through zone and rejected
    ADVERSE_BAR       = "adverse_bar"        # Current bar moving strongly against zone
    LIQUIDITY_TAKEN   = "liquidity_taken"    # Buy/sell-side liquidity cleared
    MOMENTUM_ALIGNED  = "momentum_aligned"   # Consecutive bars confirming direction
    ZONE_VIOLATED     = "zone_violated"      # Zone failed mid-bar


@dataclass
class Zone:
    symbol:     str
    zone_type:  ZoneType
    direction:  int           # +1 = bullish zone (entry long), -1 = bearish
    price_lo:   float
    price_hi:   float
    formed_bar: int           # bar index when zone was detected
    formed_time: datetime
    strength:   float = 1.0  # 0-1; weakens on touch, resets on rejection
    state:      ZoneState = ZoneState.FRESH
    touch_count: int = 0
    last_event: Optional[IntraBarEvent] = None
    notes:      list[str] = field(default_factory=list)

    @property
    def midpoint(self) -> float:
        return (self.price_lo + self.price_hi) / 2

    @property
    def size(self) -> float:
        return self.price_hi - self.price_lo

    def is_alive(self) -> bool:
        return self.state not in (ZoneState.VIOLATED,)

    def price_inside(self, price: float) -> bool:
        return self.price_lo <= price <= self.price_hi


@dataclass
class BarEvent:
    """A zone-related event raised at bar close."""
    bar_time:   datetime
    symbol:     str
    event:      IntraBarEvent
    zone:       Zone
    price:      float
    note:       str = ""


class MarketReader:
    """Stateful zone tracker for a single symbol.

    Called on each M15 bar close with the fresh OHLCV dataframe.
    Updates zone states and raises IntraBarEvents when price interacts
    with tracked zones.
    """

    MAX_ZONES     = 20   # maximum live zones per direction
    MAX_ZONE_AGE  = 96   # bars (~24h on M15) after which FRESH zones expire

    def __init__(self, symbol: str) -> None:
        self.symbol  = symbol
        self._zones: list[Zone] = []
        self._bar_idx = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, df: pd.DataFrame) -> list[BarEvent]:
        """Process a new M15 bar. Returns events raised this bar."""
        if df is None or len(df) < 3:
            return []
        self._bar_idx += 1
        events = []

        last   = df.iloc[-1]
        hi     = float(last["high"])
        lo     = float(last["low"])
        close  = float(last["close"])
        open_  = float(last["open"])
        try:
            bar_time = pd.to_datetime(last["time"], utc=True).to_pydatetime()
        except Exception:
            bar_time = datetime.now(timezone.utc)

        # Expire stale zones
        self._expire_old_zones()

        # Detect new zones from the last 3 bars
        new_zones = self._detect_zones(df)
        for z in new_zones:
            if len([x for x in self._zones if x.direction == z.direction]) < self.MAX_ZONES:
                self._zones.append(z)

        # Update existing zone states
        for zone in list(self._zones):
            ev = self._evaluate_zone(zone, hi, lo, close, open_, bar_time)
            if ev:
                events.append(ev)

        # Prune consumed and violated zones
        self._zones = [z for z in self._zones if z.is_alive()]

        return events

    def get_live_zones(self, direction: Optional[int] = None) -> list[Zone]:
        """Return all alive zones, optionally filtered by direction."""
        if direction is None:
            return [z for z in self._zones if z.is_alive()]
        return [z for z in self._zones if z.is_alive() and z.direction == direction]

    def add_zone(self, zone: Zone) -> None:
        """Manually register a zone (e.g. from strategy output)."""
        self._zones.append(zone)

    # ── Zone detection ────────────────────────────────────────────────────────

    def _detect_zones(self, df: pd.DataFrame) -> list[Zone]:
        """Detect FVG and manipulation patterns from the last 3 bars."""
        zones: list[Zone] = []
        if len(df) < 3:
            return zones

        b0, b1, b2 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        try:
            bar_time = pd.to_datetime(b1["time"], utc=True).to_pydatetime()
        except Exception:
            bar_time = datetime.now(timezone.utc)

        # Bullish FVG: b0.low > b2.high (gap between bars 0 and 2, bar 1 impulse up)
        if float(b0["low"]) > float(b2["high"]):
            zones.append(Zone(
                symbol=self.symbol, zone_type=ZoneType.FVG, direction=1,
                price_lo=float(b2["high"]), price_hi=float(b0["low"]),
                formed_bar=self._bar_idx - 1, formed_time=bar_time,
            ))

        # Bearish FVG: b0.high < b2.low
        if float(b0["high"]) < float(b2["low"]):
            zones.append(Zone(
                symbol=self.symbol, zone_type=ZoneType.FVG, direction=-1,
                price_lo=float(b0["high"]), price_hi=float(b2["low"]),
                formed_bar=self._bar_idx - 1, formed_time=bar_time,
            ))

        # Bullish manipulation W (sweep low then reverse — JP mentor "W" pattern)
        b1_lo = float(b1["low"])
        b0_lo = float(b0["low"])
        b1_cl = float(b1["close"])
        b1_op = float(b1["open"])
        if b1_lo < b0_lo and b1_cl > b1_op:   # swept lower, closed bullish
            zones.append(Zone(
                symbol=self.symbol, zone_type=ZoneType.MANIPULATION, direction=1,
                price_lo=b1_lo, price_hi=float(b1["high"]),
                formed_bar=self._bar_idx - 1, formed_time=bar_time, strength=1.2,
            ))

        # Bearish manipulation M
        b1_hi = float(b1["high"])
        b0_hi = float(b0["high"])
        if b1_hi > b0_hi and b1_cl < b1_op:   # swept higher, closed bearish
            zones.append(Zone(
                symbol=self.symbol, zone_type=ZoneType.MANIPULATION, direction=-1,
                price_lo=float(b1["low"]), price_hi=b1_hi,
                formed_bar=self._bar_idx - 1, formed_time=bar_time, strength=1.2,
            ))

        return zones

    # ── Zone evaluation ───────────────────────────────────────────────────────

    def _evaluate_zone(
        self, zone: Zone, hi: float, lo: float,
        close: float, open_: float, bar_time: datetime,
    ) -> Optional[BarEvent]:
        """Advance zone state based on this bar's OHLCV. Returns event if notable."""

        if zone.direction == 1:   # bullish zone — price approaching from above
            if lo <= zone.price_hi:
                if close > zone.price_hi:
                    # Price dipped into zone and closed back above: SWEEP_REJECTED (strong signal)
                    zone.state       = ZoneState.CONSUMED
                    zone.last_event  = IntraBarEvent.SWEEP_REJECTED
                    zone.touch_count += 1
                    return BarEvent(
                        bar_time=bar_time, symbol=self.symbol,
                        event=IntraBarEvent.SWEEP_REJECTED, zone=zone,
                        price=close,
                        note=f"Bull zone swept and rejected: lo={lo:.5f} zone=[{zone.price_lo:.5f},{zone.price_hi:.5f}]",
                    )
                elif close < zone.price_lo:
                    # Closed below zone: violated
                    zone.state      = ZoneState.VIOLATED
                    zone.last_event = IntraBarEvent.ZONE_VIOLATED
                    return BarEvent(
                        bar_time=bar_time, symbol=self.symbol,
                        event=IntraBarEvent.ZONE_VIOLATED, zone=zone,
                        price=close,
                        note=f"Bull zone violated: close={close:.5f} < lo={zone.price_lo:.5f}",
                    )
                else:
                    # Inside zone
                    zone.state       = ZoneState.TOUCHED
                    zone.touch_count += 1
                    zone.strength   *= 0.9   # weakens slightly on touch

        else:   # bearish zone
            if hi >= zone.price_lo:
                if close < zone.price_lo:
                    zone.state       = ZoneState.CONSUMED
                    zone.last_event  = IntraBarEvent.SWEEP_REJECTED
                    zone.touch_count += 1
                    return BarEvent(
                        bar_time=bar_time, symbol=self.symbol,
                        event=IntraBarEvent.SWEEP_REJECTED, zone=zone,
                        price=close,
                        note=f"Bear zone swept and rejected: hi={hi:.5f} zone=[{zone.price_lo:.5f},{zone.price_hi:.5f}]",
                    )
                elif close > zone.price_hi:
                    zone.state      = ZoneState.VIOLATED
                    zone.last_event = IntraBarEvent.ZONE_VIOLATED
                    return BarEvent(
                        bar_time=bar_time, symbol=self.symbol,
                        event=IntraBarEvent.ZONE_VIOLATED, zone=zone,
                        price=close,
                        note=f"Bear zone violated: close={close:.5f} > hi={zone.price_hi:.5f}",
                    )
                else:
                    zone.state       = ZoneState.TOUCHED
                    zone.touch_count += 1
                    zone.strength   *= 0.9

        return None

    def _expire_old_zones(self) -> None:
        for zone in list(self._zones):
            age = self._bar_idx - zone.formed_bar
            if zone.state == ZoneState.FRESH and age > self.MAX_ZONE_AGE:
                self._zones.remove(zone)
