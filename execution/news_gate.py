"""News gate — economic calendar awareness for live trade management.

Fetches ForexFactory calendar JSON every 30 min (background thread).
Direction is derived from Forecast vs Previous (consensus expectation), NOT actuals —
the FF feed does not publish actuals. This means the gate tracks market consensus,
not confirmed prints. It functions as an AMPLIFIER only: +1 score when a setup
aligns with macro consensus; 0 otherwise. It never blocks or force-closes.

For detection of actual prints (beat/miss), price-action post-event logic is needed
separately in TradingEngine — this module handles schedule and expected direction only.

Integration points:
  1. Pre-entry: score +1 if new signal aligns with consensus direction
  2. Logging: know what events are scheduled so engine can log them
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Ordered backends — Agent-Reach pattern: try each in order, first success wins.
_CALENDAR_BACKENDS = [
    ("ForexFactory JSON",  "https://nfs.faireconomy.media/ff_calendar_thisweek.json"),
    ("FF Mirror",          "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json"),
]
_FF_URL       = _CALENDAR_BACKENDS[0][1]   # kept for backwards-compat references
_REFRESH_SECS = 1800   # 30 min

# ── Instrument → currency mapping ─────────────────────────────────────────────

_INSTRUMENT_CURRENCY: dict[str, str] = {
    "XAUUSD":      "USD",
    "XAGUSD":      "USD",
    "US100.cash":  "USD",
    "US30.cash":   "USD",
    "US500.cash":  "USD",
    "US2000.cash": "USD",
    "UK100.cash":  "GBP",
    "JP225.cash":  "JPY",
}

_METALS = {"XAUUSD", "XAGUSD"}

# USD event name → category: "inflation" or "growth"
_USD_INFLATION = ["cpi", "ppi", "pce", "core inflation", "inflation expectations", "import prices"]
_USD_GROWTH    = [
    "nonfarm", "nfp", "gdp", "retail sales", "ism manufacturing", "ism services",
    "adp", "unemployment claims", "durable goods", "industrial production",
    "consumer confidence", "business confidence", "philly fed",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class NewsEvent:
    impact:     str             # "High", "Medium", "Low"
    event_time: datetime        # UTC
    currency:   str
    name:       str
    prev:       Optional[float] = None
    forecast:   Optional[float] = None

    @property
    def minutes_since(self) -> float:
        return (datetime.now(timezone.utc) - self.event_time).total_seconds() / 60

    @property
    def minutes_until(self) -> float:
        return -self.minutes_since

    @property
    def has_fired(self) -> bool:
        return self.minutes_since >= 0

    @property
    def consensus_direction(self) -> int:
        """Direction implied by Fcst vs Prev (+1 = positive surprise expected, -1 = negative).
        None if one of the values is missing. Only valid for events with numeric data."""
        if self.forecast is None or self.prev is None:
            return 0
        if self.forecast > self.prev:
            return 1
        if self.forecast < self.prev:
            return -1
        return 0

    def expected_direction_for(self, symbol: str) -> int:
        """Translate consensus surprise expectation into directional impact for this symbol."""
        ccy = _INSTRUMENT_CURRENCY.get(symbol, "")
        if ccy != self.currency or self.consensus_direction == 0:
            return 0

        name_low = self.name.lower()
        beat_expected = self.consensus_direction == 1  # Fcst > Prev

        if self.currency == "USD":
            if symbol in _METALS:
                # Gold/silver move INVERSE to USD strength
                if any(k in name_low for k in _USD_INFLATION + _USD_GROWTH):
                    return -1 if beat_expected else 1
            else:
                # US index instruments
                if any(k in name_low for k in _USD_INFLATION):
                    # Higher inflation expected → rate-hike fear → bearish indices
                    return -1 if beat_expected else 1
                if any(k in name_low for k in _USD_GROWTH):
                    # Stronger growth expected → risk-on → bullish indices
                    return 1 if beat_expected else -1

        elif self.currency == "GBP" and symbol == "UK100.cash":
            if any(k in name_low for k in ["gdp", "retail", "pmi", "employment", "trade"]):
                return 1 if beat_expected else -1
            if any(k in name_low for k in ["cpi", "inflation"]):
                return -1 if beat_expected else 1

        elif self.currency == "JPY" and symbol == "JP225.cash":
            # JPY strength bearish for Nikkei
            if any(k in name_low for k in ["cpi", "gdp", "pmi"]):
                return -1 if beat_expected else 1

        return 0


@dataclass
class NewsContext:
    """Snapshot of news state."""
    fired_high:    list[NewsEvent] = field(default_factory=list)  # HIGH events fired in last 2h
    upcoming_high: list[NewsEvent] = field(default_factory=list)  # HIGH within next 60 min
    fired_medium:  list[NewsEvent] = field(default_factory=list)

    def consensus_direction(self, symbol: str) -> int:
        """Net consensus direction from recent HIGH events for this symbol."""
        votes = [e.expected_direction_for(symbol) for e in self.fired_high]
        votes = [v for v in votes if v != 0]
        if not votes:
            return 0
        pos = sum(1 for v in votes if v > 0)
        neg = sum(1 for v in votes if v < 0)
        if pos > neg:
            return 1
        if neg > pos:
            return -1
        return 0

    def score_modifier(self, symbol: str, signal_dir: int) -> int:
        """Score +1 if recent macro consensus agrees with signal direction; 0 otherwise.
        Never returns negative — this gate is an AMPLIFIER only, not a blocker."""
        net = self.consensus_direction(symbol)
        if net != 0 and net == signal_dir:
            return 1
        return 0

    # Keywords that identify a central bank rate decision — JP mentor v7:
    # "Do you want to surf in the tsunami wave?" → hard block on FOMC/rate days.
    _RATE_KEYWORDS = (
        "fomc", "federal funds", "interest rate decision", "interest rate statement",
        "rate decision", "monetary policy statement", "fed rate", "boe rate",
        "ecb rate", "boj rate", "reserve bank",
    )

    def is_rate_decision_window(self, window_min: float = 240.0) -> bool:
        """True if a central bank rate decision fires within window_min minutes.

        Any upcoming HIGH event whose name matches _RATE_KEYWORDS is treated as a
        tsunami event — block entries entirely per JP mentor v7 guidance.
        """
        all_events = list(self.upcoming_high) + list(self.fired_high)
        for e in all_events:
            name_low = e.name.lower()
            if any(kw in name_low for kw in self._RATE_KEYWORDS):
                if not e.has_fired and e.minutes_until <= window_min:
                    return True
                if e.has_fired and e.minutes_since <= 60.0:
                    return True
        return False

    def upcoming_summary(self) -> str:
        """Human-readable string of upcoming HIGH events."""
        if not self.upcoming_high:
            return ""
        return "; ".join(
            f"{e.name} [{e.currency}] in {e.minutes_until:.0f}min"
            for e in self.upcoming_high
        )

    def fired_summary(self, symbol: str) -> str:
        """Human-readable string of recent HIGH events with their expected direction."""
        if not self.fired_high:
            return ""
        parts = []
        for e in self.fired_high:
            d = e.expected_direction_for(symbol)
            arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
            parts.append(f"{e.name} {arrow}")
        return "; ".join(parts)


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_value(s: Optional[str]) -> Optional[float]:
    if not s or s.strip() in ("", "—", "N/A"):
        return None
    s = s.strip().replace(",", "")
    mult = 1.0
    if s.endswith("B"):
        s = s[:-1]; mult = 1e9
    elif s.endswith("K"):
        s = s[:-1]; mult = 1e3
    elif s.endswith("M"):
        s = s[:-1]; mult = 1e6
    s = s.rstrip("%")
    try:
        return float(s) * mult
    except (ValueError, TypeError):
        return None


def _parse_events(raw: list[dict]) -> list[NewsEvent]:
    events: list[NewsEvent] = []
    for item in raw:
        impact_raw = item.get("impact", "").strip()
        if impact_raw not in ("High", "Medium", "Low"):
            continue

        date_str = item.get("date", "")
        try:
            if "T" in date_str:
                # ISO format with timezone: "2026-07-14T08:30:00-04:00"
                dt = datetime.fromisoformat(date_str).astimezone(timezone.utc)
            else:
                # Fallback: "Jul 14, 2026" + separate "time" field
                time_str = item.get("time", "12:00am")
                combined = f"{date_str} {time_str}"
                dt = datetime.strptime(combined, "%b %d, %Y %I:%M%p").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        events.append(NewsEvent(
            impact     = impact_raw,
            event_time = dt,
            currency   = item.get("country", "").upper(),
            name       = item.get("title", ""),
            prev       = _parse_value(item.get("previous")),
            forecast   = _parse_value(item.get("forecast")),
        ))

    return events


# ── News gate ─────────────────────────────────────────────────────────────────

class NewsGate:
    """Thread-safe economic calendar cache. Amplifies entries on macro consensus alignment."""

    def __init__(self, refresh_secs: int = _REFRESH_SECS):
        self._refresh_secs    = refresh_secs
        self._events:    list[NewsEvent] = []
        self._lock            = threading.Lock()
        self._last_fetch      = 0.0
        self._active_backend  = _CALENDAR_BACKENDS[0][0]
        self._thread:    Optional[threading.Thread] = None
        self._stop            = threading.Event()

    def start(self) -> None:
        self._fetch()
        self._thread = threading.Thread(
            target=self._refresh_loop, name="NewsGate", daemon=True
        )
        self._thread.start()
        logger.info("[NewsGate] Started — refreshing every %ds", self._refresh_secs)

    def stop(self) -> None:
        self._stop.set()

    def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._refresh_secs)
            if not self._stop.is_set():
                self._fetch()

    def _fetch(self) -> None:
        for backend_name, url in _CALENDAR_BACKENDS:
            try:
                req  = urllib.request.Request(url, headers={"User-Agent": "AiDEN-NewsGate/1.0"})
                resp = urllib.request.urlopen(req, timeout=10)
                raw  = json.loads(resp.read())
                events = _parse_events(raw)
                with self._lock:
                    self._events     = events
                    self._last_fetch = time.time()
                    self._active_backend = backend_name
                logger.info("[NewsGate] Fetched %d events via %s", len(events), backend_name)
                return
            except Exception as exc:
                logger.warning("[NewsGate] Backend %s failed: %s", backend_name, exc)
        logger.warning("[NewsGate] All backends failed — using stale cache (%d events)", len(self._events))

    def get_context(
        self,
        fired_window_min:    float = 120.0,
        upcoming_window_min: float = 60.0,
    ) -> NewsContext:
        with self._lock:
            events = list(self._events)

        fired_high   = []
        upcoming_high = []
        fired_medium  = []

        for e in events:
            mins_since = e.minutes_since
            mins_until = e.minutes_until

            if e.has_fired and 0 <= mins_since <= fired_window_min:
                if e.impact == "High":
                    fired_high.append(e)
                elif e.impact == "Medium":
                    fired_medium.append(e)

            if not e.has_fired and 0 <= mins_until <= upcoming_window_min:
                if e.impact == "High":
                    upcoming_high.append(e)

        return NewsContext(
            fired_high   = fired_high,
            upcoming_high = upcoming_high,
            fired_medium  = fired_medium,
        )
