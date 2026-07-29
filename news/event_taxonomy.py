"""Event taxonomy — classify NewsEvents into tiers, categories, and directional impacts.

Extends the directional logic in execution/news_gate.py with:
  - Tier classification (1=market-moving, 2=notable, 3=noise)
  - Strength decay half-lives per tier
  - Richer category coverage (employment, housing, sentiment, central bank)
  - FX pair directional mapping (GBPUSD, EURUSD, USDJPY)
"""
from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from execution.news_gate import NewsEvent


class EventTier(IntEnum):
    TIER_1 = 1   # NFP, CPI prints, GDP advance, FOMC decision
    TIER_2 = 2   # ISM, Retail Sales, PPI, PCE, ADP, PMI flash
    TIER_3 = 3   # Secondary releases — contextual only


# Score bonus at full strength per tier (decays over time)
TIER_SCORE: dict[int, int] = {1: 2, 2: 1, 3: 0}

# Decay rate k in strength = base * exp(-k * minutes_since)
# Half-life: TIER_1=60min, TIER_2=30min, TIER_3=15min
TIER_DECAY_RATE: dict[int, float] = {
    1: math.log(2) / 60,
    2: math.log(2) / 30,
    3: math.log(2) / 15,
}

TIER_BASE_STRENGTH: dict[int, float] = {1: 1.0, 2: 0.75, 3: 0.4}

# ── Keyword → tier mapping ────────────────────────────────────────────────────

_TIER1_KEYWORDS = [
    "nonfarm payroll", "nfp", "non-farm payroll",
    "consumer price index", " cpi ",
    "core cpi", "cpi y/y", "cpi m/m",
    "federal funds rate", "fomc statement", "interest rate decision",
    "gdp q/q", "gdp advance", "gdp preliminary",
    "retail sales",  # high-impact only when impact==High already
    "unemployment rate",
    "core pce",
]

_TIER2_KEYWORDS = [
    "ppi", "producer price",
    "pce", "personal consumption",
    "adp nonfarm", "adp employment",
    "ism manufacturing", "ism services", "ism non-manufacturing",
    "pmi",
    "durable goods",
    "industrial production",
    "trade balance",
    "gdp",                        # revisions, not advance
    "initial jobless claims", "unemployment claims",
    "consumer confidence", "consumer sentiment",
    "business confidence", "business climate",
    "retail",
    "philly fed", "empire state",
    "bank of england", "ecb press", "boj",
    "monetary policy",
    "building permits", "housing starts", "existing home", "new home",
]

# ── Symbol directional mapping ────────────────────────────────────────────────
# Returns +1 (bullish for symbol) or -1 (bearish) or 0 (no clear impact)
# given an event and whether the consensus surprised to the upside (beat=True).

_METALS          = {"XAUUSD", "XAGUSD"}
_USD_INSTRUMENTS = {"XAUUSD", "XAGUSD", "US30.cash", "US100.cash", "US500.cash", "US2000.cash"}
_GBP_INSTRUMENTS = {"UK100.cash", "GBPUSD"}
_JPY_INSTRUMENTS = {"JP225.cash", "USDJPY"}
_EUR_INSTRUMENTS = {"EURUSD"}

_USD_INFLATION_KW  = ["cpi", "ppi", "pce", "core inflation", "inflation", "import price"]
_USD_GROWTH_KW     = [
    "nonfarm", "nfp", "gdp", "retail", "ism", "adp", "unemployment",
    "durable", "industrial production", "consumer confidence", "philly fed", "empire state",
]
_USD_EMPLOYMENT_KW = ["nonfarm", "nfp", "adp", "unemployment rate", "jobless claims"]


def classify_tier(event: "NewsEvent") -> EventTier:
    """Assign an EventTier based on event name + impact level."""
    name = event.name.lower()
    # Only High-impact events can be TIER_1 or TIER_2
    if event.impact == "High":
        for kw in _TIER1_KEYWORDS:
            if kw in name:
                return EventTier.TIER_1
        for kw in _TIER2_KEYWORDS:
            if kw in name:
                return EventTier.TIER_2
        return EventTier.TIER_2   # unrecognised High = still notable
    if event.impact == "Medium":
        for kw in _TIER2_KEYWORDS:
            if kw in name:
                return EventTier.TIER_2
        return EventTier.TIER_3
    return EventTier.TIER_3


def event_direction_for(event: "NewsEvent", symbol: str) -> int:
    """Directional impact of an event on a symbol (+1/-1/0).

    Extends NewsEvent.expected_direction_for() with FX pair coverage and
    finer category sub-routing (inflation vs growth vs employment).
    """
    ccy      = event.currency.upper()
    name     = event.name.lower()
    beat     = event.consensus_direction == 1  # Fcst > Prev

    if event.consensus_direction == 0:
        return 0   # no consensus data

    # ── USD events ────────────────────────────────────────────────────────────
    if ccy == "USD":
        inflation_hit = any(kw in name for kw in _USD_INFLATION_KW)
        growth_hit    = any(kw in name for kw in _USD_GROWTH_KW)

        if symbol in _METALS:
            # Gold/silver: inverse USD strength
            if inflation_hit or growth_hit:
                return -1 if beat else 1

        elif symbol in {"US30.cash", "US100.cash", "US500.cash", "US2000.cash"}:
            if inflation_hit:
                # Higher inflation → rate-hike fear → bearish for equities
                return -1 if beat else 1
            if growth_hit:
                # Stronger growth → risk-on → bullish
                return 1 if beat else -1

        elif symbol == "GBPUSD":
            # USD side: stronger USD = lower GBPUSD
            if inflation_hit or growth_hit:
                return -1 if beat else 1

        elif symbol == "EURUSD":
            # USD side: stronger USD = lower EURUSD
            if inflation_hit or growth_hit:
                return -1 if beat else 1

        elif symbol == "USDJPY":
            # USD side: stronger USD = higher USDJPY
            if inflation_hit or growth_hit:
                return 1 if beat else -1

    # ── GBP events ────────────────────────────────────────────────────────────
    elif ccy == "GBP":
        gbp_growth_kw    = ["gdp", "retail", "pmi", "employment", "trade", "production"]
        gbp_inflation_kw = ["cpi", "inflation", "rpi"]

        if symbol == "UK100.cash":
            if any(kw in name for kw in gbp_growth_kw):
                return 1 if beat else -1
            if any(kw in name for kw in gbp_inflation_kw):
                return -1 if beat else 1   # rate-hike fear

        elif symbol == "GBPUSD":
            if any(kw in name for kw in gbp_growth_kw + gbp_inflation_kw):
                return 1 if beat else -1   # stronger GBP data = GBPUSD up

    # ── EUR events ────────────────────────────────────────────────────────────
    elif ccy == "EUR":
        eur_kw = ["gdp", "cpi", "pmi", "ifo", "sentiment", "retail", "employment", "ecb"]
        if symbol == "EURUSD":
            if any(kw in name for kw in eur_kw):
                return 1 if beat else -1

    # ── JPY events ────────────────────────────────────────────────────────────
    elif ccy == "JPY":
        jpy_kw = ["cpi", "gdp", "pmi", "trade", "industrial"]
        if symbol == "JP225.cash":
            # JPY strength bearish for Nikkei (export companies hurt)
            if any(kw in name for kw in jpy_kw):
                return -1 if beat else 1
        elif symbol == "USDJPY":
            # Stronger JPY data → JPY up → USDJPY down
            if any(kw in name for kw in jpy_kw):
                return -1 if beat else 1

    return 0


def decayed_strength(tier: EventTier, minutes_since: float) -> float:
    """Strength of an event's influence at `minutes_since` minutes after firing.

    Returns 0-1. Goes to near-zero by 2-3 half-lives.
    """
    if minutes_since < 0:
        return 0.0   # hasn't fired yet
    k    = TIER_DECAY_RATE[int(tier)]
    base = TIER_BASE_STRENGTH[int(tier)]
    return base * math.exp(-k * minutes_since)


def proximity_penalty(tier: EventTier, minutes_until: float) -> float:
    """Score multiplier (0-1) for an upcoming event at `minutes_until` minutes away.

    Reduces the entry score bonus as a high-impact event approaches — we're
    about to get real information, so adding to a position now is premature.
    Returns 1.0 (no penalty) if event is far away or TIER_3.
    """
    if tier == EventTier.TIER_3:
        return 1.0
    if tier == EventTier.TIER_1:
        if minutes_until <= 15:
            return 0.0    # suppress bonus — too close to print
        if minutes_until <= 30:
            return 0.5
        if minutes_until <= 60:
            return 0.75
    if tier == EventTier.TIER_2:
        if minutes_until <= 15:
            return 0.5
        if minutes_until <= 30:
            return 0.75
    return 1.0
