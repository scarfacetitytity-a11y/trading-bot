"""NewsIntelligence — directional event scoring with decay and proximity penalty.

Wraps the existing NewsGate (which handles calendar fetching) and adds:
  - Event tier classification (TIER_1/2/3)
  - Strength decay: influence fades exponentially after the event fires
  - Proximity penalty: reduces score bonus as a high-impact event approaches
  - Richer FX directional mapping (GBPUSD, EURUSD, USDJPY)

Integration in orchestrator: replaces the flat score_modifier(+1) call
with get_signal() which returns a structured NewsSignal with a decayed
integer score_mod and metadata for logging.

Usage:
    intel = NewsIntelligence(news_gate)
    sig   = intel.get_signal("EURUSD", signal_dir=-1)
    signal_score += sig.score_mod
    if sig.upcoming_caution:
        logger.info("[EURUSD] Upcoming %s in %.0fmin — caution", sig.upcoming_name, sig.upcoming_min)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from news.event_taxonomy import (
    EventTier, classify_tier, event_direction_for,
    decayed_strength, proximity_penalty, TIER_SCORE,
)

if TYPE_CHECKING:
    from execution.news_gate import NewsGate, NewsEvent

logger = logging.getLogger(__name__)


@dataclass
class ScoredEvent:
    event:      "NewsEvent"
    tier:       EventTier
    direction:  int          # +1/-1/0 for this symbol
    strength:   float        # 0-1 decayed strength at query time
    score_contrib: int       # contribution to final score_mod


@dataclass
class NewsSignal:
    """Output of NewsIntelligence.get_signal()."""
    score_mod:        int    = 0
    strength:         float  = 0.0    # max event strength used
    dominant_event:   Optional[str] = None
    dominant_tier:    Optional[int] = None
    upcoming_caution: bool   = False  # True if a high-impact event is imminent
    upcoming_name:    Optional[str] = None
    upcoming_min:     float  = 0.0    # minutes until nearest upcoming caution event
    note:             str    = ""
    scored_events:    list[ScoredEvent] = field(default_factory=list)


class NewsIntelligence:
    """Scored, decayed, proximity-aware news signal for a given symbol + direction.

    Wraps a NewsGate. Does not own the gate's lifecycle — start/stop NewsGate
    separately, then pass it in here.
    """

    def __init__(self, gate: "NewsGate") -> None:
        self._gate = gate

    def get_signal(
        self,
        symbol:     str,
        signal_dir: int,
        fired_window_min:    float = 120.0,
        upcoming_window_min: float = 60.0,
    ) -> NewsSignal:
        """Compute the decayed news score modifier for this symbol+direction.

        Fired events:
          - Classify tier, compute decayed strength, map direction
          - Sum contributions: floor(strength * tier_score) per event
          - Cap at TIER_1 max (+2) so a pile of TIER_2 events can't massively inflate

        Upcoming events:
          - Detect imminent TIER_1/2 events and apply proximity penalty to score
          - Log caution note if within proximity window
        """
        ctx = self._gate.get_context(
            fired_window_min=fired_window_min,
            upcoming_window_min=upcoming_window_min,
        )

        # ── Score from fired events ───────────────────────────────────────────
        scored: list[ScoredEvent] = []
        for ev in ctx.fired_high + ctx.fired_medium:
            tier      = classify_tier(ev)
            direction = event_direction_for(ev, symbol)
            if direction == 0 or direction != signal_dir:
                continue  # irrelevant or opposing — no bonus
            strength = decayed_strength(tier, ev.minutes_since)
            raw_contrib = TIER_SCORE.get(int(tier), 0) * strength
            int_contrib = int(raw_contrib)  # floor: must be at full+ strength to score
            scored.append(ScoredEvent(
                event=ev, tier=tier, direction=direction,
                strength=strength, score_contrib=int_contrib,
            ))

        # Sort by contribution descending — dominant first
        scored.sort(key=lambda s: s.strength, reverse=True)

        raw_mod  = sum(s.score_contrib for s in scored)
        score_mod = min(raw_mod, 2)   # cap at +2 regardless of event pile-up

        dominant  = scored[0] if scored else None
        dom_name  = dominant.event.name if dominant else None
        dom_tier  = int(dominant.tier) if dominant else None
        dom_str   = dominant.strength if dominant else 0.0

        # ── Proximity penalty from upcoming events ────────────────────────────
        prox_mult = 1.0
        upcoming_caution = False
        upcoming_name    = None
        upcoming_min_val = float("inf")

        for ev in ctx.upcoming_high:
            tier  = classify_tier(ev)
            if tier == EventTier.TIER_3:
                continue
            mins  = ev.minutes_until
            p     = proximity_penalty(tier, mins)
            if p < prox_mult:
                prox_mult = p
                upcoming_caution = (p < 1.0)
                upcoming_name    = ev.name
                upcoming_min_val = mins

        if prox_mult < 1.0:
            score_mod = int(score_mod * prox_mult)   # dampen — can reduce to 0

        # ── Build note ────────────────────────────────────────────────────────
        parts = []
        if dom_name:
            parts.append(f"{dom_name} T{dom_tier} str={dom_str:.2f} +{score_mod}")
        if upcoming_caution:
            parts.append(f"caution: {upcoming_name} in {upcoming_min_val:.0f}min (x{prox_mult:.1f})")
        note = " | ".join(parts) if parts else "no relevant events"

        return NewsSignal(
            score_mod        = score_mod,
            strength         = dom_str,
            dominant_event   = dom_name,
            dominant_tier    = dom_tier,
            upcoming_caution = upcoming_caution,
            upcoming_name    = upcoming_name,
            upcoming_min     = upcoming_min_val if upcoming_caution else 0.0,
            note             = note,
            scored_events    = scored,
        )
