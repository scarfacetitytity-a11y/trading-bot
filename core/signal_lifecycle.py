"""Phase 3: SignalLifecycle — state machine for signal candidates.

Fixes the "pending signal never cancels" problem where a stale signal stays
live even after its zone is violated or the market structure shifts against it.

State machine per candidate:
  BORN      → First detected: score meets threshold, zone present
  WATCHING  → Awaiting M5 confirmation (pending trigger)
  ARMED     → M5 confirmed, entry order being sized
  FIRED     → Trade placed (terminal)
  INVALIDATED → Zone violated before entry (terminal)
  EXPIRED   → Max watch bars elapsed without firing (terminal)

Phase 3: AnalyzerEngine runs this shadow-only (dry-run).
Phase 4: Replaces TradingEngine's ad-hoc _pending_signal logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SignalState(str, Enum):
    BORN        = "born"
    WATCHING    = "watching"
    ARMED       = "armed"
    FIRED       = "fired"
    INVALIDATED = "invalidated"
    EXPIRED     = "expired"

    def is_terminal(self) -> bool:
        return self in (SignalState.FIRED, SignalState.INVALIDATED, SignalState.EXPIRED)


@dataclass
class SignalCandidate:
    symbol:       str
    direction:    int           # +1 long, -1 short
    state:        SignalState = SignalState.BORN
    born_time:    Optional[datetime] = None
    born_bar:     int = 0
    stack_score:  float = 0.0   # probability-stack score at birth
    base_score:   int   = 0     # raw strategy score
    score_floor:  int   = 0
    archetype:    str   = "liquidity"
    zone_lo:      float = 0.0   # zone being traded
    zone_hi:      float = 0.0
    zone_type:    str   = "fvg"

    # Lifecycle tracking
    watch_bars:   int = 0        # bars spent in WATCHING
    max_watch:    int = 8        # expire after this many bars without M5 fire
    m5_confirmed: bool = False
    zone_invalidated: bool = False

    notes: list[str] = field(default_factory=list)

    def advance(self, new_state: SignalState, note: str = "") -> None:
        if self.state.is_terminal():
            return
        old = self.state
        self.state = new_state
        msg = f"{self.symbol} dir={self.direction:+d}: {old.value}→{new_state.value}"
        if note:
            msg += f" ({note})"
        self.notes.append(msg)
        logger.debug("[SignalLifecycle] %s", msg)

    def to_dict(self) -> dict:
        return {
            "symbol":        self.symbol,
            "direction":     self.direction,
            "state":         self.state.value,
            "stack_score":   self.stack_score,
            "base_score":    self.base_score,
            "score_floor":   self.score_floor,
            "archetype":     self.archetype,
            "zone_type":     self.zone_type,
            "zone_lo":       self.zone_lo,
            "zone_hi":       self.zone_hi,
            "watch_bars":    self.watch_bars,
            "m5_confirmed":  self.m5_confirmed,
            "born_bar":      self.born_bar,
        }


class SignalLifecycleManager:
    """Manages all active SignalCandidates for a single symbol.

    Called once per M15 bar close:
      1. Invalidate candidates whose zones are violated
      2. Advance WATCHING candidates: increment watch_bars, expire if stale
      3. Return candidates ready for entry attempt (ARMED)
    """

    def __init__(self, symbol: str, max_watch_bars: int = 8) -> None:
        self.symbol         = symbol
        self.max_watch_bars = max_watch_bars
        self._candidates:   list[SignalCandidate] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, candidate: SignalCandidate) -> None:
        """Register a new BORN candidate."""
        if candidate.born_time is None:
            candidate.born_time = datetime.now(timezone.utc)
        candidate.max_watch = self.max_watch_bars
        self._candidates.append(candidate)
        logger.debug("[SLC] %s: new BORN candidate dir=%+d score=%.1f",
                     self.symbol, candidate.direction, candidate.stack_score)

    def tick(
        self,
        bar_idx:    int,
        violated_directions: set[int],  # directions whose zones were violated this bar
        m5_confirmed_directions: set[int],  # directions where M5 trigger fired
    ) -> list[SignalCandidate]:
        """Advance all candidates one bar. Returns ARMED candidates ready to enter."""
        armed = []
        for cand in self._candidates:
            if cand.state.is_terminal():
                continue

            # Invalidate if zone violated
            if cand.direction in violated_directions:
                cand.advance(SignalState.INVALIDATED, "zone violated")
                continue

            if cand.state == SignalState.BORN:
                cand.advance(SignalState.WATCHING)

            if cand.state == SignalState.WATCHING:
                cand.watch_bars += 1
                if cand.direction in m5_confirmed_directions:
                    cand.m5_confirmed = True
                    cand.advance(SignalState.ARMED, f"M5 confirmed bar+{cand.watch_bars}")
                    armed.append(cand)
                elif cand.watch_bars >= cand.max_watch:
                    cand.advance(SignalState.EXPIRED, f"no M5 in {cand.max_watch} bars")

        # Prune terminal candidates older than 24 bars
        self._candidates = [
            c for c in self._candidates
            if not c.state.is_terminal() or (bar_idx - c.born_bar) < 96
        ]
        return armed

    def mark_fired(self, direction: int) -> None:
        """Mark ARMED candidates in `direction` as FIRED."""
        for cand in self._candidates:
            if cand.state == SignalState.ARMED and cand.direction == direction:
                cand.advance(SignalState.FIRED)

    def active(self) -> list[SignalCandidate]:
        return [c for c in self._candidates if not c.state.is_terminal()]

    def all_candidates(self) -> list[SignalCandidate]:
        return list(self._candidates)
