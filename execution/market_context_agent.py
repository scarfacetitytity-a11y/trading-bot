"""MarketContextAgent — active structural intelligence layer.

Not a passive logger. This agent reads the key-level environment, decides
where price is in the structure, and outputs actionable trading directives:
  - direction_bias: +1 (favor longs), -1 (favor shorts), 0 (neutral)
  - entry_block: True if the proposed trade direction fights a major level
  - probability_lift: multiplier on base win probability (0.5 = halve it, 2.0 = double)
  - narrative: one-sentence market context for the log / council

Council members embedded:
  C02 Architect   — level classification and structural reading
  C06 App Engineer — gate logic and edge-case handling
  C11 Reality Gap  — warns when context contradicts the M15 signal
  C12 Devil's Advocate — always checks if the "obvious" level read is wrong

The agent is synchronous — it blocks entry while it computes. Speed is not
the issue; being at the wrong level costs R.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from execution.level_monitor import LevelMonitor, KeyLevel

logger = logging.getLogger(__name__)


# ── Level strength weights ─────────────────────────────────────────────────────
# Stronger level = more weight on the direction_bias and probability_lift.
# These are calibrated from market microstructure theory; updated from live data.
LEVEL_STRENGTH: dict[str, float] = {
    "Weekly":   2.0,   # weekly high/low — institutional range boundary
    "D1":       1.6,   # daily high/low — intraday S/R
    "NY":       1.3,   # prior NY session high/low
    "Asian":    1.1,   # Asian session range
    "London":   1.0,   # London 50% midpoint
    "H4":       1.4,   # H4 swing high/low
}

# Proximity tiers: how many ATR away is "at the level"
PROXIMITY_STRONG  = 0.5   # within 0.5 ATR — at the level
PROXIMITY_NEAR    = 1.5   # within 1.5 ATR — approaching the level


@dataclass
class MarketContext:
    """Output of MarketContextAgent.assess()."""
    direction_bias:    int    # +1 favor longs, -1 favor shorts, 0 neutral
    entry_block:       bool   # True = block the proposed entry
    probability_lift:  float  # multiplier on P(win); 1.0 = no effect
    at_level:          bool   # price is directly on a significant level
    level_name:        str    # which level (for logging)
    level_strength:    float  # 0-2 strength score
    narrative:         str    # one-sentence context
    council_notes:     list[str] = field(default_factory=list)


class MarketContextAgent:
    """Active structural intelligence. Consult before every entry.

    Usage:
        agent = MarketContextAgent(level_monitor)
        ctx = agent.assess(symbol, df, atr, proposed_direction)
        if ctx.entry_block:
            continue  # don't trade
        score_mult *= ctx.probability_lift
    """

    def __init__(self, level_monitor: Optional[LevelMonitor] = None):
        self._lm = level_monitor or LevelMonitor()

    def assess(
        self,
        symbol:    str,
        df:        pd.DataFrame,
        atr:       float,
        direction: int,     # proposed trade direction: +1 long, -1 short
    ) -> MarketContext:
        """Assess the structural environment and return a trading directive."""

        if df is None or len(df) < 5 or atr <= 0:
            return MarketContext(0, False, 1.0, False, "no_data", 0.0,
                                 "insufficient data — neutral")

        price = float(df["close"].iloc[-1])

        # Refresh level monitor
        try:
            approaching = self._lm.update(symbol, df, atr)
        except Exception as e:
            logger.debug("[MCAgent] level update failed: %s", e)
            approaching = []

        all_levels = self._lm.get_levels(symbol)
        if not all_levels:
            return MarketContext(0, False, 1.0, False, "no_levels", 0.0,
                                 "no key levels computed — neutral")

        # ── Find strongest nearby level ────────────────────────────────────────
        # Score each level by: proximity (closer = stronger) * level_type_weight
        best_level:   Optional[KeyLevel] = None
        best_score:   float = 0.0
        best_dist_r:  float = 999.0  # distance in ATR

        for lv in all_levels:
            dist_atr = abs(lv.price - price) / atr
            if dist_atr > PROXIMITY_NEAR:
                continue
            type_weight = LEVEL_STRENGTH.get(lv.timeframe, 1.0)
            # proximity score: 1.0 at exact level, 0 at PROXIMITY_NEAR boundary
            prox_score = max(0.0, 1.0 - dist_atr / PROXIMITY_NEAR)
            total = prox_score * type_weight
            if total > best_score:
                best_score  = total
                best_level  = lv
                best_dist_r = dist_atr

        if best_level is None:
            return MarketContext(0, False, 1.0, False, "in_range", 0.0,
                                 "price in open range — neutral, no level context")

        at_level   = best_dist_r <= PROXIMITY_STRONG
        lv_strength = LEVEL_STRENGTH.get(best_level.timeframe, 1.0)

        # ── Direction bias from the level ──────────────────────────────────────
        # A resistance level above price → favor shorts (bias = -1)
        # A support level below price    → favor longs  (bias = +1)
        # level.direction in LevelMonitor: +1 = resistance above, -1 = support below
        #
        # If proposed direction FIGHTS the level → block. Resistance + trying to long
        # = entering into a wall. That's the "buying at the Weekly High" problem.

        # lv.direction: +1 = this level is overhead resistance
        #               -1 = this level is below as support
        #               0  = neutral (range midpoint)

        if best_level.direction == +1:
            # Overhead resistance — level wants price to go DOWN
            level_bias = -1
        elif best_level.direction == -1:
            # Underside support — level wants price to go UP
            level_bias = +1
        else:
            level_bias = 0   # midpoint / neutral

        # Block the entry if: at a strong level AND direction fights it
        fighting_level = (level_bias != 0 and level_bias != direction and at_level)
        # Reduce probability if: approaching the level in the wrong direction
        approaching_against = (level_bias != 0 and level_bias != direction and not at_level)

        # ── Probability lift ───────────────────────────────────────────────────
        if at_level and level_bias == direction:
            # Trading WITH the level from exactly the right spot — maximum edge
            # e.g., short from weekly high resistance
            prob_lift = 1.0 + 0.5 * lv_strength   # 1.5 to 2.0x at strong level
        elif at_level and fighting_level:
            # Trading AGAINST the level — entering into the wall
            prob_lift = 0.3   # severe penalty; usually blocked anyway
        elif approaching_against:
            # Approaching but not at — level may absorb the move
            prob_lift = 0.6   # moderate penalty
        elif level_bias == direction:
            # Approaching with the level — partial benefit
            prob_lift = 1.0 + 0.2 * lv_strength
        else:
            prob_lift = 1.0   # neutral level

        # ── Council notes ──────────────────────────────────────────────────────
        council = []
        dist_str = f"{best_dist_r:.1f} ATR"
        lv_str   = f"{best_level.name} @ {best_level.price:.5g}"

        if fighting_level:
            council.append(
                f"[C02 Architect] BLOCKED: {direction:+d} entry at {lv_str} "
                f"({dist_str}) — this level is {_dir_name(best_level.direction)}, "
                f"not a long entry zone."
            )
            council.append(
                f"[C11 Reality Gap] M15 signal says {_dir_name(direction)} but "
                f"{lv_str} is hard {_dir_name(best_level.direction)}. "
                f"Live orders parked HERE are waiting to sell. Don't be the buyer."
            )
            council.append(
                f"[C12 Devil's Advocate] Is this actually a sweep of the level? "
                f"If price just tapped {lv_str} and rejected with strong M1 "
                f"reversal candle, a {_dir_name(-direction)} scalp from here "
                f"would be with the level, not against it."
            )
        elif at_level and level_bias == direction:
            council.append(
                f"[C02 Architect] AT KEY LEVEL: {lv_str} — entering {_dir_name(direction)} "
                f"with the structure. This is the cleanest entry zone."
            )
            council.append(
                f"[C06 App Engineer] prob_lift={prob_lift:.2f} — size up proportionally. "
                f"Being at a {best_level.timeframe} level is the edge; respect it."
            )
        elif approaching_against:
            council.append(
                f"[C11 Reality Gap] Approaching {lv_str} in the WRONG direction. "
                f"Trade may work short-term but {lv_str} will absorb the move."
            )

        # ── Narrative ─────────────────────────────────────────────────────────
        position = "AT" if at_level else f"{dist_str} from"
        if fighting_level:
            narrative = (f"BLOCKED — {position} {best_level.name} "
                         f"({best_level.timeframe} {_dir_name(best_level.direction)}), "
                         f"proposed {_dir_name(direction)} fights it")
        elif at_level and level_bias == direction:
            narrative = (f"CLEAN ENTRY — {position} {best_level.name} "
                         f"with structural bias {_dir_name(direction)}")
        elif approaching_against:
            narrative = (f"CAUTION — {position} {best_level.name}, "
                         f"approaching {_dir_name(direction)} into resistance")
        else:
            narrative = (f"{position} {best_level.name} — "
                         f"{'aligned' if level_bias == direction else 'neutral'}")

        logger.info("[MCAgent] %s | %s | lift=%.2f | block=%s",
                    symbol, narrative, prob_lift, fighting_level)
        for note in council:
            logger.info("[MCAgent] %s", note)

        return MarketContext(
            direction_bias   = level_bias,
            entry_block      = fighting_level,
            probability_lift = prob_lift,
            at_level         = at_level,
            level_name       = best_level.name,
            level_strength   = lv_strength,
            narrative        = narrative,
            council_notes    = council,
        )


def _dir_name(d: int) -> str:
    return "LONG" if d == 1 else ("SHORT" if d == -1 else "NEUTRAL")
