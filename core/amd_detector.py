"""AMD (Accumulation → Manipulation → Displacement) sweep detector.

Reads the current bar + recent history to detect a liquidity sweep pattern:
- Manipulation (bullish): bar dips below a key level (sweeping sell-side liquidity),
  then closes back above it → displacement expected UPWARD.
- Manipulation (bearish): bar spikes above a key level (sweeping buy-side liquidity),
  then closes back below → displacement expected DOWNWARD.

This is the core of the JP mentor methodology: the sweep IS the signal. Any trade
taken IN THE DIRECTION of the sweep (selling into a bullish sweep) is wrong-side.
Any trade taken AGAINST the sweep direction (buying after a bullish sweep) is
the displacement trade.

Used by the orchestrator to:
  1. Populate StackInput.sweep_present and eq_liq_cluster for shadow logging.
  2. Hard-block signals that go against a confirmed sweep (wrong side of manipulation).
  3. Boost signals that align with a sweep (displacement entry = highest conviction).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SweepEvent:
    """A confirmed liquidity sweep + close-back pattern."""
    direction: int          # +1 = bullish sweep (lows taken → expect UP), -1 = bearish
    level_label: str        # which level was swept
    swept_price: float      # the level price
    bars_ago: int           # how many bars back the sweep bar was (1 = last closed bar)
    eq_liq: bool            # True if the swept level is an equal-H/L cluster
    impulse_ratio: float    # (close_past_level) / atr — strength of the close-back move
    strong: bool = False    # True if impulse_ratio >= STRONG_THRESHOLD

    def __str__(self) -> str:
        strength = "STRONG" if self.strong else "WEAK"
        side = "BULL" if self.direction == 1 else "BEAR"
        return (f"SweepEvent({side} {strength} @ {self.swept_price:.5g} "
                f"[{self.level_label}] {self.bars_ago}b ago "
                f"impulse={self.impulse_ratio:.2f} eq_liq={self.eq_liq})")


# Minimum close-past-level distance (as ATR multiple) to count as a sweep close-back
_MIN_IMPULSE   = 0.08
# Impulse above this → "strong" sweep — hard block on counter-direction; boost on aligned
_STRONG_THRESH = 0.40
# Max bars back to look for a sweep bar
_LOOKBACK      = 4


class AMDDetector:
    """Detects liquidity sweep + displacement pattern on M15 bar data.

    Call detect() on each new bar. It checks the last _LOOKBACK bars for a
    bar that swept below (bullish) or above (bearish) any key level in the
    provided level_set, with price now closed back past the swept level.

    Returns the highest-conviction SweepEvent found, or None.
    """

    def detect(
        self,
        df: pd.DataFrame,
        levels: list,        # list of KeyLevel | HTFZone (any with .price, .label, .direction)
        atr: float,
        lookback: int = _LOOKBACK,
    ) -> Optional[SweepEvent]:
        """Return the strongest sweep event visible in the last `lookback` bars, or None.

        df: M15 OHLC dataframe (rows = bars, last row = current forming bar).
            We look at iloc[-lookback-1 : -1] (completed bars only).
        levels: list of KeyLevel or HTFZone objects with .price, .label, .direction.
        atr: current ATR for impulse ratio normalisation.
        """
        if len(df) < lookback + 2 or atr <= 0 or not levels:
            return None

        # Use only completed bars (exclude the currently forming bar)
        completed = df.iloc[-(lookback + 1): -1]
        current_close = float(df["close"].iloc[-2])   # last completed bar close
        current_low   = float(df["low"].iloc[-2])
        current_high  = float(df["high"].iloc[-2])

        best: Optional[SweepEvent] = None
        best_impulse = _MIN_IMPULSE

        for item in levels:
            level_price  = float(item.price)
            label        = str(item.label)
            level_dir    = int(getattr(item, "direction", 0))
            is_eq_liq    = getattr(item, "zone_type", None) in ("eqh", "eql")

            # ── Bullish sweep: some bar swept BELOW level_price, current close is ABOVE ──
            # Allowed when level has bullish or neutral direction (don't check bearish OBs for bullish)
            if level_dir >= 0:
                for i in range(len(completed)):
                    bar = completed.iloc[i]
                    if float(bar["low"]) < level_price and current_close > level_price:
                        impulse = (current_close - level_price) / atr
                        if impulse > best_impulse:
                            best_impulse = impulse
                            best = SweepEvent(
                                direction=+1,
                                level_label=label,
                                swept_price=level_price,
                                bars_ago=len(completed) - i,
                                eq_liq=is_eq_liq,
                                impulse_ratio=round(impulse, 3),
                                strong=impulse >= _STRONG_THRESH,
                            )

            # ── Bearish sweep: some bar swept ABOVE level_price, current close is BELOW ──
            if level_dir <= 0:
                for i in range(len(completed)):
                    bar = completed.iloc[i]
                    if float(bar["high"]) > level_price and current_close < level_price:
                        impulse = (level_price - current_close) / atr
                        if impulse > best_impulse:
                            best_impulse = impulse
                            best = SweepEvent(
                                direction=-1,
                                level_label=label,
                                swept_price=level_price,
                                bars_ago=len(completed) - i,
                                eq_liq=is_eq_liq,
                                impulse_ratio=round(impulse, 3),
                                strong=impulse >= _STRONG_THRESH,
                            )

        if best is not None:
            logger.debug("AMDDetector: %s", best)

        return best
