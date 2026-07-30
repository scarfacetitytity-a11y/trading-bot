"""AMD (Accumulation → Manipulation → Displacement) phase detector.

Three-phase market model:
  ACCUMULATION — price ranging, equal H/L forming, low momentum. Wait.
  MANIPULATION — fake move sweeps a liquidity pool, then reverses. The signal.
  DISPLACEMENT — real directional move opposite the sweep. The entry.

Accumulation detection:
  ATR relative to weekly range < threshold → ranging. Equal highs/lows forming
  at the range boundaries are the upcoming manipulation targets.

Manipulation / sweep detection:
  Bar dips below key level then closes back above → bullish (displacement UP).
  Bar spikes above key level then closes back below → bearish (displacement DOWN).

Used by the orchestrator to:
  1. Identify market phase per bar (accumulation / manipulation / displacement).
  2. Populate StackInput.sweep_present and eq_liq_cluster for the v3 stack.
  3. Hard-block counter-sweep signals (wrong side of manipulation).
  4. Score-boost aligned displacement entries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketPhase(str, Enum):
    ACCUMULATION  = "accumulation"
    MANIPULATION  = "manipulation"
    DISPLACEMENT  = "displacement"
    UNKNOWN       = "unknown"


@dataclass
class PhaseState:
    """Full AMD phase assessment for the current bar."""
    phase: MarketPhase = MarketPhase.UNKNOWN
    sweep: Optional["SweepEvent"] = None
    # Accumulation metrics
    range_compression: float = 0.0   # ATR / weekly_range (low = ranging)
    range_hi: float = 0.0
    range_lo: float = 0.0
    bars_in_range: int = 0
    # Displacement metrics (post-sweep)
    displacement_bars: int = 0       # bars since confirmed sweep
    displacement_confirmed: bool = False

    def __str__(self) -> str:
        if self.phase == MarketPhase.ACCUMULATION:
            return (f"ACCUMULATION (compression={self.range_compression:.2f} "
                    f"range={self.range_lo:.5g}–{self.range_hi:.5g} "
                    f"bars={self.bars_in_range})")
        if self.phase == MarketPhase.MANIPULATION and self.sweep:
            return f"MANIPULATION → {self.sweep}"
        if self.phase == MarketPhase.DISPLACEMENT and self.sweep:
            return (f"DISPLACEMENT {'UP' if self.sweep.direction == 1 else 'DOWN'} "
                    f"(bars_since_sweep={self.displacement_bars} "
                    f"confirmed={self.displacement_confirmed})")
        return f"{self.phase.value}"


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
    """Full AMD phase detector — Accumulation, Manipulation, Displacement.

    Per bar:
      assess_phase() → PhaseState with current market phase + any sweep event.
      detect()       → SweepEvent only (legacy, used by orchestrator sweep block).

    Phase logic:
      1. ACCUMULATION: ATR < compression_threshold * weekly_range AND price
         oscillating inside a tight band for >= min_range_bars.
      2. MANIPULATION: a sweep event is detected (bar beyond level + close back).
         Phase transitions from ACCUMULATION or UNKNOWN to MANIPULATION.
      3. DISPLACEMENT: 1–4 bars after a confirmed sweep, price is moving
         consistently in the displacement direction. Entry window.

    State is per-symbol — caller should create one AMDDetector per symbol or
    pass symbol to track last-sweep per-symbol.
    """

    # Accumulation: ATR must be < this fraction of the weekly range
    _COMPRESSION_THRESHOLD = 0.18
    # Minimum M15 bars in range to call it accumulation (8h = 32 bars)
    _MIN_RANGE_BARS = 24
    # Displacement: max bars after sweep that still counts as displacement window
    _MAX_DISPLACEMENT_BARS = 8

    def __init__(self) -> None:
        # Track last confirmed sweep per symbol for displacement phase
        self._last_sweep: dict[str, tuple[SweepEvent, int]] = {}  # symbol → (sweep, bar_idx)
        self._bar_idx: dict[str, int] = {}

    def assess_phase(
        self,
        symbol: str,
        df: pd.DataFrame,
        levels: list,
        atr: float,
        weekly_range: float = 0.0,
    ) -> PhaseState:
        """Full AMD phase assessment for the current bar.

        symbol: used to track displacement phase across calls.
        df: M15 OHLC, last row = current bar (may be forming).
        levels: KeyLevel | HTFZone list.
        atr: current ATR.
        weekly_range: high - low of current week. If 0, estimated from df.
        """
        idx = self._bar_idx.get(symbol, 0) + 1
        self._bar_idx[symbol] = idx

        state = PhaseState()

        if len(df) < _MIN_IMPULSE and atr <= 0:
            return state

        # ── Weekly range estimate if not provided ────────────────────────
        if weekly_range <= 0 and len(df) >= 96:   # ~24h of M15
            week_slice = df.iloc[-96:]
            weekly_range = float(week_slice["high"].max() - week_slice["low"].min())

        # ── 1. Accumulation check ────────────────────────────────────────
        accum = self._check_accumulation(df, atr, weekly_range)
        if accum is not None:
            state.phase           = MarketPhase.ACCUMULATION
            state.range_compression = accum["compression"]
            state.range_hi        = accum["hi"]
            state.range_lo        = accum["lo"]
            state.bars_in_range   = accum["bars"]

        # ── 2. Sweep / manipulation check ────────────────────────────────
        sweep = self.detect(df, levels, atr)
        if sweep is not None:
            state.phase = MarketPhase.MANIPULATION
            state.sweep = sweep
            self._last_sweep[symbol] = (sweep, idx)
            logger.info("[AMDDetector][%s] %s", symbol, state)
            return state

        # ── 3. Displacement check (post-sweep window) ────────────────────
        if symbol in self._last_sweep:
            last_sweep, last_idx = self._last_sweep[symbol]
            bars_since = idx - last_idx
            if 1 <= bars_since <= self._MAX_DISPLACEMENT_BARS:
                # Confirm momentum: last 2 completed bars close in sweep direction
                closes = df["close"].iloc[-3:-1]
                if len(closes) >= 2:
                    if last_sweep.direction == 1:
                        confirmed = float(closes.iloc[-1]) > float(closes.iloc[0])
                    else:
                        confirmed = float(closes.iloc[-1]) < float(closes.iloc[0])
                else:
                    confirmed = False

                state.phase                  = MarketPhase.DISPLACEMENT
                state.sweep                  = last_sweep
                state.displacement_bars       = bars_since
                state.displacement_confirmed  = confirmed
                logger.debug("[AMDDetector][%s] %s", symbol, state)
                return state
            elif bars_since > self._MAX_DISPLACEMENT_BARS:
                del self._last_sweep[symbol]

        if state.phase == MarketPhase.UNKNOWN and accum is None:
            state.phase = MarketPhase.UNKNOWN

        return state

    def _check_accumulation(
        self, df: pd.DataFrame, atr: float, weekly_range: float
    ) -> Optional[dict]:
        """Return accumulation metrics if price is ranging, else None."""
        if weekly_range <= 0 or atr <= 0:
            return None

        compression = atr / weekly_range
        if compression >= self._COMPRESSION_THRESHOLD:
            return None   # trending / volatile — not accumulating

        # Check if price has been oscillating in a tight band
        lookback = min(self._MIN_RANGE_BARS, len(df) - 2)
        if lookback < 8:
            return None

        window = df.iloc[-(lookback + 1):-1]
        hi = float(window["high"].max())
        lo = float(window["low"].min())
        band = hi - lo

        # Band must be narrow relative to ATR * lookback (price not trending)
        if band > atr * lookback * 0.35:
            return None

        return {"compression": round(compression, 3), "hi": hi, "lo": lo, "bars": lookback}


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
