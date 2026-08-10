"""Market phase detector — classifies where in the AMD cycle price currently sits.

Phases (Expansion → Retracement → Consolidation → Manipulation → Continuation):
  EXPANSION     — strong directional move, impulse candles, BOS confirmed
  RETRACEMENT   — pulling back toward OB/FVG zone after expansion; discount/premium
  CONSOLIDATION — range-bound, no structural bias, stand aside
  MANIPULATION  — liquidity sweep (false break of session high/low), reversal expected
  CONTINUATION  — post-manipulation resumption of HTF direction

Phase is computed on M15 bars using H4 context. The scenario engine reads the phase
to select the appropriate entry model (1/2/3) and required confirmations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    EXPANSION     = "expansion"
    RETRACEMENT   = "retracement"
    CONSOLIDATION = "consolidation"
    MANIPULATION  = "manipulation"
    CONTINUATION  = "continuation"
    UNKNOWN       = "unknown"


@dataclass
class PhaseResult:
    phase:        Phase
    confidence:   float        # 0-1
    h4_bias:      int          # +1 / -1 / 0
    sweep_dir:    int          # +1 (swept highs) / -1 (swept lows) / 0
    at_structure: bool         # price is at a known OB or FVG zone
    impulsive:    bool         # last N bars show strong directional move
    notes:        list[str]
    po3_phase:    str = "unknown"   # accumulation | manipulation | distribution | unknown


def detect_phase(
    df_m15:         pd.DataFrame,
    h4_bias:        int,
    atr:            float,
    lookback:       int   = 20,   # M15 bars to assess
    sweep_lookback: int   = 8,    # bars to look back for sweep
    impulse_atr:    float = 1.5,  # bar range > N*ATR = impulsive
    range_atr:      float = 0.4,  # bar range < N*ATR = consolidation candle
) -> PhaseResult:
    """Classify the current market phase from M15 bars + H4 context."""
    notes: list[str] = []

    if len(df_m15) < lookback + 5:
        return PhaseResult(Phase.UNKNOWN, 0.0, h4_bias, 0, False, False, ["insufficient bars"])

    recent  = df_m15.iloc[-(lookback + 1):-1]
    last    = df_m15.iloc[-2]
    cv      = float(df_m15["close"].iloc[-2])
    high_r  = recent["high"].values
    low_r   = recent["low"].values

    # ── 1. Sweep detection ────────────────────────────────────────────────────
    sweep_window = df_m15.iloc[-(sweep_lookback + 2):-2]
    prior_high   = float(sweep_window["high"].max())
    prior_low    = float(sweep_window["low"].min())
    bar_high     = float(last["high"])
    bar_low      = float(last["low"])
    bar_close    = float(last["close"])

    swept_highs = bar_high > prior_high and bar_close < prior_high
    swept_lows  = bar_low  < prior_low  and bar_close > prior_low
    sweep_dir   = 0
    if swept_highs:
        sweep_dir = 1
        notes.append(f"swept highs {prior_high:.5f}")
    elif swept_lows:
        sweep_dir = -1
        notes.append(f"swept lows {prior_low:.5f}")

    # ── 2. Impulse detection ──────────────────────────────────────────────────
    ranges       = recent["high"].values - recent["low"].values
    impulse_bars = int((ranges > impulse_atr * atr).sum())
    impulsive    = impulse_bars >= 2
    if impulsive:
        notes.append(f"{impulse_bars} impulse bars (>{impulse_atr}xATR)")

    # ── 3. Range detection ────────────────────────────────────────────────────
    range_bars = int((ranges < range_atr * atr).sum())
    in_range   = range_bars >= int(lookback * 0.6)
    if in_range:
        notes.append(f"{range_bars}/{lookback} small bars — consolidation")

    # ── 4. Directional structure ──────────────────────────────────────────────
    mid = lookback // 2
    hh  = high_r[-1] > high_r[-mid]
    hl  = low_r[-1]  > low_r[-mid]
    ll  = low_r[-1]  < low_r[-mid]
    lh  = high_r[-1] < high_r[-mid]

    # ── 5. Retracement zone (35–65% of lookback swing) ───────────────────────
    swing_h      = float(high_r.max())
    swing_l      = float(low_r.min())
    swing_rng    = swing_h - swing_l
    retrace_top  = swing_l + swing_rng * 0.65
    retrace_bot  = swing_l + swing_rng * 0.35
    at_retrace   = retrace_bot <= cv <= retrace_top and swing_rng > 0
    if at_retrace:
        notes.append(f"at 35-65% retrace ({retrace_bot:.5f}–{retrace_top:.5f})")

    at_structure = at_retrace

    # ── 6. Phase classification (priority: Manipulation > Expansion > Retrace)
    if swept_highs or swept_lows:
        phase      = Phase.MANIPULATION
        confidence = 0.75 if impulsive else 0.55

    elif impulsive and h4_bias != 0:
        bull_exp = h4_bias == 1  and hh and hl
        bear_exp = h4_bias == -1 and ll and lh
        if bull_exp or bear_exp:
            phase      = Phase.EXPANSION
            confidence = 0.70
        else:
            phase      = Phase.MANIPULATION
            confidence = 0.50
            notes.append("impulse against H4 bias")

    elif at_retrace and h4_bias != 0:
        phase      = Phase.RETRACEMENT
        confidence = 0.65

    elif in_range:
        phase      = Phase.CONSOLIDATION
        confidence = 0.60

    elif h4_bias != 0:
        phase      = Phase.CONTINUATION
        confidence = 0.45

    else:
        phase      = Phase.UNKNOWN
        confidence = 0.30

    # PO3 (Power of Three) AMD mapping:
    #   Accumulation → CONSOLIDATION (range building, stops loading)
    #   Manipulation → MANIPULATION (Judas Swing, liquidity sweep)
    #   Distribution → EXPANSION or CONTINUATION (true institutional move)
    _po3 = {
        Phase.CONSOLIDATION: "accumulation",
        Phase.MANIPULATION:  "manipulation",
        Phase.EXPANSION:     "distribution",
        Phase.CONTINUATION:  "distribution",
        Phase.RETRACEMENT:   "distribution",
    }.get(phase, "unknown")

    return PhaseResult(
        phase=phase,
        confidence=confidence,
        h4_bias=h4_bias,
        sweep_dir=sweep_dir,
        at_structure=at_structure,
        impulsive=impulsive,
        notes=notes,
        po3_phase=_po3,
    )
