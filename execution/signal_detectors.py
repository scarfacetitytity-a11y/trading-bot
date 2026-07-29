"""Signal detection functions for adaptive trade management.

Pure functions — no MT5 dependency. Takes pandas DataFrames, returns integer weights.
Designed to run on any timeframe; caller controls which TF data is passed.

M5 bars → full weight (caller multiplies by 1.0)
M1 bars → half weight (caller multiplies by 0.5)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


# ── Phase 0 extraction — moved verbatim to analysis/ ─────────────────────────
# Re-imported here so every existing import path (e.g. `from execution.signal_detectors
# import _swing_highs, detect_m5_entry_trigger`) keeps working unchanged.
from analysis.structure import _swing_highs, _swing_lows   # noqa: E402,F401
from analysis.momentum import (   # noqa: E402,F401
    detect_m5_entry_trigger, detect_accumulation, detect_liquidity_draw,
)


# ── CHoCH — Change of Character ───────────────────────────────────────────────

def detect_choch(
    df: pd.DataFrame,
    position_dir: int,
    lookback: int = 20,
    order: int = 3,
    atr: float = 0.0,
) -> int:
    """Price breaks the most recent validated swing against the position.

    Long: close < last swing low  → structure shifting bearish
    Short: close > last swing high → structure shifting bullish

    Returns 2 for a decisive break, 1 for a shallow break (< 0.12×ATR — likely sweep),
    0 if no CHoCH. Shallow-break penalization requires atr > 0.
    """
    if len(df) < lookback + order * 2 + 2:
        return 0

    recent = df.tail(lookback + order * 2).iloc[:-1]  # exclude current forming bar
    close  = float(df["close"].iloc[-1])

    if position_dir == 1:
        sl_mask  = _swing_lows(recent["low"], order=order)
        sl_vals  = recent["low"][sl_mask]
        if sl_vals.empty:
            return 0
        last_swing_low = float(sl_vals.iloc[-1])
        if close < last_swing_low:
            if atr > 0 and (last_swing_low - close) < 0.12 * atr:
                return 1  # shallow break — more likely sweep than reversal
            return 2

    elif position_dir == -1:
        sh_mask  = _swing_highs(recent["high"], order=order)
        sh_vals  = recent["high"][sh_mask]
        if sh_vals.empty:
            return 0
        last_swing_high = float(sh_vals.iloc[-1])
        if close > last_swing_high:
            if atr > 0 and (close - last_swing_high) < 0.12 * atr:
                return 1
            return 2

    return 0


# ── BOS — Break of Structure ──────────────────────────────────────────────────

def detect_bos(df: pd.DataFrame, position_dir: int, lookback: int = 40, order: int = 3) -> int:
    """Price breaks a major structural level — more severe than CHoCH.

    Looks for the swing low that preceded the last swing high (long) and checks if
    price has now broken below it. That level is the "trade thesis anchor."

    Returns 3 if BOS detected (trade thesis fully invalidated), 0 otherwise.
    """
    if len(df) < lookback + order * 2 + 2:
        return 0

    recent = df.tail(lookback + order * 2).iloc[:-1]
    close  = float(df["close"].iloc[-1])
    sh_mask = _swing_highs(recent["high"], order=order)
    sl_mask = _swing_lows(recent["low"],   order=order)
    sh_vals = recent["high"][sh_mask]
    sl_vals = recent["low"][sl_mask]

    if sh_vals.empty or sl_vals.empty:
        return 0

    if position_dir == 1:
        last_sh_loc  = sh_vals.index[-1]
        pre_sh_lows  = sl_vals[sl_vals.index < last_sh_loc]
        if pre_sh_lows.empty:
            return 0
        key_low = float(pre_sh_lows.iloc[-1])
        if close < key_low:
            return 3

    elif position_dir == -1:
        last_sl_loc   = sl_vals.index[-1]
        pre_sl_highs  = sh_vals[sh_vals.index < last_sl_loc]
        if pre_sl_highs.empty:
            return 0
        key_high = float(pre_sl_highs.iloc[-1])
        if close > key_high:
            return 3

    return 0


# ── Counter FVG ───────────────────────────────────────────────────────────────

def detect_counter_fvg(
    df: pd.DataFrame,
    position_dir: int,
    atr: float = 0.0,
    min_atr_mult: float = 0.08,
) -> int:
    """3-candle Fair Value Gap forming against the position.

    Bearish FVG (while long): gap between bar[-3].high and bar[-1].low — unfilled space
    Bullish FVG (while short): gap between bar[-3].low and bar[-1].high

    Only counts if gap >= min_atr_mult * ATR (filters noise).
    Returns 1 if counter FVG detected, 0 otherwise.
    """
    if len(df) < 3:
        return 0

    c1 = df.iloc[-3]
    c3 = df.iloc[-1]

    if position_dir == 1:
        gap = float(c1["high"]) - float(c3["low"])
        if gap > 0 and (atr <= 0 or gap >= min_atr_mult * atr):
            return 1

    elif position_dir == -1:
        gap = float(c3["high"]) - float(c1["low"])
        if gap > 0 and (atr <= 0 or gap >= min_atr_mult * atr):
            return 1

    return 0


# ── Momentum shift ────────────────────────────────────────────────────────────

def detect_momentum_shift(
    df: pd.DataFrame,
    position_dir: int,
    rsi_period: int = 14,
    lookback_bars: int = 6,
) -> int:
    """RSI-based momentum shift against the position.

    Long: RSI was above 52, now below 45 (momentum fading)
    Short: RSI was below 48, now above 55

    `lookback_bars` controls how far back to look for the "was above/below" condition.
    Returns 1 if shift detected, 0 otherwise.
    """
    if len(df) < rsi_period + lookback_bars + 2:
        return 0

    close = df["close"]
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_l = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan).fillna(1e10)
    rsi   = 100.0 - 100.0 / (1.0 + rs)

    cur  = float(rsi.iloc[-1])
    prev = float(rsi.iloc[-(lookback_bars + 1)])

    if position_dir == 1 and prev > 52 and cur < 45:
        return 1
    if position_dir == -1 and prev < 48 and cur > 55:
        return 1

    return 0


# ── Continuation: Higher Highs / Lower Lows ──────────────────────────────────

def detect_hh_ll(df: pd.DataFrame, position_dir: int, lookback: int = 10) -> int:
    """Price making new structure in the trade direction — continuation signal.

    Long: current high exceeds the highest high in the prior `lookback` bars
    Short: current low breaks below the lowest low in the prior `lookback` bars

    Returns 2 if confirmed, 0 otherwise.
    """
    if len(df) < lookback + 1:
        return 0

    current_bar = df.iloc[-1]
    prior       = df.iloc[-(lookback + 1):-1]

    if position_dir == 1:
        if float(current_bar["high"]) > float(prior["high"].max()):
            return 2

    elif position_dir == -1:
        if float(current_bar["low"]) < float(prior["low"].min()):
            return 2

    return 0


# ── Continuation: H4 bias alignment ──────────────────────────────────────────

def detect_htf_continuation(h4_bias: int, position_dir: int) -> int:
    """H4 trend bias still aligned with the open position.

    Returns 2 if aligned, 0 if unknown/neutral.
    Caller should apply -1 (counter) if H4 is known to oppose but neutral counts as 0.
    """
    return 2 if h4_bias == position_dir else 0


# ── Structural SL: tighter stop from recent swing ─────────────────────────────

def find_structure_sl(
    df: pd.DataFrame,
    position_dir: int,
    current_sl: float,
    lookback: int = 20,
    order: int = 3,
    buffer_mult: float = 0.1,
    atr: float = 0.0,
) -> Optional[float]:
    """Find the most recent swing level that would tighten the stop loss.

    Long: most recent swing low that is ABOVE current SL (tighter)
    Short: most recent swing high that is BELOW current SL (tighter)

    Adds a small buffer (buffer_mult * ATR) below the swing for slippage.
    Returns new SL price if found and tighter, None otherwise.
    """
    if len(df) < lookback + order * 2:
        return None

    recent  = df.tail(lookback + order * 2).iloc[:-1]
    buffer  = buffer_mult * atr if atr > 0 else 0.0

    if position_dir == 1:
        sl_mask = _swing_lows(recent["low"], order=order)
        sl_vals = recent["low"][sl_mask]
        # Only consider swing lows above current SL (would tighten)
        candidates = sl_vals[sl_vals > current_sl]
        if candidates.empty:
            return None
        new_sl = float(candidates.iloc[-1]) - buffer
        return new_sl if new_sl > current_sl else None

    elif position_dir == -1:
        sh_mask = _swing_highs(recent["high"], order=order)
        sh_vals = recent["high"][sh_mask]
        candidates = sh_vals[sh_vals < current_sl]
        if candidates.empty:
            return None
        new_sl = float(candidates.iloc[-1]) + buffer
        return new_sl if new_sl < current_sl else None

    return None


# ── Structural TP: next level for TP extension ────────────────────────────────

def find_next_structure_tp(
    df: pd.DataFrame,
    position_dir: int,
    current_tp: float,
    min_extension: float = 0.0,
    lookback: int = 60,
    order: int = 3,
) -> Optional[float]:
    """Find the next structural level beyond current TP for TP extension.

    Long: next swing high above current TP
    Short: next swing low below current TP

    Only returns a level if it extends TP by at least `min_extension` points.
    """
    if len(df) < lookback + order * 2:
        return None

    recent = df.tail(lookback + order * 2)

    if position_dir == 1:
        sh_mask   = _swing_highs(recent["high"], order=order)
        sh_vals   = recent["high"][sh_mask]
        above_tp  = sh_vals[sh_vals > current_tp + min_extension]
        if above_tp.empty:
            return None
        return float(above_tp.iloc[0])

    elif position_dir == -1:
        sl_mask   = _swing_lows(recent["low"], order=order)
        sl_vals   = recent["low"][sl_mask]
        below_tp  = sl_vals[sl_vals < current_tp - min_extension]
        if below_tp.empty:
            return None
        return float(below_tp.iloc[-1])

    return None


# ── Sweep / stop-hunt detection ───────────────────────────────────────────────

def detect_sweep_recovery(
    df: pd.DataFrame,
    position_dir: int,
    atr: float = 0.0,
    lookback: int = 20,
    order: int = 3,
    max_sweep_atr: float = 0.5,
    recover_lookback: int = 3,
) -> float:
    """Estimate probability that recent price action is a stop-hunt sweep, not reversal.

    Long: bar wicked below nearest swing low but closed above it (stops hunted, price recovered)
    Short: bar wicked above nearest swing high but closed below it

    Checks the current bar and the prior `recover_lookback` bars for the pattern.

    Returns sweep probability 0.0–1.0.
      High probability → reduce counter score weighting (likely sweep, not reversal).
      Uses depth of sweep (shallower = higher sweep prob) and recovery strength.
    """
    if len(df) < lookback + order * 2 + 2:
        return 0.0

    recent = df.tail(lookback + order * 2).iloc[:-1]

    if position_dir == 1:
        sl_mask = _swing_lows(recent["low"], order=order)
        sl_vals = recent["low"][sl_mask]
        if sl_vals.empty:
            return 0.0
        key_level = float(sl_vals.iloc[-1])

        # Check current bar and prior recover_lookback bars for sweep pattern
        swept_low   = None
        swept_close = None
        for offset in range(recover_lookback):
            bar = df.iloc[-(offset + 1)]
            lo  = float(bar["low"])
            cl  = float(bar["close"])
            if lo < key_level and cl > key_level:
                swept_low   = lo
                swept_close = cl
                break

        if swept_low is None:
            return 0.0

        sweep_depth = key_level - swept_low
        recovery    = swept_close - key_level

        depth_score    = (max(0.0, 1.0 - sweep_depth / (max_sweep_atr * atr))
                          if atr > 0 else 0.6)
        recovery_score = (min(1.0, recovery / max(atr * 0.1, 1e-10))
                          if atr > 0 else 0.6)

        return float(min(1.0, max(0.0, depth_score * 0.6 + recovery_score * 0.4)))

    elif position_dir == -1:
        sh_mask = _swing_highs(recent["high"], order=order)
        sh_vals = recent["high"][sh_mask]
        if sh_vals.empty:
            return 0.0
        key_level = float(sh_vals.iloc[-1])

        swept_high  = None
        swept_close = None
        for offset in range(recover_lookback):
            bar = df.iloc[-(offset + 1)]
            hi  = float(bar["high"])
            cl  = float(bar["close"])
            if hi > key_level and cl < key_level:
                swept_high  = hi
                swept_close = cl
                break

        if swept_high is None:
            return 0.0

        sweep_depth = swept_high - key_level
        recovery    = key_level - swept_close

        depth_score    = (max(0.0, 1.0 - sweep_depth / (max_sweep_atr * atr))
                          if atr > 0 else 0.6)
        recovery_score = (min(1.0, recovery / max(atr * 0.1, 1e-10))
                          if atr > 0 else 0.6)

        return float(min(1.0, max(0.0, depth_score * 0.6 + recovery_score * 0.4)))

    return 0.0


# detect_m5_entry_trigger, detect_accumulation, detect_liquidity_draw now live
# in analysis/momentum.py (re-imported above).
