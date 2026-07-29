"""Liquidity detectors — sweeps, sweep reversals, manipulation patterns.

Moved verbatim from strategies/aiden_index.py (Phase 0 extraction — zero
behavior change).
"""
from __future__ import annotations

import pandas as pd


# ── Liquidity sweeps ──────────────────────────────────────────────────────────

def _liq_swept_low(low: pd.Series, i: int, lookback: int) -> bool:
    """Wick below prior N-bar swing low — used for LONG setups."""
    if i < lookback + 2:
        return False
    swing_low = low.iloc[i - lookback:i - 1].min()
    return any(low.iloc[j] < swing_low for j in range(max(0, i - 5), i))


def _liq_swept_high(high: pd.Series, i: int, lookback: int) -> bool:
    """Wick above prior N-bar swing high — used for SHORT setups."""
    if i < lookback + 2:
        return False
    swing_high = high.iloc[i - lookback:i - 1].max()
    return any(high.iloc[j] > swing_high for j in range(max(0, i - 5), i))


# ── Sweep reversals ───────────────────────────────────────────────────────────

def _sweep_reversal_bull(
    close: pd.Series, low: pd.Series, i: int, lookback: int
) -> tuple:
    """Wick below prior swing low then close back above (bullish sweep reversal).

    Returns (swept_level, actual_wick_low) or (None, None).
    swept_level is the prior swing low that was taken.
    actual_wick_low is the extreme wick (used for SL placement).
    """
    if i < lookback + 3:
        return None, None
    prior_swing_lo = float(low.iloc[i - lookback:i - 1].min())
    swept = any(float(low.iloc[j]) < prior_swing_lo for j in range(max(0, i - 3), i))
    if not swept:
        return None, None
    if float(close.iloc[i]) <= prior_swing_lo:
        return None, None
    actual_lo = min(float(low.iloc[j]) for j in range(max(0, i - 3), i))
    return prior_swing_lo, actual_lo


def _sweep_reversal_bear(
    close: pd.Series, high: pd.Series, i: int, lookback: int
) -> tuple:
    """Wick above prior swing high then close back below (bearish sweep reversal).

    Returns (swept_level, actual_wick_high) or (None, None).
    """
    if i < lookback + 3:
        return None, None
    prior_swing_hi = float(high.iloc[i - lookback:i - 1].max())
    swept = any(float(high.iloc[j]) > prior_swing_hi for j in range(max(0, i - 3), i))
    if not swept:
        return None, None
    if float(close.iloc[i]) >= prior_swing_hi:
        return None, None
    actual_hi = max(float(high.iloc[j]) for j in range(max(0, i - 3), i))
    return prior_swing_hi, actual_hi


# ── Manipulation patterns ─────────────────────────────────────────────────────

def _manipulation_w(high: pd.Series, low: pd.Series, i: int, lookback: int = 20) -> bool:
    """Detect Manipulation W pattern (bullish reversal) within the last `lookback` bars.

    Structure: swing low → sweep below it (right shoulder wick) → close back above
    the prior swing low = change of character. JP mentor's W = lower-low wick that
    closes back up, indicating smart money swept retail longs then reversed.
    """
    if i < lookback + 4:
        return False
    window_lo = low.iloc[i - lookback:i + 1]
    window_hi = high.iloc[i - lookback:i + 1]
    # Find the lowest wick in the window (the sweep candle)
    sweep_idx = int(window_lo.argmin())
    if sweep_idx == 0 or sweep_idx >= lookback:
        return False
    sweep_low = float(window_lo.iloc[sweep_idx])
    # Prior swing low = min before the sweep
    prior_lo = float(window_lo.iloc[:sweep_idx].min())
    # Sweep must go below prior low (the manipulation)
    if sweep_low >= prior_lo:
        return False
    # Right shoulder: after the sweep, price makes a higher low (doesn't retake the sweep)
    post_lo = float(window_lo.iloc[sweep_idx + 1:].min())
    # Close of current bar must be above the prior swing low (change of character)
    current_close_above = float(high.iloc[i]) > prior_lo
    right_shoulder = post_lo > sweep_low
    return right_shoulder and current_close_above


def _manipulation_m(high: pd.Series, low: pd.Series, i: int, lookback: int = 20) -> bool:
    """Detect Manipulation M pattern (bearish reversal) within the last `lookback` bars.

    Structure: swing high → sweep above it → close back below = change of character.
    """
    if i < lookback + 4:
        return False
    window_hi = high.iloc[i - lookback:i + 1]
    window_lo = low.iloc[i - lookback:i + 1]
    sweep_idx = int(window_hi.argmax())
    if sweep_idx == 0 or sweep_idx >= lookback:
        return False
    sweep_high = float(window_hi.iloc[sweep_idx])
    prior_hi   = float(window_hi.iloc[:sweep_idx].max())
    if sweep_high <= prior_hi:
        return False
    post_hi = float(window_hi.iloc[sweep_idx + 1:].max())
    current_close_below = float(low.iloc[i]) < prior_hi
    right_shoulder = post_hi < sweep_high
    return right_shoulder and current_close_below
