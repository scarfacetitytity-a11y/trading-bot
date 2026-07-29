"""Market-structure detectors — H4 bias, order blocks, BOS, swings.

Moved verbatim from strategies/aiden_index.py and execution/signal_detectors.py
(Phase 0 extraction — zero behavior change).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── H4 bias ───────────────────────────────────────────────────────────────────
# (from strategies/aiden_index.py)

def _compute_h4_bias_ema(
    h4: pd.DataFrame, fast_n: int, slow_n: int
) -> tuple[pd.Series, pd.Series]:
    """Returns (bias_series, ema_spread_series).

    bias:   1=bullish, -1=bearish, 0=neutral
    spread: (fast-slow)/slow as fraction — magnitude = trend strength
    """
    close  = h4["close"]
    fast   = close.ewm(span=fast_n, adjust=False).mean()
    slow   = close.ewm(span=slow_n, adjust=False).mean()
    spread = (fast - slow) / slow.replace(0, np.nan)

    bias = pd.Series(0, index=h4.index, dtype=int)
    bias[(fast > slow) & (close > fast)] = 1
    bias[(fast < slow) & (close < fast)] = -1
    return bias, spread


def _compute_h4_bias_swing(h4: pd.DataFrame, lookback: int) -> tuple[pd.Series, pd.Series]:
    bias  = pd.Series(0, index=h4.index, dtype=int)
    highs = h4["high"].values
    lows  = h4["low"].values
    mid   = lookback // 2

    for i in range(lookback, len(h4)):
        w_hi = highs[i - lookback:i]
        w_lo = lows[i  - lookback:i]
        if w_hi[mid:].max() > w_hi[:mid].max() and w_lo[mid:].min() > w_lo[:mid].min():
            bias.iloc[i] = 1
        elif w_hi[mid:].max() < w_hi[:mid].max() and w_lo[mid:].min() < w_lo[:mid].min():
            bias.iloc[i] = -1

    spread = pd.Series(0.0, index=h4.index)  # swing method has no spread metric
    return bias, spread


# ── Order blocks ──────────────────────────────────────────────────────────────
# (from strategies/aiden_index.py)

def _find_bullish_ob(open_, close, high, low, start_i, lookback):
    """Bullish order block: bearish candle with a lower wick, followed by a bullish
    engulfing candle whose body fully covers the prior candle's body.

    JP mentor v7: "There has to be some wick sticking out here. It needs to touch a red
    wick and then the next candle's body swallows that candle — that becomes an order block."

    Returns (ob_low, ob_high) — the body zone of the bearish OB candle — or (None, None).
    """
    for j in range(start_i, max(0, start_i - lookback), -1):
        if j + 1 > start_i:
            continue
        ob_o = float(open_.iloc[j]); ob_c = float(close.iloc[j])
        ob_lo_w = float(low.iloc[j]); ob_hi_w = float(high.iloc[j])
        if ob_c >= ob_o:
            continue  # not a bearish candle
        has_lower_wick = ob_lo_w < min(ob_o, ob_c)
        if not has_lower_wick:
            continue
        # Check if the NEXT candle (j+1) is a bullish engulfing of the OB body
        nj = j + 1
        if nj > start_i:
            break
        next_o = float(open_.iloc[nj]); next_c = float(close.iloc[nj])
        if next_c <= next_o:
            continue  # next candle not bullish
        ob_body_lo = min(ob_o, ob_c); ob_body_hi = max(ob_o, ob_c)
        if next_o <= ob_body_lo and next_c >= ob_body_hi:
            return ob_body_lo, ob_body_hi
    return None, None


def _find_bearish_ob(open_, close, high, low, start_i, lookback):
    """Bearish order block: bullish candle with an upper wick, followed by a bearish
    engulfing candle whose body fully covers the prior candle's body.

    Mirror of _find_bullish_ob for short setups.
    Returns (ob_low, ob_high) — the body zone of the bullish OB candle — or (None, None).
    """
    for j in range(start_i, max(0, start_i - lookback), -1):
        if j + 1 > start_i:
            continue
        ob_o = float(open_.iloc[j]); ob_c = float(close.iloc[j])
        ob_lo_w = float(low.iloc[j]); ob_hi_w = float(high.iloc[j])
        if ob_c <= ob_o:
            continue  # not a bullish candle
        has_upper_wick = ob_hi_w > max(ob_o, ob_c)
        if not has_upper_wick:
            continue
        nj = j + 1
        if nj > start_i:
            break
        next_o = float(open_.iloc[nj]); next_c = float(close.iloc[nj])
        if next_c >= next_o:
            continue  # next candle not bearish
        ob_body_lo = min(ob_o, ob_c); ob_body_hi = max(ob_o, ob_c)
        if next_o >= ob_body_hi and next_c <= ob_body_lo:
            return ob_body_lo, ob_body_hi
    return None, None


# ── Prior battlefield (congestion CHoCH zone) ─────────────────────────────────
# (from strategies/aiden_index.py)

def _is_prior_battlefield(
    high: pd.Series, low: pd.Series, close: pd.Series,
    i: int, zone_lo: float, zone_hi: float, atr_val: float,
    lookback: int = 50,
) -> bool:
    """Return True if the zone [zone_lo, zone_hi] overlaps with a prior congestion
    area where there was a change of character (CHoCH).

    JP mentor v10: "I came back into a previous battlefield — a zone where bulls and
    bears have already fought. When price returns there it's a known reaction zone."

    Detection: scan back `lookback` bars for a run of 3+ consecutive bars where:
      1. Candle range < 0.5 × ATR (tight congestion — accumulation)
      2. The congestion midpoint is within the current FVG zone
    A prior CHoCH at/near the zone makes it a battlefield.
    """
    if i < lookback + 4 or atr_val <= 0:
        return False

    zone_mid = (zone_lo + zone_hi) / 2
    tol      = max((zone_hi - zone_lo) / 2, atr_val * 0.3)

    consecutive = 0
    for j in range(max(0, i - lookback), i - 2):
        bar_range = float(high.iloc[j]) - float(low.iloc[j])
        bar_mid   = (float(high.iloc[j]) + float(low.iloc[j])) / 2
        if bar_range < 0.5 * atr_val and abs(bar_mid - zone_mid) <= tol:
            consecutive += 1
            if consecutive >= 3:
                return True
        else:
            consecutive = 0
    return False


# ── Break of structure ────────────────────────────────────────────────────────
# (from strategies/aiden_index.py)

def _bos_bull(high: pd.Series, close: pd.Series, i: int, lookback: int = 20) -> tuple:
    """Break of structure long: close above prior N-bar swing high.

    Returns (swing_high, True) or (None, False).
    """
    if i < lookback + 2:
        return None, False
    swing_hi = float(high.iloc[i - lookback:i - 1].max())
    if float(close.iloc[i]) > swing_hi:
        return swing_hi, True
    return None, False


def _bos_bear(low: pd.Series, close: pd.Series, i: int, lookback: int = 20) -> tuple:
    """Break of structure short: close below prior N-bar swing low.

    Returns (swing_low, True) or (None, False).
    """
    if i < lookback + 2:
        return None, False
    swing_lo = float(low.iloc[i - lookback:i - 1].min())
    if float(close.iloc[i]) < swing_lo:
        return swing_lo, True
    return None, False


# ── FVG queue helper ──────────────────────────────────────────────────────────
# (from strategies/aiden_index.py)

def _alt_dup(active_fvgs: list, fvg_dir: str, anchor: float, atr_val: float) -> bool:
    """True if an equivalent alt setup already in queue (suppresses bar-by-bar re-add)."""
    return any(
        f["dir"] == fvg_dir and abs(f["fvg_lo"] - anchor) < 0.5 * atr_val
        for f in active_fvgs
    )


# ── Swing utilities ───────────────────────────────────────────────────────────
# (from execution/signal_detectors.py)

def _swing_highs(series: pd.Series, order: int = 3) -> pd.Series:
    """Boolean mask: True where series is the local max within `order` bars each side."""
    rolling_max = series.rolling(window=order * 2 + 1, center=True, min_periods=1).max()
    return series == rolling_max


def _swing_lows(series: pd.Series, order: int = 3) -> pd.Series:
    """Boolean mask: True where series is the local min within `order` bars each side."""
    rolling_min = series.rolling(window=order * 2 + 1, center=True, min_periods=1).min()
    return series == rolling_min
