"""Momentum-type detectors — M5 entry trigger, accumulation, liquidity draw.

Moved verbatim from execution/signal_detectors.py (Phase 0 extraction — zero
behavior change). Swing utilities live in analysis.structure.
"""
from __future__ import annotations

import pandas as pd

from analysis.structure import _swing_highs, _swing_lows


# ── M5 Entry Trigger ──────────────────────────────────────────────────────────

def detect_m5_entry_trigger(
    df: pd.DataFrame,
    signal_dir: int,
    lookback: int = 30,
    order: int = 2,
    atr: float = 0.0,
    impulse_bars: int = 3,
) -> bool:
    """Return True when M5 confirms the signal direction.

    Two confirmation paths — whichever fires first:

    1. BOS path: close breaks below/above a confirmed M5 swing low/high (order=2).
       Catches structured moves with clear swing points.

    2. Impulse path: last `impulse_bars` M5 closes are all bearish/bullish AND
       each close is lower/higher than the previous close (consecutive momentum).
       Catches impulsive NY-session moves where price dumps/pumps without forming
       textbook swing structure before moving.
    """
    if df is None or len(df) < lookback:
        return True  # no M5 data — don't block the entry

    highs  = df["high"].iloc[-lookback:]
    lows   = df["low"].iloc[-lookback:]
    closes = df["close"].iloc[-lookback:]
    opens  = df["open"].iloc[-lookback:]

    # ── Impulse read ──────────────────────────────────────────────────────────
    # If the last N bars are all moving in signal direction with each close
    # extending lower/higher, the market is telling us clearly where it's going.
    recent_closes = closes.iloc[-impulse_bars:]
    recent_opens  = opens.iloc[-impulse_bars:]
    if signal_dir == -1:  # short impulse: all bars bearish + consecutive lower closes
        all_bearish   = all(c < o for c, o in zip(recent_closes, recent_opens))
        making_lows   = all(recent_closes.iloc[i] < recent_closes.iloc[i - 1]
                            for i in range(1, impulse_bars))
        if all_bearish and making_lows:
            return True
    else:  # long impulse: all bars bullish + consecutive higher closes
        all_bullish   = all(c > o for c, o in zip(recent_closes, recent_opens))
        making_highs  = all(recent_closes.iloc[i] > recent_closes.iloc[i - 1]
                            for i in range(1, impulse_bars))
        if all_bullish and making_highs:
            return True

    # ── BOS path ─────────────────────────────────────────────────────────────
    if signal_dir == 1:
        sh = _swing_highs(highs, order=order)
        valid = highs[sh].iloc[:-1]
        if valid.empty:
            return True
        last_swing_high = valid.iloc[-1]
        min_break = last_swing_high + (0.02 * atr if atr > 0 else 0)
        return float(closes.iloc[-1]) > min_break

    else:
        sl = _swing_lows(lows, order=order)
        valid = lows[sl].iloc[:-1]
        if valid.empty:
            return True
        last_swing_low = valid.iloc[-1]
        min_break = last_swing_low - (0.02 * atr if atr > 0 else 0)
        return float(closes.iloc[-1]) < min_break


# ── Accumulation / Distribution Detection ─────────────────────────────────────

def detect_accumulation(
    df: pd.DataFrame,
    direction: int,
    lookback: int = 30,
    touches: int = 3,
    atr: float = 0.0,
    atr_tolerance: float = 0.3,
) -> float:
    """Detect range-bound accumulation/distribution that opposes a planned entry.

    Returns a 0–1 risk score where 1.0 means "clear opposing accumulation —
    do not enter."

    For a SHORT entry (direction=-1): checks if price has bounced from the same
    low level 3+ times without breaking down. Multiple bounces = accumulation
    = likely long setup, not short.

    For a LONG entry (direction=1): checks if price has rejected the same high
    level 3+ times without breaking up. Multiple rejections = distribution =
    likely short setup, not long.
    """
    if df is None or len(df) < lookback:
        return 0.0

    tolerance = atr * atr_tolerance if atr > 0 else (df["high"].iloc[-lookback:].mean() * 0.001)

    if direction == -1:  # planning a SHORT — check for accumulation at lows
        lows = df["low"].iloc[-lookback:]
        sl   = _swing_lows(lows, order=3)
        swing_low_vals = lows[sl].values
        if len(swing_low_vals) < touches:
            return 0.0
        recent_low = swing_low_vals[-1]
        # Count how many prior swing lows are within tolerance of the recent low
        cluster = sum(1 for v in swing_low_vals[:-1] if abs(v - recent_low) <= tolerance)
        risk = min(1.0, cluster / (touches - 1))
        return risk

    else:  # planning a LONG — check for distribution at highs
        highs = df["high"].iloc[-lookback:]
        sh    = _swing_highs(highs, order=3)
        swing_high_vals = highs[sh].values
        if len(swing_high_vals) < touches:
            return 0.0
        recent_high = swing_high_vals[-1]
        cluster = sum(1 for v in swing_high_vals[:-1] if abs(v - recent_high) <= tolerance)
        risk = min(1.0, cluster / (touches - 1))
        return risk


# ── Liquidity Pool Detection ──────────────────────────────────────────────────

def detect_liquidity_draw(
    df: pd.DataFrame,
    signal_dir: int,
    lookback: int = 50,
    eq_tolerance_atr: float = 0.15,
    atr: float = 0.0,
) -> dict:
    """Detect equal highs/lows that represent liquidity pools.

    Equal highs above price = buy-side liquidity (price likely drawn up to sweep).
    Equal lows below price  = sell-side liquidity (price likely drawn down to sweep).

    Returns dict with:
      opposing_pool: float  — distance to nearest opposing liquidity pool (in ATR)
                              A large value means clear draw AGAINST the signal direction
      aligned_pool:  float  — distance to nearest aligned liquidity pool (target)
      block_entry:   bool   — True if opposing pool is very close (< 1 ATR) and
                              there is clear draw against the signal
    """
    if df is None or len(df) < lookback or atr <= 0:
        return {"opposing_pool": 999.0, "aligned_pool": 999.0, "block_entry": False}

    highs   = df["high"].iloc[-lookback:]
    lows    = df["low"].iloc[-lookback:]
    cur     = float(df["close"].iloc[-1])
    tol     = atr * eq_tolerance_atr

    sh = _swing_highs(highs, order=3)
    sl = _swing_lows(lows, order=3)

    eq_highs = []  # equal highs above current price
    eq_lows  = []  # equal lows below current price
    high_vals = highs[sh].values
    low_vals  = lows[sl].values

    # Find clusters of equal highs above price
    for i, h in enumerate(high_vals):
        if h <= cur:
            continue
        cluster = sum(1 for h2 in high_vals if abs(h2 - h) <= tol and h2 > cur)
        if cluster >= 2:
            eq_highs.append(h)

    # Find clusters of equal lows below price
    for l in low_vals:
        if l >= cur:
            continue
        cluster = sum(1 for l2 in low_vals if abs(l2 - l) <= tol and l2 < cur)
        if cluster >= 2:
            eq_lows.append(l)

    if signal_dir == 1:  # LONG — equal highs above = opposing liquidity (above = draw up = aligned!)
        # For a long: equal lows below us are opposing (price might sweep down first)
        # Equal highs above us are the target (buy-side liquidity = draw up = good)
        nearest_opposing = min((abs(cur - l) / atr for l in eq_lows), default=999.0)
        nearest_aligned  = min((abs(h - cur) / atr for h in eq_highs), default=999.0)
        # Block if equal lows are very close below (might sweep first)
        block = nearest_opposing < 0.5 and nearest_opposing < nearest_aligned
    else:  # SHORT — equal lows below = target. Equal highs above = draw against us
        nearest_opposing = min((abs(h - cur) / atr for h in eq_highs), default=999.0)
        nearest_aligned  = min((abs(cur - l) / atr for l in eq_lows), default=999.0)
        # Block if clear equal highs very close above (price likely drawn there first)
        block = nearest_opposing < 0.5 and nearest_opposing < nearest_aligned

    return {
        "opposing_pool": nearest_opposing,
        "aligned_pool":  nearest_aligned,
        "block_entry":   block,
    }
