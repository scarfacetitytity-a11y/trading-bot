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


# ── Internal: swing detection ─────────────────────────────────────────────────

def _swing_highs(series: pd.Series, order: int = 3) -> pd.Series:
    """Boolean mask: True where series is the local max within `order` bars each side."""
    rolling_max = series.rolling(window=order * 2 + 1, center=True, min_periods=1).max()
    return series == rolling_max


def _swing_lows(series: pd.Series, order: int = 3) -> pd.Series:
    """Boolean mask: True where series is the local min within `order` bars each side."""
    rolling_min = series.rolling(window=order * 2 + 1, center=True, min_periods=1).min()
    return series == rolling_min


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
