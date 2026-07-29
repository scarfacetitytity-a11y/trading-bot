"""AiDEN Index Strategy v2 — bidirectional FVG + OB confluence, all sessions.

Cadre:
  Sage   (MEM-001) — bidirectional gate: H4 EMA bullish = longs only, bearish = shorts only
  Quant  (MEM-002) — dynamic RR: Model 3 + trend strength extend target to 3.5-5.0R
  Builder(MEM-003) — implementation, shorts mirror, regime scoring
  Scout  (MEM-004) — 7-instrument universe confirmed

OS Confluence scoring:
  HTF bias confirmed (H4 EMA direction) ......... +2  [hard gate]
  Price in discount (long) / premium (short) .... +1
  H4 impulse 50% (within 10% of H4 swing mid) .. +1  [JP mentor v5 — "50% is always a POI"]
  Liquidity swept (lows for longs / highs for shorts) +1
  OB + FVG aligned — Model 3 ................... +2
  FVG only — Model 1 ........................... +1
  Session (in active window) ................... +1
  Session prime window (NY first hour 13-15 UTC) +1  [stacks with session]
  RSI pullback zone ............................ +1
  Trend regime (H4 EMA strongly trending) ...... +1
  Manipulation W (long) / M (short) ............ +1  [JP mentor — ICC entry model]
  Asian 50% level (within 20% of Asian H/L mid)  +1  [JP mentor]
  London 50% NY POI (NY session + at London mid) +1  [JP mentor v4]
  Early leakage (Asian range broken before London)+1  [JP mentor]
  Inside day + closer target ................... +1  [JP mentor]
  PDH/PDL swept today ......................... +1  [JP mentor v2]
  D1 FVG retest (price inside daily imbalance)   +1  [JP mentor v2]
  D1 aligned (daily EMA agrees with H4) ......... +1  [optional]
  Volume spike ................................. +1  [optional]
  W1 FVG retest (weekly imbalance zone) ......... +2  [JP mentor v3]
  Weekly discount/premium (range position) ...... +1  [JP mentor v7]
  NY swept London Low/High ..................... +1  [JP mentor v7]
  Double bottom/top cluster at FVG ............. +1  [JP mentor v9]
  Multi-TF synchrony (H4 or D1 close alignment) +1  [JP mentor — candle close confluence]
  USDJPY macro aligned (DXY-linked instruments)  +1  [JP mentor v7 — UJ as DXY proxy]
  Previous battlefield (prior congestion CHoCH)   +1  [JP mentor v10]

Negative confluences (subtract from score):
  No liquidity sweep                             -1  (market uncleared)
  Opposing manipulation pattern                  -1  (M against long / W against short)
  RSI extreme against trade                      -1  (>75 for long / <25 for short)
  Untaken session liquidity in trade path        -1  (session low not swept for long /
                                                      session high not swept for short —
                                                      magnet pulls price there first)
  Both Asian H+L swept early                     -1  (JP mentor v7: "leaked early, always
                                                      a concern" — directional clarity lost)
  Mid daily range (40-60% of prior day H/L)      -1  (JP mentor v7: "market could go up or
                                                      down, I'm a bit concerned about that"
                                                      — no premium/discount edge available)

Order Block detection (upgraded v7):
  Requires wick on the OB candle + next candle body fully engulfs the OB body.
  JP mentor v7: "There has to be some wick sticking out here. It needs to touch a
  red wick and then the next candle swallows that candle — that becomes an order block."

Dynamic RR:
  Base: rr_target (default 2.5)
  Model 3 trade: + rr_model3_bonus (default +0.5)
  Strong H4 trend: + rr_trend_bonus (default +0.5)
  Cap: rr_max (default 5.0)

Parameter units:
  htf_lookback, h4_swing_lookback → H4 BARS (TF-independent after resample)
  Everything else → input-TF bars (scale ×4 for M15 vs H1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy


# ── Helpers ───────────────────────────────────────────────────────────────────

# Indicators live in the shared single-source module (strategies/indicators.py).
from strategies.indicators import atr as _atr, rsi as _rsi   # noqa: E402

# Per-instrument active session windows (UTC hours, inclusive start exclusive end).
# Session confluence fires when the bar hour falls inside ANY of the listed ranges.
# An instrument with multiple active windows (e.g. JPY trades London + Asia) can
# list both. Gold/Silver are 24h so any hour qualifies.
_INSTRUMENT_SESSIONS: dict[str, list[tuple[int, int]]] = {
    # Forex — London + NY overlap
    "GBPUSD":     [(7, 17)],
    "EURUSD":     [(7, 17)],
    "USDJPY":     [(0, 8), (12, 17)],   # Asia + NY overlap
    # Metals — near 24h, weight toward NY
    "XAUUSD":     [(0, 22)],
    "XAGUSD":     [(0, 22)],
    # US indices — NY only
    "US30":       [(13, 21)],
    "US100":      [(13, 21)],
    "US500":      [(13, 21)],
    "US2000":     [(13, 21)],
    # Asian indices — Tokyo session
    "JP225":      [(0, 8)],
    "HK50":       [(1, 9)],
}
_DEFAULT_SESSION: list[tuple[int, int]] = [(7, 21)]   # fallback for unknown symbols


def _active_session(symbol: str, hour: int) -> bool:
    """Return True if hour is within any active window for the symbol."""
    key = symbol.replace(".cash", "").replace(".fx", "").upper()
    for pat, windows in _INSTRUMENT_SESSIONS.items():
        if key.startswith(pat):
            return any(s <= hour < e for s, e in windows)
    return any(s <= hour < e for s, e in _DEFAULT_SESSION)


# Index CFDs have genuine liquidity gaps outside exchange hours — hard session gate
# remains active for these. Forex/metals trade near-24h so score alone decides.
_LIQUIDITY_GATED = {"US30", "US100", "US500", "US2000", "UK100", "JP225", "HK50"}


def _needs_session_gate(symbol: str) -> bool:
    key = symbol.replace(".cash", "").replace(".fx", "").upper()
    return any(key.startswith(s) for s in _LIQUIDITY_GATED)


def _resample_d1(df: pd.DataFrame) -> pd.DataFrame:
    times = pd.to_datetime(df["time"])
    tmp   = df[["open", "high", "low", "close"]].copy()
    tmp.index = times
    d1 = tmp.resample("1D", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
    d1 = d1.reset_index().rename(columns={"index": "time"})
    d1["time"] = pd.to_datetime(d1["time"])
    return d1


def _resample_h4(df: pd.DataFrame) -> pd.DataFrame:
    times = pd.to_datetime(df["time"])
    tmp   = df[["open", "high", "low", "close"]].copy()
    tmp.index = times
    h4 = tmp.resample("4h", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
    h4 = h4.reset_index().rename(columns={"index": "time"})
    h4["time"] = pd.to_datetime(h4["time"])
    return h4


def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    times = pd.to_datetime(df["time"])
    tmp   = df[["open", "high", "low", "close"]].copy()
    tmp.index = times
    w1 = tmp.resample("W-MON", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
    w1 = w1.reset_index().rename(columns={"index": "time"})
    w1["time"] = pd.to_datetime(w1["time"])
    return w1


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


def _alt_dup(active_fvgs: list, fvg_dir: str, anchor: float, atr_val: float) -> bool:
    """True if an equivalent alt setup already in queue (suppresses bar-by-bar re-add)."""
    return any(
        f["dir"] == fvg_dir and abs(f["fvg_lo"] - anchor) < 0.5 * atr_val
        for f in active_fvgs
    )


# ── Strategy ──────────────────────────────────────────────────────────────────

class AiDENIndexStrategy(Strategy):

    def __init__(
        self,
        # Confluence gate
        min_score: int           = 4,
        # H4 bias — in H4 BARS
        htf_lookback: int        = 20,
        h4_swing_lookback: int   = 40,
        h4_bias_method: str      = "ema",
        discount_pct: float      = 0.5,
        # FVG — in input-TF bars
        min_fvg_atr: float       = 0.10,
        max_fvg_wait: int        = 40,
        max_entry_wait: int      = 8,
        max_active_fvgs: int     = 3,
        # OB
        ob_lookback: int         = 20,
        # Liquidity sweep
        liq_lookback: int        = 10,
        # Risk / RR
        rr_target: float         = 2.5,
        rr_model3_bonus: float   = 0.5,   # extra R when OB+FVG aligned
        rr_trend_bonus: float    = 0.5,   # extra R in strong trend regime
        rr_trend_threshold: float= 0.003, # H4 EMA spread > 0.3% = strong trend
        rr_max: float            = 5.0,
        atr_period: int          = 14,
        atr_stop_buffer: float   = 0.3,
        # RSI
        rsi_period: int          = 14,
        rsi_long_lo: float       = 25.0,  # RSI zone for LONG entries
        rsi_long_hi: float       = 55.0,
        rsi_short_lo: float      = 45.0,  # RSI zone for SHORT entries
        rsi_short_hi: float      = 75.0,
        use_rsi: bool            = True,
        # Session
        session_start: int       = 12,
        session_end: int         = 21,
        session_prime_start: int = 13,    # NY open first hour bonus window
        session_prime_end: int   = 15,
        use_prime_bonus: bool    = True,
        # Volume spike — +1 score when FVG-forming bar volume > mult * 20-bar avg
        use_vol_spike: bool      = False,
        vol_spike_mult: float    = 1.5,
        # CE (Consequent Encroachment) — require price to reach FVG midpoint before entry
        require_ce: bool         = False,
        # D1 bias alignment — +1 score when Daily EMA agrees with H4 EMA
        use_d1_bias: bool        = False,
        # Direction
        long_only: bool          = False,  # False = both longs and shorts
        # Trailing stop — move SL to breakeven once trade reaches +trail_be_r profit
        # Default False: run to full SL/TP (preserves backtest win rate and RR)
        trail_to_be: bool        = False,
        trail_be_r: float        = 1.0,   # R-multiple at which SL moves to entry (BE)
        trail_lock_r: float      = 2.0,   # R-multiple at which SL trails to +1R locked
        # T1 partial close — close t1_partial_pct of position at t1_r, then move SL to BE
        # Default disabled (t1_r=0.0). trail_to_be lock still fires at trail_lock_r.
        t1_r: float              = 0.0,   # R at which T1 fires; 0 = disabled
        t1_partial_pct: float    = 0.5,   # fraction to exit at T1 (0.5 = 50%)
        # Time stop — exit if no TP progress after N bars; 0 = disabled
        time_stop_bars: int      = 0,
    ):
        self.min_score            = min_score
        self.htf_lookback         = htf_lookback
        self.h4_swing_lookback    = h4_swing_lookback
        self.h4_bias_method       = h4_bias_method
        self.discount_pct         = discount_pct
        self.min_fvg_atr          = min_fvg_atr
        self.max_fvg_wait         = max_fvg_wait
        self.max_entry_wait       = max_entry_wait
        self.max_active_fvgs      = max_active_fvgs
        self.ob_lookback          = ob_lookback
        self.liq_lookback         = liq_lookback
        self.rr_target            = rr_target
        self.rr_model3_bonus      = rr_model3_bonus
        self.rr_trend_bonus       = rr_trend_bonus
        self.rr_trend_threshold   = rr_trend_threshold
        self.rr_max               = rr_max
        self.atr_period           = atr_period
        self.atr_stop_buffer      = atr_stop_buffer
        self.rsi_period           = rsi_period
        self.rsi_long_lo          = rsi_long_lo
        self.rsi_long_hi          = rsi_long_hi
        self.rsi_short_lo         = rsi_short_lo
        self.rsi_short_hi         = rsi_short_hi
        self.use_rsi              = use_rsi
        self.session_start        = session_start
        self.session_end          = session_end
        self.session_prime_start  = session_prime_start
        self.session_prime_end    = session_prime_end
        self.use_prime_bonus      = use_prime_bonus
        self.use_vol_spike        = use_vol_spike
        self.vol_spike_mult       = vol_spike_mult
        self.require_ce           = require_ce
        self.use_d1_bias          = use_d1_bias
        self.long_only            = long_only
        self.trail_to_be          = trail_to_be
        self.trail_be_r           = trail_be_r
        self.trail_lock_r         = trail_lock_r
        self.t1_r                 = t1_r
        self.t1_partial_pct       = t1_partial_pct
        self.time_stop_bars       = time_stop_bars
        self._last_h4_bias:       int = 0   # updated on each generate_signals call
        # Gap 7 — USDJPY macro H4 bias; set externally by orchestrator before each bar.
        # +1 = USD trending up (JPY weak), -1 = USD trending down, 0 = neutral/unknown.
        self._uj_h4_bias:         int = 0
        # Set by orchestrator at startup so session windows are instrument-aware.
        self._symbol:             str = ""

    @property
    def name(self) -> str:
        direction = "long" if self.long_only else "bi"
        return (
            f"AiDEN-v2({direction}"
            f",s>={self.min_score}"
            f",{self.h4_bias_method}"
            f",rr={self.rr_target}+dyn"
            f",sess={self.session_start}-{self.session_end})"
        )

    def current_h4_bias(self, df: pd.DataFrame) -> int:
        """Return the current H4 bias (+1/-1/0) from the latest bars.
        Lightweight — just resamples and reads last value. No full loop."""
        h4 = _resample_h4(df)
        if len(h4) < 5:
            return 0
        fast_n = max(self.htf_lookback // 2, 5)
        if self.h4_bias_method == "ema":
            h4_bias, _ = _compute_h4_bias_ema(h4, fast_n, self.htf_lookback)
        else:
            h4_bias, _ = _compute_h4_bias_swing(h4, self.h4_swing_lookback)
        return int(h4_bias.iloc[-1])

    def _dynamic_rr(self, is_model3: bool, trend_strength: float) -> float:
        rr = self.rr_target
        if is_model3:
            rr += self.rr_model3_bonus
        if abs(trend_strength) > self.rr_trend_threshold:
            rr += self.rr_trend_bonus
        return min(rr, self.rr_max)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        open_  = df["open"].astype(float)
        _t     = pd.to_datetime(df["time"])
        times  = _t.dt.tz_localize(None) if _t.dt.tz is not None else _t
        hours  = times.dt.hour

        atr_s = _atr(high, low, close, self.atr_period)
        self._atr_cache = atr_s
        rsi_s = _rsi(close, self.rsi_period) if self.use_rsi else None

        vol_s    = df["tick_volume"].astype(float) if "tick_volume" in df.columns else None
        vol_mean = vol_s.rolling(20).mean() if vol_s is not None else None

        h4 = _resample_h4(df)
        fast_n = max(self.htf_lookback // 2, 5)

        if self.h4_bias_method == "ema":
            h4_bias, h4_spread = _compute_h4_bias_ema(h4, fast_n, self.htf_lookback)
        else:
            h4_bias, h4_spread = _compute_h4_bias_swing(h4, self.htf_lookback)

        # D1 bias — three-TF alignment gate
        d1_bias_at = None
        if self.use_d1_bias:
            d1 = _resample_d1(df)
            if len(d1) >= 10:
                d1_fast_n = max(3, self.htf_lookback // 4)
                d1_slow_n = max(5, self.htf_lookback // 2)
                d1_b, _   = _compute_h4_bias_ema(d1, d1_fast_n, d1_slow_n)
                _d1t_raw  = pd.to_datetime(d1["time"])
                d1_times  = _d1t_raw.dt.tz_localize(None) if _d1t_raw.dt.tz is not None else _d1t_raw
                def _d1_bias_at(bar_time, _d1b=d1_b, _d1t=d1_times):
                    _bt = bar_time.replace(tzinfo=None) if hasattr(bar_time, "tzinfo") and bar_time.tzinfo is not None else bar_time
                    idx = _d1t.searchsorted(_bt, side="right") - 1
                    return int(_d1b.iloc[idx]) if idx >= 0 else 0
                d1_bias_at = _d1_bias_at

        # ── D1 FVG zones (mentor: "I marked out the imbalance between candles on the daily") ──
        # Bullish D1 FVG: gap between bar[i-2].high and bar[i].low (bar i's low > bar i-2's high).
        # Bearish D1 FVG: gap between bar[i-2].low and bar[i].high (bar i's high < bar i-2's low).
        # Computed once from D1 resample; last 7 D1 bars scanned.
        _d1_fvg_zones: list[tuple[float, float, int]] = []  # (lo, hi, direction)
        _d1_for_fvg = _resample_d1(df)
        if len(_d1_for_fvg) >= 3:
            _d1h = _d1_for_fvg["high"].astype(float)
            _d1l = _d1_for_fvg["low"].astype(float)
            for _dj in range(2, min(len(_d1_for_fvg), 9)):
                _bull_gap_d1 = _d1l.iloc[_dj] - _d1h.iloc[_dj - 2]
                if _bull_gap_d1 > 0:
                    _d1_fvg_zones.append((_d1h.iloc[_dj - 2], _d1l.iloc[_dj], 1))
                _bear_gap_d1 = _d1l.iloc[_dj - 2] - _d1h.iloc[_dj]
                if _bear_gap_d1 > 0:
                    _d1_fvg_zones.append((_d1h.iloc[_dj], _d1l.iloc[_dj - 2], -1))

        # ── Weekly FVG zones + range (JP mentor: "closing up weekly imbalance") ──
        # Weekly FVGs are major draw targets — the largest-TF unfilled institutional
        # imbalance. +2 score when price is retesting a weekly FVG zone.
        # Weekly mid/range used for weekly premium/discount score (+1 / -1).
        _w1_fvg_zones: list[tuple[float, float, int]] = []  # (lo, hi, direction)
        _w1_mid_arr  = np.full(len(df), np.nan)
        _w1_rng_arr  = np.full(len(df), np.nan)
        _w1_df = _resample_weekly(df)
        if len(_w1_df) >= 3:
            _w1h = _w1_df["high"].astype(float)
            _w1l = _w1_df["low"].astype(float)
            for _wj in range(2, min(len(_w1_df), 6)):  # last 5 weekly candles
                _bull_gap_w1 = _w1l.iloc[_wj] - _w1h.iloc[_wj - 2]
                if _bull_gap_w1 > 0:
                    _w1_fvg_zones.append((_w1h.iloc[_wj - 2], _w1l.iloc[_wj], 1))
                _bear_gap_w1 = _w1l.iloc[_wj - 2] - _w1h.iloc[_wj]
                if _bear_gap_w1 > 0:
                    _w1_fvg_zones.append((_w1h.iloc[_wj], _w1l.iloc[_wj - 2], -1))
        # Current week mid/range for each bar — map weekly rows onto input bars
        _w1t_raw  = pd.to_datetime(_w1_df["time"]) if len(_w1_df) else pd.Series([], dtype="datetime64[ns]")
        _w1_times = _w1t_raw.dt.tz_localize(None) if len(_w1_df) and _w1t_raw.dt.tz is not None else _w1t_raw
        for _wi in range(len(_w1_df)):
            _wstart = _w1_times.iloc[_wi]
            _wend   = _w1_times.iloc[_wi + 1] if _wi + 1 < len(_w1_df) else pd.Timestamp.max
            _wmask  = (times >= _wstart) & (times < _wend)
            _wh     = float(_w1_df["high"].iloc[_wi])
            _wl     = float(_w1_df["low"].iloc[_wi])
            _w1_mid_arr[_wmask.values] = (_wh + _wl) / 2.0
            _w1_rng_arr[_wmask.values] = _wh - _wl

        # ── Pre-session range (JP mentor: Asian 50% level, early leakage, inside day) ──
        # For each UTC day, compute the high/low of bars BEFORE session_start (the
        # "Asian" or pre-market range) and the previous day's high/low. Three new
        # confluences are derived from these at FVG detection time (below).
        # _cdh/_cdl: running daily high/low up to each bar (for PDH/PDL sweep check).
        _dates_arr       = times.dt.normalize()
        _all_dates_list  = sorted(_dates_arr.unique())
        _ph  = np.full(len(df), np.nan)
        _pl  = np.full(len(df), np.nan)
        _pdh = np.full(len(df), np.nan)  # prior day high
        _pdl = np.full(len(df), np.nan)  # prior day low
        _cdh = np.full(len(df), np.nan)  # current day running high at bar i
        _cdl = np.full(len(df), np.nan)  # current day running low at bar i
        _date_idx_map: dict = {}
        for _d in _all_dates_list:
            _date_idx_map[_d] = np.where((_dates_arr == _d).values)[0]
        for _k, _d in enumerate(_all_dates_list):
            _idxs = _date_idx_map[_d]
            _pre  = np.where(hours.iloc[_idxs].values < self.session_start)[0]
            if len(_pre):
                _ph[_idxs] = float(high.iloc[_idxs[_pre]].max())
                _pl[_idxs] = float(low.iloc[_idxs[_pre]].min())
            # Running cumulative high/low within each day
            _hi_vals = high.iloc[_idxs].values
            _lo_vals = low.iloc[_idxs].values
            for _j in range(len(_idxs)):
                _cdh[_idxs[_j]] = float(np.max(_hi_vals[:_j + 1]))
                _cdl[_idxs[_j]] = float(np.min(_lo_vals[:_j + 1]))
            if _k > 0:
                _pi = _date_idx_map[_all_dates_list[_k - 1]]
                _pdh[_idxs] = float(high.iloc[_pi].max())
                _pdl[_idxs] = float(low.iloc[_pi].min())
        _presess_mid = (_ph + _pl) / 2.0
        _presess_rng = _ph - _pl

        # ── London session range (JP mentor video 4: NY taps into 50% of London range) ──
        # London session: 07:00–12:00 UTC. When NY opens (hour >= 13) and price is at
        # 50% of today's London H/L → strong POI. "NY pushed up to 50% of previous session
        # (London) and had huge wick rejection, then continued London's bearish energy."
        _lon_ph = np.full(len(df), np.nan)  # London session high per bar
        _lon_pl = np.full(len(df), np.nan)  # London session low per bar
        _london_start_h, _london_end_h = 7, 12
        for _k, _d in enumerate(_all_dates_list):
            _idxs  = _date_idx_map[_d]
            _lon_m = np.where(
                (hours.iloc[_idxs].values >= _london_start_h) &
                (hours.iloc[_idxs].values < _london_end_h)
            )[0]
            if len(_lon_m):
                _lh = float(high.iloc[_idxs[_lon_m]].max())
                _ll = float(low.iloc[_idxs[_lon_m]].min())
                _lon_ph[_idxs] = _lh
                _lon_pl[_idxs] = _ll
        _lon_mid = (_lon_ph + _lon_pl) / 2.0
        _lon_rng = _lon_ph - _lon_pl

        h4["bias"]   = h4_bias.values
        h4["spread"] = h4_spread.values
        _h4t_raw = pd.to_datetime(h4["time"])
        h4_times = _h4t_raw.dt.tz_localize(None) if _h4t_raw.dt.tz is not None else _h4t_raw

        def _h4_at(bar_time):
            _bt = bar_time.replace(tzinfo=None) if hasattr(bar_time, "tzinfo") and bar_time.tzinfo is not None else bar_time
            idx = h4_times.searchsorted(_bt, side="right") - 1
            if idx < 0:
                return 0, 0.0
            return int(h4["bias"].iloc[idx]), float(h4["spread"].iloc[idx])

        def _swing_range_at(bar_time):
            _bt = bar_time.replace(tzinfo=None) if hasattr(bar_time, "tzinfo") and bar_time.tzinfo is not None else bar_time
            idx = h4_times.searchsorted(_bt, side="right") - 1
            if idx < self.h4_swing_lookback:
                return float("nan"), float("nan")
            w = h4.iloc[idx - self.h4_swing_lookback:idx]
            return float(w["high"].max()), float(w["low"].min())

        signals      = pd.Series(0.0, index=df.index)
        self._stops  = pd.Series(float("nan"), index=df.index)
        self._scores = pd.Series(0, index=df.index)
        self._score_reasons = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)

        position         = 0
        position_size    = 0.0
        stop_loss        = None
        initial_sl       = None   # SL at entry — never modified; used for R calculations
        take_profit      = None
        entry_price      = None
        t1_hit           = False
        bars_since_entry = 0
        active_fvgs: list[dict] = []

        warmup = max(self.atr_period + 3, self.ob_lookback, self.liq_lookback,
                     self.rsi_period + 2 if self.use_rsi else 0)

        for i in range(warmup, len(df)):
            cv       = float(close.iloc[i])
            lv       = float(low.iloc[i])
            hv       = float(high.iloc[i])
            atr_val  = float(atr_s.iloc[i])
            bar_time = times.iloc[i]
            hour     = int(hours.iloc[i])

            if np.isnan(atr_val):
                signals.iloc[i]     = position * position_size
                self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                continue

            # ── 1. Manage open position ───────────────────────────────────
            if position == 1:
                bars_since_entry += 1

                # Trail — reference initial_sl so trail survives T1 BE move
                if self.trail_to_be and initial_sl is not None and entry_price is not None:
                    init_risk = entry_price - initial_sl
                    if init_risk > 0:
                        if cv >= entry_price + self.trail_lock_r * init_risk:
                            stop_loss = max(stop_loss, entry_price + init_risk)
                        elif cv >= entry_price + self.trail_be_r * init_risk and not t1_hit:
                            stop_loss = max(stop_loss, entry_price)

                # T1 partial — fires once at t1_r; moves SL to BE for runner
                if self.t1_r > 0 and not t1_hit and initial_sl is not None:
                    init_risk = entry_price - initial_sl
                    if init_risk > 0 and cv >= entry_price + self.t1_r * init_risk:
                        t1_hit = True
                        position_size = 1.0 - self.t1_partial_pct
                        stop_loss = entry_price

                # Time stop — exit if no progress after N bars
                if self.time_stop_bars > 0 and bars_since_entry >= self.time_stop_bars:
                    position = 0; position_size = 0.0; t1_hit = False
                    stop_loss = take_profit = entry_price = initial_sl = None
                    bars_since_entry = 0
                elif stop_loss is not None and lv <= stop_loss:
                    position = 0; position_size = 0.0; t1_hit = False
                    stop_loss = take_profit = entry_price = initial_sl = None
                    bars_since_entry = 0
                elif take_profit is not None and hv >= take_profit:
                    position = 0; position_size = 0.0; t1_hit = False
                    stop_loss = take_profit = entry_price = initial_sl = None
                    bars_since_entry = 0

            elif position == -1:
                bars_since_entry += 1

                if self.trail_to_be and initial_sl is not None and entry_price is not None:
                    init_risk = initial_sl - entry_price
                    if init_risk > 0:
                        if cv <= entry_price - self.trail_lock_r * init_risk:
                            stop_loss = min(stop_loss, entry_price - init_risk)
                        elif cv <= entry_price - self.trail_be_r * init_risk and not t1_hit:
                            stop_loss = min(stop_loss, entry_price)

                if self.t1_r > 0 and not t1_hit and initial_sl is not None:
                    init_risk = initial_sl - entry_price
                    if init_risk > 0 and cv <= entry_price - self.t1_r * init_risk:
                        t1_hit = True
                        position_size = 1.0 - self.t1_partial_pct
                        stop_loss = entry_price

                if self.time_stop_bars > 0 and bars_since_entry >= self.time_stop_bars:
                    position = 0; position_size = 0.0; t1_hit = False
                    stop_loss = take_profit = entry_price = initial_sl = None
                    bars_since_entry = 0
                elif stop_loss is not None and hv >= stop_loss:
                    position = 0; position_size = 0.0; t1_hit = False
                    stop_loss = take_profit = entry_price = initial_sl = None
                    bars_since_entry = 0
                elif take_profit is not None and lv <= take_profit:
                    position = 0; position_size = 0.0; t1_hit = False
                    stop_loss = take_profit = entry_price = initial_sl = None
                    bars_since_entry = 0

            if position == 0 and i >= warmup + 2:
                htf_bias, trend_strength = _h4_at(bar_time)
                swing_hi, swing_lo       = _swing_range_at(bar_time)
                in_session   = _active_session(self._symbol, hour)
                in_prime     = self.session_prime_start <= hour < self.session_prime_end
                rsi_val      = float(rsi_s.iloc[i]) if rsi_s is not None else float("nan")
                strongly_trending = abs(trend_strength) > self.rr_trend_threshold

                # Remove invalidated FVGs — expire any FVG whose bias no longer matches
                # current H4 direction. A bull FVG formed when H4 was +1 is stale if H4
                # has since flipped to 0 or -1; entering it would mean fading the current bias.
                to_expire = []
                for _fvg in active_fvgs:
                    if _fvg["dir"] == "bull" and htf_bias != 1:
                        to_expire.append(_fvg)
                    elif _fvg["dir"] == "bear" and htf_bias != -1:
                        to_expire.append(_fvg)
                for _fvg in to_expire:
                    active_fvgs.remove(_fvg)

                if htf_bias == 0:
                    self._expire_fvgs_neutral(active_fvgs, cv)
                    signals.iloc[i]     = position * position_size
                    self._stops.iloc[i] = float("nan")
                    continue

                # ── 2. Detect new FVGs ───────────────────────────────────
                h2  = float(high.iloc[i - 2])
                l2  = float(low.iloc[i - 2])

                # LONG setup — bullish FVG
                if htf_bias == 1:
                    bull_gap = lv - h2
                    if bull_gap >= self.min_fvg_atr * atr_val:
                        score = 2; reasons = ["H4 bias +2"]  # HTF bias +2

                        if not np.isnan(swing_hi) and swing_hi > swing_lo:
                            mid = swing_lo + (swing_hi - swing_lo) * self.discount_pct
                            if cv < mid:
                                score += 1; reasons.append("Discount zone")
                            # H4 impulse 50% POI: price at midpoint of H4 swing range.
                            # JP mentor video 5: "50% of the impulse is always a point of interest"
                            # — more specific than discount zone, rewards being exactly at the wall.
                            h4_rng = swing_hi - swing_lo
                            if h4_rng > 0 and abs(cv - mid) <= 0.10 * h4_rng:
                                score += 1; reasons.append("H4 impulse 50%")

                        # ── Negatives — reasons NOT to trade ─────────────────
                        # No liquidity sweep: longs need to see prior lows swept before entry.
                        # Market that hasn't swept lows hasn't cleared retail positions yet.
                        if _liq_swept_low(low, i, self.liq_lookback):
                            score += 1; reasons.append("Liquidity sweep")
                        else:
                            score -= 1; reasons.append("-No sweep")

                        # Opposing manipulation M active: if M pattern formed recently,
                        # market just signalled it wants to go DOWN. Don't fade it for a long.
                        if _manipulation_m(high, low, i, lookback=min(20, i)):
                            score -= 1; reasons.append("-Opposing Manip M")

                        # RSI extreme against trade: entering a long when RSI already > 75
                        # means the market is overbought — chasing a move that's exhausted.
                        if self.use_rsi and rsi_s is not None:
                            _rsi_now = float(rsi_s.iloc[i])
                            if not np.isnan(_rsi_now) and _rsi_now > 75.0:
                                score -= 1; reasons.append("-RSI overbought")

                        # Untaken session low below: if Asian session low hasn't been swept
                        # today, that liquidity is a magnet — price may dip there first before
                        # any long continuation. JP mentor: "liquidity is drawn to price."
                        _pl_i = _pl[i]; _cdl_i = _cdl[i]
                        if (not np.isnan(_pl_i) and not np.isnan(_cdl_i)
                                and _cdl_i > _pl_i and _pl_i < cv):
                            score -= 1; reasons.append("-Untaken session low below")

                        # Both Asian H + L swept before trade = concern (JP mentor v7:
                        # "It leaked early. I always have a concern about that.")
                        # When liquidity has been taken on BOTH sides, directional clarity
                        # is reduced — the market may reverse or go sideways.
                        _ph_i = _ph[i]; _cdh_i = _cdh[i]
                        if (not np.isnan(_ph_i) and not np.isnan(_pl_i)
                                and not np.isnan(_cdh_i) and not np.isnan(_cdl_i)
                                and _cdh_i >= _ph_i and _cdl_i <= _pl_i):
                            score -= 1; reasons.append("-Both Asian H+L swept")

                        # Mid daily range (40-60%): JP mentor v7 — "kind of in the middle
                        # of the range, market could go up or down, I'm a bit concerned."
                        # Neither premium nor discount = no directional edge from range position.
                        _pdh_i = _pdh[i]; _pdl_i = _pdl[i]
                        if (not np.isnan(_pdh_i) and not np.isnan(_pdl_i)
                                and _pdh_i > _pdl_i):
                            _d_rng = _pdh_i - _pdl_i
                            _pct   = (cv - _pdl_i) / _d_rng
                            if 0.40 <= _pct <= 0.60:
                                score -= 1; reasons.append("-Mid daily range")

                        ob_lo, ob_hi = _find_bullish_ob(open_, close, high, low, i - 2, self.ob_lookback)
                        if ob_lo is not None and min(ob_hi, lv) - max(ob_lo, h2) > 0:
                            score += 2; is_model3 = True; reasons.append("Order block (M3) +2")
                        else:
                            score += 1; is_model3 = False; reasons.append("Order block")

                        if in_session:
                            score += 1; reasons.append("Session")
                        if in_prime and self.use_prime_bonus:
                            score += 1; reasons.append("Prime window")

                        if self.use_rsi and not np.isnan(rsi_val):
                            if self.rsi_long_lo <= rsi_val <= self.rsi_long_hi:
                                score += 1; reasons.append("RSI zone")

                        if strongly_trending:
                            score += 1; reasons.append("Trend regime")

                        if self.use_vol_spike and vol_mean is not None:
                            vm = float(vol_mean.iloc[i])
                            if vm > 0 and float(vol_s.iloc[i]) > self.vol_spike_mult * vm:
                                score += 1; reasons.append("Volume spike")

                        if d1_bias_at is not None and d1_bias_at(bar_time) == 1:
                            score += 1; reasons.append("D1 aligned")

                        # ── JP Mentor confluences (Asian session structure) ──
                        _psm = _presess_mid[i]; _psr = _presess_rng[i]
                        _pdh_i = _pdh[i];       _pdl_i = _pdl[i]
                        if not np.isnan(_psm) and _psr > 0:
                            if abs(cv - _psm) <= 0.20 * _psr:
                                score += 1; reasons.append("Asian 50% level")
                            if not np.isnan(_pdl_i) and _pl[i] < _pdl_i:
                                score += 1; reasons.append("Early leakage (London sweep)")
                        # London 50% level as NY POI: when NY is open and price taps into
                        # 50% of the London session range — mentor video 4: "NY pushed up
                        # to 50% of the previous session (London) and had huge wick rejection"
                        if hour >= 13 and not np.isnan(_lon_mid[i]) and _lon_rng[i] > 0:
                            if abs(cv - _lon_mid[i]) <= 0.20 * _lon_rng[i]:
                                score += 1; reasons.append("London 50% NY POI")
                        if (not np.isnan(_pdh_i) and not np.isnan(_pdl_i)
                                and hv < _pdh_i and lv > _pdl_i
                                and (_pdh_i - cv) < (cv - _pdl_i)):
                            score += 1; reasons.append("Inside day (closer high)")
                        # Manipulation W: sweep of prior lows + right shoulder + CHoCH
                        if _manipulation_w(high, low, i, lookback=min(20, i)):
                            score += 1; reasons.append("Manipulation W")
                        # PDL swept today (+1): today's running low has taken prior day low
                        if not np.isnan(_pdl[i]) and not np.isnan(_cdl[i]) and _cdl[i] < _pdl[i]:
                            score += 1; reasons.append("PDL swept today")
                        # D1 FVG retest (+1): price is inside a daily-level imbalance zone
                        # Mentor: "I marked out the imbalance — that's a point of interest"
                        for _z in _d1_fvg_zones:
                            if _z[2] == 1 and _z[0] <= cv <= _z[1]:
                                score += 1; reasons.append("D1 FVG retest"); break

                        # Gap 6 — Weekly FVG retest (+2): highest-TF institutional imbalance
                        # JP mentor v3: "we are currently closing up a little bit of weekly imbalance"
                        for _wz in _w1_fvg_zones:
                            if _wz[2] == 1 and _wz[0] <= cv <= _wz[1]:
                                score += 2; reasons.append("W1 FVG retest +2"); break

                        # Gap 13 — Weekly range position: discount (lower half) favours longs
                        # JP mentor v7: "before I do anything, I zoom out to the weekly —
                        # are we at a weekly high, low, or in the middle?"
                        _w1_mid_i = _w1_mid_arr[i]; _w1_rng_i = _w1_rng_arr[i]
                        if not np.isnan(_w1_mid_i) and _w1_rng_i > 0:
                            if cv < _w1_mid_i:
                                score += 1; reasons.append("Weekly discount")
                            elif cv > _w1_mid_i + 0.30 * _w1_rng_i:
                                score -= 1; reasons.append("-At weekly premium (risk long)")

                        # Gap 5 — NY swept London Low (bullish): NY's first move takes the
                        # London session low, then reverses — the classic NY reversal setup.
                        # JP mentor v7: "typically NY sweeps London levels, goes takes the
                        # London low, then reverses and rallies"
                        if (hour >= 13 and not np.isnan(_lon_pl[i])
                                and not np.isnan(_cdl_i) and _cdl_i <= _lon_pl[i]):
                            score += 1; reasons.append("NY swept London Low")

                        # Gap 11 — Double bottom cluster at FVG zone (+1): two or more prior
                        # swing lows at approximately the same level as fvg_lo = accumulated
                        # resting liquidity. JP mentor v9: "double bottom... accumulation beside
                        # the liquidity is your signal."
                        _db_tol = 0.15 * atr_val
                        _db_count = sum(
                            1 for _j in range(max(0, i - 30), i - 2)
                            if abs(float(low.iloc[_j]) - h2) <= _db_tol
                        )
                        if _db_count >= 2:
                            score += 1; reasons.append("Double bottom cluster")

                        # Equal lows — liquidity cluster (+1): multiple prior lows at the same
                        # level below the FVG zone indicate resting stop-loss liquidity that
                        # price just swept. JP mentor TR4/9: "equal highs, equal highs,
                        # essentially accumulate — is this a coincidence? Liquidity taken."
                        _eq_tol = 0.12 * atr_val
                        _eq_lo_count = sum(
                            1 for _j in range(max(0, i - 40), i - 2)
                            if abs(float(low.iloc[_j]) - h2) <= _eq_tol
                        )
                        if _eq_lo_count >= 2:
                            score += 1; reasons.append("Equal lows (liq cluster)")

                        # v10 — Previous battlefield (+1): price returns to a prior
                        # congestion zone where bulls/bears already fought (prior CHoCH area).
                        # JP mentor v10: "I came back into a previous battlefield — that's
                        # how I was able to get into this trade."
                        if _is_prior_battlefield(high, low, close, i, h2, lv, atr_val):
                            score += 1; reasons.append("Prior battlefield")

                        # Gap 10 — Multi-TF synchrony (+1): H4 or D1 candle close alignment.
                        # JP mentor: entering on a candle that closes multiple TFs simultaneously
                        # concentrates institutional order flow at that moment.
                        _bar_hour = hour if isinstance(hour, int) else int(hours.iloc[i])
                        if _bar_hour % 4 == 0:
                            score += 1; reasons.append("H4 sync")
                        elif _bar_hour == 0:
                            score += 1; reasons.append("D1 sync")

                        # Gap 7 — USDJPY macro filter (+1/-1 for DXY-linked instruments).
                        # JP mentor v7: "I look at UJ — it tells me the DXY direction."
                        # EURUSD/GBPUSD/XAUUSD/XAGUSD are DXY-inverse: USDJPY bullish
                        # (USD strong) means these pairs naturally weaken → penalise longs.
                        # USDJPY bearish (USD weak) → tailwind for longs → reward.
                        if self._uj_h4_bias != 0:
                            if self._uj_h4_bias == 1:
                                score -= 1; reasons.append("-UJ macro headwind (long)")
                            else:  # _uj_h4_bias == -1
                                score += 1; reasons.append("UJ macro tailwind (long)")

                        if score >= self.min_score:
                            active_fvgs.append({
                                "dir":      "bull",
                                "fvg_lo":   h2,
                                "fvg_hi":   lv,
                                "fvg_ce":   (h2 + lv) / 2,
                                "ob_lo":    ob_lo,
                                "ob_hi":    ob_hi,
                                "score":    score,
                                "reasons":  reasons,
                                "formed":   i,
                                "tested":   False,
                                "test_bar": None,
                                "model3":   is_model3,
                                "trend_s":  trend_strength,
                            })


                # SHORT setup — bearish FVG
                if htf_bias == -1 and not self.long_only:
                    bear_gap = l2 - hv
                    if bear_gap >= self.min_fvg_atr * atr_val:
                        score = 2; reasons = ["H4 bias +2"]  # HTF bias +2

                        if not np.isnan(swing_hi) and swing_hi > swing_lo:
                            mid = swing_lo + (swing_hi - swing_lo) * self.discount_pct
                            if cv > mid:
                                score += 1; reasons.append("Premium zone")
                            # H4 impulse 50% POI (SHORT): same logic, bearish direction
                            h4_rng = swing_hi - swing_lo
                            if h4_rng > 0 and abs(cv - mid) <= 0.10 * h4_rng:
                                score += 1; reasons.append("H4 impulse 50%")

                        # ── Negatives — reasons NOT to trade ─────────────────
                        if _liq_swept_high(high, i, self.liq_lookback):
                            score += 1; reasons.append("Liquidity sweep")
                        else:
                            score -= 1; reasons.append("-No sweep")

                        # Opposing Manipulation W: market just signalled it wants to go UP.
                        if _manipulation_w(high, low, i, lookback=min(20, i)):
                            score -= 1; reasons.append("-Opposing Manip W")

                        # RSI oversold entering a short: chasing an exhausted down move.
                        if self.use_rsi and rsi_s is not None:
                            _rsi_now = float(rsi_s.iloc[i])
                            if not np.isnan(_rsi_now) and _rsi_now < 25.0:
                                score -= 1; reasons.append("-RSI oversold")

                        # Untaken session high above: if Asian session high hasn't been swept
                        # today, that liquidity is a magnet — price may rally there first before
                        # any short continuation. JP mentor: "why would it turn before taking that high?"
                        _ph_i = _ph[i]; _cdh_i = _cdh[i]
                        if (not np.isnan(_ph_i) and not np.isnan(_cdh_i)
                                and _cdh_i < _ph_i and _ph_i > cv):
                            score -= 1; reasons.append("-Untaken session high above")

                        # Both Asian H + L swept before trade = concern (JP mentor v7:
                        # "It leaked early. I always have a concern about that.")
                        _pl_i = _pl[i]; _cdl_i = _cdl[i]
                        if (not np.isnan(_ph_i) and not np.isnan(_pl_i)
                                and not np.isnan(_cdh_i) and not np.isnan(_cdl_i)
                                and _cdh_i >= _ph_i and _cdl_i <= _pl_i):
                            score -= 1; reasons.append("-Both Asian H+L swept")

                        # Mid daily range (40-60%): JP mentor v7 — "kind of in the middle
                        # of the range, market could go up or down, I'm a bit concerned."
                        _pdh_i = _pdh[i]; _pdl_i = _pdl[i]
                        if (not np.isnan(_pdh_i) and not np.isnan(_pdl_i)
                                and _pdh_i > _pdl_i):
                            _d_rng = _pdh_i - _pdl_i
                            _pct   = (cv - _pdl_i) / _d_rng
                            if 0.40 <= _pct <= 0.60:
                                score -= 1; reasons.append("-Mid daily range")

                        ob_lo, ob_hi = _find_bearish_ob(open_, close, high, low, i - 2, self.ob_lookback)
                        if ob_lo is not None and min(ob_hi, l2) - max(ob_lo, hv) > 0:
                            score += 2; is_model3 = True; reasons.append("Order block (M3) +2")
                        else:
                            score += 1; is_model3 = False; reasons.append("Order block")

                        if in_session:
                            score += 1; reasons.append("Session")
                        if in_prime and self.use_prime_bonus:
                            score += 1; reasons.append("Prime window")

                        if self.use_rsi and not np.isnan(rsi_val):
                            if self.rsi_short_lo <= rsi_val <= self.rsi_short_hi:
                                score += 1; reasons.append("RSI zone")

                        if strongly_trending:
                            score += 1; reasons.append("Trend regime")

                        if self.use_vol_spike and vol_mean is not None:
                            vm = float(vol_mean.iloc[i])
                            if vm > 0 and float(vol_s.iloc[i]) > self.vol_spike_mult * vm:
                                score += 1; reasons.append("Volume spike")

                        if d1_bias_at is not None and d1_bias_at(bar_time) == -1:
                            score += 1; reasons.append("D1 aligned")

                        # ── JP Mentor confluences (Asian session structure) ──
                        _psm = _presess_mid[i]; _psr = _presess_rng[i]
                        _pdh_i = _pdh[i];       _pdl_i = _pdl[i]
                        if not np.isnan(_psm) and _psr > 0:
                            if abs(cv - _psm) <= 0.20 * _psr:
                                score += 1; reasons.append("Asian 50% level")
                            if not np.isnan(_pdh_i) and _ph[i] > _pdh_i:
                                score += 1; reasons.append("Early leakage (London sweep)")
                        # London 50% level as NY POI (SHORT): same logic, bearish direction
                        if hour >= 13 and not np.isnan(_lon_mid[i]) and _lon_rng[i] > 0:
                            if abs(cv - _lon_mid[i]) <= 0.20 * _lon_rng[i]:
                                score += 1; reasons.append("London 50% NY POI")
                        if (not np.isnan(_pdh_i) and not np.isnan(_pdl_i)
                                and hv < _pdh_i and lv > _pdl_i
                                and (cv - _pdl_i) < (_pdh_i - cv)):
                            score += 1; reasons.append("Inside day (closer low)")
                        # Manipulation M: sweep of prior highs + right shoulder + CHoCH
                        if _manipulation_m(high, low, i, lookback=min(20, i)):
                            score += 1; reasons.append("Manipulation M")
                        # PDH swept today (+1): today's running high has taken prior day high
                        if not np.isnan(_pdh[i]) and not np.isnan(_cdh[i]) and _cdh[i] > _pdh[i]:
                            score += 1; reasons.append("PDH swept today")
                        # D1 FVG retest (+1): price is inside a bearish daily imbalance zone
                        for _z in _d1_fvg_zones:
                            if _z[2] == -1 and _z[0] <= cv <= _z[1]:
                                score += 1; reasons.append("D1 FVG retest"); break

                        # Gap 6 — Weekly FVG retest (+2): bearish weekly imbalance
                        for _wz in _w1_fvg_zones:
                            if _wz[2] == -1 and _wz[0] <= cv <= _wz[1]:
                                score += 2; reasons.append("W1 FVG retest +2"); break

                        # Gap 13 — Weekly premium favours shorts
                        _w1_mid_i = _w1_mid_arr[i]; _w1_rng_i = _w1_rng_arr[i]
                        if not np.isnan(_w1_mid_i) and _w1_rng_i > 0:
                            if cv > _w1_mid_i:
                                score += 1; reasons.append("Weekly premium")
                            elif cv < _w1_mid_i - 0.30 * _w1_rng_i:
                                score -= 1; reasons.append("-At weekly discount (risk short)")

                        # Gap 5 — NY swept London High (bearish): NY sweeps the London high
                        # then reverses — the bearish NY setup. JP mentor v7.
                        if (hour >= 13 and not np.isnan(_lon_ph[i])
                                and not np.isnan(_cdh_i) and _cdh_i >= _lon_ph[i]):
                            score += 1; reasons.append("NY swept London High")

                        # Gap 11 — Double top cluster at FVG zone (+1)
                        _dt_tol = 0.15 * atr_val
                        _dt_count = sum(
                            1 for _j in range(max(0, i - 30), i - 2)
                            if abs(float(high.iloc[_j]) - l2) <= _dt_tol
                        )
                        if _dt_count >= 2:
                            score += 1; reasons.append("Double top cluster")

                        # Equal highs — liquidity cluster (+1): multiple prior highs at the
                        # same level above the FVG zone = resting buy-stop liquidity swept.
                        # JP mentor TR4/9: mirror of equal lows concept.
                        _eq_hi_tol = 0.12 * atr_val
                        _eq_hi_count = sum(
                            1 for _j in range(max(0, i - 40), i - 2)
                            if abs(float(high.iloc[_j]) - l2) <= _eq_hi_tol
                        )
                        if _eq_hi_count >= 2:
                            score += 1; reasons.append("Equal highs (liq cluster)")

                        # v10 — Previous battlefield (+1): SHORT mirror
                        if _is_prior_battlefield(high, low, close, i, hv, l2, atr_val):
                            score += 1; reasons.append("Prior battlefield")

                        # Gap 10 — Multi-TF synchrony (+1)
                        _bar_hour_s = hour if isinstance(hour, int) else int(hours.iloc[i])
                        if _bar_hour_s % 4 == 0:
                            score += 1; reasons.append("H4 sync")
                        elif _bar_hour_s == 0:
                            score += 1; reasons.append("D1 sync")

                        # Gap 7 — USDJPY macro filter (SHORT, DXY-inverse instruments).
                        # USDJPY bullish (USD strong) → DXY-inverse pairs weaken → tailwind for shorts.
                        # USDJPY bearish (USD weak) → DXY-inverse pairs strengthen → headwind for shorts.
                        if self._uj_h4_bias != 0:
                            if self._uj_h4_bias == 1:
                                score += 1; reasons.append("UJ macro tailwind (short)")
                            else:  # _uj_h4_bias == -1
                                score -= 1; reasons.append("-UJ macro headwind (short)")

                        if score >= self.min_score:
                            active_fvgs.append({
                                "dir":      "bear",
                                "fvg_lo":   hv,
                                "fvg_hi":   l2,
                                "fvg_ce":   (hv + l2) / 2,
                                "ob_lo":    ob_lo,
                                "ob_hi":    ob_hi,
                                "score":    score,
                                "reasons":  reasons,
                                "formed":   i,
                                "tested":   False,
                                "test_bar": None,
                                "model3":   is_model3,
                                "trend_s":  trend_strength,
                            })

                # ── Alt setups: liquidity sweep reversal + BOS ────────────────
                # Fire when no clean FVG exists but structure/liquidity confirms.
                # Pushed into active_fvgs with tested=True for fast entry.
                _alt_cdh_i = _cdh[i]; _alt_cdl_i = _cdl[i]
                _alt_pdh_i = _pdh[i]; _alt_pdl_i = _pdl[i]

                if htf_bias == 1:
                    # ── Sweep reversal long ───────────────────────────────────
                    _swept_lo, _sl_lo = _sweep_reversal_bull(close, low, i, self.liq_lookback * 2)
                    if _swept_lo is not None and not _alt_dup(active_fvgs, "bull", _sl_lo, atr_val):
                        _sc = 2; _rs = ["H4 bias +2"]
                        _sc += 2; _rs.append("Liq sweep reversal +2")

                        if not np.isnan(swing_hi) and swing_hi > swing_lo:
                            _mid = swing_lo + (swing_hi - swing_lo) * self.discount_pct
                            if cv < _mid:
                                _sc += 1; _rs.append("Discount zone")
                            _h4r = swing_hi - swing_lo
                            if _h4r > 0 and abs(cv - _mid) <= 0.10 * _h4r:
                                _sc += 1; _rs.append("H4 impulse 50%")

                        if in_session:
                            _sc += 1; _rs.append("Session")
                        if in_prime and self.use_prime_bonus:
                            _sc += 1; _rs.append("Prime window")
                        if self.use_rsi and not np.isnan(rsi_val):
                            if self.rsi_long_lo <= rsi_val <= self.rsi_long_hi:
                                _sc += 1; _rs.append("RSI zone")
                        if strongly_trending:
                            _sc += 1; _rs.append("Trend regime")
                        if _manipulation_m(high, low, i, lookback=min(20, i)):
                            _sc -= 1; _rs.append("-Opposing Manip M")
                        if self.use_rsi and rsi_s is not None:
                            _rn = float(rsi_s.iloc[i])
                            if not np.isnan(_rn) and _rn > 75.0:
                                _sc -= 1; _rs.append("-RSI overbought")
                        if (not np.isnan(_alt_pdh_i) and not np.isnan(_alt_pdl_i)
                                and _alt_pdh_i > _alt_pdl_i):
                            _dr = _alt_pdh_i - _alt_pdl_i
                            if _dr > 0 and 0.40 <= (cv - _alt_pdl_i) / _dr <= 0.60:
                                _sc -= 1; _rs.append("-Mid daily range")
                        if self._uj_h4_bias == 1:
                            _sc -= 1; _rs.append("-UJ macro headwind (long)")
                        elif self._uj_h4_bias == -1:
                            _sc += 1; _rs.append("UJ macro tailwind (long)")

                        if _sc >= self.min_score:
                            active_fvgs.append({
                                "dir":      "bull",
                                "fvg_lo":   _sl_lo,
                                "fvg_hi":   _swept_lo + 0.3 * atr_val,
                                "fvg_ce":   _swept_lo + 0.15 * atr_val,
                                "ob_lo":    _sl_lo,
                                "ob_hi":    None,
                                "score":    _sc,
                                "reasons":  _rs,
                                "formed":   i,
                                "tested":   True,
                                "test_bar": i,
                                "model3":   False,
                                "trend_s":  trend_strength,
                            })

                    # ── BOS long ──────────────────────────────────────────────
                    _bos_level, _bos_hit = _bos_bull(high, close, i, self.ob_lookback)
                    if _bos_hit and _bos_level is not None and not _alt_dup(active_fvgs, "bull", _bos_level, atr_val):
                        _sc = 2; _rs = ["H4 bias +2"]
                        _sc += 1; _rs.append("BOS (structure break)")

                        if _liq_swept_low(low, i, self.liq_lookback):
                            _sc += 1; _rs.append("Liquidity sweep")
                        else:
                            _sc -= 1; _rs.append("-No sweep")

                        if not np.isnan(swing_hi) and swing_hi > swing_lo:
                            _mid = swing_lo + (swing_hi - swing_lo) * self.discount_pct
                            if cv < _mid:
                                _sc += 1; _rs.append("Discount zone")

                        if in_session:
                            _sc += 1; _rs.append("Session")
                        if in_prime and self.use_prime_bonus:
                            _sc += 1; _rs.append("Prime window")
                        if self.use_rsi and not np.isnan(rsi_val):
                            if self.rsi_long_lo <= rsi_val <= self.rsi_long_hi:
                                _sc += 1; _rs.append("RSI zone")
                        if strongly_trending:
                            _sc += 1; _rs.append("Trend regime")
                        if _manipulation_m(high, low, i, lookback=min(20, i)):
                            _sc -= 1; _rs.append("-Opposing Manip M")
                        if (not np.isnan(_alt_pdh_i) and not np.isnan(_alt_pdl_i)
                                and _alt_pdh_i > _alt_pdl_i):
                            _dr = _alt_pdh_i - _alt_pdl_i
                            if _dr > 0 and 0.40 <= (cv - _alt_pdl_i) / _dr <= 0.60:
                                _sc -= 1; _rs.append("-Mid daily range")
                        if self._uj_h4_bias == 1:
                            _sc -= 1; _rs.append("-UJ macro headwind (long)")
                        elif self._uj_h4_bias == -1:
                            _sc += 1; _rs.append("UJ macro tailwind (long)")

                        if _sc >= self.min_score:
                            active_fvgs.append({
                                "dir":      "bull",
                                "fvg_lo":   _bos_level,
                                "fvg_hi":   _bos_level + 0.5 * atr_val,
                                "fvg_ce":   _bos_level + 0.25 * atr_val,
                                "ob_lo":    float(low.iloc[i]),
                                "ob_hi":    None,
                                "score":    _sc,
                                "reasons":  _rs,
                                "formed":   i,
                                "tested":   True,
                                "test_bar": i,
                                "model3":   False,
                                "trend_s":  trend_strength,
                            })

                if htf_bias == -1 and not self.long_only:
                    # ── Sweep reversal short ──────────────────────────────────
                    _swept_hi, _sl_hi = _sweep_reversal_bear(close, high, i, self.liq_lookback * 2)
                    if _swept_hi is not None and not _alt_dup(active_fvgs, "bear", _sl_hi, atr_val):
                        _sc = 2; _rs = ["H4 bias +2"]
                        _sc += 2; _rs.append("Liq sweep reversal +2")

                        if not np.isnan(swing_hi) and swing_hi > swing_lo:
                            _mid = swing_lo + (swing_hi - swing_lo) * self.discount_pct
                            if cv > _mid:
                                _sc += 1; _rs.append("Premium zone")
                            _h4r = swing_hi - swing_lo
                            if _h4r > 0 and abs(cv - _mid) <= 0.10 * _h4r:
                                _sc += 1; _rs.append("H4 impulse 50%")

                        if in_session:
                            _sc += 1; _rs.append("Session")
                        if in_prime and self.use_prime_bonus:
                            _sc += 1; _rs.append("Prime window")
                        if self.use_rsi and not np.isnan(rsi_val):
                            if self.rsi_short_lo <= rsi_val <= self.rsi_short_hi:
                                _sc += 1; _rs.append("RSI zone")
                        if strongly_trending:
                            _sc += 1; _rs.append("Trend regime")
                        if _manipulation_w(high, low, i, lookback=min(20, i)):
                            _sc -= 1; _rs.append("-Opposing Manip W")
                        if self.use_rsi and rsi_s is not None:
                            _rn = float(rsi_s.iloc[i])
                            if not np.isnan(_rn) and _rn < 25.0:
                                _sc -= 1; _rs.append("-RSI oversold")
                        if (not np.isnan(_alt_pdh_i) and not np.isnan(_alt_pdl_i)
                                and _alt_pdh_i > _alt_pdl_i):
                            _dr = _alt_pdh_i - _alt_pdl_i
                            if _dr > 0 and 0.40 <= (cv - _alt_pdl_i) / _dr <= 0.60:
                                _sc -= 1; _rs.append("-Mid daily range")
                        if self._uj_h4_bias == 1:
                            _sc += 1; _rs.append("UJ macro tailwind (short)")
                        elif self._uj_h4_bias == -1:
                            _sc -= 1; _rs.append("-UJ macro headwind (short)")

                        if _sc >= self.min_score:
                            active_fvgs.append({
                                "dir":      "bear",
                                "fvg_lo":   _swept_hi - 0.3 * atr_val,
                                "fvg_hi":   _sl_hi,
                                "fvg_ce":   _swept_hi - 0.15 * atr_val,
                                "ob_lo":    None,
                                "ob_hi":    _sl_hi,
                                "score":    _sc,
                                "reasons":  _rs,
                                "formed":   i,
                                "tested":   True,
                                "test_bar": i,
                                "model3":   False,
                                "trend_s":  trend_strength,
                            })

                    # ── BOS short ─────────────────────────────────────────────
                    _bos_level, _bos_hit = _bos_bear(low, close, i, self.ob_lookback)
                    if _bos_hit and _bos_level is not None and not _alt_dup(active_fvgs, "bear", _bos_level - 0.5 * atr_val, atr_val):
                        _sc = 2; _rs = ["H4 bias +2"]
                        _sc += 1; _rs.append("BOS (structure break)")

                        if _liq_swept_high(high, i, self.liq_lookback):
                            _sc += 1; _rs.append("Liquidity sweep")
                        else:
                            _sc -= 1; _rs.append("-No sweep")

                        if not np.isnan(swing_hi) and swing_hi > swing_lo:
                            _mid = swing_lo + (swing_hi - swing_lo) * self.discount_pct
                            if cv > _mid:
                                _sc += 1; _rs.append("Premium zone")

                        if in_session:
                            _sc += 1; _rs.append("Session")
                        if in_prime and self.use_prime_bonus:
                            _sc += 1; _rs.append("Prime window")
                        if self.use_rsi and not np.isnan(rsi_val):
                            if self.rsi_short_lo <= rsi_val <= self.rsi_short_hi:
                                _sc += 1; _rs.append("RSI zone")
                        if strongly_trending:
                            _sc += 1; _rs.append("Trend regime")
                        if _manipulation_w(high, low, i, lookback=min(20, i)):
                            _sc -= 1; _rs.append("-Opposing Manip W")
                        if (not np.isnan(_alt_pdh_i) and not np.isnan(_alt_pdl_i)
                                and _alt_pdh_i > _alt_pdl_i):
                            _dr = _alt_pdh_i - _alt_pdl_i
                            if _dr > 0 and 0.40 <= (cv - _alt_pdl_i) / _dr <= 0.60:
                                _sc -= 1; _rs.append("-Mid daily range")
                        if self._uj_h4_bias == 1:
                            _sc += 1; _rs.append("UJ macro tailwind (short)")
                        elif self._uj_h4_bias == -1:
                            _sc -= 1; _rs.append("-UJ macro headwind (short)")

                        if _sc >= self.min_score:
                            active_fvgs.append({
                                "dir":      "bear",
                                "fvg_lo":   _bos_level - 0.5 * atr_val,
                                "fvg_hi":   _bos_level,
                                "fvg_ce":   _bos_level - 0.25 * atr_val,
                                "ob_lo":    None,
                                "ob_hi":    float(high.iloc[i]),
                                "score":    _sc,
                                "reasons":  _rs,
                                "formed":   i,
                                "tested":   True,
                                "test_bar": i,
                                "model3":   False,
                                "trend_s":  trend_strength,
                            })

                # Trim FVG queue
                if len(active_fvgs) > self.max_active_fvgs * 2:
                    active_fvgs.sort(key=lambda x: x["score"], reverse=True)
                    active_fvgs = active_fvgs[:self.max_active_fvgs * 2]

                # ── 3. Process queued FVGs ───────────────────────────────
                to_remove = []
                for fvg in active_fvgs:
                    fvg_lo = fvg["fvg_lo"]
                    fvg_hi = fvg["fvg_hi"]

                    if (i - fvg["formed"]) > self.max_fvg_wait:
                        to_remove.append(fvg); continue

                    if fvg["dir"] == "bull":
                        if cv < fvg_lo:
                            to_remove.append(fvg); continue
                        ce_level = fvg.get("fvg_ce", fvg_hi) if self.require_ce else fvg_hi
                        if not fvg["tested"] and lv <= ce_level:
                            fvg["tested"] = True; fvg["test_bar"] = i
                        if fvg["tested"]:
                            if (i - fvg["test_bar"]) > self.max_entry_wait:
                                to_remove.append(fvg)
                            elif cv > fvg_hi and position == 0:
                                if _needs_session_gate(self._symbol) and not _active_session(self._symbol, hour):
                                    to_remove.append(fvg); continue
                                stop_anchor = fvg["ob_lo"] if fvg["ob_lo"] is not None else fvg_lo
                                sl   = stop_anchor - self.atr_stop_buffer * atr_val
                                dist = cv - sl
                                if dist > 0:
                                    rr               = self._dynamic_rr(fvg["model3"], fvg["trend_s"])
                                    position         = 1
                                    position_size    = 1.0
                                    stop_loss        = sl
                                    initial_sl       = sl
                                    take_profit      = cv + rr * dist
                                    entry_price      = cv
                                    t1_hit           = False
                                    bars_since_entry = 0
                                    self._scores.iloc[i] = fvg["score"]
                                    self._score_reasons.iloc[i] = fvg.get("reasons", [])
                                to_remove.append(fvg)

                    else:  # bear
                        if cv > fvg_hi:
                            to_remove.append(fvg); continue
                        ce_level = fvg.get("fvg_ce", fvg_lo) if self.require_ce else fvg_lo
                        if not fvg["tested"] and hv >= ce_level:
                            fvg["tested"] = True; fvg["test_bar"] = i
                        if fvg["tested"]:
                            if (i - fvg["test_bar"]) > self.max_entry_wait:
                                to_remove.append(fvg)
                            elif cv < fvg_lo and position == 0:
                                if _needs_session_gate(self._symbol) and not _active_session(self._symbol, hour):
                                    to_remove.append(fvg); continue
                                stop_anchor = fvg["ob_hi"] if fvg["ob_hi"] is not None else fvg_hi
                                sl   = stop_anchor + self.atr_stop_buffer * atr_val
                                dist = sl - cv
                                if dist > 0:
                                    rr               = self._dynamic_rr(fvg["model3"], fvg["trend_s"])
                                    position         = -1
                                    position_size    = 1.0
                                    stop_loss        = sl
                                    initial_sl       = sl
                                    take_profit      = cv - rr * dist
                                    entry_price      = cv
                                    t1_hit           = False
                                    bars_since_entry = 0
                                    self._scores.iloc[i] = fvg["score"]
                                    self._score_reasons.iloc[i] = fvg.get("reasons", [])
                                to_remove.append(fvg)

                for fvg in to_remove:
                    if fvg in active_fvgs:
                        active_fvgs.remove(fvg)

            signals.iloc[i]     = position * position_size
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
            if position != 0 and i > 0 and self._scores.iloc[i] == 0:
                self._scores.iloc[i] = self._scores.iloc[i - 1]
                self._score_reasons.iloc[i] = self._score_reasons.iloc[i - 1]

        return signals

    def _expire_fvgs_neutral(self, active_fvgs: list, cv: float) -> None:
        to_remove = [
            f for f in active_fvgs
            if (f["dir"] == "bull" and cv < f["fvg_lo"]) or
               (f["dir"] == "bear" and cv > f["fvg_hi"])
        ]
        for f in to_remove:
            active_fvgs.remove(f)
