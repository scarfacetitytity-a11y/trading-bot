"""Per-trade liquidity analyst — the thesis layer above the entry signal.

The entry model (strategies/aiden_index.py) decides WHEN structure says go. This
module decides WHETHER the trade is worth taking and WHERE it's actually going —
by reading liquidity, not doing arithmetic.

Core principle (Anton, 2026-07-16): every trade is liquidity-based. The target is
a real liquidity pool price is drawn toward — equal highs/lows, a prior swing that
holds resting orders — NOT `entry +/- rr_target * stop`. Scores and confluences
sit ON TOP of that foundation; they don't replace it.

Two entry points:
  analyze_entry() -> TradePlan   : target (TP) at real liquidity, trade type,
                                    quality grade, size multiplier, written thesis.
                                    Grade C (no clean draw within reach) -> shrink
                                    to min size or skip. This is what stops the
                                    "0.1 lot gold short with TP 9% away" trade.
  analyze_exit()  -> TradeReview : post-mortem on a closed trade — did it reach the
                                    liquidity target, was the thesis valid, MFE/MAE,
                                    one lesson. Feeds the journal so the system learns.

Pure functions on OHLC DataFrames — no MT5 dependency, unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from execution.signal_detectors import _swing_highs, _swing_lows


# ── Tunables ──────────────────────────────────────────────────────────────────
MIN_RR_TRADEABLE   = 1.2    # below this, the draw is too close to be worth the risk
MAX_REACH_ATR      = 5.0    # a target beyond this many ATR is not reachable this session
EQ_TOLERANCE_ATR   = 0.15   # equal-high/low cluster tolerance
MIN_BEYOND_ATR     = 0.5    # target must sit at least this far beyond entry to count
SWING_ORDER           = 3
LOOKBACK_BARS         = 60
TARGET_LOOKBACK_BARS  = 120   # wider window for liquidity target discovery


MAX_STOP_ATR       = 6.0    # a structural stop beyond this is too wide — invalid setup
MIN_STOP_ATR       = 0.5    # a structural stop nearer than this sits inside the noise
STOP_BUFFER_ATR    = 0.20   # padding beyond the structural level for slippage

# Per-instrument minimum stop distance in points — prevents spread eating the SL
# USDJPY/JPY pairs: typical spread 0.5-1 pip = 5-10 points; need 15pt clearance minimum
_MIN_STOP_POINTS: dict[str, float] = {
    "USDJPY": 0.150,    # 15 pips minimum beyond structure
    "AUDJPY": 0.150,    # JPY cross — same pip scaling as USDJPY
    "NZDJPY": 0.150,
    "GBPJPY": 0.200,    # GBPJPY wider — higher vol, spread larger
    "JP225.cash": 5.0,
    "HK50.cash":  5.0,
    "XAUUSD":     0.50,
    "XAGUSD":     0.05,
}


@dataclass
class TradePlan:
    tradeable:   bool
    tp:          float
    stop:        float                 # STRUCTURAL stop (not ATR distance)
    rr:          float                 # real RR: (target - entry) / (entry - stop)
    trade_type:  str                   # continuation | sweep_reversal | breakout | range
    grade:       str                   # A | B | C
    size_mult:   float                 # multiply base risk by this
    target_src:  str                   # equal_highs | equal_lows | swing_high | swing_low | fvg_ce | atr_fallback
    stop_src:    str                   # sweep | swing_low/high | order_block | atr_fallback
    thesis:      str                   # human-readable WHY
    n_touches:   int = 0               # liquidity-pool touch count (higher = stronger draw)
    ote_aligned: bool = False          # price is in the ICT 62-79% Fibonacci OTE zone
    breaker_block: bool = False        # entry is at a breached-OB breaker block zone
    # Gap 12 — T1 at first structural obstacle between entry and main target.
    # JP mentor: "when it hits a level, it's going to react — go risk-free there."
    # BE is triggered at t1_price rather than at a fixed R multiple.
    t1_price:    Optional[float] = None   # nearest structural level between entry and tp
    t1_src:      str = ""


@dataclass
class TradeReview:
    hit_target:   bool
    exit_R:       float
    mfe_R:        float                # max favourable excursion in R
    mae_R:        float                # max adverse excursion in R
    thesis_valid: bool
    lesson:       str
    notes:        list = field(default_factory=list)


# ── Liquidity target finder ───────────────────────────────────────────────────

def _find_target(
    df: pd.DataFrame, direction: int, entry: float, atr: float,
) -> tuple[Optional[float], str, int]:
    """Return (target_price, source, n_touches) — the liquidity level price is
    drawn toward in the trade direction, or (None, 'none', 0)."""
    if df is None or len(df) < LOOKBACK_BARS or atr <= 0:
        return None, "none", 0

    window = df.tail(TARGET_LOOKBACK_BARS)
    tol    = atr * EQ_TOLERANCE_ATR
    beyond = atr * MIN_BEYOND_ATR
    reach  = atr * MAX_REACH_ATR

    if direction == 1:
        sh   = _swing_highs(window["high"], order=SWING_ORDER)
        vals = window["high"][sh].values
        cands = [h for h in vals if entry + beyond <= h <= entry + reach]
    else:
        sl   = _swing_lows(window["low"], order=SWING_ORDER)
        vals = window["low"][sl].values
        cands = [l for l in vals if entry - reach <= l <= entry - beyond]

    if not cands:
        return None, "none", 0

    # Primary: equal-level cluster (2+ touches) — resting order pool, strongest draw.
    best_level, best_touches, best_dist = None, 0, float("inf")
    for lvl in cands:
        touches = sum(1 for v in cands if abs(v - lvl) <= tol)
        dist    = abs(lvl - entry)
        if touches >= 2 and (touches > best_touches or (touches == best_touches and dist < best_dist)):
            best_level, best_touches, best_dist = lvl, touches, dist

    if best_level is not None:
        src = "equal_highs" if direction == 1 else "equal_lows"
        return float(best_level), src, int(best_touches)

    # Secondary: nearest single swing within reach (PDH/PDL-tier structure).
    # Single-touch significant swing: valid IPDA target, just lower conviction than a cluster.
    nearest = min(cands, key=lambda x: abs(x - entry))
    src = "swing_high" if direction == 1 else "swing_low"
    return float(nearest), src, 1


def _find_t1(
    df: pd.DataFrame, direction: int, entry: float, target: float, atr: float,
) -> tuple[Optional[float], str]:
    """Return (t1_price, source) — the nearest structural level between entry and
    the main target that price is likely to react at on the way there.

    JP mentor: "when it hits a level, it's going to react — go risk-free there."
    Used as a structural BE trigger instead of a fixed R multiple.
    Returns None if no qualifying level exists between entry and target.
    """
    if df is None or len(df) < LOOKBACK_BARS or atr <= 0 or abs(target - entry) < atr * 0.5:
        return None, ""

    window  = df.tail(LOOKBACK_BARS)
    min_gap = atr * MIN_BEYOND_ATR    # level must be meaningfully beyond entry
    max_gap = abs(target - entry)     # cap at the main target — don't look past it

    if direction == 1:
        sh   = _swing_highs(window["high"], order=SWING_ORDER)
        vals = window["high"][sh].values
        # levels between entry+min_gap and target
        cands = [h for h in vals if entry + min_gap <= h <= target]
    else:
        sl   = _swing_lows(window["low"], order=SWING_ORDER)
        vals = window["low"][sl].values
        cands = [l for l in vals if target <= l <= entry - min_gap]

    if not cands:
        return None, ""

    # Nearest structural level — the first obstacle price faces
    nearest = min(cands, key=lambda x: abs(x - entry))
    src = "swing_t1"
    return float(nearest), src


def _find_structural_stop(
    df: pd.DataFrame, direction: int, entry: float, atr: float,
    trade_type: str, swept: bool,
) -> tuple[Optional[float], str]:
    """Return (stop_price, source) placed at real structure that invalidates the
    trade — NOT an ATR distance. The level differs by trade type:

      sweep_reversal : beyond the swept wick (the stop-hunt extreme) — if it trades
                       back through, the sweep failed.
      continuation   : beyond the last swing that must hold (higher-low / lower-high).
      breakout       : beyond the broken level (now flipped support/resistance).

    ATR is used only as a small slippage buffer, never as the stop distance itself.
    """
    if df is None or len(df) < LOOKBACK_BARS or atr <= 0:
        return None, "none"

    window = df.tail(LOOKBACK_BARS)
    buf    = atr * STOP_BUFFER_ATR
    max_d  = atr * MAX_STOP_ATR
    min_d  = atr * MIN_STOP_ATR

    if direction == 1:
        # candidate structural lows below entry
        sl_mask = _swing_lows(window["low"], order=SWING_ORDER)
        lows    = window["low"][sl_mask].values
        if swept:
            # deepest recent low (the wick that grabbed liquidity)
            cand = float(window["low"].tail(SWING_ORDER * 2 + 2).min())
            src  = "sweep_low"
        else:
            # nearest swing low that is far enough to sit outside the noise
            below = [l for l in lows if l < entry - min_d]
            if not below:
                return None, "none"
            cand = float(max(below))     # nearest qualifying swing low that must hold
            src  = "swing_low"
        stop = cand - buf
        if stop >= entry:
            return None, "none"
        if entry - stop > max_d:
            return None, "too_wide"
        if entry - stop < min_d:          # too tight even after picking structure
            stop = entry - min_d
            src  = src + "+floor"
        return stop, src
    else:
        sh_mask = _swing_highs(window["high"], order=SWING_ORDER)
        highs   = window["high"][sh_mask].values
        if swept:
            cand = float(window["high"].tail(SWING_ORDER * 2 + 2).max())
            src  = "sweep_high"
        else:
            above = [h for h in highs if h > entry + min_d]
            if not above:
                return None, "none"
            cand = float(min(above))
            src  = "swing_high"
        stop = cand + buf
        if stop <= entry:
            return None, "none"
        if stop - entry > max_d:
            return None, "too_wide"
        if stop - entry < min_d:
            stop = entry + min_d
            src  = src + "+floor"
        return stop, src


def _find_order_block(
    df: pd.DataFrame, direction: int, entry: float, atr: float,
) -> tuple[Optional[float], str]:
    """Order block = the last opposite-colour candle before the move. It's a
    stronger stop anchor than a bare swing: if price trades back through the OB
    that launched the move, the thesis is dead.

    LONG  : last bearish candle (close<open) below entry -> stop below its low.
    SHORT : last bullish candle (close>open) above entry -> stop above its high.
    """
    if df is None or len(df) < 10 or atr <= 0:
        return None, "none"
    window = df.tail(LOOKBACK_BARS)
    o = window["open"].values; c = window["close"].values
    h = window["high"].values; l = window["low"].values
    min_d = atr * MIN_STOP_ATR
    max_d = atr * MAX_STOP_ATR

    if direction == 1:
        for i in range(len(window) - 2, -1, -1):        # scan backward
            if c[i] < o[i] and l[i] < entry - min_d:     # bearish candle below entry
                dist = entry - l[i]
                if dist <= max_d:
                    return float(l[i]), "order_block"
                return None, "too_wide"
    else:
        for i in range(len(window) - 2, -1, -1):
            if c[i] > o[i] and h[i] > entry + min_d:     # bullish candle above entry
                dist = h[i] - entry
                if dist <= max_d:
                    return float(h[i]), "order_block"
                return None, "too_wide"
    return None, "none"


def _find_fvg_target(
    df: pd.DataFrame, direction: int, entry: float, atr: float,
) -> Optional[float]:
    """Nearest unfilled Fair Value Gap in the trade direction beyond entry — an
    imbalance price is drawn to fill (a target candidate alongside liquidity pools).

    LONG  : bullish FVG (bar[-3].high < bar[-1].low) with the gap above entry.
    SHORT : bearish FVG (bar[-3].low  > bar[-1].high) with the gap below entry.
    """
    if df is None or len(df) < 5 or atr <= 0:
        return None
    h = df["high"].values; l = df["low"].values
    best = None
    for i in range(2, len(df)):
        if direction == 1:
            if h[i - 2] < l[i]:                              # bullish FVG
                ce = (h[i - 2] + l[i]) / 2                  # CE — 50% of gap
                if ce > entry + atr * MIN_BEYOND_ATR:
                    if best is None or ce < best:             # nearest above
                        best = ce
        else:
            if l[i - 2] > h[i]:                              # bearish FVG
                ce = (l[i - 2] + h[i]) / 2                  # CE — 50% of gap
                if ce < entry - atr * MIN_BEYOND_ATR:
                    if best is None or ce > best:             # nearest below
                        best = ce
    return float(best) if best is not None else None


def _ote_check(
    df: pd.DataFrame, direction: int, entry: float, atr: float, lookback: int = 30,
) -> bool:
    """True if entry sits in the ICT 62-79% Fibonacci OTE retracement zone of the
    most recent significant swing. Institutional re-entry zone after a BOS impulse."""
    if df is None or len(df) < lookback + 2 or atr <= 0:
        return False
    window = df.iloc[-(lookback + 1):-1]
    if direction == 1:
        A = float(window["low"].min())
        B = float(window["high"].max())
        if B - A < atr * 0.5:
            return False
        ote_hi = B - (B - A) * 0.62
        ote_lo = B - (B - A) * 0.79
    else:
        A = float(window["high"].max())
        B = float(window["low"].min())
        if A - B < atr * 0.5:
            return False
        ote_lo = B + (A - B) * 0.62
        ote_hi = B + (A - B) * 0.79
    return ote_lo <= entry <= ote_hi


def _breaker_block_check(
    df: pd.DataFrame, direction: int, entry: float, atr: float, lookback: int = 40,
) -> bool:
    """True if a breached OB is being retested at the entry price from the other side.

    Bullish breaker: a prior bearish candle whose high was later closed above by a BOS
    candle, and price is now retesting its body zone from above.
    Bearish breaker: a prior bullish candle whose low was later closed below, now retested from below.
    """
    if df is None or len(df) < lookback + 2 or atr <= 0:
        return False
    window = df.iloc[-(lookback + 1):-1]
    o = window["open"].values
    c = window["close"].values
    h = window["high"].values
    l = window["low"].values
    tol = atr * 0.1

    n = len(o)
    if direction == 1:
        for i in range(n - 4):
            if c[i] < o[i]:  # bearish candle — potential breaker
                ob_hi = h[i]; ob_lo = l[i]
                # Check if a later BOS closed above this candle's high
                bos = any(c[j] > ob_hi for j in range(i + 1, n))
                if bos and ob_lo - tol <= entry <= ob_hi + tol:
                    return True
    else:
        for i in range(n - 4):
            if c[i] > o[i]:  # bullish candle — potential breaker
                ob_hi = h[i]; ob_lo = l[i]
                bos = any(c[j] < ob_lo for j in range(i + 1, n))
                if bos and ob_lo - tol <= entry <= ob_hi + tol:
                    return True
    return False


def analyze_entry(
    df_m15:    pd.DataFrame,
    df_m5:     Optional[pd.DataFrame],
    direction: int,
    entry:     float,
    stop:      float,                  # reference/fallback stop only (e.g. strategy ATR)
    atr:       float,
    h4_bias:   int = 0,
    rr_fallback: float = 2.0,
    swept:     bool = False,
    symbol:    str = "",
) -> TradePlan:
    """Produce a fully structural trade plan — stop AND target from levels, not
    arithmetic. `stop` is a fallback reference only. `swept` = a recent sweep/
    stop-hunt was detected at entry (marks a reversal thesis)."""
    ref_dist = abs(entry - stop)

    # ── Classify trade type first (drives stop placement) ──
    if swept:
        trade_type = "sweep_reversal"        # entered on a stop-hunt reversal
    elif h4_bias == direction:
        trade_type = "continuation"          # riding the HTF draw
    else:
        trade_type = "breakout"

    # ── Structural stop: order block (strongest) > swing > reference ──
    # Prefer the order block that launched the move; fall back to swing structure.
    def _stop_from(df):
        # Order block is the launch level — highest-conviction anchor. Use it when
        # valid; otherwise fall back to swing structure.
        ob, ob_src = _find_order_block(df, direction, entry, atr)
        if ob is not None:
            return ob - atr * STOP_BUFFER_ATR if direction == 1 else ob + atr * STOP_BUFFER_ATR, ob_src
        return _find_structural_stop(df, direction, entry, atr, trade_type, swept)

    s_stop, stop_src = _stop_from(df_m5)
    if s_stop is None:
        s_stop, stop_src = _stop_from(df_m15)
    if s_stop is None:
        s_stop, stop_src = stop, "atr_fallback"

    dist = abs(entry - s_stop)

    # Per-instrument minimum stop distance — prevents spread eating through the SL
    # on tight-spread instruments (JPY pairs, indices at open)
    _min_pts = _MIN_STOP_POINTS.get(symbol, 0.0)
    if _min_pts > 0 and dist < _min_pts:
        extra = _min_pts - dist
        s_stop = (s_stop - extra) if direction == 1 else (s_stop + extra)
        dist = abs(entry - s_stop)
    if dist <= 1e-9:
        return TradePlan(False, entry, s_stop, 0.0, "range", "C", 0.0,
                         "atr_fallback", stop_src, "invalid stop distance", 0)

    # ── Target: liquidity pool (equal highs/lows) or unfilled FVG to fill ──
    # Prefer M5 precise draw, fall back to M15. Between a liquidity pool and an
    # FVG, take whichever is a valid draw that clears the min-RR bar; prefer the
    # pool (resting orders) when both qualify.
    tgt, src, touches = _find_target(df_m5, direction, entry, atr)
    if tgt is None:
        tgt, src, touches = _find_target(df_m15, direction, entry, atr)

    fvg = _find_fvg_target(df_m5, direction, entry, atr)
    if fvg is None:
        fvg = _find_fvg_target(df_m15, direction, entry, atr)

    # If no liquidity pool, or the FVG is a nearer valid draw, use the FVG.
    if fvg is not None:
        fvg_rr = abs(fvg - entry) / dist
        pool_ok = tgt is not None and (abs(tgt - entry) / dist) >= MIN_RR_TRADEABLE
        if not pool_ok and fvg_rr >= MIN_RR_TRADEABLE:
            tgt, src, touches = fvg, "fvg_ce", 0

    rr = abs(tgt - entry) / dist if tgt is not None else 0.0

    # refine type: aiming at a clean pool with no HTF alignment = breakout
    if not swept and h4_bias != direction and tgt is not None and touches >= 2:
        trade_type = "breakout"
    elif not swept and h4_bias != direction and tgt is None:
        trade_type = "range"

    # ── Grade & size ──
    if tgt is None or rr < MIN_RR_TRADEABLE:
        # No real draw within reach — this is the "stupid gold trade". Do NOT place
        # a blind arithmetic TP at full size. Cap TP conservatively, min size.
        capped_rr = max(1.0, min(rr_fallback, 1.5))
        tp = (entry + capped_rr * dist) if direction == 1 else (entry - capped_rr * dist)
        thesis = (f"NO clean liquidity draw within {MAX_REACH_ATR:.0f}xATR "
                  f"(best RR {rr:.2f}) — low-conviction, min size")
        return TradePlan(
            tradeable=False, tp=round(tp, 6), stop=round(s_stop, 6), rr=capped_rr,
            trade_type=trade_type, grade="C", size_mult=0.25,
            target_src="atr_fallback", stop_src=stop_src,
            thesis=thesis, n_touches=touches,
            ote_aligned=_ote_check(df_m15, direction, entry, atr),
            breaker_block=_breaker_block_check(df_m15, direction, entry, atr),
        )

    # Real target found. Grade on draw strength + RR + HTF alignment + stop quality.
    aligned    = (h4_bias == direction) or swept
    struct_stop = stop_src != "atr_fallback"     # stop anchored to real structure?
    if touches >= 2 and rr >= 2.0 and aligned and struct_stop:
        grade, size_mult = "A", 1.25
    elif rr >= 1.5 and (touches >= 2 or aligned) and struct_stop:
        grade, size_mult = "B", 1.0
    elif struct_stop:
        grade, size_mult = "B", 0.75
    else:
        grade, size_mult = "C", 0.5           # target ok but stop not structural

    pool = f"{touches}-touch {src}" if touches >= 2 else "swing level"
    thesis = (f"{trade_type}: stop@{stop_src} {s_stop:.5f} -> draw to {pool} @ {tgt:.5f} "
              f"(RR {rr:.2f}, H4 {'aligned' if aligned else 'neutral'})")

    # Gap 12 — T1 at first structural obstacle between entry and main target.
    # Use M5 first (more granular), fall back to M15.
    t1_px, t1_src = _find_t1(df_m5, direction, entry, tgt, atr)
    if t1_px is None:
        t1_px, t1_src = _find_t1(df_m15, direction, entry, tgt, atr)

    ote = _ote_check(df_m15, direction, entry, atr)
    bb  = _breaker_block_check(df_m15, direction, entry, atr)

    return TradePlan(
        tradeable=(grade != "C"), tp=round(tgt, 6), stop=round(s_stop, 6),
        rr=round(rr, 2), trade_type=trade_type, grade=grade, size_mult=size_mult,
        target_src=src, stop_src=stop_src, thesis=thesis, n_touches=touches,
        ote_aligned=ote, breaker_block=bb,
        t1_price=round(t1_px, 6) if t1_px is not None else None, t1_src=t1_src,
    )


# ── Adaptive live management ──────────────────────────────────────────────────

@dataclass
class ManageDecision:
    new_sl:  Optional[float] = None    # tighten-only structural trail
    new_tp:  Optional[float] = None    # extend target to next draw
    reason:  str = ""


def manage_trade(
    df:          pd.DataFrame,
    direction:   int,
    entry:       float,
    initial_sl:  float,
    current_sl:  float,
    current_tp:  float,
    price:       float,
    atr:         float,
    trade_type:  str,
    cur_r:       float,
    bank_min_r:  float = 0.0,
    asian_50:    Optional[float] = None,
    t1_price:    Optional[float] = None,
) -> ManageDecision:
    """Adaptive per-type management. Moves the stop to new structure as the trade
    develops (tighten-only) and extends the target to the next draw when price
    approaches it. Different behaviour per trade type:

      sweep_reversal : work-fast — BE by +0.5R, then trail tight behind structure.
      continuation   : ride — BE by +1R, trail behind each new higher-low/OB,
                       extend TP to the next liquidity pool.
      breakout       : BE by +1R, trail behind the reclaimed level.

    asian_50: if price reaches the Asian session 50% midpoint in the trade direction,
    go risk-free immediately — JP mentor v9: "50% of the Asian session range is a
    point of interest; I made it risk-free there." It's a known reaction level.
    """
    if df is None or len(df) < 20 or atr <= 0 or abs(entry - initial_sl) < 1e-9:
        return ManageDecision()

    rdist = abs(entry - initial_sl)
    dec   = ManageDecision()

    # ── 0. Asian session 50% → immediate breakeven (structural reaction point) ──
    # JP v9: "50% of the Asian session range is a point of interest — made myself
    # risk-free there." Trigger before the R-based ratchet so it fires first.
    if asian_50 is not None:
        reached = (direction == 1 and price >= asian_50) or \
                  (direction == -1 and price <= asian_50)
        if reached:
            be = entry
            if direction == 1 and be > current_sl:
                dec.new_sl = be
                dec.reason = "BE@Asian50%"
            elif direction == -1 and be < current_sl:
                dec.new_sl = be
                dec.reason = "BE@Asian50%"

    # ── 0b. T1 structural POI → breakeven (Gap 12) ──
    # JP mentor: "when it hits a level, it's going to react — go risk-free there."
    # Fires before the R-based ratchet: structural level takes priority over arithmetic.
    if t1_price is not None and dec.new_sl is None:
        t1_reached = (direction == 1 and price >= t1_price) or \
                     (direction == -1 and price <= t1_price)
        if t1_reached:
            be = entry
            if direction == 1 and be > current_sl:
                dec.new_sl = be
                dec.reason = "BE@T1_POI"
            elif direction == -1 and be < current_sl:
                dec.new_sl = be
                dec.reason = "BE@T1_POI"

    # ── 1. Breakeven ratchet (type-specific trigger) ──
    be_trigger = 0.5 if trade_type == "sweep_reversal" else 1.0
    if cur_r >= be_trigger:
        be = entry
        if direction == 1 and be > current_sl:
            dec.new_sl, dec.reason = be, f"BE@{be_trigger}R"
        elif direction == -1 and be < current_sl:
            dec.new_sl, dec.reason = be, f"BE@{be_trigger}R"

    # ── 2. Structural trail behind the latest swing that must hold ──
    if cur_r >= 1.0:
        s_stop, s_src = _find_structural_stop(df, direction, price, atr, trade_type, False)
        if s_stop is not None:
            better = ((direction == 1 and s_stop > (dec.new_sl or current_sl)) or
                      (direction == -1 and s_stop < (dec.new_sl or current_sl)))
            # never trail past price
            valid = (direction == 1 and s_stop < price) or (direction == -1 and s_stop > price)
            if better and valid:
                dec.new_sl = s_stop
                dec.reason = f"trail@{s_src}"

    # ── 3. Target extension — let it run to the next draw ──
    # If price is within 0.3R of TP and a further pool/FVG exists, push TP out.
    near_tp = abs(price - current_tp) <= 0.3 * rdist
    beyond_target = (direction == 1 and price >= current_tp - 0.3 * rdist) or \
                    (direction == -1 and price <= current_tp + 0.3 * rdist)
    if near_tp or beyond_target:
        nxt, src, _ = _find_target(df, direction, price, atr)
        if nxt is None:
            fvg = _find_fvg_target(df, direction, price, atr)
            nxt, src = (fvg, "fvg_fill") if fvg is not None else (None, "")
        if nxt is not None:
            extends = (direction == 1 and nxt > current_tp) or (direction == -1 and nxt < current_tp)
            if extends:
                dec.new_tp = round(nxt, 6)
                dec.reason = (dec.reason + " | " if dec.reason else "") + f"extend_tp@{src}"

    # ── 4. Bank-in (behaviour 4) — pull TP to a NEARER draw when deep in profit ──
    # Opt-in (bank_min_r>0) and guarded: only when well in profit and a real pool
    # sits ahead but closer than the current far target, so we bank at liquidity
    # instead of risking a full round-trip. Cuts a winner short by design — the
    # bank_min_r threshold is UNVALIDATED; tune it against the bar-level sim
    # before enabling live (edge-sensitive; see backtest-reality-gap).
    if bank_min_r > 0 and cur_r >= bank_min_r and dec.new_tp is None:
        nxt, src, _ = _find_target(df, direction, price, atr)
        if nxt is not None:
            ahead  = (direction == 1 and nxt > price + 0.1 * rdist) or \
                     (direction == -1 and nxt < price - 0.1 * rdist)
            nearer = (direction == 1 and nxt < current_tp) or \
                     (direction == -1 and nxt > current_tp)
            if ahead and nearer:
                dec.new_tp = round(nxt, 6)
                dec.reason = (dec.reason + " | " if dec.reason else "") + f"bank_tp@{src}"

    return dec


# ── Post-trade review ─────────────────────────────────────────────────────────

def analyze_exit(
    direction:  int,
    entry:      float,
    stop:       float,
    target:     float,
    exit_px:    float,
    path_high:  float,
    path_low:   float,
    reason:     str,
) -> TradeReview:
    """Post-mortem a closed trade. path_high/path_low = extreme prices seen while open."""
    dist = abs(entry - stop)
    if dist <= 1e-9:
        return TradeReview(False, 0.0, 0.0, 0.0, False, "invalid stop distance")

    def _R(px):
        move = (px - entry) if direction == 1 else (entry - px)
        return move / dist

    exit_R = _R(exit_px)
    mfe_R  = _R(path_high) if direction == 1 else _R(path_low)   # best it reached
    mae_R  = _R(path_low)  if direction == 1 else _R(path_high)  # worst it reached

    hit_target = ((direction == 1 and path_high >= target) or
                  (direction == -1 and path_low <= target))

    # Thesis valid if price moved at least 0.25R in the right direction.
    # Using 1.0R caused every early-exited trade to show thesis_valid=False even
    # when direction was correct — exit bugs were masking real thesis quality.
    # 0.3 flagged a 0.28-mfe trade as invalid (boundary noise) — 0.25 keeps the
    # binary honest until graded thesis quality replaces it.
    thesis_valid = mfe_R >= 0.25

    notes = []
    if mfe_R >= 2.0 and exit_R < 1.0:
        notes.append(f"gave back a {mfe_R:.1f}R winner — exit/trail too loose")
    if mae_R < -0.9 and exit_R > 0:
        notes.append(f"dipped to {mae_R:.1f}R before working — stop was well placed")
    if hit_target:
        notes.append("reached liquidity target — thesis confirmed")
    if not thesis_valid and reason == "SL":
        notes.append("never moved in our direction — entry or draw was wrong")
    if thesis_valid and not hit_target and exit_R < 0:
        notes.append("right direction but exit cut it short — management issue not thesis")

    # Specific, contextual lessons — not one static string
    if hit_target:
        lesson = "target hit — repeat this setup profile; same level/structure quality"
    elif thesis_valid and mfe_R >= 1.5:
        lesson = f"trade worked to {mfe_R:.1f}R then exited at {exit_R:.1f}R — tighten trail or extend TP earlier"
    elif thesis_valid and mfe_R >= 0.5:
        lesson = f"correct direction ({mfe_R:.1f}R MFE), early exit killed it — management or entry-cooldown bug"
    elif thesis_valid:
        lesson = f"directionally correct but only {mfe_R:.1f}R — entry was too early or level was too weak"
    elif mae_R < -0.7:
        lesson = f"never worked: dipped straight to {mae_R:.1f}R — draw/target read wrong or entry against HTF level"
    else:
        lesson = f"flat/neutral from entry (MFE {mfe_R:.1f}R, MAE {mae_R:.1f}R) — no thesis, skip this setup profile"

    return TradeReview(
        hit_target=hit_target, exit_R=round(exit_R, 2),
        mfe_R=round(mfe_R, 2), mae_R=round(mae_R, 2),
        thesis_valid=thesis_valid, lesson=lesson, notes=notes,
    )
