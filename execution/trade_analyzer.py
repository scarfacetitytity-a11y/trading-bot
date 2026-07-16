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
MAX_REACH_ATR      = 12.0   # a target beyond this many ATR is not "in reach" this session
EQ_TOLERANCE_ATR   = 0.15   # equal-high/low cluster tolerance
MIN_BEYOND_ATR     = 0.5    # target must sit at least this far beyond entry to count
SWING_ORDER        = 3
LOOKBACK_BARS      = 60


@dataclass
class TradePlan:
    tradeable:   bool
    tp:          float
    rr:          float                 # real RR: (target - entry) / risk_distance
    trade_type:  str                   # continuation | sweep_reversal | breakout | range
    grade:       str                   # A | B | C
    size_mult:   float                 # multiply base risk by this
    target_src:  str                   # equal_highs | equal_lows | swing | atr_fallback
    thesis:      str                   # human-readable WHY
    n_touches:   int = 0               # liquidity-pool touch count (higher = stronger draw)


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

    window = df.tail(LOOKBACK_BARS)
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

    # Prefer equal-level clusters (resting liquidity) over lone swings.
    best_level, best_touches, best_dist = None, 1, float("inf")
    for lvl in cands:
        touches = sum(1 for v in cands if abs(v - lvl) <= tol)
        dist    = abs(lvl - entry)
        # rank: more touches first, then nearer (nearest reachable pool)
        if touches > best_touches or (touches == best_touches and dist < best_dist):
            best_level, best_touches, best_dist = lvl, touches, dist

    src = ("equal_highs" if direction == 1 else "equal_lows") if best_touches >= 2 else "swing"
    return float(best_level), src, int(best_touches)


def analyze_entry(
    df_m15:    pd.DataFrame,
    df_m5:     Optional[pd.DataFrame],
    direction: int,
    entry:     float,
    stop:      float,
    atr:       float,
    h4_bias:   int = 0,
    rr_fallback: float = 2.0,
    swept:     bool = False,
) -> TradePlan:
    """Produce a liquidity-based trade plan. `swept` = a recent sweep/stop-hunt was
    detected at entry (marks a reversal thesis)."""
    dist = abs(entry - stop)
    if dist <= 1e-9:
        return TradePlan(False, entry, 0.0, "range", "C", 0.0, "atr_fallback",
                         "invalid stop distance", 0)

    # Prefer M5 for a precise near-term draw, fall back to M15 structure.
    tgt, src, touches = _find_target(df_m5, direction, entry, atr)
    if tgt is None:
        tgt, src, touches = _find_target(df_m15, direction, entry, atr)

    if tgt is not None:
        rr = abs(tgt - entry) / dist
    else:
        rr  = 0.0

    # ── Classify trade type ──
    if swept:
        trade_type = "sweep_reversal"        # entered on a stop-hunt reversal
    elif h4_bias == direction:
        trade_type = "continuation"          # riding the HTF draw
    elif tgt is not None and touches >= 2:
        trade_type = "breakout"              # aiming at a clean liquidity pool
    else:
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
            tradeable=False, tp=round(tp, 6), rr=capped_rr, trade_type=trade_type,
            grade="C", size_mult=0.25, target_src="atr_fallback",
            thesis=thesis, n_touches=touches,
        )

    # Real target found. Grade on draw strength + RR + HTF alignment.
    aligned = (h4_bias == direction) or swept
    if touches >= 2 and rr >= 2.0 and aligned:
        grade, size_mult = "A", 1.25
    elif rr >= 1.5 and (touches >= 2 or aligned):
        grade, size_mult = "B", 1.0
    else:
        grade, size_mult = "B", 0.75

    pool = f"{touches}-touch {src}" if touches >= 2 else "swing level"
    thesis = (f"{trade_type}: draw to {pool} @ {tgt:.5f} "
              f"(RR {rr:.2f}, H4 {'aligned' if aligned else 'neutral'})")

    return TradePlan(
        tradeable=True, tp=round(tgt, 6), rr=round(rr, 2), trade_type=trade_type,
        grade=grade, size_mult=size_mult, target_src=src,
        thesis=thesis, n_touches=touches,
    )


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

    # Thesis valid if the trade paid at least +1R of favourable movement.
    thesis_valid = mfe_R >= 1.0

    notes = []
    if mfe_R >= 2.0 and exit_R < 1.0:
        notes.append(f"gave back a {mfe_R:.1f}R winner — exit/trail too loose")
    if not thesis_valid and reason == "SL":
        notes.append("stopped with <1R favourable — entry too early or wrong draw")
    if hit_target:
        notes.append("reached liquidity target — thesis confirmed")
    if mae_R < -0.9 and exit_R > 0:
        notes.append(f"dipped to {mae_R:.1f}R before working — stop was well placed")

    if hit_target:
        lesson = "target hit — repeat this setup profile"
    elif thesis_valid:
        lesson = "right direction, exit management left R on the table"
    else:
        lesson = "thesis failed — draw/target read was wrong, review entry context"

    return TradeReview(
        hit_target=hit_target, exit_R=round(exit_R, 2),
        mfe_R=round(mfe_R, 2), mae_R=round(mae_R, 2),
        thesis_valid=thesis_valid, lesson=lesson, notes=notes,
    )
