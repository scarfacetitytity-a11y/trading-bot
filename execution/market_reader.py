"""Market Reader — synthesizes all signals into a unified market narrative.

Replaces the score-counter gate in the orchestrator. Every input feeds the
reader. The reader classifies what the market is FORMING and returns a
trade_bias. The bias is the gate — no score floor.

Core logic:
  1. HTF structure (HH/HL = bullish, LH/LL = bearish, sweeps flip the narrative)
  2. Phase (manipulation sweep = strongest, consolidation = stand aside)
  3. Premium/discount (weekly + H4 range position)
  4. Liquidity draw (where is price being pulled)
  5. Order flow + CVD + AMD (institutional fingerprint)
  6. Synthesis → trade_bias with confidence

User's key rule: "higher highs forming = bullish. But flipping a previous high
= that high is now a sell." The reader tracks sweep-and-flip as the highest
conviction signal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from execution.market_phase import Phase, PhaseResult, detect_phase

logger = logging.getLogger(__name__)


@dataclass
class MarketNarrative:
    symbol:        str
    trade_bias:    int         # +1 long, -1 short, 0 no trade
    confidence:    float       # 0.0–1.0
    htf_structure: int         # +1 bullish, -1 bearish, 0 ranging
    phase:         str         # phase name
    sweep_flip:    int         # +1 swept LOW → long narrative, -1 swept HIGH → short, 0 no sweep
    weekly_pos:    int         # +1 discount (good for longs), -1 premium (good for shorts), 0 neutral
    h4_pos:        int         # +1 discount, -1 premium, 0 neutral
    flow_aligned:  bool
    cvd_confirms:  bool
    amd_aligned:   bool
    vote:          int         # raw vote total for debug
    reasons:       list[str]   = field(default_factory=list)
    blocking:      list[str]   = field(default_factory=list)


def _htf_structure(df_m15: pd.DataFrame, lookback: int = 40) -> int:
    """Classify HTF directional structure from swing points over lookback bars.

    HH + HL = bullish (+1). LH + LL = bearish (-1). Mixed = ranging (0).
    Uses two half-windows to compare highs and lows.
    """
    if len(df_m15) < lookback + 2:
        return 0
    w   = df_m15.iloc[-(lookback + 1):-1]
    mid = lookback // 2
    h   = w["high"].values
    l   = w["low"].values
    hh  = h[-1] > h[-mid]
    hl  = l[-1]  > l[-mid]
    lh  = h[-1]  < h[-mid]
    ll  = l[-1]  < l[-mid]
    if hh and hl:
        return 1
    if lh and ll:
        return -1
    return 0


def _h4_range_position(df_m15: pd.DataFrame, desired: int, lookback: int = 40) -> int:
    """Return +1 if price is in discount (good for longs) or premium (good for shorts).

    Discount = bottom 40% of H4 swing. Premium = top 40%. Mid = 0 (neutral).
    """
    if len(df_m15) < lookback + 2:
        return 0
    w  = df_m15.iloc[-(lookback + 1):-1]
    sh = float(w["high"].max())
    sl = float(w["low"].min())
    rng = sh - sl
    if rng == 0:
        return 0
    cv  = float(df_m15["close"].iloc[-2])
    pct = (cv - sl) / rng
    if pct < 0.40:
        return 1   # discount → good for longs
    if pct > 0.60:
        return -1  # premium → good for shorts
    return 0


def read_market(
    symbol:      str,
    desired:     int,
    df_m15:      pd.DataFrame,
    df_m5:       Optional[pd.DataFrame],
    h4_bias:     int,
    atr:         float,
    amd_sweep    = None,
    of_snap      = None,
    cvd_mod:     int   = 0,
    w1_mid:      float = float("nan"),
) -> MarketNarrative:
    """Synthesize all market signals into a narrative. Returns trade_bias."""

    reasons:  list[str] = []
    blocking: list[str] = []
    vote = 0

    # ── 1. Phase ──────────────────────────────────────────────────────────────
    phase_result: PhaseResult = detect_phase(df_m15, h4_bias, atr)
    phase_name = phase_result.phase.value

    # Consolidation = institutional range building. Stand aside, no edge.
    if phase_result.phase == Phase.CONSOLIDATION:
        blocking.append("consolidation_phase")
        return MarketNarrative(
            symbol=symbol, trade_bias=0, confidence=0.0,
            htf_structure=h4_bias, phase=phase_name, sweep_flip=0,
            weekly_pos=0, h4_pos=0, flow_aligned=False, cvd_confirms=False,
            amd_aligned=False, vote=vote, reasons=reasons, blocking=blocking,
        )

    # ── 2. Sweep-and-flip (highest conviction — the user's core insight) ──────
    # A pool was swept: that level now flips the narrative.
    #   Swept HIGH + wick rejection → price rejected premium → SHORT narrative
    #   Swept LOW  + wick rejection → price rejected discount → LONG narrative
    sweep_flip = 0
    if phase_result.sweep_dir != 0:
        # sweep_dir=+1 means highs were swept → narrative flip = SHORT (-1)
        # sweep_dir=-1 means lows were swept  → narrative flip = LONG (+1)
        sweep_flip = -phase_result.sweep_dir
        if sweep_flip == desired:
            vote += 3
            reasons.append(f"sweep-flip: swept {'highs' if phase_result.sweep_dir == 1 else 'lows'} → {'+1 long' if sweep_flip == 1 else '-1 short'} narrative")
        else:
            vote -= 2
            blocking.append(f"sweep-flip_against: swept in opposite direction to signal")

    # ── 3. HTF structure ──────────────────────────────────────────────────────
    htf_struct = _htf_structure(df_m15)
    if htf_struct == desired:
        vote += 2
        reasons.append(f"HTF structure {'bullish' if desired == 1 else 'bearish'} aligns")
    elif htf_struct == -desired:
        vote -= 1
        blocking.append("HTF structure counter-signal")
    # h4_bias as tiebreaker
    if h4_bias == desired:
        vote += 1
        reasons.append("H4 EMA bias aligned")
    elif h4_bias == -desired:
        vote -= 1

    # ── 4. Phase bonus ────────────────────────────────────────────────────────
    phase_votes = {
        "manipulation":  2,   # sweep phase = prime reversal window
        "expansion":     1,   # riding a directional move
        "continuation":  1,   # post-BOS continuation
        "retracement":   1,   # pullback to structure
        "unknown":      -1,
    }
    pv = phase_votes.get(phase_name, 0)
    # Expansion/continuation only support trades aligned to the expansion direction
    if phase_name in ("expansion", "continuation") and htf_struct != 0 and htf_struct != desired:
        pv = -1  # expanding the wrong way
    vote += pv
    if pv > 0:
        reasons.append(f"phase={phase_name} supports {'+long' if desired==1 else '-short'}")
    elif pv < 0:
        blocking.append(f"phase={phase_name} unsupportive")

    # ── 5. Weekly premium/discount (hard filter) ──────────────────────────────
    weekly_pos = 0
    if not np.isnan(w1_mid):
        cv = float(df_m15["close"].iloc[-2])
        if cv > w1_mid:
            weekly_pos = -1   # above weekly mid = premium
        else:
            weekly_pos = 1    # below weekly mid = discount
        if weekly_pos == desired or weekly_pos == 0:
            vote += 1
            reasons.append(f"weekly {'discount' if weekly_pos == 1 else 'neutral'} supports {'+long' if desired==1 else '-short'}")
        else:
            vote -= 2
            blocking.append(f"weekly {'premium' if weekly_pos == -1 else 'discount'} opposes signal")

    # ── 6. H4 range position ──────────────────────────────────────────────────
    h4_pos = _h4_range_position(df_m15, desired)
    if h4_pos == desired:
        vote += 1
        reasons.append(f"H4 price in {'discount' if desired==1 else 'premium'} zone")
    elif h4_pos == -desired:
        vote -= 1
        blocking.append(f"H4 price at counter zone for signal")

    # ── 7. Order flow ─────────────────────────────────────────────────────────
    flow_aligned = False
    if of_snap is not None:
        if of_snap.imbalance_dir == desired or of_snap.dom_bias == desired:
            flow_aligned = True
            vote += 1
            reasons.append("order flow aligned")
        elif of_snap.imbalance_dir == -desired or of_snap.dom_bias == -desired:
            vote -= 1
            blocking.append("order flow opposing")

    # ── 8. CVD divergence ─────────────────────────────────────────────────────
    cvd_confirms = cvd_mod > 0
    if cvd_confirms:
        vote += 1
        reasons.append("CVD divergence confirms")

    # ── 9. AMD sweep alignment ────────────────────────────────────────────────
    amd_aligned = False
    if amd_sweep is not None:
        if amd_sweep.direction == desired:
            amd_aligned = True
            boost = 2 if amd_sweep.strong else 1
            vote += boost
            reasons.append(f"AMD sweep aligned (+{boost})")
        else:
            vote -= 1 if not amd_sweep.strong else 2
            blocking.append(f"AMD sweep opposing {'(strong)' if amd_sweep.strong else ''}")

    # ── 10. Synthesis ─────────────────────────────────────────────────────────
    # Minimum vote of 3 to trade. Confidence scales linearly from 3→8.
    MIN_VOTE = 3
    MAX_VOTE = 8
    if vote >= MIN_VOTE:
        trade_bias = desired
        confidence = min(1.0, (vote - MIN_VOTE) / (MAX_VOTE - MIN_VOTE) + 0.4)
    else:
        trade_bias = 0
        confidence = max(0.0, vote / MAX_VOTE)
        if vote < MIN_VOTE:
            blocking.append(f"insufficient vote ({vote} < {MIN_VOTE})")

    narrative = MarketNarrative(
        symbol=symbol,
        trade_bias=trade_bias,
        confidence=round(confidence, 3),
        htf_structure=htf_struct,
        phase=phase_name,
        sweep_flip=sweep_flip,
        weekly_pos=weekly_pos,
        h4_pos=h4_pos,
        flow_aligned=flow_aligned,
        cvd_confirms=cvd_confirms,
        amd_aligned=amd_aligned,
        vote=vote,
        reasons=reasons,
        blocking=blocking,
    )

    dir_str = "LONG" if desired == 1 else "SHORT"
    verdict = "TRADE" if trade_bias != 0 else "NO_TRADE"
    logger.info(
        "[MarketReader] %s %s | %s | vote=%d | phase=%s | reasons=%s | blocking=%s",
        symbol, dir_str, verdict, vote, phase_name, reasons, blocking,
    )

    return narrative
