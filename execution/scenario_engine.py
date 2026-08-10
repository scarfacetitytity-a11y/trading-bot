"""Scenario Engine — reads market phase and classifies which trade scenario is active.

Replaces score-counting with context-pattern matching. Each scenario requires
2-3 specific confirmations. If context + confirmations match → scenario GO.

Scenarios (from trading-os entry models + ICT/JP mentor framework):

  S1 POST_SWEEP_REVERSAL   — liquidity swept, wick rejection, FVG/OB within 2xATR
  S2 OB_FVG_RETRACEMENT    — HTF expansion complete, price retesting OB+FVG zone
  S3 CONTINUATION_PULLBACK — after BOS, price pulling back to displacement FVG
  S4 RANGE_SWEEP_FADE      — consolidation range, price swept boundary, fade back inside

Running in SHADOW mode: logs scenario output to logs/scenario_shadow.jsonl
but does NOT affect live trade decisions. Shadow period until validated against
live outcomes over ≥30 trades.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from execution.market_phase import Phase, PhaseResult, detect_phase

logger = logging.getLogger(__name__)

SHADOW_LOG = Path("logs") / "scenario_shadow.jsonl"
SHADOW_MODE = False  # authoritative — scenario engine controls trade qualification


@dataclass
class ScenarioResult:
    symbol:       str
    direction:    int           # +1 long / -1 short
    scenario:     str           # scenario name or "NO_SCENARIO"
    verdict:      str           # "GO" | "NO_GO" | "SHADOW_GO" | "SHADOW_NO_GO"
    phase:        str
    confirmations: list[str]   # which confirmations fired
    missing:      list[str]    # which required confirmations are absent
    confidence:   float
    ts:           str


# ── Confirmation helpers ──────────────────────────────────────────────────────

def _fvg_nearby(df: pd.DataFrame, direction: int, atr: float, lookback: int = 12) -> bool:
    """Returns True if a matching-direction FVG formed within lookback bars."""
    bars = df.iloc[-(lookback + 3):-1]
    for i in range(2, len(bars) - 1):
        h2  = float(bars["high"].iloc[i - 2])
        l2  = float(bars["low"].iloc[i - 2])
        lv  = float(bars["low"].iloc[i])
        hv  = float(bars["high"].iloc[i])
        if direction == 1 and (lv - h2) > 0.2 * atr:   # bullish FVG gap
            return True
        if direction == -1 and (l2 - hv) > 0.2 * atr:  # bearish FVG gap
            return True
    return False


def _ob_nearby(df: pd.DataFrame, direction: int, atr: float, lookback: int = 12) -> bool:
    """Returns True if price is within 1xATR of a valid OB candle."""
    bars = df.iloc[-(lookback + 2):-1]
    cv   = float(df["close"].iloc[-2])
    for i in range(1, len(bars) - 1):
        c0  = float(bars["close"].iloc[i - 1])
        c1  = float(bars["close"].iloc[i])
        h0  = float(bars["high"].iloc[i - 1])
        l0  = float(bars["low"].iloc[i - 1])
        w0  = h0 - l0
        wick_pct = (h0 - max(c0, c1)) / w0 if w0 > 0 else 0
        # Bullish OB: bearish candle with wick, then engulfed by bullish candle
        if direction == 1 and c0 < c1 and wick_pct > 0.1:
            if abs(cv - h0) < atr:
                return True
        # Bearish OB
        if direction == -1 and c0 > c1 and wick_pct > 0.1:
            if abs(cv - l0) < atr:
                return True
    return False


def _wick_rejection(df: pd.DataFrame, direction: int) -> bool:
    """Last closed bar has a significant rejection wick in trade direction."""
    bar   = df.iloc[-2]
    o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
    rng   = h - l
    if rng == 0:
        return False
    if direction == 1:
        lower_wick = min(o, c) - l
        return lower_wick / rng > 0.4
    else:
        upper_wick = h - max(o, c)
        return upper_wick / rng > 0.4


def _ltf_choch(df_m5: Optional[pd.DataFrame], direction: int) -> bool:
    """M5 CHoCH — price broke a prior swing in the trade direction on M5."""
    if df_m5 is None or len(df_m5) < 10:
        return False
    recent = df_m5.iloc[-10:-1]
    if direction == 1:
        # Look for a bullish BOS: close above prior swing high
        mid_high = float(recent["high"].iloc[:5].max())
        return float(recent["close"].iloc[-1]) > mid_high
    else:
        mid_low = float(recent["low"].iloc[:5].min())
        return float(recent["close"].iloc[-1]) < mid_low


def _discount_premium(df: pd.DataFrame, direction: int, lookback: int = 40) -> bool:
    """Price is in discount zone (long) or premium zone (short) of last H4 swing."""
    recent = df.iloc[-lookback:-1]
    swing_h = float(recent["high"].max())
    swing_l = float(recent["low"].min())
    rng     = swing_h - swing_l
    if rng == 0:
        return False
    cv   = float(df["close"].iloc[-2])
    pct  = (cv - swing_l) / rng
    if direction == 1:
        return pct < 0.45        # discount: below 45%
    else:
        return pct > 0.55        # premium: above 55%


def _volume_spike(df: pd.DataFrame, lookback: int = 20) -> bool:
    """Last bar volume is 1.5x the lookback average."""
    if "tick_volume" not in df.columns:
        return False
    recent = df.iloc[-(lookback + 1):-1]
    avg    = float(recent["tick_volume"].mean())
    last   = float(df["tick_volume"].iloc[-2])
    return avg > 0 and last > 1.5 * avg


# ── Session timing (ICT algorithmic delivery windows) ────────────────────────
# ICT: price is delivered algorithmically to hit liquidity pools. Sweeps are
# most significant at session open windows — the algorithm hunts stops before
# delivering in the true direction.
_SWEEP_WINDOWS_UTC = [
    (7,  8),    # London open — sweeps Asian range highs/lows
    (13, 14),   # NY open — sweeps London range highs/lows
    (20, 21),   # NY close MOC — final settlement sweep
]

def _cvd_absorbing(df: pd.DataFrame, direction: int, lookback: int = 5) -> bool:
    """Proxy CVD using tick_volume + bar direction. For S1: returns True when
    volume-weighted delta supports the reversal direction despite the sweep move.

    Long reversal: buyers were absorbing during the downswing (CVD positive = net buying).
    Short reversal: sellers were distributing during the upswing (CVD negative = net selling).
    """
    if "tick_volume" not in df.columns or len(df) < lookback + 2:
        return False
    recent = df.iloc[-(lookback + 1):-1]
    closes = recent["close"].values
    vols   = recent["tick_volume"].values
    delta  = 0.0
    for i in range(1, len(closes)):
        d = 1 if closes[i] > closes[i - 1] else (-1 if closes[i] < closes[i - 1] else 0)
        delta += d * float(vols[i])
    return delta > 0 if direction == 1 else delta < 0


def _session_sweep_window(ts: pd.Timestamp | None = None) -> bool:
    """Returns True if current UTC hour falls in a high-probability sweep window."""
    from datetime import datetime, timezone as tz
    if ts is not None:
        hour = ts.hour
    else:
        hour = datetime.now(tz=tz.utc).hour
    return any(s <= hour < e for s, e in _SWEEP_WINDOWS_UTC)


# ── Scenario classifiers ──────────────────────────────────────────────────────

def _s1_post_sweep_reversal(
    phase: PhaseResult, df_m15: pd.DataFrame, df_m5: Optional[pd.DataFrame],
    direction: int, atr: float
) -> tuple[bool, list[str], list[str]]:
    """S1: Liquidity swept → wick rejection → FVG or OB nearby."""
    in_window = _session_sweep_window()
    required = {
        "sweep_detected":     phase.sweep_dir != 0 and phase.sweep_dir != direction,
        "wick_rejection":     _wick_rejection(df_m15, direction),
        "fvg_or_ob_nearby":   _fvg_nearby(df_m15, direction, atr) or _ob_nearby(df_m15, direction, atr),
    }
    optional = {
        "session_sweep_window": in_window,               # London/NY open or MOC
        "cvd_absorbing":        _cvd_absorbing(df_m15, direction),  # volume delta confirms
    }
    confirmations = [k for k, v in {**required, **optional}.items() if v]
    missing       = [k for k, v in required.items() if not v]
    # Need all 3 required for S1; optionals raise confidence but don't block
    return len(missing) == 0, confirmations, missing


def _s2_ob_fvg_retracement(
    phase: PhaseResult, df_m15: pd.DataFrame, df_m5: Optional[pd.DataFrame],
    direction: int, atr: float
) -> tuple[bool, list[str], list[str]]:
    """S2: After HTF expansion, price retesting OB+FVG confluence zone."""
    fvg = _fvg_nearby(df_m15, direction, atr)
    ob  = _ob_nearby(df_m15, direction, atr)
    required = {
        "retracement_phase":  phase.phase in (Phase.RETRACEMENT, Phase.CONTINUATION),
        "ob_present":         ob,
        "fvg_present":        fvg,
        "discount_premium":   _discount_premium(df_m15, direction),
    }
    confirmations = [k for k, v in required.items() if v]
    missing       = [k for k, v in required.items() if not v]
    # Need retracement phase + (OB or FVG) + discount/premium = 3 of 4
    critical_met  = required["retracement_phase"] and (ob or fvg) and required["discount_premium"]
    return critical_met, confirmations, missing


def _s3_continuation_pullback(
    phase: PhaseResult, df_m15: pd.DataFrame, df_m5: Optional[pd.DataFrame],
    direction: int, atr: float
) -> tuple[bool, list[str], list[str]]:
    """S3: After BOS/expansion, pullback to displacement FVG — lower conviction."""
    required = {
        "expansion_or_continuation": phase.phase in (Phase.EXPANSION, Phase.CONTINUATION),
        "h4_bias_aligned":           phase.h4_bias == direction,
        "fvg_nearby":                _fvg_nearby(df_m15, direction, atr),
        "ltf_choch":                 _ltf_choch(df_m5, direction),
    }
    confirmations = [k for k, v in required.items() if v]
    missing       = [k for k, v in required.items() if not v]
    # Need all 4 — continuation requires tightest confirmation (was 11% WR without LTF CHoCH)
    return len(missing) == 0, confirmations, missing


def _s4_range_sweep_fade(
    phase: PhaseResult, df_m15: pd.DataFrame, df_m5: Optional[pd.DataFrame],
    direction: int, atr: float
) -> tuple[bool, list[str], list[str]]:
    """S4: Range boundary swept → fade back to range midpoint."""
    required = {
        "consolidation_phase": phase.phase == Phase.CONSOLIDATION,
        "sweep_detected":      phase.sweep_dir != 0 and phase.sweep_dir != direction,
        "wick_rejection":      _wick_rejection(df_m15, direction),
    }
    confirmations = [k for k, v in required.items() if v]
    missing       = [k for k, v in required.items() if not v]
    return len(missing) == 0, confirmations, missing


# ── Main engine ───────────────────────────────────────────────────────────────

class ScenarioEngine:
    """Evaluates which scenario is active and issues a GO/NO_GO verdict."""

    def evaluate(
        self,
        symbol:    str,
        direction: int,
        df_m15:    pd.DataFrame,
        df_m5:     Optional[pd.DataFrame],
        h4_bias:   int,
        atr:       float,
    ) -> ScenarioResult:
        # Called with pre-computed plan (no dataframe) — skip scenario analysis
        if df_m15 is None:
            result = ScenarioResult(
                symbol=symbol, direction=direction, scenario="NO_DATA",
                verdict="GO",  # defer to Council when no bars available
                phase="unknown", confirmations=[], missing=["df_m15_unavailable"],
                confidence=0.0, ts=datetime.now(tz=timezone.utc).isoformat(),
            )
            self._log(result, type("P", (), {"notes": ["called without dataframe"]})())
            return result

        phase = detect_phase(df_m15, h4_bias, atr)

        checks = [
            ("S1_post_sweep_reversal",    _s1_post_sweep_reversal),
            ("S2_ob_fvg_retracement",     _s2_ob_fvg_retracement),
            ("S3_continuation_pullback",  _s3_continuation_pullback),
            ("S4_range_sweep_fade",       _s4_range_sweep_fade),
        ]

        best_scenario  = "NO_SCENARIO"
        best_confirms  = []
        best_missing   = []
        best_go        = False
        best_confidence = 0.0

        for name, fn in checks:
            go, confirms, missing = fn(phase, df_m15, df_m5, direction, atr)
            if go and len(confirms) > len(best_confirms):
                best_scenario   = name
                best_confirms   = confirms
                best_missing    = missing
                best_go         = True
                best_confidence = phase.confidence

        if not best_go:
            # Find the closest match (most confirmations even if not all met)
            for name, fn in checks:
                _, confirms, missing = fn(phase, df_m15, df_m5, direction, atr)
                if len(confirms) > len(best_confirms):
                    best_scenario  = name
                    best_confirms  = confirms
                    best_missing   = missing
                    best_confidence = phase.confidence * (len(confirms) / max(1, len(confirms) + len(missing)))

        verdict_prefix = "" if not SHADOW_MODE else "SHADOW_"
        verdict = f"{verdict_prefix}GO" if best_go else f"{verdict_prefix}NO_GO"

        result = ScenarioResult(
            symbol=symbol,
            direction=direction,
            scenario=best_scenario,
            verdict=verdict,
            phase=phase.phase.value,
            confirmations=best_confirms,
            missing=best_missing,
            confidence=round(best_confidence, 3),
            ts=datetime.now(tz=timezone.utc).isoformat(),
        )

        self._log(result, phase)
        return result

    def _log(self, r: ScenarioResult, phase: PhaseResult) -> None:
        try:
            SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
            entry = asdict(r)
            entry["phase_notes"] = phase.notes
            with open(SHADOW_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("[ScenarioEngine] log failed: %s", exc)
        dir_str = "LONG" if r.direction == 1 else "SHORT"
        logger.info("[ScenarioEngine] %s %s | %s | %s | confirms=%s missing=%s",
                    r.symbol, dir_str, r.verdict, r.scenario,
                    r.confirmations, r.missing)
