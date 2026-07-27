"""Z-score regime classifier — ported from ruflo/plugins/ruflo-neural-trader.

Classifies each symbol's recent price action into one of 6 statistical
regimes using a rolling baseline Z-score. Used as a pre-entry filter:
flatline and oscillation regimes gate or redirect entries before any
structural analysis runs.

Regimes:
  spike          — maxZ > 5      : extreme momentum / breakout event
  cluster-outlier — highCount > 50% : sustained dislocation
  drift          — highCount > 30% AND |lastZ| > 1.5 : directional trend
  oscillation    — signFlips > 20% : mean-reversion range
  pattern-break  — highCount > 30% AND signFlips > 1  : regime transition
  flatline       — maxZ < 0.5    : compression / no edge
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Baseline / tail split (ruflo: 80/20)
_BASELINE_FRAC = 0.80

# Classification thresholds (ruflo signal-generation.bench.mjs)
_SPIKE_Z          = 5.0
_CLUSTER_HIGH_PCT = 0.50
_DRIFT_HIGH_PCT   = 0.30
_DRIFT_LAST_Z     = 1.5
_OSCIL_FLIP_PCT   = 0.20
_BREAK_HIGH_PCT   = 0.30
_FLATLINE_Z       = 0.50


@dataclass
class RegimeResult:
    regime: str           # spike | drift | flatline | oscillation | cluster-outlier | pattern-break
    max_z: float
    last_z: float
    high_count_pct: float
    sign_flip_pct: float
    n_bars: int


def classify_regime(closes: np.ndarray) -> Optional[RegimeResult]:
    """Classify the statistical regime of a close-price series.

    Args:
        closes: 1-D float array of close prices, oldest first.
                Minimum 20 bars required.

    Returns:
        RegimeResult or None if insufficient data.
    """
    if len(closes) < 20:
        return None

    split = max(10, int(len(closes) * _BASELINE_FRAC))
    baseline = closes[:split]
    tail     = closes[split:]

    mean = float(np.mean(baseline))
    std  = float(np.std(baseline, ddof=1))
    if std < 1e-10:
        return RegimeResult("flatline", 0.0, 0.0, 0.0, 0.0, len(tail))

    z_series = (tail - mean) / std
    abs_z    = np.abs(z_series)
    n        = len(z_series)

    max_z      = float(abs_z.max())
    last_z     = float(z_series[-1])
    high_count = int((abs_z > 2.0).sum())
    high_pct   = high_count / n

    # sign flips in Z-series
    signs      = np.sign(z_series)
    flips      = int(np.sum(signs[1:] != signs[:-1]))
    flip_pct   = flips / max(n - 1, 1)

    if max_z > _SPIKE_Z:
        regime = "spike"
    elif high_pct > _CLUSTER_HIGH_PCT:
        regime = "cluster-outlier"
    elif high_pct > _DRIFT_HIGH_PCT and abs(last_z) > _DRIFT_LAST_Z:
        regime = "drift"
    elif flip_pct > _OSCIL_FLIP_PCT:
        regime = "oscillation"
    elif high_pct > _BREAK_HIGH_PCT and flips > 1:
        regime = "pattern-break"
    else:
        regime = "flatline"

    return RegimeResult(
        regime=regime,
        max_z=round(max_z, 3),
        last_z=round(last_z, 3),
        high_count_pct=round(high_pct, 3),
        sign_flip_pct=round(flip_pct, 3),
        n_bars=n,
    )


# Entry gate decisions derived from regime
_REGIME_BLOCK = {"flatline"}            # no edge — skip entirely
_REGIME_DOWNSIZE = {"spike"}            # extreme event — half size

# regimes that permit continuation trades (non-ranging)
_REGIME_TRENDING = {"drift", "pattern-break"}

# regimes that only suit mean-reversion (sweep-reversal / breakout at level)
_REGIME_RANGING = {"oscillation", "cluster-outlier"}


def entry_gate(result: Optional[RegimeResult], desired: int, trade_type: str) -> tuple[bool, str]:
    """Return (allow_entry, reason) based on regime classification.

    Args:
        result:     RegimeResult from classify_regime(), or None (pass-through).
        desired:    +1 long, -1 short.
        trade_type: "continuation" | "sweep_reversal" | "breakout" | "range"

    Returns:
        (True, "") to allow, (False, reason) to block.
    """
    if result is None:
        return True, ""

    r = result.regime

    if r in _REGIME_BLOCK:
        return False, f"regime=flatline (maxZ={result.max_z:.2f}) — no statistical edge"

    if r in _REGIME_RANGING and trade_type == "continuation":
        return False, (
            f"regime={r} — ranging context, continuation blocked "
            f"(signFlips={result.sign_flip_pct:.0%})"
        )

    return True, ""


def size_mult(result: Optional[RegimeResult]) -> float:
    """Return a size multiplier adjustment for the regime.

    spike → 0.5x (extreme event, unpredictable)
    all others → 1.0x (no adjustment)
    """
    if result is None:
        return 1.0
    return 0.5 if result.regime == "spike" else 1.0
