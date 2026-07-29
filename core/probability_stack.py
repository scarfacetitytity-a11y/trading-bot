"""Phase 1: ProbabilityStack — shadow 0-100 scorer + JSONL logger.

Runs shadow-only alongside the live gate cascade. Every entry decision
(fired or blocked at any gate) is written to logs/shadow_stack.jsonl.
No live behavior is changed. Thresholds calibrated from these logs before
Phase 4 cutover replaces the gate cascade with the stack score.

Score formula: weighted boolean sum normalized to 0-100.
Weights per archetype mirror InstrumentProfile.confluence_weights but
scaled to the 0-100 range so calibration thresholds are intuitive.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


# ── Confluence weights per archetype (0-100 normalized later) ────────────────
# These are Phase 1 priors — calibrate from shadow logs before Phase 4.
_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Sniper (metals): sweep-to-POI entries, precision over volume
    "sniper": {
        "h4_aligned":         5.0,
        "fvg_present":        8.0,
        "ob_present":         8.0,
        "sweep_present":     12.0,
        "m5_confirmed":       6.0,
        "at_htf_level":       6.0,
        "level_strength":     5.0,
        "order_flow_aligned": 5.0,
        "dom_aligned":        4.0,
        "news_aligned":       3.0,
        "in_ict_macro":       5.0,
        "ipda_aligned":       5.0,
        "smt_divergence":     8.0,
        "eq_liq_cluster":     6.0,
        "early_leakage":      4.0,
        "scout_aligned":      5.0,
        "regime_ok":          5.0,
    },
    # Momentum (indices): trend continuation in session
    "momentum": {
        "h4_aligned":        12.0,
        "fvg_present":        6.0,
        "ob_present":         6.0,
        "sweep_present":      5.0,
        "m5_confirmed":       8.0,
        "at_htf_level":       5.0,
        "level_strength":     4.0,
        "order_flow_aligned": 8.0,
        "dom_aligned":        5.0,
        "news_aligned":       4.0,
        "in_ict_macro":       6.0,
        "ipda_aligned":       5.0,
        "smt_divergence":     4.0,
        "eq_liq_cluster":     4.0,
        "early_leakage":      3.0,
        "scout_aligned":      6.0,
        "regime_ok":          9.0,
    },
    # Liquidity (FX): engineered sweeps around session opens
    "liquidity": {
        "h4_aligned":         8.0,
        "fvg_present":        7.0,
        "ob_present":         7.0,
        "sweep_present":     12.0,
        "m5_confirmed":       6.0,
        "at_htf_level":       7.0,
        "level_strength":     5.0,
        "order_flow_aligned": 6.0,
        "dom_aligned":        4.0,
        "news_aligned":       4.0,
        "in_ict_macro":       5.0,
        "ipda_aligned":       6.0,
        "smt_divergence":     6.0,
        "eq_liq_cluster":     8.0,
        "early_leakage":      5.0,
        "scout_aligned":      5.0,
        "regime_ok":          5.0,
    },
}

_MAX_W: Dict[str, float] = {arch: sum(w.values()) for arch, w in _WEIGHTS.items()}


@dataclass
class StackInput:
    """Confluence snapshot built incrementally through the entry decision path.

    Create at the start of `if desired != 0:`, populate as gates run,
    call shadow_logger.record() before every continue and when trade fires.
    """
    symbol: str
    signal_dir: int
    archetype: str
    archetype_threshold: int
    bar_time: Optional[str] = None

    # Gate results — keyed by gate name, value is a small dict with
    # at minimum {"blocked": bool} plus any numeric values useful for analysis.
    gates: Dict[str, dict] = field(default_factory=dict)
    blocking_gate: Optional[str] = None

    # Score accounting
    base_score: int = 0
    m5_bonus: int = 0
    order_flow_mod: int = 0
    news_mod: int = 0
    key_level_mod: int = 0
    dxy_mod: int = 0
    final_score: int = 0
    score_floor: int = 0
    floor_reason: str = ""

    # Confluence booleans
    h4_aligned: bool = False
    fvg_present: bool = True      # strategy only fires on FVG detection
    ob_present: bool = False
    sweep_present: bool = False
    m5_confirmed: bool = False
    at_htf_level: bool = False
    level_strength: float = 0.0
    order_flow_aligned: bool = False
    dom_aligned: bool = False
    news_aligned: bool = False
    in_ict_macro: bool = False
    ipda_aligned: bool = False
    smt_divergence: bool = False
    eq_liq_cluster: bool = False
    early_leakage: bool = False
    scout_aligned: bool = False
    regime: str = "unknown"
    prob_model_prob: float = 0.0

    fired: bool = False


def compute_stack_score(inp: StackInput) -> float:
    """0-100 weighted confluence score for a StackInput snapshot."""
    w   = _WEIGHTS.get(inp.archetype, _WEIGHTS["liquidity"])
    max_w = _MAX_W.get(inp.archetype, 1.0)

    raw = 0.0
    raw += w.get("h4_aligned", 0)         * float(inp.h4_aligned)
    raw += w.get("fvg_present", 0)         * float(inp.fvg_present)
    raw += w.get("ob_present", 0)          * float(inp.ob_present)
    raw += w.get("sweep_present", 0)       * float(inp.sweep_present)
    raw += w.get("m5_confirmed", 0)        * float(inp.m5_confirmed)
    raw += w.get("at_htf_level", 0)        * float(inp.at_htf_level)
    raw += w.get("level_strength", 0)      * min(inp.level_strength, 1.0)
    raw += w.get("order_flow_aligned", 0)  * float(inp.order_flow_aligned)
    raw += w.get("dom_aligned", 0)         * float(inp.dom_aligned)
    raw += w.get("news_aligned", 0)        * float(inp.news_aligned)
    raw += w.get("in_ict_macro", 0)        * float(inp.in_ict_macro)
    raw += w.get("ipda_aligned", 0)        * float(inp.ipda_aligned)
    raw += w.get("smt_divergence", 0)      * float(inp.smt_divergence)
    raw += w.get("eq_liq_cluster", 0)      * float(inp.eq_liq_cluster)
    raw += w.get("early_leakage", 0)       * float(inp.early_leakage)
    raw += w.get("scout_aligned", 0)       * float(inp.scout_aligned)
    raw += w.get("regime_ok", 0)           * float(inp.regime not in ("flatline", "oscillation"))

    return round(raw / max_w * 100, 1)


class ShadowLogger:
    """Thread-safe JSONL writer for shadow stack entries.

    One file, one writer, used by all TradingEngine threads simultaneously.
    Failures are silenced — shadow logging must never affect live execution.
    """

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._lock = threading.Lock()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def record(self, inp: StackInput) -> float:
        """Compute score, write JSONL row, return the score (0-100)."""
        try:
            score = compute_stack_score(inp)
            row = {
                "ts":                  datetime.now(timezone.utc).isoformat(),
                "symbol":              inp.symbol,
                "bar":                 inp.bar_time,
                "signal_dir":          inp.signal_dir,
                "archetype":           inp.archetype,
                "archetype_threshold": inp.archetype_threshold,
                "base_score":          inp.base_score,
                "final_score":         inp.final_score,
                "score_floor":         inp.score_floor,
                "floor_reason":        inp.floor_reason,
                "score_adj": {
                    "m5":         inp.m5_bonus,
                    "order_flow": inp.order_flow_mod,
                    "news":       inp.news_mod,
                    "key_level":  inp.key_level_mod,
                    "dxy":        inp.dxy_mod,
                },
                "confluences": {
                    "h4_aligned":         inp.h4_aligned,
                    "fvg_present":        inp.fvg_present,
                    "ob_present":         inp.ob_present,
                    "sweep_present":      inp.sweep_present,
                    "m5_confirmed":       inp.m5_confirmed,
                    "at_htf_level":       inp.at_htf_level,
                    "level_strength":     inp.level_strength,
                    "order_flow_aligned": inp.order_flow_aligned,
                    "dom_aligned":        inp.dom_aligned,
                    "news_aligned":       inp.news_aligned,
                    "in_ict_macro":       inp.in_ict_macro,
                    "ipda_aligned":       inp.ipda_aligned,
                    "smt_divergence":     inp.smt_divergence,
                    "eq_liq_cluster":     inp.eq_liq_cluster,
                    "early_leakage":      inp.early_leakage,
                    "scout_aligned":      inp.scout_aligned,
                    "regime":             inp.regime,
                    "prob_model_prob":    inp.prob_model_prob,
                },
                "gates":            inp.gates,
                "blocking_gate":    inp.blocking_gate,
                "stack_score":      score,
                "would_fire_at":    inp.archetype_threshold,
                "would_have_fired": score >= inp.archetype_threshold,
                "fired":            inp.fired,
            }
            line = json.dumps(row) + "\n"
            with self._lock:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            return score
        except Exception:
            return 0.0


# Module-level singleton — imported by orchestrator, no per-engine init needed.
shadow_logger = ShadowLogger(
    Path(__file__).parent.parent / "logs" / "shadow_stack.jsonl"
)
