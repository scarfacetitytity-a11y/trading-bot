"""Declarative per-instrument profiles — Phase 0 of the probability-stack rebuild.

Each symbol gets one InstrumentProfile carrying:
  - archetype / entry_threshold / confluence_weights — the FUTURE probability-stack
    knobs. Thresholds and weights are PLACEHOLDERS, calibrated in a later phase.
  - strategy_kwargs — the CURRENT live AiDENIndexStrategy constructor kwargs,
    derived from the same sources the orchestrator used inline before Phase 0
    (backtests.run_multi_instrument: INSTRUMENTS, OPTIMISED_PARAMS, TRAIL_CONFIGS,
    BIDIRECTIONAL, M15_PARAMS). The assembly logic below replicates the pre-Phase-0
    _build_aiden_strategy() byte-for-byte so behavior is unchanged. A later phase
    inlines these as literals and retires the backtest import.

Note: h4_bias_gate=False for XAUUSD/XAGUSD (sniper mode — H4 bias is confluence,
not a gate), matching master's _build_aiden_strategy wiring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from backtests.run_multi_instrument import (
    INSTRUMENTS, OPTIMISED_PARAMS, TRAIL_CONFIGS, BIDIRECTIONAL, M15_PARAMS,
)


@dataclass(frozen=True)
class InstrumentProfile:
    symbol: str
    archetype: str                       # "sniper" | "momentum" | "liquidity"
    entry_threshold: int                 # placeholder — calibrated later
    confluence_weights: dict[str, float] # placeholder — calibrated later
    stop_model: str
    stop_atr_mult: float
    tp_model: str
    base_rr: float
    max_hold_bars: int
    risk_mult: float
    intra_bar_entry: bool
    # CURRENT live per-symbol strategy parameters (exact AiDENIndexStrategy kwargs)
    strategy_kwargs: dict = field(default_factory=dict)


# ── Archetype assignment ─────────────────────────────────────────────────────
_ARCHETYPES: dict[str, str] = {
    "XAUUSD":      "sniper",
    "XAGUSD":      "sniper",
    "GBPUSD":      "liquidity",
    "EURUSD":      "liquidity",
    "USDJPY":      "liquidity",
    "US30.cash":   "momentum",
    "US100.cash":  "momentum",
    "US500.cash":  "momentum",
    "US2000.cash": "momentum",
    "UK100.cash":  "momentum",
    "JP225.cash":  "momentum",
}

# Placeholder entry thresholds per archetype (probability-stack scale 0-100).
_ENTRY_THRESHOLDS: dict[str, int] = {
    "sniper":    62,
    "liquidity": 65,
    "momentum":  68,
}

# Placeholder confluence weights per archetype — CALIBRATED IN A LATER PHASE.
# Keys mirror the blueprint's probability-stack confluence taxonomy.
_CONFLUENCE_WEIGHTS: dict[str, dict[str, float]] = {
    # Sniper (metals): sweep-then-reverse at a precise POI — liquidity + structure heavy
    "sniper": {
        "htf_bias": 2.0, "liquidity_sweep": 2.5, "fvg": 1.5, "ob": 1.5,
        "premium_discount": 1.0, "manipulation": 1.5, "session": 0.5, "momentum": 0.5,
    },
    # Momentum (indices): trend continuation in session — bias + momentum heavy
    "momentum": {
        "htf_bias": 2.5, "momentum": 2.0, "bos": 1.5, "fvg": 1.0,
        "session": 1.5, "liquidity_sweep": 1.0, "premium_discount": 0.5,
    },
    # Liquidity (FX): engineered sweeps around session opens — sweep + manipulation heavy
    "liquidity": {
        "liquidity_sweep": 2.5, "manipulation": 2.0, "htf_bias": 1.5, "fvg": 1.0,
        "premium_discount": 1.0, "session": 1.0, "momentum": 0.5,
    },
}


def build_strategy_kwargs(symbol: str) -> dict:
    """Exact AiDENIndexStrategy kwargs for a symbol.

    Replicates the pre-Phase-0 orchestrator._build_aiden_strategy() assembly
    verbatim (including the eager evaluation of tf_p.pop() defaults) so the
    resulting constructor arguments are identical for every symbol.
    """
    cfg   = INSTRUMENTS.get(symbol, {})
    tf_p  = M15_PARAMS.copy()
    v2    = dict(
        long_only=(symbol not in BIDIRECTIONAL),
        h4_bias_gate=(symbol not in {"XAUUSD", "XAGUSD"}),  # sniper mode: no H4 trend gate for metals
        rr_model3_bonus=0.5, rr_trend_bonus=0.5, rr_trend_threshold=0.003, rr_max=5.0,
        session_prime_start=13, session_prime_end=15,
        use_rsi=True, rsi_period=56,
        rsi_long_lo=25.0, rsi_long_hi=55.0, rsi_short_lo=45.0, rsi_short_hi=75.0,
        trail_to_be=symbol in TRAIL_CONFIGS,
        trail_be_r=TRAIL_CONFIGS[symbol].get("trail_be_r", 1.0) if symbol in TRAIL_CONFIGS else 1.0,
        trail_lock_r=TRAIL_CONFIGS[symbol].get("trail_lock_r", 2.0) if symbol in TRAIL_CONFIGS else 2.0,
        t1_r=TRAIL_CONFIGS[symbol].get("t1_r", 0.0) if symbol in TRAIL_CONFIGS else 0.0,
        t1_partial_pct=TRAIL_CONFIGS[symbol].get("t1_partial_pct", 0.5) if symbol in TRAIL_CONFIGS else 0.5,
        time_stop_bars=TRAIL_CONFIGS[symbol].get("time_stop_bars", 0) if symbol in TRAIL_CONFIGS else 0,
    )
    if symbol in OPTIMISED_PARAMS:
        opt  = OPTIMISED_PARAMS[symbol].copy()
        bias = opt.pop("h4_bias_method", tf_p.pop("h4_bias_method", "ema"))
        stop = opt.pop("atr_stop_buffer", tf_p.pop("atr_stop_buffer", 0.5))
        return dict(
            min_score=opt.get("min_score", 4),
            min_fvg_atr=opt.get("min_fvg_atr", 0.10),
            rr_target=opt.get("rr_target", 2.5),
            session_start=opt.get("session_start", cfg.get("session_start", 7)),
            session_end=opt.get("session_end", cfg.get("session_end", 21)),
            h4_bias_method=bias, atr_stop_buffer=stop,
            **{k: v for k, v in tf_p.items() if k not in ("h4_bias_method", "atr_stop_buffer")},
            **v2,
        )
    bias = tf_p.pop("h4_bias_method", "ema")
    stop = tf_p.pop("atr_stop_buffer", 0.5)
    return dict(
        min_score=4, min_fvg_atr=0.10, rr_target=2.5,
        session_start=cfg.get("session_start", 7),
        session_end=cfg.get("session_end", 21),
        h4_bias_method=bias, atr_stop_buffer=stop,
        **tf_p, **v2,
    )


def _make_profile(symbol: str) -> InstrumentProfile:
    kw        = build_strategy_kwargs(symbol)
    archetype = _ARCHETYPES[symbol]
    return InstrumentProfile(
        symbol=symbol,
        archetype=archetype,
        entry_threshold=_ENTRY_THRESHOLDS[archetype],
        confluence_weights=dict(_CONFLUENCE_WEIGHTS[archetype]),
        # Current live behavior: structural anchor (sweep low / FVG edge) + ATR buffer
        stop_model="structure_atr",
        stop_atr_mult=kw["atr_stop_buffer"],
        # Current live behavior: dynamic RR off a base target (Model3/trend bonuses)
        tp_model="rr_dynamic",
        base_rr=kw["rr_target"],
        max_hold_bars=kw["time_stop_bars"],   # 0 = disabled (current live setting)
        risk_mult=1.0,
        intra_bar_entry=False,                # bot enters on bar close today
        strategy_kwargs=kw,
    )


PROFILES: dict[str, InstrumentProfile] = {
    sym: _make_profile(sym) for sym in _ARCHETYPES
}
