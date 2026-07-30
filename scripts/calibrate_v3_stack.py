"""V3 ProbabilityStack calibration from historical processed data.

Replays M15 bars per symbol through the FVG+OB strategy, computes what each
StackInput confluence boolean would have been, runs compute_stack_score(), and
finds the score threshold that maximises (win_rate * avg_r) on the historical
sample.

Writes calibrated thresholds to config/v3_thresholds.json.
Run: python -m scripts.calibrate_v3_stack
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.probability_stack import StackInput, compute_stack_score, _WEIGHTS
from core.instrument_profile import PROFILES
from strategies.aiden_index import AiDENIndexStrategy

PROCESSED = ROOT / "data" / "processed"
OUT_PATH  = ROOT / "config" / "v3_thresholds.json"

SYMBOLS = [
    "XAUUSD", "XAGUSD", "GBPUSD",
    "US100.cash", "US30.cash", "US500.cash", "US2000.cash",
]

RR_TARGET   = 3.0
RISK_FRAC   = 0.01   # 1% per trade for P&L sim
SCORE_RANGE = range(30, 85, 5)


def _load(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    m15 = PROCESSED / f"{symbol}_M15.csv"
    if not m15.exists():
        print(f"  [SKIP] {symbol}: no M15 data")
        return None
    df = pd.read_csv(m15, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df, df   # m5 placeholder = same df for now


def _run_symbol(symbol: str, df: pd.DataFrame) -> list[dict]:
    profile  = PROFILES.get(symbol)
    arch     = profile.archetype if profile else "liquidity"
    kwargs   = (profile.strategy_kwargs if profile else {})

    strat = AiDENIndexStrategy(
        rr_target       = kwargs.get("rr_target", 3.0),
        atr_stop_buffer = kwargs.get("atr_stop_buffer", 0.3),
    )

    rows = []
    for i in range(200, len(df) - 1):
        window = df.iloc[max(0, i - 500): i + 1].copy()
        try:
            signals = strat.generate_signals(window)
            sig     = int(signals.iloc[-1])
        except Exception:
            continue

        if sig == 0:
            continue

        scores  = getattr(strat, "_scores", None)
        raw_score = int(scores.iloc[-1]) if scores is not None else 0
        if raw_score < 5:
            continue

        # Build StackInput from strategy internals
        h4_bias = getattr(strat, "_last_h4_bias", 0)
        inp = StackInput(
            symbol             = symbol,
            signal_dir         = sig,
            archetype          = arch,
            archetype_threshold= profile.entry_threshold if profile else 65,
            h4_aligned         = (h4_bias == sig),
            fvg_present        = True,
            ob_present         = raw_score >= 7,
            sweep_present      = False,   # no AMD data in historical; conservative
            m5_confirmed       = raw_score >= 8,
            at_htf_level       = raw_score >= 9,
            level_strength     = min(raw_score / 10.0, 1.0),
            order_flow_aligned = False,
            dom_aligned        = False,
            news_aligned       = False,
            in_ict_macro       = False,
            ipda_aligned       = False,
            smt_divergence     = False,
            eq_liq_cluster     = False,
            early_leakage      = False,
            scout_aligned      = False,
            regime             = "trending",
        )
        stack_score = compute_stack_score(inp)

        # Simulate outcome: next N bars (RR_TARGET * stop estimate)
        entry_bar  = df.iloc[i]
        atr_cache  = getattr(strat, "_atr_cache", None)
        if atr_cache is not None and len(atr_cache) > 0:
            atr_val = float(atr_cache.iloc[-1])
        else:
            atr_val = float(df["close"].iloc[i]) * 0.003

        sl_dist = atr_val * kwargs.get("atr_stop_buffer", 0.5)
        tp_dist = sl_dist * RR_TARGET
        entry_px = float(entry_bar["close"])
        sl_px    = entry_px - sig * sl_dist
        tp_px    = entry_px + sig * tp_dist

        outcome = 0.0
        for j in range(i + 1, min(i + 80, len(df))):
            fut = df.iloc[j]
            if sig == 1:
                if float(fut["low"]) <= sl_px:
                    outcome = -1.0; break
                if float(fut["high"]) >= tp_px:
                    outcome = RR_TARGET; break
            else:
                if float(fut["high"]) >= sl_px:
                    outcome = -1.0; break
                if float(fut["low"]) <= tp_px:
                    outcome = RR_TARGET; break

        rows.append({
            "symbol":      symbol,
            "arch":        arch,
            "v2_score":    raw_score,
            "stack_score": stack_score,
            "outcome_r":   outcome,
            "win":         outcome > 0,
        })

    return rows


def _find_threshold(rows: list[dict]) -> dict:
    if not rows:
        return {"threshold": 65, "n": 0}

    df = pd.DataFrame(rows)
    best = {"threshold": 65, "score": -999.0, "n": len(df), "wr": 0.0, "avg_r": 0.0}

    for thr in SCORE_RANGE:
        sub = df[df["stack_score"] >= thr]
        if len(sub) < 10:
            continue
        wr    = float(sub["win"].mean())
        avg_r = float(sub["outcome_r"].mean())
        metric = wr * avg_r
        if metric > best["score"]:
            best = {
                "threshold": int(thr),
                "score":     round(metric, 4),
                "n":         len(sub),
                "wr":        round(wr, 3),
                "avg_r":     round(avg_r, 3),
                "total":     len(df),
            }

    return best


def main() -> None:
    print("V3 ProbabilityStack calibration from historical data\n")
    results = {}
    all_rows: list[dict] = []

    for sym in SYMBOLS:
        print(f"[{sym}]")
        data = _load(sym)
        if data is None:
            continue
        df, _ = data
        rows = _run_symbol(sym, df)
        print(f"  signals={len(rows)}")
        if rows:
            thr = _find_threshold(rows)
            print(f"  threshold={thr['threshold']} wr={thr.get('wr', '?')} "
                  f"avg_r={thr.get('avg_r', '?')} n={thr.get('n', '?')}/{thr.get('total', '?')}")
            results[sym] = thr
            all_rows.extend(rows)

    # Global threshold across all symbols
    global_thr = _find_threshold(all_rows)
    print(f"\n[GLOBAL] threshold={global_thr['threshold']} "
          f"wr={global_thr.get('wr','?')} avg_r={global_thr.get('avg_r','?')} "
          f"n={global_thr.get('n','?')}/{global_thr.get('total','?')}")
    results["_global"] = global_thr

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
