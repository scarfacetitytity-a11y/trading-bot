"""
AiDEN autoresearch — fixed backtest harness.

This file is READ-ONLY for the autoresearch agent.
Modify research/params.py to change what is tested.

Wraps the existing adaptive FTMO simulation engine with:
- Standardized output format (key: value lines for grep)
- Reads all parameters from research/params.py
- MC trials fixed at 2000 for speed (~2-3 min runs)
- Seed fixed for reproducibility

Output format (grep-friendly):
    challenge_pass_rate: 0.420
    blow_rate: 0.010
    daily_dd_breach_rate: 0.000
    timeout_rate: 0.280
    max_dd_avg: 4.2
    median_days_to_pass: 18.0
    total_windows: 1980

Usage:
    python -m research.backtest_harness
    python -m research.backtest_harness > run.log 2>&1
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

# Add bot root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _load_params():
    spec = importlib.util.spec_from_file_location(
        "params", Path(__file__).parent / "params.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = _load_params()

    print(f"[harness] Loading params:")
    print(f"  BASE_RISK={p.BASE_RISK}  SCORE_FLOOR={p.SCORE_FLOOR}  MIN_GRADE={p.MIN_GRADE}")
    print(f"  MIN_RR={p.MIN_RR}  COUSIN_BLOCK_HOURS={p.COUSIN_BLOCK_HOURS}")
    print(f"  CAP_CONCURRENT={p.CAP_CONCURRENT}  EDGE_FILTER={p.EDGE_FILTER}")
    print(f"  SL_BUFFER_ATR={p.SL_BUFFER_ATR}  DAILY_DD_LIMIT={p.DAILY_DD_LIMIT}")

    from backtests.run_ftmo_sim_adaptive import _load_trades_with_scores, _run_mc

    # Score floor maps to integer score level (4=C, 5=B, 6=A roughly)
    # SCORE_FLOOR 65 → use score=4 (take grade C+)
    # SCORE_FLOOR 75 → use score=5 (take grade B+)
    # SCORE_FLOOR 85 → use score=6 (take grade A only)
    if p.SCORE_FLOOR >= 85:
        score_level = 6
    elif p.SCORE_FLOOR >= 75:
        score_level = 5
    else:
        score_level = 4

    print(f"[harness] score_level={score_level}  mc_trials=2000  seed=7")
    print("[harness] Building trade log...")

    trades, avg_loss_abs = _load_trades_with_scores(score_level)
    print(f"[harness] {len(trades)} trades loaded from backtest data.")

    rng = random.Random(7)
    result = _run_mc(
        trades,
        avg_loss_abs,
        base_risk=p.BASE_RISK,
        cap=p.CAP_CONCURRENT * p.BASE_RISK,  # cap = max concurrent * base risk
        mc=2000,
        adaptive=True,
        rng=rng,
    )

    # Standardized output — these lines are grepped by the autoresearch loop
    print("---")
    print(f"challenge_pass_rate: {result['pass_pct'] / 100:.6f}")
    print(f"blow_rate: {result['blow_pct'] / 100:.6f}")
    print(f"daily_dd_breach_rate: {result['daily_pct'] / 100:.6f}")
    print(f"timeout_rate: {result['timeout_pct'] / 100:.6f}")
    print(f"median_days_to_pass: {result['med_days']:.1f}")
    print(f"total_windows: {result['n']}")
    # Approximate max_dd_avg from effective risk mean
    print(f"eff_risk_mean_pct: {result['eff_mean']:.3f}")
    print(f"eff_risk_p95_pct: {result['eff_p95']:.3f}")


if __name__ == "__main__":
    main()
