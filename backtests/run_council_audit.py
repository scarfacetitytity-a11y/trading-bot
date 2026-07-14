"""Council of 12 Strategy Audit — AiDEN v2 M15 portfolio.

Runs three analyses across the full 2+ year trade log:
  1. Score distribution and per-score performance (WR, avg R, PF)
  2. Variable lot sizing simulation: score >= 6 → 1.5x, 5 → 1.0x, 4 → 0.75x
  3. Session quality breakdown (prime 13-15 UTC vs non-prime)

Usage:
    python -m backtests.run_council_audit
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.run_multi_instrument import (
    INSTRUMENTS, _generate_returns, _apply_circuit_breaker,
    _quick_metrics, _estimate_bpy,
)
from backtests.metrics import calculate_metrics

SCORE      = 4
FVG_ATR    = 0.10
RR         = 2.5
TF         = "M15"
COMMISSION = 0.0001
DAILY_HALT = 0.025
ACCOUNTS   = [10_000, 25_000, 100_000, 200_000]

# Variable sizing tiers
def _size_mult(score: int) -> float:
    if score >= 6:
        return 1.5
    if score == 5:
        return 1.0
    return 0.75   # score == 4


def _pnl_to_r(pnl_pct: float, entry: float, sl: float) -> float:
    """Convert pnl_pct to R multiple. Approximation: 1R = SL distance / entry."""
    return float("nan")  # without SL in trade log, use raw pnl rank instead


def _metrics_from_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "avg_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    pnl = trades["pnl_pct"] if "pnl_pct" in trades.columns else trades.get("pnl", pd.Series(dtype=float))
    winners = pnl[pnl > 0]
    losers  = pnl[pnl < 0]
    wr = len(winners) / len(pnl) * 100
    pf = (winners.sum() / abs(losers.sum())) if len(losers) > 0 and losers.sum() != 0 else float("inf")
    return {
        "n":        len(pnl),
        "wr":       wr,
        "pf":       pf,
        "avg_pnl":  pnl.mean() * 100,
        "avg_win":  winners.mean() * 100 if len(winners) else 0.0,
        "avg_loss": losers.mean() * 100  if len(losers)  else 0.0,
    }


def main():
    # ── Collect all trades across all instruments ─────────────────────────────
    all_trades_list = []
    per_inst_returns = {}

    print("\nRunning full backtest to collect trade log...")
    for symbol, cfg in INSTRUMENTS.items():
        r = _generate_returns(symbol, cfg, SCORE, FVG_ATR, RR, TF, COMMISSION)
        if r is None:
            continue
        _times, returns, trades = r
        per_inst_returns[symbol] = returns
        if not trades.empty:
            all_trades_list.append(trades)

    if not all_trades_list:
        print("No trade data found.")
        return

    all_trades = pd.concat(all_trades_list, ignore_index=True)
    all_trades = all_trades[all_trades["pnl_pct"] != 0].copy()

    print(f"  Total trades collected: {len(all_trades)}")
    if "score" in all_trades.columns:
        print(f"  Trades with score data: {(all_trades['score'] > 0).sum()}")

    # ── SECTION 1: Score distribution and per-score performance ───────────────
    print("\n" + "=" * 90)
    print("  SECTION 1: SCORE DISTRIBUTION & PER-SCORE PERFORMANCE")
    print("=" * 90)

    if "score" in all_trades.columns and (all_trades["score"] > 0).any():
        scored = all_trades[all_trades["score"] > 0].copy()
        score_counts = scored["score"].value_counts().sort_index()

        print(f"\n  {'Score':<8} {'Count':>7} {'% of total':>11} {'WR%':>7} {'PF':>7} "
              f"{'Avg Win%':>10} {'Avg Loss%':>11}")
        print("-" * 90)

        for s in sorted(scored["score"].unique()):
            bucket = scored[scored["score"] == s]
            m = _metrics_from_trades(bucket)
            pct = len(bucket) / len(scored) * 100
            print(f"  {s:<8} {m['n']:>7} {pct:>10.1f}% {m['wr']:>6.1f}% {m['pf']:>7.3f} "
                  f"{m['avg_win']:>9.4f}% {m['avg_loss']:>10.4f}%")

        print("-" * 90)
        all_m = _metrics_from_trades(scored)
        print(f"  {'ALL':<8} {all_m['n']:>7} {'100.0':>10}% {all_m['wr']:>6.1f}% {all_m['pf']:>7.3f} "
              f"{all_m['avg_win']:>9.4f}% {all_m['avg_loss']:>10.4f}%")

        print("\n  Score key:  4=min pass  5=+1 gate  6=+2 gates  7=+3 gates  8+=all gates")
    else:
        print("\n  Score column not available — re-run after aiden_index.py update.")

    # ── SECTION 2: Variable lot sizing simulation ─────────────────────────────
    print("\n" + "=" * 90)
    print("  SECTION 2: VARIABLE LOT SIZING  (score>=6: 1.5x | score=5: 1.0x | score=4: 0.75x)")
    print("=" * 90)

    if "score" in all_trades.columns and (all_trades["score"] > 0).any():
        scored = all_trades[all_trades["score"] > 0].copy()
        scored["size_mult"] = scored["score"].apply(_size_mult)
        scored["pnl_flat"]  = scored["pnl_pct"]
        scored["pnl_var"]   = scored["pnl_pct"] * scored["size_mult"]

        # Flat sizing stats
        flat_w = scored[scored["pnl_flat"] > 0]["pnl_flat"]
        flat_l = scored[scored["pnl_flat"] < 0]["pnl_flat"]
        flat_pf = flat_w.sum() / abs(flat_l.sum()) if flat_l.sum() != 0 else float("inf")
        flat_ret = scored["pnl_flat"].sum() * 100

        # Variable sizing stats
        var_w = scored[scored["pnl_var"] > 0]["pnl_var"]
        var_l = scored[scored["pnl_var"] < 0]["pnl_var"]
        var_pf = var_w.sum() / abs(var_l.sum()) if var_l.sum() != 0 else float("inf")
        var_ret = scored["pnl_var"].sum() * 100

        print(f"\n  {'Metric':<22} {'Flat 1%':>12} {'Variable':>12} {'Delta':>10}")
        print("-" * 60)
        print(f"  {'Total return (sum)':<22} {flat_ret:>+11.3f}% {var_ret:>+11.3f}% "
              f"{var_ret-flat_ret:>+9.3f}%")
        print(f"  {'Profit factor':<22} {flat_pf:>12.3f} {var_pf:>12.3f} "
              f"{var_pf-flat_pf:>+9.3f}")
        print(f"  {'WR%':<22} {scored['pnl_flat'].gt(0).mean()*100:>11.1f}% "
              f"{scored['pnl_flat'].gt(0).mean()*100:>11.1f}% {'same':>10}")

        # Distribution of size multipliers
        print("\n  Size distribution applied:")
        for mult in [0.75, 1.0, 1.5]:
            n = (scored["size_mult"] == mult).sum()
            pct = n / len(scored) * 100
            label = "score=4 (0.75x)" if mult == 0.75 else (
                "score=5 (1.0x)" if mult == 1.0 else "score>=6 (1.5x)")
            print(f"    {label:<22} {n:>6} trades  ({pct:.1f}%)")

        # Variable sizing MaxDD estimate (per-instrument daily sum)
        print("\n  Note: MaxDD impact requires per-instrument equity simulation.")
        print("  High-conviction sizing stacks risk if concurrent score>=6 signals fire.")
        print("  Council gate: check correlation exposure before implementing in live.")

    else:
        print("\n  Score data not available.")

    # ── SECTION 3: Session breakdown ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  SECTION 3: SESSION QUALITY BREAKDOWN")
    print("=" * 90)

    if "entry_time" in all_trades.columns:
        all_trades["entry_hour"] = pd.to_datetime(all_trades["entry_time"]).dt.hour
        all_trades["session"] = all_trades["entry_hour"].apply(
            lambda h: "prime(13-15)" if 13 <= h < 15 else
                      "ny(12-21)"    if 12 <= h < 21 else
                      "london(7-12)" if  7 <= h < 12 else
                      "asia(0-9)"    if  h <  9      else "other"
        )
        sessions = all_trades.groupby("session")
        print(f"\n  {'Session':<15} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Avg pnl%':>10}")
        print("-" * 55)
        for sess in ["prime(13-15)", "ny(12-21)", "london(7-12)", "asia(0-9)", "other"]:
            if sess not in sessions.groups:
                continue
            g = sessions.get_group(sess)
            m = _metrics_from_trades(g)
            print(f"  {sess:<15} {m['n']:>7} {m['wr']:>6.1f}% {m['pf']:>7.3f} {m['avg_pnl']:>9.4f}%")

    # ── SECTION 4: Per-instrument summary ────────────────────────────────────
    print("\n" + "=" * 90)
    print("  SECTION 4: PER-INSTRUMENT EDGE SUMMARY")
    print("=" * 90)
    print(f"\n  {'Symbol':<16} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Avg pnl%':>10} {'Score dist'}")
    print("-" * 90)
    for symbol in INSTRUMENTS:
        inst = all_trades[all_trades["symbol"] == symbol] if "symbol" in all_trades.columns else pd.DataFrame()
        if inst.empty:
            continue
        m = _metrics_from_trades(inst)
        if "score" in inst.columns:
            sc = inst[inst["score"] > 0]["score"]
            sd = " | ".join(f"s{s}:{(sc==s).sum()}" for s in sorted(sc.unique()))
        else:
            sd = "n/a"
        print(f"  {symbol:<16} {m['n']:>7} {m['wr']:>6.1f}% {m['pf']:>7.3f} {m['avg_pnl']:>9.4f}%  {sd}")

    # ── COUNCIL OF 12 VERDICT ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  COUNCIL OF 12 VERDICT ON VARIABLE LOT SIZING")
    print("=" * 90)
    print("""
  01 Principal      : Proceed with caution. Data must show score>=6 PF > score=4 PF before sizing up.
  02 Architect      : Variable sizing is one param change in _generate_returns. Low blast radius.
  03 Data Engineer  : size_mult must be stored per trade in live state — not reconstructed post-hoc.
  05 Compliance     : FTMO: 1.5x lots on 1% base = 1.5% risk. Still inside daily 5% limit with 3 concurrent.
                      If 4 concurrent score>=6 fire: 6% daily exposure — BREACH. Hard cap: max 3 concurrent.
  06 App Engineer   : _size_mult(score) is clean. Needs: (a) score passed to orchestrator, (b) lots calc updated.
  07 SRE            : No live risk until score is verified matching backtest score at same bar in live feed.
  08 Performance    : No latency impact — sizing calc is pre-order, single multiply.
  09 Test Engineer  : Need: score=4 vs score>=5 holdout test on last 6 months (unseen data).
  10 Staff          : One TRAIL_CONFIGS pattern already in place. SIZE_CONFIGS dict follows same shape.
  11 Reality Gap    : Score in backtest is computed on closed bars. Live score computed on forming bar.
                      This creates a 1-bar lag in live. Score at entry may differ by +/-1 from backtest.
  12 Devil's Adv    : The 0.75x floor on score=4 trades REDUCES the primary edge driver.
                      Score=4 is where MOST trades are. If score>=6 sample is small (< 200 trades),
                      the 1.5x multiplier is statistically underpowered. Verify n before applying.
    """)
    print("=" * 90)
    print("  Gate: run this audit, check score>=6 n-count and PF vs score=4.")
    print("  If score>=6 PF > 1.6 and n >= 150: SIZE_CONFIGS approved for backtest.")
    print("  If score>=6 PF <= score=4 PF: flat sizing stays, variable sizing rejected.")
    print("=" * 90)


if __name__ == "__main__":
    main()
