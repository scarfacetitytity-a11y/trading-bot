"""All-pairs monthly scanner — the whole universe through the resolved engine.

Runs every symbol (core 10 + expanded: palladium, platinum, copper, more indices,
oil) through the intrabar-resolved backtest, applies live spreads, and ranks by
net expectancy overall + recent months. New symbols run long-only with default
sessions (first-pass read; session tuning is a follow-up).

Usage (data pulled via pull_scan.py first):
    python -m backtests.all_pairs_scan
    python -m backtests.all_pairs_scan --months 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.resolved_backtest import run_symbol
from backtests.ftmo_resolved_sim import SPREAD as CORE_SPREAD

LOGS = Path(__file__).resolve().parent.parent / "logs"

# Expanded scan symbols (data pulled by pull_scan.py)
SCAN_NEW = [
    "XPDUSD", "XPTUSD", "XCUUSD",
    "FRA40.cash", "HK50.cash", "AUS200.cash", "EU50.cash", "SPN35.cash",
    "USOIL.cash", "UKOIL.cash",
]

CORE = ["XAUUSD", "XAGUSD", "GBPUSD", "US100.cash", "US30.cash", "US500.cash",
        "US2000.cash", "UK100.cash", "JP225.cash", "GER40.cash"]

# live spreads (price) — core from MT5 + new scan symbols
SCAN_SPREAD = {
    **CORE_SPREAD,
    "XPDUSD": 5.26, "XPTUSD": 3.55, "XCUUSD": 0.54, "FRA40.cash": 2.61,
    "HK50.cash": 6.04, "AUS200.cash": 1.17, "EU50.cash": 0.96, "SPN35.cash": 11.4,
    "USOIL.cash": 0.071, "UKOIL.cash": 0.064,
}

UNIVERSE = CORE + SCAN_NEW


def main(months=4):
    rows = []
    per_trade = []
    print(f"\n  Scanning {len(UNIVERSE)} symbols through the resolved engine...\n")
    for sym in UNIVERSE:
        try:
            trades = run_symbol(sym)
        except Exception as exc:
            print(f"  {sym:<14} ERROR: {exc}"); continue
        if not trades:
            print(f"  {sym:<14} no trades"); continue
        spread = SCAN_SPREAD.get(sym, 0.0)
        recs = []
        for t in trades:
            rd = abs(t.entry - t.stop)
            net = t.R - (spread / rd if rd > 0 else 0)
            recs.append({"symbol": sym, "entry_time": t.entry_time, "reason": t.reason,
                         "R": t.R, "net_R": net})
            per_trade.append(recs[-1])
        net = np.array([r["net_R"] for r in recs])
        hit = np.mean([1 if r["reason"] == "TP" else 0 for r in recs])
        rows.append({"symbol": sym, "trades": len(recs), "gross_expR": np.mean([r["R"] for r in recs]),
                     "net_expR": float(net.mean()), "sum_net_R": float(net.sum()),
                     "hit_tp": float(hit), "spread": spread})
        print(f"  {sym:<14} {len(recs):>4} trades  gross {np.mean([r['R'] for r in recs]):+.3f}R  "
              f"net {net.mean():+.3f}R  hit {100*hit:.0f}%")

    df = pd.DataFrame(rows).sort_values("net_expR", ascending=False)
    pt = pd.DataFrame(per_trade)
    pt["entry_time"] = pd.to_datetime(pt["entry_time"], utc=True)
    pt["month"] = pt["entry_time"].dt.strftime("%Y-%m")
    recent = sorted(pt["month"].unique())[-months:]

    print(f"\n{'='*94}")
    print(f"  ALL-PAIRS RANK — net expectancy (spread-adjusted), full sample")
    print(f"{'='*94}")
    print(f"  {'Symbol':<14} {'Trades':>7} {'grossR':>8} {'netR':>8} {'sumNet':>8} {'hitTP%':>7} {'verdict':>10}")
    print("  " + "-"*72)
    for _, r in df.iterrows():
        v = "STAR" if r.net_expR > 0.4 else ("KEEP" if r.net_expR > 0.15 else ("MARGINAL" if r.net_expR > 0 else "DROP"))
        print(f"  {r.symbol:<14} {int(r.trades):>7} {r.gross_expR:>+8.3f} {r.net_expR:>+8.3f} "
              f"{r.sum_net_R:>+8.1f} {100*r.hit_tp:>6.0f}% {v:>10}")

    # recent-months rank
    print(f"\n  RECENT {months}-MONTH net expectancy rank (min 5 trades):")
    rec = pt[pt["month"].isin(recent)]
    g = (rec.groupby("symbol").agg(trades=("net_R","size"), net=("net_R","mean"), s=("net_R","sum"))
            .reset_index())
    g = g[g["trades"] >= 5].sort_values("net", ascending=False)
    for _, r in g.iterrows():
        v = "STAR" if r.net > 0.4 else ("KEEP" if r.net > 0.15 else ("MARGINAL" if r.net > 0 else "DROP"))
        print(f"    {r.symbol:<14} {int(r.trades):>4}t  net {r.net:>+.3f}R  sum {r.s:>+.1f}R  {v}")

    df.to_csv(LOGS / "all_pairs_scan.csv", index=False)
    print(f"\n  Data: {LOGS / 'all_pairs_scan.csv'}")
    print(f"{'='*94}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=4)
    args = ap.parse_args()
    main(months=args.months)
