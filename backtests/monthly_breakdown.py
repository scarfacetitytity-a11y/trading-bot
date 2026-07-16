"""Monthly per-symbol breakdown of the resolved pool — token-cheap, no recompute.

Reads logs/resolved_trades.csv (the intrabar-resolved, honest per-trade log),
applies the spread haircut, and aggregates net expectancy + trade count by
month x symbol. Shows the recent months so you can see which pairs are working
NOW vs over the full sample.

Usage:
    python -m backtests.monthly_breakdown
    python -m backtests.monthly_breakdown --months 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtests.ftmo_resolved_sim import SPREAD

LOGS = Path(__file__).resolve().parent.parent / "logs"


def load():
    df = pd.read_csv(LOGS / "resolved_trades.csv")
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["month"] = df["entry_time"].dt.strftime("%Y-%m")
    df["rd"] = (df["entry"] - df["stop"]).abs()
    df["net_R"] = df["R"] - df["symbol"].map(SPREAD) / df["rd"]
    return df


def main(months=6):
    df = load()
    all_months = sorted(df["month"].unique())
    recent = all_months[-months:]
    symbols = sorted(df["symbol"].unique())

    print(f"\n{'='*100}")
    print(f"  MONTHLY NET EXPECTANCY (R/trade, spread-adjusted) — last {months} months")
    print(f"  Full sample: {all_months[0]} -> {all_months[-1]}  ({len(df)} trades)")
    print(f"{'='*100}")

    # Header
    hdr = f"  {'Symbol':<14}" + "".join(f"{m[2:]:>9}" for m in recent) + f"{'FULL':>9}"
    print(hdr)
    print("  " + "-"*14 + "-"*9*(len(recent)+1))

    for sym in symbols:
        s = df[df["symbol"] == sym]
        row = f"  {sym:<14}"
        for m in recent:
            mm = s[s["month"] == m]
            if len(mm):
                row += f"{mm['net_R'].mean():>+9.2f}"
            else:
                row += f"{'-':>9}"
        row += f"{s['net_R'].mean():>+9.2f}"
        print(row)

    # Trade counts by month
    print("\n  TRADE COUNT by month:")
    hdr2 = f"  {'Symbol':<14}" + "".join(f"{m[2:]:>9}" for m in recent) + f"{'FULL':>9}"
    print(hdr2)
    print("  " + "-"*14 + "-"*9*(len(recent)+1))
    for sym in symbols:
        s = df[df["symbol"] == sym]
        row = f"  {sym:<14}"
        for m in recent:
            row += f"{len(s[s['month']==m]):>9}"
        row += f"{len(s):>9}"
        print(row)

    # Portfolio monthly summary
    print("\n  PORTFOLIO by month:")
    print(f"  {'Month':<10} {'trades':>7} {'net expR':>9} {'sum R':>9} {'hit-TP%':>8} {'winR%':>7}")
    print("  " + "-"*54)
    for m in recent:
        mm = df[df["month"] == m]
        if not len(mm):
            continue
        hit = 100*(mm["reason"] == "TP").mean()
        wr  = 100*(mm["net_R"] > 0).mean()
        print(f"  {m:<10} {len(mm):>7} {mm['net_R'].mean():>+9.3f} {mm['net_R'].sum():>+9.1f} "
              f"{hit:>7.1f}% {wr:>6.1f}%")
    tot = df[df["month"].isin(recent)]
    print("  " + "-"*54)
    print(f"  {'RECENT':<10} {len(tot):>7} {tot['net_R'].mean():>+9.3f} {tot['net_R'].sum():>+9.1f} "
          f"{100*(tot['reason']=='TP').mean():>7.1f}% {100*(tot['net_R']>0).mean():>6.1f}%")

    # Rank symbols by recent (last `months`) net expectancy
    print(f"\n  SYMBOL RANK — last {months} months (net expR, min 5 trades):")
    rec = df[df["month"].isin(recent)]
    rank = (rec.groupby("symbol")
               .agg(trades=("net_R", "size"), net_expR=("net_R", "mean"), sumR=("net_R", "sum"))
               .reset_index())
    rank = rank[rank["trades"] >= 5].sort_values("net_expR", ascending=False)
    for _, r in rank.iterrows():
        flag = "DROP" if r["net_expR"] < 0.05 else ("STAR" if r["net_expR"] > 0.4 else "")
        print(f"    {r['symbol']:<14} {int(r['trades']):>4} trades  "
              f"net {r['net_expR']:>+.3f}R  sum {r['sumR']:>+.1f}R  {flag}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=6)
    args = ap.parse_args()
    main(months=args.months)
