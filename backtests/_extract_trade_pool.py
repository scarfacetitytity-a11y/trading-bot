"""Extract the combined portfolio trade stream for FTMO Monte Carlo.

Runs the TM simulation at risk_pct=1.0 for every instrument, collects every
closed trade with its exit timestamp and pnl (converted to R-multiples), then
groups them into trading-day buckets. Saves the pool to logs/trade_pool.json.

At risk_pct=1.0, a trade's pnl_pct fraction == R × 0.01, so R = pnl_pct × 100.
This decouples the realised edge from the risk level, letting the Monte Carlo
re-scale to ANY risk-per-trade without re-running the backtest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.trade_manager import TradeManager
from backtests.tm_backtest import simulate, _ALL_TM_SYMBOLS
from backtests.run_stress_test import _build_strategy, _load_symbol

LOGS = Path(__file__).resolve().parent.parent / "logs"


def extract(symbols=None, risk_pct: float = 1.0) -> dict:
    if symbols is None:
        symbols = _ALL_TM_SYMBOLS

    tm = TradeManager()
    all_trades = []

    for symbol in symbols:
        df15, df5 = _load_symbol(symbol)
        if df15 is None:
            continue
        strat = _build_strategy(symbol)
        res = simulate(
            df_m15=df15, df_m5=df5, strat=strat, trade_manager=tm,
            symbol=symbol, initial_capital=1.0,
            risk_pct=risk_pct, commission=0.0001, daily_halt_pct=2.0,
        )
        tdf = res.trades
        if tdf.empty:
            continue
        tdf = tdf.copy()
        tdf["symbol"] = symbol
        tdf["R"] = tdf["pnl_pct"] * (100.0 / risk_pct)   # normalise to R-multiples
        all_trades.append(tdf[["exit_time", "symbol", "R", "pnl_pct"]])
        print(f"  {symbol:<14} {len(tdf):>4} trades  "
              f"WR={100*(tdf['R']>0).mean():.1f}%  "
              f"avgR={tdf['R'].mean():+.3f}  "
              f"sumR={tdf['R'].sum():+.1f}")

    combined = pd.concat(all_trades, ignore_index=True)
    combined["exit_time"] = pd.to_datetime(combined["exit_time"], utc=True)
    combined = combined.sort_values("exit_time").reset_index(drop=True)
    combined["day"] = combined["exit_time"].dt.date

    # Group R-outcomes by trading day → list of daily trade batches
    day_groups = defaultdict(list)
    for _, row in combined.iterrows():
        day_groups[str(row["day"])].append(round(float(row["R"]), 4))

    daily_batches = [day_groups[d] for d in sorted(day_groups.keys())]

    # ── Distribution stats ──
    all_R = combined["R"].values
    wins  = all_R[all_R > 0]
    losses = all_R[all_R <= 0]

    stats = {
        "n_trades":       int(len(all_R)),
        "n_days":         int(len(daily_batches)),
        "win_rate":       float((all_R > 0).mean()),
        "avg_R":          float(all_R.mean()),
        "avg_win_R":      float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_R":     float(losses.mean()) if len(losses) else 0.0,
        "expectancy_R":   float(all_R.mean()),
        "std_R":          float(all_R.std()),
        "median_trades_per_day": float(np.median([len(b) for b in daily_batches])),
        "max_trades_per_day":    int(max(len(b) for b in daily_batches)),
        "best_day_R":     float(max(sum(b) for b in daily_batches)),
        "worst_day_R":    float(min(sum(b) for b in daily_batches)),
    }

    pool = {
        "stats": stats,
        "daily_batches": daily_batches,
    }

    LOGS.mkdir(exist_ok=True)
    out = LOGS / "trade_pool.json"
    with open(out, "w") as f:
        json.dump(pool, f)

    print(f"\n  === POOL STATS ===")
    for k, v in stats.items():
        print(f"  {k:<24}: {v}")
    print(f"\n  Saved {stats['n_trades']} trades across {stats['n_days']} days -> {out}")
    return pool


if __name__ == "__main__":
    extract()
