"""Per-instrument M15 parameter sweep — finds optimal config per instrument.

Tests combinations of score, rr, min_fvg_atr, h4_bias_method, atr_stop_buffer,
session windows, and RSI filter. Outputs best config per instrument and a
recommended combined portfolio.

Usage:
    python -m backtests.sweep_m15_instruments                  # all instruments
    python -m backtests.sweep_m15_instruments --sym XAUUSD     # single
    python -m backtests.sweep_m15_instruments --top 3          # show top 3 per instrument
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.engine import Backtest
from backtests.metrics import calculate_metrics
from strategies.aiden_index import AiDENIndexStrategy

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

# ── Instrument universe ───────────────────────────────────────────────────────
# session_variants allows testing multiple windows per instrument
INSTRUMENTS = {
    "XAUUSD": [
        dict(session_start=7,  session_end=21, label="London+NY"),
        dict(session_start=7,  session_end=16, label="London-only"),
        dict(session_start=12, session_end=21, label="NY-only"),
    ],
    "US100.cash": [
        dict(session_start=12, session_end=21, label="NY"),
        dict(session_start=13, session_end=20, label="NY-core"),
    ],
    "US30.cash": [
        dict(session_start=12, session_end=21, label="NY"),
        dict(session_start=13, session_end=20, label="NY-core"),
    ],
    "US500.cash": [
        dict(session_start=12, session_end=21, label="NY"),
        dict(session_start=13, session_end=20, label="NY-core"),
    ],
    "US2000.cash": [
        dict(session_start=12, session_end=21, label="NY"),
        dict(session_start=13, session_end=20, label="NY-core"),
    ],
    "GER40.cash": [
        dict(session_start=7,  session_end=17, label="Frankfurt"),
        dict(session_start=7,  session_end=12, label="EU-morning"),
        dict(session_start=8,  session_end=16, label="EU-core"),
    ],
    "JP225.cash": [
        dict(session_start=0,  session_end=9,  label="Tokyo"),
        dict(session_start=1,  session_end=8,  label="Tokyo-core"),
    ],
    # Alternates — tested as potential replacements for underperformers
    "UK100.cash": [
        dict(session_start=7,  session_end=17, label="London"),
        dict(session_start=8,  session_end=16, label="London-core"),
    ],
    "XAGUSD": [
        dict(session_start=7,  session_end=21, label="London+NY"),
        dict(session_start=12, session_end=21, label="NY-only"),
    ],
}

# ── Sweep grid ────────────────────────────────────────────────────────────────
GRID = {
    "min_score":        [3, 4, 5],
    "rr_target":        [2.5, 3.0, 3.5],
    "min_fvg_atr":      [0.10, 0.15],
    "h4_bias_method":   ["ema", "swing"],
    "atr_stop_buffer":  [0.3, 0.5],
}

# Fixed M15 params (input-TF bars scaled from H1)
M15_BASE = dict(
    htf_lookback=20,       # H4 bars
    h4_swing_lookback=40,  # H4 bars
    max_fvg_wait=160,
    max_entry_wait=32,
    ob_lookback=80,
    liq_lookback=40,
    atr_period=56,
    use_rsi=True,
    rsi_period=56,
)

COMMISSION  = 0.0001
INITIAL_CAP = 10_000
MIN_TRADES  = 15          # ignore configs with too few trades
MAX_DD_GATE = -12.0       # FTMO gate: drop configs that breach more than -12%


def _load(symbol: str) -> pd.DataFrame | None:
    path = PROCESSED / f"{symbol}_M15.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def _run_one(df: pd.DataFrame, symbol: str, params: dict) -> dict | None:
    """Run backtest with given params. Returns metrics dict or None on failure."""
    try:
        strat = AiDENIndexStrategy(**params)
        bt    = Backtest(strat, initial_capital=INITIAL_CAP, commission=COMMISSION)
        r     = bt.run(df, symbol=symbol)

        times   = pd.to_datetime(df["time"])
        returns = r.returns.copy()
        returns.index = times
        equity  = pd.Series(INITIAL_CAP * (1 + returns).cumprod(), index=times)

        bpy  = _estimate_bpy(returns)
        m    = calculate_metrics(returns, equity, r.trades, bpy)

        if m["total_trades"] < MIN_TRADES:
            return None
        if m["max_drawdown_pct"] < MAX_DD_GATE:
            return None

        return m
    except Exception:
        return None


def _estimate_bpy(returns: pd.Series) -> float:
    idx = pd.to_datetime(returns.index)
    if len(idx) < 2:
        return 252.0
    span = (idx[-1] - idx[0]).days
    return len(returns) / (span / 365.25) if span > 0 else 252.0


def sweep_instrument(symbol: str, top_n: int = 3) -> list[dict]:
    """Return sorted list of best configs for this instrument."""
    df = _load(symbol)
    if df is None:
        print(f"  {symbol}: no M15 data — skipping")
        return []

    sessions = INSTRUMENTS.get(symbol, [dict(session_start=12, session_end=21, label="default")])
    keys = list(GRID.keys())
    values = list(GRID.values())

    results = []
    total   = len(sessions) * sum(1 for _ in product(*values))
    done    = 0

    for sess in sessions:
        for combo in product(*values):
            params = dict(zip(keys, combo))
            params.update(M15_BASE)
            params["session_start"] = sess["session_start"]
            params["session_end"]   = sess["session_end"]
            params["long_only"]     = True

            m = _run_one(df, symbol, params)
            done += 1

            if m is not None:
                results.append({
                    "symbol":       symbol,
                    "session":      sess["label"],
                    "score":        params["min_score"],
                    "rr":           params["rr_target"],
                    "fvg_atr":      params["min_fvg_atr"],
                    "bias_method":  params["h4_bias_method"],
                    "stop_buf":     params["atr_stop_buffer"],
                    "sess_start":   sess["session_start"],
                    "sess_end":     sess["session_end"],
                    "return_pct":   m["total_return_pct"],
                    "max_dd":       m["max_drawdown_pct"],
                    "sharpe":       m["sharpe_ratio"],
                    "trades":       m["total_trades"],
                    "win_rate":     m["win_rate_pct"],
                    "profit_factor": m["profit_factor"],
                    "years":        m["years"],
                })

        sys.stdout.write(f"\r  {symbol}: {done}/{total} configs tested    ")
        sys.stdout.flush()

    print()

    if not results:
        print(f"  {symbol}: no valid configs found")
        return []

    # Sort: Sharpe primary, return secondary, penalise DD > -8%
    def _score(r):
        dd_penalty = max(0, (-r["max_dd"] - 8.0)) * 0.5
        return r["sharpe"] - dd_penalty + r["return_pct"] * 0.01

    results.sort(key=_score, reverse=True)
    return results[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym",  default=None, help="Single symbol to sweep")
    parser.add_argument("--top",  type=int, default=3, help="Top N configs per instrument")
    args = parser.parse_args()

    symbols = [args.sym] if args.sym else list(INSTRUMENTS.keys())
    best_per_instrument: dict[str, dict] = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  {symbol}")
        print(f"{'='*60}")

        top = sweep_instrument(symbol, top_n=args.top)
        if not top:
            continue

        print(f"\n  {'Rank':<5} {'Score':<7} {'RR':<5} {'Method':<7} "
              f"{'Stop':<6} {'FVG':<6} {'Session':<14} "
              f"{'Return':>8} {'MaxDD':>7} {'Sharpe':>7} {'Trades':>7}")
        print(f"  {'-'*110}")

        for rank, r in enumerate(top, 1):
            ftmo = "" if r["max_dd"] > -10 else " !DD"
            print(f"  {rank:<5} {r['score']:<7} {r['rr']:<5} {r['bias_method']:<7} "
                  f"{r['stop_buf']:<6} {r['fvg_atr']:<6} {r['session']:<14} "
                  f"{r['return_pct']:>+8.2f}% {r['max_dd']:>7.2f}% "
                  f"{r['sharpe']:>7.3f} {r['trades']:>7}{ftmo}")

        best = top[0]
        best_per_instrument[symbol] = best
        print(f"\n  BEST: score={best['score']} rr={best['rr']} "
              f"bias={best['bias_method']} stop={best['stop_buf']} "
              f"fvg={best['fvg_atr']} session={best['session']}")

    # ── Portfolio summary ─────────────────────────────────────────────────────
    if len(best_per_instrument) < 2:
        return

    print(f"\n\n{'='*60}")
    print("  RECOMMENDED PORTFOLIO (per-instrument best configs)")
    print(f"{'='*60}")
    print(f"\n  {'Symbol':<16} {'Score':<7} {'RR':<5} {'Method':<7} "
          f"{'Session':<14} {'Return':>8} {'MaxDD':>7} {'Sharpe':>7} {'Tr/mo':>7}")
    print(f"  {'-'*90}")

    total_trades_pm = 0
    for sym, b in best_per_instrument.items():
        trades_pm = b["trades"] / max(b["years"] * 12, 0.01)
        total_trades_pm += trades_pm
        print(f"  {sym:<16} {b['score']:<7} {b['rr']:<5} {b['bias_method']:<7} "
              f"{b['session']:<14} {b['return_pct']:>+8.2f}% "
              f"{b['max_dd']:>7.2f}% {b['sharpe']:>7.3f} {trades_pm:>7.1f}")

    print(f"\n  Projected portfolio trades/month (sum): {total_trades_pm:.1f}")
    print(f"  Instruments kept: {len(best_per_instrument)}")

    # Drop underperformers
    keepers = {s: b for s, b in best_per_instrument.items()
               if b["sharpe"] > 0.1 and b["return_pct"] > 0}
    dropped = set(best_per_instrument) - set(keepers)
    if dropped:
        print(f"  Instruments to drop (Sharpe<0.1 or return<0): {', '.join(dropped)}")
        print(f"  Remaining: {', '.join(keepers.keys())}")

    print()

    # Print config block for run_multi_instrument.py
    print("  --- Copy these per-instrument params into run_multi_instrument.py ---")
    for sym, b in keepers.items():
        print(f"  '{sym}': dict(score={b['score']}, rr={b['rr']}, "
              f"fvg_atr={b['fvg_atr']}, bias='{b['bias_method']}', "
              f"stop={b['stop_buf']}, sess=({b['sess_start']},{b['sess_end']})),")
    print()


if __name__ == "__main__":
    main()
