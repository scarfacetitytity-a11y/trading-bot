"""AiDEN Index Strategy — M2 validation run.

Quant (MEM-002): validates AiDENIndexStrategy edge on US100 and US30.
Compares against the baseline FVG-OB-Long across all available years.

Usage:
    python -m backtests.run_aiden_index
    python -m backtests.run_aiden_index --symbol US100.cash
    python -m backtests.run_aiden_index --sweep          # score threshold sweep
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from backtests.engine import Backtest
from strategies.aiden_index import AiDENIndexStrategy
from strategies.fvg_ob import FVGOrderBlockStrategy

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS   = Path(__file__).resolve().parent.parent / "logs" / "reports"

TARGETS = ["US100.cash", "US30.cash"]

STRATEGIES = {
    "AiDEN-Index(4)":  AiDENIndexStrategy(min_score=4, min_fvg_atr=0.15, rr_target=2.5),
    "AiDEN-Index(5)":  AiDENIndexStrategy(min_score=5, min_fvg_atr=0.15, rr_target=2.5),
    "AiDEN-Index(3)":  AiDENIndexStrategy(min_score=3, min_fvg_atr=0.1,  rr_target=2.0),
    "FVG-OB-Long(baseline)": FVGOrderBlockStrategy(
        min_fvg_atr=0.1, rr_target=2.0, ob_required=True, long_only=True
    ),
}


def _load(symbol: str, timeframe: str = "H1") -> pd.DataFrame | None:
    path = PROCESSED / f"{symbol}_{timeframe}.csv"
    if not path.exists():
        print(f"  SKIP {symbol}: no data at {path.name}", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def _run(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        df = _load(symbol)
        if df is None:
            continue
        for name, strat in STRATEGIES.items():
            try:
                bt = Backtest(strat, initial_capital=10_000, commission=0.0001)
                r  = bt.run(df, symbol=symbol)
                m  = r.metrics
                rows.append({
                    "Symbol":   symbol,
                    "Strategy": name,
                    "Return%":  m["total_return_pct"],
                    "CAGR%":    m["cagr_pct"],
                    "MaxDD%":   m["max_drawdown_pct"],
                    "Sharpe":   m["sharpe_ratio"],
                    "Trades":   m["total_trades"],
                    "Win%":     m["win_rate_pct"],
                    "PF":       m["profit_factor"],
                })
            except Exception as exc:
                print(f"  ERROR {symbol}/{name}: {exc}", file=sys.stderr)
    return pd.DataFrame(rows)


def _print(df: pd.DataFrame) -> None:
    for symbol in df["Symbol"].unique():
        chunk = df[df["Symbol"] == symbol].sort_values("Sharpe", ascending=False)
        print(f"\n{'='*72}")
        print(f"  {symbol}")
        print(f"{'='*72}")
        print(f"  {'Strategy':<28}  {'Return%':>8}  {'CAGR%':>7}  {'MaxDD%':>7}  {'Sharpe':>7}  {'Trades':>6}  {'Win%':>6}  {'PF':>5}")
        print(f"  {'-'*68}")
        for _, r in chunk.iterrows():
            ftmo_flag = " ** FTMO-BREACHED" if r["MaxDD%"] < -10 else ""
            print(
                f"  {r['Strategy']:<28}  {r['Return%']:>+8.2f}  {r['CAGR%']:>+7.2f}  "
                f"{r['MaxDD%']:>7.2f}  {r['Sharpe']:>7.3f}  {int(r['Trades']):>6}  "
                f"{r['Win%']:>6.1f}  {r['PF']:>5.2f}{ftmo_flag}"
            )


def _sweep(symbol: str) -> None:
    """Score threshold sweep — Quant finds the FTMO-safe configuration."""
    df = _load(symbol)
    if df is None:
        return

    configs = [
        (s, f, rr)
        for s in [3, 4, 5]
        for f in [0.1, 0.15, 0.2]
        for rr in [2.0, 2.5, 3.0]
    ]

    rows = []
    for score, fvg, rr in configs:
        strat = AiDENIndexStrategy(min_score=score, min_fvg_atr=fvg, rr_target=rr)
        bt    = Backtest(strat, initial_capital=10_000, commission=0.0001)
        r     = bt.run(df, symbol=symbol)
        m     = r.metrics
        rows.append({
            "score": score, "fvg_atr": fvg, "rr": rr,
            "Return%": m["total_return_pct"], "MaxDD%": m["max_drawdown_pct"],
            "Sharpe": m["sharpe_ratio"], "Trades": m["total_trades"],
            "Win%": m["win_rate_pct"], "PF": m["profit_factor"],
        })

    df_r = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    print(f"\nSweep: AiDENIndexStrategy on {symbol}")
    print(f"{'score':>5} {'fvg':>5} {'rr':>4}  {'Return%':>8} {'MaxDD%':>7} {'Sharpe':>7} {'Trades':>7} {'Win%':>6} {'PF':>5}  FTMO")
    print("-" * 75)
    for _, r in df_r.iterrows():
        ftmo = "OK" if r["MaxDD%"] > -10 else "BREACH"
        print(
            f"{int(r['score']):>5} {r['fvg_atr']:>5.2f} {r['rr']:>4.1f}  "
            f"{r['Return%']:>+8.2f} {r['MaxDD%']:>7.2f} {r['Sharpe']:>7.3f} "
            f"{int(r['Trades']):>7} {r['Win%']:>6.1f} {r['PF']:>5.2f}  {ftmo}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--sweep",  action="store_true")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else TARGETS

    if args.sweep:
        for s in symbols:
            _sweep(s)
    else:
        df = _run(symbols)
        if df.empty:
            print("No results.")
            sys.exit(1)
        _print(df)


if __name__ == "__main__":
    main()
