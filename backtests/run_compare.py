"""Full comparison: every strategy × every symbol × every year.

Loads all processed CSVs, slices by calendar year, runs every strategy on
every period, and prints a ranked results table.

Usage (project root, venv active):
    python -m backtests.run_compare
    python -m backtests.run_compare --symbol XAUUSD
    python -m backtests.run_compare --capital 10000 --commission 0.0001
    python -m backtests.run_compare --save          # CSV to logs/reports/
    python -m backtests.run_compare --charts        # comparison chart per symbol/year
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from config.settings import load_config
from backtests.engine import Backtest
from backtests.plot import plot_comparison
from strategies.sma_crossover import SMACrossover
from strategies.rsi import RSIStrategy
from strategies.macd import MACDStrategy
from strategies.bollinger_bands import BollingerBands

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "logs" / "reports"

STRATEGIES = {
    "SMA(20,50)":    SMACrossover(fast=20, slow=50),
    "SMA(50,200)":   SMACrossover(fast=50, slow=200),
    "RSI(14)":       RSIStrategy(period=14),
    "MACD(12,26,9)": MACDStrategy(fast=12, slow=26, signal=9),
    "BB(20,2)":      BollingerBands(period=20, std_dev=2.0),
}

MIN_BARS = 150  # skip a period slice if it has fewer bars than this


def _load_and_slice(path: Path) -> dict[str, pd.DataFrame]:
    """Return {'2022': df, '2023': df, ..., 'Full': df}."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    years = sorted(df["time"].dt.year.unique())
    slices: dict[str, pd.DataFrame] = {}
    for y in years:
        chunk = df[df["time"].dt.year == y].reset_index(drop=True)
        if len(chunk) >= MIN_BARS:
            slices[str(y)] = chunk
    if len(df) >= MIN_BARS:
        slices["Full"] = df
    return slices


def _run_all(
    symbols: list[str],
    timeframe: str,
    initial_capital: float,
    commission: float,
) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        path = PROCESSED_DIR / f"{symbol}_{timeframe}.csv"
        if not path.exists():
            print(f"  SKIP {symbol}: no processed data at {path.name}", file=sys.stderr)
            continue

        slices = _load_and_slice(path)
        if not slices:
            print(f"  SKIP {symbol}: not enough data", file=sys.stderr)
            continue

        for period, df in slices.items():
            for strat_name, strategy in STRATEGIES.items():
                try:
                    bt = Backtest(strategy, initial_capital=initial_capital, commission=commission)
                    result = bt.run(df, symbol=symbol)
                    m = result.metrics
                    rows.append({
                        "Symbol":   symbol,
                        "Period":   period,
                        "Strategy": strat_name,
                        "Return%":  m["total_return_pct"],
                        "CAGR%":    m["cagr_pct"],
                        "MaxDD%":   m["max_drawdown_pct"],
                        "Sharpe":   m["sharpe_ratio"],
                        "Trades":   m["total_trades"],
                        "Win%":     m["win_rate_pct"],
                        "PF":       m["profit_factor"],
                    })
                except Exception as exc:
                    print(f"  ERROR {symbol}/{period}/{strat_name}: {exc}", file=sys.stderr)

    return pd.DataFrame(rows)


def _print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results to display.")
        return

    for symbol in df["Symbol"].unique():
        sym_df = df[df["Symbol"] == symbol]
        for period in _period_order(sym_df["Period"].unique()):
            chunk = sym_df[sym_df["Period"] == period].copy()
            if chunk.empty:
                continue

            chunk = chunk.sort_values("Return%", ascending=False).reset_index(drop=True)
            chunk.insert(0, "Rank", range(1, len(chunk) + 1))

            print(f"\n{'='*72}")
            print(f"  {symbol}  |  {period}")
            print(f"{'='*72}")

            col_fmt = (
                f"  {'Rank':>4}  {'Strategy':<16}  {'Return%':>8}  {'CAGR%':>7}  "
                f"{'MaxDD%':>7}  {'Sharpe':>7}  {'Trades':>6}  {'Win%':>6}  {'PF':>6}"
            )
            print(col_fmt)
            print(f"  {'-'*68}")

            for _, row in chunk.iterrows():
                pf = f"{row['PF']:.2f}" if row["PF"] != float("inf") else "  inf"
                print(
                    f"  {int(row['Rank']):>4}  {row['Strategy']:<16}  "
                    f"{row['Return%']:>+8.2f}  {row['CAGR%']:>+7.2f}  "
                    f"{row['MaxDD%']:>7.2f}  {row['Sharpe']:>7.3f}  "
                    f"{int(row['Trades']):>6}  {row['Win%']:>6.1f}  {pf:>6}"
                )

    _print_winners(df)


def _print_winners(df: pd.DataFrame) -> None:
    """Print a concise 'best strategy per symbol per year' summary."""
    full_only = df[df["Period"] != "Full"]
    if full_only.empty:
        full_only = df

    best = (
        full_only.loc[full_only.groupby(["Symbol", "Period"])["Return%"].idxmax()]
        .sort_values(["Symbol", "Period"])
        [["Symbol", "Period", "Strategy", "Return%", "Sharpe"]]
        .reset_index(drop=True)
    )

    print(f"\n{'='*72}")
    print("  BEST STRATEGY PER SYMBOL / PERIOD  (by Return%)")
    print(f"{'='*72}")
    print(f"  {'Symbol':<10}  {'Period':<8}  {'Strategy':<16}  {'Return%':>8}  {'Sharpe':>7}")
    print(f"  {'-'*60}")
    for _, row in best.iterrows():
        print(
            f"  {row['Symbol']:<10}  {row['Period']:<8}  {row['Strategy']:<16}  "
            f"{row['Return%']:>+8.2f}  {row['Sharpe']:>7.3f}"
        )
    print()


def _period_order(periods) -> list[str]:
    """Sort periods: years numerically, 'Full' last."""
    years = sorted([p for p in periods if p != "Full"])
    return years + (["Full"] if "Full" in periods else [])


def _save_csv(df: pd.DataFrame, timeframe: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"comparison_{timeframe}.csv"
    df.to_csv(path, index=False)
    print(f"Results saved → {path}")


def _save_charts(df: pd.DataFrame, symbols: list, timeframe: str,
                 initial_capital: float, commission: float) -> None:
    """One comparison chart per symbol per year."""
    for symbol in symbols:
        path = PROCESSED_DIR / f"{symbol}_{timeframe}.csv"
        if not path.exists():
            continue
        slices = _load_and_slice(path)
        for period, slice_df in slices.items():
            results = []
            for strat_name, strategy in STRATEGIES.items():
                try:
                    bt = Backtest(strategy, initial_capital=initial_capital, commission=commission)
                    r = bt.run(slice_df, symbol=symbol)
                    r.strategy_name = strat_name  # override for chart label
                    results.append(r)
                except Exception:
                    pass
            if results:
                plot_comparison(results, save=True, show=False)
                print(f"Chart saved: {symbol} / {period}")


def main():
    parser = argparse.ArgumentParser(description="Compare all strategies across all symbols and years.")
    parser.add_argument("--symbol", default=None, help="Single symbol (default: all in config)")
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--commission", type=float, default=0.0001)
    parser.add_argument("--save", action="store_true", help="Save results CSV to logs/reports/")
    parser.add_argument("--charts", action="store_true", help="Save comparison charts per symbol/year")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(*([args.config] if args.config else []))
    timeframe = cfg["data"]["timeframe"].upper()
    symbols = [args.symbol] if args.symbol else cfg["data"]["symbols"]

    print(f"\nRunning comparison: {len(STRATEGIES)} strategies × {len(symbols)} symbol(s) × all years")
    print(f"Timeframe: {timeframe}  |  Capital: {args.capital:,.0f}  |  Commission: {args.commission}\n")

    results_df = _run_all(symbols, timeframe, args.capital, args.commission)

    if results_df.empty:
        print("No results — run the data pipeline first: python -m backtests.run_data_pipeline")
        sys.exit(1)

    _print_table(results_df)

    if args.save:
        _save_csv(results_df, timeframe)

    if args.charts:
        _save_charts(results_df, symbols, timeframe, args.capital, args.commission)


if __name__ == "__main__":
    main()
