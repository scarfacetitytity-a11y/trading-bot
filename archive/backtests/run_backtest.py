"""CLI entry point: run a named strategy against one or all configured symbols.

Usage (from project root, venv active):
    python -m backtests.run_backtest
    python -m backtests.run_backtest --strategy rsi --symbol XAUUSD
    python -m backtests.run_backtest --strategy all --symbol US30 --chart
    python -m backtests.run_backtest --capital 50000 --save-charts
"""
import argparse
import sys

from config.settings import load_config
from backtests.engine import Backtest
from backtests.plot import plot_result, plot_comparison
from strategies.sma_crossover import SMACrossover
from strategies.rsi import RSIStrategy
from strategies.macd import MACDStrategy
from strategies.bollinger_bands import BollingerBands
from strategies.sniper import SniperStrategy
from strategies.forex_master import ForexMasterStrategy
from strategies.sniper_master import SniperMasterStrategy
from strategies.sniper_trend import SniperTrendStrategy
from strategies.london_breakout import LondonBreakoutStrategy
from strategies.ict_smart_money import ICTSmartMoneyStrategy
from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.trend_rider import TrendRiderStrategy
from strategies.ict_amd import ICTAMDDisplacementStrategy, ICTAMDBreakerStrategy

STRATEGIES = {
    "sma_crossover": SMACrossover,
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
    "bollinger_bands": BollingerBands,
    "sniper": SniperStrategy,
    "forex_master": ForexMasterStrategy,
    "sniper_master": SniperMasterStrategy,
    "sniper_trend": SniperTrendStrategy,
    "london_breakout": LondonBreakoutStrategy,
    "ict_smart_money": ICTSmartMoneyStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "trend_rider": TrendRiderStrategy,
    "ict_amd_disp": ICTAMDDisplacementStrategy,
    "ict_amd_breaker": ICTAMDBreakerStrategy,
}


def _build_strategy(name: str):
    cls = STRATEGIES.get(name)
    if cls is None:
        print(f"Unknown strategy '{name}'. Available: {', '.join(STRATEGIES)}", file=sys.stderr)
        sys.exit(1)
    return cls()


def main():
    all_choices = list(STRATEGIES) + ["all"]
    parser = argparse.ArgumentParser(description="Run a backtest against processed OHLCV data.")
    parser.add_argument("--strategy", default="sma_crossover", choices=all_choices,
                        help="Strategy to run, or 'all' to run every strategy")
    parser.add_argument("--symbol", default=None, help="Single symbol (default: all in config)")
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--commission", type=float, default=0.0001)
    parser.add_argument("--chart", action="store_true", help="Show equity curve chart(s)")
    parser.add_argument("--save-charts", action="store_true", help="Save charts to logs/reports/")
    parser.add_argument("--compare", action="store_true",
                        help="Overlay all strategies on one chart (requires --strategy all)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(*([args.config] if args.config else []))
    timeframe = cfg["data"]["timeframe"].upper()
    symbols = [args.symbol] if args.symbol else cfg["data"]["symbols"]

    strategy_names = list(STRATEGIES) if args.strategy == "all" else [args.strategy]

    any_failed = False
    for symbol in symbols:
        symbol_results = []
        for name in strategy_names:
            strategy = _build_strategy(name)
            try:
                result = Backtest.load_and_run(
                    strategy, symbol, timeframe,
                    initial_capital=args.capital,
                    commission=args.commission,
                )
                result.print_summary()
                symbol_results.append(result)

                if (args.chart or args.save_charts) and not args.compare:
                    plot_result(result, save=args.save_charts, show=args.chart)

            except FileNotFoundError as exc:
                print(f"SKIP {symbol}/{name}: {exc}", file=sys.stderr)
                any_failed = True
            except Exception as exc:
                print(f"ERROR {symbol}/{name}: {exc}", file=sys.stderr)
                any_failed = True

        if args.compare and len(symbol_results) > 1:
            plot_comparison(symbol_results, save=args.save_charts, show=args.chart)

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
