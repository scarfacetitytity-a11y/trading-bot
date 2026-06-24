"""Vectorised backtesting engine.

Usage
-----
    from backtests.engine import Backtest
    from strategies.sma_crossover import SMACrossover

    bt = Backtest(SMACrossover(fast=20, slow=50), initial_capital=10_000, commission=0.0001)
    result = bt.run(df)   # df is a processed OHLCV DataFrame
    result.print_summary()
"""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from backtests.metrics import calculate_metrics
from strategies.base import Strategy

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: dict
    signals: pd.Series = field(repr=False)

    def print_summary(self) -> None:
        m = self.metrics
        print(f"\n{'='*52}")
        print(f"  {self.strategy_name} | {self.symbol}")
        print(f"{'='*52}")
        print(f"  Period          : {m['years']:.1f} years ({m['total_bars']:,} bars)")
        print(f"  Total return    : {m['total_return_pct']:+.2f}%")
        print(f"  CAGR            : {m['cagr_pct']:+.2f}%")
        print(f"  Max drawdown    : {m['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe ratio    : {m['sharpe_ratio']:.3f}")
        print(f"  Total trades    : {m['total_trades']}")
        print(f"  Win rate        : {m['win_rate_pct']:.1f}%")
        print(f"  Profit factor   : {m['profit_factor']:.3f}")
        print(f"  Avg win         : {m['avg_win_pct']:+.3f}%")
        print(f"  Avg loss        : {m['avg_loss_pct']:+.3f}%")
        print(f"  Avg trade       : {m['avg_trade_pct']:+.3f}%")
        print(f"{'='*52}\n")


class Backtest:
    """Run a strategy against processed OHLCV data.

    Parameters
    ----------
    strategy:        a Strategy subclass instance
    initial_capital: starting equity in account currency
    commission:      one-way commission as a fraction of trade value
                     (e.g. 0.0001 = 1 pip on a standard forex pair)
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 10_000,
        commission: float = 0.0001,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> BacktestResult:
        """Run the backtest and return a BacktestResult.

        df must have columns: time, open, high, low, close, tick_volume
        """
        df = df.reset_index(drop=True)
        signals = self.strategy.generate_signals(df)

        # Position held *during* each bar: shift signal by 1 so we act on the
        # next bar after the signal fires (no look-ahead).
        positions = signals.shift(1).fillna(0)

        close = df["close"].astype(float)
        bar_returns = close.pct_change().fillna(0)

        # Strategy return per bar = position * price return - commission on changes
        position_changes = positions.diff().abs().fillna(0)
        strat_returns = positions * bar_returns - position_changes * self.commission

        equity = pd.Series(
            self.initial_capital * (1 + strat_returns).cumprod(),
            index=df.index,
        )

        trades = self._extract_trades(df, positions, close)
        bars_per_year = _estimate_bars_per_year(df)
        metrics = calculate_metrics(strat_returns, equity, trades, bars_per_year)

        return BacktestResult(
            strategy_name=self.strategy.name,
            symbol=symbol,
            equity=equity,
            returns=strat_returns,
            trades=trades,
            metrics=metrics,
            signals=signals,
        )

    def _extract_trades(
        self, df: pd.DataFrame, positions: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        trades = []
        entry_price = None
        entry_time = None
        direction = 0

        for i in range(len(positions)):
            pos = int(positions.iloc[i])
            prev = direction

            if pos == prev:
                continue

            # Close any open trade
            if prev != 0:
                exit_price = close.iloc[i]
                raw_pnl = (exit_price - entry_price) / entry_price * prev
                commission_cost = self.commission * 2  # round trip
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": df["time"].iloc[i],
                    "direction": prev,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": raw_pnl - commission_cost,
                })

            # Open a new trade
            if pos != 0:
                entry_price = close.iloc[i]
                entry_time = df["time"].iloc[i]

            direction = pos

        return pd.DataFrame(trades)

    @classmethod
    def load_and_run(
        cls,
        strategy: Strategy,
        symbol: str,
        timeframe: str,
        initial_capital: float = 10_000,
        commission: float = 0.0001,
    ) -> BacktestResult:
        """Load a processed CSV by symbol/timeframe and run the backtest."""
        path = PROCESSED_DIR / f"{symbol}_{timeframe.upper()}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No processed data for {symbol} {timeframe}.\n"
                f"Run the data pipeline first: python -m backtests.run_data_pipeline"
            )
        df = pd.read_csv(path)
        bt = cls(strategy, initial_capital=initial_capital, commission=commission)
        return bt.run(df, symbol=symbol)


def _estimate_bars_per_year(df: pd.DataFrame) -> float:
    """Derive annualisation factor from actual bar timestamps."""
    times = pd.to_datetime(df["time"])
    if len(times) < 2:
        return 252.0
    duration_years = (times.iloc[-1] - times.iloc[0]).total_seconds() / (365.25 * 24 * 3600)
    return len(times) / duration_years if duration_years > 0 else 252.0
