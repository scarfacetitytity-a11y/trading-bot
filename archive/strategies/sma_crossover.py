"""Simple SMA crossover strategy.

Long when fast SMA > slow SMA, short when fast SMA < slow SMA.
Flat during the warm-up period (first `slow` bars).
"""
import pandas as pd

from strategies.base import Strategy


class SMACrossover(Strategy):
    def __init__(self, fast: int = 20, slow: int = 50, long_only: bool = False):
        self.fast = fast
        self.slow = slow
        self.long_only = long_only

    @property
    def name(self) -> str:
        suffix = "_long_only" if self.long_only else ""
        return f"SMACrossover({self.fast},{self.slow}){suffix}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast_ma = df["close"].rolling(self.fast).mean()
        slow_ma = df["close"].rolling(self.slow).mean()

        signals = pd.Series(0, index=df.index)
        signals[fast_ma > slow_ma] = 1
        if not self.long_only:
            signals[fast_ma < slow_ma] = -1

        # flat during warm-up
        signals.iloc[: self.slow - 1] = 0

        return signals
