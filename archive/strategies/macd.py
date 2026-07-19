"""MACD crossover strategy.

Long when the MACD line crosses above the signal line.
Short when the MACD line crosses below the signal line.
Flat during the warm-up period.
"""
import pandas as pd

from strategies.base import Strategy


class MACDStrategy(Strategy):
    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        long_only: bool = False,
    ):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.long_only = long_only

    @property
    def name(self) -> str:
        suffix = "_long_only" if self.long_only else ""
        return f"MACD({self.fast},{self.slow},{self.signal}){suffix}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)

        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()

        signals = pd.Series(0, index=df.index)
        signals[macd_line > signal_line] = 1
        if not self.long_only:
            signals[macd_line < signal_line] = -1

        # flat during warm-up (slow EMA + signal period)
        warmup = self.slow + self.signal - 1
        signals.iloc[:warmup] = 0

        return signals
