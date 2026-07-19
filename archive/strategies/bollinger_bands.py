"""Bollinger Bands mean-reversion strategy.

Enter long when price closes below the lower band (oversold).
Enter short when price closes above the upper band (overbought).
Exit when price reverts to the middle band (SMA).
Hold position between entry and exit signals.
"""
import pandas as pd

from strategies.base import Strategy


class BollingerBands(Strategy):
    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        long_only: bool = False,
    ):
        self.period = period
        self.std_dev = std_dev
        self.long_only = long_only

    @property
    def name(self) -> str:
        suffix = "_long_only" if self.long_only else ""
        return f"BollingerBands({self.period},{self.std_dev}){suffix}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        middle = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std

        signals = pd.Series(0, index=df.index)
        position = 0

        for i in range(self.period, len(df)):
            c = close.iloc[i]
            m = middle.iloc[i]

            if c < lower.iloc[i]:
                position = 1
            elif not self.long_only and c > upper.iloc[i]:
                position = -1
            elif position == 1 and c >= m:
                position = 0
            elif position == -1 and c <= m:
                position = 0

            signals.iloc[i] = position

        return signals
