"""RSI mean-reversion strategy.

Long when RSI is oversold (below oversold threshold).
Short when RSI is overbought (above overbought threshold).
Flat when RSI is in the neutral zone.
"""
import pandas as pd

from strategies.base import Strategy


class RSIStrategy(Strategy):
    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        long_only: bool = False,
    ):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.long_only = long_only

    @property
    def name(self) -> str:
        suffix = "_long_only" if self.long_only else ""
        return f"RSI({self.period},{self.oversold},{self.overbought}){suffix}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        rsi = _rsi(close, self.period)

        signals = pd.Series(0, index=df.index)
        signals[rsi < self.oversold] = 1
        if not self.long_only:
            signals[rsi > self.overbought] = -1

        signals.iloc[: self.period] = 0
        return signals


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))
