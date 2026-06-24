"""Abstract base class for all trading strategies."""
from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Subclass this and implement generate_signals().

    generate_signals() receives the full OHLCV DataFrame and must return a
    pd.Series (same index) where:
        1  = go long
       -1  = go short
        0  = flat / exit any open position

    Signals are interpreted as the *desired* position at bar close.
    The engine enters/exits at the next bar's open (approximated as current
    bar's close in a vectorised backtest).
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a signal series aligned to df's index."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
