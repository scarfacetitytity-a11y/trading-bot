"""Donchian Channel Breakout Strategy (Turtle Trading).

One of the most proven trend-following strategies in history — used by the
original Turtle Traders in the 1980s and still works today.

Entry:
  Long  : close breaks above the highest high of the last entry_period bars
  Short : close breaks below the lowest low of the last entry_period bars

Exit (whichever hits first):
  1. Trailing ATR stop — trails price to lock in profits
  2. Donchian exit     — price crosses the opposite exit_period channel
  3. Fixed ATR stop    — hard floor if trailing stop hasn't moved yet
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy


class DonchianBreakoutStrategy(Strategy):
    def __init__(
        self,
        entry_period: int = 20,  # breakout of N-bar high/low to enter
        exit_period: int = 10,   # opposite channel to exit
        atr_period: int = 14,
        atr_mult: float = 2.0,   # hard stop distance on entry
        long_only: bool = False,
    ):
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.long_only = long_only

    @property
    def name(self) -> str:
        return (
            f"DonchianBreakout(entry={self.entry_period}"
            f",exit={self.exit_period},atr={self.atr_period}x{self.atr_mult})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        # Entry channels (shift by 1 to avoid look-ahead)
        entry_high = high.shift(1).rolling(self.entry_period).max()
        entry_low  = low.shift(1).rolling(self.entry_period).min()

        # Exit channels (shorter — tighter)
        exit_high  = high.shift(1).rolling(self.exit_period).max()
        exit_low   = low.shift(1).rolling(self.exit_period).min()

        atr = _atr(high, low, close, self.atr_period)

        warmup = max(self.entry_period, self.atr_period)

        signals     = pd.Series(0, index=df.index)
        self._stops = pd.Series(float("nan"), index=df.index)
        position    = 0
        stop_loss   = None
        trail_stop  = None

        for i in range(warmup, len(df)):
            c       = close.iloc[i]
            h       = high.iloc[i]
            l       = low.iloc[i]
            atr_val = atr.iloc[i]

            # --- Trail the stop as price extends ---
            if position == 1:
                new_trail = h - self.atr_mult * atr_val
                if trail_stop is None or new_trail > trail_stop:
                    trail_stop = new_trail
                stop_loss = trail_stop

            elif position == -1:
                new_trail = l + self.atr_mult * atr_val
                if trail_stop is None or new_trail < trail_stop:
                    trail_stop = new_trail
                stop_loss = trail_stop

            # --- Trailing / hard stop hit ---
            if position == 1 and stop_loss is not None and c <= stop_loss:
                position = 0; stop_loss = None; trail_stop = None
            elif position == -1 and stop_loss is not None and c >= stop_loss:
                position = 0; stop_loss = None; trail_stop = None

            # --- Donchian exit channel ---
            if position == 1 and c <= exit_low.iloc[i]:
                position = 0; stop_loss = None; trail_stop = None
            elif position == -1 and c >= exit_high.iloc[i]:
                position = 0; stop_loss = None; trail_stop = None

            # --- Entries ---
            if position == 0 and not np.isnan(atr_val):
                if c > entry_high.iloc[i]:
                    position   = 1
                    stop_loss  = c - self.atr_mult * atr_val
                    trail_stop = stop_loss
                elif not self.long_only and c < entry_low.iloc[i]:
                    position   = -1
                    stop_loss  = c + self.atr_mult * atr_val
                    trail_stop = stop_loss

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")

        return signals


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
