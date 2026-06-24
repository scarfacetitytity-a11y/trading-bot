"""ICT Smart Money / Liquidity Grab Strategy.

Core concept: big money (banks, institutions) hunts stop losses above swing
highs and below swing lows to fill their orders. After the sweep, price
reverses sharply. We enter after the reversal is confirmed.

Entry logic:
  Long  : price sweeps BELOW the N-bar swing low (stop hunt on longs)
           AND closes ABOVE that swing low on the same bar (reversal confirmed)
           = institutions grabbed liquidity below and are now pushing up

  Short : price sweeps ABOVE the N-bar swing high (stop hunt on shorts)
           AND closes BELOW that swing high on the same bar (reversal confirmed)
           = institutions grabbed liquidity above and are now pushing down

Exit (whichever hits first):
  1. ATR stop    — entry ± atr_mult * ATR (tight, we're catching a reversal)
  2. 2:1 target  — atr_target * ATR from entry
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy


class ICTSmartMoneyStrategy(Strategy):
    def __init__(
        self,
        swing_period: int = 20,   # bars to look back for swing high/low
        atr_period: int = 14,
        atr_mult: float = 1.5,
        atr_target: float = 3.0,
        long_only: bool = False,
    ):
        self.swing_period = swing_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.atr_target = atr_target
        self.long_only = long_only

    @property
    def name(self) -> str:
        return (
            f"ICTSmartMoney(swing={self.swing_period}"
            f",atr={self.atr_period}x{self.atr_mult}/tgt{self.atr_target})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close  = df["close"].astype(float)
        open_  = df["open"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)

        # Swing high/low: highest high / lowest low over last N bars (excluding current)
        swing_high = high.shift(1).rolling(self.swing_period).max()
        swing_low  = low.shift(1).rolling(self.swing_period).min()

        atr = _atr(high, low, close, self.atr_period)

        warmup = max(self.swing_period, self.atr_period)

        signals     = pd.Series(0, index=df.index)
        position    = 0
        stop_loss   = None
        take_profit = None

        for i in range(warmup, len(df)):
            c       = close.iloc[i]
            h       = high.iloc[i]
            l       = low.iloc[i]
            o       = open_.iloc[i]
            atr_val = atr.iloc[i]
            s_high  = swing_high.iloc[i]
            s_low   = swing_low.iloc[i]

            # --- Stop hit ---
            if position == 1 and stop_loss is not None and c <= stop_loss:
                position = 0; stop_loss = None; take_profit = None
            elif position == -1 and stop_loss is not None and c >= stop_loss:
                position = 0; stop_loss = None; take_profit = None

            # --- Target hit ---
            if position == 1 and take_profit is not None and c >= take_profit:
                position = 0; stop_loss = None; take_profit = None
            elif position == -1 and take_profit is not None and c <= take_profit:
                position = 0; stop_loss = None; take_profit = None

            # --- Entries ---
            if position == 0 and not np.isnan(atr_val):
                # Bullish liquidity grab: wick below swing low, close above it
                bullish_grab = (
                    l < s_low       # wick swept below swing low
                    and c > s_low   # but closed back above (reversal)
                    and c > o       # bullish candle (closed green)
                )
                # Bearish liquidity grab: wick above swing high, close below it
                bearish_grab = (
                    not self.long_only
                    and h > s_high  # wick swept above swing high
                    and c < s_high  # but closed back below (reversal)
                    and c < o       # bearish candle (closed red)
                )

                if bullish_grab:
                    position    = 1
                    stop_loss   = l - (0.1 * atr_val)   # just below the wick
                    take_profit = c + self.atr_target * atr_val
                elif bearish_grab:
                    position    = -1
                    stop_loss   = h + (0.1 * atr_val)   # just above the wick
                    take_profit = c - self.atr_target * atr_val

            signals.iloc[i] = position

        return signals


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
