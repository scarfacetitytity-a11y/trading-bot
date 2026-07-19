"""Sniper Master Strategy.

Takes the original Sniper's high-precision entries and adds:
  - ATR stop loss (2.0x)       — cuts losers before they grow
  - Volume confirmation         — avoids low-volume false moves
  - Candle confirmation         — entry bar must close in trade direction
  - Break-even trailing stop    — moves stop to entry once price hits BB midline
  - Full target at opposite BB  — rides the full reversion instead of stopping at midline

Works on any pair and any timeframe. No trend filter — the sniper's own
confluence (RSI extreme + MACD cross + BB band breach) already identifies
genuine turning points without needing a macro trend gate.

Entry logic (ALL must be true):
  Long  : RSI < rsi_oversold                      (extreme oversold)
           AND MACD bullish cross within 10 bars   (momentum turning up)
           AND price below BB lower band           (stretched too far down)
           AND volume > 70% of 20-bar avg          (real participation)
           AND entry bar closes bullish            (candle confirms buyers)
           AND London or NY session (intraday only — skipped on D1+)

  Short : mirror with bearish conditions

Exit (whichever hits first):
  1. ATR stop (2.0x)    — hard stop below entry
  2. Break-even trail   — stop moves to entry price when BB midline is reached
  3. Full target        — exit at BB upper (longs) / BB lower (shorts)
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy


class SniperMasterStrategy(Strategy):
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        macd_cross_window: int = 10,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        atr_mult: float = 2.0,
        vol_period: int = 20,
        vol_threshold: float = 0.7,
        session_filter: bool = True,
        long_only: bool = False,
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.macd_cross_window = macd_cross_window
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.vol_period = vol_period
        self.vol_threshold = vol_threshold
        self.session_filter = session_filter
        self.long_only = long_only

    @property
    def name(self) -> str:
        return (
            f"SniperMaster(rsi={self.rsi_oversold}/{self.rsi_overbought}"
            f",bb={self.bb_period},atr={self.atr_period}x{self.atr_mult})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close  = df["close"].astype(float)
        open_  = df["open"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["tick_volume"].astype(float)

        # --- RSI ---
        rsi = _rsi(close, self.rsi_period)

        # --- MACD crossover (recent within window) ---
        ema_f       = close.ewm(span=self.macd_fast,    adjust=False).mean()
        ema_s       = close.ewm(span=self.macd_slow,    adjust=False).mean()
        macd_line   = ema_f - ema_s
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        bull_cross  = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        bear_cross  = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        recent_bull = bull_cross.rolling(self.macd_cross_window).max().astype(bool)
        recent_bear = bear_cross.rolling(self.macd_cross_window).max().astype(bool)

        # --- Bollinger Bands ---
        bb_mid   = close.rolling(self.bb_period).mean()
        bb_std_s = close.rolling(self.bb_period).std()
        bb_upper = bb_mid + self.bb_std * bb_std_s
        bb_lower = bb_mid - self.bb_std * bb_std_s

        # --- ATR ---
        atr = _atr(high, low, close, self.atr_period)

        # --- Volume: above threshold * rolling average ---
        vol_avg   = volume.rolling(self.vol_period).mean()
        vol_ok    = volume > (vol_avg * self.vol_threshold)

        # --- Candle direction ---
        bullish_bar = close > open_
        bearish_bar = close < open_

        # --- Session filter (auto-skipped on daily+ bars) ---
        if self.session_filter and "time" in df.columns:
            times = pd.to_datetime(df["time"])
            median_hours = times.diff().dt.total_seconds().median() / 3600
            if median_hours >= 20:
                in_session = pd.Series(True, index=df.index)
            else:
                hour = times.dt.hour
                in_session = ((hour >= 7) & (hour < 12)) | ((hour >= 13) & (hour < 17))
        else:
            in_session = pd.Series(True, index=df.index)

        warmup = max(self.macd_slow + self.macd_signal, self.bb_period,
                     self.atr_period, self.vol_period)

        signals      = pd.Series(0, index=df.index)
        self._stops  = pd.Series(float("nan"), index=df.index)
        position     = 0
        stop_loss    = None
        entry_price  = None
        be_triggered = False

        for i in range(warmup, len(df)):
            c       = close.iloc[i]
            atr_val = atr.iloc[i]

            # --- Trail stop to break-even at BB midline ---
            if position == 1 and not be_triggered and c >= bb_mid.iloc[i]:
                stop_loss    = entry_price
                be_triggered = True
            elif position == -1 and not be_triggered and c <= bb_mid.iloc[i]:
                stop_loss    = entry_price
                be_triggered = True

            # --- Stop hit (ATR or break-even) ---
            if position == 1 and stop_loss is not None and c <= stop_loss:
                position = 0; stop_loss = None; entry_price = None; be_triggered = False
            elif position == -1 and stop_loss is not None and c >= stop_loss:
                position = 0; stop_loss = None; entry_price = None; be_triggered = False

            # --- Full target: BB upper (longs) / BB lower (shorts) ---
            if position == 1 and c >= bb_upper.iloc[i]:
                position = 0; stop_loss = None; entry_price = None; be_triggered = False
            elif position == -1 and c <= bb_lower.iloc[i]:
                position = 0; stop_loss = None; entry_price = None; be_triggered = False

            # --- Entries ---
            if position == 0 and not np.isnan(atr_val):
                long_entry = (
                    rsi.iloc[i] < self.rsi_oversold
                    and recent_bull.iloc[i]
                    and c < bb_lower.iloc[i]
                    and vol_ok.iloc[i]
                    and bullish_bar.iloc[i]
                    and in_session.iloc[i]
                )
                short_entry = (
                    not self.long_only
                    and rsi.iloc[i] > self.rsi_overbought
                    and recent_bear.iloc[i]
                    and c > bb_upper.iloc[i]
                    and vol_ok.iloc[i]
                    and bearish_bar.iloc[i]
                    and in_session.iloc[i]
                )

                if long_entry:
                    position     = 1
                    entry_price  = c
                    stop_loss    = c - self.atr_mult * atr_val
                    be_triggered = False
                elif short_entry:
                    position     = -1
                    entry_price  = c
                    stop_loss    = c + self.atr_mult * atr_val
                    be_triggered = False

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")

        return signals


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
