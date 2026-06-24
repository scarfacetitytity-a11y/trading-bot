"""Forex Master Strategy.

Combines all indicators with forex-specific knowledge:

  Trend filter  : SMA 50 vs SMA 200 — only trade with the trend
  Entry timing  : RSI pullback + MACD momentum + Bollinger Band position
  ATR stop loss : dynamic stop at entry ± atr_mult * ATR(14) — cuts losers early
  Session filter: London (07:00-12:00 UTC) and New York (13:00-17:00 UTC) only
                  — highest liquidity, tightest spreads, strongest moves

Long  : SMA50 > SMA200 AND RSI < 55 AND MACD bullish AND price below BB midline
Short : SMA50 < SMA200 AND RSI > 45 AND MACD bearish AND price above BB midline
Exit  : ATR stop hit OR trend reversal OR RSI extreme against position
"""
import pandas as pd
import numpy as np

from strategies.base import Strategy


class ForexMasterStrategy(Strategy):
    def __init__(
        self,
        sma_fast: int = 50,
        sma_slow: int = 200,
        rsi_period: int = 14,
        rsi_long_max: float = 55.0,
        rsi_short_min: float = 45.0,
        rsi_exit_long: float = 70.0,
        rsi_exit_short: float = 30.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        atr_mult: float = 1.5,        # stop = entry ± atr_mult * ATR
        session_filter: bool = True,
        long_only: bool = False,
    ):
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.rsi_period = rsi_period
        self.rsi_long_max = rsi_long_max
        self.rsi_short_min = rsi_short_min
        self.rsi_exit_long = rsi_exit_long
        self.rsi_exit_short = rsi_exit_short
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.session_filter = session_filter
        self.long_only = long_only

    @property
    def name(self) -> str:
        return (
            f"ForexMaster(sma={self.sma_fast}/{self.sma_slow}"
            f",rsi={self.rsi_period}"
            f",macd={self.macd_fast}/{self.macd_slow}/{self.macd_signal}"
            f",bb={self.bb_period},atr={self.atr_period}x{self.atr_mult})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        # --- Trend: SMA 50 / 200 ---
        sma_fast = close.rolling(self.sma_fast).mean()
        sma_slow = close.rolling(self.sma_slow).mean()
        uptrend   = sma_fast > sma_slow
        downtrend = sma_fast < sma_slow

        # --- RSI ---
        rsi = _rsi(close, self.rsi_period)

        # --- MACD ---
        ema_f       = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_s       = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line   = ema_f - ema_s
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        macd_bullish = macd_line > signal_line
        macd_bearish = macd_line < signal_line

        # --- Bollinger Bands ---
        bb_mid   = close.rolling(self.bb_period).mean()
        bb_std_s = close.rolling(self.bb_period).std()
        near_lower = close <= bb_mid
        near_upper = close >= bb_mid

        # --- ATR ---
        atr = _atr(high, low, close, self.atr_period)

        # --- Session filter (London 07-12 UTC, New York 13-17 UTC) ---
        if self.session_filter and "time" in df.columns:
            times = pd.to_datetime(df["time"])
            median_interval_hours = times.diff().dt.total_seconds().median() / 3600
            if median_interval_hours >= 20:
                in_session = pd.Series(True, index=df.index)
            else:
                hour = times.dt.hour
                in_session = ((hour >= 7) & (hour < 12)) | ((hour >= 13) & (hour < 17))
        else:
            in_session = pd.Series(True, index=df.index)

        warmup = max(self.sma_slow, self.macd_slow + self.macd_signal,
                     self.bb_period, self.atr_period)

        signals   = pd.Series(0, index=df.index)
        position  = 0
        stop_loss = None   # price level that triggers exit

        for i in range(warmup, len(df)):
            c = close.iloc[i]

            # --- ATR stop hit ---
            if position == 1 and stop_loss is not None and c <= stop_loss:
                position  = 0
                stop_loss = None
            elif position == -1 and stop_loss is not None and c >= stop_loss:
                position  = 0
                stop_loss = None

            # --- Signal-based exits ---
            if position == 1 and (downtrend.iloc[i] or rsi.iloc[i] > self.rsi_exit_long):
                position  = 0
                stop_loss = None
            elif position == -1 and (uptrend.iloc[i] or rsi.iloc[i] < self.rsi_exit_short):
                position  = 0
                stop_loss = None

            # --- Entries ---
            if position == 0:
                atr_val = atr.iloc[i]
                if (uptrend.iloc[i]
                        and rsi.iloc[i] < self.rsi_long_max
                        and macd_bullish.iloc[i]
                        and near_lower.iloc[i]
                        and in_session.iloc[i]
                        and not np.isnan(atr_val)):
                    position  = 1
                    stop_loss = c - self.atr_mult * atr_val

                elif (not self.long_only
                        and downtrend.iloc[i]
                        and rsi.iloc[i] > self.rsi_short_min
                        and macd_bearish.iloc[i]
                        and near_upper.iloc[i]
                        and in_session.iloc[i]
                        and not np.isnan(atr_val)):
                    position  = -1
                    stop_loss = c + self.atr_mult * atr_val

            signals.iloc[i] = position

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
