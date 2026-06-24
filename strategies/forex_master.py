"""Forex Master Strategy.

Combines all indicators with forex-specific knowledge:

  Trend filter  : SMA 50 vs SMA 200 — only trade with the trend
  Entry timing  : RSI pullback + MACD momentum + Bollinger Band position
  Session filter: London (07:00-12:00 UTC) and New York (13:00-17:00 UTC) only
                  — highest liquidity, tightest spreads, strongest moves

Long  : SMA50 > SMA200 AND RSI < 55 AND MACD bullish AND price below BB midline
Short : SMA50 < SMA200 AND RSI > 45 AND MACD bearish AND price above BB midline
Exit  : trend reversal (SMA cross) OR RSI extreme against position
"""
import pandas as pd

from strategies.base import Strategy


class ForexMasterStrategy(Strategy):
    def __init__(
        self,
        sma_fast: int = 50,
        sma_slow: int = 200,
        rsi_period: int = 14,
        rsi_long_max: float = 55.0,   # RSI must be below this to go long (pullback)
        rsi_short_min: float = 45.0,  # RSI must be above this to go short (rally)
        rsi_exit_long: float = 70.0,  # exit long when RSI overbought
        rsi_exit_short: float = 30.0, # exit short when RSI oversold
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        session_filter: bool = True,  # only trade London + NY sessions
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
        self.session_filter = session_filter
        self.long_only = long_only

    @property
    def name(self) -> str:
        return (
            f"ForexMaster(sma={self.sma_fast}/{self.sma_slow}"
            f",rsi={self.rsi_period}"
            f",macd={self.macd_fast}/{self.macd_slow}/{self.macd_signal}"
            f",bb={self.bb_period})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)

        # --- Trend: SMA 50 / 200 ---
        sma_fast = close.rolling(self.sma_fast).mean()
        sma_slow = close.rolling(self.sma_slow).mean()
        uptrend = sma_fast > sma_slow
        downtrend = sma_fast < sma_slow

        # --- RSI ---
        rsi = _rsi(close, self.rsi_period)

        # --- MACD ---
        ema_f = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_f - ema_s
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        macd_bullish = macd_line > signal_line
        macd_bearish = macd_line < signal_line

        # --- Bollinger Bands ---
        bb_mid = close.rolling(self.bb_period).mean()
        bb_std = close.rolling(self.bb_period).std()
        bb_upper = bb_mid + self.bb_std * bb_std
        bb_lower = bb_mid - self.bb_std * bb_std
        near_lower = close <= bb_mid          # buying on dip (below midline)
        near_upper = close >= bb_mid          # selling on rally (above midline)

        # --- Session filter (London 07-12 UTC, New York 13-17 UTC) ---
        if self.session_filter and "time" in df.columns:
            hour = pd.to_datetime(df["time"]).dt.hour
            in_session = ((hour >= 7) & (hour < 12)) | ((hour >= 13) & (hour < 17))
        else:
            in_session = pd.Series(True, index=df.index)

        warmup = max(self.sma_slow, self.macd_slow + self.macd_signal, self.bb_period)

        signals = pd.Series(0, index=df.index)
        position = 0

        for i in range(warmup, len(df)):
            long_entry = (
                uptrend.iloc[i]
                and rsi.iloc[i] < self.rsi_long_max
                and macd_bullish.iloc[i]
                and near_lower.iloc[i]
                and in_session.iloc[i]
            )
            short_entry = (
                not self.long_only
                and downtrend.iloc[i]
                and rsi.iloc[i] > self.rsi_short_min
                and macd_bearish.iloc[i]
                and near_upper.iloc[i]
                and in_session.iloc[i]
            )

            # Exit: trend flips OR RSI extreme against position
            long_exit = position == 1 and (
                downtrend.iloc[i] or rsi.iloc[i] > self.rsi_exit_long
            )
            short_exit = position == -1 and (
                uptrend.iloc[i] or rsi.iloc[i] < self.rsi_exit_short
            )

            if long_exit or short_exit:
                position = 0
            if long_entry and position == 0:
                position = 1
            elif short_entry and position == 0:
                position = -1

            signals.iloc[i] = position

        return signals


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))
