"""AiDEN Index Strategy v2 — bidirectional FVG + OB confluence, all sessions.

Cadre:
  Sage   (MEM-001) — bidirectional gate: H4 EMA bullish = longs only, bearish = shorts only
  Quant  (MEM-002) — dynamic RR: Model 3 + trend strength extend target to 3.5-5.0R
  Builder(MEM-003) — implementation, shorts mirror, regime scoring
  Scout  (MEM-004) — 7-instrument universe confirmed

OS Confluence scoring (max 10 per setup):
  HTF bias confirmed (H4 EMA direction) ......... +2  [hard gate]
  Price in discount (long) / premium (short) .... +1
  Liquidity swept (lows for longs / highs for shorts) +1
  OB + FVG aligned — Model 3 ................... +2
  FVG only — Model 1 ........................... +1
  Session (in active window) ................... +1
  Session prime window (NY first hour 13-15 UTC) +1  [stacks with session]
  RSI pullback zone ............................ +1
  Trend regime (H4 EMA strongly trending) ...... +1

Dynamic RR:
  Base: rr_target (default 2.5)
  Model 3 trade: + rr_model3_bonus (default +0.5)
  Strong H4 trend: + rr_trend_bonus (default +0.5)
  Cap: rr_max (default 5.0)

Parameter units:
  htf_lookback, h4_swing_lookback → H4 BARS (TF-independent after resample)
  Everything else → input-TF bars (scale ×4 for M15 vs H1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _resample_h4(df: pd.DataFrame) -> pd.DataFrame:
    times = pd.to_datetime(df["time"])
    tmp   = df[["open", "high", "low", "close"]].copy()
    tmp.index = times
    h4 = tmp.resample("4h", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
    h4 = h4.reset_index().rename(columns={"index": "time"})
    h4["time"] = pd.to_datetime(h4["time"])
    return h4


def _compute_h4_bias_ema(
    h4: pd.DataFrame, fast_n: int, slow_n: int
) -> tuple[pd.Series, pd.Series]:
    """Returns (bias_series, ema_spread_series).

    bias:   1=bullish, -1=bearish, 0=neutral
    spread: (fast-slow)/slow as fraction — magnitude = trend strength
    """
    close  = h4["close"]
    fast   = close.ewm(span=fast_n, adjust=False).mean()
    slow   = close.ewm(span=slow_n, adjust=False).mean()
    spread = (fast - slow) / slow.replace(0, np.nan)

    bias = pd.Series(0, index=h4.index, dtype=int)
    bias[(fast > slow) & (close > fast)] = 1
    bias[(fast < slow) & (close < fast)] = -1
    return bias, spread


def _compute_h4_bias_swing(h4: pd.DataFrame, lookback: int) -> tuple[pd.Series, pd.Series]:
    bias  = pd.Series(0, index=h4.index, dtype=int)
    highs = h4["high"].values
    lows  = h4["low"].values
    mid   = lookback // 2

    for i in range(lookback, len(h4)):
        w_hi = highs[i - lookback:i]
        w_lo = lows[i  - lookback:i]
        if w_hi[mid:].max() > w_hi[:mid].max() and w_lo[mid:].min() > w_lo[:mid].min():
            bias.iloc[i] = 1
        elif w_hi[mid:].max() < w_hi[:mid].max() and w_lo[mid:].min() < w_lo[:mid].min():
            bias.iloc[i] = -1

    spread = pd.Series(0.0, index=h4.index)  # swing method has no spread metric
    return bias, spread


def _find_bullish_ob(open_, close, high, low, start_i, lookback):
    """Last bearish candle before start_i (body zone)."""
    for j in range(start_i, max(0, start_i - lookback), -1):
        if close.iloc[j] < open_.iloc[j]:
            return float(min(open_.iloc[j], close.iloc[j])), float(max(open_.iloc[j], close.iloc[j]))
    return None, None


def _find_bearish_ob(open_, close, high, low, start_i, lookback):
    """Last bullish candle before start_i (body zone) — bearish OB for short setups."""
    for j in range(start_i, max(0, start_i - lookback), -1):
        if close.iloc[j] > open_.iloc[j]:
            return float(min(open_.iloc[j], close.iloc[j])), float(max(open_.iloc[j], close.iloc[j]))
    return None, None


def _liq_swept_low(low: pd.Series, i: int, lookback: int) -> bool:
    """Wick below prior N-bar swing low — used for LONG setups."""
    if i < lookback + 2:
        return False
    swing_low = low.iloc[i - lookback:i - 1].min()
    return any(low.iloc[j] < swing_low for j in range(max(0, i - 5), i))


def _liq_swept_high(high: pd.Series, i: int, lookback: int) -> bool:
    """Wick above prior N-bar swing high — used for SHORT setups."""
    if i < lookback + 2:
        return False
    swing_high = high.iloc[i - lookback:i - 1].max()
    return any(high.iloc[j] > swing_high for j in range(max(0, i - 5), i))


# ── Strategy ──────────────────────────────────────────────────────────────────

class AiDENIndexStrategy(Strategy):

    def __init__(
        self,
        # Confluence gate
        min_score: int           = 4,
        # H4 bias — in H4 BARS
        htf_lookback: int        = 20,
        h4_swing_lookback: int   = 40,
        h4_bias_method: str      = "ema",
        discount_pct: float      = 0.5,
        # FVG — in input-TF bars
        min_fvg_atr: float       = 0.10,
        max_fvg_wait: int        = 40,
        max_entry_wait: int      = 8,
        max_active_fvgs: int     = 3,
        # OB
        ob_lookback: int         = 20,
        # Liquidity sweep
        liq_lookback: int        = 10,
        # Risk / RR
        rr_target: float         = 2.5,
        rr_model3_bonus: float   = 0.5,   # extra R when OB+FVG aligned
        rr_trend_bonus: float    = 0.5,   # extra R in strong trend regime
        rr_trend_threshold: float= 0.003, # H4 EMA spread > 0.3% = strong trend
        rr_max: float            = 5.0,
        atr_period: int          = 14,
        atr_stop_buffer: float   = 0.3,
        # RSI
        rsi_period: int          = 14,
        rsi_long_lo: float       = 25.0,  # RSI zone for LONG entries
        rsi_long_hi: float       = 55.0,
        rsi_short_lo: float      = 45.0,  # RSI zone for SHORT entries
        rsi_short_hi: float      = 75.0,
        use_rsi: bool            = True,
        # Session
        session_start: int       = 12,
        session_end: int         = 21,
        session_prime_start: int = 13,    # NY open first hour bonus window
        session_prime_end: int   = 15,
        # Direction
        long_only: bool          = False,  # False = both longs and shorts
    ):
        self.min_score            = min_score
        self.htf_lookback         = htf_lookback
        self.h4_swing_lookback    = h4_swing_lookback
        self.h4_bias_method       = h4_bias_method
        self.discount_pct         = discount_pct
        self.min_fvg_atr          = min_fvg_atr
        self.max_fvg_wait         = max_fvg_wait
        self.max_entry_wait       = max_entry_wait
        self.max_active_fvgs      = max_active_fvgs
        self.ob_lookback          = ob_lookback
        self.liq_lookback         = liq_lookback
        self.rr_target            = rr_target
        self.rr_model3_bonus      = rr_model3_bonus
        self.rr_trend_bonus       = rr_trend_bonus
        self.rr_trend_threshold   = rr_trend_threshold
        self.rr_max               = rr_max
        self.atr_period           = atr_period
        self.atr_stop_buffer      = atr_stop_buffer
        self.rsi_period           = rsi_period
        self.rsi_long_lo          = rsi_long_lo
        self.rsi_long_hi          = rsi_long_hi
        self.rsi_short_lo         = rsi_short_lo
        self.rsi_short_hi         = rsi_short_hi
        self.use_rsi              = use_rsi
        self.session_start        = session_start
        self.session_end          = session_end
        self.session_prime_start  = session_prime_start
        self.session_prime_end    = session_prime_end
        self.long_only            = long_only

    @property
    def name(self) -> str:
        direction = "long" if self.long_only else "bi"
        return (
            f"AiDEN-v2({direction}"
            f",s>={self.min_score}"
            f",{self.h4_bias_method}"
            f",rr={self.rr_target}+dyn"
            f",sess={self.session_start}-{self.session_end})"
        )

    def _dynamic_rr(self, is_model3: bool, trend_strength: float) -> float:
        rr = self.rr_target
        if is_model3:
            rr += self.rr_model3_bonus
        if abs(trend_strength) > self.rr_trend_threshold:
            rr += self.rr_trend_bonus
        return min(rr, self.rr_max)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        open_  = df["open"].astype(float)
        times  = pd.to_datetime(df["time"])
        hours  = times.dt.hour

        atr_s = _atr(high, low, close, self.atr_period)
        rsi_s = _rsi(close, self.rsi_period) if self.use_rsi else None

        h4 = _resample_h4(df)
        fast_n = max(self.htf_lookback // 2, 5)

        if self.h4_bias_method == "ema":
            h4_bias, h4_spread = _compute_h4_bias_ema(h4, fast_n, self.htf_lookback)
        else:
            h4_bias, h4_spread = _compute_h4_bias_swing(h4, self.htf_lookback)

        h4["bias"]   = h4_bias.values
        h4["spread"] = h4_spread.values
        h4_times     = h4["time"]

        def _h4_at(bar_time):
            idx = h4_times.searchsorted(bar_time, side="right") - 1
            if idx < 0:
                return 0, 0.0
            return int(h4["bias"].iloc[idx]), float(h4["spread"].iloc[idx])

        def _swing_range_at(bar_time):
            idx = h4_times.searchsorted(bar_time, side="right") - 1
            if idx < self.h4_swing_lookback:
                return float("nan"), float("nan")
            w = h4.iloc[idx - self.h4_swing_lookback:idx]
            return float(w["high"].max()), float(w["low"].min())

        signals     = pd.Series(0, index=df.index)
        self._stops = pd.Series(float("nan"), index=df.index)

        position    = 0
        stop_loss   = None
        take_profit = None
        active_fvgs: list[dict] = []

        warmup = max(self.atr_period + 3, self.ob_lookback, self.liq_lookback,
                     self.rsi_period + 2 if self.use_rsi else 0)

        for i in range(warmup, len(df)):
            cv       = float(close.iloc[i])
            lv       = float(low.iloc[i])
            hv       = float(high.iloc[i])
            atr_val  = float(atr_s.iloc[i])
            bar_time = times.iloc[i]
            hour     = int(hours.iloc[i])

            if np.isnan(atr_val):
                signals.iloc[i]     = position
                self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                continue

            # ── 1. Manage open position ───────────────────────────────────
            if position == 1:
                if stop_loss is not None and lv <= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and hv >= take_profit:
                    position = 0; stop_loss = take_profit = None
            elif position == -1:
                if stop_loss is not None and hv >= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and lv <= take_profit:
                    position = 0; stop_loss = take_profit = None

            if position == 0 and i >= warmup + 2:
                htf_bias, trend_strength = _h4_at(bar_time)
                swing_hi, swing_lo       = _swing_range_at(bar_time)
                in_session   = self.session_start <= hour < self.session_end
                in_prime     = self.session_prime_start <= hour < self.session_prime_end
                rsi_val      = float(rsi_s.iloc[i]) if rsi_s is not None else float("nan")
                strongly_trending = abs(trend_strength) > self.rr_trend_threshold

                # Remove invalidated FVGs
                if htf_bias == 0:
                    self._expire_fvgs_neutral(active_fvgs, cv)
                    signals.iloc[i]     = position
                    self._stops.iloc[i] = float("nan")
                    continue

                # ── 2. Detect new FVGs ───────────────────────────────────
                h2  = float(high.iloc[i - 2])
                l2  = float(low.iloc[i - 2])

                # LONG setup — bullish FVG
                if htf_bias == 1:
                    bull_gap = lv - h2
                    if bull_gap >= self.min_fvg_atr * atr_val:
                        score = 2  # HTF bias +2

                        if not np.isnan(swing_hi) and swing_hi > swing_lo:
                            mid = swing_lo + (swing_hi - swing_lo) * self.discount_pct
                            if cv < mid:
                                score += 1  # discount zone

                        if _liq_swept_low(low, i, self.liq_lookback):
                            score += 1

                        ob_lo, ob_hi = _find_bullish_ob(open_, close, high, low, i - 2, self.ob_lookback)
                        if ob_lo is not None and min(ob_hi, lv) - max(ob_lo, h2) > 0:
                            score += 2; is_model3 = True
                        else:
                            score += 1; is_model3 = False

                        if in_session:
                            score += 1
                        if in_prime:
                            score += 1  # stacks — prime window bonus

                        if self.use_rsi and not np.isnan(rsi_val):
                            if self.rsi_long_lo <= rsi_val <= self.rsi_long_hi:
                                score += 1

                        if strongly_trending:
                            score += 1  # regime bonus

                        if score >= self.min_score:
                            active_fvgs.append({
                                "dir":      "bull",
                                "fvg_lo":   h2,
                                "fvg_hi":   lv,
                                "ob_lo":    ob_lo,
                                "ob_hi":    ob_hi,
                                "score":    score,
                                "formed":   i,
                                "tested":   False,
                                "test_bar": None,
                                "model3":   is_model3,
                                "trend_s":  trend_strength,
                            })

                # SHORT setup — bearish FVG
                if htf_bias == -1 and not self.long_only:
                    bear_gap = l2 - hv
                    if bear_gap >= self.min_fvg_atr * atr_val:
                        score = 2  # HTF bias +2

                        if not np.isnan(swing_hi) and swing_hi > swing_lo:
                            mid = swing_lo + (swing_hi - swing_lo) * self.discount_pct
                            if cv > mid:
                                score += 1  # premium zone for short

                        if _liq_swept_high(high, i, self.liq_lookback):
                            score += 1

                        ob_lo, ob_hi = _find_bearish_ob(open_, close, high, low, i - 2, self.ob_lookback)
                        if ob_lo is not None and min(ob_hi, l2) - max(ob_lo, hv) > 0:
                            score += 2; is_model3 = True
                        else:
                            score += 1; is_model3 = False

                        if in_session:
                            score += 1
                        if in_prime:
                            score += 1

                        if self.use_rsi and not np.isnan(rsi_val):
                            if self.rsi_short_lo <= rsi_val <= self.rsi_short_hi:
                                score += 1

                        if strongly_trending:
                            score += 1

                        if score >= self.min_score:
                            active_fvgs.append({
                                "dir":      "bear",
                                "fvg_lo":   hv,   # bottom of bearish FVG gap
                                "fvg_hi":   l2,   # top of bearish FVG gap
                                "ob_lo":    ob_lo,
                                "ob_hi":    ob_hi,
                                "score":    score,
                                "formed":   i,
                                "tested":   False,
                                "test_bar": None,
                                "model3":   is_model3,
                                "trend_s":  trend_strength,
                            })

                # Trim FVG queue
                if len(active_fvgs) > self.max_active_fvgs * 2:
                    active_fvgs.sort(key=lambda x: x["score"], reverse=True)
                    active_fvgs = active_fvgs[:self.max_active_fvgs * 2]

                # ── 3. Process queued FVGs ───────────────────────────────
                to_remove = []
                for fvg in active_fvgs:
                    fvg_lo = fvg["fvg_lo"]
                    fvg_hi = fvg["fvg_hi"]

                    if (i - fvg["formed"]) > self.max_fvg_wait:
                        to_remove.append(fvg); continue

                    if fvg["dir"] == "bull":
                        if cv < fvg_lo:
                            to_remove.append(fvg); continue
                        if not fvg["tested"] and lv <= fvg_hi:
                            fvg["tested"] = True; fvg["test_bar"] = i
                        if fvg["tested"]:
                            if (i - fvg["test_bar"]) > self.max_entry_wait:
                                to_remove.append(fvg)
                            elif cv > fvg_hi and position == 0:
                                stop_anchor = fvg["ob_lo"] if fvg["ob_lo"] is not None else fvg_lo
                                sl   = stop_anchor - self.atr_stop_buffer * atr_val
                                dist = cv - sl
                                if dist > 0:
                                    rr       = self._dynamic_rr(fvg["model3"], fvg["trend_s"])
                                    position    = 1
                                    stop_loss   = sl
                                    take_profit = cv + rr * dist
                                to_remove.append(fvg)

                    else:  # bear
                        if cv > fvg_hi:
                            to_remove.append(fvg); continue
                        if not fvg["tested"] and hv >= fvg_lo:
                            fvg["tested"] = True; fvg["test_bar"] = i
                        if fvg["tested"]:
                            if (i - fvg["test_bar"]) > self.max_entry_wait:
                                to_remove.append(fvg)
                            elif cv < fvg_lo and position == 0:
                                stop_anchor = fvg["ob_hi"] if fvg["ob_hi"] is not None else fvg_hi
                                sl   = stop_anchor + self.atr_stop_buffer * atr_val
                                dist = sl - cv
                                if dist > 0:
                                    rr       = self._dynamic_rr(fvg["model3"], fvg["trend_s"])
                                    position    = -1
                                    stop_loss   = sl
                                    take_profit = cv - rr * dist
                                to_remove.append(fvg)

                for fvg in to_remove:
                    if fvg in active_fvgs:
                        active_fvgs.remove(fvg)

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")

        return signals

    def _expire_fvgs_neutral(self, active_fvgs: list, cv: float) -> None:
        to_remove = [
            f for f in active_fvgs
            if (f["dir"] == "bull" and cv < f["fvg_lo"]) or
               (f["dir"] == "bear" and cv > f["fvg_hi"])
        ]
        for f in to_remove:
            active_fvgs.remove(f)
