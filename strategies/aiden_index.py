"""AiDEN Index Strategy — full OS confluence stack for US100 / US30.

Cadre roles in this module:
  Sage   (MEM-001) — architecture: multi-TF H4 bias gate + H1 FVG/OB entry
  Quant  (MEM-002) — parameters tuned for US100/US30 FTMO drawdown compliance
  Builder(MEM-003) — implementation
  Scout  (MEM-004) — instrument pivot: US100 primary, US30 secondary (gold deprioritised)

OS Confluence scoring (max 7 points per setup):
  HTF bias bullish (H4 HH+HL structure) .... +2  [hard gate — 0 skips trade entirely]
  Price in discount zone (<50% H4 swing)  .. +1
  Liquidity swept before FVG ............... +1
  OB + FVG zone aligned (Model 3) .......... +2  [preferred setup]
  FVG only, no OB overlap (Model 1) ........ +1  [minimum acceptable]
  Session confirmed (NY hours) ............. +1

Trade fires only if score >= min_score (default 4).
HTF bias gate: if h4_bias != bullish, setup is not considered regardless of score.

Entry models (from trading-os/strategy/entry_models.md):
  Model 1 — FVG retest in HTF discount zone
  Model 3 — OB + FVG aligned zone (higher conviction, tighter stop off OB low)

Session defaults for indices: 12:00–21:00 UTC (pre-market + full NY session).
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


def _resample_h4(df: pd.DataFrame) -> pd.DataFrame:
    """Resample H1 OHLCV to H4, aligned to 4-hour UTC boundaries."""
    times = pd.to_datetime(df["time"])
    tmp = df[["open", "high", "low", "close"]].copy()
    tmp.index = times
    h4 = tmp.resample("4h", closed="left", label="left").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
    }).dropna()
    h4 = h4.reset_index().rename(columns={"index": "time"})
    h4["time"] = pd.to_datetime(h4["time"])
    return h4


def _compute_h4_bias(h4: pd.DataFrame, lookback: int) -> pd.Series:
    """
    For each H4 bar, classify bias: 1=bullish, -1=bearish, 0=ranging.
    Bullish = HH+HL: second half of lookback window has higher swing highs AND
    higher swing lows than the first half.
    """
    bias = pd.Series(0, index=h4.index, dtype=int)
    highs = h4["high"].values
    lows  = h4["low"].values
    mid   = lookback // 2

    for i in range(lookback, len(h4)):
        w_hi = highs[i - lookback:i]
        w_lo = lows[i  - lookback:i]
        h_first  = w_hi[:mid].max()
        h_second = w_hi[mid:].max()
        l_first  = w_lo[:mid].min()
        l_second = w_lo[mid:].min()
        if h_second > h_first and l_second > l_first:
            bias.iloc[i] = 1
        elif h_second < h_first and l_second < l_first:
            bias.iloc[i] = -1

    return bias


def _find_bullish_ob(
    open_: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    start_i: int,
    lookback: int,
) -> tuple[float | None, float | None]:
    """Last bearish candle before start_i = bullish order block zone.

    Returns (ob_low, ob_high) using candle body, or (None, None).
    """
    end_i = max(0, start_i - lookback)
    for j in range(start_i, end_i, -1):
        if close.iloc[j] < open_.iloc[j]:
            return float(min(open_.iloc[j], close.iloc[j])), float(max(open_.iloc[j], close.iloc[j]))
    return None, None


def _liquidity_swept(low: pd.Series, i: int, lookback: int) -> bool:
    """True if any candle in the last 5 bars wicked below the prior N-bar swing low."""
    if i < lookback + 2:
        return False
    swing_low = low.iloc[i - lookback:i - 1].min()
    for j in range(max(0, i - 5), i):
        if low.iloc[j] < swing_low:
            return True
    return False


# ── Strategy ──────────────────────────────────────────────────────────────────

class AiDENIndexStrategy(Strategy):
    """Multi-timeframe FVG + OB confluence strategy for US indices.

    Implements AiDEN's three-gate execution model at the strategy level:
    signal only fires when the full OS confluence stack scores >= min_score.
    """

    def __init__(
        self,
        # Confluence gate
        min_score: int          = 4,    # minimum points to trade (max 7)
        # H4 bias
        htf_lookback: int       = 20,   # H4 bars for HH/HL detection
        h4_swing_lookback: int  = 40,   # H4 bars for discount zone range
        discount_pct: float     = 0.5,  # price must be below this fraction of H4 swing
        # FVG
        min_fvg_atr: float      = 0.15,
        max_fvg_wait: int       = 40,
        max_entry_wait: int     = 8,
        max_active_fvgs: int    = 3,
        # OB
        ob_lookback: int        = 20,
        # Liquidity sweep
        liq_lookback: int       = 10,
        # Risk
        rr_target: float        = 2.5,
        atr_period: int         = 14,
        atr_stop_buffer: float  = 0.3,
        # Session (UTC hours, indices default = NY session)
        session_start: int      = 12,
        session_end: int        = 21,
        long_only: bool         = True,
    ):
        self.min_score         = min_score
        self.htf_lookback      = htf_lookback
        self.h4_swing_lookback = h4_swing_lookback
        self.discount_pct      = discount_pct
        self.min_fvg_atr       = min_fvg_atr
        self.max_fvg_wait      = max_fvg_wait
        self.max_entry_wait    = max_entry_wait
        self.max_active_fvgs   = max_active_fvgs
        self.ob_lookback       = ob_lookback
        self.liq_lookback      = liq_lookback
        self.rr_target         = rr_target
        self.atr_period        = atr_period
        self.atr_stop_buffer   = atr_stop_buffer
        self.session_start     = session_start
        self.session_end       = session_end
        self.long_only         = long_only

    @property
    def name(self) -> str:
        return (
            f"AiDEN-Index("
            f"score>={self.min_score}"
            f",fvg={self.min_fvg_atr}atr"
            f",rr={self.rr_target}"
            f",sess={self.session_start}-{self.session_end}UTC)"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        open_  = df["open"].astype(float)
        times  = pd.to_datetime(df["time"])
        hours  = times.dt.hour

        atr = _atr(high, low, close, self.atr_period)

        # Build H4 bias and swing range lookup
        h4       = _resample_h4(df)
        h4_bias  = _compute_h4_bias(h4, self.htf_lookback)
        h4["bias"] = h4_bias.values
        h4_times   = h4["time"]

        def _bias_at(bar_time) -> int:
            idx = h4_times.searchsorted(bar_time, side="right") - 1
            return int(h4["bias"].iloc[idx]) if idx >= 0 else 0

        def _swing_range_at(bar_time) -> tuple[float, float]:
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

        warmup = max(self.atr_period + 3, self.ob_lookback, self.liq_lookback)

        for i in range(warmup, len(df)):
            cv       = float(close.iloc[i])
            lv       = float(low.iloc[i])
            hv       = float(high.iloc[i])
            atr_val  = float(atr.iloc[i])
            bar_time = times.iloc[i]

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

            if position == 0 and i >= warmup + 2:
                # ── 2. HTF bias gate ──────────────────────────────────────
                htf_bias = _bias_at(bar_time)
                if htf_bias != 1:
                    # Not in bullish H4 structure — no new setups
                    self._process_active_fvgs(
                        active_fvgs, cv, lv, hv, atr_val, i,
                        signals, position, stop_loss, take_profit
                    )
                    signals.iloc[i]     = position
                    self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                    continue

                # ── 3. Detect bullish FVG ────────────────────────────────
                h2      = float(high.iloc[i - 2])
                bull_gap = lv - h2
                min_gap  = self.min_fvg_atr * atr_val

                if bull_gap >= min_gap:
                    fvg_lo = h2
                    fvg_hi = lv

                    score = 2  # HTF bias confirmed (+2)

                    # Discount zone
                    swing_hi, swing_lo = _swing_range_at(bar_time)
                    if not np.isnan(swing_hi) and swing_hi > swing_lo:
                        mid_zone = swing_lo + (swing_hi - swing_lo) * self.discount_pct
                        if cv < mid_zone:
                            score += 1

                    # Liquidity sweep
                    if _liquidity_swept(low, i, self.liq_lookback):
                        score += 1

                    # OB detection + Model 3 check
                    ob_lo, ob_hi = _find_bullish_ob(open_, close, high, low, i - 2, self.ob_lookback)
                    if ob_lo is not None:
                        overlap = min(ob_hi, fvg_hi) - max(ob_lo, fvg_lo)
                        if overlap > 0:
                            score += 2  # Model 3: OB + FVG aligned
                        # OB found but not overlapping — no bonus (FVG alone = +1 below)
                    else:
                        score += 1  # Model 1: FVG only

                    # Session
                    if self.session_start <= int(hours.iloc[i]) < self.session_end:
                        score += 1

                    if score >= self.min_score:
                        active_fvgs.append({
                            "dir":      "bull",
                            "fvg_lo":   fvg_lo,
                            "fvg_hi":   fvg_hi,
                            "score":    score,
                            "formed":   i,
                            "tested":   False,
                            "test_bar": None,
                            "ob_lo":    ob_lo,
                            "ob_hi":    ob_hi,
                        })

                    # Keep highest-scoring FVGs if over cap
                    if len(active_fvgs) > self.max_active_fvgs:
                        active_fvgs.sort(key=lambda x: x["score"], reverse=True)
                        active_fvgs = active_fvgs[:self.max_active_fvgs]

                # ── 4. Process queued FVGs ───────────────────────────────
                to_remove = []
                for fvg in active_fvgs:
                    fvg_lo = fvg["fvg_lo"]
                    fvg_hi = fvg["fvg_hi"]

                    if cv < fvg_lo:
                        to_remove.append(fvg)
                        continue

                    if not fvg["tested"] and (i - fvg["formed"]) > self.max_fvg_wait:
                        to_remove.append(fvg)
                        continue

                    if not fvg["tested"] and lv <= fvg_hi:
                        fvg["tested"]   = True
                        fvg["test_bar"] = i

                    if fvg["tested"]:
                        if (i - fvg["test_bar"]) > self.max_entry_wait:
                            to_remove.append(fvg)
                        elif cv > fvg_hi and position == 0:
                            # Stop off OB low when available (tighter — Model 3)
                            stop_anchor = fvg["ob_lo"] if fvg["ob_lo"] is not None else fvg_lo
                            sl   = stop_anchor - self.atr_stop_buffer * atr_val
                            dist = cv - sl
                            if dist > 0:
                                position    = 1
                                stop_loss   = sl
                                take_profit = cv + self.rr_target * dist
                            to_remove.append(fvg)

                for fvg in to_remove:
                    if fvg in active_fvgs:
                        active_fvgs.remove(fvg)

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")

        return signals

    def _process_active_fvgs(self, active_fvgs, cv, lv, hv, atr_val, i,
                              signals, position, stop_loss, take_profit):
        """Expire/invalidate queued FVGs when HTF bias is not bullish."""
        to_remove = []
        for fvg in active_fvgs:
            if cv < fvg["fvg_lo"]:
                to_remove.append(fvg)
        for fvg in to_remove:
            if fvg in active_fvgs:
                active_fvgs.remove(fvg)
