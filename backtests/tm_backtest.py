"""Event-driven backtest with TradeManager active.

The standard Backtest engine is vectorised (signals → bar returns).
TradeManager requires bar-by-bar state: intra-bar SL/TP hit detection,
partial closes, dynamic SL moves, and TP extensions.

This module provides an event-driven simulation loop that replicates
live execution order exactly:
  1. Check SL hit  (intra-bar, exits at SL price)
  2. Check TP hit  (intra-bar, exits at TP price)
  3. T1 partial close (at +t1_r, moves SL to BE)
  4. Trail stop    (be_r → BE, lock_r → +1R)
  5. TradeManager  (adaptive: EXIT / TIGHTEN_SL / EXTEND_TP / PARTIAL_CLOSE)
  6. New entry     (if flat and signal fires)

P&L uses fixed-fractional sizing: each entry risks risk_pct of current equity.
Each trade logs pnl_pct (total equity fraction change, including partials).

Usage:
    python -m backtests.tm_backtest XAUUSD US100.cash
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.trade_manager import TradeManager, PositionState, ActionType
from strategies.aiden_index import AiDENIndexStrategy
from backtests.metrics import calculate_metrics
from backtests.run_multi_instrument import (
    TRAIL_CONFIGS, BIDIRECTIONAL, M15_PARAMS, H1_PARAMS, OPTIMISED_PARAMS,
    INSTRUMENTS, _size_mult_from_score,
)

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class TMBacktestResult:
    symbol:        str
    equity:        pd.Series
    trades:        pd.DataFrame
    metrics:       dict
    n_tm_exits:    int = 0
    n_tm_tighten:  int = 0
    n_tm_extend:   int = 0
    n_tm_partial:  int = 0
    n_tm_wait:     int = 0
    n_tm_runner:   int = 0
    n_t1:          int = 0

    def print_summary(self) -> None:
        m = self.metrics
        print(
            f"  {self.symbol:<16} "
            f"Return: {m['total_return_pct']:>+7.2f}%  "
            f"MaxDD: {m['max_drawdown_pct']:>7.2f}%  "
            f"Sharpe: {m['sharpe_ratio']:>6.3f}  "
            f"Trades: {m['total_trades']:>4}  "
            f"WR: {m.get('win_rate_pct', 0):.1f}%  "
            f"TM[exit={self.n_tm_exits} tight={self.n_tm_tighten} "
            f"ext={self.n_tm_extend} part={self.n_tm_partial} "
            f"wait={self.n_tm_wait} run={self.n_tm_runner}]  "
            f"T1={self.n_t1}"
        )


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate(
    df_m15:          pd.DataFrame,
    df_m5:           Optional[pd.DataFrame],
    strat:           AiDENIndexStrategy,
    trade_manager:   TradeManager,
    symbol:          str   = "UNKNOWN",
    initial_capital: float = 10_000.0,
    risk_pct:        float = 1.0,
    commission:      float = 0.0001,
    daily_halt_pct:  float = 2.5,
) -> TMBacktestResult:
    """Run bar-by-bar event-driven simulation with TradeManager."""

    signals = strat.generate_signals(df_m15)
    stops   = getattr(strat, "_stops",  pd.Series(float("nan"), index=df_m15.index))
    scores  = getattr(strat, "_scores", pd.Series(0,            index=df_m15.index))

    times_m15 = pd.to_datetime(df_m15["time"], utc=True)

    m5_times:   Optional[pd.Series]    = None
    m5_cutoffs: Optional[np.ndarray]   = None   # per M15 bar: last valid M5 idx
    if df_m5 is not None and len(df_m5) > 0:
        df_m5    = df_m5.reset_index(drop=True)
        m5_times = pd.to_datetime(df_m5["time"], utc=True)
        m5_arr   = m5_times.values.astype("int64")   # nanoseconds, for searchsorted
        # For each M15 bar i, cutoff = bar_time + 10min; find last M5 bar <= cutoff.
        m15_cutoff_ns = (
            times_m15.values.astype("int64") + np.int64(10 * 60 * 1_000_000_000)
        )
        m5_cutoffs = np.searchsorted(m5_arr, m15_cutoff_ns, side="right") - 1

    trail_cfg    = TRAIL_CONFIGS.get(symbol, {})
    trail_to_be  = trail_cfg.get("trail_to_be", False)
    trail_be_r   = trail_cfg.get("trail_be_r",  1.0)
    trail_lock_r = trail_cfg.get("trail_lock_r", 2.0)
    t1_r         = getattr(strat, "t1_r",           0.0)
    t1_pct       = getattr(strat, "t1_partial_pct", 0.5)

    # position state
    position      = 0
    entry_price   = 0.0
    initial_sl    = 0.0
    current_sl    = 0.0
    current_tp    = 0.0
    position_frac = 1.0
    t1_hit        = False
    bars_in_trade = 0
    entry_score   = 0
    entry_bar_idx = 0
    trade_pnl_acc = 0.0    # accumulates pnl across partials for final trade record

    equity        = initial_capital
    current_day   = None
    day_start_eq  = initial_capital
    halted_today  = False
    equity_curve: list[float] = [equity]
    trade_log:    list[dict]  = []

    n_tm_exits   = 0
    n_tm_tighten = 0
    n_tm_extend  = 0
    n_tm_partial = 0
    n_tm_wait    = 0
    n_tm_runner  = 0
    n_t1         = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _rdist() -> float:
        return abs(entry_price - initial_sl)

    def _pnl_frac(exit_px: float, frac: float) -> float:
        rd = _rdist()
        if rd < 1e-10:
            return 0.0
        move = (exit_px - entry_price) if position == 1 else (entry_price - exit_px)
        return (move / rd) * (risk_pct / 100.0) * frac * _size_mult_from_score(entry_score)

    def _partial(exit_px: float, frac: float) -> None:
        nonlocal equity, position_frac, trade_pnl_acc
        pnl = _pnl_frac(exit_px, frac)
        equity       *= (1.0 + pnl - commission * frac)
        trade_pnl_acc += pnl
        position_frac  = max(0.0, position_frac - frac)

    def _close(exit_px: float, reason: str, bar_idx: int) -> None:
        nonlocal equity, position, position_frac, t1_hit
        nonlocal bars_in_trade, entry_bar_idx, trade_pnl_acc, entry_score
        pnl = _pnl_frac(exit_px, position_frac)
        equity *= (1.0 + pnl - commission * position_frac)
        trade_log.append({
            "entry_time":  times_m15.iloc[entry_bar_idx],
            "exit_time":   times_m15.iloc[bar_idx],
            "direction":   position,
            "entry_price": entry_price,
            "exit_price":  exit_px,
            "pnl_pct":     trade_pnl_acc + pnl,
            "reason":      reason,
            "score":       entry_score,
        })
        position = 0; position_frac = 1.0; t1_hit = False
        bars_in_trade = 0; trade_pnl_acc = 0.0

    # ── main loop ─────────────────────────────────────────────────────────────

    for i in range(1, len(df_m15)):
        bar  = df_m15.iloc[i]
        bt   = times_m15.iloc[i]
        hi   = float(bar["high"])
        lo   = float(bar["low"])
        cl   = float(bar["close"])
        op   = float(bar["open"])

        day = bt.date()
        if day != current_day:
            current_day  = day
            day_start_eq = equity
            halted_today = False
        if not halted_today and (equity - day_start_eq) / day_start_eq * 100 <= -daily_halt_pct:
            halted_today = True
            if position != 0:
                _close(cl, "DAILY_HALT", i)

        if position != 0:
            bars_in_trade += 1

            sl_hit = (position == 1 and lo <= current_sl) or (position == -1 and hi >= current_sl)
            tp_hit = (position == 1 and hi >= current_tp) or (position == -1 and lo <= current_tp)

            if sl_hit:
                _close(current_sl, "SL", i)
                equity_curve.append(equity)
                continue

            if tp_hit:
                _close(current_tp, "TP", i)
                equity_curve.append(equity)
                continue

            rd = _rdist()
            cur_r = ((cl - entry_price) / rd if position == 1 else (entry_price - cl) / rd) if rd > 0 else 0.0

            # T1 partial
            if t1_r > 0 and not t1_hit and cur_r >= t1_r:
                t1_px = (entry_price + t1_r * rd if position == 1 else entry_price - t1_r * rd)
                _partial(t1_px, t1_pct)
                current_sl = entry_price
                t1_hit = True
                n_t1  += 1

            # Trail stop
            if trail_to_be and rd > 0:
                if position == 1:
                    if cur_r >= trail_lock_r:
                        current_sl = max(current_sl, entry_price + rd)
                    elif cur_r >= trail_be_r:
                        current_sl = max(current_sl, entry_price)
                else:
                    if cur_r >= trail_lock_r:
                        current_sl = min(current_sl, entry_price - rd)
                    elif cur_r >= trail_be_r:
                        current_sl = min(current_sl, entry_price)

            # TradeManager
            df_m5_now = _get_m5_slice_fast(df_m5, m5_cutoffs, i)
            h4_bias   = (strat.current_h4_bias(df_m15.iloc[max(0, i - 120): i + 1])
                         if hasattr(strat, "current_h4_bias") else 0)
            action = trade_manager.evaluate(
                position = PositionState(
                    direction=position, entry_price=entry_price,
                    initial_sl=initial_sl, current_sl=current_sl,
                    current_tp=current_tp, current_price=cl,
                    bars_elapsed=bars_in_trade, t1_hit=t1_hit, h4_bias=h4_bias,
                ),
                df_m15 = df_m15.iloc[max(0, i - 80): i + 1],
                df_m5  = df_m5_now,
            )

            if action.action == ActionType.EXIT:
                _close(cl, "TM_EXIT", i)
                n_tm_exits += 1

            elif action.action == ActionType.PARTIAL_CLOSE and action.close_pct:
                # close_pct = fraction of REMAINING position, not original
                _partial(cl, action.close_pct * position_frac)
                if action.new_sl is not None:
                    if (position == 1 and action.new_sl > current_sl) or \
                       (position == -1 and action.new_sl < current_sl):
                        current_sl = action.new_sl
                n_tm_partial += 1

            elif action.action == ActionType.TIGHTEN_SL and action.new_sl is not None:
                if (position == 1 and action.new_sl > current_sl) or \
                   (position == -1 and action.new_sl < current_sl):
                    current_sl = action.new_sl
                    n_tm_tighten += 1

            elif action.action == ActionType.EXTEND_TP and action.new_tp is not None:
                if (position == 1 and action.new_tp > current_tp) or \
                   (position == -1 and action.new_tp < current_tp):
                    current_tp = action.new_tp
                    n_tm_extend += 1
                # also apply SL tighten if provided alongside TP extension
                if action.new_sl is not None:
                    if (position == 1 and action.new_sl > current_sl) or \
                       (position == -1 and action.new_sl < current_sl):
                        current_sl = action.new_sl

            elif action.action == ActionType.WAIT:
                n_tm_wait += 1  # hold one bar — sweep suspected, wait for confirmation

            elif action.action == ActionType.HOLD_RUNNER:
                # tighten SL to structure; TP stays open — let the winner run
                if action.new_sl is not None:
                    if (position == 1 and action.new_sl > current_sl) or \
                       (position == -1 and action.new_sl < current_sl):
                        current_sl = action.new_sl
                n_tm_runner += 1

        # new entry — edge-triggered only (signals is a persistent indicator; only
        # enter when it rises from 0, so we don't re-enter after a TM early exit)
        if position == 0 and not halted_today:
            sig  = signals.iloc[i - 1]
            sig2 = signals.iloc[i - 2] if i >= 2 else 0.0
            sl   = float(stops.iloc[i - 1]) if not pd.isna(stops.iloc[i - 1]) else 0.0
            if sig != 0 and sig2 == 0 and sl > 0:
                rd_new = abs(op - sl)
                if rd_new > 1e-6:
                    sig_dir      = 1 if sig > 0 else -1
                    rr           = getattr(strat, "rr_target", 2.5)
                    entry_price  = op
                    initial_sl   = sl
                    current_sl   = sl
                    current_tp   = (entry_price + rr * rd_new if sig_dir == 1
                                    else entry_price - rr * rd_new)
                    position     = sig_dir
                    position_frac = 1.0
                    t1_hit       = False
                    bars_in_trade = 0
                    entry_bar_idx = i
                    trade_pnl_acc = 0.0
                    entry_score  = int(scores.iloc[i - 1]) if not pd.isna(scores.iloc[i - 1]) else 0

        equity_curve.append(equity)

    eq   = pd.Series(equity_curve, dtype=float)
    rets = eq.pct_change().fillna(0.0)
    tdf  = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

    span_days     = max(1, (times_m15.iloc[-1] - times_m15.iloc[0]).days)
    bars_per_year = max(1, int(len(df_m15) / (span_days / 365.25)))
    metrics       = calculate_metrics(rets, eq, tdf, bars_per_year)

    return TMBacktestResult(
        symbol=symbol, equity=eq, trades=tdf, metrics=metrics,
        n_tm_exits=n_tm_exits, n_tm_tighten=n_tm_tighten,
        n_tm_extend=n_tm_extend, n_tm_partial=n_tm_partial,
        n_tm_wait=n_tm_wait, n_tm_runner=n_tm_runner, n_t1=n_t1,
    )


def _get_m5_slice_fast(
    df_m5:      Optional[pd.DataFrame],
    m5_cutoffs: Optional[np.ndarray],
    bar_idx:    int,
    lookback:   int = 60,
) -> Optional[pd.DataFrame]:
    """O(1) M5 slice using precomputed boundary array."""
    if df_m5 is None or m5_cutoffs is None:
        return None
    last = int(m5_cutoffs[bar_idx])
    if last < 10:
        return None
    start = max(0, last - lookback + 1)
    return df_m5.iloc[start: last + 1].reset_index(drop=True)


def _get_m5_slice(
    df_m5:    Optional[pd.DataFrame],
    m5_times: Optional[pd.Series],
    bar_time: pd.Timestamp,
    lookback: int = 60,
) -> Optional[pd.DataFrame]:
    """O(N) fallback — used outside the main simulation loop."""
    if df_m5 is None or m5_times is None:
        return None
    cutoff = bar_time + pd.Timedelta(minutes=10)
    mask   = m5_times <= cutoff
    hits   = mask[mask].index
    if len(hits) < 10:
        return None
    start = max(0, hits[-1] - lookback + 1)
    return df_m5.iloc[start: hits[-1] + 1].reset_index(drop=True)


# ── Extended instrument universe (adds pairs not in the main multi-instrument runner) ──
# GER40: London session, long-only index — has M15 data, tested here only
# GBPUSD: London+NY forex, bidirectional — M15 data to 2024-12-31, M5 active for that period
_EXTRA_INSTRUMENTS: dict[str, dict] = {
    "GER40.cash": dict(session_start=7,  session_end=17),
    "GBPUSD":     dict(session_start=7,  session_end=21),
}
# Forex pairs that trade both directions (not in BIDIRECTIONAL from run_multi_instrument)
_FX_PAIRS: set[str] = {"GBPUSD"}

# Full universe for TM backtest = production instruments + extended test pairs
_ALL_TM_SYMBOLS: list[str] = list(INSTRUMENTS.keys()) + list(_EXTRA_INSTRUMENTS.keys())


# ── Comparison runner ─────────────────────────────────────────────────────────

def run_tm_comparison(
    symbols:         Optional[list[str]] = None,
    tf:              str   = "M15",
    initial_capital: float = 10_000.0,
    commission:      float = 0.0001,
) -> None:
    if symbols is None:
        symbols = _ALL_TM_SYMBOLS

    tm = TradeManager()

    print(f"\n{'='*72}")
    print(f"  TRADE MANAGER BACKTEST  [{tf}]  —  {len(symbols)} symbols")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"{'='*72}")

    summary_rows: list[dict] = []

    for symbol in symbols:
        # cfg from production instruments, then extended extras, then empty fallback
        cfg = INSTRUMENTS.get(symbol, _EXTRA_INSTRUMENTS.get(symbol, {}))
        p15 = PROCESSED / f"{symbol}_{tf}.csv"
        p5  = PROCESSED / f"{symbol}_M5.csv"

        if not p15.exists():
            print(f"  SKIP {symbol}: no {tf} CSV")
            continue

        df_m15 = pd.read_csv(p15)
        df_m15["time"] = pd.to_datetime(df_m15["time"], utc=True, errors="coerce")
        df_m15 = df_m15.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        df_m5 = None
        if p5.exists():
            df_m5_raw = pd.read_csv(p5)
            df_m5_raw["time"] = pd.to_datetime(df_m5_raw["time"], utc=True, errors="coerce")
            df_m5_raw = df_m5_raw.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
            m5_start = df_m5_raw["time"].iloc[0]
            m5_end   = df_m5_raw["time"].iloc[-1]
            m15_start = df_m15["time"].iloc[0]
            m15_end   = df_m15["time"].iloc[-1]
            # Only use M5 if it actually overlaps with the M15 window
            if m5_start <= m15_end and m5_end >= m15_start:
                df_m5 = df_m5_raw
                # Trim M15 to M5 coverage window (M5 is the supervisor TF)
                df_m15 = df_m15[
                    (df_m15["time"] >= m5_start) & (df_m15["time"] <= m5_end)
                ].reset_index(drop=True)

        if len(df_m15) < 200:
            print(f"  SKIP {symbol}: insufficient data ({len(df_m15)} bars)")
            continue

        # FX pairs trade both directions; production BIDIRECTIONAL set covers metals
        is_long_only = (symbol not in BIDIRECTIONAL) and (symbol not in _FX_PAIRS)

        tf_params = M15_PARAMS.copy() if tf == "M15" else H1_PARAMS.copy()
        trail_cfg = TRAIL_CONFIGS.get(symbol, {})
        v2 = dict(
            long_only=is_long_only,
            rr_model3_bonus=0.5, rr_trend_bonus=0.5, rr_trend_threshold=0.003, rr_max=5.0,
            session_prime_start=13, session_prime_end=15,
            use_rsi=True, rsi_period=56,
            rsi_long_lo=25.0, rsi_long_hi=55.0, rsi_short_lo=45.0, rsi_short_hi=75.0,
            trail_to_be=trail_cfg.get("trail_to_be", False),
            trail_be_r=trail_cfg.get("trail_be_r", 1.0),
            trail_lock_r=trail_cfg.get("trail_lock_r", 2.0),
            t1_r=0.0, t1_partial_pct=0.5, time_stop_bars=0,
            use_prime_bonus=True, use_vol_spike=False, vol_spike_mult=1.5,
            require_ce=False, use_d1_bias=False,
        )

        if symbol in OPTIMISED_PARAMS:
            opt   = OPTIMISED_PARAMS[symbol].copy()
            bias  = opt.pop("h4_bias_method", "ema")
            stopb = opt.pop("atr_stop_buffer", 0.5)
            strat = AiDENIndexStrategy(
                min_score=opt.get("min_score", 4),
                min_fvg_atr=opt.get("min_fvg_atr", 0.10),
                rr_target=opt.get("rr_target", 2.5),
                session_start=opt.get("session_start", cfg.get("session_start", 7)),
                session_end=opt.get("session_end",   cfg.get("session_end",   21)),
                h4_bias_method=bias, atr_stop_buffer=stopb,
                **{k: v for k, v in tf_params.items()
                   if k not in ("h4_bias_method", "atr_stop_buffer")},
                **v2,
            )
        else:
            bias  = tf_params.pop("h4_bias_method", "ema")
            stopb = tf_params.pop("atr_stop_buffer", 0.5)
            strat = AiDENIndexStrategy(
                min_score=4, min_fvg_atr=0.10, rr_target=2.5,
                session_start=cfg.get("session_start", 7),
                session_end=cfg.get("session_end",   21),
                h4_bias_method=bias, atr_stop_buffer=stopb,
                **tf_params, **v2,
            )

        m5_tag = "M5+M15" if df_m5 is not None else "M15 only"
        dir_tag = "bi" if not is_long_only else "long"
        print(f"\n  {symbol}  [{m5_tag}  {dir_tag}]  {len(df_m15)} bars")

        result = simulate(
            df_m15=df_m15, df_m5=df_m5, strat=strat, trade_manager=tm,
            symbol=symbol, initial_capital=initial_capital,
            risk_pct=1.0, commission=commission,
        )
        result.print_summary()

        m = result.metrics
        summary_rows.append({
            "symbol":  symbol,
            "return":  m["total_return_pct"],
            "dd":      m["max_drawdown_pct"],
            "sharpe":  m["sharpe_ratio"],
            "trades":  m["total_trades"],
            "wr":      m.get("win_rate_pct", 0),
            "exits":   result.n_tm_exits,
            "wait":    result.n_tm_wait,
            "runner":  result.n_tm_runner,
            "partial": result.n_tm_partial,
            "m5":      df_m5 is not None,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    if summary_rows:
        summary_rows.sort(key=lambda x: -x["sharpe"])
        print(f"\n{'='*72}")
        print(f"  TM BACKTEST SUMMARY — sorted by Sharpe")
        print(f"{'='*72}")
        print(f"  {'Symbol':<16} {'Return':>8} {'MaxDD':>7} {'Sharpe':>7} "
              f"{'Trades':>7} {'WR':>6} {'TM-exit':>8} {'WAIT':>5} {'RUN':>5} {'M5':>4}")
        print(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*5} {'-'*5} {'-'*4}")
        for r in summary_rows:
            ftmo = "!" if r["dd"] < -10 else " "
            print(
                f"{ftmo} {r['symbol']:<16} {r['return']:>+8.2f}% {r['dd']:>7.2f}% "
                f"{r['sharpe']:>7.3f} {r['trades']:>7} {r['wr']:>5.1f}% "
                f"{r['exits']:>8} {r['wait']:>5} {r['runner']:>5} "
                f"{'y' if r['m5'] else 'n':>4}"
            )
        viable  = [r for r in summary_rows if r["dd"] > -10 and r["sharpe"] > 1.0]
        breaches = [r for r in summary_rows if r["dd"] <= -10]
        print(f"\n  Viable (DD>-10%, Sharpe>1): {len(viable)}/{len(summary_rows)}")
        if breaches:
            print(f"  DD breach (!): {', '.join(r['symbol'] for r in breaches)}")
    print(f"\n{'='*72}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="TradeManager event-driven backtest")
    ap.add_argument("symbols", nargs="*", help="Symbols to run (default: all)")
    ap.add_argument("--tf",      default="M15")
    ap.add_argument("--capital", type=float, default=10_000.0)
    args = ap.parse_args()
    run_tm_comparison(symbols=args.symbols or None, tf=args.tf, initial_capital=args.capital)
