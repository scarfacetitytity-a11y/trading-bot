"""M2P5 — FTMO Phase 1 challenge simulation.

Method
------
1. Run the 8-instrument M15 portfolio (score=4 optimal config) — same as run_multi_instrument
2. Extract ALL trades across instruments with per-trade P&L% (price basis)
3. Scale each trade to proper account-risk sizing (target_risk_pct per loss)
   Scale = target_risk / abs(avg_loss_pct_from_backtest)
4. Build a day-by-day equity curve from the scaled trade sequence
5. Apply FTMO Phase 1 rules on every overlapping 30-day window
6. Block-bootstrap for 1000+ Monte Carlo trials to get pass-rate distribution

FTMO Phase 1 rules enforced
----------------------------
  Profit target   : +10% of initial balance  (stop as soon as hit)
  Max daily loss  : 5% of initial balance per calendar day  (hard breach = fail)
  Max total loss  : 10% of initial balance cumulative  (equity floor)
  Window          : 30 calendar days (we measure up to 30 trading days)
  Circuit breaker : 2.5% combined daily loss halts remaining trades that day

Usage
-----
    python -m backtests.run_ftmo_sim
    python -m backtests.run_ftmo_sim --risk 1.0          # 1% risk per trade
    python -m backtests.run_ftmo_sim --risk 1.0 --mc 2000  # 2000 bootstrap trials
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.engine import Backtest
from strategies.aiden_index import AiDENIndexStrategy

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

# ── Same config as run_multi_instrument score=4 optimal ──────────────────────

INSTRUMENTS = {
    "XAUUSD":      dict(session_start=7,  session_end=21),
    "US100.cash":  dict(session_start=12, session_end=21),
    "US30.cash":   dict(session_start=12, session_end=21),
    "US500.cash":  dict(session_start=12, session_end=21),
    "US2000.cash": dict(session_start=12, session_end=21),
    "UK100.cash":  dict(session_start=7,  session_end=17),
    "JP225.cash":  dict(session_start=0,  session_end=9),
    "XAGUSD":      dict(session_start=7,  session_end=21),
}

BIDIRECTIONAL = {"XAUUSD", "XAGUSD"}

M15_PARAMS = dict(
    htf_lookback=20, h4_swing_lookback=40,
    max_fvg_wait=160, max_entry_wait=32,
    ob_lookback=80, liq_lookback=40,
    atr_period=56, h4_bias_method="ema",
)

V2_DEFAULTS = dict(
    rr_model3_bonus=0.5, rr_trend_bonus=0.5, rr_trend_threshold=0.003,
    rr_max=5.0, session_prime_start=13, session_prime_end=15,
    use_rsi=True, rsi_period=56,
    rsi_long_lo=25.0, rsi_long_hi=55.0,
    rsi_short_lo=45.0, rsi_short_hi=75.0,
)

OPTIMISED = {
    "US2000.cash": dict(min_score=5, rr_target=3.5, min_fvg_atr=0.10,
                        atr_stop_buffer=0.3, session_start=12, session_end=21),
    "UK100.cash":  dict(min_score=4, rr_target=3.0, min_fvg_atr=0.15,
                        atr_stop_buffer=0.3, session_start=7,  session_end=17),
    "XAGUSD":      dict(min_score=6, rr_target=3.0, min_fvg_atr=0.15,
                        atr_stop_buffer=0.3, session_start=12, session_end=21),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_strategy(symbol: str, score: int, cfg: dict) -> AiDENIndexStrategy:
    tf_params = M15_PARAMS.copy()
    v2 = V2_DEFAULTS.copy()
    v2["long_only"] = symbol not in BIDIRECTIONAL

    if symbol in OPTIMISED:
        opt = OPTIMISED[symbol].copy()
        bias = opt.pop("h4_bias_method", tf_params.pop("h4_bias_method", "ema"))
        stop = opt.pop("atr_stop_buffer", 0.5)
        return AiDENIndexStrategy(
            min_score=opt.get("min_score", score),
            min_fvg_atr=opt.get("min_fvg_atr", 0.10),
            rr_target=opt.get("rr_target", 2.5),
            session_start=opt.get("session_start", cfg["session_start"]),
            session_end=opt.get("session_end", cfg["session_end"]),
            h4_bias_method=bias, atr_stop_buffer=stop,
            **{k: v for k, v in tf_params.items() if k not in ("h4_bias_method",)},
            **v2,
        )
    else:
        bias = tf_params.pop("h4_bias_method", "ema")
        return AiDENIndexStrategy(
            min_score=score, min_fvg_atr=0.10, rr_target=2.5,
            session_start=cfg["session_start"], session_end=cfg["session_end"],
            h4_bias_method=bias, atr_stop_buffer=0.5,
            **tf_params, **v2,
        )


def _load_trades(score: int = 4) -> tuple[pd.DataFrame, float]:
    """Run all instruments and return merged trade log + avg_loss_pct."""
    all_trades = []
    for symbol, cfg in INSTRUMENTS.items():
        path = PROCESSED / f"{symbol}_M15.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        strat = _build_strategy(symbol, score, cfg)
        bt    = Backtest(strat, initial_capital=10_000, commission=0.0001)
        r     = bt.run(df, symbol=symbol)

        trades = r.trades.copy()
        if trades.empty:
            continue
        trades["symbol"] = symbol
        all_trades.append(trades)
        print(f"  {symbol:<16} {len(trades):>4} trades  "
              f"WR {(trades['pnl_pct']>0).mean()*100:.1f}%  "
              f"avg_win {trades[trades['pnl_pct']>0]['pnl_pct'].mean()*100:+.3f}%  "
              f"avg_loss {trades[trades['pnl_pct']<0]['pnl_pct'].mean()*100:+.3f}%")

    if not all_trades:
        sys.exit("No trades found — run data pipeline first.")

    all_df = pd.concat(all_trades, ignore_index=True)
    all_df["entry_time"] = pd.to_datetime(all_df["entry_time"])
    all_df["exit_time"]  = pd.to_datetime(all_df["exit_time"])
    all_df = all_df.sort_values("exit_time").reset_index(drop=True)

    losses = all_df[all_df["pnl_pct"] < 0]["pnl_pct"]
    avg_loss = losses.mean() if len(losses) > 0 else -0.005
    return all_df, float(avg_loss)


# ── FTMO window simulation ────────────────────────────────────────────────────

def _simulate_window(
    daily_pnl: list[float],
    initial: float      = 100_000,
    target_pct: float   = 0.10,
    max_dd_pct: float   = 0.10,
    daily_limit_pct: float = 0.05,
) -> dict:
    """Simulate one FTMO Phase 1 window.

    daily_pnl: list of fractional daily P&L (e.g. 0.02 = +2%)
    Returns: {'result': 'pass'|'daily_dd'|'max_dd'|'timeout',
              'day': int, 'max_dd_hit_pct': float, 'final_pct': float}
    """
    equity    = initial
    target    = initial * (1 + target_pct)
    dd_floor  = initial * (1 - max_dd_pct)
    min_eq    = initial

    for day_i, r in enumerate(daily_pnl):
        sod = equity
        eod = equity * (1 + r)
        min_eq = min(min_eq, eod)

        daily_loss_abs = sod - eod
        if daily_loss_abs > initial * daily_limit_pct:
            return dict(result="daily_dd", day=day_i + 1,
                        max_dd_hit_pct=(initial - min_eq) / initial * 100,
                        final_pct=(eod - initial) / initial * 100)

        equity = eod

        if equity <= dd_floor:
            return dict(result="max_dd", day=day_i + 1,
                        max_dd_hit_pct=(initial - min_eq) / initial * 100,
                        final_pct=(equity - initial) / initial * 100)

        if equity >= target:
            return dict(result="pass", day=day_i + 1,
                        max_dd_hit_pct=(initial - min_eq) / initial * 100,
                        final_pct=(equity - initial) / initial * 100)

    return dict(result="timeout", day=len(daily_pnl),
                max_dd_hit_pct=(initial - min_eq) / initial * 100,
                final_pct=(equity - initial) / initial * 100)


def _trades_to_daily(trades: pd.DataFrame, scale: float) -> pd.Series:
    """Sum scaled trade P&Ls into trading-day buckets."""
    trades = trades.copy()
    trades["pnl_scaled"] = trades["pnl_pct"] * scale
    trades["trade_date"] = trades["exit_time"].dt.normalize()
    daily = trades.groupby("trade_date")["pnl_scaled"].sum()
    return daily


def _apply_circuit_breaker(daily: pd.Series, halt: float = 0.025) -> pd.Series:
    result = daily.copy()
    for dt, val in daily.items():
        if val < -halt:
            result[dt] = -halt
    return result


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def _block_bootstrap(daily: pd.Series, window: int = 22, n_trials: int = 1000,
                     block_size: int = 5) -> list[list[float]]:
    """Draw n_trials sequences of `window` trading days using block bootstrap."""
    vals = daily.values
    n    = len(vals)
    if n < window:
        return [[random.choice(vals) for _ in range(window)] for _ in range(n_trials)]

    trials = []
    for _ in range(n_trials):
        seq = []
        while len(seq) < window:
            start = random.randint(0, n - 1)
            blk   = list(vals[start:min(start + block_size, n)])
            seq.extend(blk)
        trials.append(seq[:window])
    return trials


# ── Main ──────────────────────────────────────────────────────────────────────

def run_sim(
    score: int         = 4,
    risk_pcts: list[float] = None,
    mc_trials: int     = 1000,
    block_size: int    = 5,
    window_days: int   = 22,   # ≈ 30 calendar days
) -> None:
    if risk_pcts is None:
        risk_pcts = [0.5, 1.0, 1.5, 2.0]

    print(f"\n{'='*65}")
    print(f"  M2P5 — FTMO Phase 1 Simulation  (score={score}, M15, 8 instruments)")
    print(f"{'='*65}\n")

    print("Building trade log...")
    trades, avg_loss_pct = _load_trades(score)
    n_trades = len(trades)
    wr = (trades["pnl_pct"] > 0).mean()
    avg_win  = trades[trades["pnl_pct"] > 0]["pnl_pct"].mean()
    avg_loss_abs = abs(avg_loss_pct)

    print(f"\n  Total trades     : {n_trades}")
    print(f"  Win rate         : {wr*100:.1f}%")
    print(f"  Avg win (raw)    : {avg_win*100:+.4f}%")
    print(f"  Avg loss (raw)   : {avg_loss_pct*100:+.4f}%")
    print(f"  Implied RR       : {avg_win / avg_loss_abs:.2f}R")

    date_start = trades["exit_time"].min().date()
    date_end   = trades["exit_time"].max().date()
    n_days     = len(pd.bdate_range(str(date_start), str(date_end)))
    print(f"  Period           : {date_start} to {date_end} ({n_days} trading days)")
    print(f"  Trades/month     : {n_trades / (n_days / 22):.1f}")
    print()

    header = (f"{'Risk/trade':>12}  {'Scale':>7}  {'Pass%':>7}  {'Avg days':>9}  "
              f"{'Fail:DailyDD':>12}  {'Fail:MaxDD':>10}  {'Timeout':>8}  "
              f"{'Worst DD':>9}  {'Exp/month':>10}")
    print(header)
    print("-" * len(header))

    for risk_pct in risk_pcts:
        scale = risk_pct / 100.0 / avg_loss_abs

        # Build scaled daily P&L series
        daily_raw = _trades_to_daily(trades, scale)
        daily     = _apply_circuit_breaker(daily_raw, halt=0.025)

        # Historical windows (overlapping)
        daily_arr = daily.values
        n_d       = len(daily_arr)
        hist_windows = []
        for start in range(n_d - window_days + 1):
            hist_windows.append(list(daily_arr[start:start + window_days]))

        # Bootstrap windows
        boot_windows = _block_bootstrap(daily, window_days, mc_trials, block_size)
        all_windows  = hist_windows + boot_windows

        results   = [_simulate_window(w) for w in all_windows]
        passes    = [r for r in results if r["result"] == "pass"]
        daily_dd  = [r for r in results if r["result"] == "daily_dd"]
        max_dd    = [r for r in results if r["result"] == "max_dd"]
        timeouts  = [r for r in results if r["result"] == "timeout"]
        n_total   = len(results)

        pass_rate    = len(passes) / n_total * 100
        avg_days     = np.mean([r["day"] for r in passes]) if passes else float("nan")
        worst_dd     = max(r["max_dd_hit_pct"] for r in results)
        exp_per_month = daily.mean() * 22 * 100

        print(f"  {risk_pct:>10.1f}%"
              f"  {scale:>7.1f}x"
              f"  {pass_rate:>6.1f}%"
              f"  {avg_days:>9.1f}"
              f"  {len(daily_dd)/n_total*100:>11.1f}%"
              f"  {len(max_dd)/n_total*100:>9.1f}%"
              f"  {len(timeouts)/n_total*100:>7.1f}%"
              f"  {worst_dd:>8.2f}%"
              f"  {exp_per_month:>9.2f}%")

    # Detailed breakdown at the recommended risk level (1%)
    risk_rec = 1.0
    scale_rec = risk_rec / 100.0 / avg_loss_abs
    daily_rec = _apply_circuit_breaker(_trades_to_daily(trades, scale_rec), 0.025)
    windows_rec = _block_bootstrap(daily_rec, window_days, 5000, block_size)
    results_rec = [_simulate_window(w) for w in windows_rec]
    passes_rec  = [r for r in results_rec if r["result"] == "pass"]
    n_rec       = len(results_rec)

    pass_pct = len(passes_rec) / n_rec * 100
    print(f"\n{'='*65}")
    print(f"  FTMO PASS PROBABILITY @ 1% risk/trade: {pass_pct:.1f}%")
    print(f"{'='*65}")

    if passes_rec:
        day_arr = [r["day"] for r in passes_rec]
        dd_arr  = [r["max_dd_hit_pct"] for r in passes_rec]
        fin_arr = [r["final_pct"] for r in passes_rec]
        print(f"  Median days to pass  : {np.median(day_arr):.0f}")
        print(f"  P10 days (fast pass) : {np.percentile(day_arr, 10):.0f}")
        print(f"  P90 days (slow pass) : {np.percentile(day_arr, 90):.0f}")
        print(f"  Avg max DD when pass : {np.mean(dd_arr):.2f}%")
        print(f"  Avg final equity     : +{np.mean(fin_arr):.2f}%")

    print(f"\n  Fail breakdown (5000-trial MC @ 1% risk):")
    daily_dd_r = [r for r in results_rec if r["result"] == "daily_dd"]
    max_dd_r   = [r for r in results_rec if r["result"] == "max_dd"]
    timeouts_r = [r for r in results_rec if r["result"] == "timeout"]
    print(f"    Daily DD breach : {len(daily_dd_r)/n_rec*100:.1f}%")
    print(f"    Max DD breach   : {len(max_dd_r)/n_rec*100:.1f}%")
    print(f"    Timeout (30d)   : {len(timeouts_r)/n_rec*100:.1f}%")

    # FTMO expected timeline
    exp_daily = daily_rec.mean()
    exp_days  = 0.10 / exp_daily if exp_daily > 0 else float("inf")
    print(f"\n  Expected days to +10% (deterministic): {exp_days:.0f}")
    print(f"  Expected monthly return @ 1% risk : {exp_daily*22*100:.1f}%")
    print(f"  Max observed DD across all windows : "
          f"{max(r['max_dd_hit_pct'] for r in results_rec):.2f}%")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score",  type=int,   default=4)
    parser.add_argument("--risk",   type=float, default=None,
                        help="Single risk-pct override (default: sweep 0.5,1.0,1.5,2.0)")
    parser.add_argument("--mc",     type=int,   default=1000, help="MC bootstrap trials")
    parser.add_argument("--block",  type=int,   default=5,    help="Block bootstrap size (days)")
    args = parser.parse_args()

    risks = [args.risk] if args.risk else [0.5, 1.0, 1.5, 2.0]
    run_sim(score=args.score, risk_pcts=risks, mc_trials=args.mc, block_size=args.block)


if __name__ == "__main__":
    main()
