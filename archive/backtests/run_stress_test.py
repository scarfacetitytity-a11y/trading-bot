"""FTMO Stress Test — all instruments, three risk levels, three account sizes.

Runs the full TradeManager event-driven simulation (M5+M15) for every symbol
across three risk levels and reports:
  - Per-symbol: return%, maxDD%, maxDailyDD%, Sharpe, trades, WR, T1 hits, TM actions
  - Portfolio combined: shared-capital concurrent exposure analysis
  - FTMO compliance: daily DD ≤ 5%, max DD ≤ 10%
  - Dollar P&L and worst-case loss for £10k / £50k / $100k account sizes

Daily DD circuit breaker mirrors live config: -2.0% halts all entries for the day.
T1 jab fires at +1R (closes 50%), runner managed by HOLD_RUNNER (sucker punch).

Usage:
    python -m backtests.run_stress_test
    python -m backtests.run_stress_test --risk 0.5 1.0 1.5
    python -m backtests.run_stress_test --risk 1.0 --capital 10000 50000 100000
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.trade_manager import TradeManager
from backtests.tm_backtest import (
    simulate, TMBacktestResult,
    _EXTRA_INSTRUMENTS, _FX_PAIRS, _ALL_TM_SYMBOLS,
)
from backtests.run_multi_instrument import (
    INSTRUMENTS, BIDIRECTIONAL, TRAIL_CONFIGS, OPTIMISED_PARAMS,
    M15_PARAMS, H1_PARAMS, _size_mult_from_score,
)
from strategies.aiden_index import AiDENIndexStrategy

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

DAILY_HALT_PCT  = 2.0   # matches live risk_agent max_daily_loss_pct
FTMO_DAILY_LIMIT = 5.0  # FTMO hard rule
FTMO_MAX_DD     = 10.0  # FTMO hard rule

DEFAULT_RISK_LEVELS  = [0.5, 1.0, 1.5]
DEFAULT_CAPITALS     = [10_000.0, 50_000.0, 100_000.0]
CAPITAL_LABELS       = ["£10k", "£50k", "$100k"]


# ── Daily drawdown extractor ──────────────────────────────────────────────────

def _max_daily_loss(equity: pd.Series, times: pd.Series) -> float:
    """Return worst single-day equity change as a negative %, or 0."""
    df = pd.DataFrame({"eq": equity.values, "t": pd.to_datetime(times.values, utc=True)})
    df["date"] = df["t"].dt.date
    daily = df.groupby("date")["eq"].last()
    if len(daily) < 2:
        return 0.0
    pct = daily.pct_change().dropna()
    return float(pct.min() * 100) if len(pct) else 0.0


def _daily_equity(equity: pd.Series, times: pd.Series) -> pd.Series:
    """Return daily closing equity indexed by date."""
    df = pd.DataFrame({"eq": equity.values, "t": pd.to_datetime(times.values, utc=True)})
    df["date"] = df["t"].dt.date
    return df.groupby("date")["eq"].last()


# ── Instrument loader / strategy builder (shared with tm_backtest) ────────────

def _load_symbol(symbol: str) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    p15 = PROCESSED / f"{symbol}_M15.csv"
    p5  = PROCESSED / f"{symbol}_M5.csv"

    if not p15.exists():
        return None, None

    df15 = pd.read_csv(p15)
    df15["time"] = pd.to_datetime(df15["time"], utc=True, errors="coerce")
    df15 = df15.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    df5 = None
    if p5.exists():
        raw5 = pd.read_csv(p5)
        raw5["time"] = pd.to_datetime(raw5["time"], utc=True, errors="coerce")
        raw5 = raw5.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        # Only use M5 if it overlaps with M15 window
        if raw5["time"].iloc[0] <= df15["time"].iloc[-1] and \
           raw5["time"].iloc[-1] >= df15["time"].iloc[0]:
            df5 = raw5
            df15 = df15[
                (df15["time"] >= raw5["time"].iloc[0]) &
                (df15["time"] <= raw5["time"].iloc[-1])
            ].reset_index(drop=True)

    if len(df15) < 200:
        return None, None

    return df15, df5


def _build_strategy(symbol: str) -> AiDENIndexStrategy:
    cfg       = INSTRUMENTS.get(symbol, _EXTRA_INSTRUMENTS.get(symbol, {}))
    tf_params = M15_PARAMS.copy()
    trail_cfg = TRAIL_CONFIGS.get(symbol, {})
    is_long_only = (symbol not in BIDIRECTIONAL) and (symbol not in _FX_PAIRS)

    v2 = dict(
        long_only=is_long_only,
        rr_model3_bonus=0.5, rr_trend_bonus=0.5, rr_trend_threshold=0.003, rr_max=5.0,
        session_prime_start=13, session_prime_end=15,
        use_rsi=True, rsi_period=56,
        rsi_long_lo=25.0, rsi_long_hi=55.0, rsi_short_lo=45.0, rsi_short_hi=75.0,
        trail_to_be=trail_cfg.get("trail_to_be", False),
        trail_be_r=trail_cfg.get("trail_be_r", 1.0),
        trail_lock_r=trail_cfg.get("trail_lock_r", 2.0),
        t1_r=trail_cfg.get("t1_r", 0.0),
        t1_partial_pct=trail_cfg.get("t1_partial_pct", 0.5),
        time_stop_bars=0,
        use_prime_bonus=True, use_vol_spike=False, vol_spike_mult=1.5,
        require_ce=False, use_d1_bias=False,
    )

    if symbol in OPTIMISED_PARAMS:
        opt   = OPTIMISED_PARAMS[symbol].copy()
        bias  = opt.pop("h4_bias_method", "ema")
        stopb = opt.pop("atr_stop_buffer", 0.5)
        return AiDENIndexStrategy(
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
        return AiDENIndexStrategy(
            min_score=4, min_fvg_atr=0.10, rr_target=2.5,
            session_start=cfg.get("session_start", 7),
            session_end=cfg.get("session_end",   21),
            h4_bias_method=bias, atr_stop_buffer=stopb,
            **tf_params, **v2,
        )


# ── Single risk-level run ─────────────────────────────────────────────────────

@dataclass
class SymbolResult:
    symbol:        str
    risk_pct:      float
    ret_pct:       float
    max_dd_pct:    float
    max_daily_dd:  float
    sharpe:        float
    trades:        int
    wr:            float
    t1_hits:       int
    tm_exits:      int
    tm_runner:     int
    tm_wait:       int
    has_m5:        bool
    equity:        pd.Series = field(repr=False, default_factory=pd.Series)
    times:         pd.Series = field(repr=False, default_factory=pd.Series)

    @property
    def ftmo_daily_ok(self) -> bool:
        return self.max_daily_dd >= -FTMO_DAILY_LIMIT

    @property
    def ftmo_dd_ok(self) -> bool:
        return self.max_dd_pct >= -FTMO_MAX_DD


def run_single_risk(
    risk_pct:    float,
    symbols:     list[str],
    commission:  float = 0.0001,
    daily_halt:  float = DAILY_HALT_PCT,
) -> list[SymbolResult]:
    tm      = TradeManager()
    results = []

    for symbol in symbols:
        df15, df5 = _load_symbol(symbol)
        if df15 is None:
            print(f"  SKIP {symbol}: no M15 data")
            continue

        strat = _build_strategy(symbol)
        m5_tag = "M5+M15" if df5 is not None else "M15"
        print(f"  {symbol:<16} [{m5_tag}]  {len(df15)} bars  risk={risk_pct}%", end="", flush=True)

        try:
            res = simulate(
                df_m15=df15, df_m5=df5, strat=strat, trade_manager=tm,
                symbol=symbol, initial_capital=1.0,
                risk_pct=risk_pct, commission=commission,
                daily_halt_pct=daily_halt,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        m  = res.metrics
        ts = df15["time"].reset_index(drop=True)

        # align times to equity length (equity has len(df15) elements)
        eq_times = ts.iloc[:len(res.equity)].reset_index(drop=True)

        max_daily = _max_daily_loss(res.equity, eq_times)

        sr = SymbolResult(
            symbol=symbol,
            risk_pct=risk_pct,
            ret_pct=m["total_return_pct"],
            max_dd_pct=m["max_drawdown_pct"],
            max_daily_dd=max_daily,
            sharpe=m["sharpe_ratio"],
            trades=m["total_trades"],
            wr=m.get("win_rate_pct", 0.0),
            t1_hits=res.n_t1,
            tm_exits=res.n_tm_exits,
            tm_runner=res.n_tm_runner,
            tm_wait=res.n_tm_wait,
            has_m5=(df5 is not None),
            equity=res.equity,
            times=eq_times,
        )
        print(f"  return={sr.ret_pct:+.1f}%  maxDD={sr.max_dd_pct:.1f}%  "
              f"dailyDD={sr.max_daily_dd:.1f}%  Sharpe={sr.sharpe:.2f}  "
              f"trades={sr.trades}  WR={sr.wr:.0f}%")
        results.append(sr)

    return results


# ── Portfolio combined analysis ───────────────────────────────────────────────

def _portfolio_metrics(results: list[SymbolResult], risk_pct: float) -> dict:
    """
    Shared-capital portfolio model:
      Each instrument risks risk_pct of the SAME account equity.
      When N instruments are open simultaneously, max concurrent exposure = N × risk_pct.
      Portfolio P&L each day = sum of individual instrument P&L fractions.
    """
    if not results:
        return {}

    # Collect all unique dates across all instruments
    all_dates = sorted(set(
        d for r in results
        for d in _daily_equity(r.equity, r.times).index
    ))
    if not all_dates:
        return {}

    # Build combined daily P&L: each instrument contributes its daily return fraction
    combined_pnl_series: list[float] = []
    prev_equity = {r.symbol: 1.0 for r in results}

    daily_eqs: dict[str, pd.Series] = {
        r.symbol: _daily_equity(r.equity, r.times)
        for r in results
    }

    portfolio_equity = 1.0
    port_eq_curve: list[float] = [1.0]

    for date in all_dates[1:]:
        day_pnl = 0.0
        for r in results:
            deq = daily_eqs[r.symbol]
            if date in deq.index:
                today_eq = deq[date]
                prev_eq  = prev_equity.get(r.symbol, 1.0)
                # Fractional change THIS instrument contributed as a fraction of 1.0 start
                day_pnl += (today_eq - prev_eq)
                prev_equity[r.symbol] = today_eq
        portfolio_equity += day_pnl
        port_eq_curve.append(portfolio_equity)

    port_series  = pd.Series(port_eq_curve, dtype=float)
    rolling_max  = port_series.cummax()
    drawdowns    = (port_series - rolling_max) / rolling_max * 100
    max_dd       = float(drawdowns.min())

    port_pct_changes = port_series.pct_change().dropna() * 100
    max_daily_dd     = float(port_pct_changes.min()) if len(port_pct_changes) else 0.0
    total_return     = (port_series.iloc[-1] - 1.0) * 100

    n_sym           = len(results)
    max_concurrent  = n_sym * risk_pct   # worst-case if all open at once (% of account)

    return {
        "symbols":          n_sym,
        "total_return_pct": total_return,
        "max_dd_pct":       max_dd,
        "max_daily_dd_pct": max_daily_dd,
        "max_concurrent_exposure_pct": max_concurrent,
        "ftmo_dd_ok":       max_dd >= -FTMO_MAX_DD,
        "ftmo_daily_ok":    max_daily_dd >= -FTMO_DAILY_LIMIT,
        "port_equity":      port_series,
    }


# ── Full stress test output ───────────────────────────────────────────────────

def print_symbol_table(results: list[SymbolResult]) -> None:
    print(f"\n  {'Symbol':<16} {'Ret%':>7} {'MaxDD%':>7} {'DailyDD%':>9} "
          f"{'Sharpe':>7} {'Trades':>7} {'WR':>5} "
          f"{'T1':>4} {'TM-E':>5} {'Run':>5} {'M5':>3} {'FTMO'}")
    print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*9} {'-'*7} {'-'*7} {'-'*5} "
          f"{'-'*4} {'-'*5} {'-'*5} {'-'*3} {'-'*6}")
    for r in sorted(results, key=lambda x: -x.sharpe):
        dd_flag   = "!" if not r.ftmo_dd_ok else " "
        day_flag  = "!" if not r.ftmo_daily_ok else " "
        ftmo_tag  = f"DD{dd_flag} Day{day_flag}"
        print(
            f"  {r.symbol:<16} {r.ret_pct:>+7.2f}% {r.max_dd_pct:>7.2f}% "
            f"{r.max_daily_dd:>9.2f}% {r.sharpe:>7.3f} {r.trades:>7} "
            f"{r.wr:>5.1f}% {r.t1_hits:>4} {r.tm_exits:>5} {r.tm_runner:>5} "
            f"{'y' if r.has_m5 else 'n':>3}  {ftmo_tag}"
        )


def print_dollar_table(
    results:  list[SymbolResult],
    port_met: dict,
    risk_pct: float,
    capitals: list[float],
    labels:   list[str],
) -> None:
    print(f"\n  Dollar P&L @ {risk_pct}% risk per trade:")
    header = f"  {'Symbol':<16}" + "".join(f"  {l:>12}" for l in labels)
    print(header)
    print("  " + "-"*16 + ("  " + "-"*12) * len(labels))
    for r in sorted(results, key=lambda x: -x.sharpe):
        row = f"  {r.symbol:<16}"
        for cap in capitals:
            dollar = r.ret_pct / 100.0 * cap
            row += f"  {dollar:>+12,.0f}"
        print(row)
    # Portfolio total
    if port_met:
        row = f"  {'PORTFOLIO':<16}"
        for cap in capitals:
            dollar = port_met["total_return_pct"] / 100.0 * cap
            row += f"  {dollar:>+12,.0f}"
        print(row)
    # Worst-case loss (max DD in dollars)
    print(f"\n  Worst-case account loss (max DD):")
    for r in sorted(results, key=lambda x: x.max_dd_pct):
        row = f"  {r.symbol:<16}"
        for cap in capitals:
            dollar = r.max_dd_pct / 100.0 * cap
            row += f"  {dollar:>+12,.0f}"
        print(row)
    if port_met:
        row = f"  {'PORTFOLIO':<16}"
        for cap in capitals:
            dollar = port_met["max_dd_pct"] / 100.0 * cap
            row += f"  {dollar:>+12,.0f}"
        print(row)


def run_stress_test(
    risk_levels: list[float]  = DEFAULT_RISK_LEVELS,
    capitals:    list[float]  = DEFAULT_CAPITALS,
    cap_labels:  list[str]    = CAPITAL_LABELS,
    symbols:     Optional[list[str]] = None,
    commission:  float = 0.0001,
) -> None:
    if symbols is None:
        symbols = _ALL_TM_SYMBOLS

    print(f"\n{'='*80}")
    print(f"  AiDEN FTMO STRESS TEST")
    print(f"  Symbols : {', '.join(symbols)}")
    print(f"  Risk lvls: {risk_levels}%  |  Accounts: {cap_labels}")
    print(f"  Daily halt: -{DAILY_HALT_PCT}%  |  FTMO gates: -{FTMO_DAILY_LIMIT}% daily / -{FTMO_MAX_DD}% max")
    print(f"  TM: jab(T1=1R 50%) + sucker punch (HOLD_RUNNER) + WAIT sweep detect")
    print(f"{'='*80}")

    for risk in risk_levels:
        print(f"\n{'-'*80}")
        print(f"  RISK LEVEL: {risk}% per trade")
        print(f"{'-'*80}")

        results = run_single_risk(risk_pct=risk, symbols=symbols, commission=commission)

        if not results:
            print("  No results.")
            continue

        print_symbol_table(results)

        # Portfolio combined
        port = _portfolio_metrics(results, risk_pct=risk)
        if port:
            n   = port["symbols"]
            ret = port["total_return_pct"]
            dd  = port["max_dd_pct"]
            ddd = port["max_daily_dd_pct"]
            exp = port["max_concurrent_exposure_pct"]
            ok_dd  = port["ftmo_dd_ok"]
            ok_day = port["ftmo_daily_ok"]

            print(f"\n  PORTFOLIO ({n} instruments @ {risk}% each):")
            print(f"    Combined return    : {ret:+.2f}%")
            print(f"    Max DD             : {dd:.2f}%  {'PASS' if ok_dd else 'FTMO BREACH!'}")
            print(f"    Max daily DD       : {ddd:.2f}%  {'PASS' if ok_day else 'FTMO BREACH!'}")
            print(f"    Max concurrent exp : {exp:.1f}% (all {n} open at once)")

        # Dollar table
        print_dollar_table(results, port, risk_pct=risk, capitals=capitals, labels=cap_labels)

        # Viable instruments
        viable   = [r for r in results if r.ftmo_dd_ok and r.ftmo_daily_ok and r.sharpe > 1.0]
        breaches = [r for r in results if not r.ftmo_dd_ok or not r.ftmo_daily_ok]
        print(f"\n  Viable (FTMO pass + Sharpe>1) : {len(viable)}/{len(results)}")
        if breaches:
            print(f"  FTMO breaches: {', '.join(r.symbol for r in breaches)}")

    print(f"\n{'='*80}")
    print("  STRESS TEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AiDEN FTMO Stress Test")
    ap.add_argument("--risk",    type=float, nargs="+", default=DEFAULT_RISK_LEVELS,
                    help="Risk levels to test (default: 0.5 1.0 1.5)")
    ap.add_argument("--capital", type=float, nargs="+", default=DEFAULT_CAPITALS,
                    help="Account sizes for dollar P&L (default: 10000 50000 100000)")
    ap.add_argument("symbols",   nargs="*", help="Symbols to test (default: all)")
    args = ap.parse_args()

    labels = [f"£{int(c/1000)}k" if c < 100_000 else f"${int(c/1000)}k"
              for c in args.capital]

    run_stress_test(
        risk_levels=args.risk,
        capitals=args.capital,
        cap_labels=labels,
        symbols=args.symbols or None,
    )
