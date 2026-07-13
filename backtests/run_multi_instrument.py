"""AiDEN multi-instrument combined backtest — M2P4b.

Quant (MEM-002): validates combined US100 + US30 + XAUUSD portfolio at score>=5.
Sage  (MEM-001): architecture — shared capital pool, equal allocation, daily
                 circuit breaker at 2.5% portfolio loss halts all instruments.

Portfolio rules:
  - Equal capital allocation per instrument (1/N of equity)
  - All instruments run simultaneously on the same time grid
  - Daily circuit breaker: if combined daily loss > daily_halt_pct, all positions
    set to 0 for the rest of that calendar day
  - FTMO-safe target: combined MaxDD < 10%

Usage:
    python -m backtests.run_multi_instrument
    python -m backtests.run_multi_instrument --halt 0.025   # 2.5% daily halt
    python -m backtests.run_multi_instrument --score 4      # looser confluence
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.engine import Backtest
from backtests.metrics import calculate_metrics
from strategies.aiden_index import AiDENIndexStrategy

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

# Instrument configs — session hours tuned per instrument
INSTRUMENTS = {
    "US100.cash": dict(session_start=12, session_end=21),
    "US30.cash":  dict(session_start=12, session_end=21),
    "XAUUSD":     dict(session_start=7,  session_end=21),  # London + NY
}


def _load(symbol: str, tf: str = "H1") -> pd.DataFrame | None:
    path = PROCESSED / f"{symbol}_{tf}.csv"
    if not path.exists():
        print(f"  SKIP {symbol}: not found", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def _generate_returns(
    symbol: str,
    cfg: dict,
    score: int,
    fvg_atr: float,
    rr: float,
    commission: float,
) -> tuple[pd.Series, pd.Series, pd.DataFrame] | None:
    """Returns (time_series, bar_returns, trades) or None."""
    df = _load(symbol)
    if df is None:
        return None

    strat = AiDENIndexStrategy(
        min_score=score,
        min_fvg_atr=fvg_atr,
        rr_target=rr,
        session_start=cfg["session_start"],
        session_end=cfg["session_end"],
    )
    bt = Backtest(strat, initial_capital=10_000, commission=commission)
    r  = bt.run(df, symbol=symbol)

    # Align to time index
    times   = pd.to_datetime(df["time"])
    returns = r.returns.copy()
    returns.index = times

    trades = r.trades.copy()
    trades["symbol"] = symbol

    return times, returns, trades


def run_combined(
    score: int      = 5,
    fvg_atr: float  = 0.10,
    rr: float       = 2.5,
    daily_halt: float = 0.025,
    commission: float = 0.0001,
    initial_capital: float = 10_000,
) -> None:
    print(f"\nAiDEN Multi-Instrument Backtest")
    print(f"Instruments: {', '.join(INSTRUMENTS)}")
    print(f"Config: score>={score}  fvg_atr={fvg_atr}  rr={rr}  daily_halt={daily_halt*100:.1f}%")
    print(f"Capital: {initial_capital:,.0f}  Commission: {commission}\n")

    # ── 1. Generate per-instrument returns ───────────────────────────────────
    per_instrument: dict[str, pd.Series] = {}
    all_trades: list[pd.DataFrame] = []

    for symbol, cfg in INSTRUMENTS.items():
        result = _generate_returns(symbol, cfg, score, fvg_atr, rr, commission)
        if result is None:
            continue
        times, returns, trades = result
        per_instrument[symbol] = returns
        all_trades.append(trades)
        m = _quick_metrics(returns, initial_capital)
        print(f"  {symbol:<16} Return: {m['ret']:>+7.2f}%  MaxDD: {m['dd']:>7.2f}%  "
              f"Sharpe: {m['sh']:>6.3f}  Trades: {m['tr']:>4}")

    if not per_instrument:
        print("No instrument data found.")
        return

    # ── 2. Align on common time grid ─────────────────────────────────────────
    combined_returns = pd.DataFrame(per_instrument)
    combined_returns = combined_returns.sort_index().fillna(0)

    n = len(per_instrument)
    weight = 1.0 / n  # equal allocation

    # Per-bar portfolio return = weighted average of instrument returns
    portfolio_returns = combined_returns.sum(axis=1) * weight

    # ── 3. Apply daily circuit breaker ───────────────────────────────────────
    portfolio_returns = _apply_circuit_breaker(portfolio_returns, daily_halt)

    # ── 4. Combined equity curve ──────────────────────────────────────────────
    equity = pd.Series(
        initial_capital * (1 + portfolio_returns).cumprod(),
        index=portfolio_returns.index,
    )

    bars_per_year = _estimate_bpy(portfolio_returns)
    all_trade_df  = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    metrics = calculate_metrics(portfolio_returns, equity, all_trade_df, bars_per_year)

    # ── 5. Print combined results ─────────────────────────────────────────────
    ftmo_status = "SAFE" if metrics["max_drawdown_pct"] > -10 else "BREACH"
    print(f"\n{'='*60}")
    print(f"  COMBINED PORTFOLIO ({n} instruments)")
    print(f"{'='*60}")
    print(f"  Total return   : {metrics['total_return_pct']:>+8.2f}%")
    print(f"  CAGR           : {metrics['cagr_pct']:>+8.2f}%")
    print(f"  Max drawdown   : {metrics['max_drawdown_pct']:>8.2f}%   [{ftmo_status}]")
    print(f"  Sharpe         : {metrics['sharpe_ratio']:>8.3f}")
    print(f"  Total trades   : {metrics['total_trades']:>8}")
    print(f"  Win rate       : {metrics['win_rate_pct']:>8.1f}%")
    print(f"  Profit factor  : {metrics['profit_factor']:>8.3f}")
    print(f"  Period         : {metrics['years']:.1f} years")
    print(f"  Trades/year    : {metrics['total_trades'] / metrics['years']:.1f}")
    print(f"{'='*60}")

    # FTMO challenge viability estimate
    trades_per_month = metrics["total_trades"] / (metrics["years"] * 12)
    ev_per_trade = (metrics["win_rate_pct"] / 100 * rr) - ((1 - metrics["win_rate_pct"] / 100) * 1)
    expected_monthly_pct = trades_per_month * ev_per_trade * 1.0  # at 1% risk
    print(f"\n  FTMO challenge estimate (at 1% risk/trade):")
    print(f"  Trades/month     : {trades_per_month:.1f}")
    print(f"  EV per trade     : {ev_per_trade:+.3f}R  ({ev_per_trade * 1:.2f}% at 1% risk)")
    print(f"  Expected/month   : {expected_monthly_pct:+.2f}%")
    print(f"  At 2% risk/trade : {expected_monthly_pct * 2:+.2f}%")
    note = "Viable for FTMO at 2% risk" if expected_monthly_pct * 2 >= 5 else "Challenge mode needs score=4"
    print(f"  Assessment       : {note}")
    print()


def _apply_circuit_breaker(returns: pd.Series, halt_pct: float) -> pd.Series:
    """Zero out returns for any bar where the day's cumulative loss exceeds halt_pct."""
    result = returns.copy()
    dates  = returns.index.date if hasattr(returns.index, 'date') else pd.to_datetime(returns.index).date

    current_date = None
    day_cumulative = 0.0
    halted = False

    for i in range(len(result)):
        d = dates[i] if hasattr(dates, '__getitem__') else dates.iloc[i]
        r = float(result.iloc[i])

        if d != current_date:
            current_date   = d
            day_cumulative = 0.0
            halted         = False

        if halted:
            result.iloc[i] = 0.0
            continue

        day_cumulative += r
        if day_cumulative < -halt_pct:
            halted = True
            result.iloc[i] = 0.0

    return result


def _quick_metrics(returns: pd.Series, initial_capital: float) -> dict:
    eq  = initial_capital * (1 + returns).cumprod()
    dd  = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret = (eq.iloc[-1] / initial_capital - 1) * 100
    std = returns.std()
    bpy = _estimate_bpy(returns)
    sh  = (returns.mean() / std * np.sqrt(bpy)) if std > 0 else 0.0
    pos = (returns != 0).sum()
    return {"ret": ret, "dd": dd, "sh": sh, "tr": int(pos)}


def _estimate_bpy(returns: pd.Series) -> float:
    idx = pd.to_datetime(returns.index)
    if len(idx) < 2:
        return 252.0
    span_days = (idx[-1] - idx[0]).days
    return len(returns) / (span_days / 365.25) if span_days > 0 else 252.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score",   type=int,   default=5)
    parser.add_argument("--fvg",     type=float, default=0.10)
    parser.add_argument("--rr",      type=float, default=2.5)
    parser.add_argument("--halt",    type=float, default=0.025)
    parser.add_argument("--capital", type=float, default=10_000)
    args = parser.parse_args()

    run_combined(
        score=args.score,
        fvg_atr=args.fvg,
        rr=args.rr,
        daily_halt=args.halt,
        initial_capital=args.capital,
    )

    # Also run challenge mode comparison
    print("\n--- Challenge mode (score=4, 2.5% daily halt) ---")
    run_combined(score=4, fvg_atr=0.10, rr=2.5, daily_halt=0.025)


if __name__ == "__main__":
    main()
