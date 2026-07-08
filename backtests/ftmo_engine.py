"""Fixed-risk FTMO backtesting engine.

Each trade risks risk_pct of current equity, sized by initial stop distance:
    R_multiple = direction * (exit_price - entry_price) / stop_distance
    trade_equity_return = R_multiple * risk_pct

FTMO Phase 1 rules checked per 30-day window:
    - Profit target  : +10% from window start equity
    - Max overall DD : -10% from window start equity (never go below)
    - Max daily DD   : -5%  from window start equity in a single day
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.base import Strategy

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


@dataclass
class FTMOResult:
    strategy_name: str
    symbol: str
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict

    def print_summary(self) -> None:
        m = self.metrics
        print(f"\n{'='*62}")
        print(f"  {self.strategy_name} | {self.symbol}")
        print(f"{'='*62}")
        print(f"  Period          : {m['years']:.1f} years")
        print(f"  Total return    : {m['total_return_pct']:+.2f}%")
        print(f"  Max drawdown    : {m['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe ratio    : {m['sharpe_ratio']:.3f}")
        print(f"  Total trades    : {m['total_trades']}")
        print(f"  Win rate        : {m['win_rate_pct']:.1f}%")
        print(f"  Avg R per trade : {m['avg_r']:+.3f}R")
        print(f"  Trades no stop  : {m['trades_no_stop']}")
        print(f"  FTMO 30d passes : {m['ftmo_passes']}/{m['ftmo_windows']} windows")
        print(f"  FTMO pass rate  : {m['ftmo_pass_rate_pct']:.1f}%")
        print(f"{'='*62}\n")


class FTMOEngine:
    """Fixed-risk backtest engine for a single strategy."""

    def __init__(
        self,
        strategy: Strategy,
        risk_pct: float = 0.01,
        initial_capital: float = 10_000,
        ftmo_target: float = 0.10,
        ftmo_max_dd: float = 0.10,
        ftmo_daily_dd: float = 0.05,
        ftmo_window_days: int = 30,
    ):
        self.strategy = strategy
        self.risk_pct = risk_pct
        self.initial_capital = initial_capital
        self.ftmo_target = ftmo_target
        self.ftmo_max_dd = ftmo_max_dd
        self.ftmo_daily_dd = ftmo_daily_dd
        self.ftmo_window_days = ftmo_window_days

    def run(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> FTMOResult:
        df = df.reset_index(drop=True)
        signals, stops = self.strategy.generate_signals_and_stops(df)

        close = df["close"].astype(float)
        times = pd.to_datetime(df["time"])

        trades = _extract_trades(signals, stops, close, times)
        equity = _build_equity(trades, close, times, self.initial_capital, self.risk_pct)
        daily  = _daily_equity(equity, times)
        metrics = _compute_metrics(
            trades, equity, daily, times,
            self.initial_capital, self.risk_pct,
            self.ftmo_target, self.ftmo_max_dd, self.ftmo_daily_dd, self.ftmo_window_days,
        )

        return FTMOResult(
            strategy_name=self.strategy.name,
            symbol=symbol,
            equity=equity,
            trades=trades,
            metrics=metrics,
        )

    @classmethod
    def load_and_run(
        cls,
        strategy: Strategy,
        symbol: str,
        timeframe: str,
        **kwargs,
    ) -> FTMOResult:
        path = PROCESSED_DIR / f"{symbol}_{timeframe.upper()}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No processed data: {symbol} {timeframe}")
        df = pd.read_csv(path)
        return cls(strategy, **kwargs).run(df, symbol=f"{symbol} {timeframe}")


class FTMOMultiEngine:
    """Run multiple strategies on a shared equity curve.

    Each strategy still risks risk_pct per trade, but all trade P&Ls
    are applied to the same running equity. Concurrent trades are allowed.
    """

    def __init__(
        self,
        configs: list,  # list of (strategy, symbol, timeframe)
        risk_pct: float = 0.01,
        initial_capital: float = 10_000,
        ftmo_target: float = 0.10,
        ftmo_max_dd: float = 0.10,
        ftmo_daily_dd: float = 0.05,
        ftmo_window_days: int = 30,
    ):
        self.configs = configs
        self.risk_pct = risk_pct
        self.initial_capital = initial_capital
        self.ftmo_target = ftmo_target
        self.ftmo_max_dd = ftmo_max_dd
        self.ftmo_daily_dd = ftmo_daily_dd
        self.ftmo_window_days = ftmo_window_days

    def run(self) -> FTMOResult:
        all_trades: list[pd.DataFrame] = []

        for strategy, symbol, timeframe in self.configs:
            path = PROCESSED_DIR / f"{symbol}_{timeframe.upper()}.csv"
            if not path.exists():
                print(f"  [skip] {symbol} {timeframe} — data not found")
                continue
            df = pd.read_csv(path).reset_index(drop=True)
            signals, stops = strategy.generate_signals_and_stops(df)
            close = df["close"].astype(float)
            times = pd.to_datetime(df["time"])
            trades = _extract_trades(signals, stops, close, times)
            if not trades.empty:
                trades["strategy"] = strategy.name
                trades["symbol"]   = symbol
                all_trades.append(trades)

        if not all_trades:
            empty = pd.DataFrame()
            return FTMOResult("MultiStrategy", "ALL", pd.Series(dtype=float), empty, {})

        combined = pd.concat(all_trades, ignore_index=True).sort_values("exit_time")

        # Build equity from chronological trade exits
        eq = self.initial_capital
        event_times  = [combined["exit_time"].iloc[0]]
        event_equity = [eq]
        for _, trade in combined.iterrows():
            eq *= (1 + trade["r_multiple"] * self.risk_pct)
            event_times.append(trade["exit_time"])
            event_equity.append(eq)

        equity = pd.Series(event_equity, index=range(len(event_equity)))
        daily  = _daily_equity_from_events(event_times, event_equity)

        times_series = pd.Series(event_times)
        metrics = _compute_metrics(
            combined, equity, daily, times_series,
            self.initial_capital, self.risk_pct,
            self.ftmo_target, self.ftmo_max_dd, self.ftmo_daily_dd, self.ftmo_window_days,
        )
        metrics["strategy_count"] = len(self.configs)

        return FTMOResult("MultiStrategy", "ALL", equity, combined, metrics)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_trades(
    signals: pd.Series,
    stops: pd.Series,
    close: pd.Series,
    times: pd.Series,
) -> pd.DataFrame:
    """Extract trades from signal transitions; initial stop from signal bar via shift."""
    positions   = signals.shift(1).fillna(0)
    entry_stops = stops.shift(1)  # stop level recorded at the signal bar

    trades   = []
    position = 0
    entry_bar   = None
    entry_price = None
    entry_stop  = None

    for i in range(len(positions)):
        pos = int(positions.iloc[i])
        if pos == position:
            continue

        if position != 0:
            exit_price = float(close.iloc[i])
            if entry_stop is not None and not np.isnan(entry_stop):
                stop_dist = abs(entry_price - entry_stop)
                min_dist  = abs(entry_price) * 0.0001  # 1 pip minimum
                if stop_dist > min_dist:
                    r = position * (exit_price - entry_price) / stop_dist
                    r = max(-15.0, min(30.0, r))  # cap extreme outliers
                else:
                    r = 0.0
            else:
                r = 0.0
            trades.append({
                "entry_time":  times.iloc[entry_bar],
                "exit_time":   times.iloc[i],
                "direction":   position,
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "stop_price":  entry_stop,
                "r_multiple":  r,
                "has_stop":    entry_stop is not None and not np.isnan(entry_stop),
            })
            position = 0

        if pos != 0:
            entry_bar   = i
            entry_price = float(close.iloc[i])
            s = entry_stops.iloc[i]
            entry_stop  = float(s) if not np.isnan(float(s)) else None
            position    = pos

    return pd.DataFrame(trades)


def _build_equity(
    trades: pd.DataFrame,
    close: pd.Series,
    times: pd.Series,
    initial_capital: float,
    risk_pct: float,
) -> pd.Series:
    equity_arr = np.full(len(close), initial_capital, dtype=float)
    if trades.empty:
        return pd.Series(equity_arr, index=close.index)

    time_to_idx = {t: i for i, t in enumerate(times)}
    eq = initial_capital
    for _, trade in trades.sort_values("exit_time").iterrows():
        exit_idx = time_to_idx.get(trade["exit_time"])
        if exit_idx is None:
            continue
        eq *= (1 + trade["r_multiple"] * risk_pct)
        equity_arr[exit_idx:] = eq

    return pd.Series(equity_arr, index=close.index)


def _daily_equity(equity: pd.Series, times: pd.Series) -> pd.Series:
    df = pd.DataFrame({"eq": equity.values, "date": pd.to_datetime(times).dt.date})
    return df.groupby("date")["eq"].last()


def _daily_equity_from_events(
    event_times: list,
    event_equity: list,
) -> pd.Series:
    dates = [pd.to_datetime(t).date() for t in event_times]
    s = pd.Series(event_equity, index=dates)
    return s.groupby(s.index).last()


def _compute_metrics(
    trades: pd.DataFrame,
    equity: pd.Series,
    daily: pd.Series,
    times,
    initial_capital: float,
    risk_pct: float,
    ftmo_target: float,
    ftmo_max_dd: float,
    ftmo_daily_dd: float,
    ftmo_window_days: int,
) -> dict:
    m = {}

    t0 = pd.to_datetime(times.iloc[0])
    t1 = pd.to_datetime(times.iloc[-1])
    m["years"] = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1 / 365)

    final = float(equity.iloc[-1])
    m["total_return_pct"] = (final / initial_capital - 1) * 100

    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    m["max_drawdown_pct"] = float(abs(dd.min()) * 100)

    if not trades.empty and "r_multiple" in trades.columns:
        r = trades["r_multiple"].dropna()
        m["total_trades"]   = len(r)
        m["win_rate_pct"]   = float((r > 0).mean() * 100)
        m["avg_r"]          = float(r.mean())
        std_r = r.std()
        m["sharpe_ratio"]   = float(r.mean() / std_r * np.sqrt(252)) if std_r > 0 else 0.0
        m["trades_no_stop"] = int((~trades["has_stop"]).sum()) if "has_stop" in trades.columns else 0
    else:
        m["total_trades"]   = 0
        m["win_rate_pct"]   = 0.0
        m["avg_r"]          = 0.0
        m["sharpe_ratio"]   = 0.0
        m["trades_no_stop"] = 0

    passes, total = _ftmo_windows(
        daily, initial_capital, ftmo_target, ftmo_max_dd, ftmo_daily_dd, ftmo_window_days,
    )
    m["ftmo_passes"]        = passes
    m["ftmo_windows"]       = total
    m["ftmo_pass_rate_pct"] = (passes / total * 100) if total > 0 else 0.0

    return m


def _ftmo_windows(
    daily: pd.Series,
    initial_capital: float,
    ftmo_target: float,
    ftmo_max_dd: float,
    ftmo_daily_dd: float,
    window_days: int,
) -> tuple:
    dates = list(daily.index)
    if len(dates) < 5:
        return 0, 0

    passes = 0
    total  = 0
    i = 0

    while i < len(dates):
        start_date = dates[i]
        end_date   = start_date + timedelta(days=window_days)

        w_idx = [j for j, d in enumerate(dates) if start_date <= d <= end_date]
        if len(w_idx) < 5:
            break

        window   = daily.iloc[w_idx]
        start_eq = float(window.iloc[0])

        profit_floor = start_eq * (1 + ftmo_target)
        dd_floor     = start_eq * (1 - ftmo_max_dd)
        daily_limit  = start_eq * ftmo_daily_dd

        hit_target = float(window.iloc[-1]) >= profit_floor
        max_dd_ok  = float(window.min()) >= dd_floor

        daily_ok = True
        prev = start_eq
        for v in window.values:
            if (prev - v) > daily_limit:
                daily_ok = False
                break
            prev = v

        total += 1
        if hit_target and max_dd_ok and daily_ok:
            passes += 1

        # Step to next non-overlapping window
        next_start = end_date + timedelta(days=1)
        nxt = next((j for j, d in enumerate(dates) if d >= next_start), None)
        if nxt is None:
            break
        i = nxt

    return passes, total
