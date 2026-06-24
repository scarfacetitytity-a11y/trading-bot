"""Equity curve and drawdown charts for backtest results.

Usage
-----
    from backtests.plot import plot_result, plot_comparison
    plot_result(result)                          # single strategy
    plot_comparison([result1, result2, result3]) # overlay multiple
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent.parent / "logs" / "reports"


def plot_result(result, save: bool = False, show: bool = True) -> None:
    """Two-panel chart: equity curve (top) and drawdown (bottom)."""
    equity = result.equity
    times = _times(result)

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(f"{result.strategy_name}  |  {result.symbol}", fontsize=13)

    ax1.plot(times, equity, linewidth=1.2, color="#2196F3")
    ax1.set_ylabel("Equity")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.grid(True, alpha=0.3)

    m = result.metrics
    summary = (
        f"Return: {m['total_return_pct']:+.1f}%  "
        f"CAGR: {m['cagr_pct']:+.1f}%  "
        f"MaxDD: {m['max_drawdown_pct']:.1f}%  "
        f"Sharpe: {m['sharpe_ratio']:.2f}  "
        f"Trades: {m['total_trades']}  "
        f"WinRate: {m['win_rate_pct']:.1f}%"
    )
    ax1.set_title(summary, fontsize=9, color="#555555")

    ax2.fill_between(times, drawdown, 0, color="#F44336", alpha=0.5)
    ax2.plot(times, drawdown, linewidth=0.8, color="#F44336")
    ax2.set_ylabel("Drawdown %")
    ax2.set_ylim(top=0)
    ax2.grid(True, alpha=0.3)

    _format_xaxis(ax2)
    plt.tight_layout()

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / f"{result.symbol}_{result.strategy_name}.png"
        fig.savefig(path, dpi=150)
        print(f"Chart saved → {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_comparison(results: list, save: bool = False, show: bool = True) -> None:
    """Overlay equity curves (normalised to 100) for multiple results."""
    if not results:
        return

    symbol = results[0].symbol
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f"Strategy comparison  |  {symbol}", fontsize=13)

    times = _times(results[0])
    for result in results:
        equity = result.equity
        normalised = equity / equity.iloc[0] * 100
        ax.plot(times, normalised, linewidth=1.2, label=result.strategy_name)

    ax.axhline(100, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_ylabel("Normalised equity (start = 100)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _format_xaxis(ax)
    plt.tight_layout()

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        names = "_vs_".join(r.strategy_name for r in results)
        path = REPORTS_DIR / f"{symbol}_{names}.png"
        fig.savefig(path, dpi=150)
        print(f"Comparison chart saved → {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _times(result) -> pd.Series:
    try:
        return pd.to_datetime(result.signals.index)
    except Exception:
        return result.equity.index


def _format_xaxis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
