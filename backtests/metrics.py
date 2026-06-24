"""Performance metrics calculated from a backtest's returns and trade log."""
import numpy as np
import pandas as pd


def calculate_metrics(
    returns: pd.Series,
    equity: pd.Series,
    trades: pd.DataFrame,
    bars_per_year: float,
) -> dict:
    """Return a dict of performance statistics.

    Parameters
    ----------
    returns:       per-bar strategy returns (fraction, not %)
    equity:        equity curve starting at initial_capital
    trades:        DataFrame from engine._extract_trades()
    bars_per_year: annualisation factor (derived from actual data timestamps)
    """
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

    n_bars = len(returns)
    years = n_bars / bars_per_year
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    std = returns.std()
    sharpe = (returns.mean() / std * np.sqrt(bars_per_year)) if std > 0 else 0.0

    metrics = {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "total_bars": n_bars,
        "years": round(years, 2),
    }

    if trades.empty:
        metrics.update({
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "avg_trade_pct": 0.0,
        })
        return metrics

    wins = trades[trades["pnl_pct"] > 0]["pnl_pct"]
    losses = trades[trades["pnl_pct"] <= 0]["pnl_pct"]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    metrics.update({
        "total_trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
        "profit_factor": round(profit_factor, 3),
        "avg_win_pct": round(wins.mean() * 100, 3) if len(wins) else 0.0,
        "avg_loss_pct": round(losses.mean() * 100, 3) if len(losses) else 0.0,
        "avg_trade_pct": round(trades["pnl_pct"].mean() * 100, 3),
    })

    return metrics
