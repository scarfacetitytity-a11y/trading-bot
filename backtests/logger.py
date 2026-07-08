"""Structured logging for all backtest runs.

Every run writes:
  logs/runs/{timestamp}_summary.json          — metadata + metrics
  logs/strategies/{strat}_{sym}_{tf}.csv      — cumulative per-strategy trades (append)
  logs/runs/{timestamp}_{strat}_{sym}_{tf}_trades.csv — per-run trade list
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

LOGS_DIR       = Path(__file__).resolve().parent.parent / "logs"
RUNS_DIR       = LOGS_DIR / "runs"
STRATEGY_DIR   = LOGS_DIR / "strategies"


def _ensure_dirs():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)


def log_trades(
    trades: pd.DataFrame,
    strategy_name: str,
    symbol: str,
    timeframe: str,
    run_ts: str | None = None,
):
    """Append trades to the cumulative strategy log and write a per-run file."""
    if trades.empty:
        return
    _ensure_dirs()
    if run_ts is None:
        run_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    safe_name = strategy_name.replace("/", "_").replace("\\", "_")[:50]
    safe_sym  = symbol.replace(".", "_")

    # Cumulative log — append
    cumulative = STRATEGY_DIR / f"{safe_name}_{safe_sym}_{timeframe}.csv"
    df = trades.copy()
    df["run_ts"] = run_ts
    df["strategy"] = strategy_name
    df["symbol"]   = symbol
    df["timeframe"] = timeframe

    if cumulative.exists():
        df.to_csv(cumulative, mode="a", header=False, index=False)
    else:
        df.to_csv(cumulative, index=False)

    # Per-run snapshot
    run_file = RUNS_DIR / f"{run_ts}_{safe_name}_{safe_sym}_{timeframe}_trades.csv"
    trades.to_csv(run_file, index=False)


def log_run_summary(run_type: str, data, extra: dict | None = None):
    """Write a JSON summary for the whole run."""
    _ensure_dirs()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    payload = {
        "run_type":  run_type,
        "timestamp": ts,
        "extra":     extra or {},
    }

    if isinstance(data, pd.DataFrame):
        payload["results"] = data.to_dict(orient="records")
    elif isinstance(data, dict):
        payload["metrics"] = data

    out = RUNS_DIR / f"{ts}_{run_type}_summary.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"  [log] Run summary saved → {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")
    return out
