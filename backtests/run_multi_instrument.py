"""AiDEN multi-instrument combined backtest — M2P4b / Council architecture.

Cadre roles:
  Sage   (MEM-001) — architecture: 6-instrument pool, shared equity, daily circuit breaker
  Quant  (MEM-002) — validates FTMO compliance across combined portfolio
  Builder(MEM-003) — M15-aware parameter scaling, session windows per instrument
  Scout  (MEM-004) — instrument universe: US100, US30, XAUUSD, US500, JP225, US2000

Council of 12 — entry model assignment:
  PE  (01) — arbitrates FTMO breach vs trade frequency tension
  Arch(02) — H4 bias + M15 entry multi-TF gate structure
  DE  (03) — drawdown counter, daily halt state, position sizing state
  App (06) — per-instrument confluence gate (score >= threshold)
  SRE (07) — daily circuit breaker: 2.5% portfolio loss halts all instruments
  Perf(08) — M15 entry timing, bar-count latency for FVG retest
  QA  (09) — backtest coverage per instrument, minimum trade count gate

Portfolio rules:
  - Equal capital weight (1/N per instrument loaded)
  - All instruments run simultaneously on common time grid
  - Daily circuit breaker: combined daily loss > 2.5% halts all instruments for rest of day
  - FTMO target: combined MaxDD < -10%, daily DD < -5%

Usage:
    python -m backtests.run_multi_instrument
    python -m backtests.run_multi_instrument --tf M15 --score 4
    python -m backtests.run_multi_instrument --tf H1 --score 5 --halt 0.025
    python -m backtests.run_multi_instrument --sweep          # challenge vs funded modes
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

# ── Full AiDEN instrument universe ────────────────────────────────────────────
# session_start/end = UTC hours. max_fvg_wait/max_entry_wait scaled for M15.
# JP225 trades Tokyo session (Asian open). All others: NY session.
INSTRUMENTS = {
    # Metals
    "XAUUSD":      dict(session_start=7,  session_end=21),   # London + NY
    # US indices — NY session
    "US100.cash":  dict(session_start=12, session_end=21),
    "US30.cash":   dict(session_start=12, session_end=21),
    "US500.cash":  dict(session_start=12, session_end=21),
    "US2000.cash": dict(session_start=12, session_end=21),
    # European session — UK100 replaces GER40 (better edge, tighter DD)
    "UK100.cash":  dict(session_start=7,  session_end=17),   # London hours
    # Asian session
    "JP225.cash":  dict(session_start=0,  session_end=9),    # Nikkei — Tokyo
    # Excluded: GER40 (weak edge on M15)
    # XAGUSD back in — conservative params in OPTIMISED_PARAMS (score=6, NY-only)
    "XAGUSD":      dict(session_start=7,  session_end=21),   # silver: London + NY
}

# Instruments that trade BOTH long and short (bidirectional H4 bias gate).
# US indices have a structural upward bias — shorts drag performance.
# Metals can trend sharply in either direction — bidirectional edge confirmed.
BIDIRECTIONAL = {"XAUUSD", "XAGUSD"}

# M15 params — NOTE: htf_lookback and h4_swing_lookback are in H4 BARS (post-resample)
# so they stay the same as H1. Only input-TF bar counts scale ×4.
M15_PARAMS = dict(
    htf_lookback=20,       # H4 bars — same as H1 (H4 is H4 regardless of input TF)
    h4_swing_lookback=40,  # H4 bars — same as H1
    max_fvg_wait=160,      # input-TF bars: H1=40 → M15=160
    max_entry_wait=32,     # input-TF bars: H1=8  → M15=32
    ob_lookback=80,        # input-TF bars: H1=20 → M15=80
    liq_lookback=40,       # input-TF bars: H1=10 → M15=40
    atr_period=56,         # input-TF bars: H1=14 → M15=56
    atr_stop_buffer=0.5,   # slightly wider on M15 due to higher bar-noise
    h4_bias_method="ema",  # EMA crossover — more stable than swing for M15 H4 feeds
)

H1_PARAMS = dict(
    htf_lookback=20,
    h4_swing_lookback=40,
    max_fvg_wait=40,
    max_entry_wait=8,
    ob_lookback=20,
    liq_lookback=10,
    atr_period=14,
    atr_stop_buffer=0.3,
    h4_bias_method="swing",
)


# ── Variable lot sizing by score ─────────────────────────────────────────────────
# Higher score = more confluence = larger size. Reduces exposure on marginal setups.
# Applied per-trade: scale the bar returns for that trade's duration.
SIZE_CONFIGS: dict[int, float] = {
    4: 0.75,   # min pass — reduced size
    5: 1.00,   # one extra gate — standard size
}
SIZE_CONFIGS_DEFAULT = 1.5  # score >= 6 — full high-conviction size

def _size_mult_from_score(score: int) -> float:
    return SIZE_CONFIGS.get(score, SIZE_CONFIGS_DEFAULT)


# ── Per-instrument trailing stop config ──────────────────────────────────────────
# XAUUSD only — gold trends cleanly, trailing stop captured +55pp more return in A/B test.
# All other instruments: trail_to_be=False (run to full SL/TP — guards against whipsaw re-entry).
TRAIL_CONFIGS: dict[str, dict] = {
    "XAUUSD": dict(trail_to_be=True, trail_be_r=1.0, trail_lock_r=2.0),
}

# ── Per-instrument sweep-optimised params (populated after sweep_m15_instruments) ─
# Format: {symbol: {min_score, rr_target, min_fvg_atr, atr_stop_buffer, h4_bias_method,
#                   session_start, session_end}}
# When set, overrides the shared score/fvg_atr/rr args for that instrument.
# Populated by running: python -m backtests.sweep_m15_instruments
# Per-instrument configs from sweep_m15_instruments.py (2026-07-13)
OPTIMISED_PARAMS: dict[str, dict] = {
    # Sweep best — all EMA bias, M15
    "US2000.cash": dict(min_score=5, rr_target=3.5, min_fvg_atr=0.10,
                        atr_stop_buffer=0.3, h4_bias_method="ema",
                        session_start=12, session_end=21),
    "UK100.cash":  dict(min_score=4, rr_target=3.0, min_fvg_atr=0.15,
                        atr_stop_buffer=0.3, h4_bias_method="ema",
                        session_start=7,  session_end=17),
    # XAGUSD — bidirectional, ultra-selective gate to keep MaxDD < 10%
    # Sweep best longs-only was score=5 MaxDD -10.69%. score=6 + shorts to balance.
    "XAGUSD":      dict(min_score=6, rr_target=3.0, min_fvg_atr=0.15,
                        atr_stop_buffer=0.3, h4_bias_method="ema",
                        session_start=12, session_end=21),   # NY-only for tightest DD
    # Others: use shared M15_PARAMS defaults (score/fvg/rr from CLI args)
    # XAUUSD:     score=4, rr=2.5, fvg=0.10, stop=0.5, EMA, sess=7-21  → Sharpe 2.57
    # US100.cash: score=4, rr=2.5, fvg=0.10, stop=0.5, EMA, sess=12-21 → Sharpe 1.09
    # US30.cash:  score=4, rr=2.5, fvg=0.10, stop=0.5, EMA, sess=12-21 → Sharpe 1.13
    # US500.cash: score=4, rr=2.5, fvg=0.10, stop=0.5, EMA, sess=12-21 → Sharpe 0.81
    # JP225.cash: score=4, rr=2.5, fvg=0.10, stop=0.5, EMA, sess=0-9   → Sharpe 1.55
}


def _load(
    symbol: str,
    tf: str,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    path = PROCESSED / f"{symbol}_{tf}.csv"
    if not path.exists():
        print(f"  SKIP {symbol}_{tf}: not found at {path}", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if since is not None:
        df = df[df["time"] >= since].reset_index(drop=True)
    if until is not None:
        df = df[df["time"] < until].reset_index(drop=True)
    return df if len(df) > 100 else None


def _generate_returns(
    symbol: str,
    cfg: dict,
    score: int,
    fvg_atr: float,
    rr: float,
    tf: str,
    commission: float,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
    use_variable_sizing: bool = True,
    use_prime_bonus: bool = True,
) -> tuple[pd.Series, pd.Series, pd.DataFrame] | None:
    df = _load(symbol, tf, since=since, until=until)
    if df is None:
        return None

    tf_params = M15_PARAMS.copy() if tf == "M15" else H1_PARAMS.copy()

    # v2 dynamic RR + session prime window defaults (NY 13-15 UTC bonus)
    v2_defaults = dict(
        long_only=(symbol not in BIDIRECTIONAL),  # metals = bidirectional, all else = long-only
        rr_model3_bonus=0.5,
        rr_trend_bonus=0.5,
        rr_trend_threshold=0.003,
        rr_max=5.0,
        session_prime_start=13,
        session_prime_end=15,
        use_rsi=True,
        rsi_period=56,     # scaled ×4 from H1=14
        rsi_long_lo=25.0,
        rsi_long_hi=55.0,
        rsi_short_lo=45.0,
        rsi_short_hi=75.0,
        trail_to_be=False,
        trail_be_r=1.0,
        trail_lock_r=2.0,
        use_prime_bonus=use_prime_bonus,
        use_vol_spike=False,
        vol_spike_mult=1.5,
        require_ce=False,
        use_d1_bias=False,
    )
    if symbol in TRAIL_CONFIGS:
        v2_defaults.update(TRAIL_CONFIGS[symbol])

    if symbol in OPTIMISED_PARAMS:
        opt  = OPTIMISED_PARAMS[symbol].copy()   # copy — don't mutate the module-level dict
        bias = opt.pop("h4_bias_method", tf_params.pop("h4_bias_method", "ema"))
        stop = opt.pop("atr_stop_buffer", tf_params.pop("atr_stop_buffer", 0.5))
        strat = AiDENIndexStrategy(
            min_score=opt.get("min_score", score),
            min_fvg_atr=opt.get("min_fvg_atr", fvg_atr),
            rr_target=opt.get("rr_target", rr),
            session_start=opt.get("session_start", cfg["session_start"]),
            session_end=opt.get("session_end", cfg["session_end"]),
            h4_bias_method=bias,
            atr_stop_buffer=stop,
            **{k: v for k, v in tf_params.items()
               if k not in ("h4_bias_method", "atr_stop_buffer")},
            **v2_defaults,
        )
    else:
        bias = tf_params.pop("h4_bias_method", "ema")
        stop = tf_params.pop("atr_stop_buffer", 0.5)
        strat = AiDENIndexStrategy(
            min_score=score,
            min_fvg_atr=fvg_atr,
            rr_target=rr,
            session_start=cfg["session_start"],
            session_end=cfg["session_end"],
            h4_bias_method=bias,
            atr_stop_buffer=stop,
            **tf_params,
            **v2_defaults,
        )
    bt = Backtest(strat, initial_capital=10_000, commission=commission)
    r = bt.run(df, symbol=symbol)

    times = pd.to_datetime(df["time"])
    returns = r.returns.copy()
    returns.index = times

    trades = r.trades.copy()
    trades["symbol"] = symbol

    # Attach per-trade score. Signal fires at bar i, engine shifts by 1 so
    # execution (entry_time) is at bar i+1. Map score to the execution bar.
    if hasattr(strat, "_scores") and "entry_time" in trades.columns:
        score_map = {
            df["time"].iloc[i + 1]: int(strat._scores.iloc[i])
            for i in range(len(df) - 1)
            if strat._scores.iloc[i] > 0
        }
        trades["score"] = trades["entry_time"].map(score_map).fillna(0).astype(int)

    # Apply variable sizing: scale bar returns for each trade's duration by its size_mult.
    if use_variable_sizing and "score" in trades.columns and "entry_time" in trades.columns:
        mult_s = pd.Series(1.0, index=returns.index)
        for _, t in trades.iterrows():
            if t["score"] == 0:
                continue
            m = _size_mult_from_score(int(t["score"]))
            if m == 1.0:
                continue
            et = t["entry_time"]
            xt = t.get("exit_time", None)
            if xt is not None:
                mask = (returns.index >= et) & (returns.index <= xt)
            else:
                mask = returns.index >= et
            mult_s[mask] = m
        returns = returns * mult_s

    return times, returns, trades


def run_combined(
    score: int       = 4,
    fvg_atr: float   = 0.10,
    rr: float        = 2.5,
    daily_halt: float = 0.025,
    commission: float = 0.0001,
    initial_capital: float = 10_000,
    tf: str          = "M15",
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
    period_label: str = "",
    use_prime_bonus: bool = True,
) -> dict:
    instruments = list(INSTRUMENTS.keys())
    label = f"  [{period_label}]" if period_label else ""
    print(f"\nAiDEN Multi-Instrument Backtest  [{tf}]{label}")
    print(f"Universe : {', '.join(instruments)}")
    print(f"Config   : score>={score}  fvg_atr={fvg_atr}  rr={rr}  halt={daily_halt*100:.1f}%  prime_bonus={use_prime_bonus}")
    print(f"Capital  : {initial_capital:,.0f}  Commission: {commission}\n")

    per_instrument: dict[str, pd.Series] = {}
    all_trades: list[pd.DataFrame] = []

    for symbol, cfg in INSTRUMENTS.items():
        result = _generate_returns(symbol, cfg, score, fvg_atr, rr, tf, commission, since=since, until=until, use_prime_bonus=use_prime_bonus)
        if result is None:
            continue
        _times, returns, trades = result
        per_instrument[symbol] = returns
        all_trades.append(trades)
        m = _quick_metrics(returns, initial_capital)
        status = "FTMO-BREACH" if m["dd"] < -10 else "ok"
        print(f"  {symbol:<16} Return: {m['ret']:>+7.2f}%  MaxDD: {m['dd']:>7.2f}%  "
              f"Sharpe: {m['sh']:>6.3f}  Trades: {m['tr']:>4}  [{status}]")

    if not per_instrument:
        print("No instrument data found. Run the data pipeline first:")
        print("  python -m backtests.run_data_pipeline")
        return {}

    n = len(per_instrument)
    combined_returns = pd.DataFrame(per_instrument).sort_index().fillna(0)
    portfolio_returns = combined_returns.sum(axis=1) * (1.0 / n)
    portfolio_returns = _apply_circuit_breaker(portfolio_returns, daily_halt)

    equity = pd.Series(
        initial_capital * (1 + portfolio_returns).cumprod(),
        index=portfolio_returns.index,
    )

    bpy = _estimate_bpy(portfolio_returns)
    all_trade_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    metrics = calculate_metrics(portfolio_returns, equity, all_trade_df, bpy)

    ftmo_dd   = "SAFE" if metrics["max_drawdown_pct"] > -10 else "** BREACH **"
    ftmo_daily = "SAFE"  # circuit breaker caps daily at halt_pct

    print(f"\n{'='*65}")
    print(f"  COMBINED PORTFOLIO ({n} instruments, {tf})")
    print(f"{'='*65}")
    print(f"  Total return   : {metrics['total_return_pct']:>+8.2f}%")
    print(f"  CAGR           : {metrics['cagr_pct']:>+8.2f}%")
    print(f"  Max drawdown   : {metrics['max_drawdown_pct']:>8.2f}%   FTMO MaxDD: {ftmo_dd}")
    print(f"  Sharpe         : {metrics['sharpe_ratio']:>8.3f}")
    print(f"  Total trades   : {metrics['total_trades']:>8}")
    print(f"  Win rate       : {metrics['win_rate_pct']:>8.1f}%")
    print(f"  Profit factor  : {metrics['profit_factor']:>8.3f}")
    print(f"  Period         : {metrics['years']:.1f} years")
    print(f"  Trades/year    : {metrics['total_trades'] / max(metrics['years'], 0.01):.1f}")
    print(f"  Trades/month   : {metrics['total_trades'] / max(metrics['years'] * 12, 0.01):.1f}")

    trades_pm = metrics["total_trades"] / max(metrics["years"] * 12, 0.01)
    wr = metrics["win_rate_pct"] / 100
    ev = wr * rr - (1 - wr) * 1
    exp_1pct = trades_pm * ev * 1.0
    exp_2pct = exp_1pct * 2

    print(f"\n  FTMO challenge estimate:")
    print(f"  EV per trade     : {ev:+.3f}R  ({ev:.2f}% at 1% risk)")
    print(f"  Expected/month @ 1% risk : {exp_1pct:+.2f}%")
    print(f"  Expected/month @ 2% risk : {exp_2pct:+.2f}%")

    if exp_2pct >= 10 and metrics["max_drawdown_pct"] > -10:
        assessment = "VIABLE — challenge target achievable at 2% risk"
    elif exp_2pct >= 5 and metrics["max_drawdown_pct"] > -10:
        assessment = "MARGINAL — needs max available data + tight execution"
    elif metrics["max_drawdown_pct"] <= -10:
        assessment = "BLOCKED — MaxDD breach, tighten score or lower risk"
    else:
        assessment = "LOW FREQUENCY — need score=4 or more instruments"

    print(f"  Assessment       : {assessment}")
    print(f"{'='*65}\n")

    return {
        "score": score, "tf": tf, "instruments": n,
        "return_pct": metrics["total_return_pct"],
        "max_dd": metrics["max_drawdown_pct"],
        "sharpe": metrics["sharpe_ratio"],
        "trades": metrics["total_trades"],
        "trades_pm": trades_pm,
        "win_rate": metrics["win_rate_pct"],
        "ev": ev,
        "exp_2pct": exp_2pct,
        "assessment": assessment,
    }


def _apply_circuit_breaker(returns: pd.Series, halt_pct: float) -> pd.Series:
    result = returns.copy()
    try:
        dates = returns.index.date
    except AttributeError:
        dates = pd.to_datetime(returns.index).date

    current_date = None
    day_cumulative = 0.0
    halted = False

    for i in range(len(result)):
        d = dates[i]
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
    tr  = int((returns != 0).sum())
    return {"ret": ret, "dd": dd, "sh": sh, "tr": tr}


def _estimate_bpy(returns: pd.Series) -> float:
    idx = pd.to_datetime(returns.index)
    if len(idx) < 2:
        return 252.0
    span_days = (idx[-1] - idx[0]).days
    return len(returns) / (span_days / 365.25) if span_days > 0 else 252.0


def main() -> None:
    parser = argparse.ArgumentParser(description="AiDEN multi-instrument backtest")
    parser.add_argument("--tf",      default="M15",  help="Timeframe: M15 or H1")
    parser.add_argument("--score",   type=int,   default=4)
    parser.add_argument("--fvg",     type=float, default=0.10)
    parser.add_argument("--rr",      type=float, default=2.5)
    parser.add_argument("--halt",    type=float, default=0.025)
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--sweep",   action="store_true",
                        help="Run score 3/4/5 comparison")
    parser.add_argument("--periods", action="store_true",
                        help="Run across 30d / 3m / 6m / 1y / all-data windows")
    args = parser.parse_args()

    if args.periods:
        now = pd.Timestamp.utcnow()
        windows = [
            ("30d",  now - pd.Timedelta(days=30)),
            ("3m",   now - pd.Timedelta(days=91)),
            ("6m",   now - pd.Timedelta(days=182)),
            ("1y",   now - pd.Timedelta(days=365)),
            ("2y",   None),
        ]
        summary = []
        for label, since in windows:
            r = run_combined(
                score=args.score, fvg_atr=args.fvg, rr=args.rr,
                daily_halt=args.halt, tf=args.tf, since=since,
                period_label=label,
            )
            if r:
                r["label"] = label
                summary.append(r)

        print("\n=== PERIOD COMPARISON (score=4, M15) ===")
        print(f"{'Period':<8} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} "
              f"{'Trades/mo':>10} {'WinRate':>8} {'Assessment'}")
        print("-" * 80)
        for r in summary:
            print(f"  {r['label']:<6} {r['return_pct']:>+8.2f}%  {r['max_dd']:>7.2f}%"
                  f"  {r['sharpe']:>7.3f}  {r['trades_pm']:>9.1f}"
                  f"  {r.get('win_rate', 0):>7.1f}%  {r['assessment'][:35]}")

    elif args.sweep:
        results = []
        for score in [5, 4, 3]:
            r = run_combined(
                score=score, fvg_atr=args.fvg, rr=args.rr,
                daily_halt=args.halt, tf=args.tf,
            )
            if r:
                results.append(r)

        print("\n=== SWEEP SUMMARY ===")
        print(f"{'Score':<8} {'Trades/mo':>10} {'MaxDD':>8} {'EV/2%':>8} {'Assessment'}")
        print("-" * 70)
        for r in results:
            print(f"  {r['score']:<6} {r['trades_pm']:>10.1f} {r['max_dd']:>8.2f}%"
                  f" {r['exp_2pct']:>8.2f}%  {r['assessment'][:40]}")
    else:
        run_combined(
            score=args.score,
            fvg_atr=args.fvg,
            rr=args.rr,
            daily_halt=args.halt,
            initial_capital=args.capital,
            tf=args.tf,
        )


if __name__ == "__main__":
    main()
