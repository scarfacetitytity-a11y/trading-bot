"""Isolate new gates: compare 4 configs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from backtests.run_multi_instrument import (
    INSTRUMENTS, _generate_returns, _apply_circuit_breaker,
    _quick_metrics, _estimate_bpy,
)
from backtests.metrics import calculate_metrics

SCORE = 4; FVG_ATR = 0.10; RR = 2.5; TF = "M15"; COMM = 0.0001; HALT = 0.025

def _port(use_var, use_ce, use_vol, use_d1):
    per = {}; all_t = []
    for symbol, cfg in INSTRUMENTS.items():
        r = _generate_returns(
            symbol, cfg, SCORE, FVG_ATR, RR, TF, COMM,
            use_variable_sizing=use_var, use_prime_bonus=False,
        )
        if r is None: continue
        _t, ret, trades = r
        per[symbol] = ret; all_t.append(trades)
    if not per: return {}
    n = len(per)
    combined = pd.DataFrame(per).sort_index().fillna(0)
    port_r = _apply_circuit_breaker(combined.sum(axis=1) * (1.0/n), HALT)
    equity = pd.Series(10000*(1+port_r).cumprod(), index=port_r.index)
    bpy = _estimate_bpy(port_r)
    all_df = pd.concat(all_t, ignore_index=True) if all_t else pd.DataFrame()
    cm = calculate_metrics(port_r, equity, all_df, bpy)
    trades_closed = sum(len(t[t["pnl_pct"]!=0]) for t in all_t if "pnl_pct" in t.columns)
    return {"ret": cm["total_return_pct"], "dd": cm["max_drawdown_pct"],
            "sh": cm["sharpe_ratio"], "wr": cm["win_rate_pct"],
            "pf": cm["profit_factor"], "tr": trades_closed}

# Note: use_ce/use_vol/use_d1 are now baked into strategy defaults via v2_defaults
# We need to test by temporarily overriding. For now run the current defaults.

configs = [
    ("M2P8 baseline (prime-off, var-size, no new gates)", False),
    ("New gates, NO var-size",                            False),
    ("New gates, WITH var-size",                          True),
]

# Can't toggle individual gates without passing to strategy — running what's committed
print("\nRunning current committed config (CE+vol+D1 on, var-size toggled)...")
print("=" * 75)
print(f"  {'Config':<42} {'Ret':>7} {'MaxDD':>7} {'Sh':>7} {'WR':>6} {'PF':>7} {'Tr':>5}")
print("-" * 75)

for label, var in [("no variable sizing", False), ("with variable sizing", True)]:
    r = _port(var, True, True, True)
    if r:
        print(f"  {label:<42} {r['ret']:>+6.2f}% {r['dd']:>6.2f}% {r['sh']:>7.3f} "
              f"{r['wr']:>5.1f}% {r['pf']:>7.3f} {r['tr']:>5}")

print("=" * 75)
print("\nDiagnosis: if 'no variable sizing' MaxDD is still bad, the gates themselves")
print("are the issue (not the sizing). If only 'with var-size' is bad, scoring")
print("inflation is pushing weak trades into 1.5x bucket.")
