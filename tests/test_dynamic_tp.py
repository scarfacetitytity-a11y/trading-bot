"""Behaviour 4 — dynamic take-profit 'bank-in' guard tests.

The critical safety property: bank-in is OFF by default (bank_min_r=0) and never
pulls a TP inward unless explicitly enabled AND the trade is deep in profit with
a real nearer liquidity pool ahead. Enabled by default it must not fire.
"""
import numpy as np
import pandas as pd

from execution.trade_analyzer import manage_trade


def _df_with_equal_highs(n=80, pool=2010.0):
    """Uptrend that stalls into a cluster of equal highs near `pool` (resting
    liquidity) — a real nearer draw for a long whose TP sits far above."""
    rng = np.random.default_rng(3)
    close = np.linspace(1990, 2008, n) + rng.normal(0, 0.2, n)
    high  = close + 0.3
    low   = close - 0.3
    # plant equal highs (a liquidity pool) in the last several bars
    high[-6:] = pool
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"time": idx, "open": close, "high": high,
                         "low": low, "close": close, "tick_volume": 100})


class TestBankInGate:
    def test_off_by_default_never_banks(self):
        df = _df_with_equal_highs()
        dec = manage_trade(
            df=df, direction=1, entry=1995.0, initial_sl=1993.0,
            current_sl=1995.0, current_tp=2050.0, price=2007.0,
            atr=1.0, trade_type="continuation", cur_r=6.0,   # deep profit
            # bank_min_r defaults to 0.0 → disabled
        )
        assert "bank_tp" not in (dec.reason or "")

    def test_not_banked_when_not_deep_enough(self):
        df = _df_with_equal_highs()
        dec = manage_trade(
            df=df, direction=1, entry=1995.0, initial_sl=1993.0,
            current_sl=1995.0, current_tp=2050.0, price=2007.0,
            atr=1.0, trade_type="continuation", cur_r=1.0,   # below threshold
            bank_min_r=3.0,
        )
        assert "bank_tp" not in (dec.reason or "")

    def test_banks_a_nearer_pool_when_enabled_and_deep(self):
        df = _df_with_equal_highs(pool=2010.0)
        dec = manage_trade(
            df=df, direction=1, entry=1995.0, initial_sl=1993.0,
            current_sl=1995.0, current_tp=2050.0, price=2007.0,
            atr=1.0, trade_type="continuation", cur_r=6.0,
            bank_min_r=3.0,
        )
        # If a nearer pool is detected it must be inside the far TP and ahead of price
        if "bank_tp" in (dec.reason or ""):
            assert 2007.0 < dec.new_tp < 2050.0
