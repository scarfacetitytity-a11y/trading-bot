"""Portfolio optimizer — CG solver + correlation monitor + VaR/CVaR.

Adapted from ruflo/plugins/ruflo-neural-trader (conjugateGradient, RiskDecision).
Three capabilities in one module:

  1. CorrelationMatrix   — rolling pairwise return correlations between instruments
  2. correlated_size_mult() — reduce/block entry when adding correlated exposure
  3. conjugate_gradient() — optimal weight allocation via CG (ruflo port)
  4. var_cvar()           — per-trade and portfolio VaR/CVaR (95%)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Conjugate Gradient solver (ruflo port) ────────────────────────────────────

def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> np.ndarray:
    """Solve Ax = b via conjugate gradient.

    Port of ruflo sublinear-adapter.ts conjugateGradient().
    A must be symmetric positive definite (covariance matrix qualifies).

    Returns x — the solution vector (portfolio weights before normalisation).
    """
    x = np.zeros_like(b, dtype=float)
    r = b - A @ x
    p = r.copy()
    r_dot = float(r @ r)

    for _ in range(max_iter):
        if r_dot < tol ** 2:
            break
        Ap    = A @ p
        alpha = r_dot / float(p @ Ap)
        x    += alpha * p
        r    -= alpha * Ap
        r_new = float(r @ r)
        beta  = r_new / r_dot
        p     = r + beta * p
        r_dot = r_new

    return x


# ── Correlation matrix with rolling returns ───────────────────────────────────

class CorrelationMatrix:
    """Maintains rolling M15 return series per instrument and computes pairwise
    Pearson correlations on demand.  Thread-safe read (replace-not-mutate).
    """

    def __init__(self, lookback: int = 100):
        self._lookback = lookback
        self._returns: dict[str, np.ndarray] = {}

    def update(self, symbol: str, closes: np.ndarray) -> None:
        """Feed the latest close series for an instrument."""
        if len(closes) < 2:
            return
        rets = np.diff(closes.astype(float)) / np.where(
            closes[:-1] != 0, closes[:-1].astype(float), 1.0
        )
        self._returns[symbol] = rets[-self._lookback:]

    def correlation(self, sym_a: str, sym_b: str) -> float:
        """Pearson correlation between sym_a and sym_b returns. 0.0 if unknown."""
        a = self._returns.get(sym_a)
        b = self._returns.get(sym_b)
        if a is None or b is None:
            return 0.0
        n = min(len(a), len(b))
        if n < 10:
            return 0.0
        try:
            c = np.corrcoef(a[-n:], b[-n:])
            v = float(c[0, 1])
            return 0.0 if np.isnan(v) else v
        except Exception:
            return 0.0

    def correlated_size_mult(
        self,
        new_symbol: str,
        open_positions: list[tuple[str, int]],   # [(symbol, direction), ...]
    ) -> tuple[float, str]:
        """Return (size_mult, reason) based on correlation with open positions.

        Thresholds (ruflo RiskDecision portfolioCorrelation):
          corr >= 0.85 → block (same exposure)
          corr >= 0.70 → half size
          corr >= 0.50 → 0.75x
          else         → 1.0x (no adjustment)
        """
        if not open_positions:
            return 1.0, ""

        max_corr = 0.0
        max_pair  = ""
        for sym, _ in open_positions:
            if sym == new_symbol:
                corr = 1.0
            else:
                corr = abs(self.correlation(new_symbol, sym))
            if corr > max_corr:
                max_corr = corr
                max_pair = sym

        if max_corr >= 0.85:
            return 0.0, f"correlation {max_corr:.2f} with {max_pair} — same exposure, blocked"
        if max_corr >= 0.70:
            return 0.5, f"correlation {max_corr:.2f} with {max_pair} — size halved"
        if max_corr >= 0.50:
            return 0.75, f"correlation {max_corr:.2f} with {max_pair} — size 0.75x"
        return 1.0, ""

    def optimal_weights(
        self,
        symbols: list[str],
        expected_returns: dict[str, float],
    ) -> dict[str, float]:
        """CG-optimal weight allocation across a universe of instruments.

        Args:
            symbols:          candidate instruments to allocate across
            expected_returns: symbol → expected R (e.g. from P(win) * avg_win_R)

        Returns:
            symbol → weight, normalised to sum to 1.0
        """
        n = len(symbols)
        if n == 0:
            return {}
        if n == 1:
            return {symbols[0]: 1.0}

        # Build correlation matrix (dampen off-diagonals by 0.5 for stability)
        C = np.eye(n)
        for i, si in enumerate(symbols):
            for j, sj in enumerate(symbols):
                if i != j:
                    C[i, j] = self.correlation(si, sj) * 0.5

        b = np.array([float(expected_returns.get(s, 0.1)) for s in symbols])

        try:
            w = conjugate_gradient(C, b)
        except Exception:
            w = b.copy()

        w = np.clip(w, 0, None)
        total = float(w.sum())
        if total < 1e-10:
            return {s: 1.0 / n for s in symbols}
        return {s: float(w[i] / total) for i, s in enumerate(symbols)}


# ── VaR / CVaR ───────────────────────────────────────────────────────────────

@dataclass
class RiskMetrics:
    var_95:   float   # Value-at-Risk 95% as fraction of account (negative = loss)
    cvar_95:  float   # Conditional VaR 95% (expected loss given breach)
    ev:       float   # Expected value per trade in R
    p_win:    float
    win_r:    float
    loss_r:   float


def var_cvar(
    p_win:    float,
    risk_pct: float,
    win_r:    float = 1.5,
    loss_r:   float = 1.0,
) -> RiskMetrics:
    """Compute VaR(95%) and CVaR(95%) for a single trade.

    Parametric approach (binary outcome distribution):
      - P(loss) > 5%  → VaR_95 = -loss_r * risk_pct (the full stop)
      - P(loss) <= 5% → VaR_95 = EV (trade expected to profit at 95th percentile)
      - CVaR_95       = expected loss when loss occurs = -loss_r * risk_pct

    Returns fractions of account equity (negative = loss).
    """
    p_loss = 1.0 - p_win
    ev     = p_win * win_r * risk_pct - p_loss * loss_r * risk_pct

    if p_loss > 0.05:
        var_95  = -loss_r * risk_pct
        cvar_95 = -loss_r * risk_pct
    else:
        var_95  = ev
        cvar_95 = -loss_r * risk_pct   # worst case even if rare

    return RiskMetrics(
        var_95=round(var_95, 5),
        cvar_95=round(cvar_95, 5),
        ev=round(ev, 5),
        p_win=p_win,
        win_r=win_r,
        loss_r=loss_r,
    )


def portfolio_var_cvar(
    trades: list[RiskMetrics],
    correlation_matrix: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Aggregate VaR/CVaR across multiple open trades.

    Uses square-root rule with correlation: σ_p = sqrt(w^T Σ w)
    where w = individual VaR vector and Σ = correlation matrix.

    Returns (portfolio_var_95, portfolio_cvar_95) as account fractions.
    """
    if not trades:
        return 0.0, 0.0

    # Individual VaR as positive loss amounts for the matrix
    v = np.array([abs(t.var_95) for t in trades])
    n = len(v)

    if correlation_matrix is not None and correlation_matrix.shape == (n, n):
        C = correlation_matrix
    else:
        C = np.eye(n)

    port_var  = float(np.sqrt(v @ C @ v))
    port_cvar = float(np.sqrt(np.array([abs(t.cvar_95) for t in trades]) @ C @
                               np.array([abs(t.cvar_95) for t in trades])))

    return -port_var, -port_cvar
