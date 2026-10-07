"""Controlled upgrade loop: experiment → backtest → OOS → stress → compare → risk → (human review).

Connects the self-upgrade lifecycle (core/self_upgrade.py) to the existing
autoresearch / FTMO Monte Carlo infrastructure (backtests/run_ftmo_sim_adaptive).
It never promotes anything itself: the furthest it can take a proposal is
RISK_VALIDATED, where it waits for a named human reviewer.

Why not program.md's loop as-is: that loop keeps a commit whenever in-sample
challenge_pass_rate improves on one 6-month window, may raise BASE_RISK /
DAILY_DD_LIMIT / CAP_CONCURRENT, and research/backtest_harness.py only consumes
BASE_RISK, CAP_CONCURRENT and SCORE_FLOOR — so changes to MIN_RR, MIN_GRADE,
EDGE_FILTER, SL_BUFFER_ATR, COUSIN_BLOCK_HOURS and DAILY_DD_LIMIT were "kept"
or "discarded" on pure Monte Carlo noise. This loop:

  * rejects changes to parameters the evaluator cannot measure;
  * rejects any loosening of a risk parameter before spending compute;
  * splits trades chronologically (in-sample / out-of-sample);
  * stress-tests with a per-trade cost shock and multiple seeds;
  * requires the PromotionCriteria evidence bar.

Usage:
    python -m research.upgrade_loop propose --title "Raise score floor" \\
        --param SCORE_FLOOR=75 --observation "..." --problem "..." --hypothesis "..."
    python -m research.upgrade_loop run UPG-...
    python -m research.upgrade_loop review UPG-... --reviewer Jay --approve
    python -m research.upgrade_loop list
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.self_upgrade import SelfUpgrade, Stage, UpgradeProposal  # noqa: E402


@dataclass
class Metrics:
    pass_rate: float
    blow_rate: float          # max_dd + daily_dd breaches
    n_trades: int
    n_windows: int

    def as_dict(self) -> dict:
        return {"pass_rate": self.pass_rate, "blow_rate": self.blow_rate,
                "n_trades": self.n_trades, "n_windows": self.n_windows}


class Evaluator(Protocol):
    measurable_params: frozenset

    def evaluate(self, params: dict, split: str, stress: bool = False) -> Metrics: ...


def load_params(path: Path = _ROOT / "research" / "params.py") -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("aiden_research_params", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {k: getattr(mod, k) for k in dir(mod) if k.isupper()}


def _score_level(score_floor: float) -> int:
    # Same mapping as research/backtest_harness.py.
    if score_floor >= 85:
        return 6
    if score_floor >= 75:
        return 5
    return 4


class HarnessEvaluator:
    """Wraps backtests.run_ftmo_sim_adaptive with a chronological IS/OOS split,
    a hard daily halt (so DAILY_DD_LIMIT is actually measured) and a stress mode."""

    measurable_params = frozenset({"BASE_RISK", "CAP_CONCURRENT", "SCORE_FLOOR", "DAILY_DD_LIMIT"})

    def __init__(self, oos_fraction: float = 0.3, mc: int = 1000, seeds: tuple[int, ...] = (7, 11, 23),
                 stress_cost_frac: float = 0.15):
        self.oos_fraction = oos_fraction
        self.mc = mc
        self.seeds = seeds
        self.stress_cost_frac = stress_cost_frac
        self._cache: dict[int, tuple[Any, float]] = {}

    def _trades(self, level: int):
        if level not in self._cache:
            from backtests.run_ftmo_sim_adaptive import _load_trades_with_scores
            self._cache[level] = _load_trades_with_scores(level)
        return self._cache[level]

    def evaluate(self, params: dict, split: str, stress: bool = False) -> Metrics:
        import numpy as np
        import pandas as pd
        from backtests.run_ftmo_sim_adaptive import WINDOW_DAYS, _simulate_challenge

        trades, avg_loss = self._trades(_score_level(float(params["SCORE_FLOOR"])))
        cut = int(len(trades) * (1 - self.oos_fraction))
        part = trades.iloc[:cut] if split == "is" else trades.iloc[cut:]
        part = part.reset_index(drop=True)
        if stress:
            part = part.copy()
            # Cost shock: every trade pays an extra fraction of an average loss
            # (slippage/spread widening), which hits marginal edges hardest.
            part["pnl_pct"] = part["pnl_pct"] - self.stress_cost_frac * avg_loss

        base = float(params["BASE_RISK"])
        cap = float(params["CAP_CONCURRENT"]) * base
        daily = float(params.get("DAILY_DD_LIMIT", 0.0)) / 100.0
        ent = part["entry_time"].values
        first, last = part["entry_time"].min(), part["entry_time"].max()
        span = max(1, ((last - pd.Timedelta(days=WINDOW_DAYS)) - first).days)

        results = []
        for seed in self.seeds:
            rng = random.Random(seed)
            for _ in range(self.mc // len(self.seeds)):
                start = first + pd.Timedelta(days=rng.randint(0, span))
                end = start + pd.Timedelta(days=WINDOW_DAYS)
                lo = np.searchsorted(ent, start.to_datetime64())
                hi = np.searchsorted(ent, end.to_datetime64())
                if hi - lo < 3:
                    continue
                results.append(_simulate_challenge(part.iloc[lo:hi], avg_loss, base, cap, end,
                                                   adaptive=True, daily_halt=daily)["result"])
        n = len(results) or 1
        return Metrics(
            pass_rate=sum(r == "pass" for r in results) / n,
            blow_rate=sum(r in ("max_dd", "daily_dd") for r in results) / n,
            n_trades=len(part), n_windows=len(results),
        )


def run_proposal(p: UpgradeProposal, evaluator: Evaluator, su: SelfUpgrade,
                 baseline: Optional[dict] = None) -> UpgradeProposal:
    """Drive a proposal from HYPOTHESIS to RISK_VALIDATED (or REJECTED)."""
    if p.stage_enum is not Stage.HYPOTHESIS:
        raise ValueError(f"{p.id} is at {p.stage}; run_proposal starts at hypothesis")
    baseline = dict(baseline or load_params())

    unmeasurable = sorted(set(p.changes) - set(evaluator.measurable_params))
    if unmeasurable:
        return su.reject(p, f"evaluator cannot measure {unmeasurable}; no evidence is possible — "
                            "extend the evaluator first")

    candidate = dict(baseline)
    for k, (_old, new) in p.change_pairs().items():
        candidate[k] = new

    su.start_experiment(p, {"evaluator": type(evaluator).__name__, "baseline": baseline,
                            "candidate": candidate, "metric": "pass_rate",
                            "split": "chronological IS/OOS"})
    b_is, c_is = evaluator.evaluate(baseline, "is"), evaluator.evaluate(candidate, "is")
    su.record_backtest(p, {"metric": "pass_rate", "baseline": b_is.pass_rate, "candidate": c_is.pass_rate,
                           "n_trades": c_is.n_trades, "baseline_detail": b_is.as_dict(),
                           "candidate_detail": c_is.as_dict()})
    b_oos, c_oos = evaluator.evaluate(baseline, "oos"), evaluator.evaluate(candidate, "oos")
    su.record_oos(p, {"metric": "pass_rate", "baseline": b_oos.pass_rate, "candidate": c_oos.pass_rate,
                      "n_trades": c_oos.n_trades, "baseline_detail": b_oos.as_dict(),
                      "candidate_detail": c_oos.as_dict()})
    b_st, c_st = evaluator.evaluate(baseline, "oos", stress=True), evaluator.evaluate(candidate, "oos", stress=True)
    su.record_stress(p, {"baseline_blow_rate": b_st.blow_rate, "candidate_blow_rate": c_st.blow_rate,
                         "baseline_pass_rate": b_st.pass_rate, "candidate_pass_rate": c_st.pass_rate,
                         "scenario": "OOS + per-trade cost shock"})
    p = su.compare_to_baseline(p)
    if p.stage_enum is Stage.REJECTED:
        return p
    return su.validate_risk(p)


def _parse_param(s: str) -> tuple[str, Any]:
    k, _, v = s.partition("=")
    try:
        return k.strip(), float(v) if "." in v else int(v)
    except ValueError:
        return k.strip(), v.strip()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AiDEN controlled upgrade loop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("propose")
    pr.add_argument("--title", required=True)
    pr.add_argument("--param", action="append", default=[], help="NAME=value (research/params.py)")
    pr.add_argument("--observation", required=True)
    pr.add_argument("--problem", required=True)
    pr.add_argument("--hypothesis", required=True)
    pr.add_argument("--source", default="human")
    rn = sub.add_parser("run")
    rn.add_argument("id")
    rv = sub.add_parser("review")
    rv.add_argument("id")
    rv.add_argument("--reviewer", required=True)
    g = rv.add_mutually_exclusive_group(required=True)
    g.add_argument("--approve", action="store_true")
    g.add_argument("--reject", action="store_true")
    rv.add_argument("--note", default="")
    sub.add_parser("list")
    a = ap.parse_args(argv)

    su = SelfUpgrade()
    if a.cmd == "propose":
        current = load_params()
        changes = {}
        for raw in a.param:
            k, v = _parse_param(raw)
            changes[k] = (current.get(k), v)
        p = su.propose(title=a.title, kind="parameter", source=a.source, observation=a.observation,
                       problem=a.problem, hypothesis=a.hypothesis, changes=changes)
        print(f"{p.id}  {p.stage}  {p.decision or ''}")
    elif a.cmd == "run":
        p = run_proposal(su.ledger.get(a.id), HarnessEvaluator(), su)
        print(f"{p.id}  {p.stage}  {p.decision or 'awaiting human review'}")
    elif a.cmd == "review":
        p = su.review(su.ledger.get(a.id), a.reviewer, approve=a.approve, note=a.note)
        if p.stage_enum is Stage.REVIEWED:
            p = su.promote(p)
        print(f"{p.id}  {p.stage}  v{p.version or '-'}  {p.decision or ''}")
    else:
        for p in su.ledger.all().values():
            print(f"{p.id}  {p.stage:<18} {p.kind:<10} {p.title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
