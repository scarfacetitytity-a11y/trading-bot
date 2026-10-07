"""core.self_upgrade lifecycle + research.upgrade_loop with a fake evaluator."""
import pytest

from core.self_upgrade import (LifecycleError, PromotionCriteria, SelfUpgrade, Stage,
                               UpgradeLedger)
from research.upgrade_loop import Metrics, run_proposal


@pytest.fixture()
def su(tmp_path):
    return SelfUpgrade(ledger=UpgradeLedger(tmp_path / "ledger.jsonl"),
                       changelog=tmp_path / "CHANGELOG.md")


def _propose(su, changes=None, kind="parameter"):
    return su.propose(title="t", kind=kind, source="quant", observation="o",
                      problem="p", hypothesis="h", changes=changes or {"SCORE_FLOOR": (65, 75)})


def _evidence(su, p, is_=(0.40, 0.46, 200), oos=(0.38, 0.43, 80), stress=(0.10, 0.09)):
    su.start_experiment(p, {"design": "x"})
    su.record_backtest(p, {"metric": "pass_rate", "baseline": is_[0], "candidate": is_[1], "n_trades": is_[2]})
    su.record_oos(p, {"metric": "pass_rate", "baseline": oos[0], "candidate": oos[1], "n_trades": oos[2]})
    su.record_stress(p, {"baseline_blow_rate": stress[0], "candidate_blow_rate": stress[1]})
    return su.compare_to_baseline(p)


def test_requires_observation_problem_hypothesis(su):
    with pytest.raises(LifecycleError):
        su.propose(title="t", kind="parameter", source="x", observation="o", problem="", hypothesis="h")


def test_unknown_kind_rejected(su):
    with pytest.raises(LifecycleError):
        _propose(su, kind="vibes")


def test_happy_path_promotes_with_version_and_changelog(su, tmp_path):
    p = _propose(su)
    assert p.stage == Stage.HYPOTHESIS.value
    p = _evidence(su, p)
    p = su.validate_risk(p)
    p = su.review(p, "Jay", approve=True)
    p = su.promote(p)
    assert p.stage == Stage.DOCUMENTED.value
    assert p.version == "0.0.1" and p.invariants_fingerprint
    assert "SCORE_FLOOR" in (tmp_path / "CHANGELOG.md").read_text()
    assert su.ledger.get(p.id).stage == Stage.DOCUMENTED.value


def test_stages_cannot_be_skipped(su):
    p = _propose(su)
    with pytest.raises(LifecycleError):
        su.record_oos(p, {"baseline": 0, "candidate": 1, "n_trades": 100})


def test_cannot_promote_without_review(su):
    p = su.validate_risk(_evidence(su, _propose(su)))
    with pytest.raises(LifecycleError):
        su.promote(p)


@pytest.mark.parametrize("reviewer", ["", "Sage", "claude", "autoresearch"])
def test_agents_cannot_review(su, reviewer):
    p = su.validate_risk(_evidence(su, _propose(su)))
    with pytest.raises(LifecycleError):
        su.review(p, reviewer, approve=True)


def test_risk_loosening_rejected_automatically_at_proposal(su):
    p = _propose(su, changes={"BASE_RISK": (0.35, 0.6)})
    assert p.stage == Stage.REJECTED.value and "invariant" in p.decision


def test_disabling_approval_rejected(su):
    p = _propose(su, changes={"require_approval": (True, False)})
    assert p.stage == Stage.REJECTED.value


def test_risk_tightening_allowed(su):
    p = _propose(su, changes={"BASE_RISK": (0.5, 0.35)}, kind="risk_limit")
    assert p.stage == Stage.HYPOTHESIS.value


def test_single_backtest_is_not_enough(su):
    p = _propose(su)
    su.start_experiment(p, {})
    su.record_backtest(p, {"metric": "pass_rate", "baseline": 0.4, "candidate": 0.6, "n_trades": 500})
    ok, reasons = su.evaluate_evidence(p)
    assert not ok and "out-of-sample" in reasons[0]


def test_overfit_rejected(su):
    # Big in-sample gain, almost nothing survives out of sample.
    p = _evidence(su, _propose(su), is_=(0.40, 0.60, 300), oos=(0.40, 0.41, 100))
    assert p.stage == Stage.REJECTED.value and "overfit" in p.decision


def test_oos_degradation_rejected(su):
    p = _evidence(su, _propose(su), oos=(0.40, 0.35, 100))
    assert p.stage == Stage.REJECTED.value


def test_small_sample_rejected(su):
    p = _evidence(su, _propose(su), is_=(0.40, 0.46, 40), oos=(0.38, 0.43, 10))
    assert p.stage == Stage.REJECTED.value and "trades" in p.decision


def test_stress_blowup_rejected(su):
    p = _evidence(su, _propose(su), stress=(0.10, 0.18))
    assert p.stage == Stage.REJECTED.value and "stress" in p.decision


def test_rejected_is_terminal(su):
    p = su.reject(_propose(su), "no")
    with pytest.raises(LifecycleError):
        su.start_experiment(p, {})


def test_review_reject(su):
    p = su.validate_risk(_evidence(su, _propose(su)))
    p = su.review(p, "Jay", approve=False, note="not convinced")
    assert p.stage == Stage.REJECTED.value


def test_strategy_kind_bumps_minor(su):
    p = su.validate_risk(_evidence(su, _propose(su, kind="strategy")))
    p = su.promote(su.review(p, "Jay", True))
    assert p.version == "0.1.0"


def test_ledger_is_append_only(su):
    p = _propose(su)
    n1 = len(su.ledger.path.read_text().splitlines())
    su.start_experiment(p, {})
    assert len(su.ledger.path.read_text().splitlines()) == n1 + 1


# ── upgrade loop with a deterministic fake evaluator ────────────────────────

class FakeEvaluator:
    measurable_params = frozenset({"SCORE_FLOOR", "BASE_RISK", "CAP_CONCURRENT", "DAILY_DD_LIMIT"})

    def __init__(self, gain_is=0.06, gain_oos=0.05, stress_blow_delta=-0.01):
        self.g = {"is": gain_is, "oos": gain_oos}
        self.sd = stress_blow_delta

    def evaluate(self, params, split, stress=False):
        cand = params["SCORE_FLOOR"] != 65
        n = 300 if split == "is" else 120
        pr = 0.40 + (self.g[split] if cand else 0.0)
        blow = 0.10 + (self.sd if (cand and stress) else 0.0)
        return Metrics(pass_rate=pr, blow_rate=blow, n_trades=n, n_windows=500)


BASE = {"SCORE_FLOOR": 65, "BASE_RISK": 0.35, "CAP_CONCURRENT": 3, "DAILY_DD_LIMIT": 2.0, "MIN_RR": 1.5}


def test_loop_stops_at_risk_validated_awaiting_human(su):
    p = run_proposal(_propose(su), FakeEvaluator(), su, baseline=BASE)
    assert p.stage == Stage.RISK_VALIDATED.value
    assert p.reviewer is None and p.version is None


def test_loop_rejects_unmeasurable_params(su):
    p = _propose(su, changes={"MIN_GRADE": ("C", "B")})
    p = run_proposal(p, FakeEvaluator(), su, baseline=BASE)
    assert p.stage == Stage.REJECTED.value and "cannot measure" in p.decision


def test_loop_rejects_overfit(su):
    p = run_proposal(_propose(su), FakeEvaluator(gain_is=0.10, gain_oos=0.01), su, baseline=BASE)
    assert p.stage == Stage.REJECTED.value


def test_loop_rejects_stress_failure(su):
    p = run_proposal(_propose(su), FakeEvaluator(stress_blow_delta=0.05), su, baseline=BASE)
    assert p.stage == Stage.REJECTED.value


# ── HarnessEvaluator against the real FTMO Monte Carlo engine ───────────────

def _synthetic_trades(n=400, seed=3):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    entry = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 300, n)), unit="D")
    win = rng.random(n) < 0.42
    pnl = np.where(win, 0.010, -0.005)
    return pd.DataFrame({"entry_time": entry, "exit_time": entry + pd.Timedelta(hours=6),
                         "pnl_pct": pnl, "score": rng.integers(4, 7, n)}), 0.005


def test_harness_evaluator_runs_real_engine():
    from research.upgrade_loop import HarnessEvaluator
    ev = HarnessEvaluator(mc=60, seeds=(1, 2))
    ev._cache = {lvl: _synthetic_trades() for lvl in (4, 5, 6)}
    params = {"BASE_RISK": 0.35, "CAP_CONCURRENT": 3, "SCORE_FLOOR": 65, "DAILY_DD_LIMIT": 2.0}
    is_m, oos_m = ev.evaluate(params, "is"), ev.evaluate(params, "oos")
    assert is_m.n_trades == 280 and oos_m.n_trades == 120
    assert 0.0 <= is_m.pass_rate <= 1.0 and is_m.n_windows > 0
    stressed = ev.evaluate(params, "oos", stress=True)
    assert stressed.pass_rate <= oos_m.pass_rate + 1e-9


def test_harness_evaluator_end_to_end_proposal(su):
    from research.upgrade_loop import HarnessEvaluator
    ev = HarnessEvaluator(mc=60, seeds=(1, 2))
    ev._cache = {lvl: _synthetic_trades() for lvl in (4, 5, 6)}
    p = _propose(su, changes={"BASE_RISK": (0.35, 0.30)}, kind="risk_limit")
    p = run_proposal(p, ev, su, baseline=dict(BASE))
    # Whatever the verdict, it must be a legal terminal-or-waiting state with evidence recorded.
    assert p.stage in (Stage.REJECTED.value, Stage.RISK_VALIDATED.value)
    assert Stage.BACKTESTED.value in p.evidence and Stage.STRESS_TESTED.value in p.evidence
