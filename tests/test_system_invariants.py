"""core.system_invariants — the safety envelope."""
import pytest

from core import system_invariants as inv
from core.system_invariants import INVARIANTS


def test_invariants_are_immutable():
    with pytest.raises(Exception):
        INVARIANTS.max_daily_loss_pct = 10.0


def test_envelope_values_preserve_ftmo_buffers():
    assert INVARIANTS.max_daily_loss_pct < 5.0          # FTMO daily 5%
    assert INVARIANTS.max_total_kill_pct < 10.0         # FTMO max 10%
    assert INVARIANTS.max_soft_dd_pct < INVARIANTS.max_total_kill_pct
    assert INVARIANTS.min_reward_risk >= 1.5            # OS hard floor
    assert INVARIANTS.require_human_approval_live is True
    assert INVARIANTS.require_hard_stop is True


def test_current_vps_config_is_inside_envelope():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).parent.parent / "config" / "config_vps.yaml").read_text())
    assert inv.validate_trade_config(cfg["trading"]) == []


@pytest.mark.parametrize("key,value", [
    ("daily_halt_pct", 4.5), ("total_kill_pct", 10.0), ("soft_dd_halt_pct", 9.6),
    ("risk_pct", 2.0), ("max_portfolio_risk_pct", 6.0), ("max_concurrent_trades", 20),
    ("max_daily_entries", 12), ("weekly_dd_halt_pct", 9.0), ("min_rr", 1.0),
])
def test_looser_config_is_violation_and_clamped(key, value):
    cfg = {key: value}
    assert inv.validate_trade_config(cfg)
    clamped, violations = inv.enforce_trade_config(cfg)
    assert violations
    assert inv.validate_trade_config(clamped) == []


def test_tighter_config_untouched():
    cfg = {"daily_halt_pct": 2.0, "risk_pct": 0.5, "min_rr": 2.0}
    clamped, violations = inv.enforce_trade_config(cfg)
    assert violations == [] and clamped == cfg


def test_soft_halt_must_precede_kill():
    clamped, violations = inv.enforce_trade_config({"soft_dd_halt_pct": 9.0, "total_kill_pct": 8.0})
    assert violations
    assert clamped["soft_dd_halt_pct"] < clamped["total_kill_pct"]


def test_clamp_risk_pct():
    assert inv.clamp_risk_pct(5.0) == INVARIANTS.max_risk_per_trade_pct
    assert inv.clamp_risk_pct(0.5) == 0.5


# ── approval ────────────────────────────────────────────────────────────────

def test_default_mode_is_live_and_fail_closed():
    p = inv.approval_policy({}, is_real_account=False)
    assert p.required and p.fail_closed and not p.allow_auto_approve
    assert p.mode == "live"


def test_unknown_mode_treated_as_live():
    assert inv.approval_policy({"execution_mode": "yolo"}, False).mode == "live"


def test_real_account_ignores_demo_autonomous():
    p = inv.approval_policy({"execution_mode": "demo_autonomous"}, is_real_account=True)
    assert p.required and p.fail_closed and not p.allow_auto_approve


def test_demo_autonomous_must_be_explicit():
    p = inv.approval_policy({"execution_mode": "demo_autonomous"}, is_real_account=False)
    assert not p.required and not p.fail_closed


# ── emergency halt ──────────────────────────────────────────────────────────

def test_emergency_halt_roundtrip(tmp_path):
    assert inv.is_emergency_halted(tmp_path) == (False, "")
    inv.set_emergency_halt("manual stop", tmp_path)
    halted, why = inv.is_emergency_halted(tmp_path)
    assert halted and why == "manual stop"
    assert inv.clear_emergency_halt(tmp_path)
    assert not inv.is_emergency_halted(tmp_path)[0]


def test_emergency_halt_env(tmp_path, monkeypatch):
    monkeypatch.setenv(inv.EMERGENCY_HALT_ENV, "1")
    assert inv.is_emergency_halted(tmp_path)[0]


def test_corrupt_halt_flag_counts_as_halted(tmp_path):
    (tmp_path / inv.EMERGENCY_HALT_FILE).write_text("{not json")
    assert inv.is_emergency_halted(tmp_path)[0]


# ── pre-order check ─────────────────────────────────────────────────────────

def test_pre_order_ok(tmp_path):
    r = inv.check_pre_order(1, entry=100.0, sl=99.0, tp=102.0, risk_pct=0.5, logs_dir=tmp_path)
    assert r.ok, r.reasons


@pytest.mark.parametrize("kwargs,code", [
    (dict(direction=1, entry=100.0, sl=None, tp=102.0, risk_pct=0.5), "NO_HARD_STOP"),
    (dict(direction=1, entry=100.0, sl=100.5, tp=102.0, risk_pct=0.5), "STOP_WRONG_SIDE"),
    (dict(direction=-1, entry=100.0, sl=99.5, tp=98.0, risk_pct=0.5), "STOP_WRONG_SIDE"),
    (dict(direction=1, entry=100.0, sl=99.0, tp=101.0, risk_pct=0.5), "RR_BELOW_FLOOR"),
    (dict(direction=1, entry=100.0, sl=99.0, tp=102.0, risk_pct=1.5), "RISK_ABOVE_CEILING"),
])
def test_pre_order_blocks(tmp_path, kwargs, code):
    r = inv.check_pre_order(logs_dir=tmp_path, **kwargs)
    assert not r.ok and any(code in x for x in r.reasons)


def test_pre_order_blocked_by_emergency_halt(tmp_path):
    inv.set_emergency_halt("test", tmp_path)
    r = inv.check_pre_order(1, 100.0, 99.0, 102.0, 0.5, logs_dir=tmp_path)
    assert not r.ok and "EMERGENCY_HALT" in r.reasons[0]


def test_no_tp_runner_is_allowed(tmp_path):
    assert inv.check_pre_order(1, 100.0, 99.0, None, 0.5, logs_dir=tmp_path).ok


# ── change classification ───────────────────────────────────────────────────

@pytest.mark.parametrize("name,old,new,expected", [
    ("BASE_RISK", 0.35, 0.5, "loosens"), ("BASE_RISK", 0.5, 0.35, "tightens"),
    ("MIN_RR", 1.5, 2.0, "tightens"), ("MIN_RR", 2.0, 1.6, "loosens"),
    ("CAP_CONCURRENT", 3, 5, "loosens"), ("SCORE_FLOOR", 65, 75, "neutral"),
    ("require_approval", True, False, "loosens"), ("DAILY_DD_LIMIT", 2.0, "abc", "loosens"),
])
def test_classify_param_change(name, old, new, expected):
    assert inv.classify_param_change(name, old, new) == expected


def test_any_automatic_risk_loosening_is_violation_even_inside_envelope():
    v = inv.risk_change_violations({"BASE_RISK": (0.35, 0.5)})   # 0.5 < 1.0 ceiling
    assert v and "loosening" in v[0].rule


def test_strategy_params_evolve_freely():
    assert inv.risk_change_violations({"SCORE_FLOOR": (65, 75), "EDGE_FILTER": (1.0, 1.2)}) == []


def test_fingerprint_stable():
    assert inv.fingerprint() == inv.fingerprint() and len(inv.fingerprint()) == 16
