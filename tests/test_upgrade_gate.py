"""execution.upgrade_gate — what may reach live execution."""
import json
import subprocess

import pytest

from core import system_invariants as inv
from execution import upgrade_gate as gate


def _cfg(**trading):
    base = {"risk_pct": 0.75, "daily_halt_pct": 4.0, "soft_dd_halt_pct": 9.0, "total_kill_pct": 9.5}
    base.update(trading)
    return {"trading": base, "v3_cutover": {"live": ["XAUUSD"], "go_no_go_min_samples": 50}}


def test_clean_config_passes(tmp_path):
    r = gate.startup_check(_cfg(), logs_dir=tmp_path, check_git=False)
    assert r.violations == [] and not r.block_entries


def test_loose_config_clamped(tmp_path):
    r = gate.startup_check(_cfg(risk_pct=3.0, daily_halt_pct=6.0), logs_dir=tmp_path, check_git=False)
    assert r.trade_cfg["risk_pct"] == inv.INVARIANTS.max_risk_per_trade_pct
    assert r.trade_cfg["daily_halt_pct"] == inv.INVARIANTS.max_daily_loss_pct
    assert len(r.violations) == 2


def test_cutover_sample_bypass_closed(tmp_path):
    cfg = _cfg()
    cfg["v3_cutover"]["go_no_go_min_samples"] = 0
    r = gate.startup_check(cfg, logs_dir=tmp_path, check_git=False)
    assert r.v3_cfg["go_no_go_min_samples"] == inv.INVARIANTS.min_cutover_samples
    assert any("bypass" in v for v in r.violations)


def test_emergency_halt_blocks_entries(tmp_path):
    inv.set_emergency_halt("x", tmp_path)
    assert gate.startup_check(_cfg(), logs_dir=tmp_path, check_git=False).block_entries


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_dirty_protected_file_sets_emergency_halt(tmp_path):
    repo = tmp_path / "repo"
    (repo / "core").mkdir(parents=True)
    f = repo / "core" / "system_invariants.py"
    f.write_text("A = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    assert gate.protected_files_dirty(repo) == []
    f.write_text("A = 2\n")
    logs = tmp_path / "logs"
    r = gate.startup_check(_cfg(), repo_root=repo, logs_dir=logs)
    assert r.block_entries
    assert inv.is_emergency_halted(logs)[0]


def test_protected_dirty_none_without_git(tmp_path):
    assert gate.protected_files_dirty(tmp_path / "not-a-repo") is None


# ── lift proposals ──────────────────────────────────────────────────────────

LIFTS = {"fvg_present": 1.3, "ob_present": 1.2}


def test_small_sample_lifts_not_applied():
    acc, rej = gate.gate_lift_proposals({"confluences": {"fvg_present": 1.4}, "n_trades": 20}, LIFTS)
    assert acc == {} and "n=20" in rej[0]


def test_missing_sample_size_not_applied():
    acc, _ = gate.gate_lift_proposals({"confluences": {"fvg_present": 1.4}}, LIFTS)
    assert acc == {}


def test_quant_key_name_accepted():
    acc, _ = gate.gate_lift_proposals({"confluences": {"fvg_present": 1.4}, "n_trades_analysed": 40}, LIFTS)
    assert acc == {"fvg_present": 1.4}


def test_lift_bounds_and_unknown_keys():
    acc, rej = gate.gate_lift_proposals(
        {"confluences": {"fvg_present": 9.9, "nope": 1.0, "ob_present": True}, "n_trades": 50}, LIFTS)
    assert acc == {} and len(rej) == 3


# ── runtime risk ────────────────────────────────────────────────────────────

def test_telegram_risk_clamped():
    pct, msg = gate.gate_runtime_risk_change(0.75, 5.0)
    assert pct == inv.INVARIANTS.max_risk_per_trade_pct and "clamped" in msg


def test_telegram_risk_nonpositive_rejected():
    pct, msg = gate.gate_runtime_risk_change(0.75, 0)
    assert pct == 0.75 and msg.startswith("rejected")


def test_telegram_risk_within_envelope():
    assert gate.gate_runtime_risk_change(0.75, 0.5)[0] == 0.5


# ── structural proposals ────────────────────────────────────────────────────

def test_structural_proposal_recorded_once(tmp_path):
    led = tmp_path / "ledger.jsonl"
    kw = dict(title="Reduce X", observation="o", problem="p", hypothesis="h",
              changes={"MAX_REACH_ATR": (None, 5.0)}, kind="strategy", ledger_path=led)
    a = gate.record_structural_proposal(**kw)
    b = gate.record_structural_proposal(**kw)
    assert a.id == b.id and a.stage == "hypothesis"


def test_builder_prompt_lists_protected_files():
    rule = gate.protected_files_rule()
    for f in gate.PROTECTED_FILES:
        assert f in rule


def test_learning_loop_no_longer_edits_source(tmp_path, monkeypatch):
    from execution import learning_loop as ll
    monkeypatch.setattr(ll, "LOG_DIR", tmp_path)
    import execution.upgrade_gate as ug
    calls = []
    monkeypatch.setattr(ug, "record_structural_proposal",
                        lambda **kw: calls.append(kw) or type("P", (), {"id": "UPG-x", "stage": "hypothesis"})())
    from pathlib import Path
    ta = Path(ll.__file__).parent / "trade_analyzer.py"
    before = ta.read_text(encoding="utf-8")
    result = {"trades": 40, "_trades_raw": [], "proposals": [
        {"council": "11", "title": "Liquidity target hit rate 0%", "confidence": "high", "evidence": "e"}]}
    applied = ll.auto_apply(result)
    assert ta.read_text(encoding="utf-8") == before
    assert calls and any("proposal" in a.lower() for a in applied)
