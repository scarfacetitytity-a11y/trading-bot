"""Upgrade gate — stops unverified changes reaching live execution.

PROTECTED FILE.

The gate sits on every path by which behaviour can change at runtime:

  1. startup_check()          — live config is clamped to the invariant envelope,
                                the v3 cutover sample floor is enforced, and
                                uncommitted edits to protected safety files block
                                new entries (emergency halt).
  2. gate_lift_proposals()    — Quant / learning-loop Bayesian lift proposals are
                                applied only with a sufficient sample and within
                                bounds (ProbabilityModel calls this).
  3. gate_runtime_risk_change() — runtime risk edits (Telegram /risk) can only
                                tighten or stay inside the envelope.
  4. record_structural_proposal() — learning-loop structural fixes become
                                self-upgrade proposals instead of source edits.

CLI:
    python -m execution.upgrade_gate --check        # report, no changes
    python -m execution.upgrade_gate --halt "reason"
    python -m execution.upgrade_gate --clear-halt   # human only
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core import system_invariants as inv

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent

PROTECTED_FILES: tuple[str, ...] = (
    "core/system_invariants.py",
    "core/self_upgrade.py",
    "execution/upgrade_gate.py",
    "execution/risk_agent.py",
)


def protected_files_rule() -> str:
    """Instruction injected into autonomous agent prompts (Builder)."""
    files = ", ".join(PROTECTED_FILES)
    return (
        "SAFETY RULE — you must NOT modify these protected files: "
        f"{files}; nor any risk limit, approval setting or halt logic in config/*.yaml "
        "or execution/orchestrator.py (RiskGuard, halt checks, approval gate). "
        "If the fix needs such a change, do not make it: write an upgrade proposal "
        "(docs/SELF_UPGRADE.md) describing the problem and hypothesis instead. "
        "Uncommitted edits to protected files trigger an emergency halt at next start."
    )


# ── Protected-file integrity ─────────────────────────────────────────────────

def protected_files_dirty(repo_root: Path = _ROOT) -> Optional[list[str]]:
    """Protected files with uncommitted changes, or None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", *PROTECTED_FILES],
            cwd=str(repo_root), capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return [line[3:].strip() for line in out.stdout.splitlines() if line.strip()]


# ── Startup ──────────────────────────────────────────────────────────────────

@dataclass
class GateReport:
    trade_cfg: dict
    v3_cfg: dict
    violations: list[str] = field(default_factory=list)
    block_entries: bool = False
    notes: list[str] = field(default_factory=list)


def startup_check(cfg: dict, repo_root: Path = _ROOT, logs_dir: Optional[Path] = None,
                  check_git: bool = True, set_halt: bool = True) -> GateReport:
    """set_halt=False reports without writing the emergency halt (CLI --check)."""
    trade_cfg = dict(cfg.get("trading", {}) or {})
    v3_cfg = dict(cfg.get("v3_cutover", {}) or {})

    clamped, violations = inv.enforce_trade_config(trade_cfg)
    report = GateReport(trade_cfg=clamped, v3_cfg=v3_cfg,
                        violations=[str(v) for v in violations])

    samples = v3_cfg.get("go_no_go_min_samples")
    if samples is not None and int(samples) < inv.INVARIANTS.min_cutover_samples:
        report.violations.append(
            f"v3_cutover.go_no_go_min_samples={samples} < {inv.INVARIANTS.min_cutover_samples} "
            "(cutover validation bypass)")
        v3_cfg["go_no_go_min_samples"] = inv.INVARIANTS.min_cutover_samples

    if check_git:
        dirty = protected_files_dirty(repo_root)
        if dirty:
            reason = f"protected safety files modified without commit: {', '.join(dirty)}"
            if set_halt:
                inv.set_emergency_halt(reason, logs_dir)
            report.block_entries = True
            report.violations.append(reason)
        elif dirty is None:
            report.notes.append("git unavailable — protected-file integrity not verified")

    halted, why = inv.is_emergency_halted(logs_dir)
    if halted:
        report.block_entries = True
        report.notes.append(f"emergency halt active: {why}")
    return report


# ── Bayesian lift proposals ──────────────────────────────────────────────────

def gate_lift_proposals(data: dict, current_lifts: dict[str, float],
                        min_trades: Optional[int] = None) -> tuple[dict[str, float], list[str]]:
    """Return (accepted {key: lift}, rejection reasons).

    Previously any proposal file was applied (Quant routinely analysed only the
    last 20 trades; the learning loop wrote proposals from n >= 3)."""
    min_trades = inv.INVARIANTS.min_lift_proposal_trades if min_trades is None else min_trades
    n = data.get("n_trades", data.get("n_trades_analysed"))
    try:
        n = int(n) if n is not None else 0
    except (TypeError, ValueError):
        n = 0
    if n < min_trades:
        return {}, [f"sample n={n} < {min_trades}: proposal recorded, not applied"]
    accepted: dict[str, float] = {}
    rejected: list[str] = []
    for key, val in (data.get("confluences") or {}).items():
        if key not in current_lifts:
            rejected.append(f"{key}: unknown confluence")
            continue
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            rejected.append(f"{key}: non-numeric")
            continue
        cur = float(current_lifts[key])
        if not (0.5 * cur <= float(val) <= 2.0 * cur):
            rejected.append(f"{key}: {val} outside 0.5x–2x of current {cur}")
            continue
        accepted[key] = float(val)
    return accepted, rejected


# ── Runtime risk changes ─────────────────────────────────────────────────────

def gate_runtime_risk_change(current_pct: float, requested_pct: float) -> tuple[float, str]:
    """Telegram /risk: inside the envelope only. Returns (applied_pct, message)."""
    if requested_pct <= 0:
        return current_pct, f"rejected: risk must be > 0 (kept {current_pct:.2f}%)"
    applied = inv.clamp_risk_pct(requested_pct)
    if applied < requested_pct:
        return applied, (f"clamped to invariant ceiling {applied:.2f}% "
                         f"(requested {requested_pct:.2f}%)")
    return applied, f"risk_pct={applied:.2f}%"


# ── Learning-loop structural fixes ───────────────────────────────────────────

def record_structural_proposal(title: str, observation: str, problem: str, hypothesis: str,
                               changes: dict[str, tuple[Any, Any]], source: str = "learning_loop",
                               kind: str = "parameter", ledger_path: Optional[Path] = None):
    from core.self_upgrade import SelfUpgrade, Stage, UpgradeLedger
    su = SelfUpgrade(ledger=UpgradeLedger(ledger_path) if ledger_path else None)
    # The learning loop runs repeatedly; one open proposal per title is enough.
    terminal = {Stage.REJECTED.value, Stage.DOCUMENTED.value}
    for existing in su.ledger.all().values():
        if existing.title == title and existing.source == source and existing.stage not in terminal:
            return existing
    return su.propose(title=title, kind=kind, source=source, observation=observation,
                      problem=problem, hypothesis=hypothesis, changes=changes)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AiDEN upgrade gate")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--halt", metavar="REASON")
    g.add_argument("--clear-halt", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)

    if a.halt:
        print(f"emergency halt set: {inv.set_emergency_halt(a.halt)}")
        return 0
    if a.clear_halt:
        print("emergency halt cleared" if inv.clear_emergency_halt() else "no emergency halt was set")
        return 0

    from config.settings import load_config
    cfg = load_config(Path(a.config)) if a.config else load_config()
    r = startup_check(cfg, check_git=True, set_halt=False)
    for v in r.violations:
        print(f"VIOLATION  {v}")
    for n in r.notes:
        print(f"NOTE       {n}")
    print(f"block_entries={r.block_entries}  invariants={inv.fingerprint()}")
    return 1 if r.violations or r.block_entries else 0


if __name__ == "__main__":
    sys.exit(main())
