"""AiDEN self-upgrade lifecycle.

PROTECTED FILE (see execution/upgrade_gate.PROTECTED_FILES).

Every improvement — whether proposed by Quant, Sage, the learning loop,
autoresearch or a human — moves through one ordered lifecycle:

  OBSERVED → PROBLEM_IDENTIFIED → HYPOTHESIS → EXPERIMENT → BACKTESTED
  → OOS_TESTED → STRESS_TESTED → BASELINE_COMPARED → RISK_VALIDATED
  → REVIEWED → PROMOTED → VERSIONED → DOCUMENTED
                         ↘ REJECTED (terminal, from any stage)

Rules this module enforces (not just documents):
  * stages cannot be skipped;
  * risk validation auto-rejects any change that loosens a risk limit
    (core.system_invariants.risk_change_violations);
  * promotion requires in-sample AND out-of-sample AND stress evidence that
    meets PromotionCriteria — one good backtest is never enough;
  * promotion requires a named human reviewer; agents cannot review;
  * the ledger is append-only JSONL, so every decision is auditable and a
    promotion can be rolled back by reverting the commit that applied it.

Promotion records the decision. Applying a promoted parameter change to live
config is a human commit — the self-upgrade system never edits config itself.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from core import system_invariants as inv

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER = _ROOT / "research" / "upgrades" / "ledger.jsonl"
DEFAULT_CHANGELOG = _ROOT / "research" / "upgrades" / "CHANGELOG.md"

AGENT_REVIEWERS = {"sage", "quant", "builder", "scout", "claude", "agent", "autoresearch", "learning_loop"}


class Stage(str, Enum):
    OBSERVED = "observed"
    PROBLEM_IDENTIFIED = "problem_identified"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    BACKTESTED = "backtested"
    OOS_TESTED = "oos_tested"
    STRESS_TESTED = "stress_tested"
    BASELINE_COMPARED = "baseline_compared"
    RISK_VALIDATED = "risk_validated"
    REVIEWED = "reviewed"
    PROMOTED = "promoted"
    VERSIONED = "versioned"
    DOCUMENTED = "documented"
    REJECTED = "rejected"


ORDER: list[Stage] = [s for s in Stage if s is not Stage.REJECTED]

KINDS = {"parameter", "strategy", "probability_model", "feature", "data_source", "code", "risk_limit"}


class LifecycleError(RuntimeError):
    pass


@dataclass
class PromotionCriteria:
    """Evidence bar. Defaults follow the OS: 'all parameters are provisional
    until validated by 50+ trade sample' and 'opinions do not change rules'."""
    min_is_trades: int = 100
    min_oos_trades: int = 50
    min_is_improvement: float = 0.0        # candidate - baseline on primary metric, must be >
    min_oos_improvement: float = 0.0       # must also improve out of sample
    min_oos_retention: float = 0.5         # OOS gain must keep >= 50% of IS gain (overfit guard)
    max_stress_blow_increase: float = 0.0  # stressed blow rate may not exceed baseline's
    higher_is_better: bool = True


@dataclass
class UpgradeProposal:
    title: str
    kind: str
    source: str
    observation: str
    problem: str = ""
    hypothesis: str = ""
    changes: dict[str, list[Any]] = field(default_factory=dict)   # name -> [old, new]
    id: str = field(default_factory=lambda: f"UPG-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:6]}")
    stage: str = Stage.OBSERVED.value
    evidence: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    decision: Optional[str] = None
    reviewer: Optional[str] = None
    version: Optional[str] = None
    invariants_fingerprint: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def stage_enum(self) -> Stage:
        return Stage(self.stage)

    def change_pairs(self) -> dict[str, tuple[Any, Any]]:
        return {k: (v[0], v[1]) for k, v in self.changes.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "UpgradeProposal":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class UpgradeLedger:
    """Append-only JSONL. The latest line for an id is its current state."""

    def __init__(self, path: Path = DEFAULT_LEDGER):
        self.path = Path(path)

    def append(self, p: UpgradeProposal) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(p), default=str) + "\n")

    def all(self) -> dict[str, UpgradeProposal]:
        out: dict[str, UpgradeProposal] = {}
        if not self.path.exists():
            return out
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = UpgradeProposal.from_dict(json.loads(line))
                out[p.id] = p
        return out

    def get(self, proposal_id: str) -> UpgradeProposal:
        p = self.all().get(proposal_id)
        if p is None:
            raise KeyError(proposal_id)
        return p

    def current_version(self) -> str:
        versions = [p.version for p in self.all().values() if p.version]
        if not versions:
            return "0.0.0"
        return max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))


def _bump(version: str, kind: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    if kind in ("strategy", "risk_limit"):
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


class SelfUpgrade:
    def __init__(self, ledger: Optional[UpgradeLedger] = None,
                 criteria: Optional[PromotionCriteria] = None,
                 changelog: Path = DEFAULT_CHANGELOG):
        self.ledger = ledger or UpgradeLedger()
        self.criteria = criteria or PromotionCriteria()
        self.changelog = Path(changelog)

    # ── creation ────────────────────────────────────────────────────────────
    def propose(self, *, title: str, kind: str, source: str, observation: str,
                problem: str, hypothesis: str,
                changes: Optional[dict[str, tuple[Any, Any]]] = None) -> UpgradeProposal:
        if kind not in KINDS:
            raise LifecycleError(f"unknown kind {kind!r}; expected one of {sorted(KINDS)}")
        if not (observation and problem and hypothesis):
            raise LifecycleError("observation, problem and hypothesis are all required")
        p = UpgradeProposal(title=title, kind=kind, source=source, observation=observation,
                            changes={k: [o, n] for k, (o, n) in (changes or {}).items()})
        self._move(p, Stage.OBSERVED, "created")
        p.problem = problem
        self._move(p, Stage.PROBLEM_IDENTIFIED, problem)
        p.hypothesis = hypothesis
        self._move(p, Stage.HYPOTHESIS, hypothesis)
        # Fail fast: a proposal that loosens risk never reaches an experiment.
        if self._risk_violations(p):
            return p
        self.ledger.append(p)
        return p

    # ── evidence stages ─────────────────────────────────────────────────────
    def start_experiment(self, p: UpgradeProposal, design: dict) -> UpgradeProposal:
        return self._record(p, Stage.EXPERIMENT, design)

    def record_backtest(self, p: UpgradeProposal, metrics: dict) -> UpgradeProposal:
        return self._record(p, Stage.BACKTESTED, metrics)

    def record_oos(self, p: UpgradeProposal, metrics: dict) -> UpgradeProposal:
        return self._record(p, Stage.OOS_TESTED, metrics)

    def record_stress(self, p: UpgradeProposal, metrics: dict) -> UpgradeProposal:
        return self._record(p, Stage.STRESS_TESTED, metrics)

    def compare_to_baseline(self, p: UpgradeProposal) -> UpgradeProposal:
        verdict, reasons = self.evaluate_evidence(p)
        p.evidence[Stage.BASELINE_COMPARED.value] = {"pass": verdict, "reasons": reasons}
        if not verdict:
            return self.reject(p, "baseline comparison failed: " + "; ".join(reasons))
        return self._record(p, Stage.BASELINE_COMPARED, p.evidence[Stage.BASELINE_COMPARED.value])

    def validate_risk(self, p: UpgradeProposal) -> UpgradeProposal:
        if self._risk_violations(p):
            return p
        return self._record(p, Stage.RISK_VALIDATED,
                            {"violations": [], "invariants": inv.fingerprint()})

    # ── human decision ──────────────────────────────────────────────────────
    def review(self, p: UpgradeProposal, reviewer: str, approve: bool, note: str = "") -> UpgradeProposal:
        if not reviewer or reviewer.strip().lower() in AGENT_REVIEWERS:
            raise LifecycleError("review requires a named human reviewer; agents cannot approve upgrades")
        self._require_stage(p, Stage.RISK_VALIDATED)
        p.reviewer = reviewer.strip()
        if not approve:
            return self.reject(p, f"rejected in review by {p.reviewer}: {note}")
        return self._record(p, Stage.REVIEWED, {"reviewer": p.reviewer, "note": note})

    def promote(self, p: UpgradeProposal) -> UpgradeProposal:
        self._require_stage(p, Stage.REVIEWED)
        # Re-check at promotion time: evidence or invariants may have changed.
        verdict, reasons = self.evaluate_evidence(p)
        if not verdict:
            return self.reject(p, "evidence no longer sufficient at promotion: " + "; ".join(reasons))
        if self._risk_violations(p):
            return p
        p.decision = "promoted"
        p.invariants_fingerprint = inv.fingerprint()
        self._move(p, Stage.PROMOTED, f"approved by {p.reviewer}")
        p.version = _bump(self.ledger.current_version(), p.kind)
        self._move(p, Stage.VERSIONED, p.version)
        self._document(p)
        self._move(p, Stage.DOCUMENTED, str(self.changelog))
        self.ledger.append(p)
        return p

    def reject(self, p: UpgradeProposal, reason: str) -> UpgradeProposal:
        p.decision = f"rejected: {reason}"
        self._move(p, Stage.REJECTED, reason)
        self.ledger.append(p)
        return p

    # ── evidence evaluation ─────────────────────────────────────────────────
    def evaluate_evidence(self, p: UpgradeProposal) -> tuple[bool, list[str]]:
        c = self.criteria
        reasons: list[str] = []
        bt = p.evidence.get(Stage.BACKTESTED.value)
        oos = p.evidence.get(Stage.OOS_TESTED.value)
        st = p.evidence.get(Stage.STRESS_TESTED.value)
        if not bt:
            return False, ["missing in-sample backtest"]
        if not oos:
            return False, ["missing out-of-sample test — one backtest is never enough"]
        if not st:
            return False, ["missing stress test"]
        sign = 1.0 if c.higher_is_better else -1.0
        is_gain = sign * (bt["candidate"] - bt["baseline"])
        oos_gain = sign * (oos["candidate"] - oos["baseline"])
        if bt.get("n_trades", 0) < c.min_is_trades:
            reasons.append(f"in-sample trades {bt.get('n_trades', 0)} < {c.min_is_trades}")
        if oos.get("n_trades", 0) < c.min_oos_trades:
            reasons.append(f"OOS trades {oos.get('n_trades', 0)} < {c.min_oos_trades}")
        if is_gain <= c.min_is_improvement:
            reasons.append(f"no in-sample improvement ({is_gain:+.4f})")
        if oos_gain <= c.min_oos_improvement:
            reasons.append(f"no out-of-sample improvement ({oos_gain:+.4f})")
        if is_gain > 0 and oos_gain < c.min_oos_retention * is_gain:
            reasons.append(f"overfit: OOS gain {oos_gain:+.4f} < {c.min_oos_retention:.0%} of IS gain {is_gain:+.4f}")
        blow_inc = st["candidate_blow_rate"] - st["baseline_blow_rate"]
        if blow_inc > c.max_stress_blow_increase:
            reasons.append(f"stress blow rate +{blow_inc:.4f} vs baseline")
        return (not reasons), reasons

    # ── internals ───────────────────────────────────────────────────────────
    def _risk_violations(self, p: UpgradeProposal) -> list[inv.Violation]:
        violations = inv.risk_change_violations(p.change_pairs())
        if violations:
            p.evidence[Stage.RISK_VALIDATED.value] = {"violations": [str(v) for v in violations]}
            self.reject(p, "invariant violation: " + "; ".join(str(v) for v in violations))
        return violations

    def _record(self, p: UpgradeProposal, stage: Stage, payload: dict) -> UpgradeProposal:
        p.evidence[stage.value] = payload
        self._move(p, stage, "")
        self.ledger.append(p)
        return p

    def _require_stage(self, p: UpgradeProposal, stage: Stage) -> None:
        if p.stage_enum is not stage:
            raise LifecycleError(f"{p.id} is at {p.stage}; expected {stage.value}")

    def _move(self, p: UpgradeProposal, stage: Stage, note: str) -> None:
        current = p.stage_enum
        if current is Stage.REJECTED:
            raise LifecycleError(f"{p.id} was rejected; open a new proposal instead")
        if stage is not Stage.REJECTED:
            if not p.history and stage is Stage.OBSERVED:
                pass
            elif ORDER.index(stage) != ORDER.index(current) + 1:
                raise LifecycleError(f"{p.id}: cannot move {current.value} → {stage.value} (stages cannot be skipped)")
        p.stage = stage.value
        p.history.append({"stage": stage.value, "ts": datetime.now(timezone.utc).isoformat(), "note": note})

    def _document(self, p: UpgradeProposal) -> None:
        self.changelog.parent.mkdir(parents=True, exist_ok=True)
        new = not self.changelog.exists()
        with self.changelog.open("a", encoding="utf-8") as f:
            if new:
                f.write("# AiDEN verified upgrades\n\nAppend-only. Generated by core/self_upgrade.py.\n")
            bt = p.evidence.get(Stage.BACKTESTED.value, {})
            oos = p.evidence.get(Stage.OOS_TESTED.value, {})
            changes = ", ".join(f"`{k}`: {v[0]} → {v[1]}" for k, v in p.changes.items()) or "n/a"
            f.write(
                f"\n## v{p.version} — {p.title} ({p.id})\n\n"
                f"- Kind: {p.kind} · Source: {p.source} · Reviewer: {p.reviewer}\n"
                f"- Changes: {changes}\n"
                f"- Problem: {p.problem}\n- Hypothesis: {p.hypothesis}\n"
                f"- In-sample {bt.get('metric', 'metric')}: {bt.get('baseline')} → {bt.get('candidate')} (n={bt.get('n_trades')})\n"
                f"- Out-of-sample: {oos.get('baseline')} → {oos.get('candidate')} (n={oos.get('n_trades')})\n"
                f"- Invariants fingerprint: {p.invariants_fingerprint}\n"
            )
