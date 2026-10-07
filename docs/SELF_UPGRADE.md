# How AiDEN evolves

AI does the heavy thinking. Algorithms do repetitive measurement. Hard-coded
controls protect capital. The human remains the final authority.

**Self-improvement never means self-overriding safety.** Strategy logic,
research parameters, data sources, probability models, features and code may
evolve. Risk limits, human approval and safety invariants cannot be weakened
automatically.

## Components

| Concept | File | Role |
|---|---|---|
| Invariants | `core/system_invariants.py` | Immutable envelope, approval policy, emergency halt, pre-order check, change classification. |
| Lifecycle | `core/self_upgrade.py` | `UpgradeProposal`, ordered stages, evidence bar (`PromotionCriteria`), human review, versioning, append-only ledger. |
| Upgrade loop | `research/upgrade_loop.py` | Runs a proposal through backtest → OOS → stress → compare → risk on the existing FTMO Monte Carlo engine. Stops at review. |
| Upgrade gate | `execution/upgrade_gate.py` | Everything that can change live behaviour passes here: startup config, lift proposals, Telegram `/risk`, structural fixes, protected-file integrity. |
| Ledger | `research/upgrades/ledger.jsonl` | Every state change of every proposal (commit it). |
| Changelog | `research/upgrades/CHANGELOG.md` | One entry per promoted upgrade, with evidence and invariants fingerprint. |

## The loop

```
Scout / Observer → Market Context → Strategy Engine → Probability Stack → Risk Agent
   → Human Approval → Execution → Trade Journal → Outcome Analysis (learning_loop, Quant)
   → UpgradeProposal (core/self_upgrade) → upgrade_loop: backtest / OOS / stress / compare
   → risk validation (invariants) → human review → PROMOTED → VERSIONED → DOCUMENTED
   → human applies the change in a commit → upgrade_gate checks it at next start → Scout
```

## Lifecycle

```
OBSERVED → PROBLEM_IDENTIFIED → HYPOTHESIS → EXPERIMENT → BACKTESTED → OOS_TESTED
  → STRESS_TESTED → BASELINE_COMPARED → RISK_VALIDATED → REVIEWED → PROMOTED
  → VERSIONED → DOCUMENTED                         (REJECTED from any stage, terminal)
```

Enforced in code, not just described:

- Stages cannot be skipped or revisited; a rejected proposal stays rejected.
- **Risk validation** runs at creation *and* again at promotion. Any automatic
  loosening of a risk parameter is rejected, even if it stays inside the envelope.
- **Evidence bar** (`PromotionCriteria`, defaults):
  - ≥ 100 in-sample trades and ≥ 50 out-of-sample trades (chronological split);
  - the candidate beats the baseline both in-sample **and** out-of-sample;
  - out-of-sample gain ≥ 50% of in-sample gain (overfitting guard);
  - stressed blow rate (OOS + per-trade cost shock) no worse than baseline.
- **Review** needs a named human. `Sage`, `Quant`, `Builder`, `Scout`, `claude`,
  `autoresearch`, `learning_loop` are refused as reviewers.
- **Promotion records a decision, it does not deploy.** Applying a promoted
  change to `config_vps.yaml` or code is a normal commit by a human, so every
  live change is a reviewable, revertible diff.
- Parameters the evaluator cannot measure are rejected ("no evidence possible")
  instead of being kept or discarded on noise.

## Who can propose

| Source | How |
|---|---|
| Human | `python -m research.upgrade_loop propose ...` |
| Learning loop | `auto_apply` tier-2 findings → `upgrade_gate.record_structural_proposal` (deduplicated) |
| Quant | Lift proposals file → gate requires n ≥ 30 and 0.5x–2x bounds; strategy findings as proposals |
| Sage / Scout | Brain-vault notes phrased as hypotheses; a human or the loop turns them into proposals |
| Autoresearch | Exploration only; winners go through `upgrade_loop` (see `research/program.md`) |
| Builder | Bug fixes only; must not touch protected files (prompt rule + startup integrity check) |

## Protected files

`core/system_invariants.py`, `core/self_upgrade.py`, `execution/upgrade_gate.py`,
`execution/risk_agent.py`. Uncommitted edits to any of them set the persistent
emergency halt at the next bot start. Change them only through a reviewed PR.

## Operating it

```powershell
python -m execution.upgrade_gate --check              # config vs envelope, halt state
python -m execution.upgrade_gate --halt "reason"      # persistent emergency stop
python -m execution.upgrade_gate --clear-halt         # human only

python -m research.upgrade_loop propose --title "Raise score floor to 75" --param SCORE_FLOOR=75 `
  --observation "C-grade entries 18% WR over 60 live trades" `
  --problem "Low-grade setups dilute expectancy" `
  --hypothesis "SCORE_FLOOR 75 raises pass rate without raising blow rate"
python -m research.upgrade_loop run UPG-20261007-abc123
python -m research.upgrade_loop review UPG-20261007-abc123 --reviewer Jay --approve
python -m research.upgrade_loop list
```

`run` needs `data/processed/<SYMBOL>_M15.csv` (run the data pipeline first).

## Not automated on purpose

- Widening the envelope (`core/system_invariants.py`).
- Switching `execution_mode` to `demo_autonomous`.
- Applying a promoted change to live config.
- Clearing an emergency halt.
