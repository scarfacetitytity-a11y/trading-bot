# research/upgrades

- `ledger.jsonl` — append-only history of every upgrade proposal (core/self_upgrade.py). Commit it.
- `CHANGELOG.md` — one entry per promoted upgrade, with evidence and the invariants fingerprint.

Both are written by `python -m research.upgrade_loop ...`. See docs/SELF_UPGRADE.md.
