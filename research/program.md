# AiDEN Autoresearch Program

Adapted from Karpathy's autoresearch. Autonomous parameter optimization for the FTMO challenge.

## Setup

To set up a new research run:

1. **Agree on a run tag** based on today's date (e.g. `jul26`). The branch `research/<tag>` must not exist yet — this is a fresh run.
2. **Create the branch**: `git checkout -b research/<tag>` (if using git).
3. **Read the in-scope files**:
   - `research/params.py` — the only file you modify. All risk parameters live here.
   - `research/backtest_harness.py` — fixed. Do NOT modify. Contains the backtest runner, evaluation metric, and result output.
   - `research/results.tsv` — your log of experiments (untracked by git).
4. **Verify data**: Check that historical OHLC data exists at the path specified in `backtest_harness.py`. If not, tell the human.
5. **Initialize results.tsv**: Create it with the header row. Baseline runs first.
6. **Confirm and go**.

## Experimentation

Each experiment runs the full backtest on the last 6 months of data. The backtest runs for a **fixed window** regardless of parameters. The metric is **challenge_pass_rate** — the percentage of simulated challenges that pass FTMO rules (profit target reached without breaching daily or max DD limits).

**What you CAN modify in `params.py`:**
- `BASE_RISK` — risk per trade (percent of account). Primary lever.
- `DAILY_DD_LIMIT` — daily drawdown halt threshold (percent).
- `SCORE_FLOOR` — minimum score to take a trade.
- `MIN_GRADE` — minimum grade (A/B/C/D).
- `MIN_RR` — minimum risk/reward ratio.
- `COUSIN_BLOCK_HOURS` — how long to block correlated instruments after a loss.
- `CAP_CONCURRENT` — max simultaneous open positions.
- `EDGE_FILTER` — minimum edge ratio to enter.
- `SL_BUFFER_ATR` — stop loss buffer as ATR multiple.

**What you CANNOT modify:**
- `backtest_harness.py` — the evaluator is fixed. It IS the ground truth.
- FTMO rules built into the harness (5% daily, 10% max DD, 10% profit target).
- Instrument configuration (pip values, max lots, correlation pairs).

**The goal**: maximize `challenge_pass_rate`. Secondary metrics: `avg_profit_pct` (average profit at end of challenge window), `max_dd_avg` (average max DD across runs). Lower DD for same pass rate is a simplification win.

**Simplicity criterion** (from autoresearch): A 0.5% improvement in pass rate that adds complex conditional logic? Probably not worth it. A 0.5% improvement from simplifying params? Keep. All else equal, the simpler param set is better.

## Output Format

The backtest prints:

```
---
challenge_pass_rate: 0.420
avg_profit_pct: 6.8
max_dd_avg: 4.2
blow_rate: 0.12
timeout_rate: 0.28
total_challenges: 500
```

Extract the key metric:
```
grep "^challenge_pass_rate:" run.log
```

## Logging Results

Log to `results.tsv` (tab-separated):

```
tag	challenge_pass_rate	max_dd_avg	status	description
```

Example:
```
tag	challenge_pass_rate	max_dd_avg	status	description
baseline	0.420	4.2	keep	baseline config
jul26a	0.445	4.1	keep	lower base_risk to 0.3%
jul26b	0.390	5.1	discard	higher score_floor to 80
jul26c	0.000	0.0	crash	invalid param combination
```

## The Experiment Loop

LOOP:
1. Look at current params.py — know exactly what changed from the last run.
2. Modify ONE parameter at a time (or two if they're tightly coupled like rr+edge).
3. Commit the change: `git commit -m "research: [description]"`
4. Run: `python -m research.backtest_harness > run.log 2>&1`
5. Read: `grep "^challenge_pass_rate:\|^blow_rate:\|^max_dd_avg:" run.log`
6. If empty: the run crashed. Read `tail -30 run.log`, attempt fix. If fundamentally broken, discard and log "crash".
7. Log the result to results.tsv.
8. If `challenge_pass_rate` improved: keep the commit, advance.
9. If equal or worse: `git reset --hard HEAD~1`, restore previous params.

**Timeout**: Each backtest should complete in under 5 minutes. If it exceeds 10 minutes, kill and treat as failure.

**Never stop**: Once the loop begins, do not pause to ask if you should continue. The human may be asleep. Run until manually stopped. If stuck, re-read the sweep results in Obsidian Brain, check `2026-07-26-adaptive-sweep-results.md`, try untested regions of parameter space.

## Reporting

At any pause point (when human returns), output:
1. How many experiments ran
2. Best `challenge_pass_rate` achieved and the params that achieved it
3. What was tried and discarded
4. One paragraph on what the data suggests about the parameter space
5. Recommended next params to try

Write a brain note to Obsidian:
```python
from execution.council_obsidian import write_brain_note
write_brain_note(
    slug=f'research-{tag}-session',
    title=f'Autoresearch Session — {tag}',
    body='[full experiment log]',
    tags=['research', 'parameters', 'backtest', 'autoresearch', 'aiden'],
    related=['AiDEN', 'Brain', 'Backtest'],
    note_type='insight',
)
```
