"""Phase 0 verification — zero-behavior-change gate.

1. Imports the NEW _build_aiden_strategy (via core.instrument_profile.PROFILES).
2. Extracts the OLD _build_aiden_strategy source from git HEAD and execs it
   in a namespace wired to the same AiDENIndexStrategy + backtest param dicts.
3. Compares vars(strategy) for all 11 symbols. Must be identical.

Run from repo root:  venv\\Scripts\\python.exe _verify_phase0.py
"""
import ast
import subprocess
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

SYMBOLS = ["GBPUSD", "EURUSD", "USDJPY", "XAUUSD", "XAGUSD", "US30.cash",
           "US100.cash", "US500.cash", "US2000.cash", "UK100.cash", "JP225.cash"]

# ── New implementation ────────────────────────────────────────────────────────
from execution.orchestrator import _build_aiden_strategy as build_new

# ── Old implementation, resurrected from git HEAD ─────────────────────────────
old_src = subprocess.run(
    ["git", "show", "HEAD:execution/orchestrator.py"],
    capture_output=True, text=True, cwd=ROOT, check=True,
    encoding="utf-8", errors="replace",
).stdout
tree = ast.parse(old_src)
fn = next(n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "_build_aiden_strategy")
fn_src = ast.get_source_segment(old_src, fn)

from strategies.aiden_index import AiDENIndexStrategy
from backtests.run_multi_instrument import (
    INSTRUMENTS, OPTIMISED_PARAMS, TRAIL_CONFIGS, BIDIRECTIONAL, M15_PARAMS,
)
ns = dict(AiDENIndexStrategy=AiDENIndexStrategy, INSTRUMENTS=INSTRUMENTS,
          OPTIMISED_PARAMS=OPTIMISED_PARAMS, TRAIL_CONFIGS=TRAIL_CONFIGS,
          BIDIRECTIONAL=BIDIRECTIONAL, M15_PARAMS=M15_PARAMS)
exec(compile(fn_src, "<old_orchestrator>", "exec"), ns)
build_old = ns["_build_aiden_strategy"]

# ── Compare ───────────────────────────────────────────────────────────────────
failures = 0
for sym in SYMBOLS:
    old_v = vars(build_old(sym))
    new_v = vars(build_new(sym))
    if old_v == new_v:
        print(f"  OK   {sym}")
        continue
    failures += 1
    print(f"  FAIL {sym}")
    keys = sorted(set(old_v) | set(new_v))
    for k in keys:
        if old_v.get(k, "<missing>") != new_v.get(k, "<missing>"):
            print(f"       {k}: old={old_v.get(k, '<missing>')!r} new={new_v.get(k, '<missing>')!r}")

# ── Import smoke test ─────────────────────────────────────────────────────────
import strategies.aiden_index          # noqa: F401
import execution.signal_detectors      # noqa: F401
import execution.orchestrator          # noqa: F401
import analysis.structure, analysis.liquidity, analysis.momentum  # noqa: F401
from core.instrument_profile import PROFILES
assert len(PROFILES) == 11, f"expected 11 profiles, got {len(PROFILES)}"

# Old aiden_index helper import paths must still resolve
from strategies.aiden_index import _compute_h4_bias_ema, _resample_h4  # noqa: F401
from execution.signal_detectors import (  # noqa: F401
    _swing_highs, _swing_lows, detect_m5_entry_trigger,
    detect_accumulation, detect_liquidity_draw,
)

print()
if failures:
    print(f"RESULT: FAIL — {failures}/{len(SYMBOLS)} symbols differ")
    sys.exit(1)
print(f"RESULT: PASS — all {len(SYMBOLS)} symbols identical, imports clean")
