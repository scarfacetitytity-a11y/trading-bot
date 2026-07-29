"""AiDEN analysis package — pure, stateless market-structure detectors.

Phase 0 extraction: functions moved verbatim from strategies/aiden_index.py
and execution/signal_detectors.py. Zero behavior change — the original
modules re-import these names so every existing import path still resolves.

Modules:
  structure  — H4 bias, order blocks, BOS, swing utilities, FVG-queue helpers
  liquidity  — sweeps, sweep reversals, manipulation W/M patterns
  momentum   — M5 entry trigger, accumulation/distribution, liquidity draw
"""
