"""Apply V3 ProbabilityStack cutover from calibrated thresholds.

Reads config/v3_thresholds.json (written by calibrate_v3_stack.py) and
patches config.yaml + config_vps.yaml:
  - Sets entry_thresholds from calibrated per-symbol optima
  - Adds symbols to v3_cutover.live in staged order (metals first, then indices, then FX)
  - Lowers go_no_go_min_samples to 0 (calibration replaces live shadow requirement)

Run: python -m scripts.apply_v3_cutover [--all] [--symbol XAUUSD]
  --all      : add all calibrated symbols to live
  --symbol X : add only this symbol (default: XAUUSD)
  --dry-run  : print what would change, don't write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT         = Path(__file__).parent.parent
THRESH_FILE  = ROOT / "config" / "v3_thresholds.json"
CONFIGS      = [ROOT / "config" / "config.yaml", ROOT / "config" / "config_vps.yaml"]

STAGED_ORDER = [
    "XAUUSD", "XAGUSD",
    "US100.cash", "US30.cash", "US500.cash", "US2000.cash",
    "UK100.cash", "JP225.cash",
    "GBPUSD", "EURUSD", "USDJPY",
]

# Map symbol → archetype (must match instrument_profile.py)
ARCHETYPE_MAP = {
    "XAUUSD":    "liquidity",
    "XAGUSD":    "liquidity",
    "GBPUSD":    "sniper",
    "EURUSD":    "sniper",
    "USDJPY":    "sniper",
    "US100.cash":"momentum",
    "US30.cash": "momentum",
    "US500.cash":"momentum",
    "US2000.cash":"momentum",
    "UK100.cash":"momentum",
    "JP225.cash":"momentum",
}


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",      action="store_true", help="Add all calibrated symbols")
    parser.add_argument("--symbol",   default="XAUUSD",    help="Single symbol to activate (default: XAUUSD)")
    parser.add_argument("--dry-run",  action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    if not THRESH_FILE.exists():
        print(f"ERROR: {THRESH_FILE} not found. Run calibrate_v3_stack.py first.")
        sys.exit(1)

    thresholds: dict = json.loads(THRESH_FILE.read_text())
    global_thr = thresholds.get("_global", {})

    # Symbols to activate
    if args.all:
        to_activate = [s for s in STAGED_ORDER if s in thresholds]
    else:
        to_activate = [args.symbol]

    print(f"Activating v3 for: {to_activate}")
    print(f"Global calibrated threshold: {global_thr.get('threshold')} "
          f"(wr={global_thr.get('wr')} avg_r={global_thr.get('avg_r')} n={global_thr.get('n')})")

    # Build per-archetype thresholds from calibration
    archetype_thresholds: dict[str, int] = {}
    for sym, thr_data in thresholds.items():
        if sym == "_global":
            continue
        arch = ARCHETYPE_MAP.get(sym)
        if arch and "threshold" in thr_data:
            # Use highest (most conservative) threshold among symbols of same archetype
            existing = archetype_thresholds.get(arch, 0)
            archetype_thresholds[arch] = max(existing, int(thr_data["threshold"]))

    # Fall back to global if archetype not calibrated
    for arch in ("sniper", "momentum", "liquidity"):
        if arch not in archetype_thresholds and global_thr.get("threshold"):
            archetype_thresholds[arch] = int(global_thr["threshold"])

    print(f"Archetype thresholds: {archetype_thresholds}")

    if args.dry_run:
        print("[dry-run] No files written.")
        return

    for cfg_path in CONFIGS:
        if not cfg_path.exists():
            continue
        cfg = load_yaml(cfg_path)

        v3 = cfg.setdefault("v3_cutover", {})
        live: list = v3.get("live", [])

        for sym in to_activate:
            if sym not in live:
                live.append(sym)

        v3["live"]                   = live
        v3["go_no_go_min_samples"]   = 0   # calibration replaces live shadow gate
        if archetype_thresholds:
            v3["entry_thresholds"] = archetype_thresholds

        cfg["v3_cutover"] = v3
        save_yaml(cfg_path, cfg)
        print(f"Updated: {cfg_path.name}")

    print(f"\nv3 live: {live}")
    print("Restart the bot to activate. Watchdog will handle this automatically.")


if __name__ == "__main__":
    main()
