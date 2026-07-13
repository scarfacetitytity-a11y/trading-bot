"""Scan MT5 for available symbols matching AiDEN criteria.

Prints all index/metal/commodity CFDs available on the broker,
grouped by category, with spread info. Helps identify correct
symbol names and any additional candidates.

Run with MT5 open:
    python -m backtests.scan_symbols
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not installed.")
    sys.exit(1)

from config.settings import load_config

# Keywords that indicate institutional CFD instruments
INDEX_KEYWORDS   = ["US100", "US30", "US500", "US2000", "JP225", "GER40", "UK100",
                    "AUS200", "NAS", "DOW", "SPX", "DAX", "FTSE", "NIKKEI",
                    "HK50", "FRA40", "CHINA50", "VIX"]
METAL_KEYWORDS   = ["XAU", "XAG", "GOLD", "SILVER", "PLAT", "PALLAD"]
ENERGY_KEYWORDS  = ["OIL", "WTI", "BRENT", "NGAS", "USOIL", "UKOIL"]
CRYPTO_KEYWORDS  = ["BTC", "ETH", "XRP", "CRYPTO"]


def categorise(name: str) -> str:
    n = name.upper()
    for k in INDEX_KEYWORDS:
        if k in n:
            return "INDEX"
    for k in METAL_KEYWORDS:
        if k in n:
            return "METAL"
    for k in ENERGY_KEYWORDS:
        if k in n:
            return "ENERGY"
    for k in CRYPTO_KEYWORDS:
        if k in n:
            return "CRYPTO"
    return "OTHER"


def spread_pips(info) -> float:
    try:
        return round((info.ask - info.bid) / info.point, 1) if info.point > 0 else 0.0
    except Exception:
        return 0.0


def main() -> None:
    config = load_config()
    mt5_cfg = config.get("mt5", {})
    terminal = mt5_cfg.get("terminal_path") or None

    init_kwargs = {}
    if terminal:
        init_kwargs["path"] = terminal

    if not mt5.initialize(**init_kwargs):
        print(f"MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print(f"MT5 connected: build {mt5.terminal_info().build}\n")

    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        print("No symbols returned.")
        mt5.shutdown()
        sys.exit(1)

    categorised: dict[str, list] = {
        "INDEX": [], "METAL": [], "ENERGY": [], "CRYPTO": [], "OTHER": []
    }

    for s in all_symbols:
        cat = categorise(s.name)
        if cat == "OTHER":
            continue
        info = mt5.symbol_info(s.name)
        if info is None:
            continue
        sp = spread_pips(info)
        categorised[cat].append((s.name, sp, info.trade_mode))

    # Print each category
    for cat in ["INDEX", "METAL", "ENERGY", "CRYPTO"]:
        items = sorted(categorised[cat], key=lambda x: x[0])
        if not items:
            continue
        print(f"\n{'-'*55}")
        print(f"  {cat}  ({len(items)} symbols)")
        print(f"{'-'*55}")
        for name, sp, mode in items:
            tradeable = "TRADE" if mode > 0 else "view-only"
            print(f"  {name:<22}  spread: {sp:>7.1f} pts  [{tradeable}]")

    # Highlight AiDEN universe — show which names exist and suggest alternatives
    print(f"\n{'='*55}")
    print("  AiDEN universe check")
    print("="*55)

    targets = {
        "US100.cash": ["US100", "US100.cash", "NAS100", "NASDAQ"],
        "US30.cash":  ["US30", "US30.cash",  "DOW30",  "WALLST"],
        "US500.cash": ["US500", "US500.cash", "SPX500", "SP500"],
        "JP225.cash": ["JP225", "JP225.cash", "NIKKEI", "JPN225"],
        "US2000.cash":["US2000","US2000.cash","RUSSELL","RUS2000"],
        "XAUUSD":     ["XAUUSD", "GOLD", "XAUUSD."],
    }

    all_names = {s.name for s in all_symbols}

    for target, candidates in targets.items():
        found = [c for c in candidates if c in all_names]
        if found:
            status = f"FOUND: {', '.join(found)}"
        else:
            # Fuzzy — check if any symbol contains target prefix
            prefix = target.replace(".cash", "").replace("USD", "")
            fuzzy = [s for s in all_names if prefix.upper() in s.upper()][:5]
            status = f"NOT FOUND — close matches: {fuzzy}" if fuzzy else "NOT FOUND — not on broker"
        print(f"  {target:<16}  {status}")

    mt5.shutdown()
    print()


if __name__ == "__main__":
    main()
