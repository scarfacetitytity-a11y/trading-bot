"""Backfill existing Obsidian trade notes with queryable YAML frontmatter fields."""
from __future__ import annotations

import re
from pathlib import Path

VAULT_TRADES = Path(r"C:\Users\anton\OneDrive\Desktop\Aiden\AiDEN\Trades")

_TABLE = {
    "Symbol":      ("symbol",      str),
    "Direction":   ("direction",   str),
    "Outcome":     ("outcome",     str),
    "Score":       ("score",       lambda v: int(v.split("/")[0])),
    "Grade":       ("grade",       str),
    "R-Multiple":  ("r_multiple",  lambda v: float(v.strip("`R").replace(",", ""))),
    "P&L":         ("pnl_usd",     lambda v: float(v.strip("`$").replace(",", ""))),
    "Lots":        ("lots",        lambda v: float(v.replace(",", ""))),
    "Entry":       ("entry_price", lambda v: float(v.strip("`").replace(",", ""))),
    "SL":          ("sl_price",    lambda v: float(v.strip("`").replace(",", ""))),
    "TP":          ("tp_price",    lambda v: v.strip("`") if v not in ("—", "") else ""),
    "MFE":         ("mfe_r",       lambda v: float(v.strip("R").replace(",", ""))),
    "MAE":         ("mae_r",       lambda v: float(v.strip("R").replace(",", ""))),
    "Open":        ("open_time",   lambda v: v.replace(" UTC", "")),
    "Close":       ("close_time",  lambda v: v.replace(" UTC", "")),
}

_WIKILINK = re.compile(r"\[\[(.+?)\]\]")
_NEW_FIELDS = set(k for _, (k, _) in _TABLE.items() if k not in
                  ("symbol", "direction", "outcome"))  # these already in relatedTo tags


def _extract_table_value(raw: str) -> str:
    raw = raw.strip()
    raw = _WIKILINK.sub(lambda m: m.group(1), raw)
    # handle the old f-string bug: "52095 if tp else `—`"
    if " if tp else" in raw:
        raw = raw.split(" if tp else")[0].strip("`")
    return raw


def _parse_note(text: str) -> dict:
    vals: dict = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 2:
            continue
        field, value = parts[0], parts[1]
        if field in _TABLE:
            key, cast = _TABLE[field]
            try:
                vals[key] = cast(_extract_table_value(value))
            except (ValueError, AttributeError):
                vals[key] = _extract_table_value(value)
    return vals


def _already_backfilled(fm: str) -> bool:
    return "r_multiple:" in fm or "score:" in fm


def _build_new_frontmatter(old_fm: str, vals: dict) -> str:
    lines = old_fm.strip().splitlines()
    new_lines = []
    for line in lines:
        new_lines.append(line)
    # append new fields after existing ones
    for key in ["symbol", "direction", "outcome", "score", "grade",
                 "r_multiple", "pnl_usd", "lots", "entry_price", "sl_price",
                 "tp_price", "mfe_r", "mae_r", "open_time", "close_time"]:
        if key in vals:
            v = vals[key]
            if isinstance(v, str) and (" " in v or v == ""):
                new_lines.append(f'{key}: "{v}"')
            else:
                new_lines.append(f"{key}: {v}")
    return "\n".join(new_lines)


def backfill(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return False
    end = text.index("---", 3)
    old_fm = text[3:end].strip()
    body = text[end + 3:]

    if _already_backfilled(old_fm):
        return False

    vals = _parse_note(body)
    if not vals:
        return False

    new_fm = _build_new_frontmatter(old_fm, vals)
    path.write_text(f"---\n{new_fm}\n---{body}", encoding="utf-8")
    return True


def main() -> None:
    notes = list(VAULT_TRADES.rglob("*.md"))
    updated = 0
    skipped = 0
    for note in notes:
        if note.name.startswith("_"):
            continue
        try:
            if backfill(note):
                updated += 1
                print(f"  updated: {note.name}")
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR {note.name}: {e}")
    print(f"\n{updated} updated, {skipped} already done / skipped")


if __name__ == "__main__":
    main()
