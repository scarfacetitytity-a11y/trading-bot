"""Test isolation: nothing a test does may touch the real vault or logs.

Before this file existed the suite wrote Obsidian notes to a literal
'C:\\Users\\anton\\OneDrive\\...' directory inside the repo on non-Windows hosts.
Set at import time because several modules resolve paths at import.
"""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="aiden-tests-"))
os.environ.setdefault("AIDEN_VAULT_PATH", str(_TMP / "vault"))
os.environ.setdefault("AIDEN_LOGS_DIR", str(_TMP / "logs"))
os.environ.setdefault("AIDEN_FIRM_DB", str(_TMP / "firm.db"))
os.environ.pop("AIDEN_EMERGENCY_HALT", None)
