"""Read-only MetaTrader 5 connection helpers.

This module only attaches to an already-running, already-logged-in MT5
terminal and reports its status. It never places trades and never attempts
to log in with credentials.
"""
import logging
import os
from typing import Optional

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

_TRADE_MODE_NAMES = {
    mt5.ACCOUNT_TRADE_MODE_DEMO: "DEMO",
    mt5.ACCOUNT_TRADE_MODE_CONTEST: "CONTEST",
    mt5.ACCOUNT_TRADE_MODE_REAL: "REAL",
}


def connect(terminal_path: Optional[str] = None) -> bool:
    """Attach to a running MT5 terminal and verify it is logged in.

    Returns True if connected to a logged-in account, False otherwise.
    Always logs the reason on failure.
    """
    initialized = mt5.initialize(path=terminal_path) if terminal_path else mt5.initialize()

    if not initialized:
        logger.error("MT5 initialize() failed: %s", mt5.last_error())
        logger.error(
            "Checklist: MT5 terminal installed? Terminal open? "
            "'terminal_path' in config.yaml correct? Logged into an account?"
        )
        return False

    # If MT5_LOGIN is set in the environment, ensure the terminal is on that
    # account — switch via mt5.login() when it isn't.
    _env_login = os.environ.get("MT5_LOGIN")
    if _env_login:
        _acct = mt5.account_info()
        if _acct is None or str(_acct.login) != str(_env_login):
            _ok = mt5.login(
                int(_env_login),
                password=os.environ.get("MT5_PASSWORD", ""),
                server=os.environ.get("MT5_SERVER", ""),
            )
            if not _ok:
                logger.error("MT5 login(%s) failed: %s", _env_login, mt5.last_error())
                mt5.shutdown()
                return False
            logger.info("Switched MT5 terminal to account %s", _env_login)

    terminal = mt5.terminal_info()
    if terminal is None:
        logger.error("Could not read MT5 terminal info: %s", mt5.last_error())
        mt5.shutdown()
        return False

    if not terminal.connected:
        logger.error("MT5 terminal is open but not connected to a trade server.")
        mt5.shutdown()
        return False

    account = mt5.account_info()
    if account is None:
        logger.error(
            "MT5 terminal is connected but no account is logged in. "
            "Open MT5 and log into a demo account."
        )
        mt5.shutdown()
        return False

    mode = _TRADE_MODE_NAMES.get(account.trade_mode, f"UNKNOWN({account.trade_mode})")
    logger.info(
        "Connected to MT5 | account=%s | server=%s | mode=%s | terminal=%s build %s",
        account.login, account.server, mode, terminal.name, terminal.build,
    )

    if mode != "DEMO":
        logger.warning(
            "Account trade mode is '%s', not DEMO. This pipeline is read-only "
            "and will not place trades, but confirm this is the account you intend to use.",
            mode,
        )

    return True


def disconnect() -> None:
    """Close the MT5 connection."""
    mt5.shutdown()
    logger.info("MT5 connection closed.")
