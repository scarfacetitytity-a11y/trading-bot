"""Position sizing helpers.

Two modes:
  fixed     - use the lot_size from config directly
  risk_pct  - size lots so that a stop-loss hit costs exactly risk_pct % of balance
"""
import logging

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


def calculate_lots(symbol: str, trade_cfg: dict, balance: float) -> float:
    """Return the lot size to use for the next trade.

    trade_cfg keys used:
        risk_mode         : 'fixed' (default) or 'risk_pct'
        lot_size          : used when risk_mode='fixed'
        risk_pct          : % of balance to risk (risk_mode='risk_pct')
        stop_loss_points  : SL distance in points (risk_mode='risk_pct')
    """
    mode = trade_cfg.get("risk_mode", "fixed")

    if mode == "risk_pct":
        return _risk_pct_lots(symbol, balance, trade_cfg)

    lots = float(trade_cfg.get("lot_size", 0.01))
    return _clamp_lots(symbol, lots)


def calculate_sl_tp(
    symbol: str,
    direction: int,
    trade_cfg: dict,
) -> tuple[float | None, float | None]:
    """Return (stop_loss_price, take_profit_price) or (None, None) if not configured."""
    sl_points = int(trade_cfg.get("stop_loss_points", 0))
    tp_points = int(trade_cfg.get("take_profit_points", 0))

    if sl_points == 0 and tp_points == 0:
        return None, None

    info = mt5.symbol_info(symbol)
    if info is None:
        logger.warning("Could not get symbol info for %s; SL/TP not set.", symbol)
        return None, None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None, None

    point = info.point
    price = tick.ask if direction == 1 else tick.bid

    sl = (price - sl_points * point) if direction == 1 else (price + sl_points * point)
    tp = (price + tp_points * point) if direction == 1 else (price - tp_points * point)

    return (round(sl, info.digits) if sl_points else None,
            round(tp, info.digits) if tp_points else None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _risk_pct_lots(symbol: str, balance: float, trade_cfg: dict) -> float:
    risk_pct = float(trade_cfg.get("risk_pct", 1.0)) / 100
    sl_points = int(trade_cfg.get("stop_loss_points", 100))

    info = mt5.symbol_info(symbol)
    if info is None:
        logger.warning("Could not get symbol info for %s; falling back to lot_size.", symbol)
        return float(trade_cfg.get("lot_size", 0.01))

    # Value of 1 point movement for 1 lot, in account currency
    # For forex: point_value = contract_size * point
    # For indices: similar, but tick_value / tick_size gives the rate
    if info.trade_tick_size > 0:
        point_value_per_lot = info.trade_tick_value / info.trade_tick_size * info.point
    else:
        point_value_per_lot = info.point * info.trade_contract_size

    risk_amount = balance * risk_pct
    sl_value_per_lot = sl_points * point_value_per_lot

    if sl_value_per_lot <= 0:
        logger.warning("SL value per lot is zero for %s; falling back to lot_size.", symbol)
        return float(trade_cfg.get("lot_size", 0.01))

    lots = risk_amount / sl_value_per_lot
    return _clamp_lots(symbol, lots)


def _clamp_lots(symbol: str, lots: float) -> float:
    """Clamp to broker min/max/step and round correctly."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return round(max(lots, 0.01), 2)

    step = info.volume_step
    min_lot = info.volume_min
    max_lot = info.volume_max

    # Round down to nearest step
    if step > 0:
        lots = int(lots / step) * step

    lots = max(min_lot, min(lots, max_lot))
    digits = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
    return round(lots, max(digits, 2))
