"""
Entry and smart exit signal generation.

Entry: BB breach + RSI extreme + volume spike + session active (all required)
Exit:  TP / SL hit OR smart early exit (2-of-3 conditions)
"""

import numpy as np
import pandas as pd
from config import settings


def generate_entry_signals(df: pd.DataFrame,
                            rsi_oversold: float = settings.RSI_OVERSOLD,
                            rsi_overbought: float = settings.RSI_OVERBOUGHT) -> pd.DataFrame:
    """
    Add 'signal_long' and 'signal_short' boolean columns.
    Requires: BB_lower, BB_upper, RSI, vol_spike, session_active columns.
    """
    df = df.copy()

    # Trend filter: only trade with the trend, block counter-trend signals
    # trend_up   → longs allowed, shorts blocked
    # trend_down → shorts allowed, longs blocked
    # neither    → both allowed (choppy/ranging — mean reversion works best)
    trend_up   = df.get("trend_up",   pd.Series(False, index=df.index))
    trend_down = df.get("trend_down", pd.Series(False, index=df.index))
    long_trend_ok  = trend_up | (~trend_up & ~trend_down)   # uptrend or chop
    short_trend_ok = trend_down | (~trend_up & ~trend_down) # downtrend or chop

    # Long: price below lower BB + RSI oversold + volume spike + session active + trend ok
    df["signal_long"] = (
        (df["close"] < df["BB_lower"]) &
        (df["RSI"] < rsi_oversold) &
        df["vol_spike"] &
        df["session_active"] &
        long_trend_ok
    )

    # Short: price above upper BB + RSI overbought + volume spike + session active + trend ok
    df["signal_short"] = (
        (df["close"] > df["BB_upper"]) &
        (df["RSI"] > rsi_overbought) &
        df["vol_spike"] &
        df["session_active"] &
        short_trend_ok
    )

    return df


def compute_exit_levels(entry_price: float, direction: str, atr: float,
                         sl_mult: float = settings.ATR_SL_MULTIPLIER,
                         tp_mult: float = settings.ATR_TP_MULTIPLIER) -> tuple:
    """
    Returns (take_profit, stop_loss) prices.
    direction: 'long' or 'short'
    """
    sl_dist = atr * sl_mult
    tp_dist = atr * tp_mult
    if direction == "long":
        return entry_price + tp_dist, entry_price - sl_dist
    else:
        return entry_price - tp_dist, entry_price + sl_dist


def check_smart_exit(candles_in_trade: pd.DataFrame,
                     entry_price: float,
                     direction: str,
                     stop_loss: float,
                     rsi_midline: float = settings.SMART_EXIT_RSI_MIDLINE,
                     sl_retrace_pct: float = settings.SMART_EXIT_SL_RETRACE,
                     stall_candles: int = settings.SMART_EXIT_STALL_CANDLES) -> bool:
    """
    Returns True if smart exit should fire (2-of-3 conditions met).
    candles_in_trade: all candles since entry (including current).
    """
    if len(candles_in_trade) < 2:
        return False

    current = candles_in_trade.iloc[-1]
    prev = candles_in_trade.iloc[-2]
    sl_dist = abs(entry_price - stop_loss)

    conditions_met = 0

    # Condition 1: RSI crossed back through 50 against position
    if direction == "long":
        rsi_crossed = prev["RSI"] < rsi_midline and current["RSI"] >= rsi_midline
    else:
        rsi_crossed = prev["RSI"] > rsi_midline and current["RSI"] <= rsi_midline
    if rsi_crossed:
        conditions_met += 1

    # Condition 2: Price 50%+ retraced toward SL AND stalled for N candles
    if direction == "long":
        retrace = (entry_price - current["close"]) / sl_dist
    else:
        retrace = (current["close"] - entry_price) / sl_dist

    if retrace >= sl_retrace_pct and len(candles_in_trade) >= stall_candles:
        recent = candles_in_trade.iloc[-stall_candles:]
        if direction == "long":
            no_progress = recent["close"].max() <= candles_in_trade.iloc[-stall_candles - 1]["close"] if len(candles_in_trade) > stall_candles else True
        else:
            no_progress = recent["close"].min() >= candles_in_trade.iloc[-stall_candles - 1]["close"] if len(candles_in_trade) > stall_candles else True
        if no_progress:
            conditions_met += 1

    # Condition 3: Opposing BB signal fired
    if direction == "long" and current.get("signal_short", False):
        conditions_met += 1
    elif direction == "short" and current.get("signal_long", False):
        conditions_met += 1

    return conditions_met >= 2
