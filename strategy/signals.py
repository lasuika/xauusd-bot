"""
Entry and smart exit signal generation.
Three strategies available:
  1. mean_reversion  — BB + RSI + Volume (original)
  2. vwap_rsi        — VWAP deviation + RSI confirmation
  3. ema_momentum    — EMA crossover + momentum breakout
"""

import numpy as np
import pandas as pd
from config import settings


# ── Strategy 1: Mean Reversion ────────────────────────────────────────────────

def generate_entry_signals(df: pd.DataFrame,
                            rsi_oversold: float = settings.RSI_OVERSOLD,
                            rsi_overbought: float = settings.RSI_OVERBOUGHT,
                            strategy: str = settings.STRATEGY) -> pd.DataFrame:
    """Dispatch to the selected strategy's signal generator."""
    if strategy == "vwap_rsi":
        return _signals_vwap_rsi(df, rsi_oversold, rsi_overbought)
    elif strategy == "ema_momentum":
        return _signals_ema_momentum(df)
    else:
        return _signals_mean_reversion(df, rsi_oversold, rsi_overbought)


def _signals_mean_reversion(df: pd.DataFrame,
                              rsi_oversold: float,
                              rsi_overbought: float) -> pd.DataFrame:
    """Original: BB breach + RSI extreme + volume spike + trend filter."""
    df = df.copy()

    trend_up   = df.get("trend_up",   pd.Series(False, index=df.index))
    trend_down = df.get("trend_down", pd.Series(False, index=df.index))
    long_trend_ok  = trend_up | (~trend_up & ~trend_down)
    short_trend_ok = trend_down | (~trend_up & ~trend_down)

    df["signal_long"] = (
        (df["close"] < df["BB_lower"]) &
        (df["RSI"] < rsi_oversold) &
        df["vol_spike"] &
        df["session_active"] &
        long_trend_ok
    )
    df["signal_short"] = (
        (df["close"] > df["BB_upper"]) &
        (df["RSI"] > rsi_overbought) &
        df["vol_spike"] &
        df["session_active"] &
        short_trend_ok
    )
    return df


# ── Strategy 2: VWAP + RSI Divergence ────────────────────────────────────────

def _signals_vwap_rsi(df: pd.DataFrame,
                       rsi_oversold: float,
                       rsi_overbought: float) -> pd.DataFrame:
    """
    Long  when: price > 1.5 std below VWAP AND RSI < oversold AND volume spike AND session active
    Short when: price > 1.5 std above VWAP AND RSI > overbought AND volume spike AND session active
    VWAP resets each trading day (00:00 UTC).
    """
    df = df.copy()

    # Session VWAP (resets daily)
    df["_date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["_tp_vol"] = typical * df["volume"]

    df["VWAP"] = (
        df.groupby("_date")["_tp_vol"].cumsum() /
        df.groupby("_date")["volume"].cumsum()
    )

    # VWAP standard deviation bands (rolling 20-period within the day)
    df["_vwap_diff_sq"] = (df["close"] - df["VWAP"]) ** 2
    df["VWAP_std"] = (
        df.groupby("_date")["_vwap_diff_sq"]
        .transform(lambda x: x.expanding().mean() ** 0.5)
    )

    vwap_dev = settings.VWAP_STD_THRESHOLD
    df["signal_long"] = (
        (df["close"] < df["VWAP"] - vwap_dev * df["VWAP_std"]) &
        (df["RSI"] < rsi_oversold) &
        df["vol_spike"] &
        df["session_active"]
    )
    df["signal_short"] = (
        (df["close"] > df["VWAP"] + vwap_dev * df["VWAP_std"]) &
        (df["RSI"] > rsi_overbought) &
        df["vol_spike"] &
        df["session_active"]
    )

    df = df.drop(columns=["_date", "_tp_vol", "_vwap_diff_sq"])
    return df


# ── Strategy 3: EMA Momentum Breakout ────────────────────────────────────────

def _signals_ema_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long  when: EMA8 crosses above EMA21, price above EMA50, RSI 45-65, volume spike
    Short when: EMA8 crosses below EMA21, price below EMA50, RSI 35-55, volume spike
    Trend-following scalp — works best in trending gold sessions.
    """
    df = df.copy()

    df["EMA8"]  = df["close"].ewm(span=8,  adjust=False).mean()
    df["EMA21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Crossover detection
    ema_cross_up   = (df["EMA8"] > df["EMA21"]) & (df["EMA8"].shift(1) <= df["EMA21"].shift(1))
    ema_cross_down = (df["EMA8"] < df["EMA21"]) & (df["EMA8"].shift(1) >= df["EMA21"].shift(1))

    df["signal_long"] = (
        ema_cross_up &
        (df["close"] > df["EMA50"]) &
        (df["RSI"] > 45) & (df["RSI"] < 65) &
        df["vol_spike"] &
        df["session_active"]
    )
    df["signal_short"] = (
        ema_cross_down &
        (df["close"] < df["EMA50"]) &
        (df["RSI"] > 35) & (df["RSI"] < 55) &
        df["vol_spike"] &
        df["session_active"]
    )
    return df


# ── Exit logic (shared across all strategies) ─────────────────────────────────

def compute_exit_levels(entry_price: float, direction: str, atr: float,
                         sl_mult: float = settings.ATR_SL_MULTIPLIER,
                         tp_mult: float = settings.ATR_TP_MULTIPLIER) -> tuple:
    """Returns (take_profit, stop_loss) prices."""
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
    """Returns True if smart exit should fire (2-of-3 conditions met)."""
    if len(candles_in_trade) < 2:
        return False

    current = candles_in_trade.iloc[-1]
    prev    = candles_in_trade.iloc[-2]
    sl_dist = abs(entry_price - stop_loss)
    if sl_dist == 0:
        return False

    conditions_met = 0

    # Condition 1: RSI crossed back through midline against position
    if direction == "long":
        rsi_crossed = prev["RSI"] < rsi_midline and current["RSI"] >= rsi_midline
    else:
        rsi_crossed = prev["RSI"] > rsi_midline and current["RSI"] <= rsi_midline
    if rsi_crossed:
        conditions_met += 1

    # Condition 2: Price 50%+ retraced toward SL AND stalled
    if direction == "long":
        retrace = (entry_price - current["close"]) / sl_dist
    else:
        retrace = (current["close"] - entry_price) / sl_dist

    if retrace >= sl_retrace_pct and len(candles_in_trade) >= stall_candles:
        recent = candles_in_trade.iloc[-stall_candles:]
        if direction == "long":
            no_progress = recent["close"].max() <= candles_in_trade.iloc[-stall_candles - 1]["close"] \
                if len(candles_in_trade) > stall_candles else True
        else:
            no_progress = recent["close"].min() >= candles_in_trade.iloc[-stall_candles - 1]["close"] \
                if len(candles_in_trade) > stall_candles else True
        if no_progress:
            conditions_met += 1

    # Condition 3: Opposing signal fired
    if direction == "long" and current.get("signal_short", False):
        conditions_met += 1
    elif direction == "short" and current.get("signal_long", False):
        conditions_met += 1

    return conditions_met >= 2
