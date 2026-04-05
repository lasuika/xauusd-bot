"""
Pure pandas/numpy indicator calculations.
All functions accept a DataFrame and return a new DataFrame with added columns.
"""

import numpy as np
import pandas as pd
from config import settings


def add_bollinger_bands(df: pd.DataFrame,
                        period: int = settings.BB_PERIOD,
                        std: float = settings.BB_STD) -> pd.DataFrame:
    """Add BB_mid, BB_upper, BB_lower columns."""
    df = df.copy()
    df["BB_mid"] = df["close"].rolling(period).mean()
    rolling_std = df["close"].rolling(period).std()
    df["BB_upper"] = df["BB_mid"] + std * rolling_std
    df["BB_lower"] = df["BB_mid"] - std * rolling_std
    return df


def add_rsi(df: pd.DataFrame, period: int = settings.RSI_PERIOD) -> pd.DataFrame:
    """Add RSI column using Wilder's smoothing (standard)."""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df: pd.DataFrame, period: int = settings.ATR_PERIOD) -> pd.DataFrame:
    """Add ATR column (Average True Range)."""
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return df


def add_volume_signal(df: pd.DataFrame,
                      multiplier: float = settings.VOLUME_MULTIPLIER,
                      period: int = settings.VOLUME_ROLLING_PERIOD) -> pd.DataFrame:
    """Add vol_avg and vol_spike (bool) columns."""
    df = df.copy()
    df["vol_avg"] = df["volume"].rolling(period).mean()
    df["vol_spike"] = df["volume"] > (multiplier * df["vol_avg"])
    return df


def add_trend_filter(df: pd.DataFrame,
                     ema_fast: int = settings.TREND_EMA_FAST,
                     ema_slow: int = settings.TREND_EMA_SLOW) -> pd.DataFrame:
    """
    Add EMA trend filter columns:
      trend_up   = close > EMA_fast > EMA_slow  (bullish — longs only)
      trend_down = close < EMA_fast < EMA_slow  (bearish — shorts only)
      trend_chop = neither                       (both directions allowed)
    """
    df = df.copy()
    df["EMA_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["EMA_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
    df["trend_up"]   = (df["close"] > df["EMA_fast"]) & (df["EMA_fast"] > df["EMA_slow"])
    df["trend_down"] = (df["close"] < df["EMA_fast"]) & (df["EMA_fast"] < df["EMA_slow"])
    return df


def add_all_indicators(df: pd.DataFrame,
                       bb_period: int = settings.BB_PERIOD,
                       bb_std: float = settings.BB_STD,
                       rsi_period: int = settings.RSI_PERIOD,
                       atr_period: int = settings.ATR_PERIOD,
                       vol_multiplier: float = settings.VOLUME_MULTIPLIER,
                       vol_period: int = settings.VOLUME_ROLLING_PERIOD) -> pd.DataFrame:
    """Compute all indicators in one pass. Returns df with all columns added."""
    df = add_bollinger_bands(df, bb_period, bb_std)
    df = add_rsi(df, rsi_period)
    df = add_atr(df, atr_period)
    df = add_volume_signal(df, vol_multiplier, vol_period)
    df = add_trend_filter(df)
    return df
