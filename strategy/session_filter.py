"""
ATR percentile-based session filter.
Marks candles as tradeable only when ATR is in the top Nth percentile
of the rolling 7-day window — naturally capturing high-liquidity sessions.
"""

import pandas as pd
from config import settings


def add_session_filter(df: pd.DataFrame,
                       window_days: int = settings.SESSION_ATR_WINDOW_DAYS,
                       percentile: int = settings.SESSION_ATR_PERCENTILE) -> pd.DataFrame:
    """
    Add 'session_active' boolean column.
    True when current ATR > rolling percentile threshold.
    Requires 'ATR' column to already exist (from indicators.add_atr).
    """
    if "ATR" not in df.columns:
        raise ValueError("ATR column missing. Run add_atr() first.")

    df = df.copy()
    window_candles = window_days * 24 * 60  # 7 days of 1-min candles

    # Rolling percentile threshold
    df["ATR_threshold"] = (
        df["ATR"]
        .rolling(window=window_candles, min_periods=window_candles // 7)
        .quantile(percentile / 100)
    )

    df["session_active"] = df["ATR"] > df["ATR_threshold"]
    return df
