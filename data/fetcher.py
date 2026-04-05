"""
Historical OHLCV fetcher with local Parquet cache.
Supports two sources:
  - Bybit REST API (primary, requires access)
  - yfinance GC=F gold futures (fallback, no API key needed)
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from config import settings


def check_symbol(symbol: str = settings.SYMBOL) -> None:
    """Quick check to verify the symbol exists and API is reachable."""
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": "1", "limit": 1}
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    print(f"API check: retCode={data.get('retCode')}  retMsg={data.get('retMsg')}")
    rows = data.get("result", {}).get("list", [])
    print(f"Sample candle: {rows[0] if rows else 'NONE — symbol may not exist on Bybit linear'}")


def _fetch_chunk(symbol: str, interval: str, start_ms: int, end_ms: int,
                 category: str = "linear") -> list:
    """Fetch one chunk of kline data from Bybit REST API."""
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "start": start_ms,
        "end": end_ms,
        "limit": 1000,
    }
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") == 0:
                return data["result"]["list"]
            else:
                print(f"  API error: {data.get('retMsg')} (retCode={data.get('retCode')})")
                return []
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return []


def fetch_historical(
    symbol: str = settings.SYMBOL,
    interval: str = settings.INTERVAL,
    years: int = settings.BACKTEST_YEARS,
    cache_path: str = settings.CACHE_FILE,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data and cache as Parquet.
    On subsequent runs, only fetches missing candles (incremental update).
    Returns a DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_dt = datetime.now(timezone.utc) - timedelta(days=years * 365)
    start_ms = int(start_dt.timestamp() * 1000)

    # Load existing cache
    existing_df = pd.DataFrame()
    if os.path.exists(cache_path):
        existing_df = pd.read_parquet(cache_path)
        if not existing_df.empty:
            last_ts = int(existing_df["timestamp"].max())
            start_ms = last_ts + 60_000  # resume from next candle
            if start_ms >= now_ms - 60_000:
                print(f"Cache is up to date. {len(existing_df):,} candles loaded.")
                return existing_df
            print(f"Cache found ({len(existing_df):,} candles). Fetching from {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}...")

    # Fetch in 1000-candle chunks (1000 minutes = ~16.7 hours per chunk)
    chunk_ms = 1000 * 60 * 1000  # 1000 minutes in milliseconds
    all_rows = []
    current_start = start_ms
    total_expected = (now_ms - start_ms) // 60_000

    print(f"Fetching ~{total_expected:,} candles for {symbol} ({years}y of 1m data)...")

    consecutive_empty = 0
    while current_start < now_ms:
        current_end = min(current_start + chunk_ms, now_ms)
        chunk = _fetch_chunk(symbol, interval, current_start, current_end)
        if not chunk:
            consecutive_empty += 1
            current_start = current_end + 1  # skip ahead instead of stopping
            if consecutive_empty == 1:
                print(f"  No data at {datetime.fromtimestamp(current_start/1000, tz=timezone.utc).strftime('%Y-%m-%d')} — skipping forward...")
            if consecutive_empty >= 50:  # give up after 50 empty chunks (~35 days of gaps)
                print(f"  50 consecutive empty chunks — stopping.")
                break
            time.sleep(0.05)
            continue
        consecutive_empty = 0
        all_rows.extend(chunk)
        current_start = current_end + 1
        fetched = len(all_rows)
        print(f"  Fetched {fetched:,} / ~{total_expected:,} candles...", end="\r")
        time.sleep(0.05)  # respect rate limit

    print(f"\nFetched {len(all_rows):,} new candles.")

    if not all_rows:
        return existing_df

    # Bybit returns [startTime, open, high, low, close, volume, turnover]
    new_df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    new_df = new_df.drop(columns=["turnover"])
    new_df = new_df.astype({
        "timestamp": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
    })
    new_df = new_df.sort_values("timestamp").reset_index(drop=True)

    # Merge with existing cache
    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    else:
        combined = new_df

    # Add human-readable datetime column
    combined["datetime"] = pd.to_datetime(combined["timestamp"], unit="ms", utc=True)

    # Save to Parquet
    combined.to_parquet(cache_path, index=False)
    print(f"Saved {len(combined):,} candles to {cache_path}")
    return combined


def fetch_yfinance(
    years: int = settings.BACKTEST_YEARS,
    cache_path: str = settings.CACHE_FILE,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Fetch XAUUSD-equivalent data from yfinance using GC=F (Gold Futures).
    GC=F tracks XAUUSD spot price very closely (~$2-5 difference).

    yfinance limits:
      interval='1m'  → last 7 days only
      interval='5m'  → last 60 days
      interval='1h'  → last 730 days (2 years) ← recommended for full backtest
      interval='1d'  → years of data

    interval param maps to yfinance interval strings: '1m','5m','15m','1h','1d'
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Map friendly names
    yf_interval_map = {"1": "1m", "5": "5m", "15": "15m", "60": "1h", "D": "1d"}
    yf_interval = yf_interval_map.get(interval, interval)

    # yfinance period strings
    period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max"}
    period = period_map.get(yf_interval, "730d")

    print(f"Fetching GC=F (Gold Futures) from yfinance: interval={yf_interval}, period={period}")
    print("Note: GC=F closely tracks XAUUSD spot price.")

    ticker = yf.Ticker("GC=F")
    raw = ticker.history(period=period, interval=yf_interval, auto_adjust=True)

    if raw.empty:
        raise ValueError("yfinance returned no data. Check your internet connection.")

    raw = raw.reset_index()

    # Normalize column names
    time_col = "Datetime" if "Datetime" in raw.columns else "Date"
    df = pd.DataFrame({
        "datetime": pd.to_datetime(raw[time_col], utc=True),
        "open":     raw["Open"].astype("float64"),
        "high":     raw["High"].astype("float64"),
        "low":      raw["Low"].astype("float64"),
        "close":    raw["Close"].astype("float64"),
        "volume":   raw["Volume"].astype("float64"),
    })

    # Use .timestamp() for reliable ms conversion regardless of pandas datetime precision
    df["timestamp"] = df["datetime"].apply(lambda x: int(x.timestamp() * 1000))
    df = df.sort_values("timestamp").reset_index(drop=True)

    df.to_parquet(cache_path, index=False)
    print(f"Saved {len(df):,} candles ({yf_interval}) to {cache_path}")
    print(f"Date range: {df['datetime'].min().strftime('%Y-%m-%d')} -> {df['datetime'].max().strftime('%Y-%m-%d')}")
    return df


def load_cached(cache_path: str = settings.CACHE_FILE) -> pd.DataFrame:
    """Load cached data without fetching."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"No cache found at {cache_path}. Run fetch_historical() first.")
    df = pd.read_parquet(cache_path)
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)
