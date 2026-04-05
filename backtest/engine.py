"""
Vectorized backtesting engine.
Simulates trade entries/exits on historical data with spread + slippage costs.
"""

import numpy as np
import pandas as pd
from typing import Optional
from config import settings
from strategy.signals import compute_exit_levels, check_smart_exit


def run_backtest(df: pd.DataFrame,
                 initial_equity: float = 1000.0,
                 risk_pct: float = settings.RISK_BASE_PCT,
                 sl_mult: float = settings.ATR_SL_MULTIPLIER,
                 tp_mult: float = settings.ATR_TP_MULTIPLIER,
                 spread_pips: float = settings.SPREAD_PIPS,
                 slippage_pips: float = settings.SLIPPAGE_PIPS,
                 max_concurrent: int = settings.MAX_CONCURRENT_POSITIONS) -> dict:
    """
    Run a full backtest on pre-processed DataFrame.
    df must have: signal_long, signal_short, ATR, RSI, close, high, low columns.
    Returns a dict with trades list and equity curve.
    """
    df = df.copy().reset_index(drop=True)
    equity = initial_equity
    trades = []
    equity_curve = [equity]

    # Track open positions: list of dicts
    open_positions = []

    total_cost_per_trade = (spread_pips + slippage_pips) * 2  # entry + exit

    for i in range(1, len(df)):
        row = df.iloc[i]

        # ── Check exits for open positions ────────────────────────────────────
        still_open = []
        for pos in open_positions:
            entry_idx = pos["entry_idx"]
            trade_candles = df.iloc[entry_idx:i + 1]

            direction = pos["direction"]
            tp = pos["take_profit"]
            sl = pos["stop_loss"]
            entry_price = pos["entry_price"]
            position_size = pos["position_size"]

            closed = False
            exit_price = None
            exit_reason = None

            # Check TP/SL hit within this candle
            if direction == "long":
                if row["low"] <= sl:
                    exit_price = sl
                    exit_reason = "stop_loss"
                    closed = True
                elif row["high"] >= tp:
                    exit_price = tp
                    exit_reason = "take_profit"
                    closed = True
            else:
                if row["high"] >= sl:
                    exit_price = sl
                    exit_reason = "stop_loss"
                    closed = True
                elif row["low"] <= tp:
                    exit_price = tp
                    exit_reason = "take_profit"
                    closed = True

            # Smart exit check (only if not already hit TP/SL)
            if not closed and len(trade_candles) >= 2:
                if check_smart_exit(trade_candles, entry_price, direction, sl):
                    exit_price = row["close"]
                    exit_reason = "smart_exit"
                    closed = True

            if closed:
                raw_pnl = (exit_price - entry_price) * position_size if direction == "long" \
                    else (entry_price - exit_price) * position_size
                cost = total_cost_per_trade * position_size
                net_pnl = raw_pnl - cost
                equity += net_pnl

                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_time": df.iloc[entry_idx]["datetime"],
                    "exit_time": row["datetime"],
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "take_profit": tp,
                    "stop_loss": sl,
                    "position_size": position_size,
                    "pnl": net_pnl,
                    "exit_reason": exit_reason,
                    "duration_candles": i - entry_idx,
                    "equity_after": equity,
                })
            else:
                still_open.append(pos)

        open_positions = still_open

        # ── Check new entries ──────────────────────────────────────────────────
        if len(open_positions) < max_concurrent:
            for direction, signal_col in [("long", "signal_long"), ("short", "signal_short")]:
                if not row.get(signal_col, False):
                    continue
                # Don't open same direction if already have one
                if any(p["direction"] == direction for p in open_positions):
                    continue
                if len(open_positions) >= max_concurrent:
                    break

                entry_price = row["close"]
                atr = row["ATR"]
                if pd.isna(atr) or atr == 0:
                    continue

                tp, sl = compute_exit_levels(entry_price, direction, atr, sl_mult, tp_mult)

                sl_dist = abs(entry_price - sl)
                risk_amount = equity * risk_pct
                position_size = risk_amount / sl_dist if sl_dist > 0 else 0

                if position_size <= 0:
                    continue

                open_positions.append({
                    "entry_idx": i,
                    "direction": direction,
                    "entry_price": entry_price,
                    "take_profit": tp,
                    "stop_loss": sl,
                    "position_size": position_size,
                    "atr_at_entry": atr,
                })

        equity_curve.append(equity)

    return {
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
        "equity_curve": pd.Series(equity_curve),
        "final_equity": equity,
        "initial_equity": initial_equity,
    }


def run_walkforward(df: pd.DataFrame,
                    initial_equity: float = 1000.0,
                    train_months: int = settings.WALKFORWARD_TRAIN_MONTHS,
                    test_months: int = settings.WALKFORWARD_TEST_MONTHS,
                    **backtest_kwargs) -> dict:
    """
    Walk-forward validation: train on first N months, test on remaining M months.
    Returns test period results only.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    split_date = df["datetime"].min() + pd.DateOffset(months=train_months)
    test_df = df[df["datetime"] >= split_date].reset_index(drop=True)

    print(f"Walk-forward: training ends {split_date.strftime('%Y-%m-%d')}")
    print(f"Test period: {test_df['datetime'].min().strftime('%Y-%m-%d')} → {test_df['datetime'].max().strftime('%Y-%m-%d')}")
    print(f"Test candles: {len(test_df):,}")

    return run_backtest(test_df, initial_equity=initial_equity, **backtest_kwargs)


def run_optimization(df: pd.DataFrame,
                     initial_equity: float = 1000.0,
                     strategies: list = None) -> pd.DataFrame:
    """
    Grid search over parameter combinations across all strategies.
    Returns ranked results by Sharpe.
    """
    from itertools import product
    from backtest.metrics import compute_metrics
    from strategy.indicators import add_all_indicators
    from strategy.session_filter import add_session_filter
    from strategy.signals import generate_entry_signals

    if strategies is None:
        strategies = ["mean_reversion", "vwap_rsi", "ema_momentum"]

    results = []
    grid = list(product(
        settings.OPT_BB_PERIODS,
        settings.OPT_BB_STDS,
        settings.OPT_RSI_OVERSOLD,
        settings.OPT_RSI_OVERBOUGHT,
        settings.OPT_ATR_SL_MULT,
        settings.OPT_VOLUME_MULT,
    ))

    total = len(grid) * len(strategies)
    print(f"Running grid search: {len(grid)} param combos x {len(strategies)} strategies = {total} tests...")

    tested = 0
    for strategy in strategies:
        print(f"\n  Testing strategy: {strategy}")
        for bb_p, bb_std, rsi_os, rsi_ob, sl_m, vol_m in grid:
            try:
                d = add_all_indicators(df, bb_period=bb_p, bb_std=bb_std, vol_multiplier=vol_m)
                d = add_session_filter(d)
                d = generate_entry_signals(d, rsi_oversold=rsi_os,
                                           rsi_overbought=rsi_ob, strategy=strategy)
                d = d.dropna().reset_index(drop=True)

                result = run_backtest(d, initial_equity=initial_equity, sl_mult=sl_m,
                                      tp_mult=sl_m * settings.ATR_TP_MULTIPLIER)
                m = compute_metrics(result)

                if m["total_trades"] < 10:
                    tested += 1
                    continue

                results.append({
                    "strategy": strategy,
                    "bb_period": bb_p,
                    "bb_std": bb_std,
                    "rsi_oversold": rsi_os,
                    "rsi_overbought": rsi_ob,
                    "sl_mult": sl_m,
                    "vol_mult": vol_m,
                    **m,
                })
            except Exception:
                pass
            tested += 1
            if tested % 50 == 0:
                print(f"  {tested}/{total} tested...", end="\r")

    print(f"\nOptimization complete. {len(results)} valid results.")
    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    return results_df
