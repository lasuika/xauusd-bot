"""
XAUUSD Scalping Bot — Backtesting Entry Point

Usage:
  python main_backtest.py --fetch-only                    # Bybit data (default)
  python main_backtest.py --fetch-only --source yfinance  # yfinance GC=F, no API key needed
  python main_backtest.py --fetch-only --source yfinance --yf-interval 1h  # hourly (2yr)
  python main_backtest.py --fetch-only --source yfinance --yf-interval 1m  # 1-min (7 days only)
  python main_backtest.py --backtest --equity 5000
  python main_backtest.py --optimize --equity 5000
  python main_backtest.py --walkforward --equity 5000
  python main_backtest.py --equity 5000                   # full pipeline
"""

import argparse
import sys
import pandas as pd

from data.fetcher import fetch_historical, fetch_yfinance, load_cached
from strategy.indicators import add_all_indicators
from strategy.session_filter import add_session_filter
from strategy.signals import generate_entry_signals
from backtest.engine import run_backtest, run_walkforward, run_optimization
from backtest.metrics import compute_metrics, print_metrics
from backtest.reporter import generate_full_report
from config import settings


def prepare_data(df: pd.DataFrame,
                 bb_period: int = settings.BB_PERIOD,
                 rsi_oversold: float = settings.RSI_OVERSOLD,
                 rsi_overbought: float = settings.RSI_OVERBOUGHT,
                 vol_multiplier: float = settings.VOLUME_MULTIPLIER) -> pd.DataFrame:
    """Add all indicators + signals, drop NaN rows."""
    print("Computing indicators...")
    df = add_all_indicators(df, bb_period=bb_period, vol_multiplier=vol_multiplier)
    df = add_session_filter(df)
    df = generate_entry_signals(df, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought)
    df = df.dropna(subset=["BB_lower", "BB_upper", "RSI", "ATR"]).reset_index(drop=True)
    print(f"Data ready: {len(df):,} candles with indicators")

    signals_long = df["signal_long"].sum()
    signals_short = df["signal_short"].sum()
    session_active = df["session_active"].sum()
    print(f"Signal summary | Long: {signals_long:,}  Short: {signals_short:,}  "
          f"Session active: {session_active:,} ({session_active/len(df)*100:.1f}%)")
    return df


def _apply_best_params(best) -> None:
    """Overwrite config/settings.py with the best optimization parameters."""
    import re
    path = "config/settings.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    replacements = {
        r"^BB_PERIOD\s*=.*$":         f"BB_PERIOD = {int(best['bb_period'])}",
        r"^RSI_OVERSOLD\s*=.*$":      f"RSI_OVERSOLD = {best['rsi_oversold']}",
        r"^RSI_OVERBOUGHT\s*=.*$":    f"RSI_OVERBOUGHT = {best['rsi_overbought']}",
        r"^ATR_SL_MULTIPLIER\s*=.*$": f"ATR_SL_MULTIPLIER = {best['sl_mult']}",
        r"^VOLUME_MULTIPLIER\s*=.*$": f"VOLUME_MULTIPLIER = {best['vol_mult']}",
    }

    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nconfig/settings.py updated with best parameters:")
    print(f"  BB_PERIOD={int(best['bb_period'])}  RSI={best['rsi_oversold']}/{best['rsi_overbought']}"
          f"  SL_MULT={best['sl_mult']}  VOL_MULT={best['vol_mult']}")


def cmd_fetch(args):
    print("=" * 55)
    print("  FETCHING HISTORICAL DATA")
    print("=" * 55)
    source = getattr(args, "source", "bybit")
    if source == "yfinance":
        yf_interval = getattr(args, "yf_interval", "1h")
        df = fetch_yfinance(interval=yf_interval)
    else:
        df = fetch_historical()
    print(f"\nData range: {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"Total candles: {len(df):,}")


def cmd_backtest(args):
    print("=" * 55)
    print("  RUNNING BACKTEST")
    print("=" * 55)
    df = load_cached()
    df = prepare_data(df)

    print("\nRunning backtest simulation...")
    result = run_backtest(df, initial_equity=args.equity)
    metrics = compute_metrics(result)
    print_metrics(metrics)
    generate_full_report(result, metrics)

    # Save trades to CSV
    if not result["trades"].empty:
        trades_path = f"{settings.BACKTEST_OUTPUT_DIR}/trades.csv"
        result["trades"].to_csv(trades_path, index=False)
        print(f"Trades saved to {trades_path}")


def cmd_walkforward(args):
    print("=" * 55)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 55)
    df = load_cached()
    df = prepare_data(df)

    print("\nRunning walk-forward test (test period only)...")
    result = run_walkforward(df, initial_equity=args.equity)
    metrics = compute_metrics(result)
    print("\n[WALK-FORWARD TEST PERIOD RESULTS]")
    print_metrics(metrics)
    generate_full_report(result, metrics)


def cmd_optimize(args):
    print("=" * 55)
    print("  PARAMETER OPTIMIZATION")
    print("=" * 55)
    df = load_cached()
    # Use base indicators only — optimizer will recompute per combination
    df = df.dropna().reset_index(drop=True)

    results = run_optimization(df, initial_equity=args.equity)

    print("\nTop 10 parameter combinations by Sharpe Ratio:")
    print("-" * 90)
    cols = ["bb_period", "rsi_oversold", "rsi_overbought", "sl_mult", "vol_mult",
            "win_rate", "sharpe", "profit_factor", "max_drawdown", "trades_per_week"]
    print(results[cols].head(10).to_string(index=False))

    # Save full results
    out_path = f"{settings.BACKTEST_OUTPUT_DIR}/optimization_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    # Print best params
    best = results.iloc[0]
    print("\n[BEST PARAMETERS]")
    print(f"  BB Period     : {int(best['bb_period'])}")
    print(f"  RSI Oversold  : {best['rsi_oversold']}")
    print(f"  RSI Overbought: {best['rsi_overbought']}")
    print(f"  ATR SL Mult   : {best['sl_mult']}")
    print(f"  Volume Mult   : {best['vol_mult']}")
    print(f"  Win Rate      : {best['win_rate']*100:.1f}%")
    print(f"  Sharpe        : {best['sharpe']:.3f}")

    if getattr(args, "apply_best", False):
        _apply_best_params(best)
        print("\nRe-running backtest with best parameters...")
        df2 = load_cached()
        df2 = prepare_data(df2)
        result = run_backtest(df2, initial_equity=args.equity,
                              sl_mult=best["sl_mult"],
                              tp_mult=best["sl_mult"] * settings.ATR_TP_MULTIPLIER)
        metrics = compute_metrics(result)
        print_metrics(metrics)
        generate_full_report(result, metrics)
    else:
        print(f"\nTip: rerun with --apply-best to auto-update settings and see final results.")


def cmd_full(args):
    print("=" * 55)
    print("  FULL PIPELINE: FETCH -> BACKTEST -> REPORT")
    print("=" * 55)
    source = getattr(args, "source", "bybit")
    if source == "yfinance":
        fetch_yfinance(interval=getattr(args, "yf_interval", "1h"))
    else:
        fetch_historical()
    df = load_cached()
    df = prepare_data(df)

    print("\nRunning backtest...")
    result = run_backtest(df, initial_equity=args.equity)
    metrics = compute_metrics(result)
    print_metrics(metrics)
    generate_full_report(result, metrics)

    if not result["trades"].empty:
        trades_path = f"{settings.BACKTEST_OUTPUT_DIR}/trades.csv"
        result["trades"].to_csv(trades_path, index=False)
        print(f"Trades CSV: {trades_path}")

    print(f"\nOpen backtest/output/report.html to view full report.")


def main():
    parser = argparse.ArgumentParser(description="XAUUSD Scalping Bot — Backtester")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch/update historical data")
    parser.add_argument("--backtest", action="store_true", help="Run backtest with current settings")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--optimize", action="store_true", help="Run parameter grid search")
    parser.add_argument("--equity", type=float, default=1000.0, help="Starting equity (default: 1000)")
    parser.add_argument("--apply-best", action="store_true",
                        help="After optimization, auto-update settings.py and re-run backtest")
    parser.add_argument("--source", choices=["bybit", "yfinance"], default="bybit",
                        help="Data source: bybit (default) or yfinance (no API key, works anywhere)")
    parser.add_argument("--yf-interval", default="1h",
                        choices=["1m", "5m", "15m", "1h", "1d"],
                        help="yfinance interval (default: 1h). 1m=7 days, 1h=2 years")
    args = parser.parse_args()

    if args.fetch_only:
        cmd_fetch(args)
    elif args.backtest:
        cmd_backtest(args)
    elif args.walkforward:
        cmd_walkforward(args)
    elif args.optimize:
        cmd_optimize(args)
    else:
        cmd_full(args)


if __name__ == "__main__":
    main()
