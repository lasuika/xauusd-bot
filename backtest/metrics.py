"""
Performance metrics for backtest results.
"""

import numpy as np
import pandas as pd


def compute_metrics(result: dict) -> dict:
    """
    Compute full performance metrics from backtest result dict.
    Returns a flat dict of metric values.
    """
    trades = result["trades"]
    equity_curve = result["equity_curve"]
    initial_equity = result["initial_equity"]
    final_equity = result["final_equity"]

    if trades.empty or len(trades) < 5:
        return _empty_metrics()

    total_trades = len(trades)
    winners = trades[trades["pnl"] > 0]
    losers = trades[trades["pnl"] <= 0]

    win_rate = len(winners) / total_trades
    avg_win = winners["pnl"].mean() if len(winners) > 0 else 0
    avg_loss = abs(losers["pnl"].mean()) if len(losers) > 0 else 0
    profit_factor = (winners["pnl"].sum() / abs(losers["pnl"].sum())
                     if losers["pnl"].sum() != 0 else np.inf)

    total_return = (final_equity - initial_equity) / initial_equity

    # Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()

    # Sharpe (annualized, assuming 1-min equity curve, 525,600 minutes/year)
    returns = equity_curve.pct_change().dropna()
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(525_600)
    else:
        sharpe = 0.0

    # Trade frequency
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        delta = trades["exit_time"].max() - trades["entry_time"].min()
        duration_days = delta.total_seconds() / 86400
        trades_per_week = (total_trades / duration_days * 7) if duration_days > 0 else 0
        avg_duration_candles = trades["duration_candles"].mean()
    else:
        trades_per_week = 0
        avg_duration_candles = 0

    # Exit breakdown
    exit_counts = trades["exit_reason"].value_counts().to_dict() if "exit_reason" in trades.columns else {}

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "total_return": round(total_return, 4),
        "max_drawdown": round(max_drawdown, 4),
        "sharpe": round(sharpe, 3),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "rr_actual": round(avg_win / avg_loss, 3) if avg_loss > 0 else 0,
        "trades_per_week": round(trades_per_week, 1),
        "avg_duration_candles": round(avg_duration_candles, 1),
        "final_equity": round(final_equity, 2),
        "exit_tp": exit_counts.get("take_profit", 0),
        "exit_sl": exit_counts.get("stop_loss", 0),
        "exit_smart": exit_counts.get("smart_exit", 0),
    }


def _empty_metrics() -> dict:
    return {
        "total_trades": 0, "win_rate": 0, "profit_factor": 0,
        "total_return": 0, "max_drawdown": 0, "sharpe": -999,  # sorts to bottom
        "avg_win": 0, "avg_loss": 0, "rr_actual": 0,
        "trades_per_week": 0, "avg_duration_candles": 0,
        "final_equity": 0, "exit_tp": 0, "exit_sl": 0, "exit_smart": 0,
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print metrics to console."""
    targets = {
        "win_rate": (">=70%", lambda v: v >= 0.70),
        "trades_per_week": ("~400", lambda v: 300 <= v <= 500),
        "max_drawdown": ("<20%", lambda v: v > -0.20),
        "sharpe": (">2.0", lambda v: v > 2.0),
        "profit_factor": (">2.0", lambda v: v > 2.0),
    }

    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS")
    print("=" * 55)
    print(f"  Total Trades      : {metrics['total_trades']:,}")
    print(f"  Trades/Week       : {metrics['trades_per_week']:.1f}  (target: ~400)")

    wr = metrics['win_rate'] * 100
    wr_ok = "[OK]" if metrics['win_rate'] >= 0.70 else "[!!]"
    print(f"  Win Rate          : {wr:.1f}%  {wr_ok} (target: >=70%)")

    pf_ok = "[OK]" if metrics['profit_factor'] > 2.0 else "[!!]"
    print(f"  Profit Factor     : {metrics['profit_factor']:.3f}  {pf_ok} (target: >2.0)")

    sh_ok = "[OK]" if metrics['sharpe'] > 2.0 else "[!!]"
    print(f"  Sharpe Ratio      : {metrics['sharpe']:.3f}  {sh_ok} (target: >2.0)")

    dd = metrics['max_drawdown'] * 100
    dd_ok = "[OK]" if metrics['max_drawdown'] > -0.20 else "[!!]"
    print(f"  Max Drawdown      : {dd:.1f}%  {dd_ok} (target: <20%)")

    ret = metrics['total_return'] * 100
    print(f"  Total Return      : {ret:.1f}%")
    print(f"  Final Equity      : ${metrics['final_equity']:,.2f}")
    print(f"  Avg Win / Loss    : ${metrics['avg_win']:.4f} / ${metrics['avg_loss']:.4f}")
    print(f"  Actual RR         : {metrics['rr_actual']:.3f}:1")
    print(f"  Avg Duration      : {metrics['avg_duration_candles']:.0f} candles")
    print(f"  Exits: TP={metrics['exit_tp']}  SL={metrics['exit_sl']}  Smart={metrics['exit_smart']}")
    print("=" * 55 + "\n")
