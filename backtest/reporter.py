"""
Generates visual reports from backtest results.
Outputs PNG charts + an HTML summary to backtest/output/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from config import settings


OUTPUT_DIR = settings.BACKTEST_OUTPUT_DIR


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_equity_curve(result: dict, metrics: dict) -> str:
    """Equity curve with drawdown overlay. Returns file path."""
    _ensure_output_dir()
    equity = result["equity_curve"]
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["top"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["right"].set_color("#30363d")

    ax1.plot(equity.values, color="#58a6ff", linewidth=1.2, label="Equity")
    ax1.fill_between(range(len(equity)), result["initial_equity"], equity.values,
                     alpha=0.15, color="#58a6ff")
    ax1.set_title(f"Equity Curve — Win Rate: {metrics['win_rate']*100:.1f}%  |  "
                  f"Sharpe: {metrics['sharpe']:.2f}  |  "
                  f"Max DD: {metrics['max_drawdown']*100:.1f}%",
                  color="#c9d1d9", fontsize=12)
    ax1.set_ylabel("Equity ($)", color="#c9d1d9")
    ax1.legend(facecolor="#21262d", labelcolor="#c9d1d9")
    ax1.yaxis.label.set_color("#c9d1d9")

    ax2.fill_between(range(len(drawdown)), 0, drawdown.values,
                     color="#f85149", alpha=0.7)
    ax2.set_ylabel("Drawdown %", color="#c9d1d9")
    ax2.set_xlabel("Candles", color="#c9d1d9")
    ax2.yaxis.label.set_color("#c9d1d9")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "equity_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def plot_monthly_heatmap(result: dict) -> str:
    """Monthly P&L heatmap. Returns file path."""
    _ensure_output_dir()
    trades = result["trades"]
    if trades.empty or "entry_time" not in trades.columns:
        return ""

    trades = trades.copy()
    trades["year"] = trades["entry_time"].dt.year
    trades["month"] = trades["entry_time"].dt.month
    monthly = trades.groupby(["year", "month"])["pnl"].sum().reset_index()
    pivot = monthly.pivot(index="year", columns="month", values="pnl").fillna(0)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 1.2)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    cmap = LinearSegmentedColormap.from_list("rg", ["#f85149", "#161b22", "#3fb950"])
    vmax = max(abs(pivot.values.max()), abs(pivot.values.min()), 1)
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, color="#c9d1d9")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color="#c9d1d9")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"${val:.0f}", ha="center", va="center",
                    color="white", fontsize=8)

    plt.colorbar(im, ax=ax, label="P&L ($)")
    ax.set_title("Monthly P&L Heatmap", color="#c9d1d9", fontsize=13)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "monthly_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def plot_hourly_distribution(result: dict) -> str:
    """Trade distribution by hour of day. Returns file path."""
    _ensure_output_dir()
    trades = result["trades"]
    if trades.empty or "entry_time" not in trades.columns:
        return ""

    trades = trades.copy()
    trades["hour"] = trades["entry_time"].dt.hour
    hourly_count = trades.groupby("hour").size().reindex(range(24), fill_value=0)
    hourly_pnl = trades.groupby("hour")["pnl"].sum().reindex(range(24), fill_value=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    ax1.bar(hourly_count.index, hourly_count.values, color="#58a6ff", alpha=0.8)
    ax1.set_title("Trade Count by Hour (UTC)", color="#c9d1d9")
    ax1.set_ylabel("Trades", color="#c9d1d9")
    ax1.set_xticks(range(24))

    colors = ["#3fb950" if v >= 0 else "#f85149" for v in hourly_pnl.values]
    ax2.bar(hourly_pnl.index, hourly_pnl.values, color=colors, alpha=0.8)
    ax2.set_title("P&L by Hour (UTC)", color="#c9d1d9")
    ax2.set_ylabel("P&L ($)", color="#c9d1d9")
    ax2.set_xlabel("Hour (UTC)", color="#c9d1d9")
    ax2.set_xticks(range(24))
    ax2.axhline(0, color="#30363d", linewidth=0.8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "hourly_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def generate_html_report(metrics: dict, chart_paths: dict) -> str:
    """Generate a standalone HTML report. Returns file path."""
    _ensure_output_dir()

    def badge(ok: bool) -> str:
        color = "#3fb950" if ok else "#f85149"
        symbol = "✓" if ok else "✗"
        return f'<span style="color:{color};font-weight:bold">{symbol}</span>'

    rows = [
        ("Total Trades", f"{metrics['total_trades']:,}", True),
        ("Trades / Week", f"{metrics['trades_per_week']:.1f}", 300 <= metrics['trades_per_week'] <= 500),
        ("Win Rate", f"{metrics['win_rate']*100:.2f}%", metrics['win_rate'] >= 0.70),
        ("Profit Factor", f"{metrics['profit_factor']:.3f}", metrics['profit_factor'] > 2.0),
        ("Sharpe Ratio", f"{metrics['sharpe']:.3f}", metrics['sharpe'] > 2.0),
        ("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%", metrics['max_drawdown'] > -0.20),
        ("Total Return", f"{metrics['total_return']*100:.2f}%", metrics['total_return'] > 0),
        ("Final Equity", f"${metrics['final_equity']:,.2f}", True),
        ("Avg Win", f"${metrics['avg_win']:.4f}", True),
        ("Avg Loss", f"${metrics['avg_loss']:.4f}", True),
        ("Actual RR", f"{metrics['rr_actual']:.3f}:1", True),
        ("Avg Duration", f"{metrics['avg_duration_candles']:.0f} candles", True),
        ("TP Exits", str(metrics['exit_tp']), True),
        ("SL Exits", str(metrics['exit_sl']), True),
        ("Smart Exits", str(metrics['exit_smart']), True),
    ]

    table_rows = "\n".join(
        f"<tr><td>{name}</td><td>{val}</td><td>{badge(ok)}</td></tr>"
        for name, val, ok in rows
    )

    imgs = ""
    for label, path in chart_paths.items():
        if path and os.path.exists(path):
            fname = os.path.basename(path)
            imgs += f'<h2>{label}</h2><img src="{fname}" style="max-width:100%;border-radius:8px;margin-bottom:24px">\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>XAUUSD Backtest Report</title>
<style>
  body {{ background:#0d1117; color:#c9d1d9; font-family:system-ui,sans-serif; padding:32px; }}
  h1 {{ color:#58a6ff; }} h2 {{ color:#79c0ff; margin-top:32px; }}
  table {{ border-collapse:collapse; width:420px; margin-bottom:32px; }}
  td {{ padding:8px 16px; border-bottom:1px solid #30363d; }}
  tr:hover {{ background:#161b22; }}
  td:first-child {{ color:#8b949e; }}
  td:last-child {{ text-align:center; }}
</style>
</head>
<body>
<h1>XAUUSD Scalping Bot — Backtest Report</h1>
<table>{table_rows}</table>
{imgs}
</body>
</html>"""

    path = os.path.join(OUTPUT_DIR, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved to {path}")
    return path


def generate_full_report(result: dict, metrics: dict) -> None:
    """Run all charts + HTML report."""
    print("Generating report...")
    charts = {
        "Equity Curve": plot_equity_curve(result, metrics),
        "Monthly P&L Heatmap": plot_monthly_heatmap(result),
        "Hourly Trade Distribution": plot_hourly_distribution(result),
    }
    generate_html_report(metrics, charts)
    print(f"Report saved to {OUTPUT_DIR}/")
