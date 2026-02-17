"""
main.py — ML Trading Signal Generator: Full Pipeline

Run this single script to execute the entire pipeline across
multiple tickers and confidence thresholds:
    1. Download historical data for each ticker
    2. Engineer technical indicator features
    3. Train XGBoost with walk-forward validation
    4. Backtest the strategy against buy-and-hold
    5. Generate performance charts
    6. Output a comparison summary across all configurations

Usage:
    python main.py

Output:
    - Console: performance metrics for each configuration
    - results/<TICKER>_threshold_<XX>/: charts and CSVs per config
    - results/comparison_summary.csv: all metrics side by side
    - results/comparison_chart.png: visual comparison across configs
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from data_loader import load_data
from features import build_features, get_feature_columns
from model import train_walk_forward
from backtest import run_backtest


# =============================================================================
# Configuration — easy to tweak
# =============================================================================

TICKERS = ["META", "SPY"]
THRESHOLDS = [0.5, 0.6]
PERIOD = "5y"
INITIAL_CAPITAL = 100_000
RESULTS_DIR = "results"


# =============================================================================
# Chart Generation
# =============================================================================

def plot_equity_curve(equity_df: pd.DataFrame, metrics: dict,
                      ticker: str, threshold: float, save_path: str):
    """
    Plot portfolio value over time: ML Strategy vs Buy & Hold.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1],
                              gridspec_kw={"hspace": 0.3})

    dates = pd.to_datetime(equity_df["Date"])

    # --- Top: Equity curves ---
    ax1 = axes[0]
    ax1.plot(dates, equity_df["Strategy_Equity"], color="#2563EB",
             linewidth=1.8, label="ML Strategy")
    ax1.plot(dates, equity_df["Benchmark_Equity"], color="#9CA3AF",
             linewidth=1.2, linestyle="--", label="Buy & Hold", alpha=0.8)
    ax1.fill_between(dates, equity_df["Strategy_Equity"],
                      equity_df["Benchmark_Equity"],
                      where=equity_df["Strategy_Equity"] >= equity_df["Benchmark_Equity"],
                      color="#2563EB", alpha=0.08)
    ax1.fill_between(dates, equity_df["Strategy_Equity"],
                      equity_df["Benchmark_Equity"],
                      where=equity_df["Strategy_Equity"] < equity_df["Benchmark_Equity"],
                      color="#EF4444", alpha=0.08)

    ax1.set_title(f"{ticker} — ML Strategy vs Buy & Hold (Threshold: {threshold:.0%})",
                  fontsize=16, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax1.legend(fontsize=11, loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.3)

    # Add metric annotations
    s = metrics["strategy"]
    b = metrics["benchmark"]
    textstr = (f"Strategy: {s['Total Return (%)']:.1f}% return | "
               f"Sharpe {s['Sharpe Ratio']:.2f} | "
               f"Max DD {s['Max Drawdown (%)']:.1f}%\n"
               f"Benchmark: {b['Total Return (%)']:.1f}% return | "
               f"Sharpe {b['Sharpe Ratio']:.2f} | "
               f"Max DD {b['Max Drawdown (%)']:.1f}%")
    ax1.text(0.02, 0.05, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    # --- Bottom: Drawdown ---
    ax2 = axes[1]
    strategy_cummax = equity_df["Strategy_Equity"].cummax()
    strategy_dd = (equity_df["Strategy_Equity"] - strategy_cummax) / strategy_cummax * 100
    benchmark_cummax = equity_df["Benchmark_Equity"].cummax()
    benchmark_dd = (equity_df["Benchmark_Equity"] - benchmark_cummax) / benchmark_cummax * 100

    ax2.fill_between(dates, strategy_dd, 0, color="#2563EB", alpha=0.3, label="Strategy DD")
    ax2.fill_between(dates, benchmark_dd, 0, color="#9CA3AF", alpha=0.2, label="Benchmark DD")
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved equity curve → {save_path}")


def plot_feature_importance(feature_importance: pd.DataFrame, ticker: str,
                            threshold: float, save_path: str):
    """
    Horizontal bar chart of top 15 features by importance.
    """
    top_n = feature_importance.head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_n)))
    ax.barh(top_n["Feature"], top_n["Importance"], color=colors)
    ax.set_xlabel("Importance (avg across walk-forward folds)", fontsize=12)
    ax.set_title(f"{ticker} — Top 15 Features (Threshold: {threshold:.0%})",
                 fontsize=16, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved feature importance → {save_path}")


def plot_signals(equity_df: pd.DataFrame, ticker: str,
                 threshold: float, save_path: str):
    """
    Price chart with BUY signals marked.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = pd.to_datetime(equity_df["Date"])
    close = equity_df["Benchmark_Equity"]

    ax.plot(dates, close, color="#1F2937", linewidth=1, label=f"{ticker} Price (scaled)")

    buy_mask = equity_df["Predicted"] == 1
    ax.fill_between(dates, close.min() * 0.95, close.max() * 1.05,
                     where=buy_mask, color="#10B981", alpha=0.12, label="Invested (BUY)")

    ax.set_title(f"{ticker} — Trading Signals (Threshold: {threshold:.0%})",
                 fontsize=16, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved signals chart → {save_path}")


def plot_comparison(summary_df: pd.DataFrame, save_path: str):
    """
    Bar chart comparing Sharpe ratio, total return, and max drawdown
    across all configurations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    labels = summary_df["Config"]
    x = np.arange(len(labels))
    width = 0.35

    color_strategy = "#2563EB"
    color_benchmark = "#9CA3AF"

    # --- Sharpe Ratio ---
    ax = axes[0]
    ax.bar(x - width/2, summary_df["Strategy Sharpe"], width,
           label="ML Strategy", color=color_strategy)
    ax.bar(x + width/2, summary_df["Benchmark Sharpe"], width,
           label="Buy & Hold", color=color_benchmark)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # --- Total Return ---
    ax = axes[1]
    ax.bar(x - width/2, summary_df["Strategy Return (%)"], width,
           label="ML Strategy", color=color_strategy)
    ax.bar(x + width/2, summary_df["Benchmark Return (%)"], width,
           label="Buy & Hold", color=color_benchmark)
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Total Return Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # --- Max Drawdown ---
    ax = axes[2]
    ax.bar(x - width/2, summary_df["Strategy Max DD (%)"], width,
           label="ML Strategy", color=color_strategy)
    ax.bar(x + width/2, summary_df["Benchmark Max DD (%)"], width,
           label="Buy & Hold", color=color_benchmark)
    ax.set_ylabel("Max Drawdown (%)")
    ax.set_title("Max Drawdown Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Strategy Performance Across Tickers & Confidence Thresholds",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[main] Saved comparison chart → {save_path}")


# =============================================================================
# Run Pipeline for One Configuration
# =============================================================================

def run_pipeline(ticker: str, threshold: float, period: str,
                 initial_capital: float, results_dir: str) -> dict:
    """
    Run the full pipeline for a single ticker + threshold combination.
    """
    config_name = f"{ticker}_threshold_{int(threshold * 100)}"
    config_dir = os.path.join(results_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Running: {ticker} | Threshold: {threshold:.0%}")
    print(f"{'='*65}")

    # Step 1: Load data
    print(f"\n  📥 Loading {ticker} data...")
    df = load_data(ticker, period=period)

    # Step 2: Engineer features
    print(f"  🔧 Engineering features...")
    df = build_features(df)
    feature_cols = get_feature_columns(df)

    # Step 3: Train model
    print(f"  🤖 Training XGBoost (walk-forward)...")
    model_results = train_walk_forward(df, feature_cols,
                                        confidence_threshold=threshold)

    # Step 4: Backtest
    print(f"  📊 Running backtest...")
    metrics, equity_df = run_backtest(model_results["predictions"],
                                      initial_capital=initial_capital)

    # Step 5: Generate charts
    print(f"  📈 Generating charts...")
    plot_equity_curve(equity_df, metrics, ticker, threshold,
                      os.path.join(config_dir, "equity_curve.png"))
    plot_feature_importance(model_results["feature_importance"], ticker, threshold,
                           os.path.join(config_dir, "feature_importance.png"))
    plot_signals(equity_df, ticker, threshold,
                 os.path.join(config_dir, "signals.png"))

    # Save CSVs
    model_results["predictions"].to_csv(
        os.path.join(config_dir, "predictions.csv"), index=False)
    model_results["feature_importance"].to_csv(
        os.path.join(config_dir, "feature_importance.csv"), index=False)

    # Return summary row
    s = metrics["strategy"]
    b = metrics["benchmark"]
    return {
        "Config": f"{ticker} ({threshold:.0%})",
        "Ticker": ticker,
        "Threshold": threshold,
        "Strategy Return (%)": round(s["Total Return (%)"], 1),
        "Strategy Sharpe": round(s["Sharpe Ratio"], 2),
        "Strategy Max DD (%)": round(s["Max Drawdown (%)"], 1),
        "Strategy Win Rate (%)": round(s["Win Rate (%)"], 1),
        "Strategy Profit Factor": round(s["Profit Factor"], 2),
        "Strategy Exposure (%)": round(s["Exposure (%)"], 1),
        "Benchmark Return (%)": round(b["Total Return (%)"], 1),
        "Benchmark Sharpe": round(b["Sharpe Ratio"], 2),
        "Benchmark Max DD (%)": round(b["Max Drawdown (%)"], 1),
        "Final Value ($)": round(s["Final Value ($)"], 0),
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """Execute the full ML trading pipeline across all configurations."""

    print("=" * 65)
    print("  ML Trading Signal Generator — Multi-Config Analysis")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print(f"  Thresholds: {', '.join(f'{t:.0%}' for t in THRESHOLDS)}")
    print(f"  Period: {PERIOD} | Capital: ${INITIAL_CAPITAL:,.0f}")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Run all configurations ---
    summary_rows = []
    for ticker in TICKERS:
        for threshold in THRESHOLDS:
            row = run_pipeline(ticker, threshold, PERIOD,
                               INITIAL_CAPITAL, RESULTS_DIR)
            summary_rows.append(row)

    # --- Build comparison summary ---
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[main] Saved comparison summary → {summary_path}")

    # --- Print comparison table ---
    print(f"\n{'='*90}")
    print("  COMPARISON SUMMARY — All Configurations")
    print(f"{'='*90}")
    print(f"\n{'Config':<18} {'Return':>10} {'Sharpe':>10} {'Max DD':>10} "
          f"{'Win Rate':>10} {'Exposure':>10} {'BH Return':>10}")
    print("-" * 90)
    for _, row in summary_df.iterrows():
        print(f"  {row['Config']:<16} {row['Strategy Return (%)']:>9.1f}% "
              f"{row['Strategy Sharpe']:>10.2f} {row['Strategy Max DD (%)']:>9.1f}% "
              f"{row['Strategy Win Rate (%)']:>9.1f}% {row['Strategy Exposure (%)']:>9.1f}% "
              f"{row['Benchmark Return (%)']:>9.1f}%")
    print(f"{'='*90}")

    # --- Generate comparison chart ---
    plot_comparison(summary_df,
                    os.path.join(RESULTS_DIR, "comparison_chart.png"))

    print("\n" + "=" * 65)
    print("  ✅ Pipeline complete! Check the results/ folder for outputs.")
    print("=" * 65)


if __name__ == "__main__":
    main()
