"""
main.py — ML Trading Signal Generator: Full Pipeline

Run this single script to execute the entire pipeline:
    1. Download META historical data
    2. Engineer technical indicator features
    3. Train XGBoost with walk-forward validation
    4. Backtest the strategy against buy-and-hold
    5. Generate performance charts

Usage:
    python main.py

Output:
    - Console: performance metrics and model summary
    - results/equity_curve.png: portfolio growth over time
    - results/feature_importance.png: which features drive predictions
    - results/signals.png: buy/sell signals overlaid on price chart
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

TICKER = "META"
PERIOD = "5y"
INITIAL_CAPITAL = 100_000
CONFIDENCE_THRESHOLD = 0.5  # Only trade when model is >50% confident
RESULTS_DIR = "results"


# =============================================================================
# Chart Generation
# =============================================================================

def plot_equity_curve(equity_df: pd.DataFrame, metrics: dict, save_path: str):
    """
    Plot portfolio value over time: ML Strategy vs Buy & Hold.

    This is the most important chart — it shows at a glance whether
    the model adds value over passive investing.
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

    ax1.set_title(f"{TICKER} — ML Strategy vs Buy & Hold", fontsize=16, fontweight="bold")
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
    print(f"[main] Saved equity curve → {save_path}")


def plot_feature_importance(feature_importance: pd.DataFrame, save_path: str):
    """
    Horizontal bar chart of top 15 features by importance.

    This chart tells recruiters: "I don't just run models —
    I understand what's driving the predictions."
    """
    top_n = feature_importance.head(15).iloc[::-1]  # Reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_n)))
    ax.barh(top_n["Feature"], top_n["Importance"], color=colors)
    ax.set_xlabel("Importance (avg across walk-forward folds)", fontsize=12)
    ax.set_title("Top 15 Features by Importance", fontsize=16, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[main] Saved feature importance → {save_path}")


def plot_signals(equity_df: pd.DataFrame, save_path: str):
    """
    Price chart with BUY signals marked.

    Shows when the model was "in the market" vs sitting in cash.
    Green shading = invested, white = cash.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = pd.to_datetime(equity_df["Date"])
    # Reconstruct close price from equity returns
    close = equity_df["Benchmark_Equity"]  # Proportional to price

    ax.plot(dates, close, color="#1F2937", linewidth=1, label=f"{TICKER} Price (scaled)")

    # Shade BUY periods in green
    buy_mask = equity_df["Predicted"] == 1
    ax.fill_between(dates, close.min() * 0.95, close.max() * 1.05,
                     where=buy_mask, color="#10B981", alpha=0.12, label="Invested (BUY)")

    ax.set_title(f"{TICKER} — Model Trading Signals", fontsize=16, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[main] Saved signals chart → {save_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """Execute the full ML trading pipeline."""

    print("=" * 65)
    print(f"  ML Trading Signal Generator — {TICKER}")
    print(f"  Period: {PERIOD} | Capital: ${INITIAL_CAPITAL:,.0f}")
    print("=" * 65)

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Step 1: Load data ---
    print("\n📥 STEP 1: Loading data...")
    df = load_data(TICKER, period=PERIOD)

    # --- Step 2: Engineer features ---
    print("\n🔧 STEP 2: Engineering features...")
    df = build_features(df)
    feature_cols = get_feature_columns(df)

    # --- Step 3: Train model ---
    print("\n🤖 STEP 3: Training XGBoost (walk-forward validation)...")
    model_results = train_walk_forward(
        df, feature_cols, confidence_threshold=CONFIDENCE_THRESHOLD
    )

    # --- Step 4: Backtest ---
    print("\n📊 STEP 4: Running backtest...")
    metrics, equity_df = run_backtest(
        model_results["predictions"], initial_capital=INITIAL_CAPITAL
    )

    # --- Step 5: Generate charts ---
    print("\n📈 STEP 5: Generating charts...")
    plot_equity_curve(equity_df, metrics,
                      os.path.join(RESULTS_DIR, "equity_curve.png"))
    plot_feature_importance(model_results["feature_importance"],
                           os.path.join(RESULTS_DIR, "feature_importance.png"))
    plot_signals(equity_df,
                 os.path.join(RESULTS_DIR, "signals.png"))

    # --- Step 6: Save predictions to CSV ---
    model_results["predictions"].to_csv(
        os.path.join(RESULTS_DIR, "predictions.csv"), index=False
    )
    model_results["feature_importance"].to_csv(
        os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False
    )
    print(f"\n[main] Saved predictions & feature importance CSVs → {RESULTS_DIR}/")

    # --- Done ---
    print("\n" + "=" * 65)
    print("  ✅ Pipeline complete! Check the results/ folder for outputs.")
    print("=" * 65)


if __name__ == "__main__":
    main()
