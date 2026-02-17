"""
backtest.py — Strategy Backtesting Engine

Simulates trading META using the model's buy/sell signals and calculates
professional financial performance metrics.

Strategy logic:
    - Model predicts BUY (1) → we are invested (capture that day's return)
    - Model predicts SELL (0) → we are in cash (0% return)

Benchmark:
    - Buy-and-hold META for the entire period

Usage:
    from backtest import run_backtest
    metrics, equity_df = run_backtest(predictions_df, initial_capital=100000)
"""

import pandas as pd
import numpy as np


def run_backtest(
    predictions_df: pd.DataFrame,
    initial_capital: float = 100_000,
) -> tuple:
    """
    Backtest the model's signals against buy-and-hold.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Output from model.train_walk_forward()["predictions"].
        Must have columns: Date, Close, Actual, Predicted, Probability.
    initial_capital : float
        Starting portfolio value (default: $100,000).

    Returns
    -------
    tuple of (metrics_dict, equity_df)
        - metrics_dict: performance metrics for both strategy and benchmark
        - equity_df: daily equity curve for plotting
    """
    df = predictions_df.copy()

    # --- Calculate daily returns ---
    df["Daily_Return"] = df["Close"].pct_change()

    # --- Strategy returns: only invested on BUY days ---
    # If model says BUY (1), we capture that day's return
    # If model says SELL (0), we're in cash (0% return)
    df["Strategy_Return"] = df["Daily_Return"] * df["Predicted"]

    # --- Benchmark returns: buy and hold the entire time ---
    df["Benchmark_Return"] = df["Daily_Return"]

    # Drop the first row (no return to calculate)
    df = df.iloc[1:].copy()

    # --- Build equity curves ---
    # Cumulative product of (1 + daily return) gives growth of $1
    df["Strategy_Equity"] = initial_capital * (1 + df["Strategy_Return"]).cumprod()
    df["Benchmark_Equity"] = initial_capital * (1 + df["Benchmark_Return"]).cumprod()

    # --- Calculate metrics for both strategy and benchmark ---
    strategy_metrics = _calculate_metrics(
        df["Strategy_Return"], df["Strategy_Equity"], "ML Strategy"
    )
    benchmark_metrics = _calculate_metrics(
        df["Benchmark_Return"], df["Benchmark_Equity"], "Buy & Hold"
    )

    # --- Additional strategy-specific metrics ---
    trading_days = df["Predicted"].sum()
    total_days = len(df)

    # Win rate: of the days we traded, how many were profitable?
    traded_returns = df.loc[df["Predicted"] == 1, "Daily_Return"]
    win_rate = (traded_returns > 0).mean() if len(traded_returns) > 0 else 0

    # Profit factor: gross profits / gross losses
    gross_profits = traded_returns[traded_returns > 0].sum()
    gross_losses = abs(traded_returns[traded_returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf

    strategy_metrics.update({
        "Trading Days": int(trading_days),
        "Total Days": int(total_days),
        "Exposure (%)": trading_days / total_days * 100,
        "Win Rate (%)": win_rate * 100,
        "Profit Factor": profit_factor,
    })

    # --- Print summary ---
    _print_summary(strategy_metrics, benchmark_metrics)

    # --- Prepare equity DataFrame for plotting ---
    equity_df = df[["Date", "Strategy_Equity", "Benchmark_Equity",
                     "Strategy_Return", "Benchmark_Return", "Predicted"]].copy()

    return {"strategy": strategy_metrics, "benchmark": benchmark_metrics}, equity_df


def _calculate_metrics(
    returns: pd.Series,
    equity: pd.Series,
    name: str,
) -> dict:
    """
    Calculate standard financial performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    equity : pd.Series
        Daily portfolio value.
    name : str
        Label for this strategy.

    Returns
    -------
    dict of metric_name: value
    """
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    # Annualized return (assuming 252 trading days/year)
    n_days = len(returns)
    n_years = n_days / 252
    annualized_return = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100

    # Sharpe Ratio: annualized (return / risk)
    # Risk-free rate assumed to be 0 for simplicity
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Maximum Drawdown: worst peak-to-trough decline
    cumulative_max = equity.cummax()
    drawdown = (equity - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100

    # Volatility (annualized)
    annual_volatility = returns.std() * np.sqrt(252) * 100

    return {
        "Name": name,
        "Total Return (%)": total_return,
        "Annualized Return (%)": annualized_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_drawdown,
        "Annual Volatility (%)": annual_volatility,
        "Final Value ($)": equity.iloc[-1],
    }


def _print_summary(strategy: dict, benchmark: dict) -> None:
    """Print a formatted comparison of strategy vs benchmark."""

    print("\n" + "=" * 65)
    print("                    BACKTEST RESULTS")
    print("=" * 65)

    print(f"\n{'Metric':<28} {'ML Strategy':>16} {'Buy & Hold':>16}")
    print("-" * 65)

    comparison_metrics = [
        ("Total Return (%)",       "{:>15.1f}%",  "{:>15.1f}%"),
        ("Annualized Return (%)",  "{:>15.1f}%",  "{:>15.1f}%"),
        ("Sharpe Ratio",           "{:>16.2f}",   "{:>16.2f}"),
        ("Max Drawdown (%)",       "{:>15.1f}%",  "{:>15.1f}%"),
        ("Annual Volatility (%)",  "{:>15.1f}%",  "{:>15.1f}%"),
        ("Final Value ($)",        "{:>12,.0f}   ", "{:>12,.0f}   "),
    ]

    for metric, fmt_s, fmt_b in comparison_metrics:
        s_val = strategy[metric]
        b_val = benchmark[metric]
        print(f"  {metric:<26} {fmt_s.format(s_val)} {fmt_b.format(b_val)}")

    print("-" * 65)
    print(f"\n  Strategy-specific:")
    print(f"    Trading Days:  {strategy['Trading Days']:,} / "
          f"{strategy['Total Days']:,} "
          f"({strategy['Exposure (%)']:.1f}% exposure)")
    print(f"    Win Rate:      {strategy['Win Rate (%)']:.1f}%")
    print(f"    Profit Factor: {strategy['Profit Factor']:.2f}")
    print("=" * 65)


# ---- Quick test when run directly ----
if __name__ == "__main__":
    from data_loader import load_data
    from features import build_features, get_feature_columns
    from model import train_walk_forward

    df = load_data("META", period="5y")
    df = build_features(df)
    feature_cols = get_feature_columns(df)

    results = train_walk_forward(df, feature_cols)
    metrics, equity_df = run_backtest(results["predictions"])

    print(f"\nEquity curve shape: {equity_df.shape}")
    print(f"Date range: {equity_df['Date'].iloc[0]} to {equity_df['Date'].iloc[-1]}")
