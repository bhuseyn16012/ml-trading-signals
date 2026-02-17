"""
backtest.py — Strategy Backtesting Engine

Simulates trading using the model's buy/sell signals and calculates
professional financial performance metrics.

Strategy logic:
    - Model predicts BUY (1) → we are invested (capture that day's return)
    - Model predicts SELL (0) → we are in cash (0% return)

Includes:
    - Transaction cost modeling (10bps per trade)
    - Comparison against buy-and-hold benchmark

Usage:
    from backtest import run_backtest
    metrics, equity_df = run_backtest(predictions_df, initial_capital=100000)
"""

import pandas as pd
import numpy as np


def run_backtest(
    predictions_df: pd.DataFrame,
    initial_capital: float = 100_000,
    transaction_cost_bps: float = 10.0,
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
    transaction_cost_bps : float
        Cost per trade in basis points (default: 10 = 0.1%).
        Applied each time the model switches position (BUY→SELL or SELL→BUY).

    Returns
    -------
    tuple of (metrics_dict, equity_df)
        - metrics_dict: performance metrics for both strategy and benchmark
        - equity_df: daily equity curve for plotting
    """
    df = predictions_df.copy()
    tc_rate = transaction_cost_bps / 10_000  # Convert bps to decimal

    # --- Calculate daily returns ---
    df["Daily_Return"] = df["Close"].pct_change()

    # --- Detect position changes (trades) ---
    # A trade occurs when the signal changes from 0→1 or 1→0
    df["Position_Change"] = df["Predicted"].diff().abs().fillna(0)
    total_trades = int(df["Position_Change"].sum())

    # --- Strategy returns with transaction costs ---
    # On BUY days: capture the return minus any transaction cost
    # On SELL days: 0% return (in cash), minus transaction cost if we just sold
    df["Trade_Cost"] = df["Position_Change"] * tc_rate
    df["Strategy_Return"] = (df["Daily_Return"] * df["Predicted"]) - df["Trade_Cost"]

    # --- Benchmark returns: buy and hold (no costs after initial purchase) ---
    df["Benchmark_Return"] = df["Daily_Return"]

    # Drop the first row (no return to calculate)
    df = df.iloc[1:].copy()

    # --- Build equity curves ---
    df["Strategy_Equity"] = initial_capital * (1 + df["Strategy_Return"]).cumprod()
    df["Benchmark_Equity"] = initial_capital * (1 + df["Benchmark_Return"]).cumprod()

    # --- Calculate metrics ---
    strategy_metrics = _calculate_metrics(
        df["Strategy_Return"], df["Strategy_Equity"], "ML Strategy"
    )
    benchmark_metrics = _calculate_metrics(
        df["Benchmark_Return"], df["Benchmark_Equity"], "Buy & Hold"
    )

    # --- Additional strategy-specific metrics ---
    trading_days = df["Predicted"].sum()
    total_days = len(df)

    traded_returns = df.loc[df["Predicted"] == 1, "Daily_Return"]
    win_rate = (traded_returns > 0).mean() if len(traded_returns) > 0 else 0

    gross_profits = traded_returns[traded_returns > 0].sum()
    gross_losses = abs(traded_returns[traded_returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf

    total_cost = df["Trade_Cost"].sum() * initial_capital
    strategy_metrics.update({
        "Trading Days": int(trading_days),
        "Total Days": int(total_days),
        "Exposure (%)": trading_days / total_days * 100,
        "Win Rate (%)": win_rate * 100,
        "Profit Factor": profit_factor,
        "Total Trades": total_trades,
        "Transaction Costs ($)": total_cost,
    })

    # --- Print summary ---
    _print_summary(strategy_metrics, benchmark_metrics, transaction_cost_bps)

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
    """
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    # Annualized return (assuming 252 trading days/year)
    n_days = len(returns)
    n_years = n_days / 252
    annualized_return = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100

    # Sharpe Ratio: annualized (return / risk)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Maximum Drawdown
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


def _print_summary(strategy: dict, benchmark: dict,
                    transaction_cost_bps: float) -> None:
    """Print a formatted comparison of strategy vs benchmark."""

    print("\n" + "=" * 65)
    print("                    BACKTEST RESULTS")
    print(f"          (Transaction costs: {transaction_cost_bps:.0f} bps per trade)")
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
    print(f"    Trading Days:     {strategy['Trading Days']:,} / "
          f"{strategy['Total Days']:,} "
          f"({strategy['Exposure (%)']:.1f}% exposure)")
    print(f"    Win Rate:         {strategy['Win Rate (%)']:.1f}%")
    print(f"    Profit Factor:    {strategy['Profit Factor']:.2f}")
    print(f"    Total Trades:     {strategy['Total Trades']:,}")
    print(f"    Transaction Costs: ${strategy['Transaction Costs ($)']:,.2f}")
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
