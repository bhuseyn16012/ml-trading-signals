"""
features.py — Feature Engineering for ML Trading Signals

Transforms raw OHLCV data into technical indicator features that
the XGBoost model uses to predict next-day price direction.

Feature categories:
    - Trend:      Moving averages, MACD
    - Momentum:   RSI, Rate of Change
    - Volatility: Bollinger Bands, ATR
    - Volume:     Volume moving average ratio
    - Returns:    Lagged returns over multiple windows

Usage:
    from data_loader import load_data
    from features import build_features
    df = load_data("META")
    df = build_features(df)
"""

import pandas as pd
import numpy as np


# =============================================================================
# Trend Indicators
# =============================================================================

def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple Moving Averages (5, 10, 20, 50 day) and price-to-MA ratios.

    Ratios tell the model: "Is the price above or below the trend?"
    - Ratio > 1 → price is above the average (bullish)
    - Ratio < 1 → price is below the average (bearish)
    """
    for window in [5, 10, 20, 50]:
        # Raw moving average
        df[f"MA_{window}"] = df["Close"].rolling(window=window).mean()
        # Price relative to MA (more useful for the model than raw MA value)
        df[f"Close_to_MA_{window}"] = df["Close"] / df[f"MA_{window}"]

    return df


def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).

    - MACD line = 12-day EMA - 26-day EMA
    - Signal line = 9-day EMA of MACD line
    - Histogram = MACD - Signal (positive = bullish momentum)
    """
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    return df


# =============================================================================
# Momentum Indicators
# =============================================================================

def _add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    RSI (Relative Strength Index) — 0 to 100 scale.

    - Above 70 → overbought (potential sell signal)
    - Below 30 → oversold (potential buy signal)

    Uses the Wilder smoothing method (exponential moving average).
    """
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def _add_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rate of Change — percentage price change over N periods.

    Captures momentum at multiple time scales.
    """
    for window in [5, 10, 20]:
        df[f"ROC_{window}"] = df["Close"].pct_change(periods=window) * 100

    return df


# =============================================================================
# Volatility Indicators
# =============================================================================

def _add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Bollinger Bands — volatility channel around a moving average.

    - Upper band = MA + 2 * std
    - Lower band = MA - 2 * std
    - BB_Position: where the price sits within the band (0 = lower, 1 = upper)

    BB_Position is the most useful feature — it normalizes across different
    price levels and volatility regimes.
    """
    ma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()

    df["BB_Upper"] = ma + (2 * std)
    df["BB_Lower"] = ma - (2 * std)
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    return df


def _add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    ATR (Average True Range) — measures daily price volatility.

    True Range = max of:
        - High - Low (today's range)
        - |High - Previous Close| (gap up)
        - |Low - Previous Close| (gap down)

    We normalize by Close to make it comparable across price levels.
    """
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=window).mean()
    # Normalized ATR (percentage of price) — better for the model
    df["ATR_Pct"] = df["ATR"] / df["Close"] * 100

    return df


# =============================================================================
# Volume Features
# =============================================================================

def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume relative to its moving average.

    - Ratio > 1 → above-average volume (strong conviction behind the move)
    - Ratio < 1 → below-average volume (weak move, might reverse)
    """
    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"]

    return df


# =============================================================================
# Return Features
# =============================================================================

def _add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lagged returns over multiple lookback windows.

    These capture:
    - Short-term momentum (1-5 days)
    - Medium-term trends (10-20 days)
    """
    for window in [1, 3, 5, 10, 20]:
        df[f"Return_{window}d"] = df["Close"].pct_change(periods=window) * 100

    return df


# =============================================================================
# Target Variable
# =============================================================================

def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary classification target: will tomorrow's return be positive?

    - 1 → tomorrow's close > today's close (BUY signal)
    - 0 → tomorrow's close <= today's close (SELL/HOLD signal)

    We use shift(-1) to look one day ahead. The last row will be NaN
    (we don't know tomorrow's return yet) and gets dropped.
    """
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df


# =============================================================================
# Master Function
# =============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps and return a clean DataFrame.

    Steps:
        1. Add all technical indicator features
        2. Add the target variable (next-day direction)
        3. Drop rows with NaN values (first ~50 rows due to MA_50 lookback)
        4. Drop raw price columns that would cause data leakage

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data from data_loader.load_data().

    Returns
    -------
    pd.DataFrame
        Feature matrix with target column, ready for model training.
    """
    df = df.copy()  # Don't modify the original

    # --- Add all features ---
    df = _add_moving_averages(df)
    df = _add_macd(df)
    df = _add_rsi(df)
    df = _add_rate_of_change(df)
    df = _add_bollinger_bands(df)
    df = _add_atr(df)
    df = _add_volume_features(df)
    df = _add_return_features(df)
    df = _add_target(df)

    # --- Drop rows with NaN (first ~50 rows + last 1 row) ---
    rows_before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[features] Created {len(get_feature_columns(df))} features. "
          f"Dropped {rows_before - len(df)} rows with NaN. "
          f"{len(df)} rows remaining.")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of feature column names (excludes Date, raw OHLCV, and Target).

    This ensures we never accidentally feed raw prices or the target
    variable into the model as features.
    """
    exclude = ["Date", "Open", "High", "Low", "Close", "Volume",
               "Volume_MA_20", "Target",
               "MA_5", "MA_10", "MA_20", "MA_50",  # raw MAs leak price level
               "BB_Upper", "BB_Lower", "ATR"]       # raw values leak price level
    return [col for col in df.columns if col not in exclude]


# ---- Quick test when run directly ----
if __name__ == "__main__":
    from data_loader import load_data

    df = load_data("META", period="5y")
    df = build_features(df)

    print("\nFeature columns:")
    for i, col in enumerate(get_feature_columns(df), 1):
        print(f"  {i:2d}. {col}")

    print(f"\nTarget distribution:")
    print(f"  Up days (1): {df['Target'].sum()} ({df['Target'].mean():.1%})")
    print(f"  Down days (0): {(1 - df['Target']).sum():.0f} ({1 - df['Target'].mean():.1%})")

    print(f"\nSample row:")
    print(df[get_feature_columns(df)].iloc[-1])
