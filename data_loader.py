"""
data_loader.py — Stock Price Data Retrieval & Cleaning

Downloads historical OHLCV data from Yahoo Finance and returns
a clean pandas DataFrame ready for feature engineering.

Usage:
    from data_loader import load_data
    df = load_data("META", period="5y")
"""

import yfinance as yf
import pandas as pd


def load_data(ticker: str = "META", period: str = "5y") -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (default: "META").
    period : str
        How far back to fetch. Options: "1y", "2y", "5y", "10y", "max"
        (default: "5y" — balances sample size with regime relevance).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Volume.
        Indexed by integer (Date is a column, not the index).
    """
    print(f"[data_loader] Downloading {period} of daily data for {ticker}...")

    # Download from Yahoo Finance
    raw = yf.download(ticker, period=period, interval="1d", progress=False)

    # yfinance sometimes returns MultiIndex columns for single tickers — flatten
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only the columns we need
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = raw[required_cols].copy()

    # Reset index so Date becomes a regular column
    df.reset_index(inplace=True)

    # Drop any rows with missing data (weekends/holidays already excluded by yfinance)
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)

    if rows_before != rows_after:
        print(f"[data_loader] Dropped {rows_before - rows_after} rows with missing values.")

    print(f"[data_loader] Loaded {len(df)} trading days "
          f"({df['Date'].iloc[0].strftime('%Y-%m-%d')} to "
          f"{df['Date'].iloc[-1].strftime('%Y-%m-%d')})")

    return df


# ---- Quick test when run directly ----
if __name__ == "__main__":
    df = load_data("META", period="5y")
    print("\nFirst 5 rows:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
