# 📈 ML Trading Signal Generator

An end-to-end machine learning pipeline that generates buy/sell signals using XGBoost and technical indicator features, validated with walk-forward backtesting to prevent look-ahead bias.

## Overview

This project builds a binary classification model that predicts whether a stock's next-day return will be positive or negative. Unlike naive approaches that use random train/test splits, this pipeline uses **expanding-window walk-forward validation** — the same methodology used by professional quant researchers — to ensure all predictions are made on truly unseen future data.

The pipeline runs across **multiple tickers** (META, SPY) and **multiple confidence thresholds** (50%, 60%) to analyze how the model performs under different conditions.

### Key Features
- **20+ engineered technical features** across trend, momentum, volatility, volume, and return categories
- **Walk-forward validation** eliminates look-ahead bias by training strictly on past data
- **Multi-ticker, multi-threshold comparison** to analyze strategy behavior across different assets
- **Transaction cost modeling** (10bps per trade) for realistic performance estimates
- **Professional financial metrics**: Sharpe ratio, max drawdown, profit factor, win rate
- **Feature importance analysis** to understand what drives predictions

## Results

> Run `python main.py` to generate all results in the `results/` folder.

### Cross-Configuration Comparison
![Comparison Chart](results/comparison_chart.png)

### META — 50% Confidence Threshold
![META 50% Equity Curve](results/META_threshold_50/equity_curve.png)

### META — 60% Confidence Threshold
![META 60% Equity Curve](results/META_threshold_60/equity_curve.png)

### SPY — 50% Confidence Threshold
![SPY 50% Equity Curve](results/SPY_threshold_50/equity_curve.png)

### SPY — 60% Confidence Threshold
![SPY 60% Equity Curve](results/SPY_threshold_60/equity_curve.png)

### Feature Importance (META)
![Feature Importance](results/META_threshold_50/feature_importance.png)

## Strategy Analysis

### Key Findings

**1. Buy-and-hold is a tough benchmark during bull markets.**
META returned over 200% during the test period (2021–2025), driven by an exceptional AI-fueled rally. Any strategy that spends time in cash will underperform in a straight-up bull market — this is a well-known challenge in quantitative finance.

**2. Confidence threshold impacts trade frequency and risk.**
Raising the threshold from 50% to 60% reduces the number of trades and market exposure. This typically reduces both drawdown risk and total return. The trade-off between selectivity and opportunity cost is a core challenge in signal-based strategies.

**3. Different assets respond differently to the same model.**
SPY (S&P 500) is a more diversified, less volatile asset than META. The model's behavior on SPY vs META illustrates how the same technical features can have different predictive power depending on the underlying asset's characteristics.

**4. Transaction costs matter.**
The 10bps cost per trade may seem small, but it compounds — especially for strategies that trade frequently. This is why professional quant strategies focus heavily on minimizing turnover.

### Why the Model Underperforms Buy-and-Hold

Predicting daily stock direction is one of the hardest problems in quantitative finance. Technical indicators alone capture a limited set of market dynamics. The model faces several structural challenges:

- **Signal-to-noise ratio**: Daily returns are extremely noisy. Even a model with 52% accuracy can be profitable, but achieving consistent edge is difficult.
- **Regime dependence**: Patterns that work in high-volatility periods may fail in trending markets, and vice versa. Our features don't explicitly model market regime.
- **Feature limitations**: Technical indicators only capture price and volume dynamics. They miss fundamentals, sentiment, macroeconomic factors, and cross-asset correlations.

### Honest Self-Assessment

This model does not beat buy-and-hold. That's a realistic and instructive result — most academic models don't survive contact with real markets. The value of this project is in the methodology:
- Proper walk-forward validation (no look-ahead bias)
- Realistic transaction cost modeling
- Multi-asset comparison
- Transparent analysis of results

### Potential Improvements
- **Alternative data**: Add sentiment (news/social media), earnings data, or macroeconomic indicators
- **Regime detection**: Add a layer that identifies bull/bear/sideways markets and adapts the strategy
- **Position sizing**: Instead of all-in/all-out, scale position size by prediction confidence
- **Ensemble methods**: Combine multiple model types (XGBoost + LSTM + linear) for more robust signals
- **Longer history with regime labels**: Train on 10+ years but include explicit market regime features
- **Short selling**: Allow the model to profit from predicted down days, not just avoid them

## Methodology

### 1. Data
- **Source**: Yahoo Finance via `yfinance`
- **Tickers**: META (Meta Platforms), SPY (S&P 500 ETF)
- **Period**: 5 years of daily OHLCV data (~1,250 trading days per ticker)
- **Rationale**: 5 years balances sample size with regime relevance

### 2. Feature Engineering
Features are grouped into five categories, all computed using backward-looking rolling windows to prevent data leakage:

| Category | Features | Purpose |
|---|---|---|
| **Trend** | Moving Average ratios (5/10/20/50-day), MACD, MACD Signal, MACD Histogram | Capture directional bias |
| **Momentum** | RSI (14-day), Rate of Change (5/10/20-day) | Detect overbought/oversold conditions |
| **Volatility** | Bollinger Band position, ATR (% of price) | Measure risk and mean-reversion potential |
| **Volume** | Volume/MA ratio | Gauge conviction behind price moves |
| **Returns** | Lagged returns (1/3/5/10/20-day) | Short and medium-term momentum |

**Leakage prevention**: Raw price levels are excluded from the feature set. Only scale-invariant ratios and normalized values are used.

### 3. Model
- **Algorithm**: XGBoost (Gradient Boosted Trees)
- **Target**: Binary — will tomorrow's close be higher than today's?
- **Hyperparameters**: Conservative defaults (max_depth=4, learning_rate=0.05, subsample=0.8) to prioritize generalization. Parameters are intentionally **not** tuned on test data.

### 4. Walk-Forward Validation

```
Fold 1: [===TRAIN (Year 1)===]     [TEST (Year 2)]
Fold 2: [======TRAIN (Years 1-2)======]     [TEST (Year 3)]
Fold 3: [==========TRAIN (Years 1-3)==========]     [TEST (Year 4)]
Fold 4: [=============TRAIN (Years 1-4)==============]     [TEST (Year 5)]
```

Each fold trains **only on past data** and predicts the next unseen period. This mimics how a real trading system would operate.

### 5. Backtesting
- **Strategy**: Invest on days the model predicts UP; hold cash otherwise
- **Benchmark**: Buy-and-hold for the entire period
- **Transaction costs**: 10 basis points (0.1%) per trade
- **Starting capital**: $100,000

## Project Structure

```
ml-trading-signals/
├── main.py              # Entry point — runs full pipeline across all configs
├── data_loader.py       # Downloads and cleans stock price data
├── features.py          # Engineers technical indicator features
├── model.py             # XGBoost training with walk-forward validation
├── backtest.py          # Simulates trading with transaction costs
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── results/             # Generated after running main.py
    ├── comparison_summary.csv
    ├── comparison_chart.png
    ├── META_threshold_50/
    │   ├── equity_curve.png
    │   ├── feature_importance.png
    │   ├── signals.png
    │   └── predictions.csv
    ├── META_threshold_60/
    ├── SPY_threshold_50/
    └── SPY_threshold_60/
```

## Installation & Usage

### Prerequisites
- Python 3.9+

### Setup
```bash
# Clone the repository
git clone https://github.com/bhuseyn16012/ml-trading-signals.git
cd ml-trading-signals

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

### Customization
Edit the configuration at the top of `main.py`:
```python
TICKERS = ["META", "SPY"]       # Add any Yahoo Finance tickers
THRESHOLDS = [0.5, 0.6]         # Add more threshold levels to compare
PERIOD = "5y"                    # "1y", "2y", "5y", "10y", "max"
INITIAL_CAPITAL = 100_000        # Starting portfolio value
```

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Past performance does not guarantee future results. Do not use this system for live trading without extensive additional validation, risk management, and regulatory compliance.
