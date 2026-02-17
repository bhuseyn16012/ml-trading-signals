# Interview Prep Guide: ML Trading Signal Generator

## How to Introduce Your Project (30-second pitch)

"I built an end-to-end ML trading pipeline that uses XGBoost to predict daily stock direction based on technical indicators. The model uses walk-forward validation to prevent look-ahead bias, includes transaction cost modeling, and compares performance across multiple tickers and confidence thresholds. The model didn't beat buy-and-hold — which is actually the expected result for daily prediction with technicals alone — and the README includes an honest analysis of why and what I'd improve."

---

## Most Likely Interview Questions & How to Answer Them

### Q1: "Walk me through your project."

**Answer framework:** Data → Features → Model → Validation → Results → Learnings

"I start by downloading 5 years of daily OHLCV data from Yahoo Finance. Then I engineer over 20 technical features across five categories: trend indicators like moving average ratios, momentum indicators like RSI, volatility indicators like Bollinger Band position, volume ratios, and lagged returns.

I feed these into an XGBoost classifier that predicts whether tomorrow's close will be higher than today's. The key design choice is walk-forward validation — I use an expanding window where each fold only trains on past data and tests on the next unseen year. This prevents look-ahead bias.

I then backtest the signals with transaction cost modeling at 10 basis points per trade, and compare the strategy against buy-and-hold across META and SPY at different confidence thresholds."

---

### Q2: "Why did you use XGBoost?"

**Good answer:**
"XGBoost handles non-linear relationships between features well, which matters because financial signals often interact — for example, low RSI combined with high volume might predict a bounce, but neither alone is a strong signal. It's also robust to noise, handles mixed feature types well, and provides feature importance scores out of the box which help with interpretability. It's widely used in quantitative finance for tabular data."

**If they push further — "Why not a neural network?"**
"For tabular data with around 1,000 rows and 20 features, gradient boosted trees typically outperform neural networks. Deep learning shines with sequential data or very large datasets. If I were to add sequential modeling, I'd consider an LSTM on top of the XGBoost signals as an ensemble."

---

### Q3: "What is data leakage and how did you prevent it?"

**This is a critical question — nail this one.**

"Data leakage is when the model accidentally sees information during training that it wouldn't have access to at prediction time. In trading, the most dangerous form is look-ahead bias — using future data to predict the past.

I prevented it in four ways:
- First, I used walk-forward validation instead of random train/test splits, so the model only trains on past data and predicts the future.
- Second, I used scale-invariant features like price-to-MA ratios instead of raw prices, so the model learns patterns rather than memorizing price levels.
- Third, all feature calculations use backward-looking rolling windows — no future data leaks into the features.
- Fourth, I intentionally didn't tune hyperparameters on the test data. The conservative defaults are fixed before any testing."

---

### Q4: "Your model lost money. Why should I be impressed?"

**This is your chance to show maturity. Don't be defensive.**

"Predicting daily stock direction from technical indicators alone is one of the hardest problems in quantitative finance. If it were easy, everyone would do it. The model underperformed buy-and-hold primarily because META and SPY were in strong bull markets during the test period — any strategy that spends time in cash will underperform in a straight-up rally.

The value of this project isn't the returns — it's the methodology. Walk-forward validation, transaction cost modeling, multi-asset comparison, and honest analysis of why the strategy fails. Most student projects show suspiciously high returns because of overfitting or data leakage. I'd rather show a realistic result with sound methodology."

**Follow-up they might ask — "What would you change to improve it?"**

"Three things. First, I'd add alternative data sources like news sentiment or earnings surprises — technical indicators only capture price and volume dynamics. Second, I'd add a regime detection layer so the model knows whether it's in a bull, bear, or sideways market and adapts accordingly. Third, I'd allow short selling so the model can profit from predicted down days instead of just sitting in cash."

---

### Q5: "Explain walk-forward validation."

"In a normal train/test split, you randomly divide data — but with time series, that means you might train on December data and test on March data from the same year. The model has seen the future.

Walk-forward validation respects the time ordering. In fold one, I train on year one and test on year two. In fold two, I train on years one and two and test on year three. The training window expands each fold, but the test set is always strictly in the future. This mimics how you'd actually trade — you only know the past, and you're making predictions about tomorrow."

---

### Q6: "What does the Sharpe ratio mean and why does it matter?"

"Sharpe ratio is the risk-adjusted return — it's the average excess return divided by the standard deviation of returns, annualized. A Sharpe of 1 means you're getting 1 unit of return per unit of risk.

It matters more than total return because it accounts for risk. A strategy that returns 50% with 10% volatility (Sharpe ~5) is far better than one that returns 100% with 80% volatility (Sharpe ~1.25). Most hedge funds target a Sharpe above 1, and anything above 2 is considered excellent."

---

### Q7: "What are the most important features in your model?"

"The feature importance varies between META and SPY, which is itself an interesting finding. Generally, the lagged return features and RSI tend to rank highly — they capture recent momentum and overbought/oversold conditions. Moving average ratios also contribute, especially the shorter windows like the 5-day and 10-day.

One thing I'd note is that feature importance in XGBoost measures how often a feature is used for splits, not necessarily its predictive power in isolation. A feature could be important for the model's decision tree structure but not individually correlated with returns."

---

### Q8: "Why did you choose a 5-year lookback period?"

"It's a trade-off between having enough data for the model to learn and keeping the data relevant. With daily data, 5 years gives roughly 1,250 trading days — enough for XGBoost to find patterns across walk-forward folds. Going back further, say 10 years, risks including data from a very different market regime — META's business in 2014 was fundamentally different from today. Going shorter, like 2 years, doesn't give the model enough training data, especially since walk-forward validation consumes the first portion for the initial training set."

---

### Q9: "What's the difference between your 50% and 60% confidence thresholds?"

"At 50%, the model trades whenever it thinks there's any edge — essentially every day it believes the probability of an up move exceeds a coin flip. This gives high market exposure but includes a lot of low-conviction trades.

At 60%, the model only trades when it's more confident, which reduces the number of trades and market exposure. In theory this should improve win rate because you're filtering out the marginal signals. The trade-off is that you miss some profitable days where the model was only 55% confident. Comparing the two shows how the selectivity-opportunity trade-off plays out empirically."

---

### Q10: "How would you deploy this in production?"

"Several things would need to change. First, I'd need real-time data feeds instead of Yahoo Finance — something like polygon.io or a direct exchange feed. Second, I'd need an execution system connected to a broker API to actually place trades. Third, I'd add much more robust risk management — position sizing limits, maximum drawdown circuit breakers, and correlation monitoring if running multiple strategies.

I'd also need to retrain the model regularly as new data comes in, monitor for model drift, and have alerting if the model's live performance deviates significantly from backtest expectations. The current project is a research tool — production would require infrastructure that's a separate engineering challenge."

---

## Technical Concepts You Should Know Cold

**Alpha**: Returns above the benchmark. If SPY returned 10% and your strategy returned 12%, your alpha is 2%.

**Sharpe Ratio**: (Mean return - Risk-free rate) / Standard deviation of returns, annualized by multiplying by sqrt(252). Higher is better.

**Max Drawdown**: The largest peak-to-trough decline in portfolio value. Measures the worst-case pain.

**Win Rate**: Percentage of trades that were profitable. 55% is considered good for daily strategies.

**Profit Factor**: Gross profits divided by gross losses. Above 1.0 means you're making more than you're losing.

**Basis Points (bps)**: 1 basis point = 0.01%. So 10bps = 0.1%. Used to express small percentages like transaction costs.

**Overfitting**: When a model memorizes training data patterns (including noise) instead of learning generalizable signals. Shows great backtest results but fails on new data.

**Regime**: A market "state" — bull market, bear market, high volatility, low volatility. Strategies often work in one regime but fail in another.

---

## Red Flags to Avoid in Interviews

- Never claim your model "beats the market" if it doesn't
- Never say you used a random train/test split for time series data
- Never say "I just used the default parameters" without explaining why defaults are reasonable
- Never dismiss poor results — always explain why and what you'd improve
- Never say "I found this code online and modified it" — own every line

## Green Flags That Impress

- Explaining leakage prevention unprompted
- Acknowledging limitations honestly
- Having a clear plan for improvements
- Understanding why results look the way they do
- Connecting technical choices to real-world trading concerns
