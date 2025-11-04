# PriceDirectionLearner V1

This project is a complete, end-to-end experiment to answer one question:

> **Can I learn short-term price direction from historical market data and turn it into a tradeable signal?**

To do that, I built a small ML pipeline around daily stock prices and evaluated it with a realistic (time-based) backtest.

---

## 1. Motivation

I wanted to build a trading algorithm that is more than a fixed technical rule (like “buy when MA crosses”).  
The goal was to show that I can:

1. collect real market data,
2. engineer features that represent market behavior,
3. train a supervised model on past data **without leaking future information**, and
4. convert the model’s predictions into actual trading signals and measure performance.

This version (V1) is meant to be clear and explainable, not perfect or high-frequency.

---

## 2. Data

- **Source:** Yahoo Finance via the `yfinance` Python library  
- **Ticker used:** `AAPL` (Apple Inc.)  
- **Frequency:** Daily  
- **Duration:** Last **5 years**  
- **Columns:** Date, Open, High, Low, Close, Adj Close, Volume  
- **Why Yahoo Finance?** It’s a widely used, easily accessible source that mirrors exchange data closely enough for research and portfolio projects.

This dataset is large enough to capture different market regimes (rallies, pullbacks, sideways periods) but still small enough to train quickly.

---

## 3. System Architecture / Pipeline

The project follows a straight ML-for-trading pipeline:

1. **Data Ingestion**  
   - download daily OHLCV data from Yahoo Finance  
   - clean and drop missing values

2. **Feature Engineering**  
   - 1-day return (short-term momentum)  
   - moving average spread (MA10 − MA30) to capture trend  
   - 5-day volatility to capture recent risk

3. **Target Construction**  
   - binary label: `1` if tomorrow’s close > today’s close, else `0`  
   - turns the problem into a classification task

4. **Time-Based Train/Test Split**  
   - first 80% of the dates → training  
   - last 20% → testing  
   - avoids data leakage (model never sees the future)

5. **Model Training**  
   - model: **GradientBoostingClassifier** (from scikit-learn)

6. **Signal Generation & Backtesting**  
   - prediction = 1 → long  
   - prediction = 0 → flat  
   - apply signals with 1-day lag  
   - build strategy equity curve and compare to buy & hold

7. **Evaluation**  
   - classification metrics (accuracy, precision, recall)  
   - trading metrics (total return, max drawdown, volatility, Sharpe)  
   - Plotly charts for price and strategy performance

You can think of it as:

> **Data → Features → Label → Model → Signal → Backtest → Metrics**

---

## 4. Why Gradient Boosting?

I used **Gradient Boosting** because:

- it handles nonlinear relationships between features (trend + volatility together),
- it usually performs better than a single decision tree or pure logistic regression on small tabular datasets,
- it’s fast enough to train on daily data,
- it’s a standard model in many real ML trading setups.

A simpler model (like Logistic Regression) could be added as a baseline in V2.

---

## 5. What This Version Shows

- I can build a trading algorithm **from scratch** (not just call an API).
- I can avoid **time-series data leakage** by using a chronological split.
- I can turn model predictions into actual trades and evaluate them.
- I can explain why it didn’t beat buy-and-hold all the time (because the model predicts direction, not magnitude, and there are no costs yet).

---

## 6. How to Run

```bash
# create env (optional)
pip install -r requirements.txt

# open the notebook
jupyter notebook PriceDirectionLearnerV1.ipynb


