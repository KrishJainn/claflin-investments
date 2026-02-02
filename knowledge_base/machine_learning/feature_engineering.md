# Feature Engineering for Quantitative Trading

## Overview

Feature engineering is the process of transforming raw market data into informative inputs
for predictive models. In quantitative finance, feature quality is the primary determinant
of model performance -- far more important than model architecture. This document covers
the construction, selection, normalization, and validation of features for trading models.

---

## 1. Technical Indicators as Features

### 1.1 Trend Features
- **Moving averages**: SMA(n), EMA(n) for various lookbacks (5, 10, 20, 50, 200 days)
- **Moving average crossovers**: SMA(fast) - SMA(slow), normalized by volatility
  - MACD: EMA(12) - EMA(26), with signal line EMA(9) of the MACD
- **Price relative to moving average**: (Price - SMA(n)) / SMA(n)
  - Captures mean-reversion or momentum depending on lookback
- **Linear regression slope**: slope of price over trailing n bars, normalized by price level
- **ADX (Average Directional Index)**: trend strength (0-100), not direction

### 1.2 Momentum Features
- **Returns over multiple horizons**: r(1d), r(5d), r(21d), r(63d), r(252d)
  - Log returns preferred for stationarity: ln(P_t / P_{t-n})
- **Rate of change (ROC)**: (P_t - P_{t-n}) / P_{t-n}
- **RSI (Relative Strength Index)**: 100 - 100 / (1 + avg_gain / avg_loss) over n periods
- **Momentum ratios**: r(short_horizon) / r(long_horizon) -- captures acceleration
- **52-week high proximity**: P_t / max(P over trailing 252 days)

### 1.3 Volatility Features
- **Realized volatility**: std(log returns) over trailing n days, annualized
  - Multiple timescales: 5d, 10d, 21d, 63d
- **Volatility ratio**: vol(short) / vol(long) -- captures volatility regime shifts
- **ATR (Average True Range)**: average of max(H-L, |H-C_prev|, |L-C_prev|) over n bars
- **Garman-Klass volatility**: uses OHLC data for more efficient vol estimation
  - GK = 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
- **Parkinson volatility**: ln(H/L)^2 / (4 * ln(2))
- **Implied vs realized spread**: IV - RV as a risk premium feature

### 1.4 Volume Features
- **Volume ratio**: V_t / SMA(V, n) -- relative volume
- **On-Balance Volume (OBV)**: cumulative signed volume
- **Volume-price trend**: cumulative (return * volume)
- **Amihud illiquidity**: |return| / dollar_volume, averaged over n days
  - Higher = more illiquid; useful as a liquidity risk feature
- **Volume at price levels**: volume profile (POC, value area) as features

### 1.5 Order Book Features (for HFT/intraday)
- **Bid-ask spread**: (ask - bid) / mid, in basis points
- **Book imbalance**: (bid_vol - ask_vol) / (bid_vol + ask_vol) at top N levels
- **Weighted mid-price**: (bid * ask_vol + ask * bid_vol) / (bid_vol + ask_vol)
- **Depth slope**: rate at which cumulative volume increases away from mid-price
- **Trade flow imbalance**: net signed volume over trailing n seconds/minutes

---

## 2. Cross-Sectional Features

### 2.1 Relative Value
- **Cross-sectional z-score**: (x_i - mean(x)) / std(x) across universe of assets
  - Apply to any raw feature to create a relative ranking feature
- **Sector-neutral features**: demean features within each sector/industry
  - Removes common sector effects, isolates stock-specific signal
- **Percentile rank**: rank of asset's feature value within the universe (0 to 1)
  - Robust to outliers compared to z-scores

### 2.2 Factor Exposures
- **Market beta**: rolling regression of asset return on market return
- **Size**: log(market capitalization)
- **Value**: book-to-market ratio, earnings yield, cash flow yield
- **Quality**: ROE, debt/equity, earnings stability
- **Residual features**: orthogonalize features to known factors
  - Regress feature on factor exposures; use residual as the "pure" signal

### 2.3 Pair and Basket Features
- **Spread z-score**: (spread - mean(spread)) / std(spread) for pairs
  - spread = log(P_A) - beta * log(P_B) from cointegration regression
- **Correlation changes**: rolling corr(A, B) - long-term corr(A, B)
- **Relative strength**: return(A) - return(B) over various horizons
- **ETF vs components**: ETF price vs NAV; premium/discount as feature

---

## 3. Lagged Features and Rolling Statistics

### 3.1 Lagged Features
- Include features at multiple lags: f(t), f(t-1), f(t-2), ..., f(t-k)
- Captures autocorrelation and delayed effects
- Critical rule: NEVER include f(t) if it uses information not available at prediction time
  - E.g., using today's close price to predict today's close is lookahead bias
- Typical lags: 1, 2, 3, 5, 10, 21 trading days

### 3.2 Rolling Statistics
- **Rolling mean**: mean(feature) over trailing window of n bars
- **Rolling standard deviation**: std(feature) over trailing window
- **Rolling skewness**: captures asymmetry of recent feature distribution
- **Rolling kurtosis**: captures tail heaviness
- **Rolling quantiles**: median, 25th, 75th percentile over trailing window
- **Rolling z-score**: (current - rolling_mean) / rolling_std
  - Widely used for mean-reversion signals

### 3.3 Exponential Weighting
- EMA variants: give more weight to recent observations
- Halflife parameterization: hl = -ln(2) / ln(decay_factor)
  - halflife of 10 days: decay ~ 0.933 per day
- Apply exponential weighting to any rolling statistic
- Advantages: smoother, more responsive than simple rolling windows
- Multiple halflifes capture different timescales of information

### 3.4 Change and Acceleration Features
- **First difference**: delta_f = f(t) - f(t-n)
- **Percentage change**: (f(t) - f(t-n)) / |f(t-n)|
- **Second difference (acceleration)**: delta_f(t) - delta_f(t-n)
- **Z-score of change**: normalize change by its rolling standard deviation
- These capture the DYNAMICS of features, not just levels

---

## 4. Feature Selection

### 4.1 Filter Methods
- **Mutual information**: I(X; Y) = sum P(x,y) * log(P(x,y) / (P(x) * P(y)))
  - Captures nonlinear dependencies; no assumption of functional form
  - Compute between each candidate feature and target variable
  - Rank by MI score; select top-k or those above a threshold
- **Correlation**: Spearman rank correlation (robust to outliers)
  - corr(feature_rank, return_rank) over rolling windows
  - Called "Information Coefficient (IC)" in quant finance
  - IC > 0.02 is generally considered meaningful for daily return prediction

### 4.2 Embedded Methods (LASSO and Elastic Net)
- **LASSO (L1 regularization)**: min ||y - X*beta||^2 + alpha * ||beta||_1
  - Drives coefficients to exactly zero: automatic feature selection
  - alpha controls sparsity: higher alpha = fewer features retained
  - Use cross-validated alpha selection (but with proper time-series CV!)
- **Elastic Net**: combines L1 and L2 penalties
  - min ||y - X*beta||^2 + alpha * (rho * ||beta||_1 + (1-rho) * ||beta||_2^2)
  - Better than LASSO when features are correlated (groups of correlated features)

### 4.3 Wrapper Methods
- **Recursive Feature Elimination (RFE)**: train model, remove least important features, repeat
- **Sequential Feature Selection**: add/remove features one at a time based on CV score
- Computationally expensive but directly optimizes model performance
- Use walk-forward CV (NOT random CV) to avoid lookahead

### 4.4 Feature Importance from Tree Models
- **Random Forest importance**: mean decrease in impurity (MDI) or permutation importance
  - MDI is biased toward high-cardinality features; prefer permutation importance
- **Gradient boosting importance**: gain, split count, or SHAP values
- **SHAP (SHapley Additive exPlanations)**: theoretically grounded feature attribution
  - SHAP value for feature j = average marginal contribution across all subsets
  - Computationally expensive; use TreeSHAP for tree-based models

### 4.5 Stability of Feature Importance
- Feature importance should be stable across time periods and subsamples
- Method: compute importance on multiple non-overlapping time windows
- Retain features that rank consistently high (e.g., in top-20 in >70% of windows)
- Unstable features likely capture noise or regime-specific patterns

---

## 5. Normalization for Financial Data

### 5.1 Why Normalization Matters
- Financial features span vastly different scales (price: $10-$1000; volume: 1K-100M)
- Many ML algorithms are scale-sensitive (linear models, SVMs, neural networks)
- Non-stationary data (trending prices) violates model assumptions
- Outliers in financial data can dominate unnormalized features

### 5.2 Cross-Sectional Normalization
- At each time step, normalize features across the universe of assets
- Z-score: (x_i - mean(x)) / std(x) -- sensitive to outliers
- Rank normalization: replace values with their percentile rank [0, 1]
  - Most robust to outliers; recommended as default for cross-sectional models
- Winsorization: clip values at 1st and 99th percentile before z-scoring

### 5.3 Time-Series Normalization
- Rolling z-score: (x_t - rolling_mean) / rolling_std
  - Window: 60-252 days depending on feature frequency
  - Creates a stationary, zero-mean, unit-variance feature
- Returns instead of prices: use log returns to achieve stationarity
- Volatility scaling: divide feature by rolling volatility estimate
  - Accounts for time-varying volatility; stabilizes feature distribution

### 5.4 Normalization Pitfalls
- **Lookahead in normalization**: use ONLY past data for rolling stats
  - Common mistake: normalize using the entire dataset (future-contaminated)
- **Mean and std estimation**: use expanding or rolling windows, NEVER the full sample
- **Rank ties**: handle ties consistently (e.g., average rank for tied values)
- **Zero variance**: add epsilon to denominator: (x - mean) / (std + 1e-8)
- **Non-Gaussian distributions**: z-score assumes normality; rank transform is safer

---

## 6. Common Pitfalls in Feature Engineering

1. **Lookahead bias**: using future information in feature construction
   - Always verify: at time t, does this feature use ONLY data from time <= t?
2. **Survivorship bias**: features computed only on currently listed assets
   - Include delisted, bankrupt, and acquired companies in historical features
3. **Data snooping**: testing thousands of features and selecting ex-post winners
   - Use out-of-sample testing; adjust for multiple comparisons
4. **Multicollinearity**: highly correlated features destabilize linear models
   - Drop one of each highly correlated pair (|corr| > 0.9)
   - Or use PCA to orthogonalize features
5. **Feature leakage**: target information encoded in features (e.g., next-day return in same row)
6. **Ignoring transaction costs**: features that predict tiny returns may not survive costs
7. **Overfitting to feature count**: more features != better; use selection and regularization

---

## 7. Implementation Workflow

1. **Data pipeline**: clean raw data -> compute features -> store in feature store
2. **Feature registry**: document each feature (name, formula, lookback, data source, known issues)
3. **Validation**: check for NaNs, infinities, constant features, extreme outliers
4. **Normalization**: apply appropriate method (cross-sectional rank + time-series z-score)
5. **Selection**: compute IC, MI, and model-based importance; retain stable, significant features
6. **Monitoring**: track feature distributions and IC decay over time in production
7. **Refresh**: re-evaluate feature set quarterly; retire degraded features, test new candidates

---

## 8. Key References

- **Lopez de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Wiley. -- Definitive reference on ML feature engineering for finance.
- **Kakushadze, Z. (2016)**. "101 Formulaic Alphas." *Wilmott Magazine*, 2016(84), 72-81. -- Systematic catalog of quantitative features.
- **Tibshirani, R. (1996)**. "Regression Shrinkage and Selection via the LASSO." *Journal of the Royal Statistical Society B*, 58(1), 267-288. -- LASSO for feature selection.
- **Lundberg, S. & Lee, S.I. (2017)**. "A Unified Approach to Interpreting Model Predictions." *NeurIPS*, 4765-4774. -- SHAP values for feature importance.
- **Cover, T. & Thomas, J. (2006)**. *Elements of Information Theory*. Wiley. -- Information-theoretic foundations for mutual information.
- **Jegadeesh, N. & Titman, S. (1993)**. "Returns to Buying Winners and Selling Losers." *Journal of Finance*, 48(1), 65-91. -- Momentum features.
- **Fama, E. & French, K. (1993)**. "Common Risk Factors in the Returns on Stocks and Bonds." *Journal of Financial Economics*, 33(1), 3-56. -- Factor-based features.
- **Amihud, Y. (2002)**. "Illiquidity and Stock Returns." *Journal of Financial Markets*, 5(1), 31-56. -- Illiquidity feature.
