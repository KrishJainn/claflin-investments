# Mean Reversion Trading Strategies

## 1. Overview

Mean reversion is the theory that asset prices and returns eventually move back toward
their long-run mean or equilibrium level. Unlike momentum, which bets on trend
continuation, mean reversion profits from the tendency of prices to oscillate around a
central value.

Mean reversion operates across multiple time horizons: intraday microstructure effects,
short-term (days to weeks) overreaction reversals, and long-term (years) valuation-driven
reversion. This document focuses on short-to-medium-term statistical mean reversion
strategies applicable to algorithmic trading.

---

## 2. Theoretical Basis

**Efficient Market Counterargument:**
In an efficient market, prices follow a random walk and no mean reversion exists. Mean
reversion strategies exploit deviations from efficiency caused by:
- Liquidity shocks (temporary price impact from large orders)
- Overreaction to news (behavioral finance: anchoring, representativeness)
- Market microstructure noise (bid-ask bounce)
- Inventory management by market makers

**Random Walk vs Mean-Reverting Process:**

    Random Walk:    dX(t) = mu * dt + sigma * dW(t)
    Mean-Reverting: dX(t) = theta * (mu - X(t)) * dt + sigma * dW(t)

The key difference is the drift term theta * (mu - X(t)) which pulls the process back
toward mu. When X(t) > mu, the drift is negative; when X(t) < mu, the drift is positive.

---

## 3. Ornstein-Uhlenbeck (OU) Process

The canonical continuous-time mean-reverting process used in quantitative finance.

**Continuous Form:**

    dX(t) = theta * (mu - X(t)) * dt + sigma * dW(t)

    Parameters:
        theta  = speed of mean reversion (higher = faster reversion)
        mu     = long-run equilibrium level
        sigma  = volatility of the process
        W(t)   = standard Wiener process (Brownian motion)

**Discrete-Time (AR(1)) Approximation:**

    X(t) = a + b * X(t-1) + epsilon(t)

    Mapping:
        theta = -ln(b) / dt
        mu    = a / (1 - b)
        sigma_ou = sigma_epsilon * sqrt(-2 * ln(b) / (dt * (1 - b^2)))

    Mean-reverting condition: |b| < 1 (equivalently, theta > 0)

**Estimation via OLS:**

    Regress X(t) on X(t-1):
        X(t) = a_hat + b_hat * X(t-1) + residuals

    If b_hat is statistically less than 1 (via t-test or ADF), the series mean-reverts.

---

## 4. Half-Life of Mean Reversion

The half-life measures how long it takes for a deviation from the mean to decay by 50%.

**Formula:**

    half_life = -ln(2) / ln(b)

    where b is the AR(1) coefficient from the regression X(t) = a + b * X(t-1) + e(t)

**Interpretation:**

    | Half-Life   | Implication                                    |
    |-------------|------------------------------------------------|
    | < 1 day     | Microstructure noise; too fast for most algos  |
    | 1-5 days    | High-frequency mean reversion; requires low    |
    |             | latency execution                              |
    | 5-30 days   | Sweet spot for systematic mean reversion       |
    | 30-90 days  | Viable but requires patience and capital       |
    | > 90 days   | Likely too slow; opportunity cost is high       |

**Trading Application:**
The half-life informs the holding period and stop-loss timing. A spread with a half-life
of 10 days should be expected to converge within approximately 20-30 days (2-3 half-lives).

---

## 5. Hurst Exponent

The Hurst exponent H quantifies the long-range dependence in a time series and
distinguishes between trending and mean-reverting behavior.

**Interpretation:**

    H < 0.5  -->  Mean-reverting (anti-persistent)
    H = 0.5  -->  Random walk (Geometric Brownian Motion)
    H > 0.5  -->  Trending (persistent)

**Estimation Methods:**

*Rescaled Range (R/S) Analysis:*

    For each sub-period of length n:
        1. Compute mean-adjusted cumulative deviations Y(t) = SUM(X(i) - X_mean, i=1..t)
        2. R(n) = max(Y) - min(Y)  (range)
        3. S(n) = std(X)           (standard deviation)
        4. E[R(n)/S(n)] ~ C * n^H

    Regress ln(R/S) on ln(n) to estimate H.

*Variance Ratio Method:*

    VR(q) = Var(X(t) - X(t-q)) / (q * Var(X(t) - X(t-1)))

    If VR(q) < 1 for q > 1, the series is mean-reverting.
    If VR(q) = 1, the series is a random walk.
    If VR(q) > 1, the series is trending.

**Practical Note:**
The Hurst exponent can be unstable over short samples. Use a rolling window (e.g., 252
trading days) and monitor regime changes.

---

## 6. Augmented Dickey-Fuller (ADF) Test

The ADF test is the standard statistical test for stationarity (mean reversion).

**Hypothesis:**

    H0: The series has a unit root (random walk, non-stationary)
    H1: The series is stationary (mean-reverting)

**Test Regression:**

    delta_X(t) = alpha + beta * X(t-1) + SUM(gamma_i * delta_X(t-i), i=1..p) + epsilon(t)

    Test statistic = beta_hat / SE(beta_hat)

    Reject H0 if test statistic < critical value (e.g., -2.86 at 5% for constant only)

**Implementation Considerations:**
- Choose lag order p using AIC or BIC
- Include constant (drift) and optionally a trend term
- Critical values are non-standard (use MacKinnon tables, not normal/t-distribution)
- Low power in small samples: a non-rejection does not confirm random walk

**For Pairs/Spreads:**
Apply ADF to the spread series s(t) = Y(t) - beta * X(t). If s(t) is stationary, the
pair is cointegrated and suitable for mean reversion trading.

---

## 7. Bollinger Band Strategy

A widely used mean-reversion indicator based on moving averages and standard deviations.

**Construction:**

    Middle Band:  MA(t)    = (1/N) * SUM(P(t-i), i=0..N-1)
    Upper Band:   UB(t)    = MA(t) + k * sigma(t)
    Lower Band:   LB(t)    = MA(t) - k * sigma(t)

    where:
        N     = lookback window (typically 20 days)
        k     = number of standard deviations (typically 2.0)
        sigma = rolling standard deviation of price over N periods

**Trading Rules:**

    Entry Long:   P(t) < LB(t)   [price touches or crosses below lower band]
    Entry Short:  P(t) > UB(t)   [price touches or crosses above upper band]
    Exit:         P(t) crosses MA(t) [return to mean]
    Stop-loss:    P(t) < LB(t) - k2 * sigma(t) [extended deviation, k2 > k]

**Enhancements:**
- Bandwidth filter: only trade when bandwidth = (UB - LB) / MA is above a threshold
  (avoids low-volatility, trendless regimes)
- Volume confirmation: require above-average volume on the signal bar
- RSI confirmation: combine with RSI < 30 for long entries, RSI > 70 for short entries

---

## 8. Z-Score Based Signals

The z-score normalizes the deviation from the mean, providing a standardized signal.

**Computation:**

    z(t) = (X(t) - MA(X, N)) / std(X, N)

    where X(t) is the price or spread value

**Signal Thresholds:**

    | Z-Score    | Action          | Rationale                          |
    |------------|-----------------|------------------------------------|
    | z > +2.0   | Enter Short     | Price is 2 sigma above mean        |
    | z > +1.0   | Partial Short   | Moderate overvaluation              |
    | -0.5 < z   | Exit / Neutral  | Price near equilibrium              |
    |   < +0.5   |                 |                                    |
    | z < -1.0   | Partial Long    | Moderate undervaluation             |
    | z < -2.0   | Enter Long      | Price is 2 sigma below mean         |

**Dynamic Thresholds:**
Instead of fixed thresholds, use rolling percentiles of the z-score distribution:

    entry_threshold(t) = percentile(|z|, 95th, rolling_window=252)

This adapts to changing volatility regimes automatically.

**Exponentially Weighted Z-Score:**

    EWMA_mean(t) = lambda * X(t) + (1 - lambda) * EWMA_mean(t-1)
    EWMA_var(t)  = lambda * (X(t) - EWMA_mean(t))^2 + (1 - lambda) * EWMA_var(t-1)
    z_ew(t)      = (X(t) - EWMA_mean(t)) / sqrt(EWMA_var(t))

    Typical lambda = 2 / (N + 1) for span N

---

## 9. Cointegration for Mean Reversion

When individual price series are non-stationary (I(1)), their linear combination may
be stationary (I(0)). This is cointegration, and the stationary residual is the
mean-reverting spread.

**Engle-Granger Two-Step Method:**

    Step 1: Estimate the cointegrating regression via OLS
        Y(t) = alpha + beta * X(t) + epsilon(t)

    Step 2: Test residuals epsilon_hat(t) for stationarity using ADF
        If ADF rejects unit root, Y and X are cointegrated with vector [1, -beta]

**The Spread:**

    s(t) = Y(t) - beta_hat * X(t)

    Trade the spread using z-score signals applied to s(t)

**Rolling Estimation:**
The hedge ratio beta is not constant. Use a rolling window (e.g., 60-120 days) to
re-estimate beta and rebalance the hedge:

    beta_hat(t) = OLS(Y(t-W..t), X(t-W..t))

Or preferably use a Kalman filter for continuous adaptation (see pairs_trading.md).

---

## 10. Implementation for Algorithmic Trading

### Signal Pipeline

    1. Universe filtering: select liquid instruments with sufficient history
    2. Stationarity screening: ADF test, Hurst exponent, variance ratio
    3. Half-life estimation: ensure half-life is within target range (5-60 days)
    4. Signal generation: compute z-score or Bollinger band position
    5. Position sizing: scale inversely with half-life and volatility
       w(t) = -z(t) * (target_vol / realized_vol) * (base_halflife / halflife)
    6. Risk limits: max position per name, max portfolio gross exposure

### Execution

    - Use limit orders at or near the signal price (mean reversion is not urgent)
    - Passive execution suits mean reversion better than momentum (no urgency)
    - Avoid crossing the spread unnecessarily; patient limit order placement
      improves realized P&L by 5-15 bps per trade

### Risk Management

    - Time-based stop: exit if trade has not converged within 3x half-life
    - Loss-based stop: exit if unrealized loss exceeds 2x expected profit
    - Regime detection: monitor Hurst exponent and ADF p-value in real time;
      if the series transitions to H > 0.5 or ADF fails to reject, exit positions
    - Drawdown control: reduce position sizes after portfolio drawdown exceeds threshold

---

## 11. Common Pitfalls

1. **Assuming stationarity is permanent**: A stationary spread can become non-stationary
   due to structural breaks (mergers, regulation changes, macro regime shifts)
2. **Overfitting lookback and threshold parameters**: Optimize on out-of-sample data;
   use walk-forward analysis with expanding or rolling windows
3. **Ignoring the spread between mean and median**: For skewed distributions, the mean
   may not be the appropriate central tendency measure
4. **Neglecting transaction costs for high-frequency reversion**: Bid-ask bounce creates
   apparent mean reversion that vanishes after costs
5. **Confusing mean reversion with low volatility**: A low-volatility sideways market is
   not the same as statistically significant mean reversion
6. **Using price levels instead of log prices or returns**: Raw prices can create
   spurious regression results; use log prices for spread estimation

---

## 12. Seminal References

- Poterba, J. and Summers, L. (1988). "Mean Reversion in Stock Prices: Evidence and
  Implications." Journal of Financial Economics, 22(1), 27-59.
- Lo, A. and MacKinlay, C. (1988). "Stock Market Prices Do Not Follow Random Walks:
  Evidence from a Simple Specification Test." Review of Financial Studies, 1(1), 41-66.
- Uhlenbeck, G.E. and Ornstein, L.S. (1930). "On the Theory of the Brownian Motion."
  Physical Review, 36(5), 823-841.
- Engle, R.F. and Granger, C.W.J. (1987). "Co-Integration and Error Correction:
  Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.
- Hurst, H.E. (1951). "Long-Term Storage Capacity of Reservoirs." Transactions of the
  American Society of Civil Engineers, 116, 770-799.
- Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators for
  Autoregressive Time Series with a Unit Root." Journal of the American Statistical
  Association, 74(366), 427-431.
- Bollinger, J. (2001). "Bollinger on Bollinger Bands." McGraw-Hill.
- Avellaneda, M. and Lee, J.H. (2010). "Statistical Arbitrage in the US Equities
  Market." Quantitative Finance, 10(7), 761-782.
