# Pairs Trading Strategies

## 1. Overview

Pairs trading is a market-neutral strategy that identifies two historically correlated
securities, monitors the divergence of their price relationship, and trades the
convergence back to equilibrium. It is the simplest and most intuitive form of
statistical arbitrage.

The strategy was pioneered in the mid-1980s by quantitative analysts at Morgan Stanley,
led by Nunzio Tartaglia. It remains one of the most widely deployed stat arb strategies
due to its conceptual simplicity and robustness.

**Core Logic:**

    1. Identify a pair of securities (A, B) with a stable historical relationship
    2. Construct a spread: s(t) = P_A(t) - beta * P_B(t)
    3. When the spread deviates significantly from its mean, trade it:
       - Spread too high: short A, long B (expect convergence)
       - Spread too low: long A, short B (expect convergence)
    4. Exit when the spread returns to its mean

---

## 2. Distance Method

The distance method (Gatev, Goetzmann, Rouwenhorst 2006) is the simplest pair
selection and trading approach.

**Pair Selection (Formation Period):**

    1. Normalize all price series to start at $1:
       P_norm(i, t) = P(i, t) / P(i, 0)

    2. For each pair (i, j), compute the sum of squared deviations (SSD):
       SSD(i, j) = SUM( (P_norm(i, t) - P_norm(j, t))^2, t=1..T )

    3. Rank all pairs by SSD in ascending order
    4. Select the top N pairs with lowest SSD (e.g., top 20 pairs)

**Trading Rules (Trading Period):**

    spread(t) = P_norm(i, t) - P_norm(j, t)
    mu = mean(spread) over formation period
    sigma = std(spread) over formation period

    Open: |spread(t) - mu| > 2 * sigma
    Close: spread(t) crosses mu (return to mean)

**Performance Characteristics:**
- Gatev et al. found average excess returns of ~11% annually (1962-2002)
- Returns declined significantly after 2002 as the strategy became widely known
- Transaction costs and short-selling constraints reduce returns substantially

**Limitations:**
- No statistical foundation for the price relationship (purely empirical)
- Normalized prices are non-stationary; SSD can be misleading
- No guarantee the spread is mean-reverting (no cointegration test)

---

## 3. Cointegration Approach: Engle-Granger Method

The cointegration framework provides a rigorous statistical basis for pairs trading.

**Concept:**
Two non-stationary I(1) time series are cointegrated if a linear combination of them
is stationary I(0). This stationary combination is the mean-reverting spread.

**Engle-Granger Two-Step Procedure:**

    Step 1: Cointegrating Regression (OLS)

        log(P_A(t)) = alpha + beta * log(P_B(t)) + epsilon(t)

        beta_hat = Cov(log P_A, log P_B) / Var(log P_B)

        Note: Use log prices to ensure the spread is in return space and to
        handle the non-negativity constraint of prices.

    Step 2: Test Residuals for Stationarity (ADF Test)

        epsilon_hat(t) = log(P_A(t)) - alpha_hat - beta_hat * log(P_B(t))

        Run ADF test on epsilon_hat(t):
            H0: epsilon_hat has a unit root (no cointegration)
            H1: epsilon_hat is stationary (cointegration exists)

        Use Engle-Granger critical values (more conservative than standard ADF)
        because epsilon_hat is estimated, not observed.

**The Spread:**

    s(t) = log(P_A(t)) - beta_hat * log(P_B(t)) - alpha_hat

    This is the error-correction term. When s(t) > 0, A is expensive relative to B.
    When s(t) < 0, A is cheap relative to B.

**Advantages Over Distance Method:**
- Statistically grounded: cointegration implies the spread is mean-reverting
- The hedge ratio beta is estimated, not assumed to be 1
- Provides a framework for testing whether the relationship is genuine

**Disadvantages:**
- Assumes a single, fixed cointegrating vector (may change over time)
- The Engle-Granger method can only find one cointegrating relationship
- Sensitive to which variable is the dependent variable (asymmetry)

---

## 4. Cointegration Approach: Johansen Method

The Johansen method addresses limitations of Engle-Granger by using a system-based
approach.

**Vector Error Correction Model (VECM):**

    delta_Y(t) = Pi * Y(t-1) + SUM(Gamma_i * delta_Y(t-i), i=1..p-1) + epsilon(t)

    where:
        Y(t) = [log P_A(t), log P_B(t)]' (2 x 1 vector)
        Pi = alpha * beta'  (2 x 2 matrix, decomposed into adjustment and
                             cointegrating vectors)
        rank(Pi) = number of cointegrating relationships

**Testing Procedure:**

    Trace Test:
        H0: rank(Pi) <= r   vs   H1: rank(Pi) > r

        trace_stat = -T * SUM(ln(1 - lambda_hat_i), i=r+1..n)

    Maximum Eigenvalue Test:
        H0: rank(Pi) = r    vs   H1: rank(Pi) = r + 1

        max_eigen_stat = -T * ln(1 - lambda_hat_(r+1))

    For pairs trading (n=2): test whether rank = 0 (no cointegration) vs rank = 1

**Advantages:**
- Symmetric: does not require choosing a dependent variable
- Can handle more than two variables (basket cointegration)
- Provides estimates of adjustment speeds (alpha) in addition to the cointegrating
  vector (beta)
- More powerful test than Engle-Granger in finite samples

**Interpretation of Adjustment Speeds:**

    If alpha_A < 0 and alpha_B > 0:
        Both A and B adjust toward equilibrium
        |alpha_A| / (|alpha_A| + |alpha_B|) = fraction of adjustment by A

    Faster adjustment speeds imply quicker mean reversion (shorter half-life).

---

## 5. Spread Construction

The spread is the tradeable signal in pairs trading. Proper construction is critical.

**Log Price Spread:**

    s(t) = log(P_A(t)) - beta * log(P_B(t))

    Interpretation: percentage mispricing between A and B

**Dollar-Neutral Spread:**

    For each $1 long A, short $beta of B:
        Number of shares of A: N_A = notional / P_A
        Number of shares of B: N_B = beta * notional / P_B
        Spread P&L = N_A * delta_P_A - N_B * delta_P_B

**Ratio Spread:**

    ratio(t) = P_A(t) / P_B(t)

    Trade when ratio deviates from its historical mean.
    Simpler but assumes beta = 1 and ignores the intercept.

**Normalized Spread (Z-Score):**

    z(t) = (s(t) - mean(s, window)) / std(s, window)

    Standard approach: window = 60-120 trading days (rolling)

    This is the primary signal for entry and exit decisions.

---

## 6. Entry and Exit Signals

**Standard Z-Score Signals:**

    | Condition             | Action                                    |
    |-----------------------|-------------------------------------------|
    | z(t) > +z_entry       | Short the spread (short A, long B)        |
    | z(t) < -z_entry       | Long the spread (long A, short B)         |
    | |z(t)| < z_exit       | Close position (spread near equilibrium)  |
    | |z(t)| > z_stop       | Stop-loss: close position (divergence)    |
    | t - t_entry > T_max   | Time stop: close position                 |

**Typical Parameters:**

    z_entry = 2.0 standard deviations
    z_exit  = 0.0 to 0.5 standard deviations
    z_stop  = 3.5 to 4.0 standard deviations
    T_max   = 3 * half_life (time-based stop)

**Graduated Entry (Scaling In):**

    | Z-Score Range  | Position Size  |
    |----------------|----------------|
    | |z| > 1.5      | 33% of target  |
    | |z| > 2.0      | 66% of target  |
    | |z| > 2.5      | 100% of target |

    Scaling in reduces the risk of entering too early and allows averaging into
    a better price if the spread continues to diverge.

**Signal Confirmation Filters:**
- Volume spike: require above-average volume on divergence day
- Momentum filter: avoid entering if the divergence has strong momentum (trending)
- Fundamental filter: check for corporate events (earnings, M&A) that explain divergence
- Sector confirmation: ensure the divergence is pair-specific, not sector-wide

---

## 7. Kalman Filter for Dynamic Hedging

The hedge ratio beta is not constant; it evolves over time. The Kalman filter provides
an optimal framework for tracking the time-varying hedge ratio.

**State-Space Formulation:**

    Observation equation:
        log(P_A(t)) = alpha(t) + beta(t) * log(P_B(t)) + v(t),  v(t) ~ N(0, R)

    State transition equations:
        alpha(t) = alpha(t-1) + w_alpha(t),  w_alpha ~ N(0, Q_alpha)
        beta(t)  = beta(t-1)  + w_beta(t),   w_beta  ~ N(0, Q_beta)

    State vector: x(t) = [alpha(t), beta(t)]'
    Observation matrix: H(t) = [1, log(P_B(t))]

**Kalman Filter Recursion:**

    Prediction Step:
        x_hat(t|t-1) = x_hat(t-1|t-1)          [state prediction, random walk model]
        P(t|t-1) = P(t-1|t-1) + Q               [covariance prediction]

    Update Step:
        y(t) = log(P_A(t)) - H(t) * x_hat(t|t-1)   [innovation / spread]
        S(t) = H(t) * P(t|t-1) * H(t)' + R          [innovation covariance]
        K(t) = P(t|t-1) * H(t)' / S(t)               [Kalman gain]
        x_hat(t|t) = x_hat(t|t-1) + K(t) * y(t)      [state update]
        P(t|t) = (I - K(t) * H(t)) * P(t|t-1)        [covariance update]

**Trading Signal:**
The innovation y(t) is the spread (prediction error). The standardized innovation:

    z_kalman(t) = y(t) / sqrt(S(t))

is approximately N(0,1) if the model is correct. Trade when |z_kalman| exceeds a
threshold (e.g., 2.0).

**Parameter Tuning:**

    Q (process noise covariance): controls how quickly beta adapts
        Q too large: beta is noisy, hedge ratio changes too fast
        Q too small: beta is sticky, slow to adapt to regime changes
        Typical: Q_beta ~ 1e-5 to 1e-4

    R (observation noise variance): controls how much weight is given to each observation
        R too large: filter is slow to update, spread is wide
        R too small: filter overreacts to noise
        Typical: R ~ variance of the OLS regression residuals

**Advantages Over Rolling OLS:**
- Optimal weighting of old and new information
- Continuous adaptation without window-length selection
- Provides uncertainty estimates for the hedge ratio (from P(t|t))
- The spread (innovation) has well-defined statistical properties

---

## 8. Sector-Based Pair Selection

Restricting pair selection to within-sector pairs improves the economic rationale and
stability of the trading relationship.

**Rationale:**
- Companies in the same sector face similar economic drivers
- Fundamental shocks are more likely to be temporary (mean-reverting) within sectors
- Regulatory and industry-specific risks are shared
- Reduces the risk of structural breaks from divergent business evolution

**Selection Pipeline:**

    1. Define sector/industry groups (GICS, ICB classification)
    2. Within each sector, identify candidate pairs:
       a. Correlation filter: rolling 252-day correlation > 0.7
       b. Cointegration test: Engle-Granger ADF p-value < 0.05
       c. Half-life filter: half-life between 5 and 60 trading days
       d. Liquidity filter: both names have ADV > $5M
    3. Rank qualifying pairs by:
       - Cointegration test statistic (stronger is better)
       - Half-life (shorter is better, within limits)
       - Historical Sharpe ratio of the spread trading strategy
    4. Select top M pairs per sector (e.g., 3-5 per sector)

**Sector-Specific Considerations:**

    | Sector        | Typical Drivers              | Pair Examples            |
    |---------------|------------------------------|--------------------------|
    | Energy        | Oil price, refining margins   | Upstream producers       |
    | Financials    | Interest rates, credit cycle  | Large banks              |
    | Technology    | Growth/value rotation         | Cloud/SaaS competitors   |
    | Utilities     | Regulation, rates             | Regional utilities       |
    | Healthcare    | Drug approvals, demographics  | Pharma within sub-sector |

**Cross-Sector Pairs:**
Occasionally valid (e.g., gold miners vs gold ETF, airlines vs oil), but require
stronger economic justification and more robust cointegration evidence.

---

## 9. Implementation for Algorithmic Trading

### Full Pipeline

    1. Universe Construction:
       - Liquid equities within defined sectors
       - Minimum ADV: $5M; minimum market cap: $500M
       - Exclude recent IPOs (< 1 year), stocks with corporate actions pending

    2. Pair Identification (Monthly):
       - Run cointegration tests on all within-sector pairs
       - Filter by half-life, correlation, spread variance
       - Score and rank pairs; select top candidates

    3. Daily Signal Generation:
       - Update Kalman filter estimates for beta(t), alpha(t)
       - Compute spread: s(t) = log(P_A(t)) - beta(t) * log(P_B(t))
       - Compute z-score: z(t) = (s(t) - mean_kalman) / sqrt(S(t))
       - Generate entry/exit/stop signals

    4. Portfolio Construction:
       - Allocate equal risk to each active pair
       - Position size: notional / n_pairs, adjusted for pair volatility
       - Ensure aggregate dollar neutrality and beta neutrality

    5. Execution:
       - Execute both legs simultaneously (or near-simultaneously) to avoid leg risk
       - Use basket/pair execution algorithms offered by prime brokers
       - Monitor fill ratios: both legs must fill to avoid directional exposure

    6. Monitoring:
       - Track spread z-score in real time
       - Monitor cointegration stability (rolling ADF p-value)
       - Alert on corporate events affecting pair constituents

### Position Sizing

    For each pair p:
        vol_spread(p) = std(s_p) * sqrt(252)
        notional(p) = (target_risk / n_pairs) / vol_spread(p)
        shares_A(p) = notional(p) / P_A
        shares_B(p) = beta(p) * notional(p) / P_B

### Risk Management

    Per-Pair Limits:
        Max loss per pair: 1-2% of portfolio NAV
        Max holding period: 3 * half_life
        Z-score stop-loss: |z| > 4.0

    Portfolio-Level Limits:
        Max number of active pairs: 20-40
        Max gross exposure: 200-400% of NAV
        Max net exposure (dollar): +/- 5% of NAV
        Max sector concentration: 30% of gross in any single sector
        Max daily turnover: 20% of gross

    Correlation Risk:
        Monitor pairwise correlation of pair spreads
        If pair spreads become correlated (> 0.5), the portfolio is less diversified
        than assumed; reduce overall position sizes

---

## 10. Common Pitfalls

1. **Leg risk during execution**: If only one leg of the pair fills, the portfolio has
   unintended directional exposure. Always use paired execution and monitor fill ratios.
2. **Ignoring structural breaks**: Mergers, spinoffs, regulatory changes, and business
   model shifts can permanently alter the pair relationship. Monitor corporate event
   calendars and set cointegration stability alerts.
3. **Overfitting the hedge ratio**: Using the full in-sample period to estimate beta
   maximizes in-sample stationarity but does not guarantee out-of-sample performance.
   Use walk-forward estimation.
4. **Neglecting borrowing costs**: Short positions incur borrowing fees that can
   exceed 5-10% annualized for hard-to-borrow names, eroding strategy returns.
5. **Confusing correlation with cointegration**: High correlation does not imply
   cointegration. Two trending stocks can be perfectly correlated but not cointegrated.
   Always test for cointegration explicitly.
6. **Symmetric entry/exit assumptions**: The long and short sides of a pair may have
   different dynamics (e.g., short squeezes affect one side more). Consider asymmetric
   thresholds.
7. **Ignoring the intercept drift**: The alpha term in the cointegrating regression
   can drift over time. A Kalman filter handles this; static OLS does not.

---

## 11. Seminal References

- Gatev, E., Goetzmann, W., and Rouwenhorst, K.G. (2006). "Pairs Trading: Performance
  of a Relative-Value Arbitrage Rule." Review of Financial Studies, 19(3), 797-827.
- Engle, R.F. and Granger, C.W.J. (1987). "Co-Integration and Error Correction:
  Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.
- Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors in
  Gaussian Vector Autoregressive Models." Econometrica, 59(6), 1551-1580.
- Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis." Wiley.
- Elliott, R.J., van der Hoek, J., and Malcolm, W.P. (2005). "Pairs Trading."
  Quantitative Finance, 5(3), 271-276.
- Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems."
  Journal of Basic Engineering, 82(1), 35-45.
- Hamilton, J.D. (1994). "Time Series Analysis." Princeton University Press.
- Pole, A. (2007). "Statistical Arbitrage: Algorithmic Trading Insights and Techniques."
  Wiley.
