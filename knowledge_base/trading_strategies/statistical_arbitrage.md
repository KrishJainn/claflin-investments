# Statistical Arbitrage Trading Strategies

## 1. Overview

Statistical arbitrage (stat arb) is a class of market-neutral trading strategies that
exploit temporary mispricings among related securities using quantitative models. Unlike
pure arbitrage, stat arb does not guarantee a riskless profit; it relies on statistical
relationships that hold on average over many trades.

Stat arb strategies emerged in the 1980s at Morgan Stanley under Nunzio Tartaglia's
group, which pioneered pairs trading and evolved into systematic, multi-factor,
PCA-based approaches. Modern stat arb encompasses factor models, eigenportfolios,
machine learning-based signals, and high-frequency market making.

---

## 2. Concept and History

**Core Principle:**
If a group of securities is driven by common factors, then the idiosyncratic (residual)
component of each security's return should be mean-reverting. Stat arb strategies
identify and trade these residual mispricings.

**Historical Evolution:**

    1980s: Pairs trading at Morgan Stanley (Tartaglia group)
    1990s: Multi-factor residual reversion models (D.E. Shaw, Renaissance)
    2000s: PCA-based eigenportfolios, sector-neutral construction
    2007:  Quant meltdown (August 2007) - crowding risk materialized
    2010s: Machine learning integration, alternative data signals
    2020s: Capacity constraints; alpha decay accelerating

**The August 2007 Quant Crisis:**
Many stat arb funds experienced simultaneous large drawdowns in August 2007. The likely
cause was forced liquidation by a large multi-strategy fund, which triggered a cascade
of losses among funds with similar positions. This event highlighted the systemic risk
of crowded stat arb strategies.

---

## 3. PCA-Based Approaches

Principal Component Analysis (PCA) decomposes the covariance structure of returns into
orthogonal factors, providing a data-driven alternative to pre-specified factor models.

**Methodology:**

    1. Construct the return matrix R (T x N) for N securities over T periods
    2. Compute the sample covariance matrix: Sigma = (1/T) * R' * R
    3. Eigendecompose: Sigma = V * Lambda * V'
       where Lambda = diag(lambda_1, ..., lambda_N) with lambda_1 >= lambda_2 >= ...
       and V = [v_1, v_2, ..., v_N] are the eigenvectors
    4. Select top K eigenvectors capturing sufficient variance (e.g., 50-70%)
    5. Factor returns: F(t) = R(t) * V_K    (T x K matrix)
    6. Factor loadings (betas): B = V_K      (N x K matrix)

**Residual Computation:**

    r_residual(i, t) = r(i, t) - SUM(B(i, k) * F(k, t), k=1..K)

    The residual r_residual is the idiosyncratic return unexplained by the top K factors.
    Stat arb trades the cumulative residual when it deviates significantly from zero.

**Cumulative Residual (S-Score):**

    s(i, t) = SUM(r_residual(i, j), j=t-W..t) / std(residual_i)

    where W is the estimation window (typically 60 trading days)

    Trade: long if s(i, t) < -s_entry, short if s(i, t) > +s_entry
    Exit:  |s(i, t)| < s_exit

**Typical Parameters:**
    s_entry = 1.5 to 2.0 standard deviations
    s_exit  = 0.5 to 0.75 standard deviations
    W       = 60 trading days
    K       = 5 to 15 principal components

---

## 4. Factor Models

Pre-specified factor models provide an alternative to PCA with more interpretable factors.

**Return Decomposition:**

    r(i, t) = alpha(i) + SUM(beta(i, k) * f(k, t), k=1..K) + epsilon(i, t)

    where:
        f(k, t) = return of factor k at time t
        beta(i, k) = loading of security i on factor k
        epsilon(i, t) = idiosyncratic return (the trading signal)

**Standard Factor Sets:**

    | Factor        | Long Leg              | Short Leg              |
    |---------------|-----------------------|------------------------|
    | Market        | Full universe         | Risk-free rate         |
    | Size (SMB)    | Small-cap stocks      | Large-cap stocks       |
    | Value (HML)   | High B/M stocks       | Low B/M stocks         |
    | Momentum(UMD) | Past 12-1m winners    | Past 12-1m losers      |
    | Quality (QMJ) | High profitability    | Low profitability      |
    | Low Vol (BAB) | Low-beta stocks       | High-beta stocks       |

**Residual Alpha:**
After regressing out factor exposures, the residual represents the security-specific
mispricing. A positive cumulative residual suggests the stock has outperformed its
factor-implied return, and a mean-reversion bet would short it (expecting reversion).

**Cross-Sectional Regression (Fama-MacBeth):**

    At each time t:
        r(i, t) = gamma_0(t) + SUM(gamma_k(t) * beta(i, k)) + e(i, t)

    Time-series average of gamma_k estimates the factor risk premium.
    The cross-sectional residuals e(i, t) are the stat arb signals.

---

## 5. Eigenportfolios

Eigenportfolios are portfolios whose weights correspond to the eigenvectors of the
return covariance matrix.

**Construction:**

    Portfolio k weights: w_k = v_k / ||v_k||_1

    where v_k is the k-th eigenvector and ||.||_1 is the L1 norm for normalization

**Properties:**
- The first eigenportfolio approximates the market portfolio (all positive weights)
- Higher-order eigenportfolios represent long-short sector/style bets
- Eigenportfolios are orthogonal: their returns are uncorrelated by construction

**Trading Strategy:**

    1. Compute eigenportfolios from trailing covariance matrix (e.g., 252 days)
    2. Project each stock's return onto the eigenportfolio space
    3. The residual (unexplained component) is the stat arb signal
    4. Trade the residual using z-score or s-score methodology

**Stability Considerations:**
- Eigenvectors are notoriously unstable: small changes in the covariance matrix can
  cause large rotations in eigenvectors (eigenvalue crossing problem)
- Use shrinkage estimators (Ledoit-Wolf) for more stable covariance estimation
- Apply random matrix theory (RMT) to distinguish signal eigenvalues from noise:
  eigenvalues below the Marchenko-Pastur upper bound are likely noise

    Marchenko-Pastur upper bound: lambda_+ = sigma^2 * (1 + sqrt(N/T))^2

---

## 6. Market-Neutral Portfolio Construction

Stat arb portfolios are constructed to be market-neutral and ideally factor-neutral.

**Dollar Neutrality:**

    SUM(w_i, for all i) = 0  or equivalently  SUM(w_i, longs) = -SUM(w_i, shorts)

**Beta Neutrality:**

    SUM(w_i * beta_i) = 0

    Ensures the portfolio has zero sensitivity to the broad market.

**Factor Neutrality (General):**

    For each factor k:  SUM(w_i * beta(i, k)) = 0

**Optimization Framework:**

    maximize:   SUM(w_i * alpha_hat(i))      [expected alpha]
    subject to:
        SUM(w_i * beta(i, k)) = 0            for k = 1..K  [factor neutrality]
        SUM(|w_i|) <= L                       [gross leverage constraint]
        |w_i| <= w_max                        [position size limit]
        w' * Sigma * w <= sigma_target^2      [risk constraint]
        turnover(w, w_prev) <= tau_max        [turnover constraint]

This is a convex optimization problem (quadratic with linear constraints) solvable
with standard solvers (CVXPY, Gurobi, MOSEK).

**Sector Neutrality:**

    For each sector s: SUM(w_i, i in sector s) = 0

This ensures the portfolio does not have directional sector bets, reducing exposure
to sector-specific shocks.

---

## 7. Risk Decomposition

Understanding the sources of portfolio risk is critical for stat arb.

**Total Variance Decomposition:**

    Var(r_portfolio) = Var(systematic) + Var(idiosyncratic)

    Var(systematic) = w' * B * Sigma_F * B' * w
    Var(idiosyncratic) = w' * D * w

    where:
        B = factor loading matrix (N x K)
        Sigma_F = factor covariance matrix (K x K)
        D = diagonal matrix of idiosyncratic variances

**For a Well-Constructed Stat Arb Portfolio:**
- Systematic risk should be near zero (if properly neutralized)
- Idiosyncratic risk dominates; this is desired as it diversifies across many names
- Idiosyncratic risk decreases as sqrt(N) with number of positions
  (assuming independence)

**Risk Metrics:**

    Ex-ante volatility:    sigma_p = sqrt(w' * Sigma * w) * sqrt(252)
    Tracking error:        TE = sigma_p (for dollar-neutral portfolios)
    Sharpe ratio target:   SR > 1.5 for institutional stat arb
    Maximum drawdown:      typically constrained to < 10-15%
    Conditional VaR (CVaR): P(loss > VaR at 99%) with expected shortfall

**Stress Testing:**
- Replay August 2007 quant crisis returns through current portfolio
- Simulate factor shocks: +/- 3 sigma move in each factor independently
- Liquidity stress: assume 50% reduction in daily volume for position exit

---

## 8. Capacity Constraints

Stat arb strategies face significant capacity limitations.

**Sources of Capacity Constraints:**

    1. Market impact: trading moves prices adversely
       Impact model: impact(i) = eta * sigma(i) * sqrt(|trade_i| / ADV(i))
       where eta ~ 0.1-0.5, ADV = average daily volume

    2. Alpha decay: signals weaken as more capital chases the same opportunities
       Half-life of alpha: typically 1-5 days for residual reversion signals

    3. Crowding: correlated positions among stat arb managers amplify drawdowns
       Crowding measure: overlap = (w_A' * w_B) / (||w_A|| * ||w_B||)

    4. Borrowing costs: short positions incur lending fees, especially for
       hard-to-borrow stocks (can exceed 10% annualized for crowded shorts)

**Capacity Estimation:**

    Capacity ~ (acceptable_cost_fraction * alpha) * SUM(ADV_i * participation_limit)

    Example:
        Alpha = 5% annualized
        Acceptable cost = 20% of alpha (1% cost budget)
        Average ADV per name = $10M
        Participation limit = 5% of ADV
        100 positions: Capacity ~ 100 * $10M * 0.05 = $50M per side

**Scaling Strategies:**
- Expand universe: more names dilute per-name impact
- Reduce turnover: composite signals with longer horizons
- Multi-strategy: combine stat arb with other alpha sources
- Trade less liquid names where competition is lower but impact is higher (tradeoff)

---

## 9. Implementation for Algorithmic Trading

### Daily Pipeline

    1. Data: download adjusted prices, volumes, corporate actions, factor returns
    2. Estimation: rolling PCA or factor regression (252-day window)
    3. Residuals: compute cumulative residuals, s-scores, z-scores
    4. Signal generation: rank s-scores, identify entry/exit candidates
    5. Portfolio optimization: solve for target weights with constraints
    6. Trade generation: diff target weights vs current weights
    7. Execution: break orders into child orders, execute via TWAP/VWAP
    8. Risk monitoring: real-time factor exposure, drawdown, concentration checks
    9. Reconciliation: end-of-day position and P&L reconciliation

### Backtesting Considerations

    - Use point-in-time data to avoid lookahead bias
    - Account for survivorship bias: include delisted securities
    - Model transaction costs: spread + impact + borrowing costs
    - Test for robustness: parameter sensitivity, different time periods
    - Walk-forward optimization: never optimize in-sample and test in-sample
    - Out-of-sample Sharpe ratio should be at least 50% of in-sample

### Technology Requirements

    - Low-latency data feed: real-time prices for intraday risk monitoring
    - Portfolio optimizer: CVXPY, Gurobi, or MOSEK for quadratic programs
    - Execution management system: order routing, smart order routing
    - Risk system: real-time factor decomposition, exposure monitoring
    - Database: time-series database for historical data (kdb+, InfluxDB, Arctic)

---

## 10. Common Pitfalls

1. **Overfitting the number of principal components**: Using too many PCs captures
   noise; too few misses important factors. Use cross-validation or RMT bounds.
2. **Ignoring transaction costs in alpha estimation**: A 2% residual alpha with 500%
   annual turnover requires < 0.4 bps per trade to be profitable.
3. **Stale covariance matrices**: Using 252-day trailing windows during regime changes
   produces misleading factor structures. Consider exponential weighting.
4. **Crowding risk**: Monitor overlap with known stat arb factor exposures. August 2007
   demonstrated that crowded stat arb can have fat-tailed drawdowns.
5. **Short squeeze risk**: Concentrated short positions in hard-to-borrow names can
   experience violent squeezes (e.g., GameStop January 2021 dynamics).
6. **Survivorship bias in backtests**: Not including delisted/bankrupt stocks inflates
   long-side returns and understates short-side borrowing costs.
7. **Assuming factor neutrality implies market neutrality**: Factor-neutral portfolios
   can still have residual market exposure through higher-order moments.

---

## 11. Seminal References

- Avellaneda, M. and Lee, J.H. (2010). "Statistical Arbitrage in the US Equities
  Market." Quantitative Finance, 10(7), 761-782.
- Gatev, E., Goetzmann, W., and Rouwenhorst, K.G. (2006). "Pairs Trading: Performance
  of a Relative-Value Arbitrage Rule." Review of Financial Studies, 19(3), 797-827.
- Khandani, A. and Lo, A. (2007). "What Happened to the Quants in August 2007?"
  Journal of Investment Management, 5(4), 5-54.
- Ledoit, O. and Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional
  Covariance Matrices." Journal of Multivariate Analysis, 88(2), 365-411.
- Fama, E.F. and MacBeth, J.D. (1973). "Risk, Return, and Equilibrium: Empirical
  Tests." Journal of Political Economy, 81(3), 607-636.
- Ross, S.A. (1976). "The Arbitrage Theory of Capital Asset Pricing." Journal of
  Economic Theory, 13(3), 341-360.
- Laloux, L., Cizeau, P., Bouchaud, J.P., and Potters, M. (1999). "Noise Dressing of
  Financial Correlation Matrices." Physical Review Letters, 83(7), 1467-1470.
- Lo, A. (2010). "Hedge Funds: An Analytic Perspective." Princeton University Press.
