# Modern Portfolio Theory and Advanced Portfolio Construction

## Overview

Modern Portfolio Theory (MPT), developed by Harry Markowitz in 1952, provides the mathematical
framework for constructing portfolios that maximize expected return for a given level of risk.
While the original formulation has well-known limitations, it remains the foundation upon which
all modern portfolio construction techniques are built.

---

## Markowitz Mean-Variance Optimization

### Core Framework

- **Input requirements**: Expected returns vector (mu), covariance matrix (Sigma), and
  constraints (budget, long-only, sector limits).
- **Objective**: Minimize portfolio variance for a target expected return, or equivalently
  maximize expected return for a target variance.
- **Mathematical formulation**:
  ```
  Minimize:  w' * Sigma * w
  Subject to: w' * mu >= target_return
              w' * 1 = 1  (fully invested)
              w >= 0       (long-only, optional)
  ```
- **Solution**: Quadratic programming yields optimal weight vector w*.

### Estimation Challenges

The primary practical problem with mean-variance optimization is **estimation error**:

1. **Expected returns**: The most difficult input to estimate. Small changes in expected
   returns produce large swings in optimal weights. Historical mean returns are extremely
   noisy estimators of future expected returns.
2. **Covariance matrix**: More stable than returns but still noisy, especially for large
   universes. Sample covariance matrix is singular when T < N (more assets than observations).
3. **Estimation error amplification**: MVO effectively maximizes estimation error -- it
   overweights assets with overestimated returns and underweights those with underestimated
   returns, producing extreme, concentrated, unstable portfolios.

### Practical Solutions to Estimation Error

| Technique                  | Description                                         |
|----------------------------|-----------------------------------------------------|
| Shrinkage estimators       | Ledoit-Wolf shrinks sample covariance toward structured target |
| Resampled efficiency       | Michaud: Monte Carlo resampling of inputs, average optimal portfolios |
| Robust optimization        | Worst-case optimization over uncertainty sets for inputs |
| Bayesian estimation        | Prior beliefs regularize return estimates            |
| Factor model covariance    | Impose factor structure on covariance matrix for stability |
| Constraints as information | Position and sector limits serve as implicit shrinkage |

---

## The Efficient Frontier

### Definition

- The efficient frontier is the set of portfolios offering the highest expected return for
  each level of risk (standard deviation).
- Portfolios below the frontier are **dominated** -- a higher-return portfolio exists at the
  same risk level.
- The frontier is a hyperbola in mean-standard deviation space (parabola in mean-variance).

### Key Properties

- **Minimum variance portfolio**: The leftmost point on the frontier; the portfolio with
  the lowest possible volatility achievable from the asset universe.
- **Two-fund separation**: Any efficient portfolio can be constructed as a combination of
  two efficient portfolios (Merton, 1972).
- **Diversification benefit**: The frontier lies to the left of all individual assets,
  demonstrating that diversification reduces risk without sacrificing return.

### Practical Considerations

- The efficient frontier is estimated with error and is unstable over time.
- In practice, the realized efficient frontier looks very different from the ex-ante estimate.
- Constraints (no short selling, position limits) alter the frontier shape and can actually
  improve out-of-sample performance by reducing estimation error sensitivity.

---

## Capital Market Line (CML)

- When a risk-free asset is available, the efficient frontier becomes the **Capital Market
  Line**: a straight line from the risk-free rate tangent to the efficient frontier.
- **Tangency portfolio**: The point where the CML touches the efficient frontier; the
  optimal risky portfolio for all investors (under CAPM assumptions).
- **Separation theorem**: All investors hold the same risky portfolio (tangency portfolio),
  adjusting risk through leverage (borrowing/lending at the risk-free rate).
- In practice, the tangency portfolio is sensitive to expected return estimates and
  borrowing rates differ from lending rates, breaking the clean theoretical result.

---

## Portfolio Diversification

### Types of Diversification

1. **Naive diversification (1/N)**: Equal-weight across N assets. DeMiguel, Garlappi,
   Uppal (2009) showed 1/N outperforms many optimization methods out-of-sample for
   reasonable N, due to zero estimation error.
2. **Risk-based diversification**: Equal risk contribution, maximum diversification ratio.
3. **Factor diversification**: Diversify across return-generating factors rather than
   just assets.
4. **Temporal diversification**: Spread execution across time to diversify timing risk.

### Diversification Limits

- Correlation increases during market stress ("diversification fails when you need it most").
- Within-country equity diversification: most idiosyncratic risk is eliminated with 25-30
  stocks; systematic risk cannot be diversified away.
- International diversification adds value but correlations have increased with globalization.
- Alternative assets (real estate, commodities, private equity) offer diversification but
  with liquidity and transparency trade-offs.

---

## Limitations of Modern Portfolio Theory

1. **Normal distribution assumption**: Returns exhibit fat tails, skewness, and kurtosis
   that MPT ignores. Tail events occur far more frequently than Gaussian models predict.
2. **Static single-period framework**: MPT is a single-period model; real portfolios are
   managed dynamically over multiple periods with changing opportunity sets.
3. **Estimation sensitivity**: Small input changes produce radically different portfolios.
4. **Correlation instability**: Correlations are regime-dependent; MPT assumes they are constant.
5. **Ignores liquidity**: No consideration of transaction costs, market impact, or
   the ability to actually execute the recommended trades.
6. **Ignores higher moments**: Investors care about skewness (prefer positive) and kurtosis
   (prefer low) beyond just mean and variance.
7. **Utility function assumption**: Mean-variance is optimal only for quadratic utility or
   normally distributed returns -- neither holds in practice.

---

## Black-Litterman Model

### Motivation

Black and Litterman (1992) addressed the key practical problems of MVO:
- Traditional MVO produces extreme, unintuitive portfolios.
- Reverse-engineering expected returns from market cap weights (equilibrium returns)
  provides a stable starting point.
- Investors can then overlay subjective views with controlled confidence levels.

### Framework

1. **Start with equilibrium returns**: Implied excess returns from market cap weights
   assuming the market portfolio is mean-variance efficient.
   ```
   Pi = delta * Sigma * w_market
   ```
   where delta is the risk aversion parameter.

2. **Express views**: Investor views as linear combinations of expected returns with
   associated uncertainty.
   ```
   Q = P * mu + epsilon,  epsilon ~ N(0, Omega)
   ```

3. **Combine via Bayesian updating**: Posterior expected returns blend equilibrium and views.
   ```
   mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]
   ```

### Advantages

- Produces stable, diversified portfolios even without strong views.
- Views are expressed with confidence levels, naturally controlling deviation from benchmark.
- Widely adopted by institutional investors and quant funds.
- Naturally handles the "garbage in, garbage out" problem of traditional MVO.

---

## Risk Parity

### Philosophy

- Traditional 60/40 portfolios are dominated by equity risk (equities contribute ~90%
  of portfolio volatility despite being 60% of capital).
- Risk parity equalizes the **risk contribution** of each asset class, achieving true
  diversification of risk rather than capital.

### Implementation

- **Equal Risk Contribution (ERC)**: Each asset contributes equally to total portfolio risk.
  ```
  w_i * (Sigma * w)_i = w_j * (Sigma * w)_j  for all i, j
  ```
- Typically requires leverage on low-vol assets (bonds) to achieve target return.
- Bridgewater's All Weather Fund is the most famous risk parity implementation.

### Variants

1. **Naive risk parity**: Inverse volatility weighting (ignores correlations).
2. **True risk parity (ERC)**: Accounts for correlations in risk contribution.
3. **Factor risk parity**: Equalizes risk contribution from underlying factors.
4. **Hierarchical Risk Parity (HRP)**: Lopez de Prado (2016) uses hierarchical clustering
   and inverse-variance allocation along the dendrogram; avoids covariance matrix inversion.

### Pros and Cons

- **Pro**: More diversified risk exposure; historically strong risk-adjusted returns; less
  sensitive to return estimation errors.
- **Con**: Requires leverage; sensitive to correlation estimation; underperforms in strong
  equity bull markets; low-rate environments reduce bond contribution.

---

## Portfolio Construction for Indian Markets

### Asset Universe Considerations

1. **Equity segments**: Nifty 50 (large-cap), Nifty Midcap 150, Nifty Smallcap 250 each
   have distinct risk-return characteristics and correlation structures.
2. **Fixed income**: Government securities (G-secs), corporate bonds, SDL bonds, T-bills.
   Indian bond market is less liquid than equity, affecting portfolio rebalancing.
3. **Gold**: SGBs (Sovereign Gold Bonds) provide gold exposure with 2.5% annual interest;
   culturally significant asset class in India with genuine diversification benefits.
4. **REITs/InvITs**: Emerging asset class in India (Embassy, Mindspace, Brookfield) offering
   real estate exposure with liquidity.
5. **International**: RBI's Liberalized Remittance Scheme (LRS) allows USD 250K/year for
   international diversification; growing allocation via Motilal Oswal S&P 500 index fund, etc.

### India-Specific Portfolio Constraints

| Constraint                | Details                                              |
|---------------------------|------------------------------------------------------|
| Mutual fund regulations   | SEBI categorization limits overlap across fund categories |
| Tax efficiency            | LTCG (>1yr) taxed at 12.5% above Rs 1.25L; STCG at 20% |
| STT (Securities Transaction Tax) | Affects high-turnover strategies significantly  |
| Sectoral concentration    | Top 5 Nifty stocks can represent 35%+ of market cap  |
| Currency risk             | International allocations face INR/USD volatility     |
| Liquidity tiers           | Sharp drop in liquidity below Nifty 500 constituents  |

### Practical Portfolio Construction Framework

1. **Strategic asset allocation**: Define long-term target weights across equity (large/mid/small),
   fixed income, gold, and international based on risk tolerance and investment horizon.
2. **Tactical overlays**: Adjust weights based on regime indicators (growth/inflation
   quadrant, VIX levels, FII flow regimes).
3. **Within-equity construction**: Use factor tilts (momentum, quality, low-vol) within
   each capitalization segment.
4. **Rebalancing protocol**: Calendar-based (quarterly/semi-annual) or threshold-based
   (rebalance when allocation drifts >5% from target). Consider tax implications of
   rebalancing in India -- avoid triggering STCG when possible.
5. **Liquidity management**: Maintain 5-10% in liquid instruments (liquid funds, T-bills)
   for opportunistic deployment during market dislocations.

### Risk Management Overlay

- **Position limits**: Maximum single-stock exposure (e.g., 5% for large-cap, 3% for mid-cap).
- **Sector limits**: Maximum sector exposure (e.g., 25%) to avoid concentration.
- **Drawdown controls**: Systematic de-risking when portfolio drawdown exceeds thresholds.
- **Correlation monitoring**: Track rolling correlations between portfolio components;
  high correlation regimes signal reduced diversification effectiveness.
- **Stress testing**: Simulate portfolio performance under historical stress scenarios
  (2008 GFC, 2020 COVID, 2018 IL&FS, demonetization 2016).

---

## Key References

- Markowitz, H. (1952). "Portfolio Selection."
- Black, F. & Litterman, R. (1992). "Global Portfolio Optimization."
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive Diversification."
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios."
- Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample."
- Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix."
