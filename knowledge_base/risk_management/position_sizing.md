# Position Sizing Methods for Systematic Trading

## Overview

Position sizing determines HOW MUCH capital to allocate to each trade. It is
arguably the most important component of a trading system, yet receives the least
attention compared to entry and exit signals. Two traders with identical signals
will produce vastly different results based on position sizing alone.

Position sizing answers: "Given a trade signal, how many shares/contracts/units
should I trade?" The answer must balance return maximization against risk control,
accounting for portfolio-level constraints.

---

## Fixed Fractional Position Sizing

The simplest method: risk a fixed percentage of current equity on each trade.

    position_size = (equity * risk_fraction) / risk_per_unit

Where:
- equity = current account value
- risk_fraction = percentage of equity risked per trade (typically 0.5% to 2%)
- risk_per_unit = distance from entry to stop loss (in dollar terms per share)

### Example
Equity = $100,000, risk 1% per trade, stop loss 5% below entry at $50:
    risk_per_unit = $50 * 0.05 = $2.50
    position_size = ($100,000 * 0.01) / $2.50 = 400 shares

### Properties
- Anti-martingale: position size grows with equity, shrinks with losses
- Guarantees account cannot go to zero (asymptotic decline)
- After a drawdown of D%, you need a gain of D/(1-D)% to recover
- The risk fraction directly controls drawdown severity

### Choosing the Risk Fraction
A common heuristic links risk fraction to maximum tolerable drawdown:

    risk_fraction ~ max_tolerable_drawdown / (max_consecutive_losses * loss_factor)

For 20% max drawdown tolerance with expected worst streak of 10 losses:
    risk_fraction ~ 0.20 / (10 * 1.5) = 1.3%

The 1.5 factor accounts for the fact that consecutive losses compound.

---

## Fixed Ratio Position Sizing (Ryan Jones Method)

Increases position size at fixed profit intervals rather than proportionally.

    contracts = 1 + floor(equity_gain / delta)

Where delta is the required profit increment per additional contract.

### Properties
- Slower size increase during early growth (more conservative than fixed fractional)
- Faster recovery from drawdowns (size decreases more slowly)
- Requires choosing delta, which depends on the strategy's expected per-contract profit
- Better suited for futures/options with discrete contract sizes

### Choosing Delta
    delta = max_loss_per_contract * desired_safety_factor

For a futures strategy with worst-case loss of $2,000 per contract and safety factor 2:
    delta = $2,000 * 2 = $4,000

You add one contract for every $4,000 in cumulative profits.

---

## Volatility-Based Position Sizing

Size positions inversely proportional to their volatility so that each position
contributes approximately equal risk.

    position_size = (equity * target_risk) / (instrument_volatility * price)

Where:
- target_risk = desired portfolio volatility contribution per position
- instrument_volatility = annualized or per-period standard deviation of returns

### Using Dollar Volatility

    dollar_vol = price * percentage_volatility
    position_value = (equity * target_risk) / dollar_vol
    shares = position_value / price

### Example
Equity $500,000, target risk per position = 0.5%, stock at $100, daily vol = 2%:
    dollar_vol = $100 * 0.02 = $2.00
    position_value = ($500,000 * 0.005) / $2.00 = $1,250
    shares = $1,250 / $100 = 12.5 -> 12 shares (round down)

Wait -- that seems small. Note that target_risk = 0.5% means each position
moves your portfolio by ~0.5% per day on a 1-sigma basis. With 20 positions,
portfolio daily vol ~ 0.5% * sqrt(20) ~ 2.2% (if uncorrelated).

### Advantages
- Equalizes risk contribution across diverse instruments
- Naturally reduces exposure to volatile (risky) instruments
- Adapts dynamically as volatility changes
- Foundation of trend-following sizing (used by CTAs like AHL, Winton)

---

## ATR-Based Position Sizing

Average True Range (ATR) is the standard volatility proxy for position sizing.

    ATR = exponential or simple moving average of True Range over N periods
    True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)

### ATR Position Sizing Formula

    position_size = (equity * risk_fraction) / (ATR_N * ATR_multiplier)

Where ATR_multiplier converts ATR to your stop distance. Common: stop = 2 * ATR.

### Example (Turtle Trading Style)
Equity $1,000,000, risk 1% per trade, 20-day ATR of gold futures = $15 per oz,
contract size = 100 oz:

    dollar_risk_per_contract = $15 * 100 = $1,500
    position_size = ($1,000,000 * 0.01) / $1,500 = 6.67 -> 6 contracts

### ATR Lookback Period Selection
- Short (10-14 days): responsive to recent volatility, can whipsaw
- Medium (20 days): standard choice, balances responsiveness and stability
- Long (50-60 days): smooth, may lag during volatility regime changes
- Blended: use max(short_ATR, long_ATR) for conservative sizing

### Dynamic ATR Adjustment
Volatility clusters (GARCH effects) mean ATR changes over time:
- Rising ATR -> smaller positions (risk is increasing)
- Falling ATR -> larger positions (risk is decreasing)
This creates a natural counter-cyclical sizing mechanism.

---

## Risk Parity Position Sizing

Allocate capital so that each position contributes equally to total portfolio risk.

### Equal Risk Contribution (ERC) Formulation

For portfolio weight w_i, the risk contribution of asset i is:

    RC_i = w_i * (Sigma * w)_i / sigma_portfolio

Risk parity requires: RC_i = RC_j for all i, j.

The optimization problem:
    minimize: sum_i (RC_i - RC_target)^2
    subject to: sum(w_i) = 1, w_i >= 0

### Simplified Risk Parity (Inverse Volatility)

When correlations are equal or ignored:

    w_i = (1 / sigma_i) / sum_j(1 / sigma_j)

This is "naive risk parity" and works surprisingly well in practice because
cross-asset correlations are often small and unstable.

### Full Risk Parity with Correlations

Requires iterative optimization. Spinu (2013) provides an efficient algorithm:

    1. Start with inverse-volatility weights
    2. Compute risk contributions RC_i
    3. Adjust weights: w_i_new = w_i * (RC_target / RC_i)^kappa
    4. Normalize weights: w_i = w_i / sum(w_j)
    5. Repeat until convergence

Where kappa in (0.5, 1.0) controls step size.

### Practical Considerations
- Risk parity often implies leverage (bonds have lower vol, need more weight)
- Correlation estimates are unstable; use shrinkage estimators
- Rebalance frequency: monthly is common; more frequent adds transaction costs
- During crises, correlations spike toward 1.0, breaking the diversification assumption

---

## Correlation-Adjusted Position Sizing

Reduce position sizes when holding correlated positions to avoid concentration.

### Pairwise Adjustment

For two correlated positions with individual sizes S_1 and S_2 and
correlation rho:

    adjusted_S_1 = S_1 * 1 / sqrt(1 + (n-1) * avg_rho)

Where n = number of correlated positions and avg_rho = average pairwise correlation.

### Portfolio Heat Approach

Define "portfolio heat" as the total risk across all open positions:

    portfolio_heat = sum_i(position_size_i * stop_distance_i * correlation_factor_i)
    max_heat = equity * max_risk_fraction (e.g., 6% of equity)

If adding a new position would exceed max_heat, either:
1. Reduce the new position size proportionally
2. Reduce all existing positions to accommodate
3. Skip the trade entirely

### Sector/Factor Correlation Grouping

Group positions by correlation cluster:
- Within-group correlation > 0.5: treat as partially redundant
- Reduce individual position sizes by factor sqrt(1 / (1 + (n_group - 1) * rho))
- Example: 3 tech stocks with avg rho = 0.6:
  adjustment = 1 / sqrt(1 + 2 * 0.6) = 1 / sqrt(2.2) = 0.674
  Each position sized at 67.4% of what standalone sizing would suggest

---

## Practical Implementation Framework

### Layered Position Sizing System

    Layer 1: Base size from volatility/ATR method
    Layer 2: Adjust for correlation with existing portfolio
    Layer 3: Apply conviction scaling (signal strength)
    Layer 4: Enforce hard limits (max position, max sector, max leverage)
    Layer 5: Liquidity filter (size < X% of average daily volume)

### Signal Strength Scaling

Scale position size by signal confidence:

    final_size = base_size * signal_strength_multiplier

Where signal_strength_multiplier in [0.5, 1.5] based on:
- Number of confirming indicators
- Magnitude of the signal (e.g., z-score of mean-reversion signal)
- Regime classification confidence

Caution: signal-scaled sizing increases concentration risk. Cap the multiplier
and ensure total portfolio risk stays within bounds.

### Liquidity Constraints

    max_shares = ADV_20 * max_participation_rate / trading_days_to_fill
    position_size = min(desired_size, max_shares)

Where:
- ADV_20 = 20-day average daily volume
- max_participation_rate = 0.01 to 0.05 (1% to 5% of daily volume)
- trading_days_to_fill = 1 for liquid names, up to 5 for illiquid

Never size a position that would take more than a few days to liquidate
at reasonable participation rates.

### Rebalancing Protocol

    1. Daily: check that no position exceeds hard limits
    2. Weekly: recompute volatility-based sizes, adjust if deviation > 20%
    3. Monthly: full portfolio rebalance including correlation adjustment
    4. On signal change: resize immediately if conviction changes meaningfully

---

## Comparison of Methods

| Method          | Complexity | Adaptiveness | Best For                    |
|-----------------|------------|--------------|-----------------------------|
| Fixed Fractional| Low        | Medium       | Single-strategy accounts    |
| Fixed Ratio     | Low        | Low          | Futures, discrete sizing    |
| Volatility-Based| Medium     | High         | Multi-asset portfolios      |
| ATR-Based       | Medium     | High         | Trend following, CTA-style  |
| Risk Parity     | High       | High         | Asset allocation, macro     |
| Correlation-Adj | High       | High         | Equity stat-arb, multi-strat|

---

## Key References

- Van Tharp (1998). "Trade Your Way to Financial Freedom." Fixed fractional sizing.
- Jones, R. (1999). "The Trading Game." Fixed ratio method.
- Vince, R. (1990). "Portfolio Management Formulas." Optimal f and leverage space.
- Turtle Trading Rules (1983). Original ATR-based unit sizing system.
- Qian, E. (2006). "On the Financial Interpretation of Risk Contribution." Risk parity.
- Maillard, Roncalli, Teiletche (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." JoIM.
- Spinu, F. (2013). "An Algorithm for Computing Risk Parity Weights." SSRN.
- Roncalli, T. (2013). "Introduction to Risk Parity and Budgeting." Chapman & Hall.
- de Prado, M. Lopez (2018). "Advances in Financial Machine Learning." Ch. 10 on bet sizing.
