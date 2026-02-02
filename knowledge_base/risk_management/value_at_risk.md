# Value at Risk (VaR)

## Definition

VaR answers: "What is the maximum loss over a given time horizon at a given confidence level?"

Example: "1-day 95% VaR of $100,000" means there is a 5% chance of losing more than $100,000 in one day.

## Calculation Methods

### 1. Historical Simulation

```python
def historical_var(returns, confidence=0.95):
    """
    Calculate VaR from historical returns.
    """
    return np.percentile(returns, (1 - confidence) * 100)
```

### 2. Parametric (Variance-Covariance)

```python
def parametric_var(portfolio_value, volatility, confidence=0.95, days=1):
    """
    Assumes normal distribution of returns.
    """
    from scipy.stats import norm
    z_score = norm.ppf(1 - confidence)
    var = portfolio_value * volatility * z_score * np.sqrt(days)
    return abs(var)
```

### 3. Monte Carlo Simulation

```python
def monte_carlo_var(portfolio_value, returns, n_simulations=10000, confidence=0.95):
    """
    Simulate future returns and calculate VaR.
    """
    mu = returns.mean()
    sigma = returns.std()
    
    simulated = np.random.normal(mu, sigma, n_simulations)
    simulated_pnl = portfolio_value * simulated
    
    return np.percentile(simulated_pnl, (1 - confidence) * 100)
```

## Limitations of VaR

1. **Tail Risk**: VaR doesn't tell you how bad losses can get beyond the threshold
2. **Non-Subadditivity**: VaR of combined portfolios can exceed sum of individual VaRs
3. **Normal Assumption**: Markets have fat tails
4. **Static**: Assumes volatility and correlations are stable

## Expected Shortfall (CVaR)

Average of losses beyond VaR - better captures tail risk:

```python
def expected_shortfall(returns, confidence=0.95):
    """
    Also called Conditional VaR or CVaR.
    """
    var = np.percentile(returns, (1 - confidence) * 100)
    return returns[returns <= var].mean()
```

## AQTIS Risk Limits

| Metric | Limit |
|--------|-------|
| 1-day 95% VaR | 3% of portfolio |
| Expected Shortfall | 5% of portfolio |
| Max Drawdown | 15% |

## References

- Jorion, P. "Value at Risk"
- Hull, J. "Risk Management and Financial Institutions"
