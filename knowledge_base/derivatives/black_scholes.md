# Black-Scholes Option Pricing Model

## The Formula

For a European call option:

```
C = S*N(d1) - K*e^(-rT)*N(d2)

d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d2 = d1 - σ√T
```

For a European put:
```
P = K*e^(-rT)*N(-d2) - S*N(-d1)
```

## Python Implementation

```python
from scipy.stats import norm
import numpy as np

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes option pricing.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: "call" or "put"
    
    Returns:
        Option price
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price
```

## Model Assumptions

1. **Geometric Brownian Motion**: Stock follows dS = μS dt + σS dW
2. **Constant volatility**: σ is fixed (violated in practice)
3. **No dividends**: Or adjusted for known dividends
4. **European exercise**: No early exercise
5. **No transaction costs**: Frictionless trading
6. **Risk-free rate constant**: r is fixed
7. **Log-normal returns**: Returns are normally distributed

## Implied Volatility

The volatility that makes Black-Scholes price equal market price:

```python
from scipy.optimize import brentq

def implied_volatility(market_price, S, K, T, r, option_type="call"):
    """
    Calculate implied volatility from market price.
    """
    def objective(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - market_price
    
    try:
        iv = brentq(objective, 0.001, 5.0)
        return iv
    except:
        return np.nan
```

## Limitations

1. **Volatility Smile**: Real markets show non-constant IV across strikes
2. **Fat Tails**: Actual returns have more extreme moves than normal
3. **Jumps**: Markets can gap, violating continuous price assumption
4. **Early Exercise**: American options need different models

## When to Use

- Quick option pricing estimates
- Greeks calculation
- IV calculation from market prices
- Educational understanding of option mechanics

## References

- Black, F., Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Hull, J. "Options, Futures, and Other Derivatives"
