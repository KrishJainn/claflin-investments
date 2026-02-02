# Momentum Trading Strategies

## Core Concept

Momentum is the tendency for assets that have performed well (poorly) to continue performing well (poorly) over intermediate horizons (3-12 months).

## Types of Momentum

### 1. Cross-Sectional (Relative) Momentum

Compare assets to each other. Go long top performers, short bottom performers.

```python
def cross_sectional_momentum(returns, lookback=252, hold=21):
    # Skip most recent month (short-term reversal)
    formation_returns = returns.iloc[-(lookback + hold):-hold].sum()
    ranks = formation_returns.rank(pct=True)
    
    signal = pd.Series(0, index=returns.columns)
    signal[ranks > 0.8] = 1   # Long top 20%
    signal[ranks < 0.2] = -1  # Short bottom 20%
    return signal
```

### 2. Time-Series (Absolute) Momentum

Compare asset to itself. Go long if trending up, short/flat if trending down.

```python
def time_series_momentum(prices, lookback=252):
    returns = prices.pct_change(lookback).iloc[-1]
    if returns > 0:
        return 1
    elif returns < 0:
        return -1
    return 0
```

### 3. Dual Momentum (Antonacci)

Combine both: Use time-series momentum as filter, then apply cross-sectional.

## Momentum Crashes

Momentum strategies are vulnerable to sharp reversals after market stress:
- 2009: -73% drawdown in momentum
- Frequency: ~2-3 times per decade

**Mitigation**:
```python
def momentum_with_crash_protection(prices, vol_threshold=0.3):
    realized_vol = prices.pct_change().iloc[-21:].std() * np.sqrt(252)
    if realized_vol > vol_threshold:
        exposure = 0.5 * (vol_threshold / realized_vol)
    else:
        exposure = 1.0
    return exposure
```

## Regime Considerations

| Regime | Momentum Performance | Adjustment |
|--------|---------------------|------------|
| Trending Up | Strong | Full allocation |
| Low Volatility | Moderate | Full allocation |
| Mean Reverting | Weak | Reduce allocation |
| High Volatility | Crash risk | Reduce significantly |

## References

- Jegadeesh & Titman (1993). "Returns to Buying Winners and Selling Losers"
- Daniel & Moskowitz (2016). "Momentum Crashes"
