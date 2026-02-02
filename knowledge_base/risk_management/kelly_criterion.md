# Kelly Criterion for Position Sizing

## Core Formula

The Kelly Criterion determines optimal bet size to maximize long-term geometric growth rate of capital:

```
f* = (p * b - q) / b
```

Where:
- **f*** = Optimal fraction of capital to risk
- **p** = Probability of winning
- **q** = Probability of losing (1 - p)
- **b** = Odds ratio (average win / average loss)

For trading with equal win/loss amounts: `f* = 2p - 1`

## Practical Trading Application

### Quarter Kelly (Recommended for AQTIS)

Full Kelly is mathematically optimal but produces extreme drawdowns (50%+ possible). AQTIS uses quarter Kelly:

```python
def calculate_kelly(trades):
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    if not losses or not wins:
        return 0

    win_rate = len(wins) / len(trades)
    avg_win = np.mean([t.pnl_percent for t in wins])
    avg_loss = abs(np.mean([t.pnl_percent for t in losses]))

    if avg_loss == 0:
        return 0

    b = avg_win / avg_loss
    kelly = (win_rate * b - (1 - win_rate)) / b

    # Quarter Kelly for safety
    return max(0, kelly * 0.25)
```

### Why Fractional Kelly?

| Kelly Fraction | Expected Growth | Max Drawdown Risk | Use Case |
|----------------|-----------------|-------------------|----------|
| Full (1.0x) | Maximum | 50%+ | Theoretical only |
| Half (0.5x) | 75% of max | ~25% | Aggressive |
| Quarter (0.25x) | 50% of max | ~12% | AQTIS default |
| Eighth (0.125x) | 25% of max | ~6% | Very conservative |

## Regime-Based Adjustments

```python
def regime_adjusted_kelly(base_kelly, regime):
    adjustments = {
        "low_volatility": 1.0,
        "trending_up": 1.0,
        "mean_reverting": 0.8,
        "trending_down": 0.5,
        "high_volatility": 0.5,
    }
    return base_kelly * adjustments.get(regime, 0.75)
```

## When Kelly Fails

1. **Estimation Error**: Win rate and payoffs are estimated from historical data
2. **Non-Stationarity**: Markets change. Historical win rates may not persist
3. **Fat Tails**: Kelly assumes bounded losses. Markets can gap
4. **Serial Correlation**: Kelly assumes independent bets

## References

- Kelly, J.L. (1956). "A New Interpretation of Information Rate"
- Thorp, E.O. (2006). "The Kelly Criterion in Blackjack and the Stock Market"
