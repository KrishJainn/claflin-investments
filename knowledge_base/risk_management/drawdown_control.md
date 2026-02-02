# Drawdown Control and Recovery Analysis

## Overview

Drawdown is the decline in portfolio value from a peak to a subsequent trough.
It is the single most psychologically and operationally important risk measure
for traders and allocators. While volatility is symmetric, drawdown captures only
the downside -- the actual experience of losing money.

Drawdown governs:
- Investor redemptions (funds are shut down by drawdowns, not by volatility)
- Strategy credibility windows (how long you have before being fired)
- Leverage sustainability (margin calls are triggered by drawdowns)
- Psychological resilience (traders blow up from drawdown-induced tilt)

---

## Drawdown Measurement

### Point-to-Point Drawdown

At time t, the drawdown is:

    DD(t) = (Peak(t) - Value(t)) / Peak(t)

Where Peak(t) = max(Value(s)) for all s <= t.

The drawdown is always non-negative and equals zero when the portfolio is at a
new high-water mark.

### Maximum Drawdown (MDD)

The largest peak-to-trough decline over a given period:

    MDD = max over all t of DD(t)
    MDD = max over all t of (Peak(t) - Trough_after_peak(t)) / Peak(t)

MDD is a single number summarizing the worst historical loss experience.

### Drawdown Duration

Time from a peak until the portfolio recovers to that peak level:

    Duration = recovery_time - peak_time

Maximum drawdown duration (longest underwater period) is often more
psychologically relevant than maximum drawdown depth. A 15% drawdown lasting
18 months is worse for most investors than a 25% drawdown lasting 3 months.

### Average Drawdown

    Avg_DD = (1/T) * sum(DD(t)) for all t

Average drawdown captures the typical underwater experience, not just the
extreme. It is related to the Ulcer Index:

    Ulcer_Index = sqrt((1/T) * sum(DD(t)^2))

The Ulcer Index penalizes deep drawdowns more heavily (squared terms) and
is used in the Ulcer Performance Index (Martin ratio):

    Martin_Ratio = (return - risk_free) / Ulcer_Index

---

## Expected Maximum Drawdown

For a random walk with drift mu and volatility sigma over T periods, the
expected maximum drawdown can be approximated:

    E[MDD] ~ (sigma * sqrt(T)) * f(mu * sqrt(T) / sigma)

Where f is a complex function. Key approximations:

For a strategy with Sharpe ratio S over T trading days:

    E[MDD] ~ sigma_daily * sqrt(T) * g(S * sqrt(T / 252))

Magdon-Ismail and Atiya (2004) provide tables and closed-form approximations.

### Rule of Thumb Relationships

For normally distributed returns:

    E[MDD] ~ 2 * annual_vol / Sharpe_ratio  (for Sharpe > 0.5)

A strategy with 15% vol and Sharpe 1.5:
    E[MDD] ~ 2 * 0.15 / 1.5 = 20%

A strategy with 20% vol and Sharpe 0.5:
    E[MDD] ~ 2 * 0.20 / 0.5 = 80% (essentially certain ruin without adjustment)

### Recovery Time

Expected time to recover from a drawdown of depth d:

    E[recovery] ~ d / (mu - 0.5 * sigma^2)

For geometric returns. A 30% drawdown with mu = 10% annualized:
    E[recovery] ~ 0.30 / 0.10 = 3 years

This assumes the strategy continues performing at its historical rate, which
is not guaranteed. Recovery time uncertainty is enormous.

---

## Drawdown-Based Position Reduction

### Linear Scaling

Reduce position size linearly as drawdown deepens:

    size_multiplier = max(0, 1 - DD(t) / DD_max_threshold)

If DD_max_threshold = 20%:
- At 0% drawdown: full size (multiplier = 1.0)
- At 10% drawdown: half size (multiplier = 0.5)
- At 20% drawdown: flat (multiplier = 0.0)

### Stepped Reduction (Drawdown Tranches)

    DD < 5%:          100% of target size
    5% <= DD < 10%:   75% of target size
    10% <= DD < 15%:  50% of target size
    15% <= DD < 20%:  25% of target size
    DD >= 20%:        0% (fully stopped out)

### Exponential Reduction

    size_multiplier = exp(-k * DD(t))

Where k controls sensitivity. For k = 10:
- 5% DD: multiplier = 0.61
- 10% DD: multiplier = 0.37
- 20% DD: multiplier = 0.14

### Choosing a Reduction Schedule

The optimal schedule depends on whether drawdowns are caused by:

1. **Regime change** (strategy is broken): Aggressive reduction is correct.
   The strategy's edge has disappeared and continued trading increases losses.

2. **Normal variance** (strategy is fine, just unlucky): Reduction locks in
   losses and reduces the recovery rate. You are cutting size at exactly the
   wrong time.

This is the fundamental dilemma: you cannot know in real-time which case applies.

**Practical compromise:** Use moderate reduction (half position at 10-15% DD)
with a regime detection overlay. If regime indicators confirm a structural break,
accelerate the reduction. If indicators suggest normal variance, maintain a
minimum position to participate in recovery.

---

## Circuit Breakers

Hard rules that halt or drastically reduce trading when thresholds are breached.

### Daily Loss Limit

    If daily_PnL < -X% of equity:
        Stop trading for the remainder of the day
        Reduce tomorrow's position sizes by 50%

Typical X = 2% to 5% depending on strategy volatility.

### Weekly / Monthly Loss Limit

    If MTD_PnL < -Y% of starting_monthly_equity:
        Reduce all positions to 25% of target
        Do not add new positions until month-end review

Typical Y = 5% to 10%.

### Consecutive Loss Circuit Breaker

    If N consecutive losing trades (N = 5 to 10):
        Reduce position sizes by 50% for the next 2N trades
        Flag strategy for parameter review

This addresses the scenario where the market has changed and the strategy's
signals have lost their edge.

### Volatility Circuit Breaker

    If realized_vol > 2 * expected_vol:
        Scale positions by expected_vol / realized_vol
        Maintain until vol normalizes (5-day rolling vol < 1.5 * expected)

This automatically de-risks during volatility spikes (earnings, macro events,
crises) without relying on drawdown as a lagging indicator.

### Implementation Notes

- Circuit breakers must be coded as pre-trade checks, not post-trade analysis
- Log every circuit breaker trigger with timestamp, trigger condition, and action taken
- Review circuit breaker triggers monthly: too many false triggers waste opportunity;
  too few mean the thresholds are too loose
- Never override a circuit breaker manually during live trading (if tempted, that
  is precisely when you need it most)

---

## Drawdown in Strategy Evaluation

### Calmar Ratio

    Calmar = annualized_return / max_drawdown

Higher is better. Calmar > 1.0 is good; Calmar > 2.0 is excellent.
Timeframe-dependent: typically computed over 3 years.

### Sterling Ratio

    Sterling = annualized_return / (average_annual_max_drawdown + 10%)

The +10% is a buffer to avoid division by small numbers for low-DD strategies.
Some variants omit the buffer or use average of N worst drawdowns.

### Burke Ratio

    Burke = annualized_return / sqrt(sum(DD_i^2) / T)

Where DD_i are the individual drawdown depths. Penalizes frequent and deep
drawdowns via the root-mean-square formulation.

### Pain Ratio

    Pain_Ratio = annualized_return / Pain_Index
    Pain_Index = (1/T) * sum(DD(t))

Uses average drawdown rather than maximum. More stable than Calmar and less
sensitive to single outlier events.

### Drawdown-Based Strategy Comparison

When comparing strategies, always examine:

1. **Maximum drawdown depth** -- the worst single event
2. **Maximum drawdown duration** -- the longest recovery period
3. **Drawdown frequency** -- how often drawdowns exceed 5%, 10%, 15%
4. **Drawdown clustering** -- do drawdowns come in bursts? (regime-dependent)
5. **Recovery profile** -- V-shaped (quick snap back) vs. L-shaped (slow grind)

A strategy with lower max DD but frequent moderate drawdowns may be harder
to live with than one with rare but sharp drawdowns followed by quick recoveries.

---

## Recovery Analysis

### The Asymmetry Problem

Drawdown recovery is asymmetric:

    Loss     | Gain Required to Recover
    ---------|------------------------
    10%      | 11.1%
    20%      | 25.0%
    30%      | 42.9%
    40%      | 66.7%
    50%      | 100.0%
    60%      | 150.0%
    75%      | 300.0%
    90%      | 900.0%

This asymmetry is the fundamental argument for drawdown control: prevention
is vastly more efficient than recovery.

### Recovery Rate Analysis

Track the rate of recovery as a diagnostic:

    recovery_rate = (equity - trough) / (peak - trough)

Plot recovery_rate over time after each drawdown event. Patterns to watch:
- Slowing recovery rate: strategy may be degrading
- Recovery rate faster than historical: possibly taking too much risk in recovery
- Recovery stalls at 50-70%: common pattern when traders size up too aggressively
  trying to recover, then take another hit

### Monte Carlo Drawdown Analysis

Simulate the distribution of drawdowns using bootstrap or parametric methods:

    1. Generate 10,000 simulated equity curves from strategy return distribution
    2. Compute MDD for each simulation
    3. Report: median MDD, 95th percentile MDD, worst-case MDD
    4. Compare realized MDD to this distribution

If realized MDD exceeds the 95th percentile of simulated MDD, the strategy
is likely broken (not just unlucky). Consider:
- Halting the strategy pending review
- Reducing allocation permanently
- Investigating structural market changes

### Conditional Drawdown Analysis

    P(DD > 20% | DD > 10%) -- probability drawdown worsens once it starts
    P(DD > 30% | DD > 20%) -- probability of catastrophic extension

For normally distributed strategies, these conditional probabilities are stable.
For fat-tailed strategies, P(worse | bad) is much higher than normal models
predict. This is why "it can't get worse" is the most dangerous phrase in trading.

---

## Practical Drawdown Control Framework

### Pre-Trade Risk Budget

Before entering any position:

    1. Compute current portfolio drawdown
    2. Look up drawdown-adjusted position multiplier
    3. Compute base position size (volatility or ATR method)
    4. Apply drawdown multiplier
    5. Check against all circuit breakers
    6. Check portfolio heat (total risk across all positions)
    7. If all checks pass, execute. Otherwise, reduce or skip.

### Post-Trade Monitoring

    1. Real-time P&L tracking with drawdown computation
    2. Alert thresholds at 25%, 50%, 75% of max tolerable drawdown
    3. Automated position reduction at pre-specified drawdown levels
    4. End-of-day drawdown report with attribution (which positions contributed)
    5. Weekly drawdown regime assessment (is current DD normal variance or regime shift?)

### Strategy Lifecycle Management

    Phase 1 (Incubation): Paper trade or small size. Max DD tolerance = 15%.
    Phase 2 (Scaling): Increase size gradually. Max DD tolerance = 20%.
    Phase 3 (Full Allocation): Target size. Max DD tolerance = 25%.
    Phase 4 (Review): If MDD breached, reduce to Phase 1 size and evaluate.

Never skip phases. A strategy that draws down 30% in Phase 1 should be killed,
not given more capital because "it's due for a recovery."

---

## Key References

- Magdon-Ismail, Atiya (2004). "Maximum Drawdown." Risk Magazine.
  Closed-form approximations for expected MDD.
- Chekhlov, Uryasev, Zabarankin (2005). "Drawdown Measure in Portfolio Optimization."
  Conditional Drawdown at Risk (CDaR) framework.
- Grossman, Zhou (1993). "Optimal Investment Strategies for Controlling Drawdowns."
  Mathematical Finance. Optimal portfolio insurance.
- Hamelink, Hoesli (2004). "Maximum Drawdown and the Allocation to Real Estate."
- Martin, P. and McCann, B. (1989). "The Investor's Guide to Fidelity Funds."
  Origin of the Ulcer Index.
- Young, T.W. (1991). "Calmar Ratio: A Smoother Tool." Futures Magazine.
- Lo, Mamaysky, Wang (2000). "Foundations of Technical Analysis." Journal of Finance.
  Statistical properties of drawdowns.
- Burghardt, Duncan, Liu (2003). "Understanding Drawdowns." Working paper.
  Distributional properties and duration analysis.
