# Option Greeks

## Overview

The Greeks measure the sensitivity of an option's price to changes in underlying parameters.
They are the partial derivatives of the option pricing function and form the foundation of
options risk management, hedging, and algorithmic trading. A quantitative trader who does not
understand Greeks deeply will not survive in options markets.

---

## First-Order Greeks

### Delta (d_C/d_S)

Delta measures the rate of change of the option price with respect to the underlying price.

**Call Delta = N(d1)** (ranges from 0 to +1)
**Put Delta = N(d1) - 1 = -N(-d1)** (ranges from -1 to 0)

Key properties:
- ATM options have delta ~ 0.50 (calls) or ~ -0.50 (puts)
- Deep ITM calls approach delta = 1.0; deep OTM calls approach delta = 0.0
- Delta approximates the probability of finishing ITM under the risk-neutral measure
  (more precisely, it is the hedge ratio, not the exact probability)
- Delta is the number of shares needed to hedge one option contract

**Practical hedging**: To delta-hedge a portfolio of options, compute the aggregate delta
across all positions and take an offsetting position in the underlying:

Shares to trade = -Portfolio Delta * Contract Multiplier

Rebalance frequency depends on gamma exposure, transaction costs, and the trader's risk
tolerance. More frequent rebalancing reduces hedging error but increases costs.

### Theta (d_C/d_t)

Theta measures time decay -- the rate at which the option loses value as time passes,
all else equal.

**Theta_call = -[S * n(d1) * sigma] / [2 * sqrt(T)] - r * K * e^(-rT) * N(d2)**
**Theta_put = -[S * n(d1) * sigma] / [2 * sqrt(T)] + r * K * e^(-rT) * N(-d2)**

Where n(x) is the standard normal PDF.

Key properties:
- Theta is almost always negative for long option positions (time decay hurts)
- ATM options have the highest absolute theta (most time value to lose)
- Theta accelerates as expiration approaches (gamma-theta tradeoff)
- For ATM options, theta is approximately: Theta ~ -S * sigma / (2 * sqrt(2*pi*T))

**The Gamma-Theta Tradeoff**: For a delta-hedged portfolio, daily P&L is approximately:

Daily P&L ~ (1/2) * Gamma * (dS)^2 + Theta * dt

If realized moves are larger than implied, gamma profits exceed theta costs (long vol wins).
If realized moves are smaller, theta costs dominate (short vol wins). This is the fundamental
equation of options market-making.

### Vega (d_C/d_sigma)

Vega measures sensitivity to changes in implied volatility.

**Vega = S * sqrt(T) * n(d1)**

Note: Vega is identical for calls and puts with the same strike and expiry (follows from
put-call parity).

Key properties:
- Always positive for long options (higher vol = higher option value)
- ATM options have the highest vega
- Vega increases with time to expiration (longer-dated options are more vol-sensitive)
- Vega is quoted as price change per 1 percentage point change in IV

**Vega in algorithmic trading**:
- Vega exposure is the primary risk factor for volatility trading strategies
- A delta-hedged option position is essentially a bet on volatility: long vega profits
  when IV rises, short vega profits when IV falls
- Vega-neutral portfolios can be constructed by combining options with different expirations,
  though this introduces vanna and volga risk

### Rho (d_C/d_r)

Rho measures sensitivity to changes in the risk-free interest rate.

**Rho_call = K * T * e^(-rT) * N(d2)**
**Rho_put = -K * T * e^(-rT) * N(-d2)**

Key properties:
- Calls have positive rho (higher rates increase call value)
- Puts have negative rho (higher rates decrease put value)
- Rho is largest for long-dated, deep ITM options
- Generally the least important Greek for short-dated equity options
- Becomes critical for long-dated options (LEAPS) and interest rate derivatives

---

## Second-Order Greeks

### Gamma (d^2_C/d_S^2 = d_Delta/d_S)

Gamma measures the rate of change of delta with respect to the underlying price.

**Gamma = n(d1) / (S * sigma * sqrt(T))**

Gamma is identical for calls and puts with the same strike and expiry.

Key properties:
- Always positive for long options (convexity)
- Highest for ATM options near expiration (gamma spike)
- Approaches zero for deep ITM/OTM options
- Short gamma positions face explosive risk near expiration (pin risk)

**Gamma scalping**: A delta-hedged long gamma position profits from realized volatility
exceeding implied volatility. The trader repeatedly rebalances delta as the underlying moves:
- Underlying rises -> delta increases -> sell shares to rebalance -> lock in profit
- Underlying falls -> delta decreases -> buy shares to rebalance -> lock in profit
Each rebalance captures realized convexity. The cost is theta decay.

**Dollar Gamma**: Gamma * S^2 / 100. Represents the dollar P&L impact of a 1% move in
the underlying. More useful than raw gamma for comparing across different-priced underlyings.

### Vanna (d^2_C / (d_S * d_sigma) = d_Delta/d_sigma = d_Vega/d_S)

Vanna measures how delta changes with volatility, or equivalently, how vega changes with
the underlying price.

**Vanna = -n(d1) * d2 / sigma**

Key properties:
- Important for managing the interaction between directional and volatility risk
- Drives the skew dynamics: when spot drops and vol rises simultaneously (leverage effect),
  vanna amplifies the delta change
- Critical for understanding how the volatility surface moves with spot

### Volga / Vomma (d^2_C/d_sigma^2 = d_Vega/d_sigma)

Volga measures the sensitivity of vega to changes in volatility -- the convexity of the
option price with respect to volatility.

**Volga = Vega * d1 * d2 / sigma**

Key properties:
- Positive for OTM and ITM options; near zero for ATM options
- Traders use volga to understand how their vega exposure changes as vol moves
- Important for pricing exotic options via vanna-volga method

### Charm (d_Delta/d_t = d_Theta/d_S)

Charm measures the rate of change of delta with respect to time.

**Charm = -n(d1) * [2*(r-q)*T - d2*sigma*sqrt(T)] / [2*T*sigma*sqrt(T)]**

Key properties:
- Tells you how your delta hedge drifts overnight even without a move in spot
- Critical for managing overnight gamma risk
- Largest for near-ATM options approaching expiration

### Speed (d_Gamma/d_S = d^3_C/d_S^3)

Speed measures how gamma changes with the underlying price.

**Speed = -Gamma * (1 + d1 / (sigma * sqrt(T))) / S**

Important for large-gamma portfolios where the curvature of the hedging function matters.

---

## Greeks in Risk Management

### Portfolio-Level Greek Aggregation

For a portfolio of N options positions, the aggregate Greek is:

**Portfolio_Greek = sum over i of [quantity_i * multiplier_i * Greek_i]**

Risk limits are typically set on aggregate Greeks:
- **Delta limit**: Maximum directional exposure (e.g., +/- $5M delta)
- **Gamma limit**: Maximum convexity exposure, often per underlying and per expiry bucket
- **Vega limit**: Maximum volatility exposure, often decomposed by tenor
- **Theta limit**: Maximum daily time decay (both positive and negative)

### Scenario Analysis

Greeks provide a first-order (delta) and second-order (gamma) Taylor expansion of P&L:

**dP ~ Delta * dS + (1/2) * Gamma * dS^2 + Theta * dt + Vega * d_sigma + Rho * dr**

For large moves, include cross-terms:

**+ Vanna * dS * d_sigma + Volga * (1/2) * d_sigma^2 + Charm * dS * dt + ...**

Full revaluation (repricing all options under the scenario) is preferred for stress testing,
as the Taylor expansion breaks down for large moves.

### Hedging Strategies Using Greeks

| Objective              | Primary Greek | Hedging Instrument       |
|------------------------|---------------|--------------------------|
| Eliminate directional  | Delta         | Underlying shares/futures|
| Reduce convexity risk  | Gamma         | Options (same expiry)    |
| Reduce time decay      | Theta         | Spread against other options|
| Neutralize vol risk    | Vega          | Options (different expiry)|
| Interest rate hedge    | Rho           | Interest rate instruments|

**Delta-gamma hedging** requires at minimum two instruments: the underlying (for delta) and
another option (for gamma). Solving the system of equations:

q_underlying * 1 + q_option * Delta_option = -Portfolio_Delta
q_underlying * 0 + q_option * Gamma_option = -Portfolio_Gamma

---

## Greeks in Algorithmic Trading

### Market Making
Options market makers continuously quote bid/ask prices and manage inventory. Greeks are
computed in real-time across the entire book:
- Delta is hedged in the underlying, typically aggregated per underlying
- Gamma and theta represent the core P&L driver: market makers are usually net short gamma,
  collecting theta as compensation for convexity risk
- Vega is managed across the term structure, often with calendar spreads

### Volatility Arbitrage
- Compare implied vol (from option prices) to forecasted realized vol
- If IV > forecast RV: sell options (short vega, collect theta, short gamma)
- If IV < forecast RV: buy options (long vega, pay theta, long gamma)
- Delta-hedge to isolate the volatility bet
- P&L attribution: Vega * d_IV + (1/2) * Gamma * (realized_move^2 - implied_move^2)

### Dynamic Hedging Algorithms
Production delta-hedging systems must decide:
1. **Rebalance trigger**: Time-based (every N minutes), delta-threshold (when delta drift
   exceeds X), or optimal (minimizing variance + cost). Zakamouline (2006) provides an
   optimal hedging bandwidth.
2. **Hedging instrument**: Stock, futures, or other options
3. **Order type**: Aggressive (market orders for immediate delta reduction) vs. passive
   (limit orders to capture spread, accepting temporary delta drift)

### Greeks-Based Signal Generation
- **Gamma exposure (GEX)**: Aggregate gamma of market makers across strikes. When GEX is
  large and positive, market makers dampen moves (buy dips, sell rallies). When negative,
  they amplify moves. GEX is now a widely tracked indicator.
- **Vanna flows**: As spot moves and dealers rehedge vanna, it creates systematic flows
  in the underlying. This is particularly strong around large expirations (OPEX).
- **Charm flows**: Overnight delta drift due to charm creates predictable hedging flows
  at the open.

---

## Common Pitfalls

1. **Assuming Greeks are constant**: Greeks change continuously. A position that is
   delta-neutral now may not be in an hour. Monitor charm and gamma.
2. **Ignoring cross-Greeks**: Vanna and volga matter, especially for OTM options and
   in volatile markets. A pure delta/gamma/vega framework misses important risk.
3. **Discrete vs. continuous Greeks**: BSM Greeks assume continuous hedging. In practice,
   discrete hedging introduces path-dependent P&L that can deviate significantly from
   Greek-implied estimates.
4. **Greeks for American options**: BSM Greeks are for European options. American option
   Greeks require numerical differentiation of the pricing model (bump-and-reprice).
5. **Vega is not flat across strikes**: Different strikes have different vegas. Treating
   vega as a single number ignores the risk of the smile moving non-parallel.
6. **Theta is not smooth**: Weekend and holiday theta must be modeled carefully. Some
   systems distribute weekend theta across Friday-Monday, others assign it to Friday.

---

## Key References

- Taleb, N.N. "Dynamic Hedging: Managing Vanilla and Exotic Options" - Wiley
- Hull, J.C. "Options, Futures, and Other Derivatives" - Standard reference
- Wilmott, P. "Paul Wilmott on Quantitative Finance" - Comprehensive treatment
- Zakamouline, V. (2006). "European Option Pricing and Hedging with Both Fixed and
  Proportional Transaction Costs" - Journal of Economic Dynamics and Control
- Bouchaud, J.P. & Potters, M. "Theory of Financial Risk and Derivative Pricing" -
  Cambridge University Press
