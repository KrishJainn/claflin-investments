# Slippage: Measurement, Modeling, and Mitigation

## Overview

Slippage is the difference between the expected price of a trade and the actual execution price.
It is one of the most significant hidden costs in trading, and a primary reason that backtested
strategies underperform in live markets. Understanding, measuring, and reducing slippage is
critical for profitable algorithmic trading.

---

## 1. Definition and Types

### 1.1 Formal Definition
- Slippage = Execution Price - Expected Price
  - For buys: positive slippage means you paid MORE than expected (unfavorable)
  - For sells: positive slippage means you received LESS than expected (unfavorable)
- Expected price may be defined as: mid-price at decision time, arrival price, or limit price

### 1.2 Types of Slippage
- **Spread slippage**: crossing the bid-ask spread to obtain immediate execution
  - Cost = half-spread for a single-side trade against mid-price benchmark
- **Impact slippage**: price movement caused by your own order consuming book depth
  - Increases nonlinearly with order size relative to available liquidity
- **Delay slippage (timing cost)**: price drift between decision and execution
  - Caused by latency, queuing, or deliberate pacing of execution
- **Opportunity cost**: when limit orders fail to fill and the price moves away
  - Unmeasured but critical: the cost of trades you DIDN'T make

---

## 2. Causes of Slippage

### 2.1 Market Structure Factors
- **Bid-ask spread**: wider spreads mean higher base slippage cost
  - Spread is a function of: volatility, tick size, adverse selection, competition
- **Market depth**: thin books amplify impact; walking the book is expensive
- **Fragmentation**: liquidity split across venues increases routing complexity
- **Latency**: delay between signal and execution allows price to move

### 2.2 Order-Specific Factors
- **Order size**: larger orders have disproportionately higher slippage
  - Rule of thumb: slippage ~ C * sigma * sqrt(Q / ADV)
  - sigma = daily volatility, Q = order quantity, ADV = average daily volume
- **Order type**: market orders guarantee fill but maximize spread cost
- **Urgency**: aggressive time horizon forces liquidity consumption
- **Information content**: orders correlated with future price moves suffer adverse selection

### 2.3 Market Condition Factors
- **Volatility**: higher volatility widens spreads and increases timing risk
- **News events**: liquidity evaporates around announcements; spreads spike
- **Time of day**: wider spreads at open/close; thinnest liquidity midday for some assets
- **Holiday/low-volume days**: reduced depth amplifies all slippage components

---

## 3. Measurement Frameworks

### 3.1 Arrival Price Benchmark (Perold's Implementation Shortfall)
- Defined by Perold (1988) as the difference between paper portfolio return and actual return
- Implementation Shortfall = (Paper Return) - (Actual Return)
- Decomposition:
  - IS = Delay Cost + Trading Cost + Opportunity Cost + Fees
  - Delay Cost: price move from decision to first execution
  - Trading Cost: price move from first execution to completion (spread + impact)
  - Opportunity Cost: unfilled portion * price move over horizon
- This is the gold standard benchmark for institutional execution quality analysis

### 3.2 VWAP Slippage
- VWAP Slippage = Avg Execution Price - VWAP over execution window
- Useful for measuring whether you traded better or worse than the market average
- Limitations: VWAP is an ex-post benchmark; your own trades influence it
- Appropriate for orders that are small relative to total volume

### 3.3 Reversion-Adjusted Slippage
- Measures permanent vs temporary components of observed slippage
- Temporary impact: the portion that reverts after trading completes
- Permanent impact: the portion that persists in the new equilibrium price
- Method: compare execution price to price 30-60 minutes after completion
  - Reversion = Execution Price - Post-Trade Price
  - Permanent component = Total Slippage - Reversion

### 3.4 Basis Point Measurement
- Slippage (bps) = ((Execution Price - Benchmark Price) / Benchmark Price) * 10000
- Standardizing in bps allows comparison across assets with different price levels
- Track rolling distributions: mean, median, standard deviation, and tail percentiles
- Alert thresholds: flag executions beyond 2 standard deviations from historical mean

---

## 4. Slippage Models

### 4.1 Linear Model
- Slippage = alpha + beta * (Q / ADV) + epsilon
- Simple but underestimates impact for large orders
- Useful as a first approximation for small participation rates (<5% of ADV)

### 4.2 Square-Root Model (Empirical Standard)
- Slippage = sigma * C * sqrt(Q / ADV)
- C = constant typically calibrated between 0.1 and 0.5 depending on asset class
- sigma = daily volatility
- Empirically validated across equities, futures, and FX
- Derived from information-theoretic arguments (Kyle, 1985; Barra, Kissell)
- For equities: C ~ 0.3 is a common starting point

### 4.3 Power-Law Model
- Slippage = eta * (Q / ADV)^delta
- delta typically between 0.4 and 0.7 (square-root corresponds to delta = 0.5)
- Calibrate eta and delta from historical execution data
- More flexible than square-root; captures asset-specific impact profiles

### 4.4 Almgren-Chriss Temporary Impact
- Instantaneous cost per share: g(v) = eta * |v|^alpha * sign(v)
- v = trading rate (shares per unit time)
- alpha ~ 0.5 to 1.0 (linear to concave)
- Integrates with optimal execution framework for scheduling

### 4.5 Regime-Dependent Models
- Slippage parameters vary with market regime (volatility, volume, spread regime)
- Fit separate models for: normal conditions, high-volatility, low-liquidity, event-driven
- Use regime detection (HMM or threshold-based) to select appropriate model
- Improves accuracy of pre-trade cost estimates by 20-40% vs single-regime models

---

## 5. Reducing Slippage in Algorithmic Trading

### 5.1 Execution Algorithm Selection
- **Small orders (<1% ADV)**: market orders or aggressive limits are fine; slippage minimal
- **Medium orders (1-10% ADV)**: VWAP or TWAP to blend with market flow
- **Large orders (>10% ADV)**: Implementation Shortfall algo with multi-day scheduling
- Match algorithm urgency to alpha decay: fast alpha -> aggressive execution

### 5.2 Venue Selection and Smart Routing
- Route to venues with highest displayed depth at NBBO
- Use dark pools for large blocks to avoid signaling (but monitor fill quality)
- Internalization: some brokers fill against internal flow at midpoint
- Measure venue toxicity: track adverse selection per venue

### 5.3 Order Sizing and Scheduling
- Break large orders into child orders: never exceed 5-10% of interval volume
- Randomize child order timing to reduce footprint detection
- Use minimum quantity (MinQty) to avoid information leakage from small fills
- Reserve a portion for opportunistic fills on favorable price dislocations

### 5.4 Passive Execution Strategies
- Post limit orders at or near the spread to earn the spread rather than pay it
- Use pegged orders to maintain position relative to NBBO
- Midpoint peg in dark pools: trade at the midpoint, zero spread cost
- Risk: adverse selection; only use for low-urgency, low-information-content orders

### 5.5 Real-Time Adaptation
- Monitor realized slippage vs pre-trade estimate during execution
- If slippage exceeds threshold: pause, reassess, adjust schedule
- Use real-time volatility and spread estimates to re-optimize remaining schedule
- Kill switch: abort execution if total cost exceeds risk budget

---

## 6. Slippage in Backtesting

### 6.1 The Backtesting Illusion
- Backtests without realistic slippage models massively overstate performance
- Common mistake: assume fills at the close price with zero cost
- Even 1-2 bps of slippage per trade can destroy high-frequency strategies

### 6.2 Realistic Slippage Modeling in Backtests
- **Minimum**: add half-spread cost to every trade
- **Better**: use square-root model: sigma * C * sqrt(Q / ADV) per trade
- **Best**: use tick-by-tick order book simulation with queue models
- Calibrate slippage model from historical execution data, not assumptions

### 6.3 Volume Participation Limits
- Never assume you can trade more than 5-10% of bar volume without impact
- If backtest signal triggers in a 5-minute bar with 1000 shares volume:
  - Maximum realistic fill = 50-100 shares (5-10% participation)
  - Remainder must be spread across subsequent bars with additional slippage
- Apply capacity constraints: total strategy AUM limited by worst-case slippage budget

### 6.4 Latency and Fill Assumptions
- Assume minimum latency of 1-10ms for co-located systems; 50-200ms for retail
- Model fill probability for limit orders: not all posted limits will execute
- Account for partial fills: a 1000-share limit may only fill 200 shares
- In backtesting, penalize missed fills as opportunity cost

### 6.5 Slippage Sensitivity Analysis
- Run backtest across a range of slippage assumptions: 0.5x, 1x, 2x, 3x baseline
- A strategy that only works at 0.5x slippage is not robust
- Minimum robustness check: strategy profitable at 2x estimated slippage
- Plot Sharpe ratio vs slippage multiplier: identify the breakeven slippage level

---

## 7. Common Pitfalls

1. **Ignoring slippage entirely**: the most common and most damaging error in backtesting
2. **Using fixed slippage (e.g., 5 bps flat)**: slippage varies by order size, volatility, and asset
3. **Not tracking slippage in live trading**: compare expected vs realized to detect execution decay
4. **Assuming you are the only participant**: your orders affect the market; model your own impact
5. **Over-relying on dark pools**: fill rates are uncertain; signaling risk exists in some venues
6. **Ignoring opportunity cost**: unfilled orders have a cost; measure it explicitly
7. **Using daily OHLC for fill simulation**: intra-bar prices may not have been accessible in sequence

---

## 8. Implementation Checklist

- [ ] Build slippage estimator using square-root model calibrated to your asset universe
- [ ] Log every execution: timestamp, price, size, venue, benchmark price
- [ ] Compute IS decomposition daily: delay, impact, opportunity, fees
- [ ] Create slippage dashboards: rolling averages by strategy, asset, venue, time-of-day
- [ ] Run weekly regression: actual slippage vs model prediction; re-calibrate quarterly
- [ ] Integrate slippage model into backtester with volume participation constraints
- [ ] Perform slippage sensitivity analysis on every new strategy before deployment

---

## 9. Key References

- **Perold, A.F. (1988)**. "The Implementation Shortfall: Paper vs. Reality." *Journal of Portfolio Management*, 14(3), 4-9. -- Defined the implementation shortfall framework.
- **Kissell, R. & Glantz, M. (2003)**. *Optimal Trading Strategies*. AMACOM. -- Comprehensive slippage and execution cost modeling.
- **Almgren, R. et al. (2005)**. "Direct Estimation of Equity Market Impact." *Risk*, 18(7), 58-62. -- Empirical impact estimation.
- **Gatheral, J. (2010)**. "No-Dynamic-Arbitrage and Market Impact." *Quantitative Finance*, 10(7), 749-759. -- Theoretical foundations of impact models.
- **Bacidore, J. et al. (2003)**. "Order Submission Strategies, Liquidity Supply, and Trading in Pennies on the New York Stock Exchange." *Journal of Financial Markets*, 6(3), 337-362. -- Empirical order submission.
- **Frazzini, A., Israel, R., & Moskowitz, T. (2018)**. "Trading Costs." Working Paper. -- Large-scale empirical analysis of institutional trading costs.
