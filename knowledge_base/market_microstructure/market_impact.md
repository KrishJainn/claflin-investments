# Market Impact: Theory, Models, and Strategy Capacity

## Overview

Market impact is the effect that a trade has on the price of the asset being traded. It is
the single largest component of transaction costs for institutional-sized orders and the
primary constraint on strategy capacity. Understanding and modeling market impact is essential
for optimal execution, portfolio construction, and assessing how much capital a strategy can
deploy before returns erode.

---

## 1. Taxonomy of Market Impact

### 1.1 Temporary Impact
- Price displacement that occurs during execution and reverts afterward
- Caused by short-term supply/demand imbalance from your order
- Timescale: reverts within minutes to hours after execution completes
- Think of it as a "liquidity premium" paid for immediacy
- Modeled as a function of trading rate v: g(v)
- Primarily affects execution cost but does NOT permanently move the market

### 1.2 Permanent Impact
- Persistent shift in the equilibrium price due to information revealed by trading
- Reflects the market's inference about your information from observing your order flow
- Does NOT revert: the new price level persists indefinitely
- Modeled as a function of total quantity traded: h(Q)
- Key insight: even uninformed traders cause permanent impact because the market
  cannot distinguish informed from uninformed flow in real time

### 1.3 Realized Impact (Total)
- Total impact = Temporary + Permanent components
- Measured as: (Average Execution Price) - (Pre-Trade Benchmark Price)
- The decomposition is estimated by observing post-trade price reversion
  - At time T+tau after trade completion:
  - Permanent = Price(T+tau) - Price(0)
  - Temporary = Avg Execution Price - Price(T+tau)
  - Total = Permanent + Temporary

### 1.4 Mechanical vs. Informational Impact
- Mechanical: direct book depletion from consuming resting limit orders
  - Immediate and proportional to book depth consumed
- Informational: market makers infer information and adjust quotes
  - Indirect and depends on perceived information content of order flow
  - Higher for concentrated flow, momentum-aligned trades, and unusual size

---

## 2. The Square-Root Law of Market Impact

### 2.1 Empirical Formulation
- The most robust empirical finding in market microstructure:
- Impact = Y * sigma * sqrt(Q / V)
  - Y = constant (asset/market specific, typically 0.1 to 1.0)
  - sigma = daily volatility
  - Q = number of shares traded
  - V = average daily volume (ADV)
  - Q/V = participation rate (fraction of daily volume)

### 2.2 Empirical Evidence
- Validated across: US equities, European equities, futures, FX, options
- Remarkably stable across time periods, asset classes, and market regimes
- Key studies:
  - Almgren et al. (2005): direct estimation on US equities
  - Toth et al. (2011): theory connecting to latent order book
  - Bershova & Rakhlin (2013): meta-analysis confirming universality
- The square-root form implies concavity: doubling order size does NOT double impact
  - This creates economies of scale up to a point

### 2.3 Why Square-Root?
- Several theoretical justifications have been proposed:
  - **Kyle (1985) equilibrium**: informed trader optimally trades proportional to volume;
    market maker's pricing function yields sqrt relationship
  - **Latent order book theory (Toth et al.)**: supply/demand curves near the price are
    approximately flat, and the marginal price impact of consuming Q orders from a
    uniform distribution scales as sqrt(Q)
  - **Dimensional analysis**: impact must scale with sigma (volatility units) and the only
    dimensionless ratio is Q/V; concavity from price-discovery arguments gives sqrt

### 2.4 Limitations
- Breaks down for very small orders (spread dominates) and very large orders (>30% ADV)
- Does not capture time-dependence: same Q traded over 1 hour vs 1 day has different impact
- Assumes stationary volatility and volume; fails during regime changes
- Cross-impact (trading asset A affecting asset B) not captured

---

## 3. Kyle's Lambda

### 3.1 Kyle (1985) Model Setup
- Three types of agents: one informed trader, noise traders, and a market maker
- Informed trader knows the true value V of the asset
- Noise traders submit random orders u ~ N(0, sigma_u^2)
- Market maker observes total order flow y = x + u (cannot distinguish informed from noise)
- Market maker sets price as: P = mu + lambda * y
  - mu = prior expected value
  - lambda = Kyle's lambda (price impact coefficient)

### 3.2 Equilibrium Solution
- lambda = sigma_v / (2 * sigma_u)
  - sigma_v = standard deviation of asset value innovation
  - sigma_u = standard deviation of noise trading volume
- Informed trader's optimal strategy: x = (V - mu) / (2 * lambda)
  - Trades proportionally to information advantage, inversely to impact
- lambda measures market depth: low lambda = deep market, high lambda = shallow
- lambda is inversely related to market liquidity

### 3.3 Practical Estimation of Lambda
- Regress price changes on signed order flow:
  - Delta_P_t = alpha + lambda * OFI_t + epsilon_t
  - OFI_t = order flow imbalance (net signed volume) in period t
- lambda estimated from intraday data (5-min or 15-min bars)
- Typical values: 0.01 to 0.10 (cents per share per signed volume unit)
- Time-varying: estimate with rolling windows; higher around events
- Used to estimate cost of a marginal order: cost per share ~ lambda * order_size

### 3.4 Extensions of Kyle's Model
- Multi-period Kyle: informed trader spreads information across periods
- Multiple informed traders: competition reduces impact per trader
- Kyle-Back: continuous-time version with more realistic dynamics
- Glosten-Milgrom (1985): sequential trade model as complement to Kyle

---

## 4. The Almgren-Chriss Model

### 4.1 Setup
- Liquidate X shares over T periods (or continuous time [0, T])
- Holdings trajectory: x(t), with x(0) = X and x(T) = 0
- Trading rate: v(t) = -dx/dt (positive when selling)
- Price dynamics:
  - S(t) = S(0) + sigma * W(t) - g_permanent(v(t)) * t
  - W(t) = standard Brownian motion
- Execution price includes temporary impact:
  - S_exec(t) = S(t) - g_temporary(v(t))

### 4.2 Cost Functional
- Expected cost: E[C] = integral from 0 to T of [x(t) * gamma * v(t) + eta * v(t)^2] dt
  - First term: permanent impact cost (gamma = permanent impact parameter)
  - Second term: temporary impact cost (eta = temporary impact parameter)
- Variance of cost: Var[C] = sigma^2 * integral from 0 to T of x(t)^2 dt
- Objective: minimize E[C] + lambda * Var[C]
  - lambda = risk aversion parameter (Lagrange multiplier on variance constraint)

### 4.3 Optimal Solution (Linear Impact)
- With linear temporary and permanent impact, the optimal trajectory is:
  - x(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)
  - kappa = sqrt(lambda * sigma^2 / eta)
- Properties:
  - kappa -> 0 (low risk aversion): uniform liquidation (TWAP)
  - kappa -> infinity (high risk aversion): immediate liquidation
  - Intermediate kappa: front-loaded, smooth curve
- The efficient frontier: plot E[C] vs sqrt(Var[C]) to visualize the tradeoff

### 4.4 Calibration
- Permanent impact gamma: regress post-trade price change on total signed volume
  - Typical: gamma ~ 0.05 to 0.5 bps per 1% of ADV
- Temporary impact eta: regress execution shortfall on trading rate
  - Typical: eta ~ 0.1 to 1.0 bps per (shares/minute)
- Volatility sigma: estimate from recent returns (e.g., 20-day realized volatility)
- Risk aversion lambda: set based on portfolio-level risk budget or utility preference
  - Common approach: target a specific probability of exceeding cost budget

---

## 5. Optimal Trade Scheduling

### 5.1 Single-Asset Scheduling
- Given the Almgren-Chriss solution, discretize into N trading intervals
- For each interval k: trade n_k shares according to the optimal trajectory
- Practical adjustments:
  - Clip to maximum participation rate (e.g., 20% of interval volume)
  - Round to lot sizes
  - Skip intervals with abnormally low volume or wide spreads

### 5.2 Multi-Asset (Portfolio) Scheduling
- When rebalancing a portfolio, trades in correlated assets interact
- Cross-impact: buying asset A may move the price of correlated asset B
- Optimal portfolio execution minimizes total cost across all assets simultaneously
- Requires cross-impact matrix estimation (challenging; limited empirical work)
- Simplified approach: schedule each asset independently, then stagger correlated legs

### 5.3 Adaptive Scheduling
- Re-optimize remaining schedule based on realized prices and market conditions
- If price moved favorably: slow down (reduce urgency)
- If price moved adversely: speed up (increase urgency) or pause
- Almgren-Lorenz (2007): adaptive extension with dynamic programming
- Must balance adaptation benefit against execution predictability

---

## 6. Impact on Strategy Capacity

### 6.1 Capacity Definition
- Strategy capacity = maximum AUM before net returns fall below a threshold
- As AUM increases: trade sizes grow, slippage increases, net alpha decreases
- Capacity is reached when marginal slippage equals marginal alpha

### 6.2 Capacity Estimation Framework
- Gross alpha per trade: alpha_gross (in bps)
- Slippage per trade: sigma * C * sqrt(Q / ADV) = sigma * C * sqrt((AUM * w) / (P * ADV))
  - w = portfolio weight, P = price per share
- Net alpha: alpha_net = alpha_gross - slippage
- Capacity: solve for AUM where alpha_net = target (e.g., 0 or minimum acceptable)
- AUM_max = ADV * P * ((alpha_gross - target) / (sigma * C))^2

### 6.3 Capacity by Strategy Type
- **High-frequency (seconds)**: very low per-trade capacity; offset by high turnover
  - Typical: $10M - $100M
- **Statistical arbitrage (days)**: moderate capacity; diversified across many assets
  - Typical: $100M - $1B
- **Momentum/trend (weeks-months)**: higher per-trade tolerance; larger capacity
  - Typical: $1B - $10B
- **Fundamental/value (months-years)**: highest capacity; low turnover
  - Typical: $10B+
- Capacity scales with: number of traded instruments, ADV, and holding period

### 6.4 Capacity-Aware Portfolio Construction
- Penalize positions in illiquid assets: add slippage cost to transaction cost in optimizer
- Tilt toward liquid names when capacity is binding
- Use turnover constraints to limit trading frequency as AUM grows
- Monitor realized slippage vs capacity model; re-estimate quarterly

---

## 7. Common Pitfalls

1. **Ignoring permanent impact in backtest P&L**: your trades move the market; future signals are affected
2. **Using linear impact for large orders**: sqrt is more accurate; linear underestimates large-order cost
3. **Assuming zero cross-impact**: correlated assets co-move; trading one affects the other
4. **Static impact parameters**: impact varies with regime; recalibrate frequently
5. **Conflating temporary and permanent impact**: they have different implications for optimal scheduling
6. **Overestimating strategy capacity**: use conservative (high) impact estimates for capacity analysis
7. **Ignoring decay of alpha signal**: if alpha decays quickly, faster execution is optimal despite higher impact

---

## 8. Implementation Notes

- Build a pre-trade cost estimator: input = (asset, quantity, urgency) -> output = (expected cost, confidence interval)
- Use the estimator in the portfolio optimizer to make capacity-aware allocation decisions
- Post-trade: decompose realized cost into spread, temporary impact, permanent impact, timing
- Maintain a database of impact estimates by asset, time-of-day, and volatility regime
- A/B test execution algorithms: randomize between strategies and compare realized impact
- Monitor for changes in market structure that may invalidate impact model assumptions

---

## 9. Key References

- **Kyle, A.S. (1985)**. "Continuous Auctions and Insider Trading." *Econometrica*, 53(6), 1315-1335. -- Foundational model of strategic trading and price impact.
- **Almgren, R. & Chriss, N. (2001)**. "Optimal Execution of Portfolio Transactions." *Journal of Risk*, 3(2), 5-39. -- Optimal liquidation with temporary and permanent impact.
- **Almgren, R. et al. (2005)**. "Direct Estimation of Equity Market Impact." *Risk*, 18(7), 58-62. -- Empirical calibration of square-root impact model.
- **Gatheral, J. (2010)**. "No-Dynamic-Arbitrage and Market Impact." *Quantitative Finance*, 10(7), 749-759. -- Constraints on impact models from no-arbitrage.
- **Toth, B. et al. (2011)**. "Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets." *Physical Review X*, 1(2), 021006. -- Theoretical basis for square-root law.
- **Bouchaud, J.P. et al. (2004)**. "Fluctuations and Response in Financial Markets." *Quantitative Finance*, 4(2), 176-190. -- Propagator models of price response.
- **Glosten, L. & Milgrom, P. (1985)**. "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders." *Journal of Financial Economics*, 14(1), 71-100. -- Sequential trade model complementing Kyle.
- **Bershova, N. & Rakhlin, D. (2013)**. "The Non-Linear Market Impact of Large Trades." *Quantitative Finance*, 13(11), 1759-1778. -- Meta-analysis confirming sqrt universality.
- **Almgren, R. & Lorenz, J. (2007)**. "Adaptive Arrival Price." *Algorithmic Trading III*, 59-66. -- Adaptive optimal execution.
