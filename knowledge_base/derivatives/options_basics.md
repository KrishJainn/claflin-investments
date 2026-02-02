# Options Basics

## Overview

Options are derivative contracts granting the holder the right, but not the obligation, to buy
or sell an underlying asset at a predetermined price (strike) on or before a specified date
(expiration). They are foundational instruments in quantitative finance, serving as building
blocks for hedging, speculation, and complex structured products.

---

## Core Definitions

### Call Option
A call option gives the holder the right to **buy** the underlying asset at the strike price K.

- **Long Call Payoff at Expiry**: max(S_T - K, 0)
- **Short Call Payoff at Expiry**: -max(S_T - K, 0) = min(K - S_T, 0)

A trader buys calls when expecting the underlying to rise. The maximum loss is the premium paid;
the theoretical profit is unlimited.

### Put Option
A put option gives the holder the right to **sell** the underlying asset at the strike price K.

- **Long Put Payoff at Expiry**: max(K - S_T, 0)
- **Short Put Payoff at Expiry**: -max(K - S_T, 0) = min(S_T - K, 0)

A trader buys puts when expecting the underlying to fall or to hedge downside risk. The maximum
profit is K minus the premium (underlying goes to zero); the maximum loss is the premium paid.

---

## Moneyness

Moneyness describes the relationship between the current spot price S and the strike price K.

| State            | Call Condition | Put Condition | Intrinsic Value |
|------------------|---------------|---------------|-----------------|
| In-the-Money     | S > K         | S < K         | Positive        |
| At-the-Money     | S ~ K         | S ~ K         | ~Zero           |
| Out-of-the-Money | S < K         | S > K         | Zero            |

**Log-moneyness** is commonly used in quantitative work: m = ln(S / K). This normalizes moneyness
across different price levels and is the natural x-axis for volatility surfaces.

**Standardized moneyness**: m* = ln(S / K) / (sigma * sqrt(T)), which accounts for both
volatility and time to expiry. This is the preferred metric for comparing options across
different underlyings and tenors.

---

## Intrinsic and Extrinsic Value

The option premium (market price) decomposes into two components:

**Option Price = Intrinsic Value + Extrinsic Value (Time Value)**

- **Intrinsic Value**: The payoff if exercised immediately. For a call: max(S - K, 0).
  For a put: max(K - S, 0). Always non-negative.
- **Extrinsic Value**: The portion of the premium above intrinsic value. Reflects the
  probability that the option will gain additional intrinsic value before expiry. Driven
  primarily by volatility and time remaining.

Key properties of extrinsic value:
- Highest for ATM options (maximum uncertainty about finishing ITM or OTM)
- Decays as expiration approaches (theta decay)
- Increases with implied volatility
- At expiration, extrinsic value is zero; only intrinsic value remains

---

## American vs. European Options

### European Options
- Can only be exercised at expiration
- Easier to price analytically (Black-Scholes applies directly)
- Most index options (e.g., SPX) are European-style

### American Options
- Can be exercised at any time up to and including expiration
- Always worth at least as much as the equivalent European option
- American Call on non-dividend stock: Early exercise is never optimal (proven by Merton, 1973)
- American Put: Early exercise can be optimal when deep ITM (time value of receiving K now
  exceeds optionality value)
- Most single-stock options in the US are American-style
- Pricing requires numerical methods: binomial trees, trinomial trees, or finite difference
  methods. The Barone-Adesi and Whaley (1987) approximation provides a fast closed-form estimate.

### Early Exercise Premium
The difference between American and European option prices:

EEP = C_american - C_european >= 0

For calls on dividend-paying stocks, early exercise may be optimal just before ex-dividend dates
when the dividend exceeds the remaining time value of the option.

---

## Put-Call Parity

The fundamental arbitrage relationship linking European calls and puts with the same strike
and expiry:

**C - P = S - K * e^(-rT)**

Where:
- C = European call price
- P = European put price
- S = Current spot price
- K = Strike price
- r = Risk-free rate (continuously compounded)
- T = Time to expiration (in years)

### Derivation Intuition
A portfolio of long call + short put with the same strike and expiry replicates a forward
contract on the underlying. Both portfolios have identical payoffs at expiry:
- If S_T > K: call pays (S_T - K), put expires worthless -> net payoff = S_T - K
- If S_T < K: call expires worthless, put costs (K - S_T) -> net payoff = S_T - K

Therefore, C - P = PV(Forward) = S - K * e^(-rT).

### Extensions
- **With continuous dividends**: C - P = S * e^(-qT) - K * e^(-rT)
- **With discrete dividends**: C - P = S - PV(Dividends) - K * e^(-rT)
- **American options**: Put-call parity becomes an inequality:
  S - K <= C_am - P_am <= S - K * e^(-rT)

### Algorithmic Trading Applications
- **Parity violations** signal potential arbitrage or data errors. In practice, transaction
  costs, bid-ask spreads, and execution risk consume most apparent arbitrage.
- **Synthetic positions**: Use parity to construct synthetic longs/shorts when direct
  positions are expensive or unavailable. Synthetic long stock = Long Call + Short Put.
- **Implied dividend extraction**: Rearranging parity yields the market-implied dividend,
  useful for dividend forecasting strategies.
- **Conversion/reversal arbitrage**: The classic options market-making strategy. A conversion
  is long stock + long put + short call. A reversal is the opposite.

---

## Payoff Diagram Construction

Payoff diagrams are essential for visualizing option strategies. For algorithmic systems,
generating payoff profiles programmatically enables rapid strategy screening.

For a portfolio of N option legs, the aggregate payoff at expiry is:

**Payoff(S_T) = sum over i of [q_i * payoff_i(S_T)] - Net Premium Paid**

Where q_i is the signed quantity (+1 for long, -1 for short) and payoff_i is the individual
leg payoff function.

Key breakeven calculations:
- Long Call breakeven: K + Premium
- Long Put breakeven: K - Premium
- Bull Call Spread breakeven: K_lower + Net Debit
- Iron Condor breakevens: K_short_put + Net Credit and K_short_call - Net Credit

---

## Common Pitfalls

1. **Ignoring dividends**: Failing to account for dividends causes systematic mispricing,
   especially for deep ITM calls near ex-dates.
2. **Confusing payoff with P&L**: Payoff diagrams show terminal value, not profit. Always
   subtract the initial premium to get P&L.
3. **American vs European mismatch**: Applying Black-Scholes to American puts on
   dividend-paying stocks without adjustment leads to underpricing.
4. **Bid-ask spread neglect**: Theoretical parity violations often vanish after accounting
   for realistic spreads. Always use mid prices or model the spread explicitly.
5. **Pin risk at expiry**: ATM options at expiration create delta uncertainty. Short gamma
   positions near expiry require careful management.
6. **Assignment risk**: American-style short options can be assigned at any time. Short ITM
   calls before ex-dividend dates carry significant assignment risk.

---

## Implementation Notes for Algorithmic Trading

- **Option chain data**: Ensure your data feed provides Greeks, implied vol, and open interest
  alongside price. OPRA feed in the US provides ~1.5 million quotes per second.
- **Strategy screening**: Parameterize strategies by risk/reward metrics (max loss, max gain,
  breakeven, probability of profit) for automated scanning.
- **Execution**: Options markets are less liquid than equity markets. Use limit orders, target
  natural prices (at or between bid/ask), and consider legging into multi-leg strategies.
- **Settlement**: Know whether the option is PM-settled (standard) or AM-settled (many index
  options). AM settlement uses the opening price on expiration morning.

---

## Key References

- Hull, J.C. "Options, Futures, and Other Derivatives" - The standard reference textbook
- Merton, R.C. (1973). "Theory of Rational Option Pricing" - Bell Journal of Economics
- Stoll, H.R. (1969). "The Relationship Between Put and Call Option Prices" - Journal of Finance
- Barone-Adesi, G. & Whaley, R.E. (1987). "Efficient Analytic Approximation of American
  Option Values" - Journal of Finance
- Cox, J.C., Ross, S.A., & Rubinstein, M. (1979). "Option Pricing: A Simplified Approach" -
  Journal of Financial Economics (Binomial model)
