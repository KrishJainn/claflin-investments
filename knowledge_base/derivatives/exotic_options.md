# Exotic Options

## Overview

Exotic options are derivatives with payoffs more complex than standard European or American
vanilla options. They arise to meet specific hedging needs, express precise market views, or
provide cost-efficient exposure to particular risk factors. Exotics are a major revenue source
for derivatives desks and demand sophisticated pricing, hedging, and risk management techniques.
Understanding their behavior is essential for quantitative traders in structured products,
volatility trading, and systematic strategies.

---

## Barrier Options

Barrier options are activated (knock-in) or extinguished (knock-out) when the underlying
crosses a specified barrier level during the option's life.

### Types
- **Down-and-Out Call (DOC)**: Standard call that ceases to exist if S falls below barrier B
- **Down-and-In Call (DIC)**: Standard call that activates only if S falls below barrier B
- **Up-and-Out Put (UOP)**: Standard put that ceases to exist if S rises above barrier B
- **Up-and-In Put (UIP)**: Standard put that activates only if S rises above barrier B
- **Double barrier**: Both upper and lower barriers simultaneously

### Key Relationship
For European barriers with no rebate:

**Knock-In + Knock-Out = Vanilla**

This parity allows pricing a knock-in from the vanilla and knock-out (or vice versa).

### Closed-Form Pricing
For continuously monitored single barriers under BSM, closed-form solutions exist
(Merton 1973, Reiner & Rubinstein 1991):

**DOC = C_vanilla - (S/B)^(1-2r/sigma^2) * C_reflected**

Where C_reflected is a BSM call with adjusted parameters reflecting the barrier.

### Practical Considerations
- **Continuous vs. discrete monitoring**: Most traded barriers are monitored daily (at close)
  or at discrete fixing times. Continuous monitoring formulas overestimate knock-out
  probability. Broadie, Glasserman & Kou (1997) provide a continuity correction:

  B_adjusted = B * exp(+/- beta * sigma * sqrt(dt))

  Where beta ~ 0.5826 (the Zeta(1/2)/sqrt(2*pi) constant) and dt is the monitoring interval.

- **Barrier shift / rebate**: Many contracts pay a small rebate upon knock-out to compensate
  the holder. This must be priced as a separate digital component.

- **Hedging challenges**: Barrier options have discontinuous payoffs at the barrier, creating
  large gamma and delta spikes near the barrier. The hedging P&L becomes highly path-dependent.
  Near the barrier, the option behaves like a digital, requiring careful management.

- **Pin risk at barrier**: If the underlying is near the barrier at a fixing time, the
  option can rapidly oscillate between active and inactive states. This creates enormous
  hedging costs and is one of the primary risks on exotic desks.

---

## Asian Options

Asian options have payoffs dependent on the average price of the underlying over some period,
rather than the terminal price alone. They are widely used in commodity and FX markets.

### Types
- **Average price call**: Payoff = max(A - K, 0), where A is the average
- **Average price put**: Payoff = max(K - A, 0)
- **Average strike call**: Payoff = max(S_T - A, 0)
- **Average strike put**: Payoff = max(A - S_T, 0)

### Averaging Methods
- **Arithmetic average**: A = (1/N) * sum(S_ti). Most common in practice but has no
  closed-form BSM solution because the sum of log-normals is not log-normal.
- **Geometric average**: A = (product(S_ti))^(1/N). Has a closed-form solution under BSM
  because the product of log-normals is log-normal. Used as a control variate for arithmetic
  Asian pricing.

### Pricing Approaches
1. **Geometric average (exact)**: Under BSM, the geometric average is log-normal with
   adjusted volatility sigma_G = sigma * sqrt((2N+1)/(6*(N+1))) and adjusted drift.

2. **Moment matching (Turnbull-Wakeman)**: Match the first two moments of the arithmetic
   average to a log-normal distribution. Fast and reasonably accurate.

3. **Monte Carlo**: The standard approach for arithmetic Asians, especially with discrete
   fixings. Use geometric Asian as control variate for variance reduction (factor of 10-50x
   reduction typical).

4. **PDE approach**: Requires an additional state variable (running average), making it
   a 2D PDE. Finite differences work but are computationally expensive.

### Trading Applications
- **Commodity hedging**: Producers and consumers hedge average prices over delivery periods
- **Reduced vol**: Asian options are cheaper than vanilla because averaging reduces effective
  volatility. The reduction factor depends on the averaging window relative to total tenor.
- **Reduced manipulation risk**: The average price is harder to manipulate than spot at expiry

---

## Lookback Options

Lookback options have payoffs depending on the extreme value (max or min) of the underlying
over the option's life.

### Types
- **Floating strike lookback call**: Payoff = S_T - min(S_t) for t in [0, T]
  (buy at the lowest price)
- **Floating strike lookback put**: Payoff = max(S_t) - S_T for t in [0, T]
  (sell at the highest price)
- **Fixed strike lookback call**: Payoff = max(max(S_t) - K, 0)
- **Fixed strike lookback put**: Payoff = max(K - min(S_t), 0)

### Pricing
Under BSM with continuous monitoring, closed-form solutions exist (Goldman, Sosin & Gatto,
1979). For a floating strike lookback call:

The price depends on S, the current running minimum M = min(S_t), sigma, r, and T.
These are significantly more expensive than vanilla options due to the optionality on the
path extremum.

### Practical Considerations
- **Expensive**: Lookbacks are among the most expensive exotic options because they capture
  the full range of price movement. Premiums can be 2-3x vanilla equivalents.
- **Discrete monitoring**: Practical lookbacks are monitored at discrete intervals (daily).
  This reduces the value relative to continuous monitoring. The Broadie-Glasserman-Kou
  correction applies here as well.
- **Hedging**: Lookbacks have high gamma near the running extremum and are difficult to
  hedge. Delta can jump when a new high/low is established.

---

## Digital (Binary) Options

Digital options pay a fixed amount if a condition is met at expiration.

### Types
- **Cash-or-nothing call**: Pays Q if S_T > K, else 0
- **Cash-or-nothing put**: Pays Q if S_T < K, else 0
- **Asset-or-nothing call**: Pays S_T if S_T > K, else 0
- **Asset-or-nothing put**: Pays S_T if S_T < K, else 0

### Pricing Under BSM
**Cash-or-nothing call** = Q * e^(-rT) * N(d2)
**Asset-or-nothing call** = S * N(d1)

Note that a vanilla call = Asset-or-nothing call - K * Cash-or-nothing call (strike K, payout K).

### The Overhedge Problem
Digital options have a payoff discontinuity at the strike, creating infinite gamma at
expiration. In practice, digitals are hedged using tight call spreads (static replication):

**Digital(K) ~ [C(K - eps) - C(K + eps)] / (2 * eps)**

This call spread approximation has finite gamma and is the standard hedging approach.
The choice of eps balances model risk (wider eps) against replication accuracy (narrower eps).

### Skew Sensitivity
Digital prices are highly sensitive to the volatility skew near the strike. BSM digital
prices (using flat vol) can differ substantially from prices computed under a skewed surface.
For accurate pricing, use the model-free formula:

**Digital_call = -dC/dK**

evaluated numerically from the vanilla call price curve, which automatically captures skew.

---

## Quanto Options

Quanto (quantity-adjusted) options have payoffs in one currency based on an underlying
denominated in a different currency, with the exchange rate fixed (quantoed).

### Example
A US investor buys a call on the Nikkei 225 (JPY-denominated) with a quanto feature. The
payoff in USD is:

**Payoff = max(Nikkei_T - K, 0) * fixed_USD_per_index_point**

The investor has no FX exposure; the payoff converts at a predetermined rate.

### Pricing Adjustment
Under BSM, the quanto adjustment modifies the drift of the underlying:

**Quanto drift = r_domestic - r_foreign - rho_SX * sigma_S * sigma_X**

Where:
- r_domestic, r_foreign = domestic and foreign risk-free rates
- rho_SX = correlation between the asset and the exchange rate
- sigma_S = asset volatility
- sigma_X = exchange rate volatility

The additional term (rho_SX * sigma_S * sigma_X) is the quanto adjustment. It arises because
the underlying and FX rate are correlated, creating a covariance drift under the domestic
risk-neutral measure.

### Trading Applications
- Allow foreign exposure without FX risk
- Popular in structured products (auto-callables on foreign indices)
- Correlation rho_SX is a key risk factor; changes in correlation directly affect quanto prices
- The quanto adjustment can be substantial: for rho = -0.3, sigma_S = 20%, sigma_X = 10%,
  the drift adjustment is +0.6% per year

---

## Variance and Volatility Swaps

### Variance Swaps
A variance swap pays the difference between realized variance and a fixed strike:

**Payoff = Notional * (sigma_realized^2 - K_var)**

Where sigma_realized^2 = (252/N) * sum[ln(S_{i+1}/S_i)]^2 (annualized realized variance
from daily log returns).

**Pricing**: The fair strike K_var can be replicated model-free using a portfolio of OTM
options across all strikes (Carr & Madan, Demeterfi et al., 1999):

**K_var = (2/T) * [integral from 0 to F of P(K)/K^2 dK + integral from F to inf of C(K)/K^2 dK]**

This is essentially a weighted sum of OTM put and call prices. The VIX index is computed
using a discretized version of this formula (VIX^2 = K_var for 30-day tenor).

### Volatility Swaps
A volatility swap pays on realized volatility rather than variance:

**Payoff = Notional * (sigma_realized - K_vol)**

Volatility swaps cannot be replicated model-free because sqrt is a concave function of
variance (Jensen's inequality). The fair strike satisfies:

**K_vol ~ sqrt(K_var) - (Var[sigma^2]) / (8 * K_var^(3/2))**

The convexity adjustment (second term) is negative, making vol swap strikes lower than
sqrt(var swap strikes).

### Trading Applications
- **Pure volatility exposure**: No delta, no gamma, no path dependency (for variance swaps).
  Cleanest way to express a view on realized volatility.
- **Variance risk premium**: Historically, implied variance (var swap strike) exceeds
  realized variance by 2-4 vol points annualized. Selling variance swaps captures this
  premium, but with significant tail risk.
- **Dispersion via variance swaps**: Sell index variance, buy single-stock variance.
  Captures the correlation risk premium.

---

## Pricing Approaches for Exotics

### 1. Analytical (Closed-Form)
Available for: European barriers (continuous), geometric Asians, lookbacks (continuous),
digitals under BSM. Fast but limited to simple models and payoffs.

### 2. Monte Carlo Simulation
The most flexible approach. Works for any payoff and any model:
- Generate paths under the risk-neutral measure
- Evaluate the payoff along each path
- Average discounted payoffs

**Variance reduction techniques** are critical for efficiency:
- **Antithetic variates**: Use both W and -W paths. Reduces variance by ~50% for symmetric payoffs.
- **Control variates**: Use a correlated quantity with known expectation (e.g., geometric
  Asian for arithmetic Asian). Can reduce variance by 90%+.
- **Importance sampling**: Shift the sampling distribution to focus on important regions
  (e.g., near barriers). Dramatically improves convergence for rare events.
- **Stratified sampling / Sobol sequences**: Quasi-Monte Carlo with low-discrepancy sequences.
  Convergence rate improves from O(1/sqrt(N)) to nearly O(1/N).

### 3. PDE / Finite Differences
Solve the pricing PDE on a discrete grid. Excellent for 1D and 2D problems:
- 1D: Vanilla, barriers, digital options
- 2D: Stochastic vol models (Heston), quanto options
- 3D+: Becomes computationally prohibitive (curse of dimensionality)

Use Crank-Nicolson for stability and second-order accuracy. For barrier options, align grid
points with the barrier to avoid oscillations.

### 4. Fourier / Transform Methods
For models with known characteristic functions (Heston, VG, CGMY):
- **Carr-Madan FFT**: Price vanilla options across all strikes simultaneously in O(N*log(N))
- **COS method** (Fang & Oosterlee, 2008): Uses Fourier-cosine expansion. Very fast and
  accurate for European-style payoffs.
- Can be extended to some exotics (barriers via Hilbert transform, Asians via convolution)

---

## Common Pitfalls

1. **Model risk**: Exotic prices are far more model-dependent than vanillas. Two models
   calibrated to the same vanilla surface can produce very different exotic prices. Always
   compute model risk reserves by pricing under multiple models.
2. **Barrier monitoring frequency**: Using continuous-barrier formulas for daily-monitored
   barriers leads to systematic mispricing. Always apply discrete monitoring corrections.
3. **Correlation assumptions**: Quantos, basket options, and multi-asset exotics depend
   critically on correlation. Correlation is notoriously difficult to estimate and unstable
   over time. Stress-test correlation assumptions aggressively.
4. **Early exercise for American exotics**: Barrier + American exercise, Asian + American
   exercise, etc. create complex optimal exercise boundaries. These require Longstaff-Schwartz
   Monte Carlo or specialized lattice methods.
5. **Hedging discontinuities**: Digital payoffs, barrier knockouts, and auto-callable
   triggers create payoff discontinuities that are impossible to hedge perfectly. Budget
   for hedging slippage in P&L attribution.
6. **Smile dynamics matter**: Exotic prices depend on how the smile moves, not just the
   current smile level. Local vol, stochastic vol, and jump-diffusion models imply very
   different smile dynamics, leading to different exotic prices even from the same vanilla
   calibration. This is the core challenge of exotic pricing.

---

## Key References

- Taleb, N.N. "Dynamic Hedging: Managing Vanilla and Exotic Options" - Wiley
- Gatheral, J. "The Volatility Surface: A Practitioner's Guide" - Wiley Finance
- Broadie, M., Glasserman, P. & Kou, S.G. (1997). "A Continuity Correction for Discrete
  Barrier Options" - Mathematical Finance
- Goldman, M.B., Sosin, H.B. & Gatto, M.A. (1979). "Path-Dependent Options: Buy at the
  Low, Sell at the High" - Journal of Finance
- Carr, P. & Madan, D. (1999). "Option Valuation Using the Fast Fourier Transform" -
  Journal of Computational Finance
- Demeterfi, K., Derman, E., Kamal, M. & Zou, J. (1999). "A Guide to Volatility and
  Variance Swaps" - Journal of Derivatives
- Longstaff, F.A. & Schwartz, E.S. (2001). "Valuing American Options by Simulation" -
  Review of Financial Studies
- Fang, F. & Oosterlee, C.W. (2008). "A Novel Pricing Method for European Options Based
  on Fourier-Cosine Series Expansions" - SIAM Journal on Scientific Computing
- Reiner, E. & Rubinstein, M. (1991). "Breaking Down the Barriers" - Risk Magazine
