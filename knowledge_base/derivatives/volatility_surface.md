# Volatility Surface

## Overview

The volatility surface is a three-dimensional representation of implied volatility as a
function of strike price (or moneyness) and time to expiration. It encodes the market's
collective view on the distribution of future returns and is the single most important object
in options trading. A quantitative trader's ability to model, interpret, and trade the
volatility surface directly determines their edge in derivatives markets.

---

## The Volatility Smile and Skew

### Historical Context
Under BSM assumptions, implied volatility should be constant across all strikes and expirations.
Prior to the 1987 crash, equity option IVs were indeed relatively flat. After Black Monday,
the market permanently repriced tail risk, and OTM puts became systematically more expensive
than ATM options. This pattern has persisted for nearly four decades.

### Equity Index Skew
For equity indices (S&P 500, Euro Stoxx 50, etc.), the typical pattern is:
- OTM puts have higher IV than ATM options (downside protection demand)
- OTM calls have lower IV than ATM options (upside is sold via covered calls)
- The relationship is approximately linear in log-moneyness for moderate strikes

A common parameterization of the skew at fixed expiry:

**sigma(K) ~ sigma_ATM + skew * [ln(K/F)] + convexity * [ln(K/F)]^2**

Where F is the forward price. Typical equity index skew values:
- 1-month SPX skew (25-delta put IV minus 25-delta call IV): 5-10 vol points in normal markets
- Skew steepens dramatically during sell-offs (can exceed 20 vol points)
- Short-dated skew is steeper than long-dated skew

### FX and Commodity Smiles
- **FX options**: Exhibit a more symmetric smile (both OTM puts and calls are expensive)
  because currency moves are more symmetric than equity moves. Quoted using risk reversals
  (RR = call IV - put IV) and butterflies (BF = (call IV + put IV)/2 - ATM IV).
- **Commodity options**: Smiles vary by commodity. Energy options often show positive skew
  (OTM calls expensive due to supply disruption risk). Agricultural options vary seasonally.

---

## The Volatility Term Structure

The term structure describes how ATM implied volatility varies across expirations:

- **Normal (contango)**: Short-term IV < Long-term IV. The typical state, reflecting
  volatility mean reversion. Short-term vol is low (calm market) and expected to rise
  toward the long-run mean.
- **Inverted (backwardation)**: Short-term IV > Long-term IV. Occurs during market stress
  when near-term uncertainty is elevated. Expected to normalize as the crisis passes.
- **Humped**: A local maximum at an intermediate expiry, often around a known event
  (earnings, FOMC meeting, election).

Key metrics:
- **VIX term structure**: VIX (30-day) vs. VIX3M (90-day) vs. VIX6M. The VIX/VIX3M ratio
  is a common regime indicator. Ratio > 1 signals stress.
- **Calendar spread IV**: The difference in IV between two expirations at the same strike.
  Drives the pricing of calendar and diagonal spreads.

---

## Local Volatility

### Dupire's Framework (1994)

Local volatility sigma_L(S, t) is the unique deterministic volatility function that is
consistent with all observed European option prices. It answers: "What instantaneous
volatility must the underlying have at each point (S, t) to reproduce the market?"

**Dupire's formula**:

sigma_L^2(K, T) = [dC/dT + (r - q) * K * dC/dK + q * C] / [(1/2) * K^2 * d^2C/dK^2]

Where C = C(K, T) is the market call price surface.

### Strengths
- Perfectly calibrates to the entire observed volatility surface
- Provides a unique, arbitrage-free model
- Useful for pricing path-dependent exotics by Monte Carlo or PDE with sigma_L(S, t)

### Weaknesses
- Assumes volatility is a deterministic function of spot and time (no randomness in vol)
- The "smile dynamics" are wrong: local vol predicts that the smile flattens when spot
  moves, but empirically the smile tends to move with spot ("sticky delta" behavior)
- Extremely sensitive to interpolation of the input surface. Noisy inputs produce unstable
  local vol surfaces. Requires careful smoothing (e.g., SVI parameterization) before
  applying Dupire's formula.

---

## Stochastic Volatility Models

### The Heston Model (1993)

The most widely used stochastic volatility model. Variance follows a CIR (Cox-Ingersoll-Ross)
mean-reverting process:

**dS = mu * S * dt + sqrt(V) * S * dW_1**
**dV = kappa * (theta - V) * dt + xi * sqrt(V) * dW_2**
**Corr(dW_1, dW_2) = rho**

Parameters:
- V: instantaneous variance (V = sigma^2)
- kappa: mean-reversion speed of variance
- theta: long-run variance level
- xi: volatility of variance (vol of vol)
- rho: correlation between spot and variance (typically negative for equities: rho ~ -0.7)

**Key features**:
- Negative rho produces downside skew (spot drops -> vol rises -> OTM puts become expensive)
- xi (vol of vol) controls the curvature/convexity of the smile
- kappa and theta control the term structure
- Semi-analytical pricing via characteristic function and Fourier inversion (Carr-Madan FFT)

**Calibration**:
- Minimize sum of squared IV errors (or price errors weighted by vega) across the surface
- Use Levenberg-Marquardt or differential evolution for the 5-parameter optimization
- Typical calibration takes 1-5 seconds for ~100 option quotes with FFT pricing

**Limitations**:
- Cannot perfectly fit very short-dated smiles (not enough curvature)
- The model is diffusion-based; adding jumps (Bates model = Heston + jumps) improves
  short-dated fit significantly

### The SABR Model (Hagan et al., 2002)

SABR (Stochastic Alpha Beta Rho) is the standard model for interest rate options
(swaptions, caps/floors) and is also used for equity and FX options:

**dF = alpha * F^beta * dW_1**
**d_alpha = nu * alpha * dW_2**
**Corr(dW_1, dW_2) = rho**

Parameters:
- F: forward price
- alpha: initial volatility level
- beta: CEV exponent (controls the backbone; beta=1 is log-normal, beta=0 is normal)
- nu: vol of vol
- rho: correlation between forward and volatility

**Hagan's approximation** gives implied volatility in closed form:

sigma_B(K, F) ~ alpha / [(FK)^((1-beta)/2)] * {z / x(z)} *
  {1 + [(1-beta)^2/24 * alpha^2/((FK)^(1-beta)) + rho*beta*nu*alpha/(4*(FK)^((1-beta)/2))
  + (2-3*rho^2)/24 * nu^2] * T}

Where z = (nu/alpha) * (FK)^((1-beta)/2) * ln(F/K) and x(z) = ln[(sqrt(1-2*rho*z+z^2)+z-rho)/(1-rho)].

**Advantages**:
- Fast closed-form IV approximation (no numerical inversion needed)
- Intuitive parameters with clear market interpretation
- Handles both log-normal and normal volatility conventions
- Excellent for interpolation and extrapolation within an expiry

**Disadvantages**:
- Hagan's formula can produce arbitrage (negative densities) for extreme strikes
- Each expiry is calibrated independently; no built-in term structure dynamics
- The approximation degrades for long-dated options or very high vol of vol

---

## Volatility Surface Construction in Practice

### Step 1: Raw Data Collection
- Collect option quotes (bid, ask, mid, last) across all listed strikes and expirations
- Filter for liquidity: minimum open interest, maximum bid-ask spread, minimum volume
- Convert to implied volatility using a robust IV solver

### Step 2: Parameterization and Smoothing
- **SVI (Stochastic Volatility Inspired)** by Gatheral (2004) is the industry standard
  for parameterizing the smile at a fixed expiry:

  **w(k) = a + b * [rho_svi * (k - m) + sqrt((k - m)^2 + sigma_svi^2)]**

  Where w = sigma^2 * T (total implied variance), k = ln(K/F) (log-moneyness).
  Five parameters: a, b, rho_svi, m, sigma_svi.

- **SSVI (Surface SVI)** by Gatheral and Jacquier (2014) extends SVI to the full surface
  with arbitrage-free constraints across expirations.

### Step 3: Arbitrage Checks
The surface must satisfy:
- **No calendar spread arbitrage**: Total variance w(k, T) must be non-decreasing in T
  for each k. Equivalently, forward-start variance must be non-negative.
- **No butterfly arbitrage**: The implied density must be non-negative everywhere.
  Equivalent to d^2C/dK^2 >= 0 for all K.
- **No vertical spread arbitrage**: Call prices must be decreasing in K; put prices must
  be increasing in K.

### Step 4: Interpolation and Extrapolation
- Between strikes: Use the fitted SVI parameters (smooth by construction)
- Between expirations: Interpolate in total variance space (linear in T ensures no
  calendar arbitrage). Cubic interpolation in sqrt(T) is also common.
- Wing extrapolation: The tails of the smile (deep OTM puts/calls) have limited market
  data. SVI's asymptotic behavior provides reasonable wing extrapolation, but tail risk
  pricing remains model-dependent.

---

## Trading the Volatility Surface

### Skew Trading
- **Risk reversals**: Buy OTM puts + sell OTM calls (long skew) or the reverse (short skew).
  Profit if the skew steepens or flattens relative to entry.
- **Skew is mean-reverting**: After extreme skew moves (e.g., panic-driven), skew tends to
  normalize. Selling extreme skew (via risk reversals) has positive expected value historically.
- **Skew as a signal**: Steep skew can indicate institutional hedging demand or fear. Changes
  in skew often precede directional moves.

### Term Structure Trading
- **Calendar spreads**: Buy longer-dated options + sell shorter-dated options (long calendar).
  Profit from term structure normalization or from vol rising in the back month.
- **VIX futures curve trades**: Analogous to term structure trades but executed via VIX
  futures and options. The persistent contango in VIX futures creates a systematic roll yield.

### Relative Value
- **Cross-asset vol spreads**: Compare IV of correlated assets (e.g., SPY vs. QQQ, or
  stock vs. index). Dislocations create statistical arbitrage opportunities.
- **Dispersion trading**: Sell index vol + buy constituent single-stock vol. Profits from
  the systematic premium of index vol over implied correlation-weighted single-stock vol
  (the correlation risk premium).

---

## Common Pitfalls

1. **Overfitting the surface**: Too many parameters cause instability. The surface should
   be smooth; excessive wiggles indicate overfitting or noisy input data.
2. **Ignoring arbitrage constraints**: An arbitrage-free surface is a hard requirement for
   any pricing engine. Arbitrageable surfaces produce negative densities and nonsensical
   exotic prices.
3. **Stale quotes**: Options on illiquid strikes may have stale prices. Using these without
   filtering contaminates the surface. Always check timestamps and spreads.
4. **Confusing sticky strike vs. sticky delta**: Local vol models imply sticky strike
   dynamics (smile fixed in strike space). Markets behave more like sticky delta (smile
   moves with spot). Stochastic vol models capture this better.
5. **Extrapolation risk**: The wings of the smile are poorly constrained by market data.
   Exotic option prices (barriers, digitals) are highly sensitive to wing behavior. Always
   stress-test wing assumptions.
6. **Event vol handling**: Earnings, dividends, and macro events create discontinuities
   in the term structure. Naive interpolation through events produces misleading results.
   Decompose IV into event vol and base vol.

---

## Key References

- Gatheral, J. "The Volatility Surface: A Practitioner's Guide" - Wiley Finance
- Dupire, B. (1994). "Pricing with a Smile" - Risk Magazine
- Heston, S.L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility" -
  Review of Financial Studies
- Hagan, P.S. et al. (2002). "Managing Smile Risk" - Wilmott Magazine (SABR model)
- Gatheral, J. & Jacquier, A. (2014). "Arbitrage-Free SVI Volatility Surfaces" -
  Quantitative Finance
- Carr, P. & Madan, D. (1999). "Option Valuation Using the Fast Fourier Transform" -
  Journal of Computational Finance
- Bergomi, L. "Stochastic Volatility Modeling" - Chapman & Hall (advanced treatment)
