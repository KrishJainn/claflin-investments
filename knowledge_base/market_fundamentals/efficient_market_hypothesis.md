# Efficient Market Hypothesis (EMH)

## Overview

The Efficient Market Hypothesis is the foundational theory in financial economics that asserts
asset prices fully reflect all available information. Proposed by Eugene Fama in 1970, EMH has
profound implications for quantitative trading, portfolio management, and market regulation.

---

## Three Forms of EMH

### Weak Form Efficiency

- Prices reflect all **past trading information** (historical prices, volume, returns).
- Technical analysis should not produce consistent excess returns.
- Random walk theory: price changes are independent and identically distributed.
- Implication for quant systems: pure momentum or mean-reversion strategies based solely on
  price history should not work if weak form holds strictly.
- Empirical evidence: serial correlation tests, runs tests, and variance ratio tests show
  mixed results -- short-term momentum and long-term reversals exist.

### Semi-Strong Form Efficiency

- Prices reflect all **publicly available information** (financial statements, news, analyst reports).
- Fundamental analysis should not yield consistent alpha.
- Event studies are the primary testing methodology: measuring abnormal returns around
  information releases (earnings, M&A announcements, macro data).
- Evidence: post-earnings announcement drift (PEAD) violates semi-strong form, suggesting
  markets underreact to earnings surprises.

### Strong Form Efficiency

- Prices reflect **all information**, including private/insider information.
- Even insider trading should not produce excess returns.
- Universally rejected: insider trading studies consistently show abnormal returns.
- SEC enforcement actions and SEBI regulations exist precisely because insiders can profit.

---

## Evidence For EMH

1. **Professional fund manager underperformance**: The SPIVA scorecard consistently shows
   that 80-90% of active managers underperform their benchmarks over 10+ year horizons.
2. **Speed of price adjustment**: In liquid markets, prices adjust to news within milliseconds,
   leaving little room for systematic exploitation.
3. **Failure of technical trading rules**: Many simple technical strategies lose profitability
   after transaction costs and once popularized.
4. **Index fund growth**: The rise of passive investing reflects rational acceptance that
   beating the market consistently is extremely difficult.
5. **Random walk characteristics**: Short-horizon returns approximate a random walk in most
   developed equity markets.

## Evidence Against EMH

1. **Anomalies and factor premiums**: Value, momentum, size, and low-volatility effects
   persist across markets and time periods.
2. **Bubbles and crashes**: The dot-com bubble (1999-2000), housing crisis (2007-2008), and
   crypto manias suggest prices can deviate massively from fundamentals.
3. **Excess volatility**: Shiller (1981) demonstrated stock prices are far more volatile than
   justified by subsequent dividend changes.
4. **Post-earnings announcement drift**: Stock prices continue drifting in the direction of
   earnings surprises for 60-90 days, violating semi-strong efficiency.
5. **Limits to arbitrage**: Short-selling constraints, funding liquidity risk, and noise
   trader risk prevent rational arbitrageurs from correcting mispricings.
6. **Calendar effects**: January effect, day-of-week effects, and turn-of-month effects have
   been documented (though many have weakened over time).

---

## The Grossman-Stiglitz Paradox

Grossman and Stiglitz (1980) presented a fundamental logical contradiction in EMH:

- If markets are perfectly efficient, there is no incentive to spend resources gathering
  and analyzing information.
- If no one gathers information, prices cannot reflect information, so markets cannot
  be efficient.
- **Resolution**: Markets must be inefficient enough to compensate informed traders for the
  cost of information acquisition and processing.
- This implies an **equilibrium level of inefficiency** where the marginal cost of
  information gathering equals the marginal benefit of trading on it.
- Implication for quant trading: alpha exists but is a **scarce resource** that gets
  arbitraged away as more capital chases the same signals.
- The paradox suggests the correct question is not "are markets efficient?" but rather
  "how inefficient are markets, and for whom?"

---

## Adaptive Markets Hypothesis (Andrew Lo)

Andrew Lo (2004) proposed the Adaptive Markets Hypothesis as a reconciliation of EMH with
behavioral finance, drawing on evolutionary biology:

### Core Principles

1. **Individuals act in their self-interest** but make mistakes due to cognitive biases.
2. **Learning and adaptation**: Market participants learn from experience and adapt behavior.
3. **Competition drives adaptation**: Successful strategies attract capital, reducing their
   profitability over time.
4. **Natural selection operates on strategies**: Strategies that lose money are abandoned;
   profitable ones proliferate.
5. **Market efficiency is not a fixed state** but varies over time and across markets.

### Implications for Quant Trading

- **Strategy lifecycle**: Every alpha source goes through birth, growth, maturity, and decay.
- **Regime dependence**: Strategy performance depends on the competitive ecology of the market.
- **Innovation premium**: Novel strategies earn excess returns until competitors replicate them.
- **Risk premia are time-varying**: The compensation for bearing risk fluctuates with market
  conditions, investor demographics, and institutional structures.
- **Survival of the fittest**: Quant systems must continuously evolve or face obsolescence.

### Practical Applications

- Monitor strategy crowding indicators to detect alpha decay.
- Maintain a pipeline of new strategy ideas to replace decaying signals.
- Use regime detection to dynamically adjust strategy weights.
- Accept that no single strategy works in all environments.

---

## Market Efficiency in Emerging Markets (India Focus)

### Why Emerging Markets May Be Less Efficient

1. **Information asymmetry**: Corporate disclosure standards are improving but historically
   weaker than in developed markets. SEBI has progressively tightened requirements.
2. **Retail investor dominance**: Indian markets have a high proportion of retail participants
   who may exhibit stronger behavioral biases.
3. **Analyst coverage gaps**: Mid-cap and small-cap stocks in India often have thin or no
   analyst coverage, creating information voids.
4. **Market microstructure**: Circuit breakers (5%, 10%, 20% bands), T+1 settlement (implemented
   in 2023), and periodic illiquidity affect price discovery.
5. **Regulatory evolution**: SEBI regulations are rapidly evolving, creating transitional
   inefficiencies.

### Documented Inefficiencies in Indian Markets

- **Momentum effect**: Strong evidence of momentum profits in Indian equities (Sehgal &
  Balakrishnan, 2002; multiple subsequent studies).
- **Value premium**: Book-to-market and earnings yield factors show persistent premiums.
- **Small-cap premium**: Small-cap stocks outperform on a risk-adjusted basis, partially
  explained by illiquidity.
- **Post-budget drift**: Stock prices show predictable patterns around Union Budget announcements.
- **FII flow predictability**: Foreign Institutional Investor flows exhibit persistence and
  predictably impact prices.
- **Expiry week effects**: Derivatives expiry creates systematic patterns in Nifty/Bank Nifty.

### Improving Efficiency Over Time

- Algorithmic trading now accounts for over 50% of NSE turnover.
- DMA (Direct Market Access) and co-location facilities have reduced latency advantages.
- SEBI's insider trading regulations and surveillance systems are strengthening.
- Growing institutional participation (mutual funds, PMS) is reducing anomaly persistence.

---

## Implications for Quantitative Trading System Design

1. **Accept partial efficiency**: Design systems assuming markets are mostly but not perfectly
   efficient -- seek small, persistent edges rather than large mispricings.
2. **Focus on information processing speed**: Even if information is public, faster and better
   processing of that information can generate alpha.
3. **Exploit structural inefficiencies**: Regulatory constraints, institutional mandates, and
   market microstructure create predictable supply/demand imbalances.
4. **Capacity awareness**: Size alpha estimates relative to strategy capacity -- large alpha
   in small-cap illiquids may not be exploitable at scale.
5. **Continuous research**: Under AMH, yesterday's alpha decays -- maintain an active research
   pipeline to discover new signals.
6. **Transaction cost sensitivity**: Many apparent inefficiencies disappear after realistic
   transaction cost modeling, especially in less liquid segments.
7. **Ensemble approaches**: Combine multiple weak signals rather than relying on any single
   anomaly, as individual anomalies can disappear.

---

## Key References

- Fama, E. (1970). "Efficient Capital Markets: A Review of Theory and Empirical Work."
- Grossman, S. & Stiglitz, J. (1980). "On the Impossibility of Informationally Efficient Markets."
- Lo, A. (2004). "The Adaptive Markets Hypothesis."
- Shiller, R. (1981). "Do Stock Prices Move Too Much to be Justified by Subsequent Changes in Dividends?"
- Lo, A. (2017). "Adaptive Markets: Financial Evolution at the Speed of Thought."
