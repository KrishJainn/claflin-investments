# Market Regimes and Regime Detection

## Overview

Market regimes are distinct, persistent states of market behavior characterized by different
return distributions, volatility levels, correlations, and factor dynamics. Identifying the
current regime and adapting strategy allocation accordingly is one of the most impactful
capabilities a quantitative trading system can develop.

---

## Fundamental Regime Types

### Bull Market

- Characterized by sustained upward price trends, typically 20%+ from recent lows.
- Features: rising earnings expectations, expanding P/E multiples, positive sentiment,
  increasing retail participation, declining credit spreads.
- Duration: historically averages 3-5 years in US markets; Indian bull cycles have
  lasted 2-4 years (2003-2008, 2013-2018, 2020-2024).
- Strategy implications: momentum, trend-following, and high-beta strategies thrive;
  short-selling and mean-reversion tend to underperform.

### Bear Market

- Sustained decline of 20%+ from recent highs.
- Features: deteriorating fundamentals, rising volatility, widening credit spreads,
  correlation spikes (diversification fails when most needed), liquidity withdrawal.
- Duration: typically shorter than bull markets (1-2 years), but with sharper moves.
- Strategy implications: short-selling, defensive quality, low-volatility, and
  tail-risk hedging strategies perform best.

### Sideways/Range-Bound Market

- No sustained directional trend; prices oscillate within a range.
- Features: moderate volatility, mixed economic signals, sector rotation dominance.
- Often the most common regime (markets trend only ~30% of the time).
- Strategy implications: mean-reversion strategies excel; trend-following suffers
  from whipsaws; options selling (short volatility) can be profitable.

### Crisis/Dislocation Regime

- Extreme volatility spikes, liquidity evaporation, correlation breakdowns.
- Examples: 2008 GFC, COVID crash (March 2020), 2018 IL&FS crisis in India.
- Duration: typically weeks to months of acute stress.
- Strategy implications: most systematic strategies suffer; survival and capital
  preservation take priority; post-crisis recovery presents the best opportunities.

---

## VIX as a Regime Indicator

### VIX Levels and Market States

| VIX Level | Regime            | Market Characteristics                        |
|-----------|-------------------|-----------------------------------------------|
| < 12      | Complacency       | Very low vol, trending up, crowded carry trades |
| 12-18     | Normal            | Healthy market, moderate vol, mixed signals   |
| 18-25     | Elevated Anxiety  | Increased uncertainty, possible correction    |
| 25-35     | Fear              | Significant stress, bear market conditions    |
| > 35      | Panic             | Crisis mode, extreme dislocations             |

### India VIX

- India VIX is calculated from Nifty 50 options prices (similar methodology to CBOE VIX).
- Typical range: 10-25 in normal markets.
- Spikes above 30 during elections, global crises, and domestic shocks.
- India VIX tends to spike sharply before election results and budget announcements.
- Important caveat: India VIX can be less reliable in illiquid option strike ranges.

### Using VIX in Regime Models

- VIX level classifies current volatility regime.
- VIX term structure (VIX vs. VIX3M or VIX futures curve) indicates
  contango (normal) vs. backwardation (crisis).
- VIX rate of change: sharp VIX increases signal regime transitions.
- Combine VIX with realized volatility: high VIX relative to realized vol suggests
  fear premium (potential opportunity); low VIX relative to realized suggests complacency.

---

## Macro Regime Classification

### Growth-Inflation Quadrant Framework

This framework classifies economic regimes along two dimensions:

```
                    HIGH INFLATION
                         |
     STAGFLATION         |         OVERHEATING
     (Low Growth,        |         (High Growth,
      High Inflation)    |          High Inflation)
                         |
  ------------------------------------------------
                         |
     DEFLATION/          |         GOLDILOCKS
     RECESSION           |         (High Growth,
     (Low Growth,        |          Low Inflation)
      Low Inflation)     |
                         |
                    LOW INFLATION
```

### Asset Class Performance by Quadrant

| Quadrant      | Equities  | Bonds     | Commodities | Gold    | Real Estate |
|---------------|-----------|-----------|-------------|---------|-------------|
| Goldilocks    | Strong +  | Moderate + | Moderate    | Weak    | Strong +    |
| Overheating   | Mixed     | Weak -    | Strong +    | Moderate| Mixed       |
| Stagflation   | Weak -    | Mixed     | Strong +    | Strong +| Weak -      |
| Recession     | Weak -    | Strong +  | Weak -      | Strong +| Weak -      |

### Measuring Growth and Inflation Regimes

- **Growth indicators**: PMI, industrial production, GDP growth, corporate earnings growth,
  credit growth, employment data.
- **Inflation indicators**: CPI, WPI (important in India), core inflation, inflation
  expectations (breakeven rates), commodity prices.
- Use rate of change (acceleration/deceleration) rather than levels for regime classification.
- Leading indicators are more useful than coincident or lagging indicators for trading.

---

## Regime-Switching Models

### Hidden Markov Models (HMM)

- Most widely used statistical framework for regime detection.
- Assumes markets switch between discrete hidden states with different return distributions.
- Parameters: transition probabilities, emission distributions (mean and variance per state).
- Typically 2-3 states work best: low-vol bull, high-vol bull, and bear/crisis.
- Estimation via Baum-Welch (EM) algorithm; state inference via Viterbi algorithm.
- Challenges: regime assignments are probabilistic and backward-looking; real-time
  detection lags actual transitions.

### Threshold Models

- Define regime based on observable indicators crossing thresholds.
- Example: VIX above 25 = high-vol regime; below 15 = low-vol regime.
- Simpler and more interpretable than HMM but less flexible.
- Multiple indicators can be combined via scoring systems.

### Machine Learning Approaches

- **Clustering (K-means, Gaussian Mixture Models)**: Group historical periods by
  return/vol/correlation characteristics.
- **Change-point detection**: Identify structural breaks in time series properties.
- **Random forests/gradient boosting**: Classify regimes using multiple features.
- **Deep learning (LSTM, Transformers)**: Capture complex temporal patterns in regime
  transitions but require large datasets and are prone to overfitting.

### Practical Implementation

```python
# Simplified regime detection framework
def classify_regime(market_data):
    """
    Multi-signal regime classification.
    Returns: 'risk_on', 'risk_off', 'neutral', 'crisis'
    """
    signals = {
        'trend': compute_trend_signal(market_data),      # Price vs. moving averages
        'volatility': compute_vol_regime(market_data),    # Realized vol percentile
        'breadth': compute_breadth(market_data),          # Advance/decline ratios
        'credit': compute_credit_signal(market_data),     # Credit spread changes
        'macro': compute_macro_regime(market_data),       # PMI, CLI indicators
    }
    # Aggregate signals with weights
    regime_score = weighted_average(signals, regime_weights)
    return map_score_to_regime(regime_score)
```

---

## Strategy Performance Across Regimes

### Momentum Strategies

- **Best**: Trending bull markets with moderate volatility.
- **Worst**: Regime transitions, especially bear-to-bull reversals (momentum crash).
- **Adaptation**: Reduce position size or switch off during high-vol regime transitions.

### Mean Reversion Strategies

- **Best**: Sideways, range-bound markets with normal volatility.
- **Worst**: Strong trending markets and crisis periods (mean reversion becomes value trap).
- **Adaptation**: Widen entry/exit bands during high-vol regimes; reduce size during trends.

### Carry/Yield Strategies

- **Best**: Low-vol, goldilocks environments.
- **Worst**: Crisis periods when carry trades unwind violently.
- **Adaptation**: Pre-emptive reduction when VIX term structure inverts.

### Trend Following (CTA-style)

- **Best**: Sustained trends in any direction; crisis periods with clear directional moves.
- **Worst**: Choppy, range-bound markets with frequent reversals.
- **Adaptation**: Shorten lookback periods in fast markets; lengthen in slow markets.

### Volatility Selling

- **Best**: Low-vol, range-bound markets with elevated implied volatility.
- **Worst**: Volatility expansion / crisis (unlimited downside risk on short vol).
- **Adaptation**: Reduce notional and shift to defined-risk structures in elevated VIX.

---

## Indian Market Regime Characteristics

### Unique Regime Drivers in India

1. **Monsoon cycle**: Agricultural output (15% of GDP) depends on monsoon quality;
   deficient monsoons historically trigger rural consumption slowdowns and inflation.
2. **FII/FPI flow regimes**: Foreign portfolio investor flows drive large-cap indices;
   persistent outflows (e.g., 2022 -- Rs 1.2 lakh crore outflow) create distinct regimes.
3. **Election cycles**: State and national elections create uncertainty regimes; Nifty
   has historically rallied post-election result clarity.
4. **RBI monetary policy cycles**: Rate hike cycles vs. rate cut cycles create distinct
   bond and equity regime shifts.
5. **Global dollar/DXY regime**: Rupee depreciation during strong dollar periods affects
   FII returns and creates additional volatility.
6. **Crude oil prices**: India imports ~85% of oil; high crude creates stagflationary
   pressure, affecting overall market regime.

### Indian Market Regime Indicators

- **India VIX**: Primary volatility regime indicator.
- **FII/DII flow data**: Published daily by NSE; persistent flow direction indicates regime.
- **Nifty 50 vs. Nifty 500 breadth**: Narrow leadership (Nifty 50 diverging from broader
  market) signals late-cycle regime.
- **Bank Nifty / Nifty ratio**: Financial sector leadership indicates growth regime.
- **INR/USD rate of change**: Rapid depreciation signals risk-off regime.
- **10-year G-sec yield**: Rising yields signal tightening regime; falling signals easing.
- **Credit growth (RBI data)**: Accelerating credit growth signals expansionary regime.

### Regime-Aware Strategy Design for Indian Markets

1. Monitor multiple regime indicators simultaneously; no single indicator is reliable alone.
2. During election uncertainty regimes, reduce directional exposure and increase hedging.
3. FII outflow regimes disproportionately affect large-caps; rotate to domestic-flow-driven
   mid/small-caps if system capacity allows.
4. Monsoon season (June-September) introduces agricultural/inflation regime uncertainty;
   consider reducing exposure to rural consumption and inflation-sensitive sectors.
5. Budget announcement (typically February 1st) creates a short-lived but intense
   uncertainty regime; options-based strategies can capitalize on elevated premiums.

---

## Key References

- Hamilton, J. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series."
- Ang, A. & Bekaert, G. (2002). "Regime Switches in Interest Rates."
- Kritzman, M. et al. (2012). "Regime Shifts: Implications for Dynamic Strategies."
- Bulla, J. et al. (2011). "Markov-Switching Asset Allocation."
- Nystrup, P. et al. (2017). "Long Memory of Financial Time Series and Hidden Markov Models."
