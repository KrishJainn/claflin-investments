# Factor Models in Quantitative Finance

## Overview

Factor models decompose asset returns into systematic risk exposures and idiosyncratic components.
They form the backbone of modern portfolio construction, risk management, and alpha generation
in quantitative trading systems.

---

## Capital Asset Pricing Model (CAPM)

### Theory

- Developed by Sharpe (1964), Lintner (1965), and Mossin (1966).
- Single-factor model: expected excess return is proportional to market beta.
- Formula: `E[Ri] - Rf = beta_i * (E[Rm] - Rf)`
- Assumes: mean-variance investors, homogeneous expectations, no transaction costs,
  unlimited borrowing/lending at risk-free rate.

### Limitations

- Empirically, the security market line is **too flat**: low-beta stocks earn higher
  risk-adjusted returns than predicted (Black, Jensen, Scholes 1972).
- Single factor insufficient to explain cross-section of returns.
- Roll's critique: the true market portfolio is unobservable, making CAPM untestable.
- Despite limitations, beta remains a critical risk measure in portfolio management.

---

## Fama-French Three-Factor Model (1993)

### Factors

1. **MKT (Market)**: Excess return of the broad market over the risk-free rate.
2. **SMB (Small Minus Big)**: Return spread between small-cap and large-cap stocks.
3. **HML (High Minus Low)**: Return spread between high book-to-market (value) and
   low book-to-market (growth) stocks.

### Formula

```
Ri - Rf = alpha_i + beta_MKT * MKT + beta_SMB * SMB + beta_HML * HML + epsilon_i
```

### Empirical Performance

- Explains approximately 90% of diversified portfolio return variation.
- Reduces average pricing errors compared to CAPM significantly.
- The value premium (HML) has averaged 3-5% annually in US markets historically.
- The size premium (SMB) has weakened since its discovery but persists in some forms.

---

## Carhart Four-Factor Model (1997)

### Added Factor

4. **UMD/WML (Up Minus Down / Winners Minus Losers)**: Momentum factor measuring the
   return spread between recent winners and recent losers (typically 12-1 month formation).

### Key Properties of Momentum

- One of the strongest and most pervasive anomalies across markets and asset classes.
- Average premium of 6-8% annually in US equities historically.
- Subject to severe **momentum crashes**: sharp reversals during market recoveries
  (e.g., March 2009 -- momentum lost 40%+ in a single month).
- Cross-sectional momentum differs from time-series momentum (trend following).
- Partially explained by behavioral biases (underreaction to information, herding).

---

## Fama-French Five-Factor Model (2015)

### Additional Factors

5. **RMW (Robust Minus Weak)**: Profitability factor -- spread between firms with
   robust (high) and weak (low) operating profitability.
6. **CMA (Conservative Minus Aggressive)**: Investment factor -- spread between firms
   with conservative (low) and aggressive (high) asset growth.

### Impact on Value Factor

- The addition of RMW and CMA renders HML (value) **redundant** in the US sample.
- This sparked debate about whether value is a genuine risk factor or a proxy for
  profitability and investment characteristics.
- International evidence is more supportive of an independent value effect.

### Model Comparison

| Model             | Factors | Avg. Pricing Error | Key Weakness              |
|--------------------|---------|--------------------|---------------------------|
| CAPM               | 1       | High               | Misses size, value        |
| Fama-French 3      | 3       | Medium             | Misses momentum           |
| Carhart 4          | 4       | Medium-Low         | Momentum crash risk       |
| Fama-French 5      | 5       | Low                | Misses momentum           |
| FF5 + Momentum     | 6       | Lowest             | High dimensionality       |

---

## Factor Investing and Smart Beta

### The Factor Investing Revolution

- Factor investing bridges active and passive management: systematic exposure to
  return-generating factors at low cost.
- Smart beta ETFs provide rules-based exposure to factors like value, momentum,
  quality, low volatility, and size.
- Global smart beta AUM has grown to trillions of dollars.

### Implementation Approaches

1. **Long-only tilt**: Overweight stocks with desired factor exposure within a benchmark.
2. **Long-short factor portfolios**: Classic academic construction, captures full premium.
3. **Factor timing**: Dynamically adjust factor allocations based on macro conditions
   or factor valuations (controversial -- evidence is mixed).
4. **Multi-factor portfolios**: Combine factors within a single portfolio for
   diversification of factor risk.

---

## Quality Factor

### Definition

Quality encompasses multiple firm characteristics:
- **Profitability**: ROE, ROA, gross profit margin, operating cash flow.
- **Earnings stability**: Low earnings volatility, consistent growth.
- **Financial strength**: Low leverage, strong interest coverage, high Altman Z-score.
- **Payout/dilution**: Shareholder-friendly capital allocation.
- **Accounting quality**: Low accruals, clean earnings.

### Empirical Evidence

- Novy-Marx (2013): Gross profitability is a strong predictor of returns.
- Asness, Frazzini, Pedersen (2019): Quality minus Junk (QMJ) factor earns significant
  risk-adjusted returns globally.
- Quality tends to perform well during market downturns (defensive characteristics).
- Quality and value are positively correlated, creating a "quality value" composite
  that outperforms either factor alone.

---

## Low Volatility Anomaly

### The Puzzle

- CAPM predicts higher risk should earn higher returns.
- Empirically, **low volatility and low beta stocks outperform** on a risk-adjusted basis
  and often on an absolute basis.
- This is one of the most robust anomalies in finance, documented since the 1970s.

### Explanations

1. **Leverage constraints**: Investors who cannot lever up buy high-beta stocks instead,
   driving up their prices and reducing their expected returns.
2. **Lottery preferences**: Behavioral preference for high-volatility, lottery-like payoffs.
3. **Benchmarking**: Fund managers tracking benchmarks prefer high-beta stocks for upside.
4. **Agency problems**: Career risk discourages managers from holding boring, low-vol stocks.

### Implementation Considerations

- Low-vol strategies have significant sector concentration (utilities, staples).
- Can underperform sharply in strong bull markets or momentum-driven rallies.
- Minimum variance portfolios combine low-vol with correlation structure for better
  diversification.
- Works best as a long-term strategic allocation rather than a tactical bet.

---

## Factor Timing

### Approaches

1. **Valuation-based timing**: Buy factors when they are cheap (e.g., value spread is wide).
2. **Macro-based timing**: Allocate to factors based on economic cycle stage.
3. **Momentum-based timing**: Trend-follow factor returns themselves.
4. **Sentiment-based timing**: Use investor sentiment indicators to time factor rotations.

### Evidence and Challenges

- Arnott et al. (2019): Factor valuations have predictive power for future factor returns.
- Macro timing: value tends to outperform in recoveries; momentum in mid-cycle;
  low-vol and quality in late-cycle and downturns.
- In practice, factor timing adds modest value with high uncertainty.
- Transaction costs and turnover from timing can erode theoretical gains.
- Most practitioners advocate **strategic multi-factor allocation** with modest tactical tilts
  rather than aggressive factor rotation.

---

## Factor Models in Indian Markets

### Factor Research Landscape

- India factor research has grown significantly since 2010 with improved data availability.
- NSE and BSE provide factor index series (Nifty Value 20, Nifty Quality 30, Nifty Low
  Volatility 30, Nifty Momentum indices).

### Documented Factor Premiums in India

| Factor         | Evidence Strength | Notes                                      |
|----------------|-------------------|--------------------------------------------|
| Value (B/M)    | Strong            | Persistent since 1990s, stronger in small-caps |
| Momentum       | Very Strong       | Among strongest globally in Indian equities |
| Size           | Moderate          | Significant but affected by liquidity       |
| Quality/Prof.  | Strong            | ROE-based quality factor works well         |
| Low Volatility | Strong            | Nifty Low Vol 30 has outperformed Nifty 50  |
| Investment     | Moderate          | Less studied, early evidence positive       |

### India-Specific Considerations

1. **Promoter holdings**: High promoter ownership affects float-adjusted size factor
   construction. Factor calculations should use free-float market cap.
2. **Sector concentration**: Nifty 50 is concentrated in financials and IT; factor
   portfolios may have extreme sector bets requiring neutralization.
3. **Liquidity constraints**: Small-cap factor strategies face execution challenges
   due to thin order books and impact costs.
4. **Foreign flow sensitivity**: FII flows disproportionately affect large-cap value
   and momentum characteristics.
5. **Tax implications**: Short-term capital gains (holding < 1 year) are taxed at
   20% vs 12.5% for long-term (Budget 2024 rates), affecting factor turnover decisions.
6. **Circuit limits**: Daily price bands (2%, 5%, 10%, 20%) affect momentum and
   volatility factor calculations.

### Building Factor Models for Indian Quant Systems

- Use NSE CM (cash market) data with proper adjustment for corporate actions.
- Source fundamental data from Capitaline, Prowess (CMIE), or Bloomberg.
- Construct factors on a monthly or weekly frequency; daily factor construction
  introduces excessive noise in Indian mid/small-caps.
- Apply winsorization to factor exposures (1st/99th percentile) to handle outliers
  from promoter-related corporate actions.
- Sector-neutralize factors to avoid unintended sector bets, especially given
  India's concentrated index composition.

---

## Key References

- Sharpe, W. (1964). "Capital Asset Prices: A Theory of Market Equilibrium."
- Fama, E. & French, K. (1993). "Common Risk Factors in the Returns on Stocks and Bonds."
- Carhart, M. (1997). "On Persistence in Mutual Fund Performance."
- Fama, E. & French, K. (2015). "A Five-Factor Asset Pricing Model."
- Novy-Marx, R. (2013). "The Other Side of Value: The Gross Profitability Premium."
- Asness, C., Frazzini, A., & Pedersen, L. (2019). "Quality Minus Junk."
- Lo, A. (2004). "The Adaptive Markets Hypothesis."
