# Paper Trading Strategy Setup

## Generated: January 27, 2026
## Test Period: 1-Year Backtest Results

---

## EXECUTIVE SUMMARY

After extensive analysis and evolution of 100+ generations across multiple runs, here are the **THREE RECOMMENDED STRATEGIES** for paper trading:

| Strategy | 1-Year Profit | Win Rate | Profit Factor | Avg Winner | Avg Loser | Risk Level |
|----------|---------------|----------|---------------|------------|-----------|------------|
| **SAFE** | $32,289 (32%) | 25.6% | 2.29 | $2,865 | -$431 | LOW |
| **BALANCED** | $15,299 (15%) | 18.2% | 1.57 | $3,012 | -$427 | MEDIUM |
| **AGGRESSIVE** | $350 (0.4%) | 19.5% | 1.01 | $1,686 | -$402 | HIGH |

**RECOMMENDATION: Use the SAFE strategy for paper trading next week.**

---

## STRATEGY 1: SAFE (Recommended)

### Performance Metrics
- **Annual Return**: 32.3%
- **Profit Factor**: 2.29 (for every $1 lost, you make $2.29)
- **Win/Loss Ratio**: 6.6:1 (avg winner is 6.6x larger than avg loser)
- **Total Trades**: 78 over 1 year (~1.5 trades/week)

### Key Indicators (Weighted)
```
BULLISH SIGNALS (BUY):
  TEMA_20:     +0.9034  (Triple EMA - Strong trend confirmation)
  KST:         +0.8221  (Know Sure Thing - Momentum)
  CMF_20:      +0.7471  (Chaikin Money Flow - Volume)
  HMA_9:       +0.6080  (Hull MA - Fast trend)
  ZSCORE_50:   +0.5693  (Statistical deviation)
  ZSCORE_20:   +0.4706  (Short-term deviation)
  VORTEX_14:   +0.3862  (Trend direction)
  BBANDS_20_2: +0.3865  (Bollinger Bands)
  SMA_100:     +0.3522  (Long-term trend)
  VWMA_10:     +0.2957  (Volume Weighted MA)

BEARISH SIGNALS (SELL/AVOID):
  ADX_20:          -1.0000  (Trend strength - avoid weak trends)
  KC_20_1.5:       -0.7425  (Keltner Channels)
  SUPERTREND_7_3:  -0.5066  (SuperTrend)
  PVI:             -0.4915  (Positive Volume Index)
  TRUERANGE:       -0.4556  (Volatility filter)
  ADX_14:          -0.4027  (Trend strength)
  LINREG_SLOPE_14: -0.3821  (Linear regression)
  DONCHIAN_20:     -0.3375  (Donchian Channels)
  T3_5:            -0.3051  (T3 Moving Average)
```

### Trading Rules
1. **Entry Signal**: When Super Indicator > 0.70 (strong bullish)
2. **Exit Signal**: When Super Indicator < 0.30 (losing momentum)
3. **Stop Loss**: 2x ATR from entry
4. **Position Size**: 2% risk per trade, max 20% per position

### Best Performing Stocks (1-Year)
| Stock | Trade | P&L | Return |
|-------|-------|-----|--------|
| M&M.NS | LONG | $7,075 | +40.0% |
| LT.NS | LONG | $5,446 | +29.7% |
| INDUSINDBK.NS | LONG | $5,404 | +28.1% |
| RELIANCE.NS | LONG | $4,236 | +22.3% |
| JSWSTEEL.NS | LONG | $3,592 | +18.3% |

---

## STRATEGY 2: BALANCED

### Performance Metrics
- **Annual Return**: 15.3%
- **Profit Factor**: 1.57
- **Win/Loss Ratio**: 7:1 (avg winner 7x larger than avg loser)
- **Total Trades**: 77 over 1 year

### Key Indicators
```
BULLISH:
  KST:         +0.9153  (Momentum)
  WILLR_28:    +0.8833  (Williams %R)
  BBANDS_20_2: +0.8088  (Bollinger Bands)
  DONCHIAN_20: +0.6942  (Breakout channels)
  NATR_20:     +0.6832  (Normalized ATR)
  PIVOTS:      +0.6279  (Pivot points)
  KC_20_1.5:   +0.6187  (Keltner Channels)
  STOCH_21_5:  +0.6101  (Stochastic)
  ZSCORE_20:   +0.5907  (Z-Score)
  CMF_20:      +0.5052  (Money Flow)

BEARISH:
  AROON_25:        -0.9460  (Aroon indicator)
  HMA_9:           -0.8518  (Hull MA)
  SUPERTREND_10_2: -0.8235  (SuperTrend)
  DEMA_10:         -0.6213  (Double EMA)
  ADX_14:          -0.5851  (Trend strength)
  KC_20_2:         -0.5488  (Keltner Channels)
  AO_5_34:         -0.5268  (Awesome Oscillator)
```

---

## STRATEGY 3: AGGRESSIVE (Higher Risk)

### Performance Metrics
- **Annual Return**: 0.4% (poor 1-year, but 3-year was $221k)
- **Profit Factor**: 1.01
- **Best for**: 3-year holding periods

### Key Indicators
```
BULLISH:
  PSAR:        +1.0000  (Parabolic SAR)
  UO_7_14_28:  +1.0000  (Ultimate Oscillator)
  NVI:         +0.9591  (Negative Volume Index)
  AROON_25:    +0.8784  (Aroon)
  NATR_20:     +0.7236  (Normalized ATR)
  WILLR_14:    +0.6319  (Williams %R)
  AO_5_34:     +0.6209  (Awesome Oscillator)
  ZSCORE_20:   +0.6194  (Z-Score)

BEARISH:
  ADX_20:          -0.9140
  COPPOCK:         -0.8994
  LINREG_SLOPE_14: -0.8898
  T3_10:           -0.8648
  SMA_100:         -0.7469
  MFI_14:          -0.7001
  PIVOTS:          -0.6529
```

---

## INDICATOR GUIDE

### Most Important Indicators Across All Strategies

1. **ZSCORE_20** (+0.56 avg) - Statistical measure of price deviation
   - Buy when price is below normal (oversold)
   - Sell when price is above normal (overbought)

2. **KST** (+0.87 avg) - Know Sure Thing momentum oscillator
   - Combines 4 different rate-of-change periods
   - Strong trend confirmation signal

3. **CMF_20** (+0.63 avg) - Chaikin Money Flow
   - Measures buying/selling pressure based on volume
   - Positive = accumulation, Negative = distribution

4. **WILLR_28** (+0.70 avg) - Williams %R
   - Momentum indicator (0 to -100 range)
   - Below -80 = oversold, Above -20 = overbought

5. **ADX_14** (-0.49 avg) - Average Directional Index
   - Measures trend strength (not direction)
   - AVOID when ADX is low (weak/ranging market)

6. **SUPERTREND** (-0.67 avg) - Trend following indicator
   - Used as a FILTER - avoid trades against SuperTrend

---

## HOW TO USE FOR PAPER TRADING

### Step 1: Check Super Indicator Value
Calculate the weighted sum of all normalized indicators:
```
SI = Σ (indicator_value × weight)
```

### Step 2: Generate Signals
- **BUY** when SI > 0.70 and currently FLAT
- **SELL** when SI < 0.30 and currently LONG
- **HOLD** otherwise

### Step 3: Position Management
- Risk 2% of capital per trade
- Set stop loss at 2x ATR
- Target profit at 3x ATR (1.5:1 R:R minimum)

### Step 4: Stock Selection
Focus on these top performers:
1. M&M.NS (Mahindra & Mahindra)
2. LT.NS (Larsen & Toubro)
3. INDUSINDBK.NS (IndusInd Bank)
4. RELIANCE.NS
5. JSWSTEEL.NS (JSW Steel)
6. HCLTECH.NS
7. NESTLEIND.NS
8. ASIANPAINT.NS
9. MARUTI.NS
10. TECHM.NS

---

## WEIGHTS JSON (For Automated Trading)

### SAFE Strategy Weights
```json
{
  "KST": 0.8221,
  "TEMA_20": 0.9034,
  "CMF_20": 0.7471,
  "HMA_9": 0.6080,
  "ZSCORE_50": 0.5693,
  "ZSCORE_20": 0.4706,
  "VORTEX_14": 0.3862,
  "BBANDS_20_2": 0.3865,
  "SMA_100": 0.3522,
  "VWMA_10": 0.2957,
  "STOCH_21_5": 0.2018,
  "MOM_10": 0.2177,
  "ADX_20": -1.0,
  "KC_20_1.5": -0.7425,
  "SUPERTREND_7_3": -0.5066,
  "PVI": -0.4915,
  "TRUERANGE": -0.4556,
  "ADX_14": -0.4027,
  "LINREG_SLOPE_14": -0.3821,
  "DONCHIAN_20": -0.3375,
  "T3_5": -0.3051
}
```

---

## EXPECTED RESULTS

Based on backtesting:

| Metric | SAFE Strategy |
|--------|---------------|
| Weekly Trades | 1-2 |
| Monthly Profit | ~$2,700 |
| Max Drawdown | ~5% |
| Best Month | +$8,000 |
| Worst Month | -$2,000 |
| Annual Return | 32% |

**DISCLAIMER**: Past performance does not guarantee future results. Paper trading first is strongly recommended before using real capital.

---

## FILES GENERATED

1. `final_strategy_report.json` - Full detailed metrics
2. `evolved_1year_strategies.json` - Evolution results
3. `backtest_results/` - Trade-by-trade CSV files
4. `strategy_weights.json` - All strategy weights

---

*Generated by Trading Evolution Engine v3.0*
