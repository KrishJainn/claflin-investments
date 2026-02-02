# Indian Derivatives Market: F&O Trading Guide

## Knowledge Base Category: Derivatives & Options Trading in India
## Last Updated: 2024-07
## Relevance: Core reference for options/futures strategies on NSE

---

## 1. NSE F&O Segment Overview

### Market Size and Significance
- NSE is the world's largest derivatives exchange by number of contracts traded
- Average daily turnover: ~300-400 lakh crore notional (2024)
- Index options dominate (~98% of F&O turnover by premium)
- Retail participation in F&O has surged post-2020 (SEBI data: 1 crore+ unique F&O traders)
- SEBI study (2023): ~89% of individual F&O traders made losses over 3 years

### Eligible Instruments
- **Index Futures**: NIFTY 50, Bank NIFTY, Nifty Financial Services, Nifty Midcap Select
- **Index Options**: Same underlyings as above; European-style (exercised only at expiry)
- **Stock Futures**: ~180 eligible stocks
- **Stock Options**: Same ~180 stocks; now European-style (changed from American in 2010)

---

## 2. Lot Sizes

### Index Contracts (2024)
| Underlying | Lot Size | Approximate Notional Value |
|---|---|---|
| NIFTY 50 | 25 | ~5.5 lakh INR |
| Bank NIFTY | 15 | ~7.5 lakh INR |
| Nifty Financial Services | 25 | ~5.0 lakh INR |
| Nifty Midcap Select | 50 | ~6.0 lakh INR |

### Stock Contracts
- Lot sizes vary by stock price and are revised periodically by NSE
- Target notional value per lot: 5-10 lakh INR
- Examples: Reliance (250), TCS (175), HDFC Bank (550), Infosys (400)
- **SEBI 2024 proposal**: Increase minimum lot value to ~15 lakh INR to curb retail speculation

### Algo Consideration
- Always fetch current lot sizes from NSE before deployment
- Lot size changes affect position sizing; historical backtests must account for lot size changes
- Store lot size history as a lookup table: {symbol: [(effective_date, lot_size), ...]}

---

## 3. Expiry Cycles

### Monthly Expiry
- Last Thursday of every month (if holiday, previous trading day)
- Available for: 3 consecutive months (near, next, far)
- Most liquid: Near-month contract

### Weekly Expiry Schedule (Post-November 2024 Rationalization)
| Day | Index |
|---|---|
| Monday | Nifty Midcap Select |
| Tuesday | Nifty Financial Services |
| Wednesday | Bank NIFTY |
| Thursday | NIFTY 50 |

- **SEBI 2024 Change**: Reduced weekly expiries to one per exchange per day (from multiple)
- Weekly options are the most actively traded instruments in Indian markets
- 0-DTE (zero days to expiry) strategies are extremely popular on expiry days

### Long-Dated Options
- Quarterly expiries available for NIFTY (March, June, September, December cycle)
- Up to 7 monthly contracts available
- Long-dated options have wider spreads and lower liquidity

---

## 4. Securities Transaction Tax (STT)

### STT Rates (Effective 2024)
| Transaction Type | STT Rate | Paid By |
|---|---|---|
| Equity Delivery Buy | 0.1% | Buyer |
| Equity Delivery Sell | 0.1% | Seller |
| Equity Intraday Sell | 0.025% | Seller |
| Futures Sell | 0.0125% | Seller |
| Options Sell (Premium) | 0.0625% (was 0.05%) | Seller |
| Options Exercise (ITM) | 0.125% on intrinsic value | Buyer |

### STT Impact on Options Trading
- **Critical Pitfall**: ITM options exercised at expiry attract STT on intrinsic value (0.125%)
- Example: Bought NIFTY 24000 CE at 50, NIFTY expires at 24200. Intrinsic = 200. STT = 0.125% x 200 x 25 = 6.25 INR per lot. Premium paid was 50 x 25 = 1,250. STT on exercise is manageable here.
- **Danger Zone**: Deep ITM options with large intrinsic values. If intrinsic is 500 points, STT = 0.125% x 500 x 25 = 15.63 per lot. Always square off ITM options before expiry rather than letting them expire.
- **Algo Rule**: Auto-square-off ITM options 5-10 minutes before expiry close to avoid exercise STT

### Commodity Transaction Tax (CTT)
- CTT: 0.01% on sell side for non-agricultural commodity derivatives
- Lower than equity STT; relevant for commodity-equity spread strategies

---

## 5. Margin Requirements

### SPAN Margin
- Standard Portfolio Analysis of Risk
- Calculated by NSE using 16 risk scenarios (price move x volatility move)
- Typically 8-15% of contract value for index futures
- Higher for stock futures (15-40% depending on volatility)
- Updated at BOD (beginning of day) and at least 5 intraday snapshots

### Exposure Margin
- Additional margin over SPAN as a safety buffer
- Index: 3% of notional value
- Stocks: 5% of notional or 1.5 standard deviations (whichever is higher)
- **Total Margin = SPAN + Exposure**

### Peak Margin Regime (SEBI 2021)
- Margin checked at random snapshots during the day (at least 4 times)
- If position margin exceeds available margin at any snapshot, penalty is levied
- Penalty: 0.5% per day for shortfall up to 1 lakh; 1% for higher shortfall
- **Algo Impact**: Cannot use intraday leverage beyond available margin even momentarily
- Must maintain margin buffers; recommended 20-30% excess margin over requirement

### Option Buying Margin
- Option buyers pay full premium upfront (no additional margin)
- Option sellers (writers) must maintain SPAN + Exposure margin
- Selling OTM options requires less margin than ATM/ITM options
- Margin changes dynamically with price movement; monitor in real-time

### Algo Implementation
- Fetch real-time margin requirements from broker API before placing orders
- Kite Connect provides margin calculator endpoint
- Pre-trade margin check: Always verify available margin > required margin + buffer
- Formula for approximate index futures margin: Notional x (SPAN% + Exposure%) where SPAN% is around 9-12%

---

## 6. Auto-Square-Off

### Broker-Level Auto-Square-Off
- Brokers square off MIS (intraday) positions at 3:15-3:20 PM
- NRML positions are NOT auto-squared off but may be if margin shortfall occurs
- Some brokers charge penalty for auto-square-off (100-50 INR per order)
- **Algo Best Practice**: Close all intraday positions by 3:10 PM; do not rely on broker auto-square-off

### RMS (Risk Management System) Square-Off
- If MTM loss causes margin shortfall, broker RMS can square off positions anytime
- No advance warning; positions closed at market price
- Keep sufficient buffer to avoid RMS-triggered liquidation

---

## 7. Physical Settlement for Stock Options

### SEBI Mandate (October 2019)
- All stock F&O contracts that expire ITM are physically settled
- If you hold a stock call option that expires ITM, you must take delivery of shares
- If you hold a stock put option that expires ITM, you must deliver shares
- **Margin Implication**: Physical delivery margin (VaR + ELM of full stock value) is levied from E-4 days before expiry
- **Algo Rule**: For stock options, auto-close positions by E-4 (4 days before expiry) to avoid physical delivery obligations

### Exemption
- Index options are cash-settled (no physical delivery)
- This is one reason index options are preferred by algo traders

---

## 8. Popular Strategies in Indian Markets

### Short Straddle on NIFTY/Bank NIFTY
- Sell ATM CE + ATM PE
- Works in range-bound markets (India VIX < 15)
- Average daily range of NIFTY: ~150-200 points; collect premium > expected range
- Risk: Unlimited on breakout; use stop-loss or hedge with OTM options
- **Indian Edge**: High implied volatility relative to realized volatility provides consistent theta decay

### Iron Condor
- Sell OTM CE + OTM PE, Buy further OTM CE + PE as hedge
- Defined risk strategy; popular for weekly expiries
- Typical setup on NIFTY: Sell strikes 200-300 points from ATM, buy hedge 100 points further
- Max profit if NIFTY stays within sold strikes range
- Win rate in India: ~60-70% on weeklies (but losses can be large)

### Expiry Day Strategies (0-DTE)
- Sell far OTM options at 9:20 AM, let theta decay work through the day
- Buy straddle at open if expecting a volatile day (event-based)
- Directional scalping based on first 15-minute range breakout
- **Warning**: SEBI is monitoring 0-DTE trading closely; regulations may tighten

### Calendar Spreads
- Sell near-week option, buy next-week option at same strike
- Profits from term structure (near-week decays faster)
- Lower margin requirement due to hedged nature
- Works well when India VIX is in contango

### Ratio Spreads
- Buy 1 ATM option, sell 2 OTM options
- Net credit or small debit entry
- Risk on the excess short leg; use only with strict stop-loss

---

## 9. India VIX

### Overview
- India VIX measures expected volatility of NIFTY 50 over next 30 days
- Calculated using Black-Scholes model on NIFTY option prices
- **Normal Range**: 10-18 (calm markets)
- **Elevated**: 18-25 (uncertain; pre-election, pre-budget)
- **Crisis**: 25-90 (COVID crash hit 83.6 in March 2020)

### Trading Signals from VIX
- VIX < 12: Sell premium strategies (straddle/strangle) are crowded; breakout risk increases
- VIX 13-18: Optimal zone for option selling strategies
- VIX > 20: Option premiums are expensive; consider buying strategies or staying out
- VIX spike + NIFTY fall: Usually marks capitulation; mean-reversion trade opportunity
- **VIX Futures**: Available on NSE but illiquid; not recommended for algo trading

### VIX-NIFTY Correlation
- Strong negative correlation: approximately -0.75 to -0.85
- When NIFTY falls, VIX rises (fear premium increases)
- Asymmetric: VIX rises more on NIFTY falls than it drops on NIFTY rises

---

## 10. Options Chain Analysis

### Key Metrics to Monitor
- **Open Interest (OI)**: Total outstanding contracts at each strike
- **Change in OI**: Indicates fresh positions being built or unwound
- **Put-Call Ratio (PCR)**: OI of puts / OI of calls
  - PCR > 1.2: Bullish (more puts written = support)
  - PCR < 0.8: Bearish (more calls written = resistance)
  - PCR ~ 1.0: Neutral
- **Implied Volatility Skew**: Compare IV across strikes to identify directional bias

### OI-Based Support and Resistance
- Highest call OI strike = key resistance level
- Highest put OI strike = key support level
- Example: If NIFTY is at 24,000 and 24,500 CE has highest call OI and 23,500 PE has highest put OI, expected range is 23,500-24,500
- **Algo Implementation**: Fetch option chain every 3-5 minutes; track OI shifts at key strikes

### Max Pain Theory
- Max Pain = strike price at which total value of outstanding options (calls + puts) is minimized
- Theory: Market makers drive price toward max pain at expiry to minimize payout
- **Indian Context**: Max pain works ~60% of the time for NIFTY weeklies
- Calculate: For each strike, sum (call OI x max(0, expiry - strike)) + (put OI x max(0, strike - expiry)); strike with minimum sum is max pain
- Best used as a reference level, not a standalone strategy
- More reliable for monthly expiries than weekly

---

## 11. Open Interest Analysis for Algo Systems

### OI Build-Up Interpretation
| Price Movement | OI Change | Interpretation |
|---|---|---|
| Price Up | OI Up | Long Build-Up (Bullish) |
| Price Up | OI Down | Short Covering (Weak Bullish) |
| Price Down | OI Up | Short Build-Up (Bearish) |
| Price Down | OI Down | Long Unwinding (Weak Bearish) |

### FII OI Data
- NSE publishes daily FII OI data in index futures and options
- Track FII long/short ratio in index futures: if > 2.0, very bullish; if < 1.0, bearish
- Available from NSE participant-wise OI reports (published daily by 7 PM)

### Practical OI Signals for Algos
- Sudden OI spike at a specific strike: Large player positioning; treat as support/resistance
- OI unwinding at support/resistance: Level likely to break
- PCR extreme readings (>1.5 or <0.5): Contrarian signal; potential reversal
- Track OI change velocity: Rate of OI change matters more than absolute OI

---

## 12. Common Pitfalls in Indian Derivatives Trading

1. **STT Trap**: Letting ITM options expire instead of squaring off; can wipe out small profits
2. **Margin Spike**: Selling options overnight; margin requirement can spike at next BOD
3. **Liquidity in Stock Options**: Many stock options have wide spreads (5-10%); stick to top 20 by OI
4. **Weekly Expiry Gamma Risk**: Last 2 hours of expiry day can see 500%+ moves in near-ATM options
5. **Physical Delivery**: Forgetting to close stock option positions before E-4
6. **Broker API Limits**: Rate limits on order placement (Zerodha: 10 orders/second)
7. **VIX Expansion on Events**: RBI policy, budget day, election results cause VIX to double overnight
8. **Illiquid Strikes**: Far OTM options (>5% from ATM) may have no buyers when you need to exit
9. **Regulatory Risk**: SEBI frequently changes F&O rules; always monitor circulars
10. **Gap Risk**: NIFTY can gap 2-3% based on global cues; overnight short option positions are exposed
