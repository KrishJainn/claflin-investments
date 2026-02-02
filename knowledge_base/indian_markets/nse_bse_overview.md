# NSE & BSE: Indian Stock Exchange Overview

## Knowledge Base Category: Indian Market Fundamentals
## Last Updated: 2024-07
## Relevance: Core reference for any India-focused quantitative trading system

---

## 1. Exchange Overview

### National Stock Exchange (NSE)
- **Founded**: 1992, commenced trading 1994
- **Headquarters**: Mumbai, Maharashtra
- **Primary Index**: NIFTY 50
- **Market Share**: ~93% of equity derivatives turnover, ~65% of cash equity turnover
- **Technology**: NEAT+ (National Exchange for Automated Trading) platform
- **Trading Members**: ~2,900+
- **Listed Companies**: ~2,200+
- **ISIN Prefix**: INE (shared across exchanges)

### Bombay Stock Exchange (BSE)
- **Founded**: 1875 (oldest in Asia)
- **Headquarters**: Dalal Street, Mumbai
- **Primary Index**: SENSEX (S&P BSE Sensex, 30 stocks)
- **Technology**: BOLT+ (BSE On-Line Trading) platform
- **Listed Companies**: ~5,500+ (more than NSE due to SME platform)
- **Relevance for Algo Trading**: Lower liquidity than NSE for most instruments; BSE is relevant for SME IPOs and certain exclusive listings

### Practical Note for Quant Systems
- Always prefer NSE for execution due to tighter spreads and deeper order books
- BSE data can be used for cross-validation of prices
- Some stocks trade exclusively on BSE; screen for these edge cases
- Arbitrage opportunities between NSE and BSE exist but are fleeting (sub-second)

---

## 2. Key Indices

### NIFTY 50
- **Composition**: 50 large-cap stocks across 13 sectors
- **Weighting**: Free-float market capitalization weighted
- **Base Year**: November 3, 1995 = 1,000
- **Rebalancing**: Semi-annual (March and September)
- **Formula**: Index Value = (Current Market Cap of Index Constituents / Base Market Cap) x Base Index Value
- **Use in Algo**: Primary benchmark; most liquid F&O underlying

### Bank NIFTY (NIFTY Bank)
- **Composition**: 12 most liquid and large-cap banking stocks
- **Significance**: Highest options trading volume globally (by number of contracts)
- **Weekly Expiry**: Every Wednesday (shifted from Thursday in 2024)
- **Volatility Profile**: Higher beta than NIFTY 50; average daily range 1.5-2.5%
- **Key Constituents**: HDFC Bank, ICICI Bank, SBI, Kotak, Axis Bank, IndusInd Bank

### NIFTY IT
- **Composition**: Major IT services companies (TCS, Infosys, Wipro, HCL Tech, Tech Mahindra)
- **Correlation**: Highly correlated with NASDAQ/USD-INR movements
- **Trading Insight**: Gaps frequently based on US market overnight moves

### Other Important Indices
- **NIFTY Midcap 150**: Mid-cap momentum strategies
- **NIFTY Smallcap 250**: Higher volatility, lower liquidity
- **NIFTY Financial Services**: Broader than Bank NIFTY (includes NBFCs, insurance)
- **India VIX**: Volatility index derived from NIFTY options; mean-reverts around 12-18

---

## 3. Market Hours and Sessions

### Pre-Open Session (9:00 AM - 9:08 AM IST)
- **9:00 - 9:08**: Order entry, modification, cancellation
- **9:08 - 9:12**: Order matching and trade confirmation (random close within this window)
- **9:12 - 9:15**: Buffer period for transition to continuous trading
- Orders are matched at a single equilibrium price determined by demand-supply
- **Algo Consideration**: Pre-open provides indicative opening price; useful for gap analysis

### Continuous Trading Session (9:15 AM - 3:30 PM IST)
- Standard continuous order matching (price-time priority)
- Most liquid period: 9:15-10:30 AM and 2:00-3:30 PM
- Lunch hour (12:00-1:00 PM) typically sees reduced volume
- **Block Deal Window**: 8:45 AM - 9:00 AM (separate window for large orders)

### Closing Session (3:30 PM - 3:40 PM IST)
- Closing price determined via weighted average of last 30 minutes
- Post-close session allows orders at closing price only

### After-Market Orders
- Some brokers allow placing AMO (After Market Orders) from 3:45 PM to 8:57 AM next day
- These are queued and sent to exchange at pre-open

### Key Timezone Notes
- IST = UTC + 5:30 (does not observe daylight saving)
- Overlap with European markets: 1:30 PM - 3:30 PM IST
- US market opens at 7:00 PM IST (summer) / 8:00 PM IST (winter)
- SGX Nifty (now GIFT Nifty) trades ~16 hours/day; use for overnight gap prediction

---

## 4. Settlement Cycle

### T+1 Settlement (Effective January 27, 2023)
- India moved from T+2 to T+1 for all equity segments
- Funds and securities are settled one business day after trade date
- **Impact on Algo Systems**:
  - Reduces capital lockup period
  - Margin recycling is faster
  - FPIs initially faced challenges due to forex settlement timing
  - Early pay-in benefit available (deliver securities before T+1 for margin benefit)

### F&O Settlement
- Daily MTM (Mark-to-Market) settlement on T+1
- Final settlement on expiry day
- Physical delivery for stock F&O since October 2019
- Index options: Cash-settled

---

## 5. Market Segments

### Equity (Cash) Segment
- Delivery-based trading (CNC) and intraday (MIS/NRML)
- No leverage on delivery in new SEBI peak margin regime
- Intraday leverage varies by broker (typically 5x-20x, reducing under SEBI rules)

### Futures & Options (F&O) Segment
- Index futures and options (NIFTY, Bank NIFTY, FinNifty, Midcap Nifty)
- Stock futures and options (~180 eligible stocks)
- Weekly expiries: NIFTY (Thursday), Bank NIFTY (Wednesday), FinNifty (Tuesday), Midcap NIFTY (Monday)
- Monthly expiry: Last Thursday of the month

### Currency Derivatives
- Pairs: USD/INR, EUR/INR, GBP/INR, JPY/INR
- Also cross-currency pairs: EUR/USD, GBP/USD, USD/JPY
- Lot size: USD/INR = 1,000 units (notional ~83,000 INR)
- Lower STT burden compared to equity F&O

### Commodity Derivatives (NSE)
- Gold, silver, crude oil, natural gas
- Relatively new on NSE; MCX remains dominant for commodities
- Useful for cross-asset correlation strategies

---

## 6. Circuit Breakers and Trading Halts

### Index-Level Circuit Breakers (Market-Wide)
| Trigger Level | Before 1:00 PM | 1:00 PM - 2:30 PM | After 2:30 PM |
|---|---|---|---|
| 10% movement | 45-min halt | 15-min halt | No halt |
| 15% movement | 1h45m halt | 45-min halt | Market closes |
| 20% movement | Market closes | Market closes | Market closes |

### Stock-Level Circuit Filters
- Stocks are placed in price bands: 2%, 5%, 10%, or 20%
- F&O stocks have NO circuit limits (but operating range exists)
- Non-F&O stocks can hit upper/lower circuits, blocking further trading in that direction
- **Algo Impact**: Always check circuit limits before placing orders on non-F&O stocks
- **Dynamic Price Bands**: Exchange can revise bands intraday for volatile stocks

---

## 7. SEBI Regulations Relevant to Algo Trading

### Key Regulations
- **Algo Trading Registration**: SEBI mandates exchange approval for all algo strategies
- **Order-to-Trade Ratio**: Maximum ratio monitored; excessive cancellations penalized
- **Co-location**: Available at NSE; SEBI ensures fair access (tick-by-tick data for all)
- **Peak Margin**: Intraday positions also require full margin (snapshot-based monitoring)
- **Upfront Margin Collection**: Brokers must collect margins before trade execution
- **API-based Trading**: SEBI 2023 circular requires all API-based orders to be tagged and approved

### Recent Changes (2023-2024)
- T+1 settlement implemented across all stocks
- Weekly expiry rationalization (one expiry per exchange per day)
- Increased F&O lot sizes for new contract introductions
- True-to-label requirements for algo strategies offered by brokers

---

## 8. FII/DII Participation

### Foreign Institutional Investors (FII/FPI)
- Daily FII buy/sell data available from NSDL by 7:30 PM
- FII flows are a strong predictor of short-term NIFTY direction
- Correlation between FII net buying and NIFTY daily returns: ~0.4-0.6
- **Algo Signal**: Track cumulative FII flows over 5/10/20 day rolling windows
- FII positions in index futures available from NSE daily reports

### Domestic Institutional Investors (DII)
- Primarily mutual funds, insurance companies (LIC), pension funds
- DIIs typically act as contrarian buyers when FIIs sell
- Monthly SIP flows (~20,000 crore/month in 2024) provide consistent buy-side demand

### Participation Mix (Approximate 2024)
- Retail: ~35% of cash market turnover
- FIIs: ~15% of cash market turnover
- DIIs: ~15% of cash market turnover
- Proprietary/Algo: ~35% of cash market turnover

---

## 9. Market Holidays (2024 Reference)

Typical holidays: 14-16 trading holidays per year including:
- Republic Day (Jan 26), Holi, Good Friday, Ambedkar Jayanti
- Maharashtra Day (May 1), Independence Day (Aug 15)
- Gandhi Jayanti (Oct 2), Dussehra, Diwali (Muhurat Trading)
- Guru Nanak Jayanti, Christmas (Dec 25)

### Algo System Considerations
- Always maintain an updated holiday calendar in the trading system
- Muhurat Trading: Special 1-hour session on Diwali evening (symbolic, low volume)
- Saturday trading sessions are rare but can be announced by exchanges
- Holiday data can be fetched from NSE website or maintained as a static config

---

## 10. Practical Pitfalls for Quant Systems

1. **Data Quality**: NSE historical data has corporate action adjustments that may not be uniform across providers
2. **Liquidity Gaps**: Midcap/smallcap stocks can have wide bid-ask spreads (2-5%)
3. **Operator Stocks**: Certain small-caps show manipulated price patterns; filter using delivery % and OI patterns
4. **Exchange Downtime**: NSE has experienced outages (Feb 2021 notable); build BSE failover logic
5. **Corporate Actions**: Bonus, splits, rights issues cause overnight price changes; adjust historical data
6. **Dividend Record Dates**: Ex-dividend dates cause price drops equal to dividend amount; factor into backtests
7. **FPI Limit Breaches**: When FPI holding approaches sector limits, stock behavior changes dramatically
8. **GIFT Nifty Gaps**: Morning gaps in NIFTY are largely explained by overnight GIFT Nifty movement

---

## Quick Reference Card

| Parameter | NSE | BSE |
|---|---|---|
| Primary Index | NIFTY 50 | SENSEX 30 |
| F&O Liquidity | Very High | Low |
| Pre-Open | 9:00-9:08 | 9:00-9:08 |
| Trading Hours | 9:15-15:30 | 9:15-15:30 |
| Settlement | T+1 | T+1 |
| Preferred for Algo | Yes | Backup |
| API Access | Via brokers | Via brokers |
