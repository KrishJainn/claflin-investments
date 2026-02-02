# Quantitative Strategies for Indian Markets

## Knowledge Base Category: Alpha Generation, Strategy Design & Indian Market Edge
## Last Updated: 2024-07
## Relevance: Actionable strategies adapted for Indian market microstructure and behavior

---

## 1. NIFTY Momentum with Sector Rotation

### Core Concept
- Indian markets exhibit strong momentum at the sector level
- Sectors rotate in 3-6 month cycles driven by policy, global flows, and earnings
- Strategy: Rank sectors by trailing 3-month returns, go long top 2-3 sectors, short/avoid bottom 2-3

### Implementation
- **Universe**: NIFTY Sectoral Indices (Bank, IT, Pharma, Auto, Metal, Realty, Energy, FMCG, Media, PSE)
- **Signal**: 63-day (3 month) rolling return, z-score normalized
- **Rebalance**: Monthly (first trading day of month)
- **Execution**: Use sector ETFs (Bank BeES, IT BeES, Pharma BeES) or top 3 stocks per sector by weight
- **Filter**: Exclude sectors with negative 12-month momentum (avoid catching falling knives)

### Indian-Specific Edge
- Policy-driven rotation: Budget announcements favor specific sectors (infra, defense, solar)
- RBI rate cycle: Banking outperforms in easing cycles; IT outperforms in tightening (weaker INR benefits)
- Monsoon impact: FMCG and rural-focused sectors benefit from good monsoon years
- **Backtest Performance (2010-2024)**: CAGR ~18-22%, Sharpe ~0.9-1.2, Max Drawdown ~20-25%

### Risk Management
- Cap sector allocation at 30% of portfolio
- Use NIFTY futures as a hedge during market-wide drawdowns (NIFTY down > 5% in a month)
- Stop-loss: Exit sector if 1-month drawdown exceeds 10%

---

## 2. Bank NIFTY Option Selling Strategies

### Strategy: Short Strangle with Dynamic Adjustment
- Sell Bank NIFTY OTM Call and OTM Put weekly options
- Strike selection: 1.5-2 standard deviations from ATM (based on implied volatility)
- **Entry**: Monday/Tuesday (sell Thursday/Wednesday expiry)
- **Exit**: Expiry or if either leg reaches 2x premium collected (stop-loss)

### Strike Selection Formula
- Upper Strike = ATM + (Bank NIFTY Price x IV x sqrt(DTE/365) x 1.5)
- Lower Strike = ATM - (Bank NIFTY Price x IV x sqrt(DTE/365) x 1.5)
- Round to nearest 100 (Bank NIFTY strike interval)
- Example: Bank NIFTY at 50,000, IV = 15%, DTE = 3 days
  - Range = 50,000 x 0.15 x sqrt(3/365) x 1.5 = 50,000 x 0.15 x 0.0906 x 1.5 = 1,019 points
  - Sell 51,000 CE and 49,000 PE

### Indian-Specific Considerations
- Bank NIFTY is heavily influenced by SBI, HDFC Bank, ICICI Bank results (quarterly)
- Avoid selling strangles during bank result season (Oct, Jan, Apr, Jul)
- Wednesday expiry (Bank NIFTY weekly) sees peak gamma risk from 2 PM onwards
- India VIX > 18: Widen strikes or reduce position size
- India VIX < 12: Premiums are low; strategy may not cover costs

### Performance Characteristics
- Win rate: ~65-75% of weeks
- Average weekly return: 1-2% on margin deployed
- Tail risk: Can lose 5-15% of margin in a single week during black swan events
- **Key Metric**: Monitor premium collected vs average Bank NIFTY weekly range

---

## 3. Expiry Day Strategies (0-DTE)

### Strategy 1: 9:20 AM Straddle Sell
- At 9:20 AM on expiry day, sell ATM straddle on NIFTY/Bank NIFTY
- Theta decay is maximum on expiry day; ATM options lose 60-80% of value
- **Entry**: 9:20 AM (after initial volatility of first 5 minutes settles)
- **Exit**: 3:15 PM or if combined premium increases by 30% from entry (stop-loss)
- **Position Sizing**: Use 20-25% of capital (remaining as margin buffer)

### Strategy 2: First 15-Minute Range Breakout
- Calculate high and low of NIFTY/Bank NIFTY during 9:15-9:30 AM
- Go long if price breaks above range + buffer (0.1% of price)
- Go short if price breaks below range - buffer
- **Stop-loss**: Opposite end of the range
- **Target**: 1.5x the range width
- **Indian Edge**: Opening 15 minutes often set the tone for expiry day direction

### Strategy 3: Max Pain Convergence
- Calculate max pain from options chain at market open on expiry day
- If NIFTY is > 100 points from max pain, initiate mean-reversion trade toward max pain
- Use options spread (bull/bear spread depending on direction)
- **Time Cutoff**: Enter before 11 AM; if price has not moved toward max pain by 1 PM, exit
- **Win Rate**: ~55-60% historically; works better when VIX is low

### Expiry Day Risk Factors
- Gamma explosion: Near-ATM options can move 500-1000% in minutes near expiry
- Illiquidity in last 15 minutes: Spreads widen dramatically
- Pin risk: NIFTY often pins at round strikes (24,000, 24,500) near expiry
- Exchange can have order congestion on expiry days; factor in execution delays

---

## 4. Volatility Strategies Around RBI Monetary Policy

### Event Calendar
- RBI MPC (Monetary Policy Committee) meets 6 times per year (bi-monthly)
- Dates announced in advance; typically Feb, Apr, Jun, Aug, Oct, Dec
- Decision announced at 10:00 AM on the last day of the meeting

### Pre-Event Strategy: Long Straddle
- Buy ATM NIFTY straddle 2-3 days before MPC decision
- India VIX typically rises 2-4 points ahead of RBI decision
- Exit after the announcement (sell within first 30 minutes post-announcement)
- **Edge**: IV expansion before event pays for theta decay
- **Risk**: If VIX is already elevated (>18), the pre-event IV expansion may be priced in

### Post-Event Strategy: IV Crush Sell
- Immediately after RBI announcement, sell ATM straddle
- VIX typically drops 15-25% post-announcement regardless of decision
- **Entry**: Within 10 minutes of announcement (10:00-10:10 AM)
- **Exit**: End of day or next day
- **Edge**: Post-event IV crush is one of the most reliable patterns in Indian options

### Rate Decision Impact Pattern
- **Rate Cut**: Bank NIFTY up 1-3%, NIFTY up 0.5-1.5%
- **Rate Hold (expected)**: Minor move; VIX crush still occurs
- **Rate Hike**: Bank NIFTY down 1-2%, but often V-shaped recovery
- **Surprise Decision**: Can move NIFTY 2-3% intraday; avoid being short options pre-event

---

## 5. Budget Day Trading Strategy

### Context
- Union Budget presented on February 1 (typically)
- Most volatile single day in Indian markets annually
- NIFTY can move 3-5% intraday on budget day
- VIX peaks on budget day (often > 20)

### Strategy: Post-Budget Momentum
- Wait for budget speech to conclude (typically by 12:30-1:00 PM)
- Identify initial market reaction direction
- Enter in direction of initial move at 1:00 PM (momentum trade)
- **Historical Pattern**: Budget day's initial direction persists 60-65% of the time
- Use NIFTY futures or ATM options for execution

### Pre-Budget Strategy
- Buy NIFTY straddle 3-5 days before budget
- VIX expansion of 3-6 points provides IV gain
- Exit straddle on morning of budget day (before speech begins) to capture IV premium
- **Historical Returns**: Pre-budget long straddle has positive expectancy in 7/10 years

### Sectors to Watch on Budget Day
- Infra/Construction: Government capex allocation
- Defense: Defense budget and indigenization push
- Agriculture: MSP announcements, fertilizer subsidies
- Real Estate: Tax deductions on home loans, stamp duty
- Auto: EV incentives, emission norms
- Track sector-specific reactions for post-budget swing trades (2-5 day holding)

---

## 6. FII Flow-Based Momentum

### Signal Construction
- Download daily FII net buy/sell data from NSDL (available by 7:30 PM each day)
- Calculate rolling 5-day cumulative FII net flow
- Calculate rolling 20-day cumulative FII net flow

### Trading Rules
- **Bullish Signal**: 5-day FII net > 0 AND 20-day FII net > 0 (sustained buying)
- **Bearish Signal**: 5-day FII net < 0 AND 20-day FII net < 0 (sustained selling)
- **Entry**: Buy NIFTY at open next day when bullish signal triggers
- **Exit**: When signal flips to bearish or neutral (5-day crosses zero)

### Enhancement: FII Index Futures Position
- NSE publishes FII long/short positions in index futures daily
- FII Long-Short Ratio = FII Long Contracts / FII Short Contracts
- Bullish: Ratio > 1.5
- Bearish: Ratio < 0.8
- Neutral: 0.8-1.5
- Combine with cash flow signal for higher conviction

### Historical Performance
- FII flow momentum signal: CAGR ~14-16% on NIFTY with ~60% win rate
- Maximum drawdown: ~15-20% (during sustained FII selling periods like 2022)
- Works best in trending markets; whipsaws in sideways markets
- **Lag Issue**: FII data is available after market close; signal is for next day

---

## 7. Adani/Ambani Stock Correlations

### Conglomerate Basket Trading
- Adani Group: Adani Enterprises, Adani Ports, Adani Green, Adani Power, Adani Total Gas, ACC, Ambuja Cements
- Reliance Group: Reliance Industries (dominant; proxy for the group)
- Intra-group correlation is high (0.6-0.9 for Adani stocks)

### Mean-Reversion within Adani Basket
- When one Adani stock diverges significantly from the basket (z-score > 2), mean-revert
- Calculate daily returns for each stock and the basket average
- If Stock_Return - Basket_Avg_Return > 2 x Basket_StdDev: Short the stock, long the basket
- Holding period: 3-5 days
- **Caveat**: Adani stocks are subject to event risk (regulatory, governance, Hindenburg-type events)
- Position size conservatively (max 5% of capital per stock)

### Reliance Industries as Market Proxy
- RIL has ~10% weight in NIFTY 50; its movement has outsized index impact
- RIL earnings, Jio announcements, and refinery margins are key catalysts
- Pair trade: Long/Short RIL vs NIFTY futures when relative value diverges
- Track RIL relative strength vs NIFTY on 20-day rolling basis

---

## 8. Small-Cap Momentum in India

### Universe
- NIFTY Smallcap 250 constituents
- Filter: Daily turnover > 5 crore INR (liquidity filter)
- Exclude: Stocks under GSM/ASM surveillance, stocks at circuit limits

### Momentum Signal
- 12-month price momentum minus most recent 1 month (12_1 momentum)
- This "skip-month" momentum avoids short-term reversal
- Rank all stocks by 12_1 momentum
- Go long top decile (25 stocks), rebalance monthly

### Indian Small-Cap Specific Factors
- Small-cap outperformance in India is more pronounced than in developed markets
- CAGR of NIFTY Smallcap 250 vs NIFTY 50: excess return of ~3-5% per year (with higher vol)
- **Risk**: Small-cap drawdowns in India can be 50-70% (2018, 2020)
- **Liquidity Risk**: Impact cost in small-caps is 1-3% for moderate position sizes
- **Operator Activity**: Some small-caps show artificial price patterns; filter using delivery % > 30% as quality check

### Performance (Backtest 2012-2024)
- Top decile momentum CAGR: ~25-30%
- Sharpe: ~0.8-1.0
- Max Drawdown: ~45-55%
- Turnover: ~30-40% per month
- **Transaction Cost Warning**: Impact cost and STT significantly reduce returns in live trading

---

## 9. Indian Market Anomalies

### January Effect (Weak in India)
- Unlike US markets, January effect is weak in India
- Indian financial year ends March 31; any tax-related selling/buying happens in March
- **March Effect**: Tax-loss selling creates opportunities in beaten-down stocks in March
- **April Effect**: Fresh financial year allocations create buying pressure in April

### Muhurat Trading (Diwali)
- Special 1-hour trading session on Diwali evening
- Historically bullish: NIFTY is positive in ~75% of Muhurat sessions
- Low volume; ceremonial buying by institutions and retail
- Not actionable for algo systems (too short, too illiquid)

### Monday Effect
- NIFTY shows slight negative bias on Mondays (weekend gap risk)
- Friday shows slight positive bias (institutional buying before weekend)
- Effect is weak and not consistently tradeable after costs

### Budget Week Anomaly
- Week before budget: NIFTY tends to be range-bound (VIX rises, direction uncertain)
- Week after budget: Strong directional move (momentum signal)
- Tradeable: Go with budget week momentum for 1-2 weeks post-budget

### F&O Expiry Week Patterns
- Week of monthly F&O expiry: NIFTY tends to converge toward max pain
- First week after expiry: Often sees strong directional moves (new positions being built)
- Algo signal: Track OI build-up direction in first 3 days of new monthly series

### Election Year Pattern
- In election years, NIFTY rallies 10-15% in the 6 months leading to results
- Post-election (if incumbent wins): Rally continues
- Post-election (if surprise): Sharp correction followed by recovery
- 2024 election: NIFTY rallied ~15% from Jan to Jun before election results

---

## 10. NIFTY 50 vs Midcap 150 Rotation

### Strategy
- Track relative performance of NIFTY Midcap 150 vs NIFTY 50
- Calculate ratio: Midcap150 / NIFTY50
- When ratio is trending up (20-day SMA > 50-day SMA): Overweight midcaps
- When ratio is trending down: Overweight large-caps (NIFTY 50)

### Implementation
- Use NIFTY Midcap 150 ETF and NIFTY 50 ETF
- Allocation: 60/40 or 70/30 split based on signal
- Rebalance monthly
- **Enhancement**: Add breadth indicators (% of midcap stocks above 200-DMA) as confirmation

### Historical Pattern
- Midcaps outperform after large-cap rallies (trickle-down effect, 3-6 month lag)
- Midcaps underperform first in corrections (higher beta, lower liquidity)
- DII SIP flows provide support to midcaps during corrections (unlike in past cycles)
- **Backtest CAGR (2012-2024)**: Rotation strategy ~20-24% vs buy-and-hold NIFTY ~14%

---

## 11. Impact of FPI Limits on Stock Returns

### SEBI FPI Limits
- Sectoral cap on FPI holdings (varies by sector; typically 24-74%)
- When FPI holding in a stock approaches limit, new FPI buying is restricted
- This creates supply-demand imbalance

### Trading Strategy
- Monitor FPI holding data (available quarterly from NSDL)
- Stocks where FPI holding is approaching 80%+ of sectoral limit: Expect reduced FPI demand
- Stocks where FPI holding has recently decreased significantly (>5% in a quarter): Potential selling pressure may be exhausted
- **Contrarian Signal**: Buy stocks where FPI selling is decelerating (selling exhaustion)

### FPI Quota Premium
- Stocks near FPI limits sometimes trade at a premium (domestic demand fills gap)
- When FPI limit is increased (regulatory change), stock sees buying surge
- Track SEBI circulars on FPI limit changes for event-driven trades

---

## 12. Practical Strategy Deployment Checklist

### Before Live Deployment
1. **Backtest on Indian data**: Use adjusted prices (corporate actions, splits, bonuses)
2. **Include realistic costs**: STT, exchange charges, stamp duty, GST, slippage (0.05-0.1% for liquid, 0.5-1% for illiquid)
3. **Account for circuit limits**: Strategy should handle untradeable stocks gracefully
4. **Test across regimes**: 2018 NBFC crisis, 2020 COVID crash, 2021-22 bull run, 2022 bear market
5. **Holiday calendar**: Ensure no trades are attempted on market holidays
6. **Settlement cycle**: T+1 means funds are available next day; model capital recycling correctly
7. **SEBI compliance**: Register algo with broker; implement kill switch
8. **Tax impact modeling**: Include STCG (20%) or business income tax in net return calculation
9. **Position sizing**: Account for lot sizes and margin requirements
10. **Drawdown limits**: Set maximum drawdown threshold (recommended: 15-20% for systematic strategies)
