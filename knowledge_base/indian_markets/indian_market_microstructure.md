# Indian Market Microstructure

## Knowledge Base Category: Exchange Architecture, Order Matching & Algo Infrastructure
## Last Updated: 2024-07
## Relevance: Technical reference for building low-latency and compliant algo trading systems in India

---

## 1. NSE NEAT+ Trading System

### Architecture Overview
- **NEAT+**: National Exchange for Automated Trading (upgraded platform)
- Built on a fault-tolerant, high-performance matching engine
- Processes 100,000+ orders per second at peak
- Average order response time: < 10 microseconds for colocated members
- Multicast tick-by-tick data feed for all market participants (SEBI mandate for fairness)

### Order Types Supported
- **Limit Order**: Specify price and quantity; most common for algo trading
- **Market Order**: Execute at best available price; avoid in illiquid stocks (slippage risk)
- **Stop-Loss Order (SL)**: Triggered when price crosses trigger price; then becomes limit order
- **Stop-Loss Market (SL-M)**: Triggered when price crosses trigger; becomes market order
- **Immediate or Cancel (IOC)**: Execute immediately or cancel; useful for algo execution
- **Good Till Date (GTD)**: Remains active until specified date (up to 45 days)
- **After Market Order (AMO)**: Queued for next day pre-open session

### Order Attributes
- **Disclosed Quantity**: Show only a portion of order to market (min 10% of total qty)
- **Validity**: Day, IOC, GTD
- **Product Types**: CNC (delivery), MIS (intraday), NRML (F&O normal)

---

## 2. Order Matching Algorithm

### Price-Time Priority (FIFO)
1. Orders are first prioritized by price (best price gets priority)
2. Among orders at the same price, earliest order gets priority (time priority)
3. No pro-rata matching; strict FIFO within each price level
4. Market orders get highest priority (treated as aggressive orders)

### Matching Process
- Buy orders sorted: Highest price first, then by time
- Sell orders sorted: Lowest price first, then by time
- Match occurs when best buy price >= best sell price
- Trade executes at the passive order's price (the order already in the book)

### Pre-Open Session Matching
- Uses call auction mechanism (not continuous matching)
- All orders collected during 9:00-9:08 window
- Single equilibrium price determined to maximize traded volume
- Unmatched orders carry forward to continuous session at limit price

### Implications for Algo Systems
- Latency matters for time priority; faster order placement gets better queue position
- Disclosed quantity orders lose time priority when new tranche is released
- IOC orders are useful for aggressive execution without leaving passive orders in book
- Always use limit orders for passive strategies; avoid market orders in low-liquidity stocks

---

## 3. Tick Size Rules

### Equity Segment
- **Standard Tick Size**: 0.05 INR (5 paise) for all stocks regardless of price
- This is notably coarse for low-priced stocks (e.g., 10 INR stock has 0.5% tick)
- For high-priced stocks (e.g., 5000 INR), tick is 0.001% (negligible)

### F&O Segment
- **Index Options**: 0.05 INR tick size on premium
- **Stock Options**: 0.05 INR tick size on premium
- **Futures**: 0.05 INR tick size
- **Currency**: 0.0025 INR tick size for USD/INR options

### Algo Considerations
- For low-priced options (premium < 5 INR), the tick size of 0.05 represents 1%+ spread
- Penny options (premium 0.05-0.50) have extremely wide relative spreads
- High tick-to-price ratio in cheap options creates lumpy P&L distributions
- When backtesting, always account for tick size rounding in fills

---

## 4. Lot Size Changes

### NSE Lot Size Revision Policy
- NSE reviews lot sizes periodically (typically every 6 months)
- Goal: Keep notional contract value between 5-10 lakh INR
- When stock price rises significantly, lot size is reduced
- When stock price falls, lot size may be increased

### Impact on Trading Systems
- Lot size changes take effect on the next new contract introduction
- Existing contracts retain old lot size until expiry
- **Critical Bug Source**: Systems that hardcode lot sizes will break on revision
- **Best Practice**: Fetch lot sizes from NSE contract master file daily
- NSE publishes lot size changes via circular; typically 2-4 weeks advance notice

### Historical Lot Size Changes (Examples)
- NIFTY: 75 -> 50 -> 25 (reduced as NIFTY price increased)
- Bank NIFTY: 25 -> 15 (reduced as Bank NIFTY price increased)
- Reliance: 500 -> 250 (adjusted for price changes)

---

## 5. Circuit Filters and Price Bands

### Stock-Level Circuit Filters
| Circuit Band | Applied To |
|---|---|
| 2% | Highly volatile/manipulated stocks (rare) |
| 5% | Volatile stocks not in F&O |
| 10% | Most non-F&O stocks |
| 20% | Standard for many stocks |
| No circuit | F&O eligible stocks (but operating range applies) |

### F&O Stocks: Operating Range
- No hard circuit limit, but NSE applies an "operating range"
- Typically 10-20% from previous close
- If breached, exchange may pause trading or extend range
- **Algo Impact**: Extreme moves in F&O stocks can still occur (no circuit to stop them)

### Dynamic Price Bands
- Exchange can revise price bands intraday based on market conditions
- Graded Surveillance Measure (GSM): Additional restrictions on suspicious stocks
- ASM (Additional Surveillance Measure): Enhanced monitoring on volatile stocks
- Stocks under GSM/ASM have reduced circuit limits and T+1 settlement restrictions

### Circuit Filter Handling in Algo Systems
- Before placing orders, check if stock is at upper/lower circuit
- If at UC, only sell orders are executable; if at LC, only buy orders
- Circuit stocks often show queue patterns; use disclosed quantity analysis
- Never place market orders on circuit-hit stocks
- Implement circuit detection: if last traded price equals circuit limit, flag the stock

---

## 6. T+1 Settlement Details

### Settlement Timeline
| Event | Timing |
|---|---|
| Trade Day (T) | 9:15 AM - 3:30 PM |
| Obligation Generation | T day evening (~7 PM) |
| Funds/Securities Pay-in | T+1 by 10:30 AM |
| Funds/Securities Payout | T+1 by 1:30 PM |

### Early Pay-In
- If securities are delivered before T+1, margin benefit is available
- Useful for covered call writers who hold shares in demat
- Reduces margin requirement for the corresponding short position

### Impact on FPI (Foreign Portfolio Investors)
- FPIs face forex settlement challenges (forex settles T+2 globally)
- Pre-funding of INR required for T+1 markets
- This has led to some FPI workflow adjustments and potential minor dislocations

---

## 7. SEBI Margin Rules

### Peak Margin Framework
- SEBI requires 100% of applicable margin to be available at all times
- Exchange takes at least 4 random snapshots per day
- Penalty for margin shortfall: 0.5% (up to 1 lakh) to 5% (above 1 crore) per day
- **End of Intraday Leverage**: Cannot use proceeds from sold positions to fund new positions intraday

### Upfront Margin Collection
- Brokers must collect VaR + ELM (Extreme Loss Margin) upfront
- For F&O: SPAN + Exposure margin must be collected before trade
- No credit or margin funding allowed for F&O positions
- Pledging of shares for margin is allowed (with haircut)

### Pledge Margin
- Clients can pledge shares held in demat for margin
- Haircut varies by stock: Large-cap index stocks: ~10-15%, Mid-caps: ~20-30%, Small-caps: ~40-60%
- Pledged margin can be used for F&O trading
- Unpledge takes T+1 to reflect

---

## 8. Broker APIs for Algo Trading

### Zerodha Kite Connect
- **Market Share**: Largest retail broker in India (~20% market share)
- **API Type**: RESTful HTTP + WebSocket for streaming
- **Rate Limits**: 10 orders/second, 200 orders/minute (contact for higher)
- **Data**: Historical candles (minute, day), live quotes, full market depth (5 levels)
- **Cost**: 2,000 INR/month for API access
- **Strengths**: Reliable, well-documented, large community
- **Weaknesses**: Rate limits can be restrictive for HFT; no tick-by-tick data
- **SDK**: Official Python SDK (kiteconnect)

### Angel One SmartAPI
- **API Type**: REST + WebSocket
- **Rate Limits**: 10 requests/second for order APIs
- **Data**: Historical data, live streaming, market depth
- **Cost**: Free API access (brokerage covers it)
- **Strengths**: No API subscription fee, decent documentation
- **Weaknesses**: Occasional reliability issues during high-volatility periods

### Upstox API v2
- **API Type**: REST + WebSocket (Protobuf for streaming)
- **Data**: Market depth, historical data, portfolio streaming
- **Cost**: Free API access
- **Strengths**: Modern API design, Protobuf for efficient streaming
- **Weaknesses**: Smaller community, documentation gaps

### Dhan API (DhanHQ)
- **API Type**: REST + WebSocket
- **Unique**: Order slicing (large orders auto-split), bracket orders via API
- **Cost**: Free API access
- **Strengths**: Good for execution-focused algos, order management features
- **Latency**: Competitive for retail-tier APIs

### Flattrade / Shoonya (Finvasia)
- **Cost**: Zero brokerage (fully free trading)
- **API**: REST + WebSocket
- **Strengths**: Zero cost makes it ideal for high-frequency options selling
- **Weaknesses**: Limited community, documentation quality

### API Comparison Matrix
| Feature | Kite | SmartAPI | Upstox | Dhan |
|---|---|---|---|---|
| API Cost | 2000/mo | Free | Free | Free |
| Order Rate Limit | 10/sec | 10/sec | 10/sec | 10/sec |
| WebSocket | Yes | Yes | Yes (Protobuf) | Yes |
| Historical Data | 1-min+ | 1-min+ | 1-min+ | 1-min+ |
| Market Depth | 5 levels | 5 levels | 5 levels | 5 levels |
| Python SDK | Official | Official | Official | Official |

---

## 9. Colocation and DMA

### NSE Colocation
- **Location**: NSE data center in Mumbai (Andheri/BKC area)
- **Access**: Available to trading members (brokers) who apply to NSE
- **Latency**: Sub-100 microsecond order execution
- **Cost**: Rack space: ~2-5 lakh/month, Tick-by-tick feed: ~1-2 lakh/month
- **SEBI Regulations**: Fair access mandated; all colo members receive data at same time

### Tick-by-Tick (TBT) Data Feed
- SEBI mandated TBT data for all members (not just colo) from 2019
- Multicast UDP feed from NSE
- Provides every order book change (not just trade ticks)
- Essential for market-making and statistical arbitrage strategies

### DMA (Direct Market Access)
- Allows institutional clients to place orders directly on exchange
- Requires broker sponsorship and SEBI approval
- Lower latency than going through broker's order management system
- **Regulatory**: All DMA orders must pass through broker's pre-trade risk checks

### CTCL (Computer-to-Computer Link)
- Legacy connectivity mode (broker terminal software connects to exchange)
- Being replaced by DMA and API-based access
- Still used by some proprietary trading desks

---

## 10. Algo Trading Regulations (SEBI Framework)

### SEBI Circular on Algo Trading (2021-2024)
- All algorithmic orders must be tagged with unique algo ID at exchange level
- Brokers must get each algo strategy approved by the exchange
- API-based retail algo orders also fall under this framework (SEBI 2023 clarification)
- **Penalty**: Unapproved algo strategies can result in trading suspension

### Key Requirements
1. **Algo Registration**: Each strategy must be registered with exchange via broker
2. **Order Tagging**: Every algo order must carry the registered algo ID
3. **Kill Switch**: Mandatory ability to stop all algo orders instantly
4. **Risk Controls**: Pre-trade checks for quantity, price, and value limits
5. **Audit Trail**: Complete log of all algo decisions and order submissions
6. **Two-Factor Authentication**: Required for API login (TOTP-based)

### Practical Compliance Steps
- Register your strategy through your broker's algo desk
- Implement kill switch accessible via API and manual override
- Log every signal, order, fill, and cancellation with timestamps
- Rate limit your order flow to stay within exchange-mandated order-to-trade ratios
- Maximum order value limits per order and per day
- Monitor for erroneous orders; implement fat-finger checks

### Recent Developments (2024)
- SEBI is developing a framework to regulate retail algo platforms (like Tradetron, Streak)
- API-based orders from retail accounts may require exchange-level algo approval
- Discussion on mandating audit trails for all API orders
- Proposed lot size increases to reduce speculative retail F&O participation

---

## 11. Latency Hierarchy in Indian Markets

### Approximate Latencies
| Access Method | Round-Trip Latency | Suitable For |
|---|---|---|
| NSE Colocation | 10-100 microseconds | Market making, statistical arbitrage |
| DMA (same city) | 1-5 milliseconds | Institutional execution |
| Broker API (cloud) | 20-100 milliseconds | Retail algo, swing strategies |
| Broker API (local) | 10-50 milliseconds | Intraday strategies |
| Manual Trading | 500ms - 5 seconds | Discretionary |

### Cloud Hosting for Algo
- AWS Mumbai (ap-south-1) or Azure Central India provides 5-20ms to broker APIs
- DigitalOcean Mumbai and Linode Mumbai are cost-effective alternatives
- **Best Practice**: Run strategy server in Mumbai region to minimize latency to broker APIs
- For non-latency-sensitive strategies (>1 min holding period), latency differences are negligible

---

## 12. Practical Microstructure Pitfalls

1. **Phantom Liquidity**: Large orders at best bid/ask that disappear when you try to hit them; use IOC to test real liquidity
2. **Opening Auction Quirks**: Pre-open equilibrium price can deviate significantly from previous close; use pre-open data for gap analysis but do not assume fill at equilibrium price
3. **Closing Price Calculation**: VWAP of last 30 minutes; closing price may differ from last traded price
4. **Frozen Stocks**: Exchange can freeze trading in a stock with zero notice (regulatory action); handle "order rejected" errors gracefully
5. **API Downtime**: Broker APIs can have connectivity issues during market hours; implement retry logic with exponential backoff
6. **Order Modifications**: Modifying an order loses time priority; cancel-replace is sometimes better
7. **Corporate Action Dates**: On ex-date, order book is reset; all GTD orders are cancelled
8. **Dividend Futures Adjustment**: Stock futures price drops by dividend amount on ex-date; account in futures-spot basis calculations
9. **Auction Session**: Illiquid stocks may enter periodic call auction instead of continuous trading; different matching rules apply
10. **New Listing Day**: No circuit limits on listing day; extreme volatility possible; many algo strategies exclude new listings for first 5 trading days
