# Indian Market Data Sources for Quantitative Trading

## Knowledge Base Category: Data Infrastructure, APIs & Data Engineering
## Last Updated: 2024-07
## Relevance: Building reliable data pipelines for Indian market quant systems

---

## 1. NSE Bhav Copy (End-of-Day Official Data)

### What is Bhav Copy?
- Official end-of-day (EOD) trade file published by NSE after market close
- Contains OHLC, close, last traded price, total traded quantity, total traded value, and number of trades
- Considered the most authoritative source for daily price data
- Published daily by ~6:30 PM IST for equity segment

### Types of Bhav Copies
| File | Content | Format |
|---|---|---|
| cm_bhav_copy | Cash market equity data | CSV |
| fo_bhav_copy | F&O segment data (futures and options) | CSV |
| mto (Market Turnover) | Delivery quantity data | CSV |
| security_bhav_copy | Combined security-level data | CSV |

### Bhav Copy URL Patterns
- Equity: https://archives.nseindia.com/content/historical/EQUITIES/{YYYY}/{MMM}/cm{DDMMYYYY}bhav.csv.zip
- F&O: https://archives.nseindia.com/content/historical/DERIVATIVES/{YYYY}/{MMM}/fo{DDMMYYYY}bhav.csv.zip
- Replace {YYYY} with year, {MMM} with 3-letter month (JAN, FEB, etc.), {DD} with date

### Parsing Bhav Copy
- **Key Columns (Equity)**: SYMBOL, SERIES (EQ for equity, BE for trade-to-trade), OPEN, HIGH, LOW, CLOSE, LAST, PREVCLOSE, TOTTRDQTY, TOTTRDVAL, TOTALTRADES
- **Series Filter**: Always filter SERIES = 'EQ' for regular equity data; 'BE' stocks have no intraday trading
- **Corporate Actions**: Bhav copy prices are NOT adjusted for splits/bonuses; you must apply adjustments separately
- **Delivery Data**: Cross-reference with MTO file for delivery quantity and delivery percentage

### Automation Tips
- Schedule daily download at 7:00 PM IST via cron job
- Handle holidays gracefully (file will not exist; skip without error)
- NSE may block automated downloads; use appropriate headers and rate limiting
- Consider using a proxy or rotating user-agents if IP is blocked
- Store raw files in date-partitioned directory structure: /data/bhav/{YYYY}/{MM}/{DD}/

---

## 2. NSE APIs and Web Scraping

### NSE Website API (Unofficial)
- NSE's website uses internal APIs that can be accessed programmatically
- **Base URL**: https://www.nseindia.com/api/
- Endpoints include: /quote-equity, /option-chain-equities, /market-data-pre-open, /equity-stockIndices
- **Authentication**: Requires session cookies from nseindia.com; must first hit homepage to get cookies
- **Rate Limiting**: Aggressive; will block after 20-30 rapid requests
- **Reliability**: NSE frequently changes API structure; not officially supported

### NSE Option Chain API
- Endpoint: /api/option-chain-indices?symbol=NIFTY
- Returns full option chain with OI, change in OI, IV, LTP, bid/ask for all strikes
- Refresh rate: ~3 minutes during market hours
- Essential for OI analysis, max pain calculation, PCR tracking

### Practical NSE Scraping Setup
- Use Python requests library with session management
- Set headers: User-Agent, Accept, Accept-Language to mimic browser
- First request to https://www.nseindia.com to obtain cookies
- Then use session cookies for API requests
- Implement exponential backoff on 403/429 errors
- **Warning**: NSE may block your IP for aggressive scraping; use judiciously

---

## 3. Yahoo Finance (*.NS Tickers)

### Ticker Convention
- NSE stocks: {SYMBOL}.NS (e.g., RELIANCE.NS, TCS.NS, INFY.NS)
- BSE stocks: {SYMBOL}.BO (e.g., RELIANCE.BO)
- Indices: ^NSEI (NIFTY 50), ^NSEBANK (Bank NIFTY), ^BSESN (SENSEX)

### Using yfinance Python Library
- Install: pip install yfinance
- Usage: yf.download("RELIANCE.NS", start="2020-01-01", end="2024-07-01")
- Provides: OHLCV data, adjusted close (corporate action adjusted), dividends, splits
- **Frequency**: Daily, weekly, monthly; intraday data available for last 60 days (1m, 5m, 15m, 1h)

### Limitations
- Intraday data limited to 60 days history (1-minute) and 730 days (1-hour)
- Data quality can have occasional gaps or errors (especially for less liquid stocks)
- Adjusted close computation may differ from NSE official adjustments
- Rate limits apply; bulk downloads may be throttled
- Not suitable as sole data source for production systems; use for research/backtesting

### Data Quality Checks
- Compare yfinance daily close with NSE bhav copy close for validation
- Check for missing dates (should match NSE trading calendar)
- Verify corporate action adjustments match NSE adjustment factors
- Dividend and split data may have date discrepancies (ex-date vs record date)

---

## 4. Jugaad Data (Python Library)

### Overview
- Python library specifically for Indian stock market data
- Fetches data from NSE directly
- Install: pip install jugaad-data
- Covers: Stock prices, derivatives data, index data

### Key Features
- **Stock History**: from jugaad_data.nse import stock_df; stock_df("RELIANCE", "01-01-2020", "01-07-2024")
- **Derivatives**: Option chain data, F&O bhav copy parsing
- **Index Data**: NIFTY, Bank NIFTY historical data
- **No API Key Required**: Direct NSE data access

### Advantages
- Purpose-built for Indian markets; understands NSE data structures
- Handles NSE cookie management internally
- Returns pandas DataFrames ready for analysis
- Open source; community maintained

### Limitations
- Depends on NSE website structure; may break if NSE changes its APIs
- Rate limited by NSE backend; not suitable for real-time data
- No intraday tick data; only EOD and periodic snapshots
- May require periodic updates as NSE changes endpoints

---

## 5. TVDatafeed (TradingView Data)

### Overview
- Unofficial Python library to fetch data from TradingView
- Install: pip install tvdatafeed (or from GitHub)
- Provides: Historical OHLCV data at various timeframes
- Covers: NSE, BSE, MCX, and global exchanges

### Usage
- from tvdatafeed import TvDatafeed, Interval
- tv = TvDatafeed() (anonymous access for limited data)
- tv = TvDatafeed(username, password) (TradingView account for more data)
- data = tv.get_hist("NIFTY", "NSE", interval=Interval.in_daily, n_bars=5000)

### Timeframes Available
- 1 minute, 3 min, 5 min, 15 min, 30 min, 45 min
- 1 hour, 2 hour, 3 hour, 4 hour
- Daily, weekly, monthly

### Advantages
- Intraday historical data beyond what yfinance provides
- Good quality data (TradingView's data infrastructure)
- Multiple Indian exchanges supported (NSE, BSE, MCX)

### Limitations
- Unofficial library; may break with TradingView changes
- TradingView may restrict access for heavy usage
- Data may have minor discrepancies with NSE official data
- Not recommended for production; use for research and validation

---

## 6. OpenBB Platform (India Support)

### Overview
- Open-source investment research platform
- Supports Indian market data through multiple providers
- Install: pip install openbb
- Provides: Fundamental data, technical analysis, news, economic data

### India-Specific Features
- Stock data via yfinance backend (*.NS tickers)
- Fundamental data: Financials, ratios, institutional holdings
- Economic data: RBI rates, inflation, GDP via FRED or local sources
- News aggregation: Can pull India market news

### Usage for Quant Systems
- Useful for fundamental factor models (value, quality, growth factors)
- Financial statement data for Indian companies
- Cross-asset analysis (equities + commodities + currencies)
- Not a primary data source but good for supplementary research data

---

## 7. Corporate Actions Data

### Sources
- **NSE Corporate Actions**: https://www.nseindia.com/companies-listing/corporate-filings-actions
- **BSE Corporate Actions**: https://www.bseindia.com/corporates/corporate_act.aspx
- Types: Bonus, Stock Split, Rights Issue, Dividend, Merger, De-listing

### Adjustment Factor Calculation
- **Stock Split (e.g., 1:5)**: Adjustment factor = 1/5 = 0.2; multiply all historical prices by 0.2
- **Bonus (e.g., 1:1)**: Adjustment factor = 1/(1+1) = 0.5; multiply all historical prices by 0.5
- **Rights Issue**: More complex; depends on rights price and ratio
- **Dividend**: Generally not price-adjusted in India (unlike total return indices)

### Data Pipeline Best Practice
- Maintain a corporate actions database with columns: symbol, ex_date, action_type, ratio, adjustment_factor
- Apply adjustments backward from most recent data
- Re-run adjustment pipeline whenever new corporate action data is received
- Cross-validate adjusted prices against NSE adjusted close or Bloomberg/Reuters

### Common Pitfalls
- Missing a corporate action causes discontinuity in price series
- Different data providers may apply adjustments on different dates (ex-date vs record date)
- Some providers adjust for dividends; others do not (total return vs price return)
- Rights issues require careful handling (not all shareholders participate)

---

## 8. FII/DII Daily Data

### NSDL FPI Data
- **Source**: https://www.fpi.nsdl.co.in/web/Reports/Latest.aspx
- Provides: Daily FPI/FII net investment in equity and debt
- Breakdown: Gross buy, gross sell, net investment
- Published by ~7:30 PM IST daily
- **Format**: Downloadable as Excel/CSV

### CDSL Participant Data
- Similar data available from CDSL
- SEBI compiles combined NSDL + CDSL data

### NSE Participant-Wise OI Data
- **Source**: NSE daily reports section
- Provides: Client, FII, DII, Pro (proprietary) positions in futures and options
- Separate data for index futures, index options, stock futures, stock options
- Shows long positions, short positions, and net positions
- **Critical for**: FII long-short ratio calculation, sentiment analysis

### Automation
- Schedule daily download after 8:00 PM IST
- Parse Excel files using openpyxl or pandas read_excel
- Store in time-series database (InfluxDB, TimescaleDB, or simple SQLite)
- Calculate derived metrics: FII Net (5-day rolling), FII L/S Ratio, DII cumulative flows

---

## 9. Popular Indian Financial Data Websites

### Moneycontrol
- **URL**: https://www.moneycontrol.com
- India's largest financial portal
- Data: Stock quotes, financial statements, mutual fund NAVs, news
- Has internal APIs (not officially documented) that can be scraped
- Useful for: Quarterly results data, consensus estimates, ownership data
- **Limitation**: Terms of service may restrict scraping

### Tickertape.in
- **URL**: https://www.tickertape.in
- Clean interface for stock screening and fundamental data
- Data: Financials, peer comparison, mutual fund overlap
- Good for: Quick fundamental checks, not for bulk data extraction
- Owned by Smallcase (fintech platform)

### Screener.in
- **URL**: https://www.screener.in
- Excellent for fundamental screening with custom queries
- Data: 10-year financial history, ratios, growth metrics
- Custom screening: Write queries like "Market Cap > 10000 AND ROE > 15 AND Debt to Equity < 0.5"
- Export: Allows CSV download of screening results (with account)
- **Quant Use**: Build fundamental factor models using screener data
- Has an unofficial API that returns JSON data

### TradingView India
- **URL**: https://www.tradingview.com (select NSE/BSE exchange)
- Charting platform with Pine Script for custom indicators
- Data: Real-time and historical for NSE, BSE, MCX
- **Quant Use**: Visual validation of strategy signals, Pine Script prototyping
- **Limitation**: Not a programmatic data source (except via TVDatafeed library)

---

## 10. Real-Time Data Feeds

### TrueData
- **Type**: Commercial real-time data feed
- **Coverage**: NSE (Equity, F&O, Currency, Commodity), BSE, MCX
- **Latency**: Low-latency tick-by-tick data
- **API**: WebSocket-based, Python SDK available
- **Cost**: Plans from ~2,000-10,000 INR/month depending on segments
- **Use Case**: Real-time algo execution, live options chain monitoring
- **Data Depth**: Full market depth (5 best bid/ask levels)

### Global Datafeeds (GDF)
- **Type**: Commercial real-time data feed
- **Coverage**: NSE, BSE, MCX, NCDEX
- **Technology**: TCP-based feed, DDE, RTD for Excel
- **API**: Available for custom integration
- **Cost**: Variable; typically 3,000-15,000 INR/month
- **Historical Data**: Provides tick-by-tick historical data (at additional cost)
- **Use Case**: Institutional-grade data for backtesting and live trading

### Broker WebSocket Feeds
- Most broker APIs provide real-time streaming via WebSocket
- **Kite Connect WebSocket**: Tick data for subscribed instruments (max 3000)
- **SmartAPI WebSocket**: Similar real-time streaming
- **Quality**: Good for retail algo trading; not suitable for sub-millisecond strategies
- **Cost**: Included with broker API subscription (or free for some brokers)

### Comparison: Real-Time Data Options
| Provider | Latency | Historical Tick Data | Cost (Monthly) | Best For |
|---|---|---|---|---|
| TrueData | ~50-100ms | Limited | 2,000-10,000 | Retail algo traders |
| Global Datafeeds | ~20-50ms | Yes (paid) | 3,000-15,000 | Institutional algo |
| Broker WebSocket | ~100-500ms | No | Free-2,000 | Basic retail algo |
| NSE Colo Feed | <1ms | Yes | 1-5 lakh | HFT/Market making |

---

## 11. Nifty Indices Website

### Source
- **URL**: https://www.niftyindices.com
- Official website for all NIFTY indices maintained by NSE Indices Limited

### Available Data
- **Daily Index Values**: OHLC for all 100+ NIFTY indices
- **Historical Data**: Downloadable CSV for any index from inception
- **Total Return Index (TRI)**: Price return and total return (with dividends)
- **Index Constituents**: Current and historical constituents with weights
- **Methodology Documents**: Full index methodology for each index

### Key Indices for Quant Systems
| Index | Symbol | Use Case |
|---|---|---|
| NIFTY 50 | NIFTY 50 | Primary benchmark |
| NIFTY Bank | NIFTY BANK | Banking sector proxy |
| NIFTY IT | NIFTY IT | IT sector, USD correlation |
| NIFTY Midcap 150 | NIFTY MIDCAP 150 | Mid-cap allocation |
| NIFTY Smallcap 250 | NIFTY SMLCAP 250 | Small-cap momentum |
| India VIX | INDIA VIX | Volatility regime detection |
| NIFTY 50 Equal Weight | NIFTY50 EQL WGT | Equal weight benchmark |
| NIFTY 500 | NIFTY 500 | Broad market benchmark |

### Data Pipeline Setup
- Download daily index data from niftyindices.com (CSV format)
- Parse and store in database alongside stock-level data
- Track index rebalancing dates (semi-annual) for constituent changes
- Constituent change data is essential for survivorship-bias-free backtesting

---

## 12. Data Quality and Pipeline Best Practices

### Common Data Issues in Indian Markets
1. **Corporate Action Gaps**: Missing adjustments for splits, bonuses in some providers
2. **Trade-to-Trade Stocks**: BE series stocks have no intraday data; only delivery trades
3. **Suspended Stocks**: Stocks suspended by exchange show zero volume; handle in universe filters
4. **Index Reconstitution Bias**: Using current index constituents for historical backtest (survivorship bias)
5. **Dividend Data Discrepancies**: Ex-date vs record date confusion across providers
6. **F&O Data Gaps**: Option chain data for illiquid strikes may have stale prices
7. **Pre-Open Data**: Pre-open session prices can distort OHLC if included incorrectly
8. **Holiday Data**: Different holiday calendars across exchanges and data providers

### Recommended Data Stack
| Layer | Tool | Purpose |
|---|---|---|
| Storage | PostgreSQL + TimescaleDB | Time-series optimized storage |
| Raw Data | NSE Bhav Copy + Jugaad Data | EOD prices and F&O data |
| Intraday | TVDatafeed or TrueData | Historical and live intraday |
| Real-Time | Broker WebSocket (Kite/Angel) | Live execution data |
| Fundamentals | Screener.in + Moneycontrol | Financial statements |
| FII/DII | NSDL + NSE Participant Data | Flow data for sentiment |
| Indices | NiftyIndices.com | Benchmark and constituent data |
| Alternative | Google Trends, News APIs | Sentiment and event data |

### Data Validation Pipeline
1. **Daily Check**: Compare bhav copy stock count with expected (~2000 for EQ series)
2. **Price Continuity**: Flag stocks with > 20% daily change (likely corporate action or error)
3. **Volume Sanity**: Flag zero-volume days for liquid stocks (data issue or suspension)
4. **Cross-Source Validation**: Compare yfinance close with bhav copy close (should match)
5. **Corporate Action Reconciliation**: Verify adjustment factors against NSE official data
6. **Holiday Calendar Sync**: Ensure no data is expected on exchange holidays
7. **Timestamp Validation**: All intraday data should fall within 9:15 AM - 3:30 PM IST
8. **Duplicate Detection**: Check for duplicate records (same symbol, same date)

### Storage Sizing Estimates
- Daily equity bhav copy: ~200 KB (compressed) x 250 trading days = ~50 MB/year
- Daily F&O bhav copy: ~2-5 MB (compressed) x 250 days = ~625 MB - 1.25 GB/year
- 1-minute candle data for NIFTY 50 stocks: ~500 MB/year
- Tick-by-tick data for F&O segment: ~50-100 GB/day (requires significant storage)
- Full option chain snapshots (every 3 min): ~5-10 GB/day
