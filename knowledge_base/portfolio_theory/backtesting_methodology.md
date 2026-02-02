# Backtesting Methodology for Quantitative Trading Systems

## Overview

Backtesting is the process of evaluating a trading strategy's performance on historical data.
While conceptually straightforward, rigorous backtesting is one of the most challenging aspects
of quantitative trading. The majority of backtested strategies that appear profitable fail in
live trading due to methodological errors, biases, and unrealistic assumptions.

---

## Walk-Forward Analysis

### Concept

Walk-forward analysis (WFA) is the gold standard for evaluating adaptive trading strategies.
It simulates the real-world process of periodically re-optimizing a strategy on recent data
and then trading it on unseen future data.

### Implementation Steps

1. **Define windows**: Choose an in-sample (IS) training window and an out-of-sample (OOS)
   testing window. Example: 2 years IS, 6 months OOS.
2. **Optimize on IS data**: Find optimal parameters using the training window.
3. **Test on OOS data**: Apply optimized parameters to the subsequent OOS window and record
   performance (no peeking, no re-optimization).
4. **Roll forward**: Slide both windows forward by the OOS length and repeat.
5. **Aggregate OOS results**: Concatenate all OOS periods to produce the walk-forward
   equity curve -- this represents realistic performance.

### Best Practices

- IS window should be 3-5x the OOS window length.
- OOS periods should cover multiple market regimes.
- The walk-forward efficiency ratio (OOS performance / IS performance) should exceed 0.5
  for a robust strategy.
- Use anchored (expanding) or rolling windows depending on whether you believe older data
  remains informative.
- Track parameter stability across windows: wildly different optimal parameters signal
  overfitting or regime sensitivity.

---

## Out-of-Sample Testing

### Three-Way Data Split

```
|---------- Training ----------|--- Validation ---|--- Test ---|
|     Model fitting            | Hyperparameter   | Final      |
|     Feature selection        | tuning           | evaluation |
|                              | Model selection  | (TOUCH ONCE)|
```

- **Training set**: Used for model fitting and feature engineering.
- **Validation set**: Used for hyperparameter tuning and model selection. Can be accessed
  multiple times.
- **Test set**: Touched exactly once for final performance evaluation. Multiple accesses
  contaminate the test set and produce overly optimistic results.

### Combinatorial Purged Cross-Validation (CPCV)

- Standard k-fold cross-validation is invalid for time series due to temporal dependence.
- **Purged CV** (Lopez de Prado): Removes observations near the train/test boundary to
  prevent information leakage from overlapping labels.
- **Embargo period**: Additional buffer after purging to account for serial correlation.
- **CPCV**: Tests all possible train/test combinations while respecting temporal order,
  producing a distribution of backtest paths rather than a single path.

---

## Common Biases in Backtesting

### Look-Ahead Bias

- **Definition**: Using information that would not have been available at the time of the
  trading decision.
- **Examples**:
  - Using adjusted close prices that incorporate future splits/dividends.
  - Using point-in-time fundamental data that was actually released with a delay.
  - Selecting universe membership based on current (not historical) index composition.
  - Using end-of-day data for intraday decisions.
- **Prevention**: Use point-in-time databases; implement strict data timestamp validation;
  lag all fundamental data by the actual reporting delay (e.g., quarterly results in India
  are available 30-45 days after quarter end).

### Survivorship Bias

- **Definition**: Testing strategies only on assets that survived the entire test period,
  ignoring those that were delisted, merged, or went bankrupt.
- **Impact**: Inflates returns by 1-4% annually depending on the strategy and universe.
  Value and small-cap strategies are most affected since distressed companies (which
  would be selected by these strategies) are disproportionately delisted.
- **Prevention**: Use survivorship-bias-free databases that include delisted securities
  with their full history and delisting returns.
- **Indian market note**: NSE provides historical index constituent data; however,
  comprehensive delisting-adjusted data for the broader market requires specialized
  providers (CMIE Prowess, Bloomberg).

### Selection Bias and Data Mining

- **Definition**: Testing many strategies or parameter combinations and reporting only the
  best results without adjusting for the number of tests conducted.
- **The multiple testing problem**: If you test 100 random strategies, approximately 5 will
  appear significant at the 5% level purely by chance.
- **Haircut Sharpe Ratio**: Harvey and Liu (2015) propose deflating the Sharpe ratio based
  on the number of strategies tested.
- **Prevention**: Apply Bonferroni correction, Benjamini-Hochberg FDR control, or require
  a minimum t-statistic of 3.0 (Harvey, Liu, Zhu 2016).

### Time Period Bias

- **Definition**: Strategy performance is dominated by a specific market regime that may
  not recur.
- **Example**: A momentum strategy tested from 2010-2020 benefits enormously from the
  post-GFC bull market; performance from 2000-2010 would look very different.
- **Prevention**: Test across multiple decades; analyze performance by regime; verify the
  strategy has a logical economic rationale that transcends specific periods.

---

## Transaction Cost Modeling

### Components of Transaction Costs

| Cost Component       | Description                                    | Typical Range (India)     |
|----------------------|------------------------------------------------|---------------------------|
| Brokerage            | Broker commission per trade                    | Rs 20/order (discount) or 0.01-0.05% |
| STT                  | Securities Transaction Tax                     | 0.1% sell (delivery), 0.025% sell (intraday), 0.0125% sell (F&O) |
| Exchange charges     | NSE/BSE transaction fees                       | 0.00345% (NSE equity)    |
| GST                  | 18% on brokerage + exchange charges            | ~18% of above             |
| Stamp duty           | State-level duty on buy transactions           | 0.015% (equity delivery)  |
| SEBI turnover fee    | Regulatory fee                                 | 0.0001%                   |
| Spread cost          | Bid-ask spread (half-spread per side)          | 0.02-0.05% (Nifty 50), 0.1-1%+ (small-cap) |
| Market impact        | Price movement caused by the trade itself      | Depends on order size vs. ADV |
| Slippage             | Difference between signal price and fill price | 0.05-0.5% depending on speed |

### Market Impact Models

1. **Square-root model (Almgren)**: Impact proportional to square root of participation rate.
   ```
   Impact = sigma * k * sqrt(V_trade / V_daily)
   ```
   where sigma is daily volatility, k is a constant (typically 0.1-0.5), V_trade is trade
   size, and V_daily is average daily volume.

2. **Linear temporary impact**: Simpler model assuming impact proportional to trade size.

3. **Decay models**: Impact decays over time (temporary vs. permanent components).

### Realistic Cost Assumptions for Indian Market Backtests

- For Nifty 50 stocks: 0.10-0.15% round-trip all-in cost for moderate size.
- For Nifty Midcap 150: 0.20-0.40% round-trip depending on individual stock liquidity.
- For small-caps below Nifty 500: 0.50-2.00% round-trip; use volume-dependent impact models.
- Always model costs as a function of trade size, not a fixed percentage.

---

## Monte Carlo Simulation for Strategy Validation

### Purpose

Monte Carlo methods test strategy robustness by simulating thousands of alternative scenarios
to assess whether observed performance could be due to luck rather than genuine alpha.

### Techniques

1. **Return shuffling (bootstrap)**: Randomly reshuffle the strategy's daily returns and
   reconstruct equity curves. Compare the actual Sharpe ratio to the distribution of
   bootstrapped Sharpe ratios. If the actual Sharpe is above the 95th percentile of the
   bootstrap distribution, the result is unlikely due to luck.

2. **Trade shuffling**: Randomly reorder trades to test if performance depends on trade
   sequence (it should not for a robust strategy, but serial correlation in returns
   can make it appear so).

3. **Parameter perturbation**: Run the strategy with randomly perturbed parameters
   (e.g., uniform noise of +/- 20% on each parameter) to test sensitivity. A robust
   strategy should perform reasonably across a neighborhood of parameter values.

4. **Synthetic data generation**: Generate synthetic price series with the same statistical
   properties as the actual data but no embedded alpha signal. The strategy should not
   show significant profits on synthetic data (if it does, it is overfitting to noise
   characteristics).

5. **White's Reality Check / Hansen's SPA test**: Test whether the best strategy from a
   set of tested strategies outperforms a benchmark after accounting for data snooping.

---

## Common Backtesting Mistakes

1. **Ignoring transaction costs**: Strategies with high turnover can appear profitable
   gross but lose money net of costs.
2. **Using unadjusted prices**: Corporate actions (splits, bonuses, dividends) must be
   properly adjusted; in India, use NSE adjusted price series or compute adjustments
   from corporate action data.
3. **Applying current universe**: Test on the universe that existed at each point in time.
4. **Ignoring execution delays**: Assume at least a 1-bar delay between signal generation
   and order execution.
5. **Perfect fill assumption**: In reality, limit orders may not fill and market orders
   incur slippage. Model partial fills for large orders.
6. **Ignoring margin and leverage constraints**: Strategies requiring leverage must model
   margin requirements and the cost of borrowing.
7. **Conflating in-sample and out-of-sample**: Any strategy optimization on the full dataset
   invalidates the entire backtest as an OOS test.
8. **Ignoring regime dependency**: A strategy that works only in bull markets is not robust.
9. **Short-selling constraints**: Not all stocks can be shorted; borrow costs vary. In India,
   stock-specific short selling is only available intraday or through F&O; there is no
   general stock lending and borrowing mechanism for retail.
10. **Overestimating capacity**: Strategies in illiquid segments cannot absorb meaningful capital.

---

## Backtesting Frameworks in Python

### Backtrader

- Event-driven framework with rich feature set.
- Supports live trading connections (Interactive Brokers, etc.).
- Good for strategy prototyping and complex order logic.
- Slower for large-scale parameter optimization.
- Community-maintained; development has slowed.

### VectorBT

- Vectorized (NumPy/Pandas based) framework for speed.
- Excellent for rapid iteration and parameter sweeps.
- Handles portfolio-level backtesting efficiently.
- Built-in performance analytics and visualization.
- Best for strategies expressible as vectorized operations.

### Zipline (and Zipline-Reloaded)

- Originally developed by Quantopian; now community-maintained as zipline-reloaded.
- Event-driven with built-in pipeline for factor computation.
- Integrates with Alphalens for factor analysis and Pyfolio for performance attribution.
- Requires specific data bundle format; integration with Indian data requires custom ingest.

### Custom Frameworks

- Many professional quant teams build custom backtesting engines tailored to their
  specific needs using NumPy/Pandas.
- Advantages: full control over assumptions, no framework overhead.
- Disadvantages: significant development and maintenance burden; risk of undiscovered bugs.

### Framework Selection Guide

| Framework   | Speed     | Flexibility | Learning Curve | Live Trading | Best For               |
|-------------|-----------|-------------|----------------|--------------|------------------------|
| Backtrader  | Moderate  | High        | Moderate       | Yes          | Complex strategies     |
| VectorBT    | Fast      | Moderate    | Low            | Limited      | Factor research, sweeps |
| Zipline     | Moderate  | Moderate    | High           | No           | Factor pipeline        |
| Custom      | Variable  | Highest     | N/A            | Custom       | Production systems     |

---

## Indian Market-Specific Backtesting Concerns

### Circuit Limits and Price Bands

- Indian stocks have daily circuit limits: 2%, 5%, 10%, or 20% depending on the stock.
- When a stock hits its circuit limit, trading is halted or severely constrained.
- **Impact on backtesting**: Strategies may generate signals on circuit-hit days when
  execution is impossible. Model circuit limits by preventing trades when the stock
  is at or near its circuit band.
- Lower circuit hits create forced selling cascades that cannot be captured by standard
  backtests assuming continuous trading.

### T+1 Settlement

- India moved to T+1 settlement in January 2023 (from T+2).
- **Impact**: Funds from selling are available one day later; buying requires upfront margin.
- For backtesting: model settlement correctly when computing available capital and margin.
- Intraday strategies are unaffected; delivery-based strategies need accurate settlement modeling.

### Corporate Action Complexity

- Indian markets have frequent corporate actions: bonuses, stock splits, rights issues,
  buybacks, demergers, mergers, and dividend payments.
- Price adjustment for backtesting must account for all these events.
- Use NSE's adjusted price series or compute adjustment factors from corporate action data.
- Rights issues are particularly tricky: they introduce dilution and optionality that
  simple price adjustment does not fully capture.

### Data Quality Issues

- Historical Indian market data often has gaps, errors, and inconsistencies.
- Verify data quality by checking for: zero-volume days (holidays misclassified as trading
  days), outlier returns (data errors vs. genuine moves), corporate action adjustment errors.
- Cross-reference multiple data sources when possible.
- BSE and NSE data may differ for dual-listed stocks; choose one exchange consistently.

### F&O-Specific Concerns

- Options backtesting requires tick-level or at minimum 1-minute data for realistic results.
- NSE options have American-style exercise for stock options and European-style for index options.
- Historical options data is expensive and often incomplete for deep OTM strikes.
- Liquidity varies enormously across strikes and expiries; model this explicitly.
- Weekly expiry backtests are only valid from when weekly contracts were introduced
  (Bank Nifty weekly: 2016; Nifty weekly: 2019; Fin Nifty: 2021).
- Margin requirements change based on SPAN + exposure margin; model these accurately
  for leveraged strategies.

### Regulatory Changes

- Indian market rules have changed frequently: STT rates, margin requirements (peak margin
  rules 2021), lot sizes, circuit limit rules, trading hours.
- Backtests spanning long periods must account for these structural changes.
- Apply the rules that were in effect at each point in time, not current rules retroactively.

---

## Backtesting Checklist

Before trusting any backtest result, verify:

- [ ] Point-in-time data used (no look-ahead bias)
- [ ] Survivorship-bias-free universe
- [ ] Realistic transaction costs modeled
- [ ] Execution delay of at least one bar
- [ ] Walk-forward or proper OOS methodology
- [ ] Multiple testing correction applied
- [ ] Strategy tested across different market regimes
- [ ] Parameter sensitivity analysis completed
- [ ] Monte Carlo validation performed
- [ ] Capacity analysis relative to target AUM
- [ ] Circuit limits and trading halts modeled (for Indian markets)
- [ ] Corporate actions properly adjusted
- [ ] Tax implications considered (STCG vs. LTCG impact on net returns)
- [ ] Margin requirements modeled for leveraged strategies

---

## Key References

- Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio."
- Harvey, C. & Liu, Y. (2015). "Backtesting."
- Harvey, C., Liu, Y., & Zhu, H. (2016). "...and the Cross-Section of Expected Returns."
- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning."
- Aronson, D. (2007). "Evidence-Based Technical Analysis."
- Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies."
