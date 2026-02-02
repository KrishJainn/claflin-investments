# AQTIS - Adaptive Quantitative Trading Intelligence System
## Product Requirements Document (PRD) v1.0

---

**Author**: Claflin Investments
**Date**: February 2026
**Status**: MVP Complete
**Repository**: [github.com/KrishJainn/claflin-investments](https://github.com/KrishJainn/claflin-investments)

---

## 1. Executive Summary

AQTIS is a multi-agent AI-powered quantitative trading intelligence system designed for a solo quant trader operating in Indian equity markets (NIFTY 50). The system continuously learns from every trade, adapts to market regimes in real-time, and compounds trading knowledge through a persistent dual-layer memory system.

The core philosophy: **every trade teaches the system something, and that knowledge is never lost.**

AQTIS wraps and extends the existing `trading_evolution` genetic-algorithm framework, adding LLM-powered reasoning, six specialized agents, semantic memory, an ML model ensemble, and continuous learning loops.

---

## 2. Problem Statement

Solo quant traders face three persistent challenges:

1. **Knowledge Decay** - Lessons from past trades are forgotten; the same mistakes recur.
2. **Regime Blindness** - Strategies that work in trending markets get deployed in mean-reverting ones.
3. **Manual Overhead** - Research, backtesting, risk validation, and post-mortem analysis consume more time than actual trading.

AQTIS solves all three by creating an autonomous intelligence layer that remembers everything, detects regimes, and automates the full pre-trade to post-trade workflow.

---

## 3. System Architecture

```
                           +---------------------+
                           |   AQTIS CLI / UI    |
                           | (Click + Streamlit) |
                           +----------+----------+
                                      |
                           +----------v----------+
                           |    Orchestrator      |
                           | (Pre/Post/Daily)     |
                           +----------+----------+
                                      |
          +-------+-------+-------+---+---+-------+-------+
          |       |       |       |       |       |       |
     +----v-+ +--v---+ +-v----+ +v-----+ +v----+ +v-----+
     |Strat | |Back- | |Risk  | |Resear| |Post | |Pred  |
     |Gen   | |test  | |Mgr   | |cher  | |Mort.| |Track |
     +----+-+ +--+---+ +-+----+ ++-----+ ++----+ ++-----+
          |      |       |       |        |       |
          +------+---+---+-------+--------+-------+
                     |
          +----------v----------+
          |    Memory Layer     |
          | SQLite + ChromaDB   |
          +----------+----------+
                     |
          +----------v----------+
          |  ML Model Ensemble  |
          | RF + Linear + LSTM  |
          | + Rules-Based       |
          +---------------------+
```

### Technology Stack

| Component           | Technology                              |
|---------------------|-----------------------------------------|
| Language            | Python 3.11                             |
| Structured Storage  | SQLite (WAL mode)                       |
| Vector Storage      | ChromaDB (PersistentClient)             |
| Embeddings          | sentence-transformers (all-MiniLM-L6-v2)|
| LLM                 | Pluggable (Gemini 2.0 Flash default)    |
| ML Models           | scikit-learn, PyTorch                   |
| Regime Detection    | hmmlearn (HMM) + KMeans fallback        |
| Market Data         | yfinance (wraps trading_evolution)       |
| Indicators          | ta library (40+ indicators via bridge)  |
| CLI                 | Click + Rich                            |
| Dashboard           | Streamlit + Plotly                      |
| Research            | arXiv API                               |

---

## 4. The Six Agents

### 4.1 Strategy Generator Agent

**Purpose**: Identify trading opportunities and select the best strategy for current market conditions.

**Capabilities**:
- Regime-aware strategy selection from the active strategy pool
- LLM-powered signal analysis incorporating historical context
- Parameter variant generation for A/B testing
- New strategy proposals based on performance gaps

**Workflow**:
1. Receives a market signal (asset + action)
2. Retrieves current market regime from the regime detector
3. Pulls active strategies with regime-specific performance
4. Selects the highest-performing strategy for the current regime
5. Queries memory for 10 similar historical trades (semantic search)
6. Sends full context to LLM for analysis
7. Returns trade decision with confidence score

**Key Methods**:
- `analyze_signal(market_signal)` - Core decision method
- `propose_strategy_improvement(strategy_id)` - Performance-driven improvements
- `generate_parameter_variants(strategy_id, n)` - Creates parameter variations
- `propose_new_strategy(constraints)` - LLM-generated new strategies

---

### 4.2 Backtesting Agent

**Purpose**: Validate every trade against historical performance before execution.

**Capabilities**:
- Instant mini-backtests (30-day lookback) using similar trade statistics
- Rolling window backtests (90-day) for strategy degradation detection
- Shadow testing of parameter variants side-by-side

**Workflow**:
1. Receives a strategy + signal combination
2. Performs semantic search for 50 similar historical trades
3. Calculates: win rate, avg return, std deviation, confidence
4. Estimates expected hold duration
5. Flags degradation if recent Sharpe drops below 70% of historical

**Key Methods**:
- `instant_backtest(strategy, signal, lookback_days=30)` - Pre-trade validation
- `rolling_window_backtest(strategy, window_days=90)` - Degradation detection
- `shadow_test_variants(base_strategy, variants, signal)` - A/B testing

---

### 4.3 Risk Management Agent

**Purpose**: Enforce risk limits and calculate optimal position sizes.

**Default Risk Limits**:
| Parameter                  | Limit  |
|---------------------------|--------|
| Max Position Size          | 10%    |
| Max Portfolio Leverage     | 2.0x   |
| Max Daily Loss             | -5%    |
| Max Drawdown               | -15%   |
| Max Correlated Exposure    | 30%    |
| Min Prediction Confidence  | 60%    |

**Validation Checks** (in order):
1. Confidence >= 60% threshold
2. Daily loss not exceeding -5%
3. Portfolio drawdown not exceeding -15%
4. Position size within 10% of portfolio

**Position Sizing**:
- Fractional Kelly Criterion (quarter Kelly = 0.25x for safety)
- Uses win/loss statistics from similar historical trades
- Applies hard caps from risk limits configuration

**Circuit Breaker**: Emergency halt that blocks all new trades when activated. Triggered by catastrophic losses or manual intervention.

**Key Methods**:
- `validate_trade(proposed_trade)` - Risk limit checks
- `calculate_position_size(prediction, portfolio_value)` - Kelly-based sizing
- `activate_circuit_breaker(reason)` - Emergency stop
- `check_portfolio_risk(positions, portfolio_value)` - Portfolio-level metrics

---

### 4.4 Research Agent

**Purpose**: Continuously ingest quantitative finance research and make it searchable.

**Capabilities**:
- Daily scan of arXiv quantitative finance papers (category: q-fin)
- LLM-powered summarization and relevance scoring (0.0 - 1.0)
- Papers scoring >= 0.6 relevance are stored in vector DB
- Semantic search across the entire research knowledge base

**What the LLM Extracts from Each Paper**:
- Key findings relevant to algorithmic trading
- Applicable trading strategies mentioned
- Market conditions tested
- Implementation notes and limitations

**Key Methods**:
- `daily_research_scan()` - Fetch + score + store recent papers
- `search_for_solution(problem)` - Find relevant research for a specific problem
- `add_paper(paper)` - Manually add research to the knowledge base

---

### 4.5 Post-Mortem Agent

**Purpose**: Deep analysis of every completed trade to extract lessons.

**Analysis Components**:
1. **Error Calculation**: Return prediction error, direction correctness, confidence calibration error
2. **Peer Comparison**: Compares trade outcome against 20 similar historical trades (avg return, win rate percentile)
3. **LLM Insights**: Why the trade worked/failed, primary factors (model/execution/randomness), actionable changes
4. **Lesson Storage**: Stores analysis as a trade pattern in vector DB for future retrieval

**Key Methods**:
- `analyze_trade(trade_id)` - Deep dive on a single trade
- `weekly_performance_review()` - Aggregate weekly insights (runs on Mondays)
- `extract_lessons(lookback_days)` - Pattern identification across trades

---

### 4.6 Prediction Tracking Agent

**Purpose**: Track every prediction vs. actual outcome and maintain calibration.

**Confidence Calibration**:
- 10 bins (0-10%, 10-20%, ..., 90-100%)
- Tracks actual win rate in each bin
- Blends model confidence with observed rate using shrinkage
- Requires 10+ predictions per bin for reliable calibration

**Model Degradation Detection**:
- Compares recent accuracy (30 days) vs. historical baseline (90 days prior)
- Alerts if recent accuracy falls below 80% of historical
- Recommends retraining or weight reduction

**Dynamic Ensemble Weight Updates**:
- Calculates per-model accuracy over configurable lookback
- Applies softmax-style normalization
- Smooth blending: `new_weight = 0.7 * old + 0.3 * accuracy_based`

**Key Methods**:
- `record_prediction(prediction)` - Log new prediction
- `record_outcome(prediction_id, outcome)` - Update with actual results
- `get_calibrated_confidence(raw_confidence)` - Adjust for calibration
- `detect_model_degradation(lookback_days)` - Monitor accuracy trends
- `get_model_weights(lookback_days)` - Calculate ensemble weights

---

## 5. Orchestrator Workflows

### 5.1 Pre-Trade Workflow

Triggered when a market signal arrives.

```
Market Signal
  |
  v
Strategy Generator --> selects strategy, LLM analysis
  |
  v
Backtester --> instant backtest on similar trades
  |
  v
Risk Manager --> validate limits + Kelly position sizing
  |
  v
Prediction Tracker --> log prediction
  |
  v
Decision: EXECUTE / SKIP / REJECT
```

**Decision States**:
- `execute` - All checks passed, trade parameters returned
- `skip` - No viable opportunity identified
- `reject` - Failed risk checks (with reason)
- `error` - System error occurred

### 5.2 Post-Trade Workflow

Triggered when a trade closes.

```
Trade Outcome
  |
  v
Prediction Tracker --> record actual vs predicted
  |
  v
Post-Mortem --> deep analysis + LLM insights
  |
  v
Memory --> store lessons as patterns
  |
  v
Strategy Generator --> queue improvements (if actionable)
```

### 5.3 Daily Routine

Runs at end of each trading day.

```
Daily Trigger
  |
  v
Research Agent --> scan arXiv for new papers
  |
  v
Prediction Tracker --> detect model degradation
  |
  v
Backtester --> rolling backtests for all active strategies
  |
  v
Post-Mortem --> weekly review (Mondays only)
```

---

## 6. Memory Layer

### 6.1 Structured Database (SQLite)

Five core tables:

**trades** (24 columns):
- Identity: trade_id, timestamp, asset, strategy_id, action
- Execution: entry_price, exit_price, position_size, leverage
- Context: market_regime, vix_level, sector_rotation_score
- Outcome: pnl, pnl_percent, max_favorable_excursion, max_adverse_excursion
- Meta: hold_duration_seconds, slippage, execution_venue, prediction_id, notes

**predictions** (21+ columns):
- Identity: prediction_id, trade_id, timestamp, strategy_id, asset
- Predicted: predicted_return, predicted_confidence, predicted_hold_seconds, win_probability
- Model: model_ensemble_weights, primary_model, feature_importance, market_features
- Actual: actual_return, actual_hold_seconds, was_profitable
- Error: return_prediction_error, direction_correct

**strategies** (17 columns):
- Identity: strategy_id, strategy_name, strategy_type, description
- Formula: mathematical_formula, parameters (JSON)
- Performance: total_trades, win_rate, sharpe_ratio, sortino_ratio, max_drawdown
- Comparison: backtest_sharpe, live_sharpe
- Regime: performance_by_regime (JSON)
- Status: is_active

**market_state** (13 columns):
- Volatility: vix, realized_vol_20d, vol_regime
- Trend: spy_trend_strength, sector_rotation
- Breadth: breadth_indicators, avg_bid_ask_spread, market_depth_score
- Correlation: asset_correlation_matrix
- Events: upcoming_events

**risk_events** (6 columns):
- Identity: event_id, timestamp, event_type
- Details: reason, portfolio_state, details

### 6.2 Vector Store (ChromaDB)

Two collections:

**trading_research**:
- Documents: Research paper abstracts and summaries
- Metadata: title, authors, URL, relevance_score, key_findings
- Use case: "Find papers about momentum crashes" returns semantically relevant research

**trade_patterns**:
- Documents: Natural language trade descriptions and post-mortem analyses
- Metadata: strategy_id, outcome, pnl, market_regime
- Use case: "BUY RELIANCE.NS in trending_up regime" returns similar historical trades

### 6.3 Memory Layer Facade

Unified interface combining both backends:

- `store_trade(trade, prediction)` - Writes to SQLite + generates embedding for ChromaDB
- `get_similar_trades(current_setup, top_k)` - Semantic search + DB enrichment
- `update_trade_outcome(trade_id, outcome)` - Updates trade and recalculates prediction errors
- `store_research(paper)` / `search_research(query)` - Vector-backed research ops
- `get_stats()` - Combined statistics from both backends

**Trade-to-Description Conversion**:
```
"BUY RELIANCE.NS using momentum_strategy. Market regime: trending_up.
 Entry: 2450.50. Exit: 2475.20. Return: 1.01%. VIX: 15.3."
```
This natural language description is embedded and stored for future similarity search.

---

## 7. ML Model Ensemble

### 7.1 Individual Models

| Model            | Type        | Config                          | Default Weight |
|-----------------|-------------|----------------------------------|----------------|
| Random Forest   | Non-linear  | 200 trees, max_depth=10, 5-fold CV | 20%        |
| Linear (Ridge)  | Baseline    | alpha=1.0, StandardScaler        | 15%            |
| LSTM            | Sequential  | 2 layers, 64 hidden, seq_len=60  | 25%            |
| Rules-Based     | Heuristic   | RSI/MACD/ADX signal rules         | 40%            |

### 7.2 Ensemble Prediction

```
ensemble_prediction = sum(model_weight[i] * model_prediction[i])
confidence = 1.0 - std(predictions) / (|mean(predictions)| + epsilon)
```

### 7.3 Dynamic Weight Updates

After each prediction outcome:
1. Calculate per-model accuracy over the last 30 days
2. Apply softmax normalization across models
3. Blend: `new = 0.7 * current_weight + 0.3 * accuracy_weight`

### 7.4 Rules-Based Signals

```
RSI < 30  -->  +2% predicted return  (oversold / bullish)
RSI > 70  -->  -2% predicted return  (overbought / bearish)
MACD > 0  -->  +1% momentum signal
ADX > 25  -->  amplify signal by 1.5x (strong trend)
```

### 7.5 Market Regime Detection

5 regimes detected using Gaussian HMM (hmmlearn) with KMeans fallback:

| Regime         | Characteristics                    |
|---------------|------------------------------------|
| low_volatility | Narrow ranges, low vol             |
| trending_up    | Sustained upward momentum          |
| mean_reverting | Range-bound, oscillating           |
| trending_down  | Sustained downward momentum        |
| high_volatility| Large swings, elevated vol         |

Features used: returns (1d/5d/20d), 20d volatility, trend strength, mean-reversion score.

Rule-based fallback when models aren't trained:
- vol > 0.25 --> high_volatility
- trend > 0.6 --> trending_up / trending_down
- mean_rev > 0.6 --> mean_reverting
- else --> low_volatility

---

## 8. CLI Reference

### Top-Level
```bash
python3 -m aqtis.cli.main [--config PATH] COMMAND
```

### Trade Commands
```bash
aqtis trade analyze SYMBOL        # Run pre-trade analysis
aqtis trade execute SYMBOL        # Execute trade (with --size, --action)
```

### Portfolio
```bash
aqtis portfolio                   # View trade stats and active strategies
```

### Strategy Commands
```bash
aqtis strategy list               # List all strategies (active/inactive)
aqtis strategy backtest ID        # Run 90-day rolling backtest
```

### Memory Commands
```bash
aqtis memory search QUERY         # Semantic search across trades
aqtis memory stats                # Database statistics (JSON)
```

### Research Commands
```bash
aqtis research scan               # Daily arXiv paper scan
aqtis research query QUERY        # Search research knowledge base
```

### System Commands
```bash
aqtis system check                # Full health check
aqtis system calibrate            # Recalibrate prediction confidence
aqtis system daily                # Run complete daily routine
```

### Entry Point
```bash
python3 -m aqtis.run                       # Status overview
python3 -m aqtis.run --analyze RELIANCE.NS # Pre-trade analysis
python3 -m aqtis.run --daily               # Full daily routine
python3 -m aqtis.run --dashboard           # Launch Streamlit
python3 -m aqtis.run --health              # System health check
```

---

## 9. Dashboard

Six-page Streamlit dashboard launched via `python3 -m aqtis.run --dashboard` or `streamlit run aqtis/dashboard/app.py`.

### Pages

1. **Overview** - Key metrics (trades, predictions, strategies), recent trades table, cumulative P&L equity curve, vector store stats
2. **Strategy Performance** - Strategy selector, per-strategy metrics (win rate, Sharpe, drawdown), parameters, trade history
3. **Prediction Analysis** - Directional accuracy, avg return error, recent predictions table, confidence calibration chart (predicted vs actual win rate)
4. **Risk Monitor** - Risk events (last 30 days), expandable event details, today's P&L, circuit breaker status
5. **Research** - Total papers count, semantic search box, results with title/relevance/preview
6. **Memory Explorer** - Similar trade search, full database statistics

---

## 10. Configuration

All settings in `aqtis_config.yaml`:

```yaml
system:
  mode: simulation          # simulation | live
  db_path: aqtis.db
  vector_store_path: aqtis_vectors
  log_level: INFO

risk:
  max_position_size: 0.10
  max_portfolio_leverage: 2.0
  max_daily_loss: -0.05
  max_drawdown: -0.15
  max_correlated_exposure: 0.30
  min_prediction_confidence: 0.60

backtesting:
  instant_backtest_lookback_days: 30
  rolling_window_days: 90
  min_similar_trades: 20
  max_similar_trades: 50

models:
  ensemble_weights:
    lstm: 0.25
    random_forest: 0.20
    linear_regression: 0.15
    rules_based: 0.40
  retrain_frequency: weekly
  min_training_samples: 100
  sequence_length: 60

memory:
  vector_db: chromadb
  embedding_model: all-MiniLM-L6-v2
  max_similar_trades: 50
  research_relevance_threshold: 0.6

llm:
  provider: gemini
  model: gemini-2.0-flash
  temperature: 0.3
  max_tokens: 4000
  timeout: 30

market_data:
  provider: yahoo
  symbols:              # 20 NIFTY 50 stocks
    - RELIANCE.NS
    - TCS.NS
    - HDFCBANK.NS
    - INFY.NS
    - ICICIBANK.NS
    - HINDUNILVR.NS
    - ITC.NS
    - SBIN.NS
    - BHARTIARTL.NS
    - BAJFINANCE.NS
    - KOTAKBANK.NS
    - LT.NS
    - HCLTECH.NS
    - AXISBANK.NS
    - ASIANPAINT.NS
    - MARUTI.NS
    - SUNPHARMA.NS
    - TITAN.NS
    - ULTRACEMCO.NS
    - WIPRO.NS
  data_years: 3
  cache_ttl_hours: 24

execution:
  order_type: market
  slippage_estimate: 0.001
  initial_capital: 100000

regime:
  n_regimes: 5
  lookback_days: 60
  update_frequency: daily
```

---

## 11. LLM Provider System

Pluggable architecture with abstract base class:

| Provider        | Status      | Notes                                    |
|----------------|-------------|------------------------------------------|
| Gemini         | Implemented | Default. Requires GEMINI_API_KEY or GOOGLE_API_KEY |
| MockLLMProvider| Implemented | Testing fallback, configurable responses  |
| Claude/OpenAI  | Extensible  | Implement `LLMProvider._call()` method    |

**Features**:
- Automatic retry with exponential backoff (3 attempts)
- JSON extraction from markdown code blocks
- Graceful degradation to MockLLMProvider when no API key is set

---

## 12. Integration with trading_evolution

AQTIS operates as a layer on top of the existing `trading_evolution` framework:

**What AQTIS imports from trading_evolution**:
- `DataFetcher` - Market data retrieval (yfinance)
- `IndicatorCalculator` - 40+ technical indicators
- `IndicatorNormalizer` - Expanding-window normalization (no lookahead bias)
- `RiskManager` - Position sizing and validation
- `BacktestEngine` - Historical simulation

**Fallback behavior**: Every bridge module has a standalone implementation that activates if `trading_evolution` is not available, making AQTIS self-contained.

---

## 13. Data Flow: End-to-End Example

**Scenario**: BUY signal for RELIANCE.NS

```
1. Signal arrives: {asset: RELIANCE.NS, action: BUY}

2. Strategy Generator:
   - Detects regime: trending_up
   - Selects: momentum_strategy (Sharpe 2.1 in this regime)
   - LLM analysis with 10 similar trades from memory
   - Decision: should_trade=true, confidence=0.72

3. Backtester:
   - 50 similar trades found
   - Win rate: 58%, avg return: +1.2%
   - Backtest confidence: 0.68

4. Risk Manager:
   - Confidence 0.72 >= 0.60 threshold      PASS
   - Daily loss -2% > -5% limit             PASS
   - Drawdown -8% > -15% limit              PASS
   - Kelly position size: 14,000 (14% capped at 10%)

5. Prediction logged: predicted_return=1.2%, confidence=0.72

6. EXECUTE: 4 shares of RELIANCE.NS at 2451.00

--- Trade closes 3.2 hours later at 2475.50 (+1.0%) ---

7. Post-Trade:
   - Actual: +1.0% vs predicted +1.2% (error: 0.2%)
   - Direction: correct
   - Post-Mortem: "Momentum continuation confirmed. Exit slightly early."
   - Lesson stored in vector DB for future reference

8. Calibration bin 70-80% updated: wins += 1
```

---

## 14. Testing

58 tests across 4 test files:

| Test File           | Tests | Coverage                                    |
|--------------------|-------|---------------------------------------------|
| test_memory.py     | 17    | SQLite CRUD, ChromaDB ops, MemoryLayer facade, Config |
| test_agents.py     | 18    | All 6 agents with MockLLMProvider           |
| test_models.py     | 11    | RF, Linear, Regime Detector, Ensemble       |
| test_orchestrator.py| 7    | Pre-trade, post-trade, daily routine workflows |
| **Total**          | **58**| **All passing**                             |

---

## 15. File Inventory

```
aqtis/
  __init__.py                    # Package init, version 0.1.0
  run.py                         # End-to-end entry point

  config/
    __init__.py
    settings.py                  # 10 dataclasses, YAML loader, validation

  memory/
    __init__.py
    database.py                  # SQLite structured storage (5 tables)
    vector_store.py              # ChromaDB vector storage (2 collections)
    memory_layer.py              # Unified facade

  llm/
    __init__.py
    base.py                      # Abstract LLMProvider + MockLLMProvider
    gemini_provider.py           # Google Gemini implementation
    embeddings.py                # sentence-transformers wrapper

  agents/
    __init__.py
    base.py                      # Abstract BaseAgent
    strategy_generator.py        # Strategy selection + LLM analysis
    backtester.py                # Instant + rolling backtests
    risk_manager.py              # Risk limits + Kelly sizing + circuit breaker
    researcher.py                # arXiv paper scanning
    post_mortem.py               # Trade analysis + lesson extraction
    prediction_tracker.py        # Calibration + degradation detection

  orchestrator/
    __init__.py
    orchestrator.py              # Pre-trade, post-trade, daily workflows

  models/
    __init__.py
    rf_model.py                  # Random Forest predictor
    linear_model.py              # Ridge/Lasso predictor
    lstm_model.py                # PyTorch LSTM predictor
    ensemble.py                  # Weighted ensemble + rules-based
    regime_detector.py           # HMM + KMeans regime detection

  data/
    __init__.py
    market_data.py               # yfinance bridge (wraps DataFetcher)

  features/
    __init__.py
    indicator_bridge.py          # Technical indicator bridge (wraps IndicatorCalculator)

  cli/
    __init__.py
    main.py                      # Click CLI (6 command groups)

  dashboard/
    app.py                       # Streamlit dashboard (6 pages)

  tests/
    __init__.py
    test_memory.py               # 17 tests
    test_agents.py               # 18 tests
    test_models.py               # 11 tests
    test_orchestrator.py         # 7 tests (pipeline integration)

aqtis_config.yaml                # Default configuration
```

**Total**: 41 files, ~6,900 lines of code

---

## 16. Dependencies

```
# Core
pyyaml
numpy
pandas
scikit-learn

# Memory
chromadb
sentence-transformers

# ML Models
torch
hmmlearn

# LLM
google-generativeai

# Market Data
yfinance
ta

# CLI
click
rich

# Dashboard
streamlit
plotly

# Research
arxiv

# Testing
pytest
```

---

## 17. Environment Variables

| Variable         | Required | Description                          |
|-----------------|----------|--------------------------------------|
| GEMINI_API_KEY  | No*      | Google Gemini API key                |
| GOOGLE_API_KEY  | No*      | Alternative Gemini API key           |

*System runs with MockLLMProvider when no key is set. LLM-dependent features (strategy analysis, paper summarization, post-mortem insights) will use fallback responses.

---

## 18. Future Roadmap

### Phase 4 (Not Yet Built)
- **Live Paper Trading**: Broker API integration (Zerodha Kite)
- **Real-time Streaming**: WebSocket market data feeds
- **Advanced Regime Detection**: Transformer-based regime classification
- **Multi-Asset Correlation**: Cross-asset signal generation
- **Alerting System**: Telegram/Discord notifications
- **Model Registry**: MLflow integration for experiment tracking
- **Distributed Backtesting**: Parallel backtest execution

---

*Built by Claflin Investments. Powered by AQTIS.*
