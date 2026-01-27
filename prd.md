# Product Requirements Document (PRD)
## Trading Evolution Engine

### 1. Executive Summary
**Product Name:** Trading Evolution Engine  
**Vision:** To democratize quantitative hedge-fund grade strategy discovery by using evolutionary biology concepts to automate the creation of trading algorithms. Instead of humans hand-crafting strategies, the system "evolves" optimal configurations (DNA) through survival of the fittest.

### 2. Problem Statement
- **Human Bias:** Traders rely on favorite indicators (RSI, MACD) without statistical validation.
- **Complexity:** There are thousands of possible indicator combinations; testing them manually is impossible.
- **Overfitting:** Traditional backtesting often curve-fits to past data rather than finding robust generalizable patterns.

### 3. Solution Overview
A generic algorithm-based engine that treats trading strategies as biological organisms ("DNA"). A population of strategies competes in a simulated market environment. The fittest strategies propagate their genes (indicator weights) to the next generation, while weaker ones go extinct.

### 4. Core Features

#### 4.1. The Genome (Super Indicator DNA)
- **Structure:** Each capability is defined by a "DNA" object containing:
  - **Active Indicators:** Which technical indicators to listen to (e.g., RSI_14, BB_UPPER).
  - **Weights:** How much importance to assign to each indicator (-1.0 to +1.0).
  - **Genetics:** Mutation parameters and lineage tracking.

#### 4.2. Indicator Universe & Normalization
- **Universe:** 82+ technical indicators including Momentum, Trend, Volatility, and Volume.
- **Smart Normalization:** Converts raw values (e.g., Price $2000) into standardized Z-scores or Percentiles (0 to 1) to make them mathematically comparable in a weighted sum.
- **Vectorized Calculation:** Pre-computes all indicators using expanding windows to allow O(N) simulation speed.

#### 4.3. Evolutionary Engine
- **Selection:** Tournament selection to pick parents.
- **Crossover:** Blending traits of two successful strategies to create offspring.
- **Mutation:** Randomly adjusting weights to introduce innovation (exploration).
- **Elitism:** Preserving the top 5% of strategies unchanged.

#### 4.4. Simulation & Risk Management
- **Paper Trading:** Simulates bar-by-bar execution.
- **Risk Controls:**
  - ATR-based Stop Loss and Take Profit.
  - Position Sizing based on volatility.
  - Max Drawdown constraints.

#### 4.5. The "Coach" (Meta-Optimization)
- An analytical layer that observes the evolution process.
- Identifies patterns in successful DNA (e.g., "Top 10 strategies all use ADX").
- Provides "guidance" to the next generation (e.g., "increase mutation rate on Volatility indicators").

### 5. Technical Architecture
- **Language:** Python 3.11+
- **Data Source:** `yfinance` (Yahoo Finance API) for Nifty 50 / Global equities.
- **Core Libraries:**
  - `pandas` / `numpy`: Data manipulation.
  - `ta`: Technical Analysis library.
- **Performance:** Optimized Vectorized Backtesting (simulates 50 strategies over 2 years in <30 seconds).

### 6. Success Metrics
- **Sharpe Ratio:** > 2.0 (Risk-adjusted return).
- **Win Rate:** > 55%.
- **Robustness:** Performance consistency across Training, Validation, and Holdout datasets.
- **Speed:** Full evolution cycle (50 generations) under 30 minutes on local hardware.

### 7. Roadmap
- **Phase 1 (MVP):** Evolution on Nifty 50 with basic indicators. [Status: LIVE]
- **Phase 2:** Multi-asset portfolio optimization (Correlation management).
- **Phase 3:** Live Broker Integration (Zeroodha/Interactive Brokers).
- **Phase 4:** Deep Reinforcement Learning (RL) agent as the "Signal Generator".
