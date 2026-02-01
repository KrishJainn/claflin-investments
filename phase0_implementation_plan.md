# Phase 0: Data + Backtest Truth Layer

## Implementation Plan
**Goal**: Build a trustworthy, deterministic backtesting foundation for NIFTY 50 intraday trading before adding AI.

---

## Current State Analysis

### What Exists ✅
| Component | File | Status |
|-----------|------|--------|
| Data Fetcher | `data/fetcher.py` | Daily bars only, needs intraday |
| Market Regime | `data/market_regime.py` | Good, needs integration |
| Execution Engine | `player/execution.py` | Basic slippage, needs Indian costs |
| Risk Manager | `player/risk_manager.py` | Position sizing, needs daily limits |
| Backtest Scripts | `backtest_strategies.py` | Works but not deterministic |

### What's Missing ❌
- **Universe Manager**: No handling of NIFTY 50 index changes over time
- **Intraday Data**: Only daily bars, need 1-5 min bars
- **Indian Cost Model**: No STT, exchange fees, GST, stamp duty
- **Deterministic Engine**: No config hashing, random seeds
- **Daily Risk Limits**: No max trades/day, daily stop loss, cooldown
- **Reproducibility**: Can't reproduce exact backtest runs

---

## Proposed Changes

### 1. Universe Manager

#### [NEW] [universe.py](file:///Users/krishjain/Desktop/New%20Project/trading_evolution/data/universe.py)

Manages NIFTY 50 constituents with historical changes:

```python
class UniverseManager:
    """
    Manages NIFTY 50 constituents by date.
    
    Features:
    - Historical constituent changes (stocks enter/exit index)
    - Corporate action adjustments (splits, bonuses)
    - Symbol validation and mapping
    """
    
    def get_constituents(self, date: datetime) -> List[str]
    def get_constituent_history(self) -> Dict[str, List[Tuple[date, date]]]
    def apply_corporate_actions(self, df: DataFrame, symbol: str) -> DataFrame
```

**Data Sources**:
- NSE India website for current constituents
- Historical index changes from archived data
- Corporate actions from yfinance/NSE

---

### 2. Intraday Data Module

#### [NEW] [intraday.py](file:///Users/krishjain/Desktop/New%20Project/trading_evolution/data/intraday.py)

Fetches and manages intraday bars:

```python
@dataclass
class IntradayConfig:
    interval: str = "5m"  # 1m, 5m, 15m
    timezone: str = "Asia/Kolkata"
    market_open: time = time(9, 15)
    market_close: time = time(15, 30)
    pre_market_start: time = time(9, 0)
    
class IntradayDataFetcher:
    """
    Fetches intraday data with IST timezone handling.
    
    Features:
    - 1m, 5m, 15m bar support
    - IST timezone conversion
    - Gap handling (market holidays)
    - Corporate action adjustment
    """
    
    def fetch_intraday(self, symbol: str, days: int = 60) -> DataFrame
    def aggregate_bars(self, df: DataFrame, target_interval: str) -> DataFrame
    def validate_trading_hours(self, df: DataFrame) -> DataFrame
```

> [!WARNING]  
> yfinance has limited intraday history (~60 days). For longer history, need alternative data source or cached data.

---

### 3. Indian Market Cost Model

#### [NEW] [indian_costs.py](file:///Users/krishjain/Desktop/New%20Project/trading_evolution/backtest/indian_costs.py)

Complete Indian market transaction costs:

```python
@dataclass
class IndianCostModel:
    """
    Indian equity market transaction costs (NSE).
    
    Components:
    - Brokerage: Fixed or percentage (broker-specific)
    - STT (Securities Transaction Tax): 0.1% on sell for delivery
    - Exchange Transaction Charges: 0.00345% (NSE)
    - SEBI Turnover Fee: 0.0001%
    - GST: 18% on brokerage + exchange charges
    - Stamp Duty: State-specific (0.003% - 0.015%)
    - Slippage: Market impact model
    """
    
    # Brokerage
    brokerage_pct: float = 0.0003  # 0.03% (discount broker)
    min_brokerage: float = 20.0    # ₹20 minimum
    
    # Statutory charges
    stt_sell_pct: float = 0.001    # 0.1% on sell (delivery)
    stt_buy_pct: float = 0.0       # No STT on buy (delivery)
    exchange_txn_pct: float = 0.0000345  # NSE
    sebi_fee_pct: float = 0.000001  # Negligible
    stamp_duty_pct: float = 0.00015  # 0.015% (varies by state)
    gst_pct: float = 0.18          # 18% on brokerage+exchange
    
    # Slippage model
    base_slippage_pct: float = 0.0005  # 0.05%
    volatility_multiplier: float = 1.0
    
    def calculate_entry_cost(self, price: float, qty: int) -> CostBreakdown
    def calculate_exit_cost(self, price: float, qty: int) -> CostBreakdown
    def calculate_round_trip(self, entry: float, exit: float, qty: int) -> CostBreakdown
```

**Cost Breakdown Example** (₹100,000 trade):
| Component | Entry | Exit | Total |
|-----------|-------|------|-------|
| Brokerage | ₹30 | ₹30 | ₹60 |
| STT | ₹0 | ₹100 | ₹100 |
| Exchange | ₹3.45 | ₹3.45 | ₹6.90 |
| GST | ₹6.02 | ₹6.02 | ₹12.04 |
| Stamp Duty | ₹15 | ₹0 | ₹15 |
| Slippage | ₹50 | ₹50 | ₹100 |
| **Total** | **₹104.47** | **₹189.47** | **₹293.94** |

---

### 4. Deterministic Backtest Engine

#### [NEW] [backtest/engine.py](file:///Users/krishjain/Desktop/New%20Project/trading_evolution/backtest/engine.py)

Event-driven, deterministic backtest engine:

```python
@dataclass
class BacktestConfig:
    """Immutable backtest configuration."""
    version: str
    random_seed: int = 42
    start_date: date
    end_date: date
    initial_capital: float = 1_000_000.0
    
    # Risk limits
    max_positions: int = 5
    max_trades_per_day: int = 10
    daily_loss_limit_pct: float = 0.02  # 2%
    cooldown_after_loss: int = 0  # bars
    
    # Execution
    cost_model: str = "indian"
    fill_assumption: str = "next_bar_open"  # or "same_bar_close"
    
    # End of day
    flatten_eod: bool = True
    flatten_time: time = time(15, 20)  # 10 min before close

    def config_hash(self) -> str:
        """Generate deterministic hash for reproducibility."""

class BacktestEngine:
    """
    Deterministic, event-driven backtest engine.
    
    Features:
    - Bar-by-bar processing
    - Limit/market order fills
    - End-of-day flatten
    - Position sizing with risk limits
    - Daily stop loss
    - Max trades per day
    - Cooldown periods
    - Reproducible with config hash
    """
    
    def run(self, 
            strategy: Strategy, 
            data: Dict[str, DataFrame],
            config: BacktestConfig) -> BacktestResult
    
    def run_single_symbol(self,
                          strategy: Strategy,
                          data: DataFrame,
                          config: BacktestConfig) -> BacktestResult
```

#### [NEW] [backtest/result.py](file:///Users/krishjain/Desktop/New%20Project/trading_evolution/backtest/result.py)

```python
@dataclass
class BacktestResult:
    """Complete backtest result with full reproducibility info."""
    
    # Identification
    run_id: str
    config_hash: str
    timestamp: datetime
    
    # Trade log
    trades: List[Trade]
    
    # Metrics
    metrics: PerformanceMetrics
    
    # Daily tracking
    daily_pnl: DataFrame
    equity_curve: DataFrame
    
    # Reproducibility
    strategy_version: str
    data_hash: str
    
    def save(self, path: str) -> None
    def load(cls, path: str) -> 'BacktestResult'
    def compare(self, other: 'BacktestResult') -> ComparisonReport
```

---

### 5. Enhanced Risk Limits

#### [MODIFY] [player/risk_manager.py](file:///Users/krishjain/Desktop/New%20Project/trading_evolution/player/risk_manager.py)

Add intraday risk controls:

```python
@dataclass
class IntradayRiskLimits:
    """Intraday-specific risk limits."""
    max_trades_per_day: int = 10
    daily_loss_limit_pct: float = 0.02  # 2% of capital
    cooldown_bars_after_loss: int = 5
    max_exposure_pct: float = 0.50  # 50% max invested
    no_new_trades_after: time = time(15, 0)
    
class IntradayRiskManager(RiskManager):
    """Extended risk manager with intraday controls."""
    
    def check_daily_limits(self) -> Tuple[bool, str]
    def register_trade(self, trade: Trade) -> None
    def reset_daily_state(self) -> None
    def is_in_cooldown(self) -> bool
```

---

### 6. Evaluation Module

#### [NEW] [backtest/evaluation.py](file:///Users/krishjain/Desktop/New%20Project/trading_evolution/backtest/evaluation.py)

Rolling metrics with regime buckets:

```python
@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    
    # Returns
    total_return: float
    annualized_return: float
    
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade_duration: timedelta
    
    # Expectancy
    expectancy: float
    expectancy_ratio: float
    
class Evaluator:
    """
    Evaluation with rolling metrics and regime analysis.
    """
    
    def calculate_metrics(self, trades: List[Trade], equity: DataFrame) -> PerformanceMetrics
    
    def rolling_metrics(self, 
                        equity: DataFrame, 
                        window: int = 30) -> DataFrame
    
    def regime_performance(self,
                           trades: List[Trade],
                           regimes: Series) -> Dict[str, PerformanceMetrics]
    
    def generate_report(self, result: BacktestResult) -> EvaluationReport
```

**Regime Buckets**:
| Regime | Description |
|--------|-------------|
| Trending Up | ADX > 25, positive slope |
| Trending Down | ADX > 25, negative slope |
| Ranging | ADX < 20 |
| High Volatility | Vol percentile > 80% |
| Low Volatility | Vol percentile < 20% |

---

### 7. Reproducibility Framework

#### [NEW] [backtest/reproducibility.py](file:///Users/krishjain/Desktop/New%20Project/trading_evolution/backtest/reproducibility.py)

```python
class ReproducibilityManager:
    """
    Ensures backtest runs are fully reproducible.
    """
    
    def generate_run_id(self) -> str
    
    def hash_config(self, config: BacktestConfig) -> str
    
    def hash_data(self, data: Dict[str, DataFrame]) -> str
    
    def hash_strategy(self, strategy: Strategy) -> str
    
    def save_run_manifest(self, 
                          result: BacktestResult,
                          path: str) -> None
    
    def verify_reproducibility(self,
                               result1: BacktestResult,
                               result2: BacktestResult) -> bool
```

---

## New Directory Structure

```
trading_evolution/
├── backtest/                    # [NEW] Backtest module
│   ├── __init__.py
│   ├── engine.py               # Deterministic backtest engine
│   ├── result.py               # BacktestResult dataclass
│   ├── evaluation.py           # Metrics and regime analysis
│   ├── indian_costs.py         # Indian market cost model
│   └── reproducibility.py      # Config hashing and verification
├── data/
│   ├── fetcher.py              # [EXISTING]
│   ├── intraday.py             # [NEW] Intraday data fetching
│   ├── universe.py             # [NEW] NIFTY 50 universe manager
│   ├── cache.py                # [EXISTING]
│   └── market_regime.py        # [EXISTING]
└── player/
    ├── risk_manager.py         # [MODIFY] Add intraday limits
    └── ...
```

---

## Verification Plan

### Automated Tests

```bash
# Test 1: Reproducibility check
python -m pytest tests/test_backtest_reproducibility.py

# Test 2: Cost model validation
python -m pytest tests/test_indian_costs.py

# Test 3: Frozen baseline strategy
python run_frozen_baseline.py --verify
```

### Manual Verification

1. **Run same backtest twice** → Identical results (hash match)
2. **Compare cost model** → Match with actual broker statements
3. **Regime detection** → Visual inspection on known market periods
4. **Daily limits** → Force breach and verify halt

---

## Implementation Order

| Step | Component | Estimated LOC | Priority |
|------|-----------|---------------|----------|
| 1 | Indian Cost Model | ~200 | P0 |
| 2 | Backtest Engine | ~400 | P0 |
| 3 | BacktestResult + Metrics | ~300 | P0 |
| 4 | Intraday Data Fetcher | ~250 | P0 |
| 5 | Reproducibility Manager | ~150 | P0 |
| 6 | Universe Manager | ~200 | P1 |
| 7 | Intraday Risk Manager | ~150 | P1 |
| 8 | Evaluation Module | ~300 | P1 |

---

## Frozen Baseline Strategy

Once complete, run this command to generate your "frozen baseline":

```bash
python -m trading_evolution.backtest.run \
    --strategy baseline_super_indicator \
    --data-range 2023-01-01:2024-12-31 \
    --interval 5m \
    --cost-model indian \
    --output frozen_baseline_v1.json \
    --verify-reproducibility
```

Output:
```
✅ Backtest complete
   Config Hash: 8a3f2b1c
   Data Hash: 4e7d9c2a
   Run ID: baseline_20260128_143000
   
   Metrics:
   - Trades: 847
   - Win Rate: 52.3%
   - Net P&L: ₹127,450
   - Sharpe: 1.84
   - Max DD: 8.2%
   
   Reproducibility verified: ✅
```

---

## Open Questions

1. **Intraday Data Source**: yfinance has ~60 day limit. Options:
   - Use cached historical data
   - Alternative provider (TrueData, Dhan, etc.)
   - Start with daily bars, migrate to intraday later

2. **Index Constituent History**: Where to source NIFTY 50 historical changes?
   - Manual compilation
   - NSE archives
   - Third-party data providers

3. **Intraday vs EOD**: Should we:
   - Start with EOD and add intraday later? (faster)
   - Build intraday from scratch? (thorough)
