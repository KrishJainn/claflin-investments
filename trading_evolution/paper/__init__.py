"""
Paper Trading Module.

Provides live paper trading capability:
- Live market data ingest with bar building
- Paper execution simulator with realistic fills
- Trade ledger with strategy versioning
- Risk management enforced in code
- Comprehensive event logging
"""

from .live_data import LiveDataManager, BarBuilder, Bar, MarketStatus
from .paper_executor import PaperExecutor, PaperFill, OrderSide, OrderType, Order
from .trade_ledger import TradeLedger, LedgerEntry, DailyStats, TradeReason
from .paper_risk_manager import PaperRiskManager, RiskLimits, RiskState, RiskAction
from .paper_trader import PaperTrader, PaperTraderConfig, BEST_STRATEGY
from .strategies import STRATEGIES, get_strategy, get_default_strategy, StrategyConfig

__all__ = [
    # Live data
    'LiveDataManager', 'BarBuilder', 'Bar', 'MarketStatus',
    # Execution
    'PaperExecutor', 'PaperFill', 'OrderSide', 'OrderType', 'Order',
    # Ledger
    'TradeLedger', 'LedgerEntry', 'DailyStats', 'TradeReason',
    # Risk
    'PaperRiskManager', 'RiskLimits', 'RiskState', 'RiskAction',
    # Trader
    'PaperTrader', 'PaperTraderConfig', 'BEST_STRATEGY',
    # Strategies
    'STRATEGIES', 'get_strategy', 'get_default_strategy', 'StrategyConfig',
]
