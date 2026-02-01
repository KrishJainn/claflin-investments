"""
Backtest module for the Trading Evolution system.

Provides:
- Deterministic, event-driven backtest engine
- Indian market cost model (STT, brokerage, GST, etc.)
- Performance evaluation with regime analysis
- Reproducibility framework
"""

from .indian_costs import IndianCostModel, CostBreakdown
from .engine import BacktestEngine, BacktestConfig
from .result import BacktestResult, Trade
from .evaluation import Evaluator, PerformanceMetrics

__all__ = [
    'IndianCostModel',
    'CostBreakdown',
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'Trade',
    'Evaluator',
    'PerformanceMetrics',
]
