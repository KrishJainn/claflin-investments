"""
AQTIS Backtest Module.

Provides token-efficient simulation runners for backtesting
strategies against Yahoo Finance historical data.
"""

from .simulation_runner import SimulationRunner
from .paper_trader import PaperTrader

__all__ = ["SimulationRunner", "PaperTrader"]
