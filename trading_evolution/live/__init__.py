"""
Live Trading module.

Generates real-time trading signals using evolved DNA.
"""

from .signal_generator import LiveSignalGenerator, LiveSignal, generate_daily_signals
from .watchlist import Watchlist, WatchlistItem, create_default_watchlist
from .alerts import AlertManager, Alert, AlertType, AlertPriority

__all__ = [
    'LiveSignalGenerator',
    'LiveSignal',
    'generate_daily_signals',
    'Watchlist',
    'WatchlistItem',
    'create_default_watchlist',
    'AlertManager',
    'Alert',
    'AlertType',
    'AlertPriority'
]
