"""
Data module.

Handles data fetching, caching, and market regime detection.
"""

from .fetcher import DataFetcher
from .cache import DataCache
from .market_regime import (
    RegimeDetector, MarketRegime, RegimeSnapshot,
    detect_regime_for_timestamp, get_regime_at_timestamps
)

__all__ = [
    'DataFetcher',
    'DataCache',
    'RegimeDetector',
    'MarketRegime',
    'RegimeSnapshot',
    'detect_regime_for_timestamp',
    'get_regime_at_timestamps'
]
