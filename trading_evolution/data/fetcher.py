"""
Data fetching module using yfinance.

Fetches historical OHLCV data for stocks with caching support.
"""

import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .cache import DataCache

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches historical market data from Yahoo Finance.

    Features:
    - Automatic caching with Parquet
    - Batch fetching for multiple symbols
    - Data validation and cleaning
    - Proper OHLCV column naming
    """

    def __init__(self, cache: DataCache = None, cache_dir: Path = None):
        """
        Initialize data fetcher.

        Args:
            cache: Optional DataCache instance
            cache_dir: Directory for cache (creates DataCache if cache not provided)
        """
        if cache:
            self.cache = cache
        elif cache_dir:
            self.cache = DataCache(cache_dir)
        else:
            self.cache = DataCache(Path("data_cache"))

    def fetch(self, symbol: str, start_date: str = None, end_date: str = None,
              years: int = 3, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single symbol.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD) or None for years ago
            end_date: End date (YYYY-MM-DD) or None for today
            years: Number of years to fetch if start_date not provided
            use_cache: Whether to use cache

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        """
        # Calculate dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start = datetime.now() - timedelta(days=years * 365)
            start_date = start.strftime('%Y-%m-%d')

        # Check cache
        if use_cache:
            cached = self.cache.get(symbol, start_date, end_date)
            if cached is not None:
                logger.debug(f"Cache hit for {symbol}")
                return cached

        # Fetch from yfinance
        logger.info(f"Fetching {symbol} from {start_date} to {end_date}")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Clean and standardize columns
            df = self._clean_data(df)

            # Cache the result
            if use_cache and df is not None:
                self.cache.put(symbol, start_date, end_date, df)

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def fetch_multiple(self, symbols: List[str], start_date: str = None,
                       end_date: str = None, years: int = 3,
                       use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of stock tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            years: Number of years to fetch if start_date not provided
            use_cache: Whether to use cache

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        failed = []

        for symbol in symbols:
            df = self.fetch(
                symbol,
                start_date=start_date,
                end_date=end_date,
                years=years,
                use_cache=use_cache
            )

            if df is not None and not df.empty:
                results[symbol] = df
            else:
                failed.append(symbol)

        if failed:
            logger.warning(f"Failed to fetch: {failed}")

        return results

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the data.

        - Rename columns to standard names
        - Handle missing values
        - Ensure proper data types
        - Remove duplicate indices
        """
        # Make a copy
        df = df.copy()

        # Standardize column names
        column_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
            'Dividends': 'dividends',
            'Stock Splits': 'splits'
        }

        # Rename columns that exist
        df.columns = [column_map.get(c, c.lower()) for c in df.columns]

        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return None

        # Remove rows with any NaN in OHLCV
        original_len = len(df)
        df = df.dropna(subset=required)
        if len(df) < original_len:
            logger.debug(f"Removed {original_len - len(df)} rows with NaN values")

        # Remove duplicate indices
        df = df[~df.index.duplicated(keep='first')]

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort by date
        df = df.sort_index()

        # Ensure numeric types
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove any remaining NaN after type conversion
        df = df.dropna(subset=required)

        return df

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate fetched data for quality.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if df is None or df.empty:
            return False, ["DataFrame is None or empty"]

        # Check for required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(f"Missing columns: {missing}")

        # Check for sufficient data
        if len(df) < 252:  # Less than 1 year
            issues.append(f"Insufficient data: only {len(df)} bars")

        # Check for gaps
        if isinstance(df.index, pd.DatetimeIndex):
            gaps = df.index.to_series().diff()
            large_gaps = gaps[gaps > timedelta(days=10)]
            if len(large_gaps) > 0:
                issues.append(f"Found {len(large_gaps)} large gaps in data")

        # Check for invalid OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low'])
        )
        if invalid_ohlc.any():
            issues.append(f"Found {invalid_ohlc.sum()} bars with invalid OHLC relationships")

        # Check for zero/negative prices
        non_positive = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        if non_positive.any():
            issues.append(f"Found {non_positive.sum()} bars with non-positive prices")

        return len(issues) == 0, issues


def split_data(market_data: Dict[str, pd.DataFrame],
               train_ratio: float = 0.67,
               validation_ratio: float = 0.165,
               holdout_ratio: float = 0.165) -> Tuple[Dict, Dict, Dict]:
    """
    Split market data into training, validation, and holdout sets.

    CRITICAL: Uses strict time-based split to prevent lookahead bias.

    Args:
        market_data: Dict of symbol -> OHLCV DataFrame
        train_ratio: Fraction for training (default 67% = ~2 years of 3)
        validation_ratio: Fraction for validation (default 16.5% = ~6 months)
        holdout_ratio: Fraction for holdout (default 16.5% = ~6 months)

    Returns:
        Tuple of (training_data, validation_data, holdout_data)
    """
    training = {}
    validation = {}
    holdout = {}

    for symbol, df in market_data.items():
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))

        training[symbol] = df.iloc[:train_end].copy()
        validation[symbol] = df.iloc[train_end:val_end].copy()
        holdout[symbol] = df.iloc[val_end:].copy()

    return training, validation, holdout


def get_date_ranges(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Get date ranges for each symbol's data.

    Args:
        market_data: Dict of symbol -> DataFrame

    Returns:
        Dict with date ranges and statistics
    """
    ranges = {}
    for symbol, df in market_data.items():
        ranges[symbol] = {
            'start': df.index.min().strftime('%Y-%m-%d'),
            'end': df.index.max().strftime('%Y-%m-%d'),
            'bars': len(df),
            'trading_days': len(df.index.unique())
        }
    return ranges
