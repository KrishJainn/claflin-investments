"""
Intraday Data Fetcher via yfinance.

Fetches and manages intraday (1m, 5m, 15m) bars for NIFTY 50 stocks:
- IST timezone handling
- Bar aggregation (1m → 5m)
- Trading hours validation
- Gap handling
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False

from .cache import DataCache

logger = logging.getLogger(__name__)


# NIFTY 50 constituents (as of Jan 2026)
NIFTY_50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS", "ONGC.NS", "NTPC.NS",
    "ULTRACEMCO.NS", "NESTLEIND.NS", "POWERGRID.NS", "M&M.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "ADANIENT.NS", "BAJAJFINSV.NS", "JSWSTEEL.NS", "HCLTECH.NS",
    "TECHM.NS", "COALINDIA.NS", "DRREDDY.NS", "INDUSINDBK.NS", "GRASIM.NS",
    "CIPLA.NS", "BRITANNIA.NS", "EICHERMOT.NS", "DIVISLAB.NS", "HEROMOTOCO.NS",
    "APOLLOHOSP.NS", "TATACONSUM.NS", "SBILIFE.NS", "HINDALCO.NS", "BPCL.NS",
    "ADANIPORTS.NS", "BAJAJ-AUTO.NS", "SHRIRAMFIN.NS", "HDFCLIFE.NS", "LTIM.NS",
]


@dataclass
class IntradayConfig:
    """Configuration for intraday data fetching."""
    
    interval: str = "5m"  # 1m, 5m, 15m
    timezone: str = "Asia/Kolkata"
    
    # Indian market hours
    pre_market_start: time = field(default_factory=lambda: time(9, 0))
    market_open: time = field(default_factory=lambda: time(9, 15))
    market_close: time = field(default_factory=lambda: time(15, 30))
    
    # Max history (yfinance limits)
    max_days_1m: int = 7        # 1m: 7 days
    max_days_5m: int = 60       # 5m: 60 days
    max_days_15m: int = 60      # 15m: 60 days
    max_days_1h: int = 730      # 1h: 730 days
    max_days_daily: int = 9999  # daily: unlimited
    
    def get_max_days(self) -> int:
        """Get maximum days for current interval."""
        interval_map = {
            '1m': self.max_days_1m,
            '5m': self.max_days_5m,
            '15m': self.max_days_15m,
            '1h': self.max_days_1h,
            '1d': self.max_days_daily,
        }
        return interval_map.get(self.interval, 60)


class IntradayDataFetcher:
    """
    Fetches intraday data with IST timezone handling.
    """
    
    def __init__(
        self,
        config: IntradayConfig = None,
        cache: DataCache = None,
        cache_dir: Path = None,
    ):
        """
        Initialize intraday data fetcher.
        
        Args:
            config: Intraday configuration
            cache: Optional data cache
            cache_dir: Directory for cache
        """
        self.config = config or IntradayConfig()
        
        if cache:
            self.cache = cache
        elif cache_dir:
            self.cache = DataCache(cache_dir)
        else:
            self.cache = DataCache(Path("data/intraday_cache"))
        
        # Set up timezone
        if PYTZ_AVAILABLE:
            self.tz = pytz.timezone(self.config.timezone)
        else:
            self.tz = None
            logger.warning("pytz not available, timezone handling may be limited")
    
    def fetch_intraday(
        self,
        symbol: str,
        days: int = None,
        start_date: date = None,
        end_date: date = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch intraday data for a symbol.
        
        Args:
            symbol: Stock ticker (e.g., 'RELIANCE.NS')
            days: Number of days to fetch (default: max for interval)
            start_date: Start date (overrides days)
            end_date: End date (default: today)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data, indexed by IST datetime
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is required for intraday data")
        
        # Determine date range (use max_days - 3 to stay within yfinance limit
        # since we add +1 day to end_date for the API call)
        max_days = self.config.get_max_days() - 3
        days = min(days or max_days, max_days)
        
        if end_date is None:
            end_date = date.today()
        
        if start_date is None:
            start_date = end_date - timedelta(days=days)
        
        # Check cache first
        cache_key_sym = f"{symbol}_{self.config.interval}"
        cache_start = str(start_date)
        cache_end = str(end_date)
        if use_cache:
            cached = self.cache.get(cache_key_sym, cache_start, cache_end)
            if cached is not None:
                logger.debug(f"Using cached data for {symbol}")
                return cached
        
        # Fetch from yfinance
        logger.info(f"Fetching {symbol} {self.config.interval} data: {start_date} to {end_date}")
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                interval=self.config.interval,
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Clean and validate
            df = self._clean_data(df)
            df = self._validate_trading_hours(df)
            
            # Cache the result
            if use_cache and len(df) > 0:
                self.cache.put(cache_key_sym, cache_start, cache_end, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple(
        self,
        symbols: List[str],
        days: int = None,
        start_date: date = None,
        end_date: date = None,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday data for multiple symbols.
        
        Args:
            symbols: List of stock tickers
            days: Number of days to fetch
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cache
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.fetch_intraday(
                    symbol=symbol,
                    days=days,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache,
                )
                if len(df) > 0:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        
        logger.info(f"Fetched {len(results)}/{len(symbols)} symbols successfully")
        return results
    
    def fetch_nifty50(
        self,
        days: int = None,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday data for all NIFTY 50 constituents.
        
        Args:
            days: Number of days to fetch
            use_cache: Whether to use cache
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        return self.fetch_multiple(
            symbols=NIFTY_50_SYMBOLS,
            days=days,
            use_cache=use_cache,
        )
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data."""
        if df.empty:
            return df
        
        # Make a copy
        df = df.copy()
        
        # Ensure column names are standardized
        column_map = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
        }
        
        df.columns = [column_map.get(c.lower(), c) for c in df.columns]
        
        # Remove any rows with all NaN
        df = df.dropna(how='all')
        
        # Forward fill small gaps (1-2 bars)
        df = df.ffill(limit=2)
        
        # Remove any remaining NaN rows
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Ensure positive volumes
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].clip(lower=0)
        
        # Sort by index
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def _validate_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to valid trading hours only."""
        if df.empty:
            return df
        
        # Localize to IST if not already
        if self.tz and df.index.tz is None:
            try:
                df.index = df.index.tz_localize('UTC').tz_convert(self.tz)
            except:
                pass
        
        # Filter to market hours
        market_open = self.config.market_open
        market_close = self.config.market_close
        
        mask = (df.index.time >= market_open) & (df.index.time <= market_close)
        df = df[mask]
        
        return df
    
    def aggregate_bars(
        self,
        df: pd.DataFrame,
        target_interval: str,
    ) -> pd.DataFrame:
        """
        Aggregate bars to a larger interval.
        
        Example: 1m bars -> 5m bars
        
        Args:
            df: DataFrame with OHLCV data
            target_interval: Target interval (e.g., '5m', '15m')
            
        Returns:
            Aggregated DataFrame
        """
        if df.empty:
            return df
        
        # Define aggregation rules
        agg_rules = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
        }
        
        # Filter to columns we have
        agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
        
        # Resample
        df_agg = df.resample(target_interval).agg(agg_rules)
        
        # Remove rows with no trades
        df_agg = df_agg.dropna()
        
        return df_agg
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic indicators needed for backtesting.
        
        Adds:
        - ATR_14
        - SMA_20
        - EMA_9, EMA_21
        - RSI_14
        - Volume SMA
        """
        if df.empty or len(df) < 20:
            return df
        
        df = df.copy()
        
        # ATR
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(14).mean()
        
        # SMAs
        df['SMA_20'] = close.rolling(20).mean()
        df['SMA_50'] = close.rolling(50).mean()
        
        # EMAs
        df['EMA_9'] = close.ewm(span=9).mean()
        df['EMA_21'] = close.ewm(span=21).mean()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Volume SMA
        if 'Volume' in df.columns:
            df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
        
        return df
    
    def get_trading_dates(
        self,
        df: pd.DataFrame,
    ) -> List[date]:
        """Get list of unique trading dates in the data."""
        if df.empty:
            return []
        
        return sorted(df.index.date.unique())
    
    def split_by_day(
        self,
        df: pd.DataFrame,
    ) -> Dict[date, pd.DataFrame]:
        """Split DataFrame by trading day."""
        result = {}
        
        for d in self.get_trading_dates(df):
            mask = df.index.date == d
            day_df = df[mask]
            if len(day_df) > 0:
                result[d] = day_df
        
        return result


def get_nifty50_symbols() -> List[str]:
    """Get list of NIFTY 50 symbols."""
    return NIFTY_50_SYMBOLS.copy()


def fetch_sample_data(
    symbol: str = "RELIANCE.NS",
    interval: str = "5m",
    days: int = 30,
) -> pd.DataFrame:
    """
    Quick function to fetch sample intraday data.
    
    Args:
        symbol: Stock ticker
        interval: Bar interval
        days: Number of days
        
    Returns:
        DataFrame with OHLCV data
    """
    config = IntradayConfig(interval=interval)
    fetcher = IntradayDataFetcher(config=config)
    
    df = fetcher.fetch_intraday(symbol=symbol, days=days)
    df = fetcher.add_indicators(df)
    
    return df
