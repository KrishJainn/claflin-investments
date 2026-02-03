"""
High-Performance Local Data Cache for AQTIS

Provides ultra-fast data access through:
1. Local pickle storage for 60-day intraday data
2. In-memory LRU caching for repeated access
3. Batch preloading for multiple symbols
4. Automatic cache invalidation and refresh
"""

import os
import pickle
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from functools import lru_cache
from collections import OrderedDict
import logging

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)


class MemoryCache:
    """Thread-safe LRU memory cache for ultra-fast repeated access."""

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[str, Tuple[pd.DataFrame, datetime]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str, max_age_seconds: int = 300) -> Optional[pd.DataFrame]:
        """Get item from cache if exists and not expired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            df, cached_at = self._cache[key]
            if (datetime.now() - cached_at).total_seconds() > max_age_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return df.copy()

    def put(self, key: str, df: pd.DataFrame) -> None:
        """Store item in cache, evicting oldest if full."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

            self._cache[key] = (df.copy(), datetime.now())

            # Evict oldest entries if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, key: Optional[str] = None) -> None:
        """Invalidate specific key or entire cache."""
        with self._lock:
            if key:
                self._cache.pop(key, None)
            else:
                self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def clear_stats(self) -> None:
        """Reset hit/miss counters."""
        with self._lock:
            self._hits = 0
            self._misses = 0


class LocalDataCache:
    """
    High-performance local data cache for AQTIS trading system.

    Features:
    - Downloads 60 days of intraday data from yfinance
    - Stores data locally in pickle format for fast I/O
    - In-memory LRU cache for ultra-fast repeated access
    - Batch preloading for multiple symbols
    - Thread-safe operations
    - Automatic cache refresh based on TTL

    Usage:
        cache = LocalDataCache(cache_dir="./data_cache")

        # Fetch single symbol (auto-caches)
        df = cache.fetch_and_cache("AAPL", days=60, interval="1h")

        # Preload multiple symbols
        cache.preload_symbols(["AAPL", "GOOGL", "MSFT"], days=60)

        # Load from cache (memory first, then disk)
        df = cache.load_from_cache("AAPL", days=60, interval="1h")
    """

    # Default intervals for intraday data
    VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo"]

    # Max days for intraday intervals (yfinance limitation)
    INTERVAL_MAX_DAYS = {
        "1m": 7,
        "2m": 60,
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "60m": 730,
        "90m": 60,
        "1h": 730,
        "1d": 10000,
        "5d": 10000,
        "1wk": 10000,
        "1mo": 10000,
    }

    def __init__(
        self,
        cache_dir: str = "./local_data_cache",
        memory_cache_size: int = 100,
        memory_cache_ttl_seconds: int = 300,
        disk_cache_ttl_hours: int = 24,
    ):
        """
        Initialize LocalDataCache.

        Args:
            cache_dir: Directory for pickle file storage
            memory_cache_size: Max items in memory cache
            memory_cache_ttl_seconds: Memory cache expiry (default 5 min)
            disk_cache_ttl_hours: Disk cache expiry (default 24 hours)
        """
        if yf is None:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._memory_cache = MemoryCache(max_size=memory_cache_size)
        self._memory_ttl = memory_cache_ttl_seconds
        self._disk_ttl = timedelta(hours=disk_cache_ttl_hours)

        self._metadata_file = self.cache_dir / "cache_metadata.pkl"
        self._metadata: Dict[str, Dict[str, Any]] = self._load_metadata()
        self._metadata_lock = threading.RLock()  # Use RLock for reentrant locking

        # Stats tracking
        self._disk_hits = 0
        self._disk_misses = 0
        self._fetches = 0

        logger.info(f"LocalDataCache initialized at {self.cache_dir}")

    def _generate_cache_key(
        self,
        symbol: str,
        days: int,
        interval: str,
    ) -> str:
        """Generate unique cache key for symbol/days/interval combo."""
        key_str = f"{symbol}_{days}_{interval}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cached data."""
        return self.cache_dir / f"{cache_key}.pkl"

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load cache metadata from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        with self._metadata_lock:
            try:
                with open(self._metadata_file, "wb") as f:
                    pickle.dump(self._metadata, f)
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")

    def _is_disk_cache_valid(self, cache_key: str) -> bool:
        """Check if disk cache entry is valid (exists and not expired)."""
        if cache_key not in self._metadata:
            return False

        meta = self._metadata[cache_key]
        cached_at = meta.get("cached_at")
        if not cached_at:
            return False

        if datetime.now() - cached_at > self._disk_ttl:
            return False

        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists()

    def _fetch_from_yfinance(
        self,
        symbol: str,
        days: int,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance API."""
        self._fetches += 1

        # Validate interval
        if interval not in self.VALID_INTERVALS:
            logger.error(f"Invalid interval: {interval}. Valid: {self.VALID_INTERVALS}")
            return None

        # Respect yfinance max days for interval
        max_days = self.INTERVAL_MAX_DAYS.get(interval, days)
        actual_days = min(days, max_days)
        if actual_days != days:
            logger.warning(f"Reduced days from {days} to {actual_days} for interval {interval}")

        try:
            ticker = yf.Ticker(symbol)

            # Use period for cleaner API call
            if actual_days <= 7:
                period = f"{actual_days}d"
            elif actual_days <= 30:
                period = "1mo"
            elif actual_days <= 90:
                period = "3mo"
            elif actual_days <= 180:
                period = "6mo"
            elif actual_days <= 365:
                period = "1y"
            elif actual_days <= 730:
                period = "2y"
            else:
                period = "max"

            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Standardize column names (lowercase)
            df.columns = df.columns.str.lower()

            # Keep only OHLCV columns
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            available_cols = [c for c in ohlcv_cols if c in df.columns]
            df = df[available_cols]

            # Clean data
            df = df.dropna()
            df = df.sort_index()

            # Validate OHLC relationships
            if all(c in df.columns for c in ["open", "high", "low", "close"]):
                valid_mask = (
                    (df["high"] >= df["low"]) &
                    (df["high"] >= df["open"]) &
                    (df["high"] >= df["close"]) &
                    (df["low"] <= df["open"]) &
                    (df["low"] <= df["close"])
                )
                invalid_count = (~valid_mask).sum()
                if invalid_count > 0:
                    logger.warning(f"Removed {invalid_count} rows with invalid OHLC for {symbol}")
                    df = df[valid_mask]

            logger.info(f"Fetched {len(df)} rows for {symbol} ({interval})")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return None

    def _save_to_disk(
        self,
        cache_key: str,
        df: pd.DataFrame,
        symbol: str,
        days: int,
        interval: str,
    ) -> bool:
        """Save DataFrame to disk cache."""
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Update metadata
            with self._metadata_lock:
                self._metadata[cache_key] = {
                    "symbol": symbol,
                    "days": days,
                    "interval": interval,
                    "cached_at": datetime.now(),
                    "rows": len(df),
                    "columns": list(df.columns),
                    "start_date": df.index.min().isoformat() if len(df) > 0 else None,
                    "end_date": df.index.max().isoformat() if len(df) > 0 else None,
                }
            self._save_metadata()

            logger.debug(f"Saved {symbol} to disk cache ({cache_path.name})")
            return True

        except Exception as e:
            logger.error(f"Failed to save {symbol} to disk: {e}")
            return False

    def _load_from_disk(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from disk cache."""
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                df = pickle.load(f)
            self._disk_hits += 1
            return df
        except Exception as e:
            logger.error(f"Failed to load from disk ({cache_key}): {e}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
            except:
                pass
            return None

    def fetch_and_cache(
        self,
        symbol: str,
        days: int = 60,
        interval: str = "1h",
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for symbol and cache it (memory + disk).

        Args:
            symbol: Ticker symbol (e.g., "AAPL", "RELIANCE.NS")
            days: Number of days of data (default 60)
            interval: Data interval (default "1h")
            force_refresh: Force re-fetch from API

        Returns:
            DataFrame with OHLCV data, or None if failed
        """
        cache_key = self._generate_cache_key(symbol, days, interval)

        # Try memory cache first (unless force refresh)
        if not force_refresh:
            df = self._memory_cache.get(cache_key, self._memory_ttl)
            if df is not None:
                logger.debug(f"Memory cache hit: {symbol}")
                return df

        # Try disk cache (unless force refresh)
        if not force_refresh and self._is_disk_cache_valid(cache_key):
            df = self._load_from_disk(cache_key)
            if df is not None:
                # Populate memory cache
                self._memory_cache.put(cache_key, df)
                logger.debug(f"Disk cache hit: {symbol}")
                return df

        self._disk_misses += 1

        # Fetch from yfinance
        df = self._fetch_from_yfinance(symbol, days, interval)
        if df is None:
            return None

        # Cache to both layers
        self._memory_cache.put(cache_key, df)
        self._save_to_disk(cache_key, df, symbol, days, interval)

        return df

    def load_from_cache(
        self,
        symbol: str,
        days: int = 60,
        interval: str = "1h",
    ) -> Optional[pd.DataFrame]:
        """
        Load data from cache only (no API fetch if not cached).

        Args:
            symbol: Ticker symbol
            days: Number of days
            interval: Data interval

        Returns:
            DataFrame if cached, None otherwise
        """
        cache_key = self._generate_cache_key(symbol, days, interval)

        # Try memory cache
        df = self._memory_cache.get(cache_key, self._memory_ttl)
        if df is not None:
            return df

        # Try disk cache
        if self._is_disk_cache_valid(cache_key):
            df = self._load_from_disk(cache_key)
            if df is not None:
                self._memory_cache.put(cache_key, df)
                return df

        return None

    def preload_symbols(
        self,
        symbols: List[str],
        days: int = 60,
        interval: str = "1h",
        force_refresh: bool = False,
        max_workers: int = 4,
    ) -> Dict[str, bool]:
        """
        Batch preload multiple symbols into cache.

        Args:
            symbols: List of ticker symbols
            days: Number of days of data
            interval: Data interval
            force_refresh: Force re-fetch all symbols
            max_workers: Number of concurrent downloads

        Returns:
            Dict mapping symbol to success status
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        def fetch_symbol(symbol: str) -> Tuple[str, bool]:
            try:
                df = self.fetch_and_cache(symbol, days, interval, force_refresh)
                return symbol, df is not None
            except Exception as e:
                logger.error(f"Failed to preload {symbol}: {e}")
                return symbol, False

        logger.info(f"Preloading {len(symbols)} symbols...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_symbol, s): s for s in symbols}

            for future in as_completed(futures):
                symbol, success = future.result()
                results[symbol] = success
                status = "✓" if success else "✗"
                logger.info(f"  {status} {symbol}")

        successful = sum(1 for v in results.values() if v)
        logger.info(f"Preloaded {successful}/{len(symbols)} symbols successfully")

        return results

    def invalidate(
        self,
        symbol: Optional[str] = None,
        days: Optional[int] = None,
        interval: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            symbol: Specific symbol to invalidate (None = all)
            days: Specific days parameter (None = all matching symbol)
            interval: Specific interval (None = all matching symbol/days)

        Returns:
            Number of entries invalidated
        """
        invalidated = 0

        with self._metadata_lock:
            keys_to_remove = []

            for cache_key, meta in self._metadata.items():
                if symbol and meta.get("symbol") != symbol:
                    continue
                if days and meta.get("days") != days:
                    continue
                if interval and meta.get("interval") != interval:
                    continue

                keys_to_remove.append(cache_key)

            for cache_key in keys_to_remove:
                # Remove from memory cache
                self._memory_cache.invalidate(cache_key)

                # Remove from disk
                cache_path = self._get_cache_path(cache_key)
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove {cache_path}: {e}")

                # Remove from metadata
                del self._metadata[cache_key]
                invalidated += 1

            if invalidated > 0:
                self._save_metadata()

        logger.info(f"Invalidated {invalidated} cache entries")
        return invalidated

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self._memory_cache.get_stats()

        disk_total = self._disk_hits + self._disk_misses

        # Calculate total cache size
        total_size_bytes = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.name != "cache_metadata.pkl":
                total_size_bytes += cache_file.stat().st_size

        return {
            "memory": memory_stats,
            "disk": {
                "entries": len(self._metadata),
                "hits": self._disk_hits,
                "misses": self._disk_misses,
                "hit_rate": self._disk_hits / disk_total if disk_total > 0 else 0.0,
                "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            },
            "api_fetches": self._fetches,
            "cache_dir": str(self.cache_dir),
        }

    def get_cached_symbols(self) -> List[Dict[str, Any]]:
        """Get list of all cached symbols with metadata."""
        result = []
        with self._metadata_lock:
            for cache_key, meta in self._metadata.items():
                if self._is_disk_cache_valid(cache_key):
                    result.append({
                        "symbol": meta.get("symbol"),
                        "days": meta.get("days"),
                        "interval": meta.get("interval"),
                        "rows": meta.get("rows"),
                        "cached_at": meta.get("cached_at"),
                        "start_date": meta.get("start_date"),
                        "end_date": meta.get("end_date"),
                    })
        return sorted(result, key=lambda x: x.get("symbol", ""))

    def cleanup_expired(self) -> int:
        """Remove expired cache entries from disk."""
        removed = 0

        with self._metadata_lock:
            keys_to_remove = []

            for cache_key in list(self._metadata.keys()):
                if not self._is_disk_cache_valid(cache_key):
                    keys_to_remove.append(cache_key)

            for cache_key in keys_to_remove:
                cache_path = self._get_cache_path(cache_key)
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                except:
                    pass
                del self._metadata[cache_key]
                self._memory_cache.invalidate(cache_key)
                removed += 1

            if removed > 0:
                self._save_metadata()

        logger.info(f"Cleaned up {removed} expired cache entries")
        return removed

    def warm_memory_cache(self) -> int:
        """Load all valid disk cache entries into memory cache."""
        loaded = 0

        for cache_key in list(self._metadata.keys()):
            if self._is_disk_cache_valid(cache_key):
                # Check if not already in memory
                if self._memory_cache.get(cache_key, self._memory_ttl) is None:
                    df = self._load_from_disk(cache_key)
                    if df is not None:
                        self._memory_cache.put(cache_key, df)
                        loaded += 1

        logger.info(f"Warmed memory cache with {loaded} entries")
        return loaded


# Convenience function for quick access
def get_cached_data(
    symbol: str,
    days: int = 60,
    interval: str = "1h",
    cache_dir: str = "./local_data_cache",
) -> Optional[pd.DataFrame]:
    """
    Quick helper to fetch cached data.

    Creates a LocalDataCache instance if needed, fetches data.
    For repeated use, create LocalDataCache instance directly.
    """
    cache = LocalDataCache(cache_dir=cache_dir)
    return cache.fetch_and_cache(symbol, days, interval)
