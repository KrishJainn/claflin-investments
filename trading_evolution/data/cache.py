"""
Data caching module using Parquet format for efficient storage.

Caches fetched market data to avoid repeated API calls.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
import hashlib
import json


class DataCache:
    """
    Parquet-based data cache for market data.

    Features:
    - Efficient columnar storage
    - Fast read/write operations
    - Automatic cache invalidation based on age
    - Content-based cache keys
    """

    def __init__(self, cache_dir: Path, max_age_days: int = 1):
        """
        Initialize data cache.

        Args:
            cache_dir: Directory for cache files
            max_age_days: Maximum age of cache before refresh (default: 1 day)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(days=max_age_days)
        self._metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load cache metadata."""
        if self._metadata_file.exists():
            with open(self._metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata."""
        with open(self._metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2)

    def _get_cache_key(self, symbol: str, start_date: str, end_date: str) -> str:
        """Generate cache key for data request."""
        key_str = f"{symbol}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.parquet"

    def is_cached(self, symbol: str, start_date: str, end_date: str) -> bool:
        """
        Check if data is cached and fresh.

        Args:
            symbol: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            True if cached and fresh, False otherwise
        """
        cache_key = self._get_cache_key(symbol, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return False

        # Check age
        if cache_key in self._metadata:
            cached_time = datetime.fromisoformat(self._metadata[cache_key]['cached_at'])
            if datetime.now() - cached_time > self.max_age:
                return False

        return True

    def get(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data.

        Args:
            symbol: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame if cached, None otherwise
        """
        cache_key = self._get_cache_key(symbol, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)

        if not self.is_cached(symbol, start_date, end_date):
            return None

        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Error reading cache for {symbol}: {e}")
            # Remove corrupted cache
            cache_path.unlink(missing_ok=True)
            return None

    def put(self, symbol: str, start_date: str, end_date: str, data: pd.DataFrame):
        """
        Store data in cache.

        Args:
            symbol: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data: DataFrame to cache
        """
        if data is None or data.empty:
            return

        cache_key = self._get_cache_key(symbol, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)

        try:
            data.to_parquet(cache_path, index=True)

            # Update metadata
            self._metadata[cache_key] = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'cached_at': datetime.now().isoformat(),
                'rows': len(data),
                'columns': list(data.columns)
            }
            self._save_metadata()

        except Exception as e:
            print(f"Error caching data for {symbol}: {e}")

    def invalidate(self, symbol: str = None):
        """
        Invalidate cache entries.

        Args:
            symbol: If provided, invalidate only this symbol. Otherwise invalidate all.
        """
        if symbol:
            # Invalidate specific symbol
            keys_to_remove = [
                k for k, v in self._metadata.items()
                if v.get('symbol') == symbol
            ]
            for key in keys_to_remove:
                cache_path = self._get_cache_path(key)
                cache_path.unlink(missing_ok=True)
                del self._metadata[key]
        else:
            # Invalidate all
            for key in list(self._metadata.keys()):
                cache_path = self._get_cache_path(key)
                cache_path.unlink(missing_ok=True)
            self._metadata = {}

        self._save_metadata()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_size = sum(
            self._get_cache_path(k).stat().st_size
            for k in self._metadata.keys()
            if self._get_cache_path(k).exists()
        )

        return {
            'num_entries': len(self._metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'symbols': list(set(v['symbol'] for v in self._metadata.values())),
            'oldest_entry': min(
                (v['cached_at'] for v in self._metadata.values()),
                default=None
            ),
            'newest_entry': max(
                (v['cached_at'] for v in self._metadata.values()),
                default=None
            )
        }

    def cleanup_old_entries(self):
        """Remove entries older than max_age."""
        now = datetime.now()
        keys_to_remove = []

        for key, meta in self._metadata.items():
            cached_time = datetime.fromisoformat(meta['cached_at'])
            if now - cached_time > self.max_age:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            cache_path = self._get_cache_path(key)
            cache_path.unlink(missing_ok=True)
            del self._metadata[key]

        if keys_to_remove:
            self._save_metadata()

        return len(keys_to_remove)
