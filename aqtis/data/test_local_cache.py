#!/usr/bin/env python3
"""
Test script for LocalDataCache

Verifies:
1. Basic fetch and cache functionality
2. Memory cache performance
3. Disk cache persistence
4. Preload batch operations
5. Cache invalidation
6. Statistics tracking
7. Edge cases and error handling
"""

import os
import sys
import time
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd


def print_header(text: str) -> None:
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_result(test_name: str, passed: bool, details: str = "") -> None:
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {test_name}")
    if details:
        print(f"         {details}")


def run_tests():
    """Run all LocalDataCache tests."""
    print_header("LocalDataCache Test Suite")
    print(f"Started: {datetime.now().isoformat()}")

    # Use temp directory for tests
    test_cache_dir = tempfile.mkdtemp(prefix="aqtis_cache_test_")
    print(f"Test cache directory: {test_cache_dir}")

    try:
        from aqtis.data.local_cache import LocalDataCache, MemoryCache

        all_passed = True

        # ============================================================
        # Test 1: Basic Initialization
        # ============================================================
        print_header("Test 1: Initialization")

        try:
            cache = LocalDataCache(
                cache_dir=test_cache_dir,
                memory_cache_size=50,
                memory_cache_ttl_seconds=60,
                disk_cache_ttl_hours=1,
            )
            print_result("LocalDataCache initialization", True)
        except Exception as e:
            print_result("LocalDataCache initialization", False, str(e))
            all_passed = False
            return

        # Verify directory created
        cache_path = Path(test_cache_dir)
        print_result("Cache directory exists", cache_path.exists())

        # ============================================================
        # Test 2: Single Symbol Fetch
        # ============================================================
        print_header("Test 2: Single Symbol Fetch")

        test_symbol = "AAPL"
        test_days = 30
        test_interval = "1h"

        try:
            start_time = time.time()
            df = cache.fetch_and_cache(test_symbol, days=test_days, interval=test_interval)
            fetch_time = time.time() - start_time

            if df is not None and len(df) > 0:
                print_result(f"Fetch {test_symbol}", True, f"{len(df)} rows in {fetch_time:.2f}s")

                # Check columns
                expected_cols = ["open", "high", "low", "close", "volume"]
                has_cols = all(c in df.columns for c in expected_cols)
                print_result("Has OHLCV columns", has_cols, str(list(df.columns)))

                # Check index is datetime
                is_datetime = isinstance(df.index, pd.DatetimeIndex)
                print_result("DatetimeIndex", is_datetime)

                # Check data quality
                has_nulls = df.isnull().any().any()
                print_result("No null values", not has_nulls)

            else:
                print_result(f"Fetch {test_symbol}", False, "No data returned")
                all_passed = False

        except Exception as e:
            print_result(f"Fetch {test_symbol}", False, str(e))
            all_passed = False

        # ============================================================
        # Test 3: Memory Cache Performance
        # ============================================================
        print_header("Test 3: Memory Cache Performance")

        try:
            # Second fetch should be from memory (ultra-fast)
            start_time = time.time()
            df2 = cache.fetch_and_cache(test_symbol, days=test_days, interval=test_interval)
            memory_fetch_time = time.time() - start_time

            print_result(
                "Memory cache retrieval",
                memory_fetch_time < 0.01,
                f"{memory_fetch_time*1000:.2f}ms"
            )

            # Verify data matches
            if df is not None and df2 is not None:
                data_matches = df.equals(df2)
                print_result("Data integrity", data_matches)
            else:
                print_result("Data integrity", False, "Missing data")

            # Check memory stats
            stats = cache.get_stats()
            mem_hits = stats["memory"]["hits"]
            print_result("Memory cache hit recorded", mem_hits > 0, f"hits={mem_hits}")

        except Exception as e:
            print_result("Memory cache test", False, str(e))
            all_passed = False

        # ============================================================
        # Test 4: Disk Cache Persistence
        # ============================================================
        print_header("Test 4: Disk Cache Persistence")

        try:
            # Create new cache instance (simulates restart)
            cache2 = LocalDataCache(
                cache_dir=test_cache_dir,
                memory_cache_size=50,
                memory_cache_ttl_seconds=60,
                disk_cache_ttl_hours=1,
            )

            # Should load from disk
            start_time = time.time()
            df3 = cache2.load_from_cache(test_symbol, days=test_days, interval=test_interval)
            disk_fetch_time = time.time() - start_time

            if df3 is not None:
                print_result("Disk cache load", True, f"{len(df3)} rows in {disk_fetch_time:.3f}s")

                # Check data matches original
                if df is not None:
                    data_matches = len(df) == len(df3)
                    print_result("Data rows match original", data_matches)
            else:
                print_result("Disk cache load", False, "No data loaded")
                all_passed = False

            # Check pickle files exist
            pkl_files = list(Path(test_cache_dir).glob("*.pkl"))
            print_result(
                "Pickle files created",
                len(pkl_files) >= 2,  # data + metadata
                f"{len(pkl_files)} files"
            )

        except Exception as e:
            print_result("Disk cache test", False, str(e))
            all_passed = False

        # ============================================================
        # Test 5: Preload Multiple Symbols
        # ============================================================
        print_header("Test 5: Preload Multiple Symbols")

        preload_symbols = ["MSFT", "GOOGL", "AMZN"]

        try:
            start_time = time.time()
            results = cache.preload_symbols(
                preload_symbols,
                days=30,
                interval="1d",
                max_workers=3,
            )
            preload_time = time.time() - start_time

            successful = sum(1 for v in results.values() if v)
            print_result(
                f"Preload {len(preload_symbols)} symbols",
                successful == len(preload_symbols),
                f"{successful}/{len(preload_symbols)} in {preload_time:.2f}s"
            )

            # Verify each symbol cached
            for symbol in preload_symbols:
                df = cache.load_from_cache(symbol, days=30, interval="1d")
                is_cached = df is not None and len(df) > 0
                print_result(f"  {symbol} cached", is_cached)
                if not is_cached:
                    all_passed = False

        except Exception as e:
            print_result("Preload test", False, str(e))
            all_passed = False

        # ============================================================
        # Test 6: Cache Invalidation
        # ============================================================
        print_header("Test 6: Cache Invalidation")

        try:
            # Invalidate single symbol
            removed = cache.invalidate(symbol="MSFT", days=30, interval="1d")
            print_result("Invalidate MSFT", removed == 1, f"removed {removed}")

            # Verify it's gone
            df = cache.load_from_cache("MSFT", days=30, interval="1d")
            print_result("MSFT no longer cached", df is None)

            # Other symbols should still be cached
            df_googl = cache.load_from_cache("GOOGL", days=30, interval="1d")
            print_result("GOOGL still cached", df_googl is not None)

        except Exception as e:
            print_result("Invalidation test", False, str(e))
            all_passed = False

        # ============================================================
        # Test 7: Statistics Tracking
        # ============================================================
        print_header("Test 7: Statistics Tracking")

        try:
            stats = cache.get_stats()

            print(f"  Cache Statistics:")
            print(f"    Memory entries: {stats['memory']['size']}")
            print(f"    Memory hits: {stats['memory']['hits']}")
            print(f"    Memory hit rate: {stats['memory']['hit_rate']:.1%}")
            print(f"    Disk entries: {stats['disk']['entries']}")
            print(f"    Disk size: {stats['disk']['total_size_mb']:.2f} MB")
            print(f"    API fetches: {stats['api_fetches']}")

            print_result("Stats tracking", True)

        except Exception as e:
            print_result("Stats test", False, str(e))
            all_passed = False

        # ============================================================
        # Test 8: Get Cached Symbols
        # ============================================================
        print_header("Test 8: Get Cached Symbols")

        try:
            cached = cache.get_cached_symbols()
            print(f"  Cached symbols: {len(cached)}")
            for item in cached[:5]:  # Show first 5
                print(f"    - {item['symbol']}: {item['rows']} rows ({item['interval']})")

            print_result("Get cached symbols", len(cached) > 0)

        except Exception as e:
            print_result("Get cached symbols test", False, str(e))
            all_passed = False

        # ============================================================
        # Test 9: Force Refresh
        # ============================================================
        print_header("Test 9: Force Refresh")

        try:
            # Get current cache timestamp
            cached_before = cache.get_cached_symbols()
            aapl_before = next((c for c in cached_before if c["symbol"] == "AAPL"), None)

            if aapl_before:
                time.sleep(1)  # Brief pause

                # Force refresh
                df = cache.fetch_and_cache(
                    "AAPL",
                    days=test_days,
                    interval=test_interval,
                    force_refresh=True
                )

                cached_after = cache.get_cached_symbols()
                aapl_after = next((c for c in cached_after if c["symbol"] == "AAPL"), None)

                if aapl_after:
                    refreshed = aapl_after["cached_at"] > aapl_before["cached_at"]
                    print_result("Force refresh updates timestamp", refreshed)
                else:
                    print_result("Force refresh", False, "AAPL not found after refresh")
            else:
                print_result("Force refresh", False, "AAPL not cached")

        except Exception as e:
            print_result("Force refresh test", False, str(e))
            all_passed = False

        # ============================================================
        # Test 10: MemoryCache Isolation Test
        # ============================================================
        print_header("Test 10: MemoryCache Isolation")

        try:
            mem_cache = MemoryCache(max_size=3)

            # Add items
            for i in range(5):
                df = pd.DataFrame({"value": [i]})
                mem_cache.put(f"key_{i}", df)

            # Should only have 3 items (LRU eviction)
            stats = mem_cache.get_stats()
            print_result("LRU eviction works", stats["size"] == 3, f"size={stats['size']}")

            # Oldest should be evicted (key_0, key_1)
            old_key = mem_cache.get("key_0", max_age_seconds=300)
            new_key = mem_cache.get("key_4", max_age_seconds=300)
            print_result("Evicts oldest entries", old_key is None and new_key is not None)

        except Exception as e:
            print_result("MemoryCache test", False, str(e))
            all_passed = False

        # ============================================================
        # Test 11: Different Intervals
        # ============================================================
        print_header("Test 11: Different Intervals")

        intervals_to_test = ["1d", "1h", "5m"]

        for interval in intervals_to_test:
            try:
                df = cache.fetch_and_cache("AAPL", days=7, interval=interval)
                success = df is not None and len(df) > 0
                rows = len(df) if df is not None else 0
                print_result(f"Interval {interval}", success, f"{rows} rows")
            except Exception as e:
                print_result(f"Interval {interval}", False, str(e))

        # ============================================================
        # Test 12: Warm Memory Cache
        # ============================================================
        print_header("Test 12: Warm Memory Cache")

        try:
            # Clear memory cache
            cache._memory_cache.invalidate()

            # Warm from disk
            warmed = cache.warm_memory_cache()
            print_result("Warm memory cache", warmed > 0, f"loaded {warmed} entries")

        except Exception as e:
            print_result("Warm memory cache test", False, str(e))
            all_passed = False

        # ============================================================
        # Summary
        # ============================================================
        print_header("Test Summary")

        final_stats = cache.get_stats()
        print(f"  Final cache state:")
        print(f"    Memory entries: {final_stats['memory']['size']}")
        print(f"    Disk entries: {final_stats['disk']['entries']}")
        print(f"    Total disk size: {final_stats['disk']['total_size_mb']:.2f} MB")
        print(f"    API calls made: {final_stats['api_fetches']}")

        if all_passed:
            print("\n  ✓ ALL TESTS PASSED")
        else:
            print("\n  ✗ SOME TESTS FAILED")

        return all_passed

    finally:
        # Cleanup test directory
        print(f"\nCleaning up test directory: {test_cache_dir}")
        try:
            shutil.rmtree(test_cache_dir)
            print("  ✓ Cleanup complete")
        except Exception as e:
            print(f"  ✗ Cleanup failed: {e}")


def quick_benchmark():
    """Run quick performance benchmark."""
    print_header("Performance Benchmark")

    from aqtis.data.local_cache import LocalDataCache

    cache = LocalDataCache(cache_dir="./benchmark_cache")

    symbol = "AAPL"
    days = 60
    interval = "1h"

    # Initial fetch (API call)
    start = time.time()
    df = cache.fetch_and_cache(symbol, days, interval, force_refresh=True)
    api_time = time.time() - start
    print(f"  API fetch: {api_time:.3f}s")

    # Memory cache read (10 iterations)
    times = []
    for _ in range(10):
        start = time.time()
        df = cache.fetch_and_cache(symbol, days, interval)
        times.append(time.time() - start)
    avg_mem = sum(times) / len(times)
    print(f"  Memory cache (avg): {avg_mem*1000:.3f}ms")

    # Clear memory, disk cache read
    cache._memory_cache.invalidate()
    start = time.time()
    df = cache.fetch_and_cache(symbol, days, interval)
    disk_time = time.time() - start
    print(f"  Disk cache: {disk_time*1000:.3f}ms")

    # Speedup
    print(f"\n  Speedup vs API:")
    print(f"    Memory cache: {api_time/avg_mem:.0f}x faster")
    print(f"    Disk cache: {api_time/disk_time:.1f}x faster")

    # Cleanup
    shutil.rmtree("./benchmark_cache", ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LocalDataCache")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark only")
    args = parser.parse_args()

    if args.benchmark:
        quick_benchmark()
    else:
        success = run_tests()
        print("\n" + "="*60)
        quick_benchmark()
        sys.exit(0 if success else 1)
