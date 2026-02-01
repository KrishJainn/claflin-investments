#!/usr/bin/env python3
"""
Paper Trading Runner.

Quick start script for paper trading with best strategy.
"""

import argparse
import logging
from datetime import datetime, date
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from trading_evolution.paper import (
    PaperTrader, 
    PaperTraderConfig, 
    BEST_STRATEGY,
    LiveDataManager,
    Bar,
)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"paper_trading_{date.today()}.log"),
        ]
    )


def run_backtest_mode(trader: PaperTrader, years: int = 1):
    """Run paper trader in backtest mode on historical data."""
    from trading_evolution.data.fetcher import DataFetcher
    from trading_evolution.data.cache import DataCache
    
    print("="*60)
    print("PAPER TRADING BACKTEST MODE")
    print(f"Strategy: {BEST_STRATEGY['dna_id']} ({BEST_STRATEGY['version']})")
    print("="*60)
    
    cache = DataCache("./cache")
    fetcher = DataFetcher(cache=cache, cache_dir="./cache")
    
    for symbol in trader.config.symbols:
        print(f"\nProcessing {symbol}...")
        
        df = fetcher.fetch(symbol, years=years)
        if df is None or len(df) < 50:
            print(f"  Skipping - insufficient data")
            continue
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            bar = Bar(
                timestamp=row.name,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume']),
            )
            
            fill = trader.process_bar(symbol, bar)
            if fill:
                print(f"  Trade: {fill.side.value} @ ₹{fill.fill_price:.2f}")
    
    # Flatten remaining
    trader.flatten_all()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    status = trader.get_status()
    print(f"Total Trades: {status['summary'].get('total_trades', 0)}")
    print(f"Win Rate: {status['summary'].get('win_rate', 0):.1f}%")
    print(f"Total P&L: ₹{status['summary'].get('total_pnl', 0):,.2f}")
    
    trader.save()
    print("\n✅ Results saved to ./paper_trades/")


def run_live_mode(trader: PaperTrader):
    """Run paper trader in live simulation mode."""
    print("="*60)
    print("PAPER TRADING LIVE MODE")
    print(f"Strategy: {BEST_STRATEGY['dna_id']} ({BEST_STRATEGY['version']})")
    print("="*60)
    print("\nNote: Live mode would poll for real-time data.")
    print("For now, use --backtest mode to test on historical data.")
    print("\nFuture implementation will include:")
    print("  - WebSocket data feeds")
    print("  - Real-time signal generation")
    print("  - Slack/Telegram alerts")


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Runner")
    parser.add_argument('--mode', choices=['backtest', 'live'], default='backtest',
                        help='Trading mode')
    parser.add_argument('--years', type=int, default=1,
                        help='Years of data for backtest')
    parser.add_argument('--capital', type=float, default=100_000,
                        help='Starting capital (USD)')
    parser.add_argument('--log-level', default='INFO',
                        help='Logging level')
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    config = PaperTraderConfig(
        strategy=BEST_STRATEGY,
        initial_capital=args.capital,
    )
    
    trader = PaperTrader(config)
    
    if args.mode == 'backtest':
        run_backtest_mode(trader, years=args.years)
    else:
        run_live_mode(trader)


if __name__ == '__main__':
    main()
