"""
LIVE PAPER TRADING SYSTEM

This script:
1. Fetches real-time data for Nifty 50 stocks
2. Calculates indicators and Super Indicator
3. Generates BUY/SELL signals
4. Tracks paper trades with P&L
5. Logs everything for analysis

Run this during market hours (9:15 AM - 3:30 PM IST)
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import yfinance as yf

from trading_evolution.indicators.universe import IndicatorUniverse
from trading_evolution.indicators.calculator import IndicatorCalculator
from trading_evolution.indicators.normalizer import IndicatorNormalizer
from trading_evolution.super_indicator.dna import SuperIndicatorDNA, IndicatorGene
from trading_evolution.super_indicator.core import SuperIndicator


# ============================================================================
# OPTIMIZED STRATEGY WEIGHTS (from top 10 Hall of Fame strategies)
# ============================================================================
STRATEGY_WEIGHTS = {
    "MFI_14": -0.8525, "TEMA_10": -0.3098, "SMA_20": 0.1499, "AO_5_34": -0.9341,
    "ATR_14": 0.5290, "NATR_20": 0.2629, "CMF_21": 0.3633, "TEMA_20": 0.4261,
    "TSI_13_25": 0.8902, "STOCH_5_3": 0.6448, "LINREG_SLOPE_14": 0.1537,
    "AROON_25": -0.9552, "CCI_20": -0.6396, "EFI_13": -0.8698, "VWMA_10": -0.9412,
    "PIVOTS": -0.5856, "DONCHIAN_50": -0.3697, "BBANDS_20_2.5": 0.4697,
    "WMA_20": -0.7451, "STOCH_14_3": -0.6456, "DEMA_20": -0.6598, "VWMA_20": -0.8193,
    "ADOSC_3_10": 0.3516, "WMA_10": -0.7413, "ADX_20": -0.8947, "ZSCORE_20": 0.5006,
    "ATR_20": 0.3462, "UO_7_14_28": -0.2402, "VORTEX_14": -0.2020, "OBV": -0.4973,
    "WILLR_14": -0.1467, "CCI_14": -0.3218, "KAMA_10": -0.1284, "KST": -0.7961,
    "SUPERTREND_7_3": -0.9663, "MASS_INDEX": 0.4535, "LINREG_SLOPE_25": -0.3096,
    "AROON_14": 0.4701, "MFI_20": -0.2728, "PVI": 0.7962, "NVI": 0.8593,
    "ICHIMOKU": -0.4294, "DEMA_10": -0.1344, "MOM_20": 0.2047, "PSAR": 0.1370,
    "EFI_20": 0.1479, "DONCHIAN_20": 0.1299, "ROC_20": -0.0827
}

# Top performing stocks from backtesting
WATCHLIST = [
    'INFY.NS', 'AXISBANK.NS', 'TECHM.NS', 'SUNPHARMA.NS', 'KOTAKBANK.NS',
    'TITAN.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS',
    'RELIANCE.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'ASIANPAINT.NS', 'NESTLEIND.NS',
    'LT.NS', 'WIPRO.NS', 'HCLTECH.NS', 'ITC.NS', 'SBIN.NS'
]

# Signal thresholds (OPTIMIZED from parameter optimization)
# Best combo: Entry=0.5, Exit=0.2, SL=2.0x ATR -> $38,854 profit, 56% WR, 2.42 PF
LONG_ENTRY_THRESHOLD = 0.50
LONG_EXIT_THRESHOLD = 0.20
SHORT_ENTRY_THRESHOLD = -0.50
SHORT_EXIT_THRESHOLD = -0.20
STOP_LOSS_ATR_MULT = 2.0

# Risk parameters
CAPITAL = 100000
RISK_PER_TRADE = 0.02  # 2%
MAX_POSITION_PCT = 0.20  # 20%


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: str
    quantity: int
    stop_loss: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    quantity: int
    gross_pnl: float
    net_pnl: float
    pnl_pct: float
    exit_reason: str


class LivePaperTrader:
    """Live paper trading system."""

    def __init__(self, capital: float = CAPITAL):
        self.capital = capital
        self.available_capital = capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.signals_log: List[Dict] = []

        # Initialize indicator components
        self.indicator_universe = IndicatorUniverse()
        self.indicator_universe.load_all()
        self.indicator_calculator = IndicatorCalculator(universe=self.indicator_universe)
        self.indicator_normalizer = IndicatorNormalizer()

        # Create DNA and Super Indicator
        self.dna = self._create_dna()
        self.super_indicator = SuperIndicator(self.dna, normalizer=self.indicator_normalizer)

        # Previous SI values for crossover detection
        self.prev_si: Dict[str, float] = {}

        # Load previous state if exists
        self._load_state()

        print(f"Live Paper Trader initialized with ${capital:,.0f}")
        print(f"Watching {len(WATCHLIST)} stocks")

    def _create_dna(self) -> SuperIndicatorDNA:
        """Create DNA from strategy weights."""
        genes = {}
        for name, weight in STRATEGY_WEIGHTS.items():
            genes[name] = IndicatorGene(
                name=name,
                weight=weight,
                active=abs(weight) > 0.05,
                category='unknown'
            )
        return SuperIndicatorDNA(dna_id="LIVE", generation=0, run_id=0, genes=genes)

    def _load_state(self):
        """Load previous state from file."""
        state_file = 'paper_trading_state.json'
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                self.positions = {k: Position(**v) for k, v in state.get('positions', {}).items()}
                self.trades = [Trade(**t) for t in state.get('trades', [])]
                self.prev_si = state.get('prev_si', {})
                self.available_capital = state.get('available_capital', self.capital)
                print(f"Loaded state: {len(self.positions)} open positions, {len(self.trades)} completed trades")

    def _save_state(self):
        """Save current state to file."""
        state = {
            'positions': {k: asdict(v) for k, v in self.positions.items()},
            'trades': [asdict(t) for t in self.trades],
            'prev_si': self.prev_si,
            'available_capital': self.available_capital,
            'last_updated': str(datetime.now())
        }
        with open('paper_trading_state.json', 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def fetch_data(self, symbol: str, days: int = 100) -> Optional[pd.DataFrame]:
        """Fetch recent price data for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d")
            if df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

    def calculate_si(self, symbol: str, df: pd.DataFrame) -> Optional[float]:
        """Calculate Super Indicator value for a symbol."""
        try:
            # Calculate all indicators
            indicators = self.indicator_calculator.calculate_all(df)
            if indicators.empty:
                return None

            # Get active indicators
            active_indicators = self.dna.get_active_indicators()
            valid_active = [ind for ind in active_indicators if ind in indicators.columns]

            if not valid_active:
                return None

            # Normalize and calculate SI
            active_df = indicators[valid_active]
            normalized = self.indicator_normalizer.normalize_all(active_df, price_series=df['close'])

            if normalized.empty:
                return None

            si_series = self.super_indicator.calculate(normalized)
            return float(si_series.iloc[-1])

        except Exception as e:
            print(f"Error calculating SI for {symbol}: {e}")
            return None

    def get_signal(self, symbol: str, si: float) -> str:
        """Determine trading signal based on SI value."""
        prev_si = self.prev_si.get(symbol, 0)

        # Check if we have a position
        has_position = symbol in self.positions
        position_dir = self.positions[symbol].direction if has_position else None

        signal = "HOLD"

        if not has_position:
            # Entry signals
            if si > LONG_ENTRY_THRESHOLD and prev_si <= LONG_ENTRY_THRESHOLD:
                signal = "BUY"
            elif si < SHORT_ENTRY_THRESHOLD and prev_si >= SHORT_ENTRY_THRESHOLD:
                signal = "SELL"
        else:
            # Exit signals
            if position_dir == 'LONG' and si < LONG_EXIT_THRESHOLD:
                signal = "EXIT_LONG"
            elif position_dir == 'SHORT' and si > SHORT_EXIT_THRESHOLD:
                signal = "EXIT_SHORT"

        # Update previous SI
        self.prev_si[symbol] = si

        return signal

    def calculate_position_size(self, price: float, atr: float) -> int:
        """Calculate position size based on risk."""
        risk_amount = self.available_capital * RISK_PER_TRADE
        stop_distance = 2 * atr  # 2x ATR stop loss

        # Position size based on risk
        if stop_distance > 0:
            quantity = int(risk_amount / stop_distance)
        else:
            quantity = int(risk_amount / (price * 0.02))  # 2% stop

        # Cap at max position size
        max_quantity = int((self.available_capital * MAX_POSITION_PCT) / price)
        quantity = min(quantity, max_quantity)

        return max(1, quantity)

    def open_position(self, symbol: str, direction: str, price: float, atr: float):
        """Open a new position."""
        quantity = self.calculate_position_size(price, atr)
        position_value = quantity * price

        if position_value > self.available_capital:
            print(f"Insufficient capital for {symbol}")
            return

        stop_loss = price - (2 * atr) if direction == 'LONG' else price + (2 * atr)

        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            entry_time=str(datetime.now()),
            quantity=quantity,
            stop_loss=stop_loss,
            current_price=price,
            unrealized_pnl=0.0
        )

        self.positions[symbol] = position
        self.available_capital -= position_value

        print(f"🟢 OPENED {direction} {symbol}: {quantity} @ ${price:.2f} (SL: ${stop_loss:.2f})")

    def close_position(self, symbol: str, price: float, reason: str = "signal"):
        """Close an existing position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Calculate P&L
        if pos.direction == 'LONG':
            gross_pnl = (price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - price) * pos.quantity

        # Subtract commission (0.1% round trip)
        commission = (pos.entry_price + price) * pos.quantity * 0.001
        net_pnl = gross_pnl - commission
        pnl_pct = (net_pnl / (pos.entry_price * pos.quantity)) * 100

        trade = Trade(
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=price,
            entry_time=pos.entry_time,
            exit_time=str(datetime.now()),
            quantity=pos.quantity,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason
        )

        self.trades.append(trade)

        # Return capital
        self.available_capital += pos.quantity * price

        # Remove position
        del self.positions[symbol]

        result = "WIN" if net_pnl > 0 else "LOSS"
        print(f"🔴 CLOSED {pos.direction} {symbol}: ${net_pnl:+.2f} ({pnl_pct:+.1f}%) [{result}] - {reason}")

    def check_stop_losses(self, prices: Dict[str, float]):
        """Check and execute stop losses."""
        for symbol, pos in list(self.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue

            if pos.direction == 'LONG' and price <= pos.stop_loss:
                self.close_position(symbol, price, "stop_loss")
            elif pos.direction == 'SHORT' and price >= pos.stop_loss:
                self.close_position(symbol, price, "stop_loss")

    def update_positions(self, prices: Dict[str, float]):
        """Update unrealized P&L for open positions."""
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.entry_price)
            pos.current_price = price

            if pos.direction == 'LONG':
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - price) * pos.quantity

    def scan_market(self):
        """Scan all watchlist stocks for signals."""
        print(f"\n{'='*60}")
        print(f"MARKET SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        signals = []
        prices = {}
        atrs = {}

        for symbol in WATCHLIST:
            df = self.fetch_data(symbol)
            if df is None or len(df) < 50:
                continue

            # Get current price and ATR
            current_price = float(df['close'].iloc[-1])
            prices[symbol] = current_price

            # Calculate ATR
            high = df['high']
            low = df['low']
            close = df['close']
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            atrs[symbol] = atr

            # Calculate Super Indicator
            si = self.calculate_si(symbol, df)
            if si is None:
                continue

            # Get signal
            signal = self.get_signal(symbol, si)

            signal_data = {
                'symbol': symbol,
                'price': current_price,
                'si': si,
                'signal': signal,
                'atr': atr,
                'time': str(datetime.now())
            }
            signals.append(signal_data)

            # Print if actionable
            if signal != "HOLD":
                print(f"⚡ {symbol}: {signal} @ ${current_price:.2f} (SI: {si:.3f})")

        # Check stop losses
        self.check_stop_losses(prices)

        # Execute signals
        for sig in signals:
            if sig['signal'] == "BUY":
                if sig['symbol'] not in self.positions and len(self.positions) < 5:
                    self.open_position(sig['symbol'], 'LONG', sig['price'], sig['atr'])

            elif sig['signal'] == "SELL":
                if sig['symbol'] not in self.positions and len(self.positions) < 5:
                    self.open_position(sig['symbol'], 'SHORT', sig['price'], sig['atr'])

            elif sig['signal'] == "EXIT_LONG" or sig['signal'] == "EXIT_SHORT":
                self.close_position(sig['symbol'], sig['price'], "signal")

        # Update positions
        self.update_positions(prices)

        # Save state
        self._save_state()

        # Log signals
        self.signals_log.extend(signals)
        with open('signals_log.json', 'w') as f:
            json.dump(self.signals_log[-1000:], f, indent=2)  # Keep last 1000

        return signals

    def print_status(self):
        """Print current portfolio status."""
        print(f"\n{'='*60}")
        print("PORTFOLIO STATUS")
        print(f"{'='*60}")

        # Capital summary
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_realized = sum(t.net_pnl for t in self.trades)
        total_pnl = total_unrealized + total_realized

        print(f"Starting Capital:  ${self.capital:,.2f}")
        print(f"Available Capital: ${self.available_capital:,.2f}")
        print(f"Unrealized P&L:    ${total_unrealized:+,.2f}")
        print(f"Realized P&L:      ${total_realized:+,.2f}")
        print(f"Total P&L:         ${total_pnl:+,.2f}")
        print(f"Return:            {(total_pnl/self.capital)*100:+.2f}%")

        # Open positions
        if self.positions:
            print(f"\nOpen Positions ({len(self.positions)}):")
            for symbol, pos in self.positions.items():
                pnl_pct = (pos.unrealized_pnl / (pos.entry_price * pos.quantity)) * 100
                print(f"  {symbol}: {pos.direction} {pos.quantity} @ ${pos.entry_price:.2f} -> ${pos.current_price:.2f} ({pos.unrealized_pnl:+.2f}, {pnl_pct:+.1f}%)")

        # Trade summary
        if self.trades:
            winners = [t for t in self.trades if t.net_pnl > 0]
            losers = [t for t in self.trades if t.net_pnl <= 0]
            print(f"\nCompleted Trades: {len(self.trades)}")
            print(f"  Winners: {len(winners)} ({len(winners)/len(self.trades)*100:.1f}%)")
            print(f"  Losers: {len(losers)} ({len(losers)/len(self.trades)*100:.1f}%)")
            if winners:
                print(f"  Avg Winner: ${np.mean([t.net_pnl for t in winners]):,.2f}")
            if losers:
                print(f"  Avg Loser: ${np.mean([t.net_pnl for t in losers]):,.2f}")

        print(f"{'='*60}\n")

    def run_continuous(self, interval_minutes: int = 15):
        """Run continuous market scanning."""
        print(f"\n🚀 Starting Live Paper Trading System")
        print(f"Scanning every {interval_minutes} minutes")
        print(f"Press Ctrl+C to stop\n")

        try:
            while True:
                self.scan_market()
                self.print_status()

                print(f"Next scan in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n\nStopping paper trader...")
            self._save_state()
            self.print_status()
            print("State saved. Goodbye!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Live Paper Trading System')
    parser.add_argument('--scan', action='store_true', help='Run single market scan')
    parser.add_argument('--status', action='store_true', help='Show portfolio status')
    parser.add_argument('--continuous', type=int, default=0, help='Run continuously with N minute intervals')
    parser.add_argument('--reset', action='store_true', help='Reset all positions and trades')

    args = parser.parse_args()

    trader = LivePaperTrader()

    if args.reset:
        trader.positions = {}
        trader.trades = []
        trader.prev_si = {}
        trader.available_capital = trader.capital
        trader._save_state()
        print("Reset complete!")
        return

    if args.status:
        trader.print_status()
        return

    if args.continuous > 0:
        trader.run_continuous(interval_minutes=args.continuous)
        return

    # Default: single scan
    trader.scan_market()
    trader.print_status()


if __name__ == '__main__':
    main()
