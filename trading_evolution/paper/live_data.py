"""
Live Market Data Manager with Bar Building.

Provides:
- Real-time data streaming (via yfinance for now)
- Bar building from tick/1m data to higher timeframes
- Market hours awareness
- Data quality validation
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import threading
import queue
import logging
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketStatus(Enum):
    """Current market status."""
    PRE_MARKET = "pre_market"
    OPEN = "open"
    CLOSED = "closed"
    HOLIDAY = "holiday"


@dataclass
class Bar:
    """A single OHLCV bar."""
    
    symbol: str
    timestamp: datetime
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Additional fields
    vwap: float = 0.0
    num_trades: int = 0
    is_complete: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'interval': self.interval,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'vwap': self.vwap,
            'is_complete': self.is_complete,
        }


@dataclass
class MarketConfig:
    """Market configuration for Indian markets (NSE)."""
    
    timezone: str = "Asia/Kolkata"
    
    # Market hours
    pre_market_start: time = field(default_factory=lambda: time(9, 0))
    market_open: time = field(default_factory=lambda: time(9, 15))
    market_close: time = field(default_factory=lambda: time(15, 30))
    post_market_end: time = field(default_factory=lambda: time(16, 0))
    
    # Bar settings
    default_interval: str = "5m"
    
    # Holidays (simplified - should load from NSE calendar)
    holidays_2026: List[date] = field(default_factory=list)


class BarBuilder:
    """
    Builds higher timeframe bars from lower timeframe data.
    
    Example: Aggregate 1m bars into 5m bars.
    """
    
    def __init__(
        self,
        symbol: str,
        source_interval: str = "1m",
        target_interval: str = "5m",
    ):
        """
        Initialize bar builder.
        
        Args:
            symbol: Stock symbol
            source_interval: Input bar interval
            target_interval: Output bar interval
        """
        self.symbol = symbol
        self.source_interval = source_interval
        self.target_interval = target_interval
        
        # Parse intervals to minutes
        self.source_minutes = self._parse_interval(source_interval)
        self.target_minutes = self._parse_interval(target_interval)
        
        if self.target_minutes % self.source_minutes != 0:
            raise ValueError(f"Target interval must be multiple of source")
        
        self.bars_per_target = self.target_minutes // self.source_minutes
        
        # Current building bar
        self._current_bar: Optional[Bar] = None
        self._source_count: int = 0
        self._volume_price_sum: float = 0.0
        
        # Completed bars
        self._completed_bars: List[Bar] = []
        
        # Callbacks
        self._on_bar_complete: Optional[Callable[[Bar], None]] = None
    
    def _parse_interval(self, interval: str) -> int:
        """Parse interval string to minutes."""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        else:
            raise ValueError(f"Unknown interval format: {interval}")
    
    def set_callback(self, callback: Callable[[Bar], None]):
        """Set callback for completed bars."""
        self._on_bar_complete = callback
    
    def add_bar(self, bar: Bar) -> Optional[Bar]:
        """
        Add a source bar and potentially emit a target bar.
        
        Args:
            bar: Source interval bar
            
        Returns:
            Completed target bar if ready, None otherwise
        """
        if self._current_bar is None:
            # Start new target bar
            self._current_bar = Bar(
                symbol=self.symbol,
                timestamp=self._round_timestamp(bar.timestamp),
                interval=self.target_interval,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                is_complete=False,
            )
            self._source_count = 1
            self._volume_price_sum = bar.volume * bar.close
        else:
            # Update current bar
            self._current_bar.high = max(self._current_bar.high, bar.high)
            self._current_bar.low = min(self._current_bar.low, bar.low)
            self._current_bar.close = bar.close
            self._current_bar.volume += bar.volume
            self._source_count += 1
            self._volume_price_sum += bar.volume * bar.close
        
        # Check if complete
        if self._source_count >= self.bars_per_target:
            return self._complete_bar()
        
        return None
    
    def _round_timestamp(self, ts: datetime) -> datetime:
        """Round timestamp to target interval boundary."""
        minutes = ts.hour * 60 + ts.minute
        rounded_minutes = (minutes // self.target_minutes) * self.target_minutes
        return ts.replace(
            hour=rounded_minutes // 60,
            minute=rounded_minutes % 60,
            second=0,
            microsecond=0,
        )
    
    def _complete_bar(self) -> Bar:
        """Complete the current bar and return it."""
        bar = self._current_bar
        bar.is_complete = True
        
        # Calculate VWAP
        if bar.volume > 0:
            bar.vwap = self._volume_price_sum / bar.volume
        else:
            bar.vwap = bar.close
        
        bar.num_trades = self._source_count
        
        self._completed_bars.append(bar)
        
        # Callback
        if self._on_bar_complete:
            self._on_bar_complete(bar)
        
        # Reset
        self._current_bar = None
        self._source_count = 0
        self._volume_price_sum = 0.0
        
        return bar
    
    def force_complete(self) -> Optional[Bar]:
        """Force complete the current bar (e.g., at market close)."""
        if self._current_bar is not None:
            return self._complete_bar()
        return None
    
    def get_completed_bars(self) -> List[Bar]:
        """Get all completed bars."""
        return self._completed_bars.copy()
    
    def get_bars_as_dataframe(self) -> pd.DataFrame:
        """Get completed bars as DataFrame."""
        if not self._completed_bars:
            return pd.DataFrame()
        
        data = [bar.to_dict() for bar in self._completed_bars]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df


class LiveDataManager:
    """
    Manages live market data for paper trading.
    
    Features:
    - Fetches latest data from yfinance
    - Builds bars at specified intervals
    - Handles market hours
    - Provides data callbacks
    """
    
    def __init__(
        self,
        symbols: List[str],
        interval: str = "5m",
        config: MarketConfig = None,
    ):
        """
        Initialize live data manager.
        
        Args:
            symbols: List of symbols to track
            interval: Bar interval
            config: Market configuration
        """
        self.symbols = symbols
        self.interval = interval
        self.config = config or MarketConfig()
        
        # Bar builders per symbol
        self._builders: Dict[str, BarBuilder] = {}
        for symbol in symbols:
            self._builders[symbol] = BarBuilder(
                symbol=symbol,
                source_interval="1m",
                target_interval=interval,
            )
        
        # Latest bars per symbol
        self._latest_bars: Dict[str, Bar] = {}
        
        # Historical data cache
        self._history: Dict[str, pd.DataFrame] = {}
        
        # Callbacks
        self._on_bar: Optional[Callable[[Bar], None]] = None
        
        # State
        self._is_running = False
        self._last_fetch: Optional[datetime] = None
    
    def set_on_bar_callback(self, callback: Callable[[Bar], None]):
        """Set callback for new bars."""
        self._on_bar = callback
        for builder in self._builders.values():
            builder.set_callback(callback)
    
    def get_market_status(self) -> MarketStatus:
        """Get current market status."""
        try:
            import pytz
            tz = pytz.timezone(self.config.timezone)
            now = datetime.now(tz)
        except ImportError:
            now = datetime.now()
        
        current_time = now.time()
        
        # Check if holiday
        if now.date() in self.config.holidays_2026:
            return MarketStatus.HOLIDAY
        
        # Check if weekend
        if now.weekday() >= 5:
            return MarketStatus.CLOSED
        
        # Check time
        if current_time < self.config.pre_market_start:
            return MarketStatus.CLOSED
        elif current_time < self.config.market_open:
            return MarketStatus.PRE_MARKET
        elif current_time <= self.config.market_close:
            return MarketStatus.OPEN
        else:
            return MarketStatus.CLOSED
    
    def fetch_latest(self) -> Dict[str, Bar]:
        """
        Fetch latest bars for all symbols.
        
        Returns:
            Dictionary of symbol -> latest Bar
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance required for live data")
        
        results = {}
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="1d", interval=self.interval)
                
                if len(df) > 0:
                    latest = df.iloc[-1]
                    bar = Bar(
                        symbol=symbol,
                        timestamp=df.index[-1].to_pydatetime(),
                        interval=self.interval,
                        open=float(latest['Open']),
                        high=float(latest['High']),
                        low=float(latest['Low']),
                        close=float(latest['Close']),
                        volume=int(latest['Volume']),
                        is_complete=True,
                    )
                    results[symbol] = bar
                    self._latest_bars[symbol] = bar
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        
        self._last_fetch = datetime.now()
        return results
    
    def fetch_history(
        self,
        days: int = 5,
        add_indicators: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for backtesting comparison.
        
        Args:
            days: Number of days of history
            add_indicators: Whether to add technical indicators
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        results = {}
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=f"{days}d", interval=self.interval)
                
                if len(df) > 0:
                    # Clean columns
                    df.columns = [c.title() for c in df.columns]
                    
                    if add_indicators:
                        df = self._add_indicators(df)
                    
                    results[symbol] = df
                    self._history[symbol] = df
                    
            except Exception as e:
                logger.error(f"Error fetching history for {symbol}: {e}")
        
        return results
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame."""
        if len(df) < 20:
            return df
        
        df = df.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(14).mean()
        
        # Moving averages
        df['SMA_20'] = close.rolling(20).mean()
        df['EMA_9'] = close.ewm(span=9).mean()
        df['EMA_21'] = close.ewm(span=21).mean()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        return df
    
    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """Get latest bar for a symbol."""
        return self._latest_bars.get(symbol)
    
    def get_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol."""
        return self._history.get(symbol)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        bar = self._latest_bars.get(symbol)
        return bar.close if bar else None
    
    def get_bid_ask_spread(self, symbol: str) -> tuple:
        """
        Estimate bid-ask spread based on volatility.
        
        Returns:
            Tuple of (bid, ask, spread_pct)
        """
        bar = self._latest_bars.get(symbol)
        if not bar:
            return None, None, None
        
        # Estimate spread based on price (higher priced stocks = tighter spread %)
        # This is a rough approximation
        price = bar.close
        
        if price < 100:
            spread_pct = 0.002  # 0.2%
        elif price < 500:
            spread_pct = 0.001  # 0.1%
        elif price < 2000:
            spread_pct = 0.0005  # 0.05%
        else:
            spread_pct = 0.0003  # 0.03%
        
        half_spread = price * spread_pct / 2
        bid = price - half_spread
        ask = price + half_spread
        
        return bid, ask, spread_pct
