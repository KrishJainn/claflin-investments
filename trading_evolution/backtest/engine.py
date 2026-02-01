"""
Deterministic Backtest Engine.

Event-driven, bar-by-bar backtesting with:
- Full reproducibility via config hashing
- Indian market cost model
- Intraday risk limits (daily stop, max trades, cooldown)
- End-of-day position flattening
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
import hashlib
import json
import uuid
import logging
import numpy as np
import pandas as pd
from enum import Enum

from .indian_costs import IndianCostModel, CostBreakdown, TradeType
from .result import Trade, BacktestResult

logger = logging.getLogger(__name__)


class FillAssumption(Enum):
    """Order fill assumptions."""
    NEXT_BAR_OPEN = "next_bar_open"
    SAME_BAR_CLOSE = "same_bar_close"
    VWAP = "vwap"


@dataclass
class BacktestConfig:
    """
    Immutable backtest configuration.
    
    All parameters that affect the backtest outcome are here
    to ensure reproducibility.
    """
    
    # Strategy identification
    strategy_name: str = "unnamed"
    strategy_version: str = "1.0"
    
    # Date range
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    
    # Capital and sizing
    initial_capital: float = 1_000_000.0
    position_size_pct: float = 0.10  # 10% of capital per trade
    max_position_value: float = 500_000.0  # Max ₹5L per position
    
    # Entry/Exit thresholds
    entry_threshold: float = 0.7
    exit_threshold: float = 0.3
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 4.0
    
    # Risk limits
    max_positions: int = 5
    max_trades_per_day: int = 10
    daily_loss_limit_pct: float = 0.02  # 2% of capital
    cooldown_bars: int = 0  # Bars to wait after stop loss
    
    # Intraday rules
    flatten_eod: bool = True
    flatten_time: time = field(default_factory=lambda: time(15, 20))
    no_new_trades_after: time = field(default_factory=lambda: time(15, 0))
    
    # Execution
    fill_assumption: FillAssumption = FillAssumption.SAME_BAR_CLOSE
    cost_model: str = "indian_intraday"
    
    # Reproducibility
    random_seed: int = 42
    
    def config_hash(self) -> str:
        """Generate deterministic hash of configuration."""
        # Create dict of all config values
        config_dict = {
            'strategy_name': self.strategy_name,
            'strategy_version': self.strategy_version,
            'initial_capital': self.initial_capital,
            'position_size_pct': self.position_size_pct,
            'max_position_value': self.max_position_value,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'stop_loss_atr_mult': self.stop_loss_atr_mult,
            'take_profit_atr_mult': self.take_profit_atr_mult,
            'max_positions': self.max_positions,
            'max_trades_per_day': self.max_trades_per_day,
            'daily_loss_limit_pct': self.daily_loss_limit_pct,
            'cooldown_bars': self.cooldown_bars,
            'flatten_eod': self.flatten_eod,
            'flatten_time': str(self.flatten_time),
            'no_new_trades_after': str(self.no_new_trades_after),
            'fill_assumption': self.fill_assumption.value,
            'cost_model': self.cost_model,
            'random_seed': self.random_seed,
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'strategy_version': self.strategy_version,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'initial_capital': self.initial_capital,
            'position_size_pct': self.position_size_pct,
            'max_position_value': self.max_position_value,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'stop_loss_atr_mult': self.stop_loss_atr_mult,
            'take_profit_atr_mult': self.take_profit_atr_mult,
            'max_positions': self.max_positions,
            'max_trades_per_day': self.max_trades_per_day,
            'daily_loss_limit_pct': self.daily_loss_limit_pct,
            'cooldown_bars': self.cooldown_bars,
            'flatten_eod': self.flatten_eod,
            'flatten_time': str(self.flatten_time),
            'no_new_trades_after': str(self.no_new_trades_after),
            'fill_assumption': self.fill_assumption.value,
            'cost_model': self.cost_model,
            'random_seed': self.random_seed,
        }


@dataclass
class Position:
    """Active position during backtest."""
    
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_time: datetime
    entry_price: float
    fill_entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    entry_signal: float
    entry_costs: CostBreakdown
    atr_at_entry: float = 0.0
    trade_id: str = ""
    indicator_snapshot: Dict[str, float] = field(default_factory=dict)


@dataclass 
class DailyState:
    """Track daily trading state for risk limits."""
    
    date: date
    trades_today: int = 0
    pnl_today: float = 0.0
    daily_limit_hit: bool = False
    cooldown_until_bar: int = 0


class BacktestEngine:
    """
    Deterministic, event-driven backtest engine.
    
    Features:
    - Bar-by-bar processing (no lookahead)
    - Configurable fill assumptions
    - Indian market cost model
    - Intraday risk limits
    - End-of-day flattening
    - Full reproducibility
    """
    
    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        
        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)
        
        # Initialize cost model
        if self.config.cost_model == "indian_intraday":
            self.cost_model = IndianCostModel.for_intraday()
        elif self.config.cost_model == "indian_delivery":
            self.cost_model = IndianCostModel.for_delivery()
        else:
            self.cost_model = IndianCostModel.for_intraday()
        
        # State
        self._positions: Dict[str, Position] = {}
        self._completed_trades: List[Trade] = []
        self._capital: float = self.config.initial_capital
        self._daily_state: DailyState = DailyState(date=date.today())
        self._current_bar_idx: int = 0
        self._equity_curve: List[Dict] = []
        
    def run(
        self,
        data: Dict[str, pd.DataFrame],
        signal_generator: Callable[[pd.DataFrame, int], float],
        indicator_extractor: Callable[[pd.DataFrame, int], Dict[str, float]] = None,
    ) -> BacktestResult:
        """
        Run backtest on multiple symbols.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame with indicators
            signal_generator: Function(df, bar_index) -> signal value (-1 to 1)
            indicator_extractor: Optional function to extract indicator snapshot
            
        Returns:
            BacktestResult with all trades and metrics
        """
        logger.info(f"Starting backtest: {self.config.strategy_name} v{self.config.strategy_version}")
        
        # Reset state
        self._reset()
        
        # Get date range from data
        all_timestamps = []
        for symbol, df in data.items():
            all_timestamps.extend(df.index.tolist())
        
        all_timestamps = sorted(set(all_timestamps))
        
        if not all_timestamps:
            raise ValueError("No data provided")
        
        start_date = all_timestamps[0].date() if hasattr(all_timestamps[0], 'date') else all_timestamps[0]
        end_date = all_timestamps[-1].date() if hasattr(all_timestamps[-1], 'date') else all_timestamps[-1]
        
        # Process bars
        bar_count = 0
        for timestamp in all_timestamps:
            bar_count += 1
            self._current_bar_idx = bar_count
            
            # Check for new day
            ts_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
            if ts_date != self._daily_state.date:
                self._new_day(ts_date)
            
            # Process each symbol at this timestamp
            for symbol, df in data.items():
                if timestamp in df.index:
                    self._process_bar(
                        symbol=symbol,
                        timestamp=timestamp,
                        bar=df.loc[timestamp],
                        df=df,
                        bar_idx=df.index.get_loc(timestamp),
                        signal_generator=signal_generator,
                        indicator_extractor=indicator_extractor,
                    )
            
            # Track equity
            self._track_equity(timestamp, data)
        
        # Close any remaining positions
        self._close_all_positions(all_timestamps[-1], data, "end_of_backtest")
        
        # Build result
        data_hash = self._hash_data(data)
        
        result = BacktestResult(
            run_id=str(uuid.uuid4())[:8],
            run_timestamp=datetime.now(),
            config_hash=self.config.config_hash(),
            strategy_name=self.config.strategy_name,
            strategy_version=self.config.strategy_version,
            start_date=start_date if isinstance(start_date, date) else date.fromisoformat(str(start_date)[:10]),
            end_date=end_date if isinstance(end_date, date) else date.fromisoformat(str(end_date)[:10]),
            symbols=list(data.keys()),
            data_hash=data_hash,
            bar_count=bar_count,
            trades=self._completed_trades,
            random_seed=self.config.random_seed,
        )
        
        # Calculate advanced metrics
        self._calculate_advanced_metrics(result)
        
        logger.info(f"Backtest complete: {result.total_trades} trades, Net P&L: ₹{result.net_pnl:,.2f}")
        
        return result
    
    def _reset(self):
        """Reset engine state."""
        self._positions = {}
        self._completed_trades = []
        self._capital = self.config.initial_capital
        self._daily_state = DailyState(date=date.today())
        self._current_bar_idx = 0
        self._equity_curve = []
        np.random.seed(self.config.random_seed)
    
    def _new_day(self, new_date: date):
        """Handle start of new trading day."""
        self._daily_state = DailyState(date=new_date)
        logger.debug(f"New trading day: {new_date}")
    
    def _process_bar(
        self,
        symbol: str,
        timestamp: datetime,
        bar: pd.Series,
        df: pd.DataFrame,
        bar_idx: int,
        signal_generator: Callable,
        indicator_extractor: Callable = None,
    ):
        """Process a single bar for a symbol."""
        
        # Extract OHLCV
        open_price = bar.get('Open', bar.get('open', 0))
        high = bar.get('High', bar.get('high', 0))
        low = bar.get('Low', bar.get('low', 0))
        close = bar.get('Close', bar.get('close', 0))
        volume = bar.get('Volume', bar.get('volume', 0))
        
        # Get ATR if available
        atr = bar.get('ATR_14', bar.get('atr', close * 0.02))  # Default 2% if no ATR
        
        # Check time of day
        bar_time = timestamp.time() if hasattr(timestamp, 'time') else time(12, 0)
        
        # Generate signal
        signal = signal_generator(df, bar_idx)
        
        # Get indicator snapshot
        indicators = {}
        if indicator_extractor:
            indicators = indicator_extractor(df, bar_idx)
        
        # Check existing position
        if symbol in self._positions:
            self._manage_position(symbol, timestamp, bar_time, high, low, close, signal, atr)
        else:
            # Check if we can open new position
            if self._can_open_position(bar_time, signal):
                self._try_open_position(symbol, timestamp, close, signal, atr, indicators)
    
    def _can_open_position(self, bar_time: time, signal: float) -> bool:
        """Check if we can open a new position."""
        # Check daily limits
        if self._daily_state.daily_limit_hit:
            return False
        
        if self._daily_state.trades_today >= self.config.max_trades_per_day:
            return False
        
        # Check position limits
        if len(self._positions) >= self.config.max_positions:
            return False
        
        # Check time of day
        if bar_time >= self.config.no_new_trades_after:
            return False
        
        # Check cooldown
        if self._current_bar_idx < self._daily_state.cooldown_until_bar:
            return False
        
        # Check signal strength
        if abs(signal) < self.config.entry_threshold:
            return False
        
        return True
    
    def _try_open_position(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        signal: float,
        atr: float,
        indicators: Dict,
    ):
        """Attempt to open a new position."""
        direction = 'LONG' if signal > 0 else 'SHORT'
        
        # Calculate position size
        position_value = min(
            self._capital * self.config.position_size_pct,
            self.config.max_position_value,
        )
        quantity = int(position_value / price)
        
        if quantity <= 0:
            return
        
        # Calculate stop loss and take profit
        if direction == 'LONG':
            stop_loss = price - (atr * self.config.stop_loss_atr_mult)
            take_profit = price + (atr * self.config.take_profit_atr_mult)
        else:
            stop_loss = price + (atr * self.config.stop_loss_atr_mult)
            take_profit = price - (atr * self.config.take_profit_atr_mult)
        
        # Calculate entry costs
        entry_costs = self.cost_model.calculate_entry_cost(price, quantity)
        
        # Create position
        position = Position(
            symbol=symbol,
            direction=direction,
            entry_time=timestamp,
            entry_price=price,
            fill_entry_price=entry_costs.fill_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_signal=signal,
            entry_costs=entry_costs,
            atr_at_entry=atr,
            trade_id=str(uuid.uuid4())[:8],
            indicator_snapshot=indicators,
        )
        
        self._positions[symbol] = position
        self._capital -= entry_costs.total_cost
        self._daily_state.trades_today += 1
        
        logger.debug(f"Opened {direction} position in {symbol} at {price:.2f}")
    
    def _manage_position(
        self,
        symbol: str,
        timestamp: datetime,
        bar_time: time,
        high: float,
        low: float,
        close: float,
        signal: float,
        atr: float,
    ):
        """Manage existing position - check stops and exit signals."""
        position = self._positions[symbol]
        
        # Check for end of day flatten
        if self.config.flatten_eod and bar_time >= self.config.flatten_time:
            self._close_position(symbol, timestamp, close, signal, "eod_flatten")
            return
        
        # Check stop loss
        if position.direction == 'LONG':
            if low <= position.stop_loss:
                self._close_position(symbol, timestamp, position.stop_loss, signal, "stop_loss")
                self._daily_state.cooldown_until_bar = self._current_bar_idx + self.config.cooldown_bars
                return
            if high >= position.take_profit:
                self._close_position(symbol, timestamp, position.take_profit, signal, "take_profit")
                return
        else:  # SHORT
            if high >= position.stop_loss:
                self._close_position(symbol, timestamp, position.stop_loss, signal, "stop_loss")
                self._daily_state.cooldown_until_bar = self._current_bar_idx + self.config.cooldown_bars
                return
            if low <= position.take_profit:
                self._close_position(symbol, timestamp, position.take_profit, signal, "take_profit")
                return
        
        # Check exit signal
        if position.direction == 'LONG' and signal < -self.config.exit_threshold:
            self._close_position(symbol, timestamp, close, signal, "signal")
        elif position.direction == 'SHORT' and signal > self.config.exit_threshold:
            self._close_position(symbol, timestamp, close, signal, "signal")
    
    def _close_position(
        self,
        symbol: str,
        timestamp: datetime,
        exit_price: float,
        exit_signal: float,
        exit_reason: str,
    ):
        """Close a position and record the trade."""
        position = self._positions.pop(symbol)
        
        # Calculate exit costs
        exit_costs = self.cost_model.calculate_exit_cost(exit_price, position.quantity)
        
        # Calculate P&L
        if position.direction == 'LONG':
            gross_pnl = (exit_costs.fill_price - position.fill_entry_price) * position.quantity
        else:
            gross_pnl = (position.fill_entry_price - exit_costs.fill_price) * position.quantity
        
        total_costs = position.entry_costs.total_cost + exit_costs.total_cost
        net_pnl = gross_pnl - exit_costs.total_cost  # Entry costs already deducted from capital
        
        # Calculate risk amount for R-multiple
        risk_amount = abs(position.entry_price - position.stop_loss) * position.quantity
        
        # Create trade record
        trade = Trade(
            trade_id=position.trade_id,
            symbol=symbol,
            direction=position.direction,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=exit_price,
            fill_entry_price=position.fill_entry_price,
            fill_exit_price=exit_costs.fill_price,
            quantity=position.quantity,
            position_value=position.entry_price * position.quantity,
            gross_pnl=gross_pnl,
            total_costs=total_costs,
            net_pnl=net_pnl,
            pnl_pct=net_pnl / (position.entry_price * position.quantity),
            exit_reason=exit_reason,
            entry_signal_value=position.entry_signal,
            exit_signal_value=exit_signal,
            stop_loss_price=position.stop_loss,
            take_profit_price=position.take_profit,
            risk_amount=risk_amount,
            atr_at_entry=position.atr_at_entry,
            indicator_snapshot=position.indicator_snapshot,
        )
        
        self._completed_trades.append(trade)
        self._capital += (position.entry_price * position.quantity) + net_pnl
        self._daily_state.pnl_today += net_pnl
        
        # Check daily loss limit
        if self._daily_state.pnl_today <= -self.config.initial_capital * self.config.daily_loss_limit_pct:
            self._daily_state.daily_limit_hit = True
            logger.warning(f"Daily loss limit hit: ₹{self._daily_state.pnl_today:,.2f}")
        
        logger.debug(f"Closed {position.direction} {symbol}: Net P&L ₹{net_pnl:,.2f} ({exit_reason})")
    
    def _close_all_positions(
        self,
        timestamp: datetime,
        data: Dict[str, pd.DataFrame],
        reason: str,
    ):
        """Close all open positions."""
        symbols_to_close = list(self._positions.keys())
        
        for symbol in symbols_to_close:
            # Get last price
            if symbol in data and len(data[symbol]) > 0:
                last_bar = data[symbol].iloc[-1]
                close = last_bar.get('Close', last_bar.get('close', 0))
            else:
                close = self._positions[symbol].entry_price
            
            self._close_position(symbol, timestamp, close, 0.0, reason)
    
    def _track_equity(self, timestamp: datetime, data: Dict[str, pd.DataFrame]):
        """Track equity curve."""
        # Calculate current unrealized P&L
        unrealized_pnl = 0.0
        
        for symbol, position in self._positions.items():
            if symbol in data and timestamp in data[symbol].index:
                current_price = data[symbol].loc[timestamp].get('Close', position.entry_price)
                if position.direction == 'LONG':
                    unrealized_pnl += (current_price - position.fill_entry_price) * position.quantity
                else:
                    unrealized_pnl += (position.fill_entry_price - current_price) * position.quantity
        
        total_equity = self._capital + unrealized_pnl
        
        self._equity_curve.append({
            'timestamp': timestamp,
            'capital': self._capital,
            'unrealized_pnl': unrealized_pnl,
            'total_equity': total_equity,
            'positions': len(self._positions),
        })
    
    def _calculate_advanced_metrics(self, result: BacktestResult):
        """Calculate advanced risk metrics."""
        if not self._equity_curve:
            return
        
        equity_df = pd.DataFrame(self._equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        returns = equity_df['total_equity'].pct_change().dropna()
        
        if len(returns) > 0:
            # Sharpe ratio (assuming 252 trading days, 78 5-min bars per day)
            bars_per_year = 252 * 78
            result.sharpe_ratio = np.sqrt(bars_per_year) * returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
            result.sortino_ratio = np.sqrt(bars_per_year) * returns.mean() / downside_std if downside_std > 0 else 0
            
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdowns = (cumulative - running_max) / running_max
            result.max_drawdown_pct = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
            result.max_drawdown = result.max_drawdown_pct * self.config.initial_capital
            
            # Calmar ratio
            if result.max_drawdown_pct > 0:
                annual_return = returns.mean() * bars_per_year
                result.calmar_ratio = annual_return / result.max_drawdown_pct
        
        result.equity_curve = equity_df
    
    def _hash_data(self, data: Dict[str, pd.DataFrame]) -> str:
        """Generate hash of input data for reproducibility."""
        hash_parts = []
        for symbol in sorted(data.keys()):
            df = data[symbol]
            hash_parts.append(f"{symbol}:{len(df)}:{df.index[0]}:{df.index[-1]}")
        
        data_str = "|".join(hash_parts)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
