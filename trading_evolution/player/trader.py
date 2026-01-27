"""
Player Agent (Trader) module.

The Player executes paper trades based on Super Indicator signals.
Handles:
- Signal processing
- Trade entry/exit decisions
- Position management
- Stop loss monitoring
"""

import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from .portfolio import Portfolio, Position, Trade
from .risk_manager import RiskManager, RiskParameters
from .execution import ExecutionEngine, ExecutionResult
from ..super_indicator.core import SuperIndicator
from ..super_indicator.signals import SignalGenerator, SignalType, PositionState

logger = logging.getLogger(__name__)


class Player:
    """
    Player Agent: Executes paper trades based on Super Indicator signals.

    Responsibilities:
    - Monitor Super Indicator signals for entry/exit
    - Execute trades with realistic costs
    - Manage open positions and stop losses
    - Track all trades for Coach analysis

    Signal Thresholds:
    - LONG_ENTRY: SI > +0.7
    - LONG_EXIT: SI < +0.3
    - SHORT_ENTRY: SI < -0.7
    - SHORT_EXIT: SI > -0.3
    """

    def __init__(self,
                 portfolio: Portfolio = None,
                 risk_manager: RiskManager = None,
                 execution: ExecutionEngine = None,
                 initial_capital: float = 100_000.0,
                 slippage_pct: float = 0.001,
                 allow_short: bool = True):
        """
        Initialize Player agent.

        Args:
            portfolio: Portfolio manager (creates new if None)
            risk_manager: Risk manager (creates new if None)
            execution: Execution engine (creates new if None)
            initial_capital: Starting capital if creating new portfolio
            slippage_pct: Slippage percentage
            allow_short: Whether to allow short positions
        """
        self.portfolio = portfolio or Portfolio(initial_capital)
        self.risk_manager = risk_manager or RiskManager()
        self.execution = execution or ExecutionEngine(slippage_pct=slippage_pct)
        self.signal_generator = SignalGenerator()

        self.allow_short = allow_short

        # Current state
        self.current_dna_id: str = ""
        self.current_generation: int = 0
        self.current_run_id: int = 0

        # Tracking
        self.signals_processed: int = 0
        self.trades_executed: int = 0

    def process_signal(self,
                       symbol: str,
                       signal: SignalType,
                       current_price: float,
                       timestamp: datetime,
                       high: float,
                       low: float,
                       atr: float = 1.0,
                       si_value: float = 0.0) -> Optional[Trade]:
        """
        Process an externally generated signal.
        """
        self.signals_processed += 1
        
        # Update existing position price
        if self.portfolio.has_position(symbol):
            self.portfolio.update_prices({symbol: current_price}, timestamp)

        # Check for stop loss hits first
        trade = self._check_stop_loss(symbol, high, low, timestamp, si_value)
        if trade:
            return trade

        # Get current position state
        position = self.portfolio.get_position(symbol)
        if position:
            if position.direction == 'LONG':
                position_state = PositionState.LONG
            else:
                position_state = PositionState.SHORT
        else:
            position_state = PositionState.FLAT

        # Process signal
        indicator_snapshot = {'SI': si_value, 'ATR': atr}
        
        if signal == SignalType.LONG_ENTRY and position_state == PositionState.FLAT:
            self._enter_position(
                symbol, 'LONG', current_price, timestamp,
                atr, si_value, indicator_snapshot
            )

        elif signal == SignalType.SHORT_ENTRY and position_state == PositionState.FLAT:
            if self.allow_short:
                self._enter_position(
                    symbol, 'SHORT', current_price, timestamp,
                    atr, si_value, indicator_snapshot
                )

        elif signal == SignalType.LONG_EXIT and position_state == PositionState.LONG:
            trade = self._exit_position(
                symbol, current_price, timestamp,
                'signal', si_value
            )
            return trade

        elif signal == SignalType.SHORT_EXIT and position_state == PositionState.SHORT:
            trade = self._exit_position(
                symbol, current_price, timestamp,
                'signal', si_value
            )
            return trade

        return None

    def process_bar(self,
                    symbol: str,
                    timestamp: datetime,
                    ohlcv: Dict[str, float],
                    super_indicator_value: float,
                    atr: float,
                    indicator_snapshot: Dict[str, float] = None) -> Optional[Trade]:
        """
        Process a single bar of data.

        This is the main entry point called for each bar during backtesting.

        Args:
            symbol: Stock ticker
            timestamp: Bar timestamp
            ohlcv: Dict with 'open', 'high', 'low', 'close', 'volume'
            super_indicator_value: Current Super Indicator value [-1, 1]
            atr: Average True Range for stop calculation
            indicator_snapshot: Optional dict of indicator values at this bar

        Returns:
            Trade if a position was closed, None otherwise
        """
        self.signals_processed += 1
        current_price = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']

        # Update existing position price
        if self.portfolio.has_position(symbol):
            self.portfolio.update_prices({symbol: current_price}, timestamp)

        # Check for stop loss hits first
        trade = self._check_stop_loss(symbol, high, low, timestamp, super_indicator_value)
        if trade:
            return trade

        # Get current position state
        position = self.portfolio.get_position(symbol)
        if position:
            if position.direction == 'LONG':
                position_state = PositionState.LONG
            else:
                position_state = PositionState.SHORT
        else:
            position_state = PositionState.FLAT

        # Determine signal
        signal = self._determine_signal(super_indicator_value, position_state)

        # Process signal
        if signal == SignalType.LONG_ENTRY and position_state == PositionState.FLAT:
            self._enter_position(
                symbol, 'LONG', current_price, timestamp,
                atr, super_indicator_value, indicator_snapshot
            )

        elif signal == SignalType.SHORT_ENTRY and position_state == PositionState.FLAT:
            if self.allow_short:
                self._enter_position(
                    symbol, 'SHORT', current_price, timestamp,
                    atr, super_indicator_value, indicator_snapshot
                )

        elif signal == SignalType.LONG_EXIT and position_state == PositionState.LONG:
            trade = self._exit_position(
                symbol, current_price, timestamp,
                'signal', super_indicator_value
            )
            return trade

        elif signal == SignalType.SHORT_EXIT and position_state == PositionState.SHORT:
            trade = self._exit_position(
                symbol, current_price, timestamp,
                'signal', super_indicator_value
            )
            return trade

        return None

    def _determine_signal(self, si_value: float,
                          position_state: PositionState) -> SignalType:
        """Determine trading signal from Super Indicator value."""
        # Entry signals
        if position_state == PositionState.FLAT:
            if si_value > 0.7:
                return SignalType.LONG_ENTRY
            elif si_value < -0.7:
                return SignalType.SHORT_ENTRY

        # Exit signals
        elif position_state == PositionState.LONG:
            if si_value < 0.3:
                return SignalType.LONG_EXIT

        elif position_state == PositionState.SHORT:
            if si_value > -0.3:
                return SignalType.SHORT_EXIT

        return SignalType.HOLD

    def _enter_position(self,
                        symbol: str,
                        direction: str,
                        price: float,
                        timestamp: datetime,
                        atr: float,
                        signal_strength: float,
                        indicator_snapshot: Dict = None):
        """Enter a new position."""
        # Check if we can open position
        can_open, reason = self.risk_manager.can_open_position(
            self.portfolio.num_positions
        )
        if not can_open:
            logger.debug(f"Cannot open position: {reason}")
            return

        # Calculate stop loss
        stop_loss = self.risk_manager.calculate_stop_loss(
            price, atr, direction
        )

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            self.portfolio.get_equity(),
            price,
            stop_loss,
            direction
        )

        if position_size <= 0:
            logger.debug(f"Position size is 0 for {symbol}")
            return

        # Execute entry
        result = self.execution.execute_entry(direction, price, position_size)
        if not result.executed:
            logger.debug(f"Entry execution failed: {result.reason}")
            return

        # Calculate take profit
        take_profit = self.risk_manager.calculate_take_profit(
            result.fill_price, stop_loss, direction
        )

        # Create position
        position = Position(
            symbol=symbol,
            direction=direction,
            quantity=position_size,
            entry_price=result.fill_price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=result.fill_price,
            atr_at_entry=atr,
            signal_strength=signal_strength,
            dna_id=self.current_dna_id,
            generation=self.current_generation
        )

        # Open position in portfolio
        if self.portfolio.open_position(position):
            self.trades_executed += 1
            logger.debug(f"Opened {direction} position in {symbol} at {result.fill_price:.2f}")

    def _exit_position(self,
                       symbol: str,
                       price: float,
                       timestamp: datetime,
                       exit_reason: str,
                       signal_at_exit: float) -> Optional[Trade]:
        """Exit an existing position."""
        position = self.portfolio.get_position(symbol)
        if not position:
            return None

        # Execute exit
        result = self.execution.execute_exit(
            position.direction, price, position.quantity
        )
        if not result.executed:
            logger.debug(f"Exit execution failed: {result.reason}")
            return None

        # Generate trade ID
        trade_id = str(uuid.uuid4())[:8]

        # Close position in portfolio
        trade = self.portfolio.close_position(
            symbol=symbol,
            exit_price=result.fill_price,
            exit_time=timestamp,
            exit_reason=exit_reason,
            signal_at_exit=signal_at_exit,
            commission=result.commission,
            slippage=result.slippage_cost,
            trade_id=trade_id,
            run_id=self.current_run_id
        )

        if trade:
            logger.debug(
                f"Closed {position.direction} in {symbol} at {result.fill_price:.2f}, "
                f"P&L: ${trade.net_pnl:.2f} ({trade.net_pnl_pct:.2%})"
            )

        return trade

    def _check_stop_loss(self,
                         symbol: str,
                         high: float,
                         low: float,
                         timestamp: datetime,
                         signal_at_exit: float) -> Optional[Trade]:
        """Check if stop loss was hit during the bar."""
        position = self.portfolio.get_position(symbol)
        if not position:
            return None

        stop_hit = False
        exit_price = position.stop_loss

        if position.direction == 'LONG':
            # For long, stop is hit if low touches stop
            if low <= position.stop_loss:
                stop_hit = True
        else:
            # For short, stop is hit if high touches stop
            if high >= position.stop_loss:
                stop_hit = True

        if stop_hit:
            return self._exit_position(
                symbol, exit_price, timestamp,
                'stop_loss', signal_at_exit
            )

        return None

    def close_all_positions(self,
                            timestamp: datetime,
                            prices: Dict[str, float],
                            reason: str = 'end_of_data') -> List[Trade]:
        """
        Close all open positions.

        Used at end of backtest or when resetting.

        Args:
            timestamp: Exit timestamp
            prices: Dict of symbol -> price
            reason: Exit reason

        Returns:
            List of closed trades
        """
        trades = []
        symbols = list(self.portfolio.positions.keys())

        for symbol in symbols:
            price = prices.get(symbol, self.portfolio.positions[symbol].current_price)
            trade = self._exit_position(symbol, price, timestamp, reason, 0.0)
            if trade:
                trades.append(trade)

        return trades

    def run_backtest(self,
                     market_data: Dict[str, pd.DataFrame],
                     normalized_indicators: Dict[str, pd.DataFrame],
                     super_indicator: SuperIndicator,
                     atr_column: str = 'ATR_14') -> List[Trade]:
        """
        Run a complete backtest.

        Args:
            market_data: Dict of symbol -> OHLCV DataFrame
            normalized_indicators: Dict of symbol -> normalized indicators DataFrame
            super_indicator: SuperIndicator instance
            atr_column: Name of ATR column in indicators

        Returns:
            List of all completed trades
        """
        all_trades = []

        # Get common date range
        all_dates = set()
        for df in market_data.values():
            all_dates.update(df.index)
        dates = sorted(all_dates)

        # Process each date
        for timestamp in dates:
            for symbol, ohlcv_df in market_data.items():
                if timestamp not in ohlcv_df.index:
                    continue

                # Get OHLCV
                bar = ohlcv_df.loc[timestamp]
                ohlcv = {
                    'open': bar['open'],
                    'high': bar['high'],
                    'low': bar['low'],
                    'close': bar['close'],
                    'volume': bar.get('volume', 0)
                }

                # Get normalized indicators
                if symbol in normalized_indicators:
                    ind_df = normalized_indicators[symbol]
                    if timestamp in ind_df.index:
                        # Calculate Super Indicator
                        si_value = super_indicator.calculate_at_timestamp(
                            ind_df, timestamp
                        )

                        # Get ATR
                        atr = ind_df.loc[timestamp, atr_column] if atr_column in ind_df.columns else 1.0
                        if pd.isna(atr):
                            atr = 1.0

                        # Get indicator snapshot
                        snapshot = ind_df.loc[timestamp].to_dict()

                        # Process bar
                        trade = self.process_bar(
                            symbol, timestamp, ohlcv,
                            si_value, atr, snapshot
                        )

                        if trade:
                            all_trades.append(trade)

        # Close remaining positions
        final_prices = {}
        for symbol, df in market_data.items():
            if len(df) > 0:
                final_prices[symbol] = df.iloc[-1]['close']

        final_trades = self.close_all_positions(
            dates[-1] if dates else datetime.now(),
            final_prices
        )
        all_trades.extend(final_trades)

        return all_trades

    def get_all_trades(self) -> List[Trade]:
        """Get all completed trades."""
        return self.portfolio.completed_trades.copy()

    def get_performance(self) -> Dict:
        """Get performance metrics."""
        return self.portfolio.get_performance_metrics()

    def reset(self):
        """Reset player state for new backtest."""
        self.portfolio.reset()
        self.signals_processed = 0
        self.trades_executed = 0

    def set_dna(self, dna_id: str, generation: int, run_id: int):
        """Set current DNA for trade tracking."""
        self.current_dna_id = dna_id
        self.current_generation = generation
        self.current_run_id = run_id
