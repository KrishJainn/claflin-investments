"""
Portfolio Management module.

Tracks:
- Cash balance
- Open positions
- Equity curve
- Performance metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from decimal import Decimal
import pandas as pd
import numpy as np


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: Optional[float] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    atr_at_entry: float = 0.0
    signal_strength: float = 0.0
    dna_id: str = ""
    generation: int = 0

    def update_price(self, price: float):
        """Update position with current price."""
        self.current_price = price
        if self.direction == 'LONG':
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

    def get_return_pct(self) -> float:
        """Get percentage return on position."""
        if self.direction == 'LONG':
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price


@dataclass
class Trade:
    """Completed trade record."""
    trade_id: str
    symbol: str
    direction: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    net_pnl_pct: float
    holding_period_bars: int
    exit_reason: str
    signal_at_entry: float
    signal_at_exit: float
    stop_loss: float
    take_profit: Optional[float]
    atr_at_entry: float
    dna_id: str
    generation: int
    run_id: int
    indicator_snapshot: Dict = field(default_factory=dict)


class Portfolio:
    """
    Portfolio manager for paper trading.

    Tracks:
    - Cash balance
    - Open positions (long and short)
    - Completed trades
    - Equity curve
    """

    def __init__(self, initial_capital: float = 100_000.0):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting cash balance
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.completed_trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self._peak_equity = initial_capital

    def get_equity(self) -> float:
        """Get current total equity (cash + position value)."""
        position_value = sum(
            p.current_price * p.quantity if p.direction == 'LONG'
            else p.entry_price * p.quantity - p.unrealized_pnl
            for p in self.positions.values()
        )
        return self.cash + position_value

    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def get_realized_pnl(self) -> float:
        """Get total realized P&L."""
        return sum(t.net_pnl for t in self.completed_trades)

    def open_position(self, position: Position) -> bool:
        """
        Open a new position.

        Args:
            position: Position to open

        Returns:
            True if successful
        """
        if position.symbol in self.positions:
            return False  # Already have position in this symbol

        # Check if we have enough cash
        required_cash = position.entry_price * position.quantity
        if position.direction == 'LONG':
            if required_cash > self.cash:
                return False
            self.cash -= required_cash
        else:
            # For short, we need margin (simplified: use same as long)
            if required_cash > self.cash:
                return False
            # Don't deduct cash for short (simplified model)

        self.positions[position.symbol] = position
        return True

    def close_position(self, symbol: str, exit_price: float,
                       exit_time: datetime, exit_reason: str,
                       signal_at_exit: float, commission: float,
                       slippage: float, trade_id: str,
                       run_id: int) -> Optional[Trade]:
        """
        Close an existing position.

        Args:
            symbol: Symbol to close
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_reason: Reason for exit
            signal_at_exit: Super Indicator value at exit
            commission: Commission charged
            slippage: Slippage cost
            trade_id: Unique trade ID
            run_id: Evolution run ID

        Returns:
            Trade record if successful, None otherwise
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # Calculate P&L
        if position.direction == 'LONG':
            gross_pnl = (exit_price - position.entry_price) * position.quantity
            # Return cash from sale
            self.cash += exit_price * position.quantity
        else:
            gross_pnl = (position.entry_price - exit_price) * position.quantity
            # For short, adjust cash by profit/loss
            self.cash += gross_pnl

        # Apply costs
        net_pnl = gross_pnl - commission - slippage

        # Calculate percentage return
        position_value = position.entry_price * position.quantity
        net_pnl_pct = net_pnl / position_value if position_value > 0 else 0

        # Calculate holding period
        holding_period = (exit_time - position.entry_time).days

        # Create trade record
        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            direction=position.direction,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=exit_time,
            gross_pnl=gross_pnl,
            commission=commission,
            slippage=slippage,
            net_pnl=net_pnl,
            net_pnl_pct=net_pnl_pct,
            holding_period_bars=holding_period,
            exit_reason=exit_reason,
            signal_at_entry=position.signal_strength,
            signal_at_exit=signal_at_exit,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            atr_at_entry=position.atr_at_entry,
            dna_id=position.dna_id,
            generation=position.generation,
            run_id=run_id
        )

        self.completed_trades.append(trade)
        del self.positions[symbol]

        return trade

    def update_prices(self, prices: Dict[str, float], timestamp: datetime = None):
        """
        Update all position prices and record equity.

        Args:
            prices: Dict of symbol -> current price
            timestamp: Current timestamp
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])

        # Record equity point
        equity = self.get_equity()
        self.equity_curve.append({
            'timestamp': timestamp or datetime.now(),
            'equity': equity,
            'cash': self.cash,
            'positions_value': equity - self.cash,
            'num_positions': len(self.positions),
            'unrealized_pnl': self.get_unrealized_pnl()
        })

        # Track peak for drawdown
        self._peak_equity = max(self._peak_equity, equity)

    def get_drawdown(self) -> float:
        """Get current drawdown from peak."""
        equity = self.get_equity()
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - equity) / self._peak_equity

    def get_max_drawdown(self) -> float:
        """Get maximum drawdown from equity curve."""
        if not self.equity_curve:
            return 0.0

        equities = [e['equity'] for e in self.equity_curve]
        peak = equities[0]
        max_dd = 0.0

        for equity in equities:
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.completed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'net_profit': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }

        # Basic counts
        total = len(self.completed_trades)
        winners = [t for t in self.completed_trades if t.net_pnl > 0]
        losers = [t for t in self.completed_trades if t.net_pnl <= 0]
        longs = [t for t in self.completed_trades if t.direction == 'LONG']
        shorts = [t for t in self.completed_trades if t.direction == 'SHORT']

        # Win rates
        win_rate = len(winners) / total if total > 0 else 0
        long_win_rate = len([t for t in longs if t.net_pnl > 0]) / len(longs) if longs else 0
        short_win_rate = len([t for t in shorts if t.net_pnl > 0]) / len(shorts) if shorts else 0

        # P&L
        gross_profit = sum(t.net_pnl for t in winners)
        gross_loss = abs(sum(t.net_pnl for t in losers))
        net_profit = gross_profit - gross_loss

        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio (annualized)
        returns = [t.net_pnl_pct for t in self.completed_trades]
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe = 0

        return {
            'total_trades': total,
            'long_trades': len(longs),
            'short_trades': len(shorts),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': win_rate,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': self.get_max_drawdown(),
            'avg_winner': np.mean([t.net_pnl for t in winners]) if winners else 0,
            'avg_loser': np.mean([t.net_pnl for t in losers]) if losers else 0,
            'largest_winner': max([t.net_pnl for t in winners]) if winners else 0,
            'largest_loser': min([t.net_pnl for t in losers]) if losers else 0,
            'avg_holding_period': np.mean([t.holding_period_bars for t in self.completed_trades]),
            'return_pct': (self.get_equity() - self.initial_capital) / self.initial_capital
        }

    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.completed_trades.clear()
        self.equity_curve.clear()
        self._peak_equity = self.initial_capital

    def get_equity_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_curve).set_index('timestamp')

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in symbol."""
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)

    @property
    def num_positions(self) -> int:
        """Number of open positions."""
        return len(self.positions)

    @property
    def long_positions(self) -> List[Position]:
        """List of long positions."""
        return [p for p in self.positions.values() if p.direction == 'LONG']

    @property
    def short_positions(self) -> List[Position]:
        """List of short positions."""
        return [p for p in self.positions.values() if p.direction == 'SHORT']
