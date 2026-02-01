"""
Backtest Result and Trade dataclasses.

Provides structured output from backtest runs with full reproducibility info.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any
import json
import hashlib
import pandas as pd
from pathlib import Path


@dataclass
class Trade:
    """
    Complete trade record with all details for analysis.
    """
    
    # Identification
    trade_id: str
    symbol: str
    
    # Direction
    direction: str  # 'LONG' or 'SHORT'
    
    # Timing
    entry_time: datetime
    exit_time: datetime
    
    # Prices (before costs)
    entry_price: float
    exit_price: float
    
    # Prices (after slippage)
    fill_entry_price: float
    fill_exit_price: float
    
    # Size
    quantity: int
    position_value: float  # entry_price * quantity
    
    # P&L
    gross_pnl: float
    total_costs: float
    net_pnl: float
    pnl_pct: float  # Net P&L as % of position value
    
    # Exit info
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit', 'eod_flatten', 'daily_limit'
    
    # Signal info
    entry_signal_value: float = 0.0
    exit_signal_value: float = 0.0
    
    # Risk info
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    risk_amount: float = 0.0
    
    # Market context
    market_regime: str = ""
    atr_at_entry: float = 0.0
    volatility_percentile: float = 0.0
    
    # Indicator snapshot at entry
    indicator_snapshot: Dict[str, float] = field(default_factory=dict)
    
    @property
    def holding_duration(self) -> timedelta:
        """Get trade holding duration."""
        return self.exit_time - self.entry_time
    
    @property
    def holding_bars(self) -> int:
        """Get approximate number of bars held (assuming 5min bars)."""
        return int(self.holding_duration.total_seconds() / 300)
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.net_pnl > 0
    
    @property
    def r_multiple(self) -> float:
        """Get R-multiple (net P&L / risk amount)."""
        if self.risk_amount > 0:
            return self.net_pnl / self.risk_amount
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'fill_entry_price': self.fill_entry_price,
            'fill_exit_price': self.fill_exit_price,
            'quantity': self.quantity,
            'position_value': self.position_value,
            'gross_pnl': self.gross_pnl,
            'total_costs': self.total_costs,
            'net_pnl': self.net_pnl,
            'pnl_pct': self.pnl_pct,
            'exit_reason': self.exit_reason,
            'entry_signal_value': self.entry_signal_value,
            'exit_signal_value': self.exit_signal_value,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'risk_amount': self.risk_amount,
            'market_regime': self.market_regime,
            'atr_at_entry': self.atr_at_entry,
            'volatility_percentile': self.volatility_percentile,
            'holding_duration_seconds': self.holding_duration.total_seconds(),
            'is_winner': self.is_winner,
            'r_multiple': self.r_multiple,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create from dictionary."""
        data = data.copy()
        # Convert time strings back to datetime
        if isinstance(data.get('entry_time'), str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        if isinstance(data.get('exit_time'), str):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        # Remove computed fields
        data.pop('holding_duration_seconds', None)
        data.pop('is_winner', None)
        data.pop('r_multiple', None)
        return cls(**data)


@dataclass
class BacktestResult:
    """
    Complete backtest result with full audit trail.
    """
    
    # Identification
    run_id: str
    run_timestamp: datetime
    
    # Configuration
    config_hash: str
    strategy_name: str
    strategy_version: str
    
    # Date range
    start_date: date
    end_date: date
    
    # Data info
    symbols: List[str]
    data_hash: str
    bar_count: int
    
    # Results
    trades: List[Trade]
    
    # Summary metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    gross_pnl: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Equity curve
    equity_curve: Optional[pd.DataFrame] = None
    daily_returns: Optional[pd.DataFrame] = None
    
    # Reproducibility
    random_seed: int = 42
    
    def __post_init__(self):
        """Calculate summary metrics from trades."""
        if self.trades and self.total_trades == 0:
            self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate all metrics from trade list."""
        if not self.trades:
            return
        
        self.total_trades = len(self.trades)
        winners = [t for t in self.trades if t.is_winner]
        losers = [t for t in self.trades if not t.is_winner]
        
        self.winning_trades = len(winners)
        self.losing_trades = len(losers)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        self.gross_pnl = sum(t.gross_pnl for t in self.trades)
        self.total_costs = sum(t.total_costs for t in self.trades)
        self.net_pnl = sum(t.net_pnl for t in self.trades)
        
        if winners:
            self.avg_win = sum(t.net_pnl for t in winners) / len(winners)
        if losers:
            self.avg_loss = abs(sum(t.net_pnl for t in losers) / len(losers))
        
        total_wins = sum(t.net_pnl for t in winners) if winners else 0
        total_losses = abs(sum(t.net_pnl for t in losers)) if losers else 0
        
        if total_losses > 0:
            self.profit_factor = total_wins / total_losses
        elif total_wins > 0:
            self.profit_factor = float('inf')
        
        self.expectancy = self.net_pnl / self.total_trades if self.total_trades > 0 else 0
    
    def verify_reproducibility(self, other: 'BacktestResult') -> bool:
        """Check if another result matches this one."""
        return (
            self.config_hash == other.config_hash and
            self.data_hash == other.data_hash and
            self.total_trades == other.total_trades and
            abs(self.net_pnl - other.net_pnl) < 0.01
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'run_timestamp': self.run_timestamp.isoformat(),
            'config_hash': self.config_hash,
            'strategy_name': self.strategy_name,
            'strategy_version': self.strategy_version,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'symbols': self.symbols,
            'data_hash': self.data_hash,
            'bar_count': self.bar_count,
            'random_seed': self.random_seed,
            'trades': [t.to_dict() for t in self.trades],
            'metrics': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.win_rate,
                'gross_pnl': self.gross_pnl,
                'total_costs': self.total_costs,
                'net_pnl': self.net_pnl,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'profit_factor': self.profit_factor,
                'expectancy': self.expectancy,
                'max_drawdown': self.max_drawdown,
                'max_drawdown_pct': self.max_drawdown_pct,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
            },
        }
    
    def save(self, path: str):
        """Save result to JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BacktestResult':
        """Load result from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert trades
        trades = [Trade.from_dict(t) for t in data.get('trades', [])]
        
        # Parse dates
        metrics = data.get('metrics', {})
        
        return cls(
            run_id=data['run_id'],
            run_timestamp=datetime.fromisoformat(data['run_timestamp']),
            config_hash=data['config_hash'],
            strategy_name=data['strategy_name'],
            strategy_version=data['strategy_version'],
            start_date=date.fromisoformat(data['start_date']),
            end_date=date.fromisoformat(data['end_date']),
            symbols=data['symbols'],
            data_hash=data['data_hash'],
            bar_count=data['bar_count'],
            random_seed=data.get('random_seed', 42),
            trades=trades,
            total_trades=metrics.get('total_trades', 0),
            winning_trades=metrics.get('winning_trades', 0),
            losing_trades=metrics.get('losing_trades', 0),
            win_rate=metrics.get('win_rate', 0),
            gross_pnl=metrics.get('gross_pnl', 0),
            total_costs=metrics.get('total_costs', 0),
            net_pnl=metrics.get('net_pnl', 0),
            avg_win=metrics.get('avg_win', 0),
            avg_loss=metrics.get('avg_loss', 0),
            profit_factor=metrics.get('profit_factor', 0),
            expectancy=metrics.get('expectancy', 0),
            max_drawdown=metrics.get('max_drawdown', 0),
            max_drawdown_pct=metrics.get('max_drawdown_pct', 0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0),
            sortino_ratio=metrics.get('sortino_ratio', 0),
            calmar_ratio=metrics.get('calmar_ratio', 0),
        )
    
    def summary(self) -> str:
        """Get human-readable summary."""
        return f"""
Backtest Result: {self.strategy_name} v{self.strategy_version}
{'='*50}
Run ID: {self.run_id}
Config Hash: {self.config_hash}
Period: {self.start_date} to {self.end_date}
Symbols: {', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}

Performance:
  Total Trades: {self.total_trades}
  Win Rate: {self.win_rate:.1%}
  Net P&L: ₹{self.net_pnl:,.2f}
  Profit Factor: {self.profit_factor:.2f}
  Expectancy: ₹{self.expectancy:,.2f}
  
Risk:
  Max Drawdown: {self.max_drawdown_pct:.1%}
  Sharpe Ratio: {self.sharpe_ratio:.2f}
  Sortino Ratio: {self.sortino_ratio:.2f}

Costs:
  Total Costs: ₹{self.total_costs:,.2f}
  Cost Impact: {(self.total_costs / self.gross_pnl * 100) if self.gross_pnl else 0:.1f}% of gross
"""
