"""
Trade Ledger with Strategy Versioning and Reason Codes.

Every trade includes:
- strategy_version: Which DNA/weights were used
- reason_code: Why the trade was taken
- Full audit trail for analysis
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import uuid
from pathlib import Path


class TradeReason(Enum):
    """Reason codes for trade entries/exits."""
    # Entry reasons
    SI_LONG_CROSS = "SI crossed above entry threshold"
    SI_SHORT_CROSS = "SI crossed below entry threshold"
    
    # Exit reasons
    SI_EXIT_SIGNAL = "SI dropped below exit threshold"
    STOP_LOSS_HIT = "Stop loss triggered"
    TAKE_PROFIT_HIT = "Take profit triggered"
    EOD_FLATTEN = "End of day flatten"
    RISK_LIMIT = "Risk limit reached"
    MANUAL_CLOSE = "Manual close"
    
    # Other
    UNKNOWN = "Unknown"


@dataclass
class LedgerEntry:
    """A single entry in the trade ledger."""
    
    # Identifiers
    entry_id: str
    trade_id: str
    symbol: str
    
    # Strategy info
    strategy_version: str
    dna_id: str
    
    # Trade details
    side: str  # "BUY" or "SELL"
    quantity: int
    price: float
    
    # Signal info
    si_value: float
    entry_reason: str
    
    # Timestamps
    timestamp: datetime
    
    # Costs
    slippage: float = 0.0
    commission: float = 0.0
    total_cost: float = 0.0

    # Indicator snapshot at entry: {indicator_name: raw_value}
    # Enables per-indicator analysis by the Coach
    indicator_snapshot: Dict[str, float] = field(default_factory=dict)

    # Optional exit info (filled when trade closes)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        if self.exit_time:
            d['exit_time'] = self.exit_time.isoformat()
        return d


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    trades: int = 0
    winners: int = 0
    losers: int = 0
    gross_pnl: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'date': self.date.isoformat(),
            'trades': self.trades,
            'winners': self.winners,
            'losers': self.losers,
            'gross_pnl': self.gross_pnl,
            'total_costs': self.total_costs,
            'net_pnl': self.net_pnl,
            'win_rate': self.winners / self.trades * 100 if self.trades > 0 else 0,
        }


class TradeLedger:
    """
    Trade ledger with full audit trail.
    
    Features:
    - Records every trade with strategy version
    - Tracks reason codes for entries/exits
    - Calculates daily P&L
    - Persists to JSON files
    """
    
    def __init__(
        self,
        strategy_version: str,
        dna_id: str,
        save_dir: str = "./paper_trades",
    ):
        """
        Initialize trade ledger.
        
        Args:
            strategy_version: Version identifier for strategy
            dna_id: DNA ID being used
            save_dir: Directory to save trade logs
        """
        self.strategy_version = strategy_version
        self.dna_id = dna_id
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # All entries
        self._entries: List[LedgerEntry] = []
        
        # Open positions (symbol -> entry)
        self._open_positions: Dict[str, LedgerEntry] = {}
        
        # Daily stats
        self._daily_stats: Dict[date, DailyStats] = {}
        
        # Counters
        self._trade_count = 0
    
    def record_entry(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        si_value: float,
        reason: TradeReason,
        timestamp: datetime = None,
        slippage: float = 0.0,
        commission: float = 0.0,
        indicator_snapshot: Dict[str, float] = None,
    ) -> LedgerEntry:
        """
        Record a trade entry.

        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            quantity: Number of shares
            price: Entry price
            si_value: Super Indicator value at entry
            reason: Reason for entry
            timestamp: Entry time
            slippage: Slippage cost
            commission: Commission cost
            indicator_snapshot: Individual indicator values at entry

        Returns:
            LedgerEntry object
        """
        self._trade_count += 1

        entry = LedgerEntry(
            entry_id=str(uuid.uuid4())[:8],
            trade_id=f"T{self._trade_count:04d}",
            symbol=symbol,
            strategy_version=self.strategy_version,
            dna_id=self.dna_id,
            side=side,
            quantity=quantity,
            price=price,
            si_value=si_value,
            entry_reason=reason.value,
            timestamp=timestamp or datetime.now(),
            slippage=slippage,
            commission=commission,
            total_cost=slippage + commission,
            indicator_snapshot=indicator_snapshot or {},
        )
        
        self._entries.append(entry)
        self._open_positions[symbol] = entry
        
        return entry
    
    def record_exit(
        self,
        symbol: str,
        exit_price: float,
        reason: TradeReason,
        exit_time: datetime = None,
        exit_slippage: float = 0.0,
        exit_commission: float = 0.0,
    ) -> Optional[LedgerEntry]:
        """
        Record a trade exit.
        
        Args:
            symbol: Stock symbol
            exit_price: Exit price
            reason: Reason for exit
            exit_time: Exit time
            exit_slippage: Exit slippage
            exit_commission: Exit commission
            
        Returns:
            Updated LedgerEntry or None if no position
        """
        if symbol not in self._open_positions:
            return None
        
        entry = self._open_positions.pop(symbol)
        
        # Calculate P&L
        if entry.side == "BUY":
            gross_pnl = (exit_price - entry.price) * entry.quantity
        else:
            gross_pnl = (entry.price - exit_price) * entry.quantity
        
        total_costs = entry.total_cost + exit_slippage + exit_commission
        net_pnl = gross_pnl - total_costs
        
        # Update entry
        entry.exit_price = exit_price
        entry.exit_time = exit_time or datetime.now()
        entry.exit_reason = reason.value
        entry.pnl = net_pnl
        entry.total_cost = total_costs
        
        # Update daily stats
        trade_date = entry.exit_time.date()
        if trade_date not in self._daily_stats:
            self._daily_stats[trade_date] = DailyStats(date=trade_date)
        
        stats = self._daily_stats[trade_date]
        stats.trades += 1
        stats.gross_pnl += gross_pnl
        stats.total_costs += total_costs
        stats.net_pnl += net_pnl
        
        if net_pnl > 0:
            stats.winners += 1
        else:
            stats.losers += 1
        
        return entry
    
    def get_open_positions(self) -> Dict[str, LedgerEntry]:
        """Get all open positions."""
        return self._open_positions.copy()
    
    def get_open_trades(self) -> Dict[str, LedgerEntry]:
        """Alias for get_open_positions."""
        return self.get_open_positions()
    
    def get_position(self, symbol: str) -> Optional[LedgerEntry]:
        """Get position for a symbol."""
        return self._open_positions.get(symbol)
    
    def get_closed_trades(self) -> List[LedgerEntry]:
        """Get all closed trades."""
        return [e for e in self._entries if e.exit_price is not None]
    
    def get_daily_stats(self, for_date: date = None) -> Optional[DailyStats]:
        """Get stats for a specific date."""
        target = for_date or date.today()
        return self._daily_stats.get(target)
    
    def get_all_daily_stats(self) -> List[DailyStats]:
        """Get all daily stats sorted by date."""
        return sorted(self._daily_stats.values(), key=lambda x: x.date)
    
    def get_total_pnl(self) -> float:
        """Get total P&L across all closed trades."""
        return sum(e.pnl for e in self._entries if e.pnl is not None)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        closed = self.get_closed_trades()
        if not closed:
            return {'total_trades': 0}
        
        pnls = [e.pnl for e in closed]
        winners = [e for e in closed if e.pnl > 0]
        
        return {
            'strategy_version': self.strategy_version,
            'dna_id': self.dna_id,
            'total_trades': len(closed),
            'open_positions': len(self._open_positions),
            'winners': len(winners),
            'losers': len(closed) - len(winners),
            'win_rate': len(winners) / len(closed) * 100,
            'total_pnl': sum(pnls),
            'avg_pnl': sum(pnls) / len(pnls),
            'best_trade': max(pnls),
            'worst_trade': min(pnls),
        }
    
    def save(self, filename: str = None):
        """Save ledger to JSON file."""
        fname = filename or f"ledger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.save_dir / fname
        
        data = {
            'strategy_version': self.strategy_version,
            'dna_id': self.dna_id,
            'saved_at': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'entries': [e.to_dict() for e in self._entries],
            'daily_stats': [s.to_dict() for s in self.get_all_daily_stats()],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def generate_daily_report(self, for_date: date = None) -> str:
        """Generate a daily P&L report."""
        stats = self.get_daily_stats(for_date)
        target = for_date or date.today()
        
        if not stats:
            return f"No trades recorded for {target}"
        
        report = f"""
================================================================================
DAILY P&L REPORT: {target}
================================================================================
Strategy: {self.strategy_version} (DNA: {self.dna_id})

TRADES:
  Total: {stats.trades}
  Winners: {stats.winners} ({stats.winners/stats.trades*100:.1f}%)
  Losers: {stats.losers} ({stats.losers/stats.trades*100:.1f}%)

P&L:
  Gross P&L: ₹{stats.gross_pnl:,.2f}
  Costs: ₹{stats.total_costs:,.2f}
  Net P&L: ₹{stats.net_pnl:,.2f}

TRADES TODAY:
"""
        # Add individual trades
        day_trades = [e for e in self._entries 
                      if e.exit_time and e.exit_time.date() == target]
        
        for t in day_trades:
            result = "WIN" if t.pnl > 0 else "LOSS"
            report += f"  {t.trade_id} | {t.symbol} | {t.side} | "
            report += f"₹{t.price:.2f} -> ₹{t.exit_price:.2f} | "
            report += f"₹{t.pnl:,.2f} ({result})\n"
            report += f"    Entry: {t.entry_reason}\n"
            report += f"    Exit: {t.exit_reason}\n"
        
        return report
