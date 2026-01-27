"""
Trade Logger module.

Logs all trades with full context to the database.
"""

from typing import Dict, List, Optional
from datetime import datetime
import json
import logging

from .database import Database

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Logs trades to the database with full context.

    Captures:
    - Entry/exit details
    - P&L calculations
    - Indicator snapshots at entry
    - Market regime at entry
    - DNA configuration used
    """

    def __init__(self, database: Database):
        """
        Initialize trade logger.

        Args:
            database: Database instance for persistence
        """
        self.db = database
        self._pending_trades: List[Dict] = []

    def log_trade(self,
                  run_id: int,
                  generation: int,
                  dna_id: str,
                  symbol: str,
                  direction: str,
                  entry_date: datetime,
                  entry_price: float,
                  exit_date: datetime,
                  exit_price: float,
                  shares: int,
                  gross_pnl: float,
                  commission: float,
                  slippage: float,
                  net_pnl: float,
                  net_pnl_pct: float,
                  stop_price: float,
                  target_price: float,
                  exit_reason: str,
                  signal_strength: float,
                  indicator_snapshot: Dict[str, float] = None,
                  market_regime: str = None,
                  notes: str = None):
        """
        Log a completed trade.

        Args:
            run_id: Evolution run ID
            generation: Generation number
            dna_id: DNA configuration ID
            symbol: Trading symbol
            direction: LONG or SHORT
            entry_date: Entry timestamp
            entry_price: Entry price
            exit_date: Exit timestamp
            exit_price: Exit price
            shares: Number of shares
            gross_pnl: Gross P&L before costs
            commission: Commission paid
            slippage: Slippage cost
            net_pnl: Net P&L after costs
            net_pnl_pct: Net P&L as percentage
            stop_price: Stop loss price
            target_price: Target price
            exit_reason: Why trade was closed
            signal_strength: Super Indicator signal at entry
            indicator_snapshot: Individual indicator values at entry
            market_regime: Market regime at entry
            notes: Additional notes
        """
        trade_data = {
            'run_id': run_id,
            'generation': generation,
            'dna_id': dna_id,
            'symbol': symbol,
            'direction': direction,
            'entry_date': entry_date.isoformat() if isinstance(entry_date, datetime) else str(entry_date),
            'entry_price': entry_price,
            'exit_date': exit_date.isoformat() if isinstance(exit_date, datetime) else str(exit_date),
            'exit_price': exit_price,
            'shares': shares,
            'gross_pnl': gross_pnl,
            'commission': commission,
            'slippage': slippage,
            'net_pnl': net_pnl,
            'net_pnl_pct': net_pnl_pct,
            'stop_price': stop_price,
            'target_price': target_price,
            'exit_reason': exit_reason,
            'signal_strength': signal_strength,
            'indicator_snapshot': json.dumps(indicator_snapshot) if indicator_snapshot else None,
            'market_regime': market_regime,
            'notes': notes
        }

        # Save to database
        self.db.log_trade(trade_data)

        logger.debug(
            f"Logged trade: {symbol} {direction} | "
            f"P&L: ${net_pnl:.2f} ({net_pnl_pct:.2%}) | "
            f"Exit: {exit_reason}"
        )

    def log_trade_from_dict(self, trade: Dict, run_id: int, generation: int, dna_id: str):
        """
        Log a trade from a dictionary (convenience method).

        Args:
            trade: Trade dictionary with all fields
            run_id: Evolution run ID
            generation: Generation number
            dna_id: DNA configuration ID
        """
        self.log_trade(
            run_id=run_id,
            generation=generation,
            dna_id=dna_id,
            symbol=trade.get('symbol'),
            direction=trade.get('direction'),
            entry_date=trade.get('entry_date'),
            entry_price=trade.get('entry_price'),
            exit_date=trade.get('exit_date'),
            exit_price=trade.get('exit_price'),
            shares=trade.get('shares', 0),
            gross_pnl=trade.get('gross_pnl', 0),
            commission=trade.get('commission', 0),
            slippage=trade.get('slippage', 0),
            net_pnl=trade.get('net_pnl', 0),
            net_pnl_pct=trade.get('net_pnl_pct', 0),
            stop_price=trade.get('stop_price', 0),
            target_price=trade.get('target_price', 0),
            exit_reason=trade.get('exit_reason', ''),
            signal_strength=trade.get('signal_strength', 0),
            indicator_snapshot=trade.get('indicator_snapshot'),
            market_regime=trade.get('market_regime'),
            notes=trade.get('notes')
        )

    def log_batch(self, trades: List[Dict], run_id: int, generation: int, dna_id: str):
        """
        Log multiple trades at once.

        Args:
            trades: List of trade dictionaries
            run_id: Evolution run ID
            generation: Generation number
            dna_id: DNA configuration ID
        """
        for trade in trades:
            self.log_trade_from_dict(trade, run_id, generation, dna_id)

        logger.info(f"Logged batch of {len(trades)} trades for generation {generation}")

    def get_trades(self,
                   run_id: int = None,
                   generation: int = None,
                   dna_id: str = None,
                   symbol: str = None,
                   direction: str = None,
                   profitable_only: bool = False) -> List[Dict]:
        """
        Retrieve trades with optional filters.

        Args:
            run_id: Filter by run ID
            generation: Filter by generation
            dna_id: Filter by DNA ID
            symbol: Filter by symbol
            direction: Filter by direction
            profitable_only: Only return profitable trades

        Returns:
            List of trade dictionaries
        """
        trades = self.db.get_trades(
            run_id=run_id,
            generation=generation,
            symbol=symbol
        )

        # Apply additional filters
        if dna_id:
            trades = [t for t in trades if t.get('dna_id') == dna_id]

        if direction:
            trades = [t for t in trades if t.get('direction') == direction]

        if profitable_only:
            trades = [t for t in trades if t.get('net_pnl', 0) > 0]

        return trades

    def get_trade_summary(self, run_id: int, generation: int = None) -> Dict:
        """
        Get summary statistics for trades.

        Args:
            run_id: Evolution run ID
            generation: Optional generation filter

        Returns:
            Dictionary with summary statistics
        """
        trades = self.get_trades(run_id=run_id, generation=generation)

        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'avg_winner': 0,
                'avg_loser': 0,
                'largest_winner': 0,
                'largest_loser': 0,
                'long_trades': 0,
                'short_trades': 0,
                'long_win_rate': 0,
                'short_win_rate': 0
            }

        pnls = [t.get('net_pnl', 0) for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        longs = [t for t in trades if t.get('direction') == 'LONG']
        shorts = [t for t in trades if t.get('direction') == 'SHORT']
        long_winners = [t for t in longs if t.get('net_pnl', 0) > 0]
        short_winners = [t for t in shorts if t.get('net_pnl', 0) > 0]

        return {
            'total_trades': len(trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(trades) if trades else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': sum(pnls) / len(pnls) if pnls else 0,
            'avg_winner': sum(winners) / len(winners) if winners else 0,
            'avg_loser': sum(losers) / len(losers) if losers else 0,
            'largest_winner': max(winners) if winners else 0,
            'largest_loser': min(losers) if losers else 0,
            'long_trades': len(longs),
            'short_trades': len(shorts),
            'long_win_rate': len(long_winners) / len(longs) if longs else 0,
            'short_win_rate': len(short_winners) / len(shorts) if shorts else 0
        }

    def get_trades_by_regime(self, run_id: int, generation: int = None) -> Dict[str, List[Dict]]:
        """
        Group trades by market regime.

        Args:
            run_id: Evolution run ID
            generation: Optional generation filter

        Returns:
            Dictionary mapping regime to list of trades
        """
        trades = self.get_trades(run_id=run_id, generation=generation)

        by_regime = {}
        for trade in trades:
            regime = trade.get('market_regime', 'unknown')
            if regime not in by_regime:
                by_regime[regime] = []
            by_regime[regime].append(trade)

        return by_regime

    def get_indicator_contribution(self,
                                   run_id: int,
                                   indicator_name: str,
                                   generation: int = None) -> Dict:
        """
        Analyze an indicator's contribution to trades.

        Args:
            run_id: Evolution run ID
            indicator_name: Name of indicator
            generation: Optional generation filter

        Returns:
            Dictionary with indicator contribution metrics
        """
        trades = self.get_trades(run_id=run_id, generation=generation)

        values_winners = []
        values_losers = []

        for trade in trades:
            snapshot = trade.get('indicator_snapshot')
            if snapshot:
                if isinstance(snapshot, str):
                    snapshot = json.loads(snapshot)

                if indicator_name in snapshot:
                    value = snapshot[indicator_name]
                    if trade.get('net_pnl', 0) > 0:
                        values_winners.append(value)
                    else:
                        values_losers.append(value)

        if not values_winners and not values_losers:
            return {'indicator': indicator_name, 'data_points': 0}

        import numpy as np

        return {
            'indicator': indicator_name,
            'data_points': len(values_winners) + len(values_losers),
            'avg_value_winners': np.mean(values_winners) if values_winners else 0,
            'avg_value_losers': np.mean(values_losers) if values_losers else 0,
            'std_value_winners': np.std(values_winners) if values_winners else 0,
            'std_value_losers': np.std(values_losers) if values_losers else 0,
            'value_difference': (
                np.mean(values_winners) - np.mean(values_losers)
                if values_winners and values_losers else 0
            )
        }
