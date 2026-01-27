"""
Pattern Detector module.

Finds patterns in winning and losing trades to improve strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradePattern:
    """Detected pattern in trades."""
    name: str
    description: str
    win_rate: float
    avg_profit: float
    trade_count: int
    confidence: float  # Statistical confidence
    indicators_involved: List[str] = field(default_factory=list)
    recommendation: str = ""


class PatternDetector:
    """
    Detects patterns in winning and losing trades.

    Patterns analyzed:
    - Indicator combinations that precede winners
    - Common characteristics of losing trades
    - Time-of-week patterns
    - Holding period patterns
    - Entry signal strength patterns
    """

    def __init__(self, min_pattern_count: int = 10):
        """
        Initialize pattern detector.

        Args:
            min_pattern_count: Minimum occurrences for valid pattern
        """
        self.min_count = min_pattern_count

    def find_patterns(self,
                      trades: List[Dict],
                      indicator_snapshots: Dict[str, Dict]) -> List[TradePattern]:
        """
        Find patterns in trade data.

        Args:
            trades: List of completed trades
            indicator_snapshots: Indicator values at trade entries

        Returns:
            List of detected patterns
        """
        if len(trades) < self.min_count:
            return []

        patterns = []

        # 1. Entry signal strength patterns
        patterns.extend(self._analyze_signal_strength(trades))

        # 2. Indicator combination patterns
        patterns.extend(self._analyze_indicator_combinations(trades, indicator_snapshots))

        # 3. Holding period patterns
        patterns.extend(self._analyze_holding_periods(trades))

        # 4. Direction patterns (long vs short)
        patterns.extend(self._analyze_direction(trades))

        # 5. Exit reason patterns
        patterns.extend(self._analyze_exit_reasons(trades))

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        return patterns

    def _analyze_signal_strength(self, trades: List[Dict]) -> List[TradePattern]:
        """Analyze patterns in entry signal strength."""
        patterns = []

        # Group by signal strength buckets
        buckets = {
            'very_strong': (0.85, 1.0),
            'strong': (0.7, 0.85),
            'moderate': (0.5, 0.7),
            'weak': (0.3, 0.5)
        }

        for bucket_name, (low, high) in buckets.items():
            bucket_trades = [
                t for t in trades
                if low <= abs(t.get('super_indicator_entry', 0)) < high
            ]

            if len(bucket_trades) >= self.min_count:
                winners = [t for t in bucket_trades if t.get('net_pnl', 0) > 0]
                win_rate = len(winners) / len(bucket_trades)
                avg_profit = np.mean([t.get('net_pnl', 0) for t in bucket_trades])

                # Compare to overall
                all_winners = [t for t in trades if t.get('net_pnl', 0) > 0]
                overall_win_rate = len(all_winners) / len(trades) if trades else 0

                # Pattern is significant if different from overall
                diff = abs(win_rate - overall_win_rate)
                confidence = min(1.0, diff * 5)  # Higher diff = higher confidence

                if diff > 0.05:  # At least 5% difference
                    direction = "better" if win_rate > overall_win_rate else "worse"
                    patterns.append(TradePattern(
                        name=f"signal_strength_{bucket_name}",
                        description=f"{bucket_name.replace('_', ' ').title()} signals "
                                    f"({low:.0%}-{high:.0%}) perform {direction}",
                        win_rate=win_rate,
                        avg_profit=avg_profit,
                        trade_count=len(bucket_trades),
                        confidence=confidence,
                        recommendation=f"{'Favor' if direction == 'better' else 'Avoid'} "
                                       f"{bucket_name} signal strength entries"
                    ))

        return patterns

    def _analyze_indicator_combinations(self,
                                        trades: List[Dict],
                                        snapshots: Dict) -> List[TradePattern]:
        """Find indicator combinations that predict winners."""
        patterns = []

        if not snapshots:
            return patterns

        # Get indicator names from first snapshot
        sample_snapshot = next(iter(snapshots.values()), {})
        indicator_names = list(sample_snapshot.keys())

        if not indicator_names:
            return patterns

        # Find indicators that are often positive/negative together in winners
        winners = [t for t in trades if t.get('net_pnl', 0) > 0]
        losers = [t for t in trades if t.get('net_pnl', 0) <= 0]

        # Count indicator agreement in winners vs losers
        for ind1_idx, ind1 in enumerate(indicator_names[:20]):  # Limit for performance
            for ind2 in indicator_names[ind1_idx + 1:30]:
                winner_agreement = 0
                winner_total = 0
                loser_agreement = 0
                loser_total = 0

                for t in winners:
                    tid = t.get('trade_id')
                    if tid in snapshots:
                        v1 = snapshots[tid].get(ind1, 0)
                        v2 = snapshots[tid].get(ind2, 0)
                        if v1 != 0 and v2 != 0:
                            winner_total += 1
                            if (v1 > 0) == (v2 > 0):  # Same sign
                                winner_agreement += 1

                for t in losers:
                    tid = t.get('trade_id')
                    if tid in snapshots:
                        v1 = snapshots[tid].get(ind1, 0)
                        v2 = snapshots[tid].get(ind2, 0)
                        if v1 != 0 and v2 != 0:
                            loser_total += 1
                            if (v1 > 0) == (v2 > 0):
                                loser_agreement += 1

                # Check if agreement rate differs between winners and losers
                if winner_total >= 10 and loser_total >= 10:
                    winner_rate = winner_agreement / winner_total
                    loser_rate = loser_agreement / loser_total

                    diff = winner_rate - loser_rate
                    if abs(diff) > 0.15:  # Significant difference
                        direction = "aligned" if diff > 0 else "divergent"
                        patterns.append(TradePattern(
                            name=f"combo_{ind1}_{ind2}",
                            description=f"{ind1} and {ind2} being {direction} "
                                        f"predicts {'winners' if diff > 0 else 'losers'}",
                            win_rate=winner_rate if diff > 0 else 1 - loser_rate,
                            avg_profit=0,  # Would need more calculation
                            trade_count=winner_total + loser_total,
                            confidence=min(1.0, abs(diff) * 3),
                            indicators_involved=[ind1, ind2],
                            recommendation=f"{'Prefer' if diff > 0 else 'Avoid'} trades when "
                                           f"{ind1} and {ind2} are {direction}"
                        ))

        return patterns[:10]  # Return top 10

    def _analyze_holding_periods(self, trades: List[Dict]) -> List[TradePattern]:
        """Analyze patterns in holding periods."""
        patterns = []

        # Group by holding period buckets
        buckets = {
            'very_short': (0, 2),
            'short': (2, 5),
            'medium': (5, 10),
            'long': (10, 30),
            'very_long': (30, float('inf'))
        }

        for bucket_name, (low, high) in buckets.items():
            bucket_trades = [
                t for t in trades
                if low <= t.get('holding_period_bars', 0) < high
            ]

            if len(bucket_trades) >= self.min_count:
                winners = [t for t in bucket_trades if t.get('net_pnl', 0) > 0]
                win_rate = len(winners) / len(bucket_trades)
                avg_profit = np.mean([t.get('net_pnl', 0) for t in bucket_trades])

                patterns.append(TradePattern(
                    name=f"holding_{bucket_name}",
                    description=f"{bucket_name.replace('_', ' ').title()} holding period "
                                f"({low}-{high if high != float('inf') else '+'} days)",
                    win_rate=win_rate,
                    avg_profit=avg_profit,
                    trade_count=len(bucket_trades),
                    confidence=min(1.0, len(bucket_trades) / 50),
                    recommendation=f"{'Optimal' if win_rate > 0.55 else 'Suboptimal'} "
                                   f"holding period"
                ))

        return patterns

    def _analyze_direction(self, trades: List[Dict]) -> List[TradePattern]:
        """Analyze long vs short performance."""
        patterns = []

        long_trades = [t for t in trades if t.get('direction') == 'LONG']
        short_trades = [t for t in trades if t.get('direction') == 'SHORT']

        if len(long_trades) >= self.min_count:
            long_winners = [t for t in long_trades if t.get('net_pnl', 0) > 0]
            long_win_rate = len(long_winners) / len(long_trades)
            long_avg = np.mean([t.get('net_pnl', 0) for t in long_trades])

            patterns.append(TradePattern(
                name="direction_long",
                description=f"Long trades performance",
                win_rate=long_win_rate,
                avg_profit=long_avg,
                trade_count=len(long_trades),
                confidence=min(1.0, len(long_trades) / 50),
                recommendation="Long trades are "
                               f"{'profitable' if long_avg > 0 else 'unprofitable'}"
            ))

        if len(short_trades) >= self.min_count:
            short_winners = [t for t in short_trades if t.get('net_pnl', 0) > 0]
            short_win_rate = len(short_winners) / len(short_trades)
            short_avg = np.mean([t.get('net_pnl', 0) for t in short_trades])

            patterns.append(TradePattern(
                name="direction_short",
                description=f"Short trades performance",
                win_rate=short_win_rate,
                avg_profit=short_avg,
                trade_count=len(short_trades),
                confidence=min(1.0, len(short_trades) / 50),
                recommendation="Short trades are "
                               f"{'profitable' if short_avg > 0 else 'unprofitable'}"
            ))

        return patterns

    def _analyze_exit_reasons(self, trades: List[Dict]) -> List[TradePattern]:
        """Analyze performance by exit reason."""
        patterns = []

        # Group by exit reason
        by_reason = defaultdict(list)
        for t in trades:
            reason = t.get('exit_reason', 'unknown')
            by_reason[reason].append(t)

        for reason, reason_trades in by_reason.items():
            if len(reason_trades) >= self.min_count:
                winners = [t for t in reason_trades if t.get('net_pnl', 0) > 0]
                win_rate = len(winners) / len(reason_trades)
                avg_profit = np.mean([t.get('net_pnl', 0) for t in reason_trades])

                patterns.append(TradePattern(
                    name=f"exit_{reason}",
                    description=f"Trades exited by {reason}",
                    win_rate=win_rate,
                    avg_profit=avg_profit,
                    trade_count=len(reason_trades),
                    confidence=min(1.0, len(reason_trades) / 30),
                    recommendation=f"{reason.replace('_', ' ').title()} exits "
                                   f"{'working well' if avg_profit > 0 else 'need improvement'}"
                ))

        return patterns

    def get_actionable_insights(self,
                                patterns: List[TradePattern]) -> List[str]:
        """Extract actionable insights from patterns."""
        insights = []

        for p in patterns:
            if p.confidence > 0.5:
                insights.append(f"[{p.confidence:.0%} confidence] {p.recommendation}")

        return insights[:10]  # Top 10 insights
