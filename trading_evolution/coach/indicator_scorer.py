"""
Indicator Scorer module.

Scores each indicator's contribution to trading performance.
Used by Coach to guide evolution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class IndicatorContribution:
    """Score for an indicator's contribution to performance."""
    name: str
    weight_used: float
    correlation_with_pnl: float = 0.0
    predictive_accuracy: float = 0.0
    signal_quality_score: float = 0.0
    avg_return_bullish: float = 0.0
    avg_return_bearish: float = 0.0
    consistency_score: float = 0.0
    information_ratio: float = 0.0
    trades_contributed: int = 0
    rank: int = 0
    long_accuracy: float = 0.0
    short_accuracy: float = 0.0


class IndicatorScorer:
    """
    Scores each indicator's contribution to trading performance.

    Scoring Methodology:
    1. Correlation: How well indicator correlates with trade P&L
    2. Predictive accuracy: % of times indicator direction matched trade outcome
    3. Information ratio: Risk-adjusted predictive power
    4. Consistency: Performance across different time periods
    5. Long/Short accuracy: Separate scores for long vs short trades
    """

    def __init__(self, min_trades: int = 10):
        """
        Initialize scorer.

        Args:
            min_trades: Minimum trades for valid scoring
        """
        self.min_trades = min_trades

    def score_indicators(self,
                         trades: List[Dict],
                         indicator_snapshots: Dict[str, Dict[str, float]],
                         weights: Dict[str, float]) -> List[IndicatorContribution]:
        """
        Score all indicators based on trade performance.

        Args:
            trades: List of completed trades with 'trade_id' and 'net_pnl'
            indicator_snapshots: Dict of trade_id -> indicator values at entry
            weights: Dict of indicator_name -> weight used

        Returns:
            List of IndicatorContribution sorted by quality score
        """
        if len(trades) < self.min_trades:
            logger.warning(f"Insufficient trades ({len(trades)}) for scoring")
            return []

        # Build DataFrame of indicator values at trade entries
        trade_data = []
        for trade in trades:
            trade_id = trade.get('trade_id')
            if trade_id not in indicator_snapshots:
                continue

            row = {
                'trade_id': trade_id,
                'pnl': trade.get('net_pnl', 0),
                'pnl_pct': trade.get('net_pnl_pct', 0),
                'direction': trade.get('direction', 'LONG'),
                **indicator_snapshots[trade_id]
            }
            trade_data.append(row)

        if not trade_data:
            return []

        df = pd.DataFrame(trade_data)

        # Score each indicator
        scores = []
        indicator_cols = [c for c in df.columns
                         if c not in ['trade_id', 'pnl', 'pnl_pct', 'direction']]

        for col in indicator_cols:
            if col not in weights:
                continue

            try:
                score = self._score_single_indicator(
                    df, col, weights[col]
                )
                scores.append(score)
            except Exception as e:
                logger.debug(f"Failed to score {col}: {e}")

        # Sort by signal quality and assign ranks
        scores.sort(key=lambda x: x.signal_quality_score, reverse=True)
        for i, score in enumerate(scores):
            score.rank = i + 1

        return scores

    def _score_single_indicator(self,
                                df: pd.DataFrame,
                                indicator_name: str,
                                weight: float) -> IndicatorContribution:
        """Score a single indicator."""
        ind_values = df[indicator_name].dropna()
        pnl_values = df.loc[ind_values.index, 'pnl']
        directions = df.loc[ind_values.index, 'direction']

        if len(ind_values) < self.min_trades:
            return IndicatorContribution(
                name=indicator_name,
                weight_used=weight,
                trades_contributed=len(ind_values)
            )

        # 1. Correlation with P&L
        correlation = ind_values.corr(pnl_values)
        if np.isnan(correlation):
            correlation = 0.0

        # 2. Predictive accuracy
        # For positive weight: positive indicator should mean positive P&L
        # For negative weight: negative indicator should mean positive P&L
        if weight >= 0:
            correct = ((ind_values > 0) & (pnl_values > 0)) | \
                      ((ind_values < 0) & (pnl_values < 0))
        else:
            correct = ((ind_values < 0) & (pnl_values > 0)) | \
                      ((ind_values > 0) & (pnl_values < 0))

        accuracy = correct.mean()

        # 3. Average return when bullish vs bearish
        bullish_mask = ind_values > 0.3
        bearish_mask = ind_values < -0.3

        avg_bullish = pnl_values[bullish_mask].mean() if bullish_mask.sum() > 3 else 0
        avg_bearish = pnl_values[bearish_mask].mean() if bearish_mask.sum() > 3 else 0

        if np.isnan(avg_bullish):
            avg_bullish = 0
        if np.isnan(avg_bearish):
            avg_bearish = 0

        # 4. Information ratio
        try:
            ir, _ = stats.spearmanr(ind_values, pnl_values)
            if np.isnan(ir):
                ir = 0.0
        except:
            ir = 0.0

        # 5. Consistency (split data and check both halves)
        mid = len(ind_values) // 2
        corr1 = ind_values.iloc[:mid].corr(pnl_values.iloc[:mid])
        corr2 = ind_values.iloc[mid:].corr(pnl_values.iloc[mid:])

        if np.isnan(corr1):
            corr1 = 0
        if np.isnan(corr2):
            corr2 = 0

        # Consistency high if both halves agree
        if corr1 * corr2 > 0:
            consistency = min(abs(corr1), abs(corr2))
        else:
            consistency = 0

        # 6. Long/Short accuracy
        long_mask = directions == 'LONG'
        short_mask = directions == 'SHORT'

        long_trades = pnl_values[long_mask]
        short_trades = pnl_values[short_mask]
        long_ind = ind_values[long_mask]
        short_ind = ind_values[short_mask]

        long_accuracy = 0.0
        short_accuracy = 0.0

        if len(long_trades) >= 5:
            long_correct = ((long_ind > 0) & (long_trades > 0)).sum()
            long_accuracy = long_correct / len(long_trades)

        if len(short_trades) >= 5:
            # For shorts, negative indicator should correlate with profit
            short_correct = ((short_ind < 0) & (short_trades > 0)).sum()
            short_accuracy = short_correct / len(short_trades)

        # 7. Composite signal quality score
        signal_quality = (
            abs(correlation) * 0.25 +
            accuracy * 0.25 +
            abs(ir) * 0.20 +
            abs(avg_bullish - avg_bearish) / max(1, abs(avg_bullish) + abs(avg_bearish)) * 0.15 +
            consistency * 0.15
        )

        return IndicatorContribution(
            name=indicator_name,
            weight_used=weight,
            correlation_with_pnl=correlation,
            predictive_accuracy=accuracy,
            signal_quality_score=signal_quality,
            avg_return_bullish=avg_bullish,
            avg_return_bearish=avg_bearish,
            consistency_score=consistency,
            information_ratio=ir,
            trades_contributed=len(ind_values),
            long_accuracy=long_accuracy,
            short_accuracy=short_accuracy
        )

    def get_top_performers(self,
                           scores: List[IndicatorContribution],
                           n: int = 10) -> List[str]:
        """Get names of top performing indicators."""
        return [s.name for s in scores[:n]]

    def get_bottom_performers(self,
                              scores: List[IndicatorContribution],
                              n: int = 10) -> List[str]:
        """Get names of worst performing indicators."""
        return [s.name for s in scores[-n:]]

    def get_by_category(self,
                        scores: List[IndicatorContribution],
                        categories: Dict[str, str]) -> Dict[str, List[IndicatorContribution]]:
        """Group scores by indicator category."""
        grouped = {}
        for score in scores:
            cat = categories.get(score.name, 'other')
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(score)
        return grouped
