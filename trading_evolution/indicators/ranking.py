"""
Indicator Ranking module.

Ranks indicators by their predictive power and contribution to trading performance.
Used by the Coach to guide evolution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class IndicatorScore:
    """Score for a single indicator."""
    name: str
    correlation_with_returns: float = 0.0
    predictive_accuracy: float = 0.0  # % of correct direction predictions
    signal_quality: float = 0.0  # Composite score
    avg_return_bullish: float = 0.0  # Return when indicator is bullish
    avg_return_bearish: float = 0.0  # Return when indicator is bearish
    information_coefficient: float = 0.0  # IC
    consistency: float = 0.0  # Consistency across time windows
    rank: int = 0


@dataclass
class RankingResult:
    """Complete ranking results for a set of indicators."""
    scores: List[IndicatorScore] = field(default_factory=list)
    top_indicators: List[str] = field(default_factory=list)
    bottom_indicators: List[str] = field(default_factory=list)
    category_leaders: Dict[str, str] = field(default_factory=dict)


class IndicatorRanker:
    """
    Ranks indicators by their predictive power for trading.

    Scoring Methodology:
    1. Correlation with forward returns
    2. Directional accuracy (did signal predict direction?)
    3. Information Coefficient (rank correlation)
    4. Consistency across different time periods
    5. Signal quality composite score
    """

    def __init__(self, forward_periods: int = 5):
        """
        Initialize ranker.

        Args:
            forward_periods: Number of periods ahead for return calculation
        """
        self.forward_periods = forward_periods

    def rank_indicators(self, indicators_df: pd.DataFrame,
                        returns: pd.Series,
                        categories: Dict[str, str] = None) -> RankingResult:
        """
        Rank all indicators by predictive power.

        Args:
            indicators_df: DataFrame with normalized indicator values
            returns: Series of forward returns
            categories: Optional dict mapping indicator -> category

        Returns:
            RankingResult with scores and rankings
        """
        scores = []

        # Align data
        common_idx = indicators_df.index.intersection(returns.index)
        indicators_aligned = indicators_df.loc[common_idx]
        returns_aligned = returns.loc[common_idx]

        for col in indicators_df.columns:
            try:
                score = self._score_indicator(
                    indicators_aligned[col],
                    returns_aligned,
                    col
                )
                scores.append(score)
            except Exception as e:
                logger.debug(f"Failed to score {col}: {e}")

        # Sort by signal quality
        scores.sort(key=lambda x: x.signal_quality, reverse=True)

        # Assign ranks
        for i, score in enumerate(scores):
            score.rank = i + 1

        # Identify top and bottom performers
        n_top = min(20, len(scores))
        top_indicators = [s.name for s in scores[:n_top]]
        bottom_indicators = [s.name for s in scores[-n_top:]] if len(scores) > n_top else []

        # Find category leaders
        category_leaders = {}
        if categories:
            for category in set(categories.values()):
                category_scores = [s for s in scores if categories.get(s.name) == category]
                if category_scores:
                    category_leaders[category] = category_scores[0].name

        return RankingResult(
            scores=scores,
            top_indicators=top_indicators,
            bottom_indicators=bottom_indicators,
            category_leaders=category_leaders
        )

    def _score_indicator(self, indicator: pd.Series,
                         returns: pd.Series, name: str) -> IndicatorScore:
        """Calculate score for a single indicator."""
        # Remove NaN
        mask = indicator.notna() & returns.notna()
        ind_clean = indicator[mask]
        ret_clean = returns[mask]

        if len(ind_clean) < 30:
            return IndicatorScore(name=name)

        # 1. Correlation with returns
        correlation = ind_clean.corr(ret_clean)
        if np.isnan(correlation):
            correlation = 0.0

        # 2. Directional accuracy
        # Bullish signal (> 0) should predict positive return
        # Bearish signal (< 0) should predict negative return
        correct_direction = (
            ((ind_clean > 0) & (ret_clean > 0)) |
            ((ind_clean < 0) & (ret_clean < 0))
        )
        accuracy = correct_direction.mean()

        # 3. Information Coefficient (Spearman rank correlation)
        try:
            ic, _ = stats.spearmanr(ind_clean, ret_clean)
            if np.isnan(ic):
                ic = 0.0
        except:
            ic = 0.0

        # 4. Return when bullish vs bearish
        bullish_mask = ind_clean > 0.3  # Strong bullish
        bearish_mask = ind_clean < -0.3  # Strong bearish

        avg_ret_bullish = ret_clean[bullish_mask].mean() if bullish_mask.sum() > 5 else 0
        avg_ret_bearish = ret_clean[bearish_mask].mean() if bearish_mask.sum() > 5 else 0

        if np.isnan(avg_ret_bullish):
            avg_ret_bullish = 0
        if np.isnan(avg_ret_bearish):
            avg_ret_bearish = 0

        # 5. Consistency (split into halves and check correlation in both)
        mid = len(ind_clean) // 2
        corr1 = ind_clean.iloc[:mid].corr(ret_clean.iloc[:mid])
        corr2 = ind_clean.iloc[mid:].corr(ret_clean.iloc[mid:])

        if np.isnan(corr1):
            corr1 = 0
        if np.isnan(corr2):
            corr2 = 0

        # Consistency score: high if both halves have same sign correlation
        if corr1 * corr2 > 0:  # Same sign
            consistency = min(abs(corr1), abs(corr2))
        else:
            consistency = 0

        # 6. Composite signal quality score
        # Weighted combination of metrics
        signal_quality = (
            abs(correlation) * 0.25 +
            accuracy * 0.25 +
            abs(ic) * 0.20 +
            (avg_ret_bullish - avg_ret_bearish) * 0.15 +
            consistency * 0.15
        )

        return IndicatorScore(
            name=name,
            correlation_with_returns=correlation,
            predictive_accuracy=accuracy,
            signal_quality=signal_quality,
            avg_return_bullish=avg_ret_bullish,
            avg_return_bearish=avg_ret_bearish,
            information_coefficient=ic,
            consistency=consistency
        )

    def rank_for_trades(self, indicators_df: pd.DataFrame,
                        trades: List[Dict],
                        indicator_snapshots: Dict[str, Dict]) -> RankingResult:
        """
        Rank indicators based on actual trade performance.

        Args:
            indicators_df: Indicator values DataFrame
            trades: List of trade dictionaries with 'net_pnl'
            indicator_snapshots: Indicator values at each trade entry

        Returns:
            RankingResult based on trade contribution
        """
        if not trades or not indicator_snapshots:
            return RankingResult()

        # Build DataFrame of indicator values at trade entries
        trade_indicators = []
        trade_returns = []

        for trade in trades:
            trade_id = trade.get('trade_id')
            if trade_id in indicator_snapshots:
                trade_indicators.append(indicator_snapshots[trade_id])
                trade_returns.append(trade.get('net_pnl', 0))

        if not trade_indicators:
            return RankingResult()

        ind_df = pd.DataFrame(trade_indicators)
        ret_series = pd.Series(trade_returns, index=ind_df.index)

        # Normalize returns for comparison
        if ret_series.std() > 0:
            ret_normalized = (ret_series - ret_series.mean()) / ret_series.std()
        else:
            ret_normalized = ret_series

        return self.rank_indicators(ind_df, ret_normalized)


def calculate_forward_returns(prices: pd.Series, periods: int = 5) -> pd.Series:
    """
    Calculate forward returns for ranking.

    Args:
        prices: Close price series
        periods: Number of periods ahead

    Returns:
        Series of forward returns
    """
    return prices.pct_change(periods).shift(-periods)


def create_category_mapping(universe) -> Dict[str, str]:
    """
    Create indicator -> category mapping from universe.

    Args:
        universe: IndicatorUniverse instance

    Returns:
        Dict mapping indicator name to category
    """
    mapping = {}
    for name in universe.get_all():
        defn = universe.get_definition(name)
        if defn:
            mapping[name] = defn.category
    return mapping
