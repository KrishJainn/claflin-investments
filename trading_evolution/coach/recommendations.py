"""
Coach Recommendations module.

Generates strategy recommendations based on analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import logging

from .indicator_scorer import IndicatorContribution
from .pattern_detector import TradePattern
from .regime_analyzer import RegimePerformance, MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class CoachRecommendation:
    """Complete recommendation from Coach analysis."""
    # Indicator changes
    indicators_to_promote: List[str] = field(default_factory=list)
    indicators_to_demote: List[str] = field(default_factory=list)
    indicators_to_remove: List[str] = field(default_factory=list)
    indicators_to_add: List[str] = field(default_factory=list)

    # Weight adjustments
    weight_adjustments: Dict[str, float] = field(default_factory=dict)

    # Insights
    regime_insights: str = ""
    pattern_insights: str = ""
    performance_insights: str = ""

    # Confidence
    confidence_level: float = 0.0

    # Specific recommendations
    entry_threshold_adjustment: float = 0.0  # +/- to current threshold
    exit_threshold_adjustment: float = 0.0
    stop_loss_adjustment: float = 0.0  # Multiplier (1.0 = no change)

    # Summary
    summary: str = ""


class RecommendationGenerator:
    """
    Generates strategy recommendations from Coach analysis.

    Combines insights from:
    - Indicator scoring
    - Pattern detection
    - Regime analysis
    - Performance metrics
    """

    def __init__(self,
                 promote_threshold: float = 0.6,
                 demote_threshold: float = 0.4,
                 remove_threshold: float = 0.25,
                 min_confidence: float = 0.5):
        """
        Initialize recommendation generator.

        Args:
            promote_threshold: Signal quality above this -> promote
            demote_threshold: Signal quality below this -> demote
            remove_threshold: Signal quality below this -> remove
            min_confidence: Minimum confidence to include recommendation
        """
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.remove_threshold = remove_threshold
        self.min_confidence = min_confidence

    def generate(self,
                 indicator_scores: List[IndicatorContribution],
                 patterns: List[TradePattern],
                 regime_performance: List[RegimePerformance],
                 performance_metrics: Dict,
                 current_weights: Dict[str, float]) -> CoachRecommendation:
        """
        Generate comprehensive recommendation.

        Args:
            indicator_scores: Scored indicators
            patterns: Detected patterns
            regime_performance: Performance by regime
            performance_metrics: Overall performance metrics
            current_weights: Current indicator weights

        Returns:
            CoachRecommendation with actionable guidance
        """
        rec = CoachRecommendation()

        # 1. Analyze indicator scores
        self._analyze_indicators(rec, indicator_scores, current_weights)

        # 2. Analyze patterns
        self._analyze_patterns(rec, patterns)

        # 3. Analyze regime performance
        self._analyze_regimes(rec, regime_performance)

        # 4. Analyze overall performance
        self._analyze_performance(rec, performance_metrics)

        # 5. Calculate overall confidence
        rec.confidence_level = self._calculate_confidence(
            indicator_scores, patterns, regime_performance
        )

        # 6. Generate summary
        rec.summary = self._generate_summary(rec)

        return rec

    def _analyze_indicators(self,
                            rec: CoachRecommendation,
                            scores: List[IndicatorContribution],
                            current_weights: Dict[str, float]):
        """Analyze indicator scores and recommend changes."""
        if not scores:
            return

        for score in scores:
            quality = score.signal_quality_score
            current_weight = current_weights.get(score.name, 0)

            # Promote top performers
            if quality >= self.promote_threshold:
                rec.indicators_to_promote.append(score.name)
                # Increase weight by 20%
                new_weight = current_weight * 1.2 if current_weight != 0 else 0.5
                rec.weight_adjustments[score.name] = np.clip(new_weight, -1, 1)

            # Demote poor performers
            elif quality < self.demote_threshold:
                rec.indicators_to_demote.append(score.name)
                # Decrease weight by 30%
                new_weight = current_weight * 0.7
                rec.weight_adjustments[score.name] = new_weight

                # Remove very poor performers
                if quality < self.remove_threshold:
                    rec.indicators_to_remove.append(score.name)
                    rec.weight_adjustments[score.name] = 0.0

        # Check for bias (all indicators same direction)
        long_biased = sum(1 for s in scores if s.long_accuracy > s.short_accuracy)
        short_biased = sum(1 for s in scores if s.short_accuracy > s.long_accuracy)

        if long_biased > len(scores) * 0.7:
            rec.pattern_insights += "Strategy is biased toward long trades. "
        elif short_biased > len(scores) * 0.7:
            rec.pattern_insights += "Strategy is biased toward short trades. "

    def _analyze_patterns(self,
                          rec: CoachRecommendation,
                          patterns: List[TradePattern]):
        """Analyze patterns and add insights."""
        if not patterns:
            return

        insights = []

        for pattern in patterns[:5]:  # Top 5 patterns
            if pattern.confidence >= self.min_confidence:
                insights.append(pattern.recommendation)

                # Signal strength patterns
                if 'signal_strength' in pattern.name:
                    if 'very_strong' in pattern.name and pattern.win_rate > 0.6:
                        rec.entry_threshold_adjustment = 0.05  # Tighter threshold
                    elif 'weak' in pattern.name and pattern.win_rate < 0.4:
                        rec.entry_threshold_adjustment = -0.05  # Looser threshold

                # Exit patterns
                if 'exit_stop' in pattern.name and pattern.avg_profit < 0:
                    rec.stop_loss_adjustment = 0.9  # Tighter stops

        rec.pattern_insights = "; ".join(insights) if insights else "No significant patterns detected"

    def _analyze_regimes(self,
                         rec: CoachRecommendation,
                         performances: List[RegimePerformance]):
        """Analyze regime performance and add insights."""
        if not performances:
            return

        insights = []

        for perf in performances:
            if perf.trade_count < 10:
                continue

            regime_name = perf.regime.value.replace('_', ' ').title()

            if perf.win_rate > 0.55:
                insights.append(f"{regime_name}: Strong ({perf.win_rate:.0%} win rate)")
            elif perf.win_rate < 0.45:
                insights.append(f"{regime_name}: Weak ({perf.win_rate:.0%} win rate)")

            # Direction bias by regime
            if perf.long_win_rate > perf.short_win_rate + 0.15:
                insights.append(f"In {regime_name}, favor longs")
            elif perf.short_win_rate > perf.long_win_rate + 0.15:
                insights.append(f"In {regime_name}, favor shorts")

        rec.regime_insights = "; ".join(insights) if insights else "Performance consistent across regimes"

    def _analyze_performance(self,
                             rec: CoachRecommendation,
                             metrics: Dict):
        """Analyze overall performance."""
        insights = []

        win_rate = metrics.get('win_rate', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)
        profit_factor = metrics.get('profit_factor', 0)

        if win_rate < 0.4:
            insights.append("Low win rate suggests entry signals need improvement")
        elif win_rate > 0.6:
            insights.append("High win rate - consider taking more risk per trade")

        if sharpe < 0.5:
            insights.append("Low Sharpe ratio - need better risk-adjusted returns")
        elif sharpe > 1.5:
            insights.append("Excellent Sharpe ratio - strategy is working well")

        if max_dd > 0.2:
            insights.append("High drawdown - consider tighter stops or smaller positions")
            rec.stop_loss_adjustment = 0.85  # Tighter stops

        if profit_factor < 1.2:
            insights.append("Low profit factor - winners not large enough vs losers")
        elif profit_factor > 2.0:
            insights.append("Excellent profit factor - strategy captures big moves")

        rec.performance_insights = "; ".join(insights) if insights else "Performance metrics acceptable"

    def _calculate_confidence(self,
                              scores: List[IndicatorContribution],
                              patterns: List[TradePattern],
                              performances: List[RegimePerformance]) -> float:
        """Calculate overall confidence in recommendations."""
        confidences = []

        # Confidence from indicator scores
        if scores:
            avg_quality = np.mean([s.signal_quality_score for s in scores])
            confidences.append(min(1.0, avg_quality * 1.5))

        # Confidence from patterns
        if patterns:
            avg_pattern_conf = np.mean([p.confidence for p in patterns[:5]])
            confidences.append(avg_pattern_conf)

        # Confidence from regime analysis
        if performances:
            total_trades = sum(p.trade_count for p in performances)
            regime_conf = min(1.0, total_trades / 100)  # More trades = more confidence
            confidences.append(regime_conf)

        return np.mean(confidences) if confidences else 0.5

    def _generate_summary(self, rec: CoachRecommendation) -> str:
        """Generate human-readable summary."""
        parts = []

        if rec.indicators_to_promote:
            parts.append(f"Promote {len(rec.indicators_to_promote)} indicators")

        if rec.indicators_to_demote:
            parts.append(f"Demote {len(rec.indicators_to_demote)} indicators")

        if rec.indicators_to_remove:
            parts.append(f"Remove {len(rec.indicators_to_remove)} underperformers")

        if rec.entry_threshold_adjustment != 0:
            direction = "tighter" if rec.entry_threshold_adjustment > 0 else "looser"
            parts.append(f"Make entry signals {direction}")

        if rec.stop_loss_adjustment != 0 and rec.stop_loss_adjustment != 1.0:
            direction = "tighter" if rec.stop_loss_adjustment < 1.0 else "looser"
            parts.append(f"Make stop losses {direction}")

        parts.append(f"Confidence: {rec.confidence_level:.0%}")

        return ". ".join(parts)
