"""
Coach Agent (Analyzer) module.

The Coach analyzes trading performance and guides evolution.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from .indicator_scorer import IndicatorScorer, IndicatorContribution
from .pattern_detector import PatternDetector, TradePattern
from .regime_analyzer import RegimeAnalyzer, RegimePerformance
from .recommendations import RecommendationGenerator, CoachRecommendation
from ..journal.database import Database

logger = logging.getLogger(__name__)


class Coach:
    """
    Coach Agent: Analyzes performance and guides evolution.

    Responsibilities:
    - Analyze trade journal to identify winning patterns
    - Rank indicators by contribution to profitable trades
    - Detect market regimes and adapt recommendations
    - Provide evolution guidance to genetic algorithm
    - Prevent overfitting through validation checks

    The Coach runs after each generation to:
    1. Score each indicator's contribution
    2. Find patterns in winning/losing trades
    3. Analyze performance by market regime
    4. Generate recommendations for next generation
    """

    def __init__(self,
                 database: Database = None,
                 indicator_scorer: IndicatorScorer = None,
                 pattern_detector: PatternDetector = None,
                 regime_analyzer: RegimeAnalyzer = None,
                 recommendation_generator: RecommendationGenerator = None):
        """
        Initialize Coach agent.

        Args:
            database: Database for storing analysis
            indicator_scorer: Indicator scoring component
            pattern_detector: Pattern detection component
            regime_analyzer: Regime analysis component
            recommendation_generator: Recommendation generation component
        """
        self.db = database
        self.indicator_scorer = indicator_scorer or IndicatorScorer()
        self.pattern_detector = pattern_detector or PatternDetector()
        self.regime_analyzer = regime_analyzer or RegimeAnalyzer()
        self.recommendation_generator = recommendation_generator or RecommendationGenerator()

        # History tracking
        self.analysis_history: List[Dict] = []

    def analyze_generation(self,
                           generation: int,
                           run_id: int,
                           trades: List[Dict],
                           indicator_snapshots: Dict[str, Dict],
                           weights: Dict[str, float],
                           market_data: Dict[str, pd.DataFrame],
                           performance_metrics: Dict) -> CoachRecommendation:
        """
        Comprehensive analysis of a generation's performance.

        Args:
            generation: Generation number
            run_id: Evolution run ID
            trades: List of completed trades
            indicator_snapshots: Indicator values at each trade entry
            weights: Current indicator weights
            market_data: Market data by symbol
            performance_metrics: Overall performance metrics

        Returns:
            CoachRecommendation with guidance for next generation
        """
        logger.info(f"Coach analyzing generation {generation}")

        # 1. Score each indicator's contribution
        indicator_scores = self.indicator_scorer.score_indicators(
            trades, indicator_snapshots, weights
        )
        logger.info(f"Scored {len(indicator_scores)} indicators")

        # 2. Detect patterns in trades
        patterns = self.pattern_detector.find_patterns(trades, indicator_snapshots)
        logger.info(f"Found {len(patterns)} patterns")

        # 3. Analyze performance by market regime
        regime_performances = []
        for symbol, df in market_data.items():
            symbol_trades = [t for t in trades if t.get('symbol') == symbol]
            if symbol_trades:
                perfs = self.regime_analyzer.analyze_performance_by_regime(
                    symbol_trades, df
                )
                regime_performances.extend(perfs)
        logger.info(f"Analyzed {len(regime_performances)} regime performances")

        # 4. Generate recommendations
        recommendation = self.recommendation_generator.generate(
            indicator_scores=indicator_scores,
            patterns=patterns,
            regime_performance=regime_performances,
            performance_metrics=performance_metrics,
            current_weights=weights
        )

        # 5. Store analysis
        analysis = {
            'generation': generation,
            'run_id': run_id,
            'num_trades': len(trades),
            'indicator_scores': indicator_scores,
            'patterns': patterns,
            'regime_performances': regime_performances,
            'recommendation': recommendation,
            'metrics': performance_metrics
        }
        self.analysis_history.append(analysis)

        # 6. Save to database if available
        if self.db:
            self._save_analysis(run_id, generation, indicator_scores, recommendation)

        logger.info(f"Coach recommendation confidence: {recommendation.confidence_level:.2%}")

        return recommendation

    def _save_analysis(self,
                       run_id: int,
                       generation: int,
                       scores: List[IndicatorContribution],
                       recommendation: CoachRecommendation):
        """Save analysis to database."""
        # Save indicator performance
        performances = []
        for score in scores:
            performances.append({
                'indicator_name': score.name,
                'weight_used': score.weight_used,
                'correlation_with_returns': score.correlation_with_pnl,
                'predictive_accuracy': score.predictive_accuracy,
                'signal_quality_score': score.signal_quality_score,
                'avg_return_bullish': score.avg_return_bullish,
                'avg_return_bearish': score.avg_return_bearish,
                'consistency_score': score.consistency_score,
                'information_ratio': score.information_ratio,
                'trades_contributed': score.trades_contributed,
                'rank_in_generation': score.rank,
                'long_accuracy': score.long_accuracy,
                'short_accuracy': score.short_accuracy
            })

        self.db.save_indicator_performance(run_id, generation, performances)

        # Save recommendation
        self.db.save_coach_recommendation(run_id, generation, {
            'promoted_indicators': recommendation.indicators_to_promote,
            'demoted_indicators': recommendation.indicators_to_demote,
            'removed_indicators': recommendation.indicators_to_remove,
            'weight_adjustments': recommendation.weight_adjustments,
            'regime_insights': recommendation.regime_insights,
            'pattern_insights': recommendation.pattern_insights,
            'confidence_level': recommendation.confidence_level
        })

    def validate_against_holdout(self,
                                 dna_fitness: float,
                                 holdout_fitness: float,
                                 previous_best: float,
                                 tolerance: float = 0.2) -> Tuple[bool, str]:
        """
        Validate DNA against holdout data to prevent overfitting.

        Args:
            dna_fitness: Fitness on training data
            holdout_fitness: Fitness on holdout data
            previous_best: Previous best holdout fitness
            tolerance: Acceptable degradation tolerance (default 20%)

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check for significant gap between training and holdout
        gap = (dna_fitness - holdout_fitness) / dna_fitness if dna_fitness > 0 else 0

        if gap > 0.3:  # More than 30% gap suggests overfitting
            return False, f"Overfitting detected: training={dna_fitness:.4f}, holdout={holdout_fitness:.4f}"

        # Check for degradation from previous best
        if previous_best > 0:
            degradation = (previous_best - holdout_fitness) / previous_best
            if degradation > tolerance:
                return False, f"Holdout performance degraded by {degradation:.1%}"

        return True, "Validation passed"

    def get_evolution_guidance(self,
                               recommendation: CoachRecommendation) -> Dict:
        """
        Convert recommendation to evolution parameters.

        Returns dict that can guide mutation/crossover operations.
        """
        return {
            'increase_mutation_for': recommendation.indicators_to_demote,
            'decrease_mutation_for': recommendation.indicators_to_promote,
            'set_to_zero': recommendation.indicators_to_remove,
            'weight_targets': recommendation.weight_adjustments,
            'confidence': recommendation.confidence_level
        }

    def get_top_indicators(self, n: int = 10) -> List[str]:
        """Get top performing indicators from latest analysis."""
        if not self.analysis_history:
            return []

        latest = self.analysis_history[-1]
        scores = latest.get('indicator_scores', [])

        return [s.name for s in scores[:n]]

    def get_bottom_indicators(self, n: int = 10) -> List[str]:
        """Get worst performing indicators from latest analysis."""
        if not self.analysis_history:
            return []

        latest = self.analysis_history[-1]
        scores = latest.get('indicator_scores', [])

        return [s.name for s in scores[-n:]]

    def get_analysis_summary(self, generation: int = None) -> str:
        """Get human-readable summary of analysis."""
        if not self.analysis_history:
            return "No analysis available"

        if generation is not None:
            analysis = next(
                (a for a in self.analysis_history if a['generation'] == generation),
                None
            )
        else:
            analysis = self.analysis_history[-1]

        if not analysis:
            return "Analysis not found"

        rec = analysis['recommendation']
        metrics = analysis.get('metrics', {})

        lines = [
            f"Generation {analysis['generation']} Analysis",
            "=" * 40,
            f"Trades analyzed: {analysis['num_trades']}",
            f"Win rate: {metrics.get('win_rate', 0):.1%}",
            f"Net profit: ${metrics.get('net_profit', 0):.2f}",
            f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"Max drawdown: {metrics.get('max_drawdown', 0):.1%}",
            "",
            "Top indicators:",
        ]

        scores = analysis.get('indicator_scores', [])
        for s in scores[:5]:
            lines.append(f"  {s.name}: {s.signal_quality_score:.3f} (rank {s.rank})")

        lines.extend([
            "",
            "Recommendations:",
            f"  Promote: {len(rec.indicators_to_promote)} indicators",
            f"  Demote: {len(rec.indicators_to_demote)} indicators",
            f"  Remove: {len(rec.indicators_to_remove)} indicators",
            f"  Confidence: {rec.confidence_level:.1%}",
            "",
            rec.summary
        ])

        return "\n".join(lines)

    def compare_generations(self,
                            gen1: int,
                            gen2: int) -> Dict:
        """Compare performance between two generations."""
        analysis1 = next(
            (a for a in self.analysis_history if a['generation'] == gen1),
            None
        )
        analysis2 = next(
            (a for a in self.analysis_history if a['generation'] == gen2),
            None
        )

        if not analysis1 or not analysis2:
            return {'error': 'Generation not found'}

        m1 = analysis1.get('metrics', {})
        m2 = analysis2.get('metrics', {})

        return {
            'win_rate_change': m2.get('win_rate', 0) - m1.get('win_rate', 0),
            'profit_change': m2.get('net_profit', 0) - m1.get('net_profit', 0),
            'sharpe_change': m2.get('sharpe_ratio', 0) - m1.get('sharpe_ratio', 0),
            'drawdown_change': m2.get('max_drawdown', 0) - m1.get('max_drawdown', 0),
            'improved': m2.get('net_profit', 0) > m1.get('net_profit', 0)
        }
