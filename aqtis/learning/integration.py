"""
Integration module for connecting LearningLog with existing AQTIS systems.

Provides adapters and hooks to automatically record learning data from:
- Backtest results
- Paper trading sessions
- Coach interventions
- Strategy updates
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from .learning_log import LearningLog

logger = logging.getLogger(__name__)


class LearningIntegration:
    """
    Integrates the LearningLog with existing AQTIS components.

    Usage:
        integration = LearningIntegration()

        # After a backtest completes
        integration.record_backtest_result(backtest_result)

        # After coach applies changes
        integration.record_coach_intervention(strategy_name, changes)

        # After paper trading session
        integration.record_paper_trade_session(trades, metrics)
    """

    def __init__(self, db_path: str = "./learning_log.db"):
        self.log = LearningLog(db_path=db_path)

    def record_backtest_result(
        self,
        result: Any,  # BacktestResult from trading_evolution
        strategy_name: str = None,
        strategy_version: int = 1,
        indicator_performance: Dict[str, Dict[str, float]] = None,
    ) -> int:
        """
        Record a backtest result to the learning log.

        Args:
            result: BacktestResult object from trading_evolution.backtest.result
            strategy_name: Override strategy name (uses result.strategy_name if not provided)
            strategy_version: Strategy version number
            indicator_performance: Optional dict of indicator performance metrics

        Returns:
            epoch_id of the recorded epoch
        """
        # Extract metrics from BacktestResult
        # Handle both dict and object formats
        if hasattr(result, 'to_dict'):
            metrics = result.to_dict()
        elif isinstance(result, dict):
            metrics = result
        else:
            metrics = vars(result)

        # Get market regime from trades if available
        market_regime = "unknown"
        if hasattr(result, 'trades') and result.trades:
            # Get most common regime from trades
            regimes = [t.market_regime for t in result.trades if hasattr(t, 'market_regime') and t.market_regime]
            if regimes:
                from collections import Counter
                market_regime = Counter(regimes).most_common(1)[0][0]

        # Get top/worst indicators from indicator_performance
        top_indicators = []
        worst_indicators = []
        if indicator_performance:
            sorted_inds = sorted(
                indicator_performance.items(),
                key=lambda x: x[1].get('win_rate', 0),
                reverse=True
            )
            top_indicators = [name for name, _ in sorted_inds[:5]]
            worst_indicators = [name for name, _ in sorted_inds[-3:]]

        # Record the epoch
        epoch_id = self.log.record_epoch(
            epoch_type="backtest",
            strategy_name=strategy_name or metrics.get('strategy_name', 'unknown'),
            strategy_version=strategy_version,
            total_trades=metrics.get('total_trades', 0),
            winning_trades=metrics.get('winning_trades', 0),
            net_pnl=metrics.get('net_pnl', 0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0),
            max_drawdown=metrics.get('max_drawdown_pct', metrics.get('max_drawdown', 0)),
            profit_factor=metrics.get('profit_factor', 0),
            market_regime=market_regime,
            top_indicators=top_indicators,
            worst_indicators=worst_indicators,
            indicator_performance=indicator_performance,
            notes=f"Backtest from {metrics.get('start_date', '?')} to {metrics.get('end_date', '?')}",
        )

        logger.info(f"Recorded backtest result as epoch {epoch_id}")
        return epoch_id

    def record_coach_intervention(
        self,
        strategy_name: str,
        strategy_version: int,
        changes: Dict[str, Any],
        previous_metrics: Dict[str, float] = None,
        new_metrics: Dict[str, float] = None,
    ) -> int:
        """
        Record a coach intervention and its effects.

        Args:
            strategy_name: Name of the strategy being updated
            strategy_version: New version number after changes
            changes: Dict of changes made (e.g., {'RSI_14_weight': 0.3 -> 0.4})
            previous_metrics: Metrics before coach intervention
            new_metrics: Metrics after coach intervention (if available)

        Returns:
            epoch_id of the recorded epoch
        """
        epoch_id = self.log.record_epoch(
            epoch_type="coach_update",
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            total_trades=new_metrics.get('total_trades', 0) if new_metrics else 0,
            winning_trades=new_metrics.get('winning_trades', 0) if new_metrics else 0,
            net_pnl=new_metrics.get('net_pnl', 0) if new_metrics else 0,
            sharpe_ratio=new_metrics.get('sharpe_ratio', 0) if new_metrics else 0,
            max_drawdown=new_metrics.get('max_drawdown', 0) if new_metrics else 0,
            coach_applied=True,
            coach_changes=changes,
            notes=f"Coach intervention applied. Changes: {list(changes.keys())}",
        )

        # Record as insight
        self.log.record_insight(
            insight_type="coach_intervention",
            category="strategy_update",
            title=f"Coach Updated {strategy_name} v{strategy_version}",
            description=f"Applied {len(changes)} changes to strategy configuration.",
            confidence=0.7,
            evidence={"changes": changes, "previous": previous_metrics, "new": new_metrics},
            actionable=False,
        )

        logger.info(f"Recorded coach intervention as epoch {epoch_id}")
        return epoch_id

    def record_paper_trade_session(
        self,
        trades: List[Dict[str, Any]],
        strategy_name: str,
        strategy_version: int = 1,
        session_date: str = None,
    ) -> int:
        """
        Record a paper trading session to the learning log.

        Args:
            trades: List of trade dicts with entry/exit prices, P&L, etc.
            strategy_name: Name of the strategy used
            strategy_version: Strategy version number
            session_date: Date of the session (defaults to today)

        Returns:
            epoch_id of the recorded epoch
        """
        if not trades:
            logger.warning("No trades to record for paper trading session")
            return -1

        # Calculate session metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get('net_pnl', t.get('pnl', 0)) > 0)
        net_pnl = sum(t.get('net_pnl', t.get('pnl', 0)) for t in trades)

        # Calculate Sharpe if we have enough data
        pnls = [t.get('net_pnl', t.get('pnl', 0)) for t in trades]
        import numpy as np
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0

        # Get market regime from trades
        regimes = [t.get('market_regime', 'unknown') for t in trades]
        from collections import Counter
        market_regime = Counter(regimes).most_common(1)[0][0] if regimes else "unknown"

        # Get symbols traded
        symbols = list(set(t.get('symbol', '') for t in trades if t.get('symbol')))

        epoch_id = self.log.record_epoch(
            epoch_type="paper_trade",
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            total_trades=total_trades,
            winning_trades=winning_trades,
            net_pnl=net_pnl,
            sharpe_ratio=sharpe,
            market_regime=market_regime,
            symbols_traded=symbols,
            notes=f"Paper trading session on {session_date or datetime.now().date()}",
        )

        logger.info(f"Recorded paper trading session as epoch {epoch_id}")
        return epoch_id

    def record_live_trade(
        self,
        trade: Dict[str, Any],
        strategy_name: str,
        strategy_version: int = 1,
    ) -> None:
        """
        Record a single live trade for learning.

        Note: This doesn't create a full epoch, but records the trade
        for model accuracy tracking.
        """
        # Record prediction accuracy if we have predicted vs actual
        if 'predicted_return' in trade and 'actual_return' in trade:
            self.log.record_prediction(
                model_name=strategy_name,
                prediction_type="return",
                predicted_value=trade['predicted_return'],
                actual_value=trade['actual_return'],
                confidence=trade.get('confidence', 0.5),
                market_regime=trade.get('market_regime', 'unknown'),
            )

    def record_indicator_discovery(
        self,
        indicator_name: str,
        category: str,
        discovery_type: str,
        description: str,
        evidence: Dict[str, Any],
        confidence: float = 0.5,
    ) -> int:
        """
        Record a discovered insight about an indicator.

        Args:
            indicator_name: Name of the indicator
            category: Category (momentum, trend, volatility, volume)
            discovery_type: Type of discovery (effectiveness, regime_dependency, correlation)
            description: Human-readable description
            evidence: Supporting evidence (stats, examples)
            confidence: Confidence in the discovery (0-1)

        Returns:
            insight_id
        """
        return self.log.record_insight(
            insight_type="indicator_discovery",
            category=category,
            title=f"{indicator_name}: {discovery_type}",
            description=description,
            confidence=confidence,
            evidence=evidence,
            actionable=confidence > 0.7,
        )

    def analyze_indicator_regime_performance(
        self,
        indicator_performance_by_regime: Dict[str, Dict[str, Dict[str, float]]],
    ) -> List[int]:
        """
        Analyze indicator performance across regimes and record discoveries.

        Args:
            indicator_performance_by_regime: {indicator: {regime: {win_rate, avg_pnl, ...}}}

        Returns:
            List of insight_ids for discoveries
        """
        insight_ids = []

        for indicator, regime_data in indicator_performance_by_regime.items():
            # Find regime-specific strengths
            best_regime = None
            best_win_rate = 0
            worst_regime = None
            worst_win_rate = 1.0

            for regime, metrics in regime_data.items():
                wr = metrics.get('win_rate', 0)
                if wr > best_win_rate:
                    best_win_rate = wr
                    best_regime = regime
                if wr < worst_win_rate:
                    worst_win_rate = wr
                    worst_regime = regime

            # Record insight if there's significant regime dependency
            if best_win_rate - worst_win_rate > 0.15:  # 15% difference
                insight_id = self.record_indicator_discovery(
                    indicator_name=indicator,
                    category="regime_dependency",
                    discovery_type="Regime-Specific Performance",
                    description=f"{indicator} performs best in {best_regime} ({best_win_rate*100:.1f}% win rate) "
                               f"and worst in {worst_regime} ({worst_win_rate*100:.1f}% win rate). "
                               f"Consider adjusting weight based on regime.",
                    evidence={
                        "best_regime": best_regime,
                        "best_win_rate": best_win_rate,
                        "worst_regime": worst_regime,
                        "worst_win_rate": worst_win_rate,
                        "regime_data": regime_data,
                    },
                    confidence=0.8,
                )
                insight_ids.append(insight_id)

        return insight_ids

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status and recent progress."""
        summary = self.log.get_learning_summary(days=7)
        trends = self.log.get_improvement_trends(periods=10)

        return {
            "summary": summary,
            "trends": trends,
            "is_improving": trends.get('sharpe_ratio', {}).get('trend') == 'improving',
            "recent_insights": summary.get('recent_insights', [])[:5],
            "recommendation": self._generate_recommendation(summary, trends),
        }

    def _generate_recommendation(self, summary: Dict, trends: Dict) -> str:
        """Generate a recommendation based on current learning status."""
        sharpe_trend = trends.get('sharpe_ratio', {}).get('trend', 'stable')
        win_rate_trend = trends.get('win_rate', {}).get('trend', 'stable')
        improvement_rate = summary.get('improvement_rate', 0)

        if sharpe_trend == 'declining' and win_rate_trend == 'declining':
            return "Consider reviewing recent strategy changes. Both Sharpe and win rate are declining."
        elif sharpe_trend == 'improving' and improvement_rate > 0.6:
            return "Strategy is learning well. Continue current approach."
        elif improvement_rate < 0.3:
            return "Low improvement rate. Consider enabling coach interventions or reviewing indicator weights."
        elif sharpe_trend == 'stable':
            return "Performance is stable. Consider testing new indicator combinations."
        else:
            return "Continue monitoring. Mixed signals in learning metrics."


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_learning_integration(db_path: str = "./learning_log.db") -> LearningIntegration:
    """Create a LearningIntegration instance."""
    return LearningIntegration(db_path=db_path)


def record_from_backtest(result: Any, db_path: str = "./learning_log.db", **kwargs) -> int:
    """Quick function to record a backtest result."""
    integration = LearningIntegration(db_path=db_path)
    return integration.record_backtest_result(result, **kwargs)
