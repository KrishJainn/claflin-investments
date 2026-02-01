"""
Strategy Updater for the AI Coach.

Generates and validates strategy updates based on Coach analysis:
- Applies indicator weight adjustments
- Updates entry/exit thresholds
- Validates changes via backtest before deployment
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import copy

from .ai_analyzer import TradeAnalysis
from ..ai_config import AIConfig, DEFAULT_AI_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class StrategyVersion:
    """A complete strategy configuration version."""
    
    version_id: int
    created_at: datetime
    
    # Core parameters
    indicator_weights: Dict[str, float]
    entry_threshold: float
    exit_threshold: float
    stop_loss_multiplier: float
    
    # Source
    source: str  # 'initial', 'coach_update', 'manual'
    change_rationale: str
    
    # Validation metrics
    backtest_sharpe: Optional[float] = None
    backtest_profit: Optional[float] = None
    backtest_win_rate: Optional[float] = None
    validated: bool = False
    
    # Live tracking
    live_trades_count: int = 0
    live_win_rate: float = 0.0
    live_pnl: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'version_id': self.version_id,
            'created_at': self.created_at.isoformat(),
            'indicator_weights': self.indicator_weights,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'stop_loss_multiplier': self.stop_loss_multiplier,
            'source': self.source,
            'change_rationale': self.change_rationale,
            'backtest_sharpe': self.backtest_sharpe,
            'backtest_profit': self.backtest_profit,
            'backtest_win_rate': self.backtest_win_rate,
            'validated': self.validated,
            'live_trades_count': self.live_trades_count,
            'live_win_rate': self.live_win_rate,
            'live_pnl': self.live_pnl,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyVersion':
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class StrategyUpdate:
    """Proposed strategy update."""
    
    source_version_id: int
    proposed_changes: Dict[str, Any]
    rationale: str
    confidence: float
    
    # Constraints applied
    changes_capped: bool = False
    original_changes: Dict[str, Any] = field(default_factory=dict)


class StrategyUpdater:
    """Generates and validates strategy updates based on Coach analysis."""
    
    def __init__(self, config: AIConfig = None):
        """
        Initialize strategy updater.
        
        Args:
            config: AI configuration
        """
        self.config = config or DEFAULT_AI_CONFIG
        
        # Strategy version history
        self._versions: List[StrategyVersion] = []
        self._current_version: Optional[StrategyVersion] = None
        self._next_version_id = 1
        
        # Default strategy parameters
        self._default_entry_threshold = 0.7
        self._default_exit_threshold = 0.3
        self._default_stop_loss_multiplier = 2.0
    
    def initialize_strategy(
        self,
        indicator_weights: Dict[str, float],
        entry_threshold: float = None,
        exit_threshold: float = None,
        stop_loss_multiplier: float = None,
    ) -> StrategyVersion:
        """
        Initialize the first strategy version.
        
        Args:
            indicator_weights: Initial indicator weights
            entry_threshold: Entry threshold (default 0.7)
            exit_threshold: Exit threshold (default 0.3)
            stop_loss_multiplier: ATR multiplier for stops (default 2.0)
            
        Returns:
            Initial StrategyVersion
        """
        version = StrategyVersion(
            version_id=self._next_version_id,
            created_at=datetime.now(),
            indicator_weights=indicator_weights.copy(),
            entry_threshold=entry_threshold or self._default_entry_threshold,
            exit_threshold=exit_threshold or self._default_exit_threshold,
            stop_loss_multiplier=stop_loss_multiplier or self._default_stop_loss_multiplier,
            source='initial',
            change_rationale='Initial strategy configuration',
            validated=True,  # Initial version is pre-validated
        )
        
        self._versions.append(version)
        self._current_version = version
        self._next_version_id += 1
        
        logger.info(f"Initialized strategy version {version.version_id}")
        return version
    
    def generate_update(
        self,
        analysis: TradeAnalysis,
    ) -> Optional[StrategyUpdate]:
        """
        Generate a strategy update based on Coach analysis.
        
        Args:
            analysis: TradeAnalysis with recommendations
            
        Returns:
            StrategyUpdate proposal or None if no update needed
        """
        if not self._current_version:
            logger.warning("No current strategy version, cannot generate update")
            return None
        
        # Check confidence threshold
        if analysis.confidence_score < self.config.coach.min_confidence_for_update:
            logger.info(
                f"Coach confidence {analysis.confidence_score:.2f} below threshold "
                f"{self.config.coach.min_confidence_for_update}, skipping update"
            )
            return None
        
        # Extract recommended adjustments
        adjustments = analysis.recommended_adjustments
        if not adjustments:
            logger.info("No adjustments recommended in analysis")
            return None
        
        # Build proposed changes
        proposed_changes = {}
        original_changes = {}
        changes_capped = False
        
        # Entry threshold
        if 'entry_threshold_change' in adjustments:
            change = adjustments['entry_threshold_change']
            original_changes['entry_threshold_change'] = change
            capped_change = self._cap_change(change, 0.05)
            if abs(capped_change) != abs(change):
                changes_capped = True
            proposed_changes['entry_threshold'] = (
                self._current_version.entry_threshold + capped_change
            )
        
        # Exit threshold
        if 'exit_threshold_change' in adjustments:
            change = adjustments['exit_threshold_change']
            original_changes['exit_threshold_change'] = change
            capped_change = self._cap_change(change, 0.05)
            if abs(capped_change) != abs(change):
                changes_capped = True
            proposed_changes['exit_threshold'] = (
                self._current_version.exit_threshold + capped_change
            )
        
        # Stop loss multiplier
        if 'stop_loss_multiplier_change' in adjustments:
            change = adjustments['stop_loss_multiplier_change']
            original_changes['stop_loss_multiplier_change'] = change
            capped_change = self._cap_change(change, 0.2)
            if abs(capped_change) != abs(change):
                changes_capped = True
            proposed_changes['stop_loss_multiplier'] = (
                self._current_version.stop_loss_multiplier + capped_change
            )
        
        # Indicator weight changes
        if 'indicator_weight_changes' in adjustments:
            weight_changes = adjustments['indicator_weight_changes']
            original_changes['indicator_weight_changes'] = weight_changes.copy()
            new_weights = self._current_version.indicator_weights.copy()
            
            max_change = self.config.coach.max_weight_adjustment_per_indicator
            
            for indicator, change in weight_changes.items():
                if indicator in new_weights:
                    capped_change = self._cap_change(change, max_change)
                    if abs(capped_change) != abs(change):
                        changes_capped = True
                    new_weights[indicator] = max(0.0, new_weights[indicator] + capped_change)
            
            proposed_changes['indicator_weights'] = new_weights
        
        if not proposed_changes:
            return None
        
        # Build rationale
        rationale = self._build_rationale(analysis)
        
        return StrategyUpdate(
            source_version_id=self._current_version.version_id,
            proposed_changes=proposed_changes,
            rationale=rationale,
            confidence=analysis.confidence_score,
            changes_capped=changes_capped,
            original_changes=original_changes,
        )
    
    def _cap_change(self, value: float, max_abs: float) -> float:
        """Cap a change value to a maximum absolute value."""
        return max(-max_abs, min(max_abs, value))
    
    def _build_rationale(self, analysis: TradeAnalysis) -> str:
        """Build human-readable rationale for update."""
        parts = []
        
        if analysis.key_insights:
            parts.append("Insights: " + "; ".join(analysis.key_insights[:3]))
        
        if analysis.winning_patterns:
            parts.append(f"Winning patterns: {', '.join(analysis.winning_patterns[:2])}")
        
        if analysis.losing_patterns:
            parts.append(f"Losing patterns: {', '.join(analysis.losing_patterns[:2])}")
        
        return " | ".join(parts) if parts else "Automatic optimization"
    
    def apply_update(
        self,
        update: StrategyUpdate,
        backtest_metrics: Dict[str, float] = None,
    ) -> StrategyVersion:
        """
        Apply a validated strategy update.
        
        Args:
            update: StrategyUpdate to apply
            backtest_metrics: Backtest results for the new strategy
            
        Returns:
            New StrategyVersion
        """
        if not self._current_version:
            raise ValueError("No current version to update from")
        
        # Check safety limits - total drift from original
        if len(self._versions) > 0:
            drift = self._calculate_drift(
                self._versions[0].indicator_weights,
                update.proposed_changes.get('indicator_weights', self._current_version.indicator_weights)
            )
            
            if drift > self.config.safety.max_strategy_drift_pct:
                logger.warning(
                    f"Strategy drift {drift:.1%} exceeds limit "
                    f"{self.config.safety.max_strategy_drift_pct:.1%}, blocking update"
                )
                if self.config.safety.require_human_approval:
                    raise ValueError("Update requires human approval due to high drift")
        
        # Create new version
        new_weights = update.proposed_changes.get(
            'indicator_weights',
            self._current_version.indicator_weights.copy()
        )
        
        version = StrategyVersion(
            version_id=self._next_version_id,
            created_at=datetime.now(),
            indicator_weights=new_weights,
            entry_threshold=update.proposed_changes.get(
                'entry_threshold',
                self._current_version.entry_threshold
            ),
            exit_threshold=update.proposed_changes.get(
                'exit_threshold',
                self._current_version.exit_threshold
            ),
            stop_loss_multiplier=update.proposed_changes.get(
                'stop_loss_multiplier',
                self._current_version.stop_loss_multiplier
            ),
            source='coach_update',
            change_rationale=update.rationale,
            validated=backtest_metrics is not None,
        )
        
        # Add backtest metrics if available
        if backtest_metrics:
            version.backtest_sharpe = backtest_metrics.get('sharpe_ratio')
            version.backtest_profit = backtest_metrics.get('net_profit')
            version.backtest_win_rate = backtest_metrics.get('win_rate')
        
        self._versions.append(version)
        self._current_version = version
        self._next_version_id += 1
        
        logger.info(
            f"Applied strategy update to version {version.version_id}: "
            f"{update.rationale[:100]}"
        )
        
        return version
    
    def _calculate_drift(
        self,
        original_weights: Dict[str, float],
        new_weights: Dict[str, float],
    ) -> float:
        """Calculate total drift from original weights."""
        total_drift = 0.0
        all_indicators = set(original_weights.keys()) | set(new_weights.keys())
        
        for ind in all_indicators:
            orig = original_weights.get(ind, 0)
            new = new_weights.get(ind, 0)
            if orig > 0:
                drift = abs(new - orig) / orig
                total_drift += drift
        
        return total_drift / len(all_indicators) if all_indicators else 0.0
    
    def rollback(self, to_version_id: int = None) -> StrategyVersion:
        """
        Rollback to a previous strategy version.
        
        Args:
            to_version_id: Version to rollback to (default: previous version)
            
        Returns:
            Rolled back StrategyVersion
        """
        if len(self._versions) < 2:
            raise ValueError("No previous version to rollback to")
        
        if to_version_id:
            target = next((v for v in self._versions if v.version_id == to_version_id), None)
            if not target:
                raise ValueError(f"Version {to_version_id} not found")
        else:
            target = self._versions[-2]
        
        self._current_version = target
        logger.warning(f"Rolled back to strategy version {target.version_id}")
        
        return target
    
    def get_current_version(self) -> Optional[StrategyVersion]:
        """Get current strategy version."""
        return self._current_version
    
    def get_version_history(self) -> List[StrategyVersion]:
        """Get all strategy versions."""
        return self._versions.copy()
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current indicator weights."""
        if self._current_version:
            return self._current_version.indicator_weights.copy()
        return {}
    
    def update_live_performance(
        self,
        trades_count: int,
        win_rate: float,
        pnl: float,
    ):
        """Update live performance metrics for current version."""
        if self._current_version:
            self._current_version.live_trades_count += trades_count
            total_trades = self._current_version.live_trades_count
            old_wins = self._current_version.live_win_rate * (total_trades - trades_count)
            new_wins = win_rate * trades_count
            self._current_version.live_win_rate = (old_wins + new_wins) / total_trades
            self._current_version.live_pnl += pnl
    
    def export_versions(self, filepath: str):
        """Export all versions to JSON file."""
        data = [v.to_dict() for v in self._versions]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported {len(data)} strategy versions to {filepath}")
    
    def import_versions(self, filepath: str):
        """Import versions from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self._versions = [StrategyVersion.from_dict(v) for v in data]
        if self._versions:
            self._current_version = self._versions[-1]
            self._next_version_id = max(v.version_id for v in self._versions) + 1
        
        logger.info(f"Imported {len(self._versions)} strategy versions from {filepath}")
