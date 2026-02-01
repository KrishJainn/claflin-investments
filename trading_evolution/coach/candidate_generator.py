"""
Candidate Generator for Coach v2.

Generates small, testable strategy patches based on:
- Trade analysis (mistakes detected)
- Market regime observations
- Indicator performance data

Each candidate includes:
- Rationale linked to mistake type
- "What could go wrong"
- Expected regime where it helps
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import uuid
import random

from .patch_language import (
    StrategyPatch,
    MistakeType,
    MarketRegime,
    WEIGHT_DELTA_CAP,
    THRESHOLD_DELTA_CAP,
    INDICATOR_WHITELIST,
    RULE_WHITELIST,
)


@dataclass
class TradeAnalysis:
    """Analysis of a trade for mistake detection."""
    trade_id: str
    symbol: str
    pnl: float
    entry_si: float
    exit_si: float
    holding_time_mins: int
    entry_hour: int
    volatility_regime: str  # "high", "low", "normal"
    trend_regime: str  # "up", "down", "ranging"
    
    # Detected issues
    early_entry: bool = False
    late_entry: bool = False
    early_exit: bool = False
    late_exit: bool = False
    whipsaw: bool = False
    news_shock: bool = False


@dataclass
class MistakePattern:
    """A pattern of mistakes detected across trades."""
    mistake_type: MistakeType
    count: int
    total_pnl_impact: float
    trade_ids: List[str]
    
    # Context
    common_hour: Optional[int] = None
    common_regime: Optional[str] = None
    common_indicators: List[str] = field(default_factory=list)


@dataclass
class CandidateExperiment:
    """
    A candidate strategy modification to test.
    
    Includes full context for decision making.
    """
    patch: StrategyPatch
    
    # Experiment metadata
    priority: int = 5  # 1-10, higher = more important
    estimated_improvement: float = 0.0  # Estimated PnL improvement
    backtest_required: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "patch": self.patch.to_dict(),
            "priority": self.priority,
            "estimated_improvement": self.estimated_improvement,
            "backtest_required": self.backtest_required,
        }


class CandidateGenerator:
    """
    Generates bounded strategy modification candidates.
    
    All generated patches are:
    - Within weight/threshold caps
    - Only using whitelisted indicators/rules
    - Include full rationale and risk assessment
    """
    
    def __init__(
        self,
        current_weights: Dict[str, float],
        current_thresholds: Dict[str, float],
    ):
        """
        Initialize generator.
        
        Args:
            current_weights: Current strategy weights
            current_thresholds: Current threshold values
        """
        self.current_weights = current_weights
        self.current_thresholds = current_thresholds
        
        # Tracking
        self._generated_patches: List[StrategyPatch] = []
    
    def generate_from_mistakes(
        self,
        mistakes: List[MistakePattern],
    ) -> List[CandidateExperiment]:
        """
        Generate candidates based on detected mistakes.
        
        Args:
            mistakes: List of detected mistake patterns
            
        Returns:
            List of candidate experiments
        """
        candidates = []
        
        for mistake in mistakes:
            # Generate patches for this mistake type
            patches = self._patches_for_mistake(mistake)
            
            for patch in patches:
                candidate = CandidateExperiment(
                    patch=patch,
                    priority=self._calculate_priority(mistake, patch),
                    estimated_improvement=mistake.total_pnl_impact * 0.3,  # Conservative estimate
                )
                candidates.append(candidate)
        
        return candidates
    
    def _patches_for_mistake(self, mistake: MistakePattern) -> List[StrategyPatch]:
        """Generate patches that could address a specific mistake type."""
        patches = []
        
        if mistake.mistake_type == MistakeType.EARLY_ENTRY:
            patches.extend(self._generate_early_entry_patches(mistake))
        
        elif mistake.mistake_type == MistakeType.LATE_ENTRY:
            patches.extend(self._generate_late_entry_patches(mistake))
        
        elif mistake.mistake_type == MistakeType.EARLY_EXIT:
            patches.extend(self._generate_early_exit_patches(mistake))
        
        elif mistake.mistake_type == MistakeType.LATE_EXIT:
            patches.extend(self._generate_late_exit_patches(mistake))
        
        elif mistake.mistake_type == MistakeType.FALSE_SIGNAL:
            patches.extend(self._generate_false_signal_patches(mistake))
        
        elif mistake.mistake_type == MistakeType.OVERTRADING:
            patches.extend(self._generate_overtrading_patches(mistake))
        
        return patches
    
    def _generate_early_entry_patches(self, mistake: MistakePattern) -> List[StrategyPatch]:
        """Generate patches for early entry mistakes."""
        patches = []
        
        # Option 1: Raise entry threshold
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            thresholds={"entry_threshold": 0.05},  # Raise by 5%
            rationale=f"Raise entry threshold to filter out weak signals. "
                      f"Detected {mistake.count} early entry mistakes causing ₹{mistake.total_pnl_impact:,.0f} loss.",
            mistake_type=MistakeType.EARLY_ENTRY.value,
            what_could_go_wrong="May miss some valid entry signals, reducing trade count",
            expected_regime=mistake.common_regime or "all",
            confidence=0.7,
        ))
        
        # Option 2: Enable momentum confirmation
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            enable_rules=["momentum_confirm"],
            rationale=f"Require momentum confirmation before entry to avoid premature signals. "
                      f"Based on {mistake.count} early entries.",
            mistake_type=MistakeType.EARLY_ENTRY.value,
            what_could_go_wrong="May add lag to entries, missing some moves",
            expected_regime="trending_up",
            confidence=0.6,
        ))
        
        # Option 3: Increase trend indicator weights
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            weights={"ADX_14": 0.10, "AROON_14": 0.08},
            rationale=f"Increase trend confirmation indicators to wait for stronger trends. "
                      f"Addresses {mistake.count} early entry mistakes.",
            mistake_type=MistakeType.EARLY_ENTRY.value,
            what_could_go_wrong="May reduce performance in ranging markets",
            expected_regime="trending_up",
            confidence=0.65,
        ))
        
        return patches
    
    def _generate_late_entry_patches(self, mistake: MistakePattern) -> List[StrategyPatch]:
        """Generate patches for late entry mistakes."""
        patches = []
        
        # Lower entry threshold
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            thresholds={"entry_threshold": -0.05},  # Lower by 5%
            rationale=f"Lower entry threshold to catch signals earlier. "
                      f"Detected {mistake.count} late entries missing ₹{abs(mistake.total_pnl_impact):,.0f}.",
            mistake_type=MistakeType.LATE_ENTRY.value,
            what_could_go_wrong="May increase false signals and whipsaws",
            expected_regime=mistake.common_regime or "trending_up",
            confidence=0.6,
        ))
        
        # Increase momentum indicator weights
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            weights={"RSI_7": 0.10, "STOCH_5_3": 0.08},
            rationale=f"Increase fast momentum indicators to detect moves earlier. "
                      f"Based on {mistake.count} late entries.",
            mistake_type=MistakeType.LATE_ENTRY.value,
            what_could_go_wrong="Fast indicators are more noisy, may increase false signals",
            expected_regime="high_volatility",
            confidence=0.55,
        ))
        
        return patches
    
    def _generate_early_exit_patches(self, mistake: MistakePattern) -> List[StrategyPatch]:
        """Generate patches for early exit mistakes (leaving money on table)."""
        patches = []
        
        # Lower exit threshold
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            thresholds={"exit_threshold": -0.05},
            rationale=f"Lower exit threshold to hold positions longer. "
                      f"Detected {mistake.count} early exits leaving ₹{abs(mistake.total_pnl_impact):,.0f} on table.",
            mistake_type=MistakeType.EARLY_EXIT.value,
            what_could_go_wrong="May give back profits on reversals",
            expected_regime="trending_up",
            confidence=0.65,
        ))
        
        # Enable trailing stop instead
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            thresholds={"take_profit_atr_mult": 0.5},  # Add/increase take profit
            rationale=f"Add take profit to capture more gains before SI-based exit. "
                      f"Based on {mistake.count} early exits.",
            mistake_type=MistakeType.EARLY_EXIT.value,
            what_could_go_wrong="Fixed take profit may cut big winners short",
            expected_regime="trending_up",
            confidence=0.6,
        ))
        
        return patches
    
    def _generate_late_exit_patches(self, mistake: MistakePattern) -> List[StrategyPatch]:
        """Generate patches for late exit mistakes (giving back gains)."""
        patches = []
        
        # Raise exit threshold
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            thresholds={"exit_threshold": 0.05},
            rationale=f"Raise exit threshold to exit positions earlier. "
                      f"Detected {mistake.count} late exits giving back ₹{abs(mistake.total_pnl_impact):,.0f}.",
            mistake_type=MistakeType.LATE_EXIT.value,
            what_could_go_wrong="May exit too early on pullbacks in strong trends",
            expected_regime="ranging",
            confidence=0.7,
        ))
        
        # Tighten stop loss
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            thresholds={"stop_loss_atr_mult": -0.3},
            rationale=f"Tighten stop loss to prevent large drawdowns. "
                      f"Based on {mistake.count} late exits.",
            mistake_type=MistakeType.LATE_EXIT.value,
            what_could_go_wrong="Tighter stops may get stopped out on normal volatility",
            expected_regime="high_volatility",
            confidence=0.6,
        ))
        
        return patches
    
    def _generate_false_signal_patches(self, mistake: MistakePattern) -> List[StrategyPatch]:
        """Generate patches for false signal mistakes."""
        patches = []
        
        # Enable volume filter
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            enable_rules=["volume_filter"],
            rationale=f"Enable volume filter to confirm signals. "
                      f"Detected {mistake.count} false signals.",
            mistake_type=MistakeType.FALSE_SIGNAL.value,
            what_could_go_wrong="May filter out valid signals in low-volume periods",
            expected_regime="all",
            confidence=0.7,
        ))
        
        # Increase confirmation indicators
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            weights={"CMF_20": 0.10, "OBV": 0.08},
            rationale=f"Increase volume-based indicator weights for confirmation. "
                      f"Based on {mistake.count} false signals.",
            mistake_type=MistakeType.FALSE_SIGNAL.value,
            what_could_go_wrong="Volume indicators can lag in fast-moving markets",
            expected_regime="all",
            confidence=0.65,
        ))
        
        # Enable trend filter
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            enable_rules=["trend_filter"],
            rationale=f"Only trade in direction of major trend. "
                      f"Addresses {mistake.count} counter-trend false signals.",
            mistake_type=MistakeType.FALSE_SIGNAL.value,
            what_could_go_wrong="Will miss reversals and ranging market opportunities",
            expected_regime="trending_up",
            confidence=0.6,
        ))
        
        return patches
    
    def _generate_overtrading_patches(self, mistake: MistakePattern) -> List[StrategyPatch]:
        """Generate patches for overtrading."""
        patches = []
        
        # Enable time filter
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            enable_rules=["time_filter"],
            rationale=f"Add time filter to avoid noisy open/close periods. "
                      f"Detected overtrading pattern with {mistake.count} excess trades.",
            mistake_type=MistakeType.OVERTRADING.value,
            what_could_go_wrong="May miss valid opportunities during filtered times",
            expected_regime="all",
            confidence=0.7,
        ))
        
        # Raise both thresholds
        patches.append(StrategyPatch(
            patch_id=self._make_id(),
            thresholds={"entry_threshold": 0.08, "exit_threshold": 0.05},
            rationale=f"Raise entry and exit thresholds to reduce trade frequency. "
                      f"Based on overtrading pattern causing ₹{abs(mistake.total_pnl_impact):,.0f} loss.",
            mistake_type=MistakeType.OVERTRADING.value,
            what_could_go_wrong="May miss some valid setups",
            expected_regime="ranging",
            confidence=0.65,
        ))
        
        return patches
    
    def generate_regime_based(
        self,
        current_regime: MarketRegime,
        regime_performance: Dict[str, float],
    ) -> List[CandidateExperiment]:
        """
        Generate candidates based on regime analysis.
        
        Args:
            current_regime: Current market regime
            regime_performance: P&L by regime
            
        Returns:
            List of candidate experiments
        """
        candidates = []
        
        # Find worst performing regime
        worst_regime = min(regime_performance.items(), key=lambda x: x[1])
        
        if worst_regime[1] < 0:
            # Generate patches for worst regime
            patches = self._patches_for_regime(worst_regime[0])
            for patch in patches:
                candidates.append(CandidateExperiment(
                    patch=patch,
                    priority=7,
                    estimated_improvement=abs(worst_regime[1]) * 0.2,
                ))
        
        return candidates
    
    def _patches_for_regime(self, regime: str) -> List[StrategyPatch]:
        """Generate patches to improve performance in a specific regime."""
        patches = []
        
        if regime == "high_volatility":
            patches.append(StrategyPatch(
                patch_id=self._make_id(),
                thresholds={"stop_loss_atr_mult": 0.5},  # Widen stops in high vol
                weights={"ATR_14": 0.10, "NATR_14": 0.08},
                rationale="Widen stops and increase volatility indicator weights for high volatility regime",
                what_could_go_wrong="Wider stops mean larger losses when wrong",
                expected_regime="high_volatility",
                confidence=0.6,
            ))
        
        elif regime == "ranging":
            patches.append(StrategyPatch(
                patch_id=self._make_id(),
                enable_rules=["volatility_filter"],
                weights={"BBANDS_20_2": 0.10, "RSI_14": 0.08},
                rationale="Enable volatility filter and mean-reversion indicators for ranging markets",
                what_could_go_wrong="May miss breakout when range ends",
                expected_regime="ranging",
                confidence=0.65,
            ))
        
        elif regime == "trending_down":
            patches.append(StrategyPatch(
                patch_id=self._make_id(),
                disable=["SUPERTREND_7_3"],
                weights={"ADX_20": 0.10},
                rationale="Reduce trend-following in downtrends, increase trend strength filter",
                what_could_go_wrong="May miss short opportunities",
                expected_regime="trending_down",
                confidence=0.5,
            ))
        
        return patches
    
    def _calculate_priority(self, mistake: MistakePattern, patch: StrategyPatch) -> int:
        """Calculate priority for a candidate."""
        base = 5
        
        # Higher priority for bigger P&L impact
        if abs(mistake.total_pnl_impact) > 10000:
            base += 2
        elif abs(mistake.total_pnl_impact) > 5000:
            base += 1
        
        # Higher priority for more frequent mistakes
        if mistake.count >= 10:
            base += 2
        elif mistake.count >= 5:
            base += 1
        
        # Adjust by confidence
        base = int(base * patch.confidence)
        
        return min(10, max(1, base))
    
    def _make_id(self) -> str:
        """Generate unique patch ID."""
        return f"patch_{uuid.uuid4().hex[:8]}"
    
    def get_all_candidates(
        self,
        mistakes: List[MistakePattern] = None,
        regime_performance: Dict[str, float] = None,
    ) -> List[CandidateExperiment]:
        """
        Generate all candidates from available data.
        
        Args:
            mistakes: List of detected mistakes
            regime_performance: P&L by regime
            
        Returns:
            All candidates, sorted by priority
        """
        candidates = []
        
        if mistakes:
            candidates.extend(self.generate_from_mistakes(mistakes))
        
        if regime_performance:
            candidates.extend(self.generate_regime_based(
                MarketRegime.ALL, regime_performance
            ))
        
        # Sort by priority
        candidates.sort(key=lambda x: x.priority, reverse=True)
        
        return candidates
