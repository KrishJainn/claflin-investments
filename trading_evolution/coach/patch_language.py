"""
Strategy Patch Language.

Defines the schema for bounded strategy modifications:
- Weight deltas with caps
- Threshold deltas with caps
- Indicator toggles from whitelist
- Rule toggles from whitelist

Patches are diffs, NOT full strategy rewrites.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any
from enum import Enum
import json


class PatchValidationError(Exception):
    """Raised when a patch fails validation."""
    pass


class MistakeType(Enum):
    """Types of trading mistakes that patches can address."""
    EARLY_ENTRY = "early_entry"
    LATE_ENTRY = "late_entry"
    EARLY_EXIT = "early_exit"
    LATE_EXIT = "late_exit"
    FALSE_SIGNAL = "false_signal"
    MISSED_SIGNAL = "missed_signal"
    OVERTRADING = "overtrading"
    UNDERTRADING = "undertrading"
    POOR_SIZING = "poor_sizing"
    WRONG_REGIME = "wrong_regime"


class MarketRegime(Enum):
    """Market regimes where patches may help."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    ALL = "all"


# ============================================================================
# BOUNDS AND WHITELISTS
# ============================================================================

# Maximum weight change per patch
WEIGHT_DELTA_CAP = 0.15  # Max ±15% change

# Maximum threshold change per patch
THRESHOLD_DELTA_CAP = 0.10  # Max ±10% threshold shift

# Indicators that can be toggled
INDICATOR_WHITELIST = {
    # Trend
    "TEMA_20", "EMA_20", "SMA_50", "SMA_100", "HMA_9",
    # Momentum
    "RSI_14", "RSI_7", "STOCH_14_3", "STOCH_5_3", "TSI_13_25", "CMO_14",
    # Volatility
    "ATR_14", "ATR_20", "NATR_14", "NATR_20", "BBANDS_20_2",
    # Volume
    "OBV", "CMF_20", "CMF_21", "MFI_14", "VWMA_10",
    # Trend Strength
    "ADX_14", "ADX_20", "AROON_14", "AROON_25", "KST",
    # Others
    "SUPERTREND_7_3", "SUPERTREND_10_2", "PSAR", "ZSCORE_20",
}

# Rules that can be toggled
RULE_WHITELIST = {
    "breakout_confirm",      # Require breakout confirmation
    "volume_filter",         # Volume above average filter
    "time_filter",           # Time of day filter
    "trend_filter",          # Trade with trend only
    "volatility_filter",     # Avoid low/high vol periods
    "momentum_confirm",      # Momentum confirmation
    "rsi_divergence",        # RSI divergence check
    "gap_filter",            # Avoid gaps
}


@dataclass
class WeightDelta:
    """A bounded weight change."""
    indicator: str
    delta: float  # -WEIGHT_DELTA_CAP to +WEIGHT_DELTA_CAP
    
    def validate(self):
        """Validate the weight delta."""
        if abs(self.delta) > WEIGHT_DELTA_CAP:
            raise PatchValidationError(
                f"Weight delta {self.delta} exceeds cap ±{WEIGHT_DELTA_CAP}"
            )
        if self.indicator not in INDICATOR_WHITELIST:
            raise PatchValidationError(
                f"Indicator '{self.indicator}' not in whitelist"
            )


@dataclass
class ThresholdDelta:
    """A bounded threshold change."""
    threshold_name: str  # entry_threshold, exit_threshold, etc.
    delta: float  # -THRESHOLD_DELTA_CAP to +THRESHOLD_DELTA_CAP
    
    VALID_THRESHOLDS = {
        "entry_threshold",
        "exit_threshold",
        "stop_loss_atr_mult",
        "take_profit_atr_mult",
    }
    
    def validate(self):
        """Validate the threshold delta."""
        if abs(self.delta) > THRESHOLD_DELTA_CAP:
            raise PatchValidationError(
                f"Threshold delta {self.delta} exceeds cap ±{THRESHOLD_DELTA_CAP}"
            )
        if self.threshold_name not in self.VALID_THRESHOLDS:
            raise PatchValidationError(
                f"Invalid threshold: '{self.threshold_name}'"
            )


@dataclass 
class StrategyPatch:
    """
    A bounded strategy modification.
    
    This is a DIFF, not a full strategy rewrite.
    All changes are capped to prevent drastic modifications.
    """
    
    # Unique ID for this patch
    patch_id: str
    
    # Weight changes (indicator -> delta)
    weights: Dict[str, float] = field(default_factory=dict)
    
    # Threshold changes
    thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Indicators to enable (must be in whitelist)
    enable: List[str] = field(default_factory=list)
    
    # Indicators to disable (must be in whitelist)
    disable: List[str] = field(default_factory=list)
    
    # Rules to enable
    enable_rules: List[str] = field(default_factory=list)
    
    # Rules to disable
    disable_rules: List[str] = field(default_factory=list)
    
    # Metadata
    rationale: str = ""
    mistake_type: Optional[str] = None
    what_could_go_wrong: str = ""
    expected_regime: str = "all"
    confidence: float = 0.5
    
    def validate(self) -> List[str]:
        """
        Validate the patch against all constraints.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate weight deltas
        for indicator, delta in self.weights.items():
            if indicator not in INDICATOR_WHITELIST:
                errors.append(f"Weight: indicator '{indicator}' not in whitelist")
            if abs(delta) > WEIGHT_DELTA_CAP:
                errors.append(f"Weight: delta {delta} for {indicator} exceeds cap ±{WEIGHT_DELTA_CAP}")
        
        # Validate threshold deltas
        valid_thresholds = {"entry_threshold", "exit_threshold", "stop_loss_atr_mult", "take_profit_atr_mult"}
        for threshold, delta in self.thresholds.items():
            if threshold not in valid_thresholds:
                errors.append(f"Threshold: '{threshold}' is invalid")
            if abs(delta) > THRESHOLD_DELTA_CAP:
                errors.append(f"Threshold: delta {delta} exceeds cap ±{THRESHOLD_DELTA_CAP}")
        
        # Validate enable list
        for ind in self.enable:
            if ind not in INDICATOR_WHITELIST:
                errors.append(f"Enable: indicator '{ind}' not in whitelist")
        
        # Validate disable list
        for ind in self.disable:
            if ind not in INDICATOR_WHITELIST:
                errors.append(f"Disable: indicator '{ind}' not in whitelist")
        
        # Check for conflicts
        overlap = set(self.enable) & set(self.disable)
        if overlap:
            errors.append(f"Conflict: indicators in both enable and disable: {overlap}")
        
        # Validate rules
        for rule in self.enable_rules:
            if rule not in RULE_WHITELIST:
                errors.append(f"Rule: '{rule}' not in whitelist")
        
        for rule in self.disable_rules:
            if rule not in RULE_WHITELIST:
                errors.append(f"Rule: '{rule}' not in whitelist")
        
        # Check rule conflicts
        rule_overlap = set(self.enable_rules) & set(self.disable_rules)
        if rule_overlap:
            errors.append(f"Conflict: rules in both enable and disable: {rule_overlap}")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if patch is valid."""
        return len(self.validate()) == 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "patch_id": self.patch_id,
            "weights": self.weights,
            "thresholds": self.thresholds,
            "enable": self.enable,
            "disable": self.disable,
            "enable_rules": self.enable_rules,
            "disable_rules": self.disable_rules,
            "rationale": self.rationale,
            "mistake_type": self.mistake_type,
            "what_could_go_wrong": self.what_could_go_wrong,
            "expected_regime": self.expected_regime,
            "confidence": self.confidence,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyPatch':
        """Create from dictionary with validation."""
        patch = cls(
            patch_id=data.get("patch_id", ""),
            weights=data.get("weights", {}),
            thresholds=data.get("thresholds", {}),
            enable=data.get("enable", []),
            disable=data.get("disable", []),
            enable_rules=data.get("enable_rules", []),
            disable_rules=data.get("disable_rules", []),
            rationale=data.get("rationale", ""),
            mistake_type=data.get("mistake_type"),
            what_could_go_wrong=data.get("what_could_go_wrong", ""),
            expected_regime=data.get("expected_regime", "all"),
            confidence=data.get("confidence", 0.5),
        )
        
        # Fail-fast: validate immediately
        errors = patch.validate()
        if errors:
            raise PatchValidationError(f"Invalid patch: {errors}")
        
        return patch
    
    def apply_to_weights(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply this patch to a set of weights.
        
        Args:
            current_weights: Current weight dictionary
            
        Returns:
            New weights with patch applied
        """
        new_weights = current_weights.copy()
        
        # Apply weight deltas
        for indicator, delta in self.weights.items():
            current = new_weights.get(indicator, 0.0)
            new_weights[indicator] = max(-1.0, min(1.0, current + delta))
        
        # Enable indicators (set small positive weight if currently 0)
        for indicator in self.enable:
            if indicator not in new_weights or abs(new_weights[indicator]) < 0.01:
                new_weights[indicator] = 0.1
        
        # Disable indicators (set to 0)
        for indicator in self.disable:
            new_weights[indicator] = 0.0
        
        return new_weights
    
    def apply_to_thresholds(self, current_thresholds: Dict[str, float]) -> Dict[str, float]:
        """
        Apply threshold changes.
        
        Args:
            current_thresholds: Current threshold values
            
        Returns:
            New thresholds with patch applied
        """
        new_thresholds = current_thresholds.copy()
        
        for name, delta in self.thresholds.items():
            current = new_thresholds.get(name, 0.5)
            new_thresholds[name] = max(0.0, min(1.0, current + delta))
        
        return new_thresholds


def validate_patch_json(json_str: str) -> tuple[bool, List[str]]:
    """
    Validate a patch from JSON string.
    
    Fail-fast guard: no fields outside schema allowed.
    
    Args:
        json_str: JSON string of patch
        
    Returns:
        Tuple of (is_valid, errors)
    """
    ALLOWED_FIELDS = {
        "patch_id", "weights", "thresholds", "enable", "disable",
        "enable_rules", "disable_rules", "rationale", "mistake_type",
        "what_could_go_wrong", "expected_regime", "confidence"
    }
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    
    errors = []
    
    # Check for unknown fields
    unknown = set(data.keys()) - ALLOWED_FIELDS
    if unknown:
        errors.append(f"Unknown fields (fail-fast): {unknown}")
        return False, errors
    
    # Try to create patch (will validate)
    try:
        patch = StrategyPatch.from_dict(data)
        return True, []
    except PatchValidationError as e:
        return False, [str(e)]
