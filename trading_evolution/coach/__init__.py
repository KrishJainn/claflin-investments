"""
Coach Module - Bounded Recommendations + Auto-Validation + News Intervention.

Phase 3:
- StrategyPatch: Diff-based strategy modifications with caps
- CandidateGenerator: Creates testable patches from mistakes
- ExperimentRunner: Tests patches with fail-fast gates

Phase 4:
- WalkForwardGate: Multi-window backtest validation
- StabilityGate: Turnover, drawdown, consistency checks
- CanaryManager: 10-20% deployment with auto-rollback
- StrategyValidator: Deterministic score() -> PASS/FAIL

Phase 5:
- EventClassifier: CRITICAL/HIGH/MEDIUM/LOW with time validity
- PostureManager: FREEZE/CONSERVATIVE/NORMAL modes
- NewsInterventionEngine: Intraday event handling with audit
"""

from .patch_language import (
    StrategyPatch,
    PatchValidationError,
    MistakeType,
    MarketRegime,
    WEIGHT_DELTA_CAP,
    THRESHOLD_DELTA_CAP,
    INDICATOR_WHITELIST,
    RULE_WHITELIST,
    validate_patch_json,
)

from .candidate_generator import (
    CandidateGenerator,
    CandidateExperiment,
    TradeAnalysis,
    MistakePattern,
)

from .experiment_runner import (
    ExperimentRunner,
    ExperimentResult,
    ExperimentGates,
)

from .validation_gates import (
    WalkForwardGate,
    WalkForwardResult,
    WindowConfig,
    WindowMetrics,
    StabilityGate,
    StabilityConfig,
    StabilityResult,
    GateResult,
)

from .canary_deployment import (
    CanaryManager,
    CanaryDeployment,
    CanaryConfig,
    CanarySession,
    DeploymentState,
)

from .validator import (
    StrategyValidator,
    ValidationReport,
    ValidationVerdict,
    quick_validate,
)

from .event_classifier import (
    EventClassifier,
    EventSeverity,
    EventType,
    MarketEvent,
)

from .posture_manager import (
    PostureManager,
    TradingPosture,
    PostureParameters,
    PostureChange,
    get_intervention_action,
    POSTURE_PRESETS,
)

from .intervention_engine import (
    NewsInterventionEngine,
    InterventionRecord,
)

from .monthly_report import (
    MonthlyReportGenerator,
    MonthlyReport,
    PerformanceMetrics,
    StrategyChange,
    ExperimentRecord,
    NewsDayReview,
    BacktestRecommendation,
)

from .approval_gate import (
    ApprovalGate,
    ApprovalGateState,
    ApprovalItem,
)

__all__ = [
    # Patch language (Phase 3)
    'StrategyPatch', 'PatchValidationError', 'MistakeType', 'MarketRegime',
    'WEIGHT_DELTA_CAP', 'THRESHOLD_DELTA_CAP', 'INDICATOR_WHITELIST', 'RULE_WHITELIST',
    'validate_patch_json',
    # Candidate generation (Phase 3)
    'CandidateGenerator', 'CandidateExperiment', 'TradeAnalysis', 'MistakePattern',
    # Experiment running (Phase 3)
    'ExperimentRunner', 'ExperimentResult', 'ExperimentGates',
    # Validation gates (Phase 4)
    'WalkForwardGate', 'WalkForwardResult', 'WindowConfig', 'WindowMetrics',
    'StabilityGate', 'StabilityConfig', 'StabilityResult', 'GateResult',
    # Canary deployment (Phase 4)
    'CanaryManager', 'CanaryDeployment', 'CanaryConfig', 'CanarySession', 'DeploymentState',
    # Validator (Phase 4)
    'StrategyValidator', 'ValidationReport', 'ValidationVerdict', 'quick_validate',
    # Event classification (Phase 5)
    'EventClassifier', 'EventSeverity', 'EventType', 'MarketEvent',
    # Posture management (Phase 5)
    'PostureManager', 'TradingPosture', 'PostureParameters', 'PostureChange',
    'get_intervention_action', 'POSTURE_PRESETS',
    # Intervention engine (Phase 5)
    'NewsInterventionEngine', 'InterventionRecord',
    # Monthly reports (Phase 6)
    'MonthlyReportGenerator', 'MonthlyReport', 'PerformanceMetrics',
    'StrategyChange', 'ExperimentRecord', 'NewsDayReview', 'BacktestRecommendation',
    # Approval gate (Phase 6)
    'ApprovalGate', 'ApprovalGateState', 'ApprovalItem',
]


