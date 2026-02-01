"""
Strategy Validator - Main Entry Point.

Deterministic validator.score(candidate) -> PASS/FAIL + report.

Combines:
- Walk-forward backtest gate
- Stability gate
- Canary deployment readiness check
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json

from .validation_gates import (
    WalkForwardGate, WalkForwardResult, WindowConfig, WindowMetrics,
    StabilityGate, StabilityConfig, StabilityResult, GateResult,
)
from .canary_deployment import CanaryConfig
from .patch_language import StrategyPatch


class ValidationVerdict(Enum):
    """Final validation verdict."""
    PASS = "PASS"
    FAIL = "FAIL"
    CONDITIONAL = "CONDITIONAL"


@dataclass
class ValidationReport:
    """Complete validation report."""
    
    # Overall
    verdict: ValidationVerdict
    score: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Gate results
    walk_forward_result: Optional[WalkForwardResult] = None
    stability_result: Optional[StabilityResult] = None
    
    # Summary
    gates_passed: List[str] = field(default_factory=list)
    gates_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Recommendation
    can_deploy_canary: bool = False
    canary_recommendation: str = ""
    
    # Human-readable summary
    summary: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "verdict": self.verdict.value,
            "score": self.score,
            "timestamp": self.timestamp.isoformat(),
            "gates_passed": self.gates_passed,
            "gates_failed": self.gates_failed,
            "warnings": self.warnings,
            "can_deploy_canary": self.can_deploy_canary,
            "canary_recommendation": self.canary_recommendation,
            "summary": self.summary,
            "walk_forward": self.walk_forward_result.to_dict() if self.walk_forward_result else None,
            "stability": self.stability_result.to_dict() if self.stability_result else None,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class StrategyValidator:
    """
    Main strategy validator.
    
    Usage:
        validator = StrategyValidator()
        report = validator.score(patch, baseline_metrics, candidate_metrics)
        
        if report.verdict == ValidationVerdict.PASS:
            # Safe to deploy to canary
    
    All validation is deterministic (non-LLM).
    """
    
    def __init__(
        self,
        walk_forward_gate: WalkForwardGate = None,
        stability_gate: StabilityGate = None,
        require_all_gates: bool = False,
    ):
        """
        Initialize validator.
        
        Args:
            walk_forward_gate: Walk-forward backtest gate
            stability_gate: Stability gate
            require_all_gates: If True, all gates must pass
        """
        self.walk_forward_gate = walk_forward_gate or WalkForwardGate()
        self.stability_gate = stability_gate or StabilityGate()
        self.require_all_gates = require_all_gates
    
    def score(
        self,
        patch: StrategyPatch,
        baseline_window_metrics: Dict[str, WindowMetrics],
        candidate_window_metrics: Dict[str, WindowMetrics],
        baseline_daily_pnls: List[float] = None,
        candidate_daily_pnls: List[float] = None,
        baseline_trades_per_day: List[int] = None,
        candidate_trades_per_day: List[int] = None,
    ) -> ValidationReport:
        """
        Score a candidate strategy patch.
        
        Args:
            patch: Strategy patch being validated
            baseline_window_metrics: Baseline metrics per window
            candidate_window_metrics: Candidate metrics per window
            baseline_daily_pnls: Daily P&L for baseline
            candidate_daily_pnls: Daily P&L for candidate
            baseline_trades_per_day: Trades per day for baseline
            candidate_trades_per_day: Trades per day for candidate
            
        Returns:
            ValidationReport with PASS/FAIL verdict and details
        """
        report = ValidationReport(
            verdict=ValidationVerdict.FAIL,
            score=0.0,
        )
        
        gates_total = 0
        gates_passed = 0
        
        # Gate 1: Walk-forward backtest
        if baseline_window_metrics and candidate_window_metrics:
            gates_total += 1
            wf_result = self.walk_forward_gate.validate(
                baseline_window_metrics, candidate_window_metrics
            )
            report.walk_forward_result = wf_result
            
            if wf_result.passed:
                report.gates_passed.append("walk_forward")
                gates_passed += 1
            else:
                report.gates_failed.append("walk_forward")
                for reason in wf_result.failure_reasons:
                    report.warnings.append(f"Walk-forward: {reason}")
        
        # Gate 2: Stability
        if candidate_daily_pnls:
            gates_total += 1
            stability_result = self.stability_gate.validate(
                baseline_daily_pnls or [],
                candidate_daily_pnls,
                baseline_trades_per_day or [],
                candidate_trades_per_day or [],
            )
            report.stability_result = stability_result
            
            if stability_result.passed:
                report.gates_passed.append("stability")
                gates_passed += 1
            else:
                report.gates_failed.append("stability")
                for reason in stability_result.checks_failed:
                    report.warnings.append(f"Stability: {reason}")
        
        # Calculate overall score
        if gates_total > 0:
            report.score = gates_passed / gates_total
        
        # Determine verdict
        if self.require_all_gates:
            if gates_passed == gates_total:
                report.verdict = ValidationVerdict.PASS
        else:
            if gates_passed >= 1:  # At least one gate passed
                if gates_passed == gates_total:
                    report.verdict = ValidationVerdict.PASS
                else:
                    report.verdict = ValidationVerdict.CONDITIONAL
        
        # Canary recommendation
        report.can_deploy_canary = report.verdict in [
            ValidationVerdict.PASS, 
            ValidationVerdict.CONDITIONAL
        ]
        
        if report.can_deploy_canary:
            report.canary_recommendation = self._generate_canary_recommendation(report, patch)
        else:
            report.canary_recommendation = "NOT RECOMMENDED - Failed validation gates"
        
        # Generate summary
        report.summary = self._generate_summary(report, patch)
        
        return report
    
    def _generate_canary_recommendation(
        self,
        report: ValidationReport,
        patch: StrategyPatch,
    ) -> str:
        """Generate canary deployment recommendation."""
        
        if report.verdict == ValidationVerdict.PASS:
            return (
                f"RECOMMENDED: Deploy to canary at 15% size for 3 sessions. "
                f"Patch {patch.patch_id} passed all gates with score {report.score:.2f}."
            )
        elif report.verdict == ValidationVerdict.CONDITIONAL:
            return (
                f"CONDITIONAL: Deploy to canary at 10% size for 5 sessions. "
                f"Patch {patch.patch_id} passed {len(report.gates_passed)}/{len(report.gates_passed) + len(report.gates_failed)} gates. "
                f"Monitor closely: {', '.join(report.gates_failed)}"
            )
        
        return "NOT RECOMMENDED"
    
    def _generate_summary(
        self,
        report: ValidationReport,
        patch: StrategyPatch,
    ) -> str:
        """Generate human-readable summary."""
        
        lines = [
            f"VALIDATION REPORT: {report.verdict.value}",
            f"Patch: {patch.patch_id}",
            f"Score: {report.score:.2f}",
            "",
            f"Gates Passed: {', '.join(report.gates_passed) or 'None'}",
            f"Gates Failed: {', '.join(report.gates_failed) or 'None'}",
        ]
        
        if report.walk_forward_result:
            wf = report.walk_forward_result
            lines.append("")
            lines.append("Walk-Forward Results:")
            for window in wf.windows_passed:
                imp = wf.pnl_improvement.get(window, 0)
                lines.append(f"  ✓ {window}: +{imp:.1f}% PnL")
            for window in wf.windows_failed:
                imp = wf.pnl_improvement.get(window, 0)
                lines.append(f"  ✗ {window}: {imp:.1f}% PnL")
        
        if report.stability_result:
            sr = report.stability_result
            lines.append("")
            lines.append("Stability Checks:")
            for check in sr.checks_passed:
                lines.append(f"  ✓ {check}")
            for check in sr.checks_failed:
                lines.append(f"  ✗ {check}")
        
        if report.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in report.warnings[:5]:  # Limit to 5
                lines.append(f"  • {w}")
        
        lines.append("")
        lines.append(f"Canary: {report.canary_recommendation}")
        
        return "\n".join(lines)


def quick_validate(
    patch: StrategyPatch,
    baseline_pnl: float,
    candidate_pnl: float,
    baseline_sharpe: float,
    candidate_sharpe: float,
    baseline_drawdown: float,
    candidate_drawdown: float,
    trades: int,
) -> Tuple[bool, str]:
    """
    Quick validation without full metrics.
    
    Returns:
        Tuple of (passed, reason)
    """
    if trades < 20:
        return False, f"Insufficient trades: {trades} < 20"
    
    pnl_improvement = candidate_pnl - baseline_pnl
    if baseline_pnl > 0:
        pnl_improvement_pct = (pnl_improvement / baseline_pnl) * 100
    else:
        pnl_improvement_pct = 100 if candidate_pnl > 0 else 0
    
    sharpe_improvement = candidate_sharpe - baseline_sharpe
    dd_change = candidate_drawdown - baseline_drawdown
    
    if pnl_improvement_pct < 2.0:
        return False, f"PnL improvement {pnl_improvement_pct:.1f}% < 2%"
    
    if dd_change > 3.0:
        return False, f"Drawdown increased {dd_change:.1f}% > 3%"
    
    if sharpe_improvement < -0.5:
        return False, f"Sharpe decreased {sharpe_improvement:.2f} < -0.5"
    
    return True, f"PASS: +{pnl_improvement_pct:.1f}% PnL, Sharpe {sharpe_improvement:+.2f}"
