"""
Monthly Supervisor Report Generator.

Generates comprehensive monthly reports:
- Performance vs baseline
- Strategy changes (diffs)
- Experiment ledger (tried/passed/failed)
- News days review
- Recommended next-month backtests

Outputs:
1. Executive summary (human readable markdown)
2. Machine summary (JSON for automation)
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


@dataclass
class PerformanceMetrics:
    """Monthly performance metrics."""
    month: str  # "2024-01"
    
    # Core metrics
    total_pnl: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    var_95: float = 0.0  # Value at Risk
    
    # Comparison
    baseline_pnl: float = 0.0
    pnl_vs_baseline: float = 0.0
    pnl_vs_baseline_pct: float = 0.0
    
    # Additional
    avg_trade_pnl: float = 0.0
    best_day_pnl: float = 0.0
    worst_day_pnl: float = 0.0
    trading_days: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "month": self.month,
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "pnl_vs_baseline": self.pnl_vs_baseline,
            "pnl_vs_baseline_pct": self.pnl_vs_baseline_pct,
        }


@dataclass
class StrategyChange:
    """Record of a strategy change during the month."""
    change_id: str
    timestamp: datetime
    change_type: str  # "weight_update", "threshold_update", "rule_toggle"
    
    # What changed
    before: Dict
    after: Dict
    delta: Dict
    
    # Context
    reason: str
    approved_by: str = "system"
    experiment_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "change_id": self.change_id,
            "timestamp": self.timestamp.isoformat(),
            "change_type": self.change_type,
            "delta": self.delta,
            "reason": self.reason,
            "approved_by": self.approved_by,
        }


@dataclass
class ExperimentRecord:
    """Record of an experiment during the month."""
    experiment_id: str
    patch_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    
    # Result
    status: str  # "passed", "failed", "pending"
    result_summary: str
    
    # Metrics
    baseline_pnl: float = 0.0
    experiment_pnl: float = 0.0
    improvement_pct: float = 0.0
    
    # Deployment
    deployed: bool = False
    deployed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "patch_id": self.patch_id,
            "started_at": self.started_at.isoformat(),
            "status": self.status,
            "result_summary": self.result_summary,
            "improvement_pct": self.improvement_pct,
            "deployed": self.deployed,
        }


@dataclass
class NewsDayReview:
    """Review of a day with significant news impact."""
    date: date
    events: List[str]
    max_severity: str
    
    # Posture taken
    posture: str
    
    # Results
    pnl: float
    trades_blocked: int
    positions_flattened: int
    
    # Assessment
    intervention_effective: bool
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date.isoformat(),
            "events": self.events,
            "max_severity": self.max_severity,
            "posture": self.posture,
            "pnl": self.pnl,
            "trades_blocked": self.trades_blocked,
            "intervention_effective": self.intervention_effective,
        }


@dataclass
class BacktestRecommendation:
    """Recommended backtest for next month."""
    recommendation_id: str
    description: str
    hypothesis: str
    priority: int  # 1-10
    
    # What to test
    test_type: str  # "weight_change", "new_indicator", "threshold_tune", etc.
    parameters: Dict
    
    # Expected outcome
    expected_improvement: str
    risk_assessment: str
    
    def to_dict(self) -> Dict:
        return {
            "recommendation_id": self.recommendation_id,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "priority": self.priority,
            "test_type": self.test_type,
            "expected_improvement": self.expected_improvement,
        }


@dataclass
class MonthlyReport:
    """Complete monthly supervisor report."""
    
    # Identification
    report_id: str
    month: str
    generated_at: datetime
    
    # Performance
    performance: PerformanceMetrics
    
    # Changes
    strategy_changes: List[StrategyChange] = field(default_factory=list)
    
    # Experiments
    experiments: List[ExperimentRecord] = field(default_factory=list)
    experiments_passed: int = 0
    experiments_failed: int = 0
    
    # News intervention
    news_days: List[NewsDayReview] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[BacktestRecommendation] = field(default_factory=list)
    
    # Approval
    requires_approval: bool = False
    approval_items: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to machine-readable JSON."""
        return {
            "report_id": self.report_id,
            "month": self.month,
            "generated_at": self.generated_at.isoformat(),
            "performance": self.performance.to_dict(),
            "strategy_changes": [c.to_dict() for c in self.strategy_changes],
            "experiments": {
                "total": len(self.experiments),
                "passed": self.experiments_passed,
                "failed": self.experiments_failed,
                "records": [e.to_dict() for e in self.experiments],
            },
            "news_days": [n.to_dict() for n in self.news_days],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "approval": {
                "requires_approval": self.requires_approval,
                "items": self.approval_items,
            },
        }
    
    def to_executive_summary(self) -> str:
        """Generate human-readable executive summary."""
        lines = [
            f"# Monthly Trading Report: {self.month}",
            f"",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M')}",
            f"",
            "---",
            "",
            "## Performance Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total P&L | ₹{self.performance.total_pnl:,.0f} |",
            f"| vs Baseline | {'+' if self.performance.pnl_vs_baseline >= 0 else ''}{self.performance.pnl_vs_baseline_pct:.1f}% |",
            f"| Total Trades | {self.performance.total_trades} |",
            f"| Win Rate | {self.performance.win_rate:.1f}% |",
            f"| Sharpe Ratio | {self.performance.sharpe_ratio:.2f} |",
            f"| Max Drawdown | {self.performance.max_drawdown_pct:.1f}% |",
            f"| Trading Days | {self.performance.trading_days} |",
            "",
        ]
        
        # Performance assessment
        if self.performance.pnl_vs_baseline_pct > 10:
            lines.append("✅ **Strong outperformance** vs baseline")
        elif self.performance.pnl_vs_baseline_pct > 0:
            lines.append("✓ Positive performance vs baseline")
        elif self.performance.pnl_vs_baseline_pct > -5:
            lines.append("⚠️ Slight underperformance vs baseline")
        else:
            lines.append("❌ **Significant underperformance** - review required")
        
        lines.extend(["", "---", "", "## Strategy Changes", ""])
        
        if self.strategy_changes:
            for i, change in enumerate(self.strategy_changes, 1):
                lines.append(f"{i}. **{change.change_type}** ({change.timestamp.strftime('%Y-%m-%d')})")
                lines.append(f"   - Reason: {change.reason}")
                lines.append(f"   - Approved by: {change.approved_by}")
        else:
            lines.append("No strategy changes this month.")
        
        lines.extend(["", "---", "", "## Experiment Ledger", ""])
        lines.append(f"| Status | Count |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Passed | {self.experiments_passed} |")
        lines.append(f"| Failed | {self.experiments_failed} |")
        lines.append(f"| Total | {len(self.experiments)} |")
        
        if self.experiments:
            lines.extend(["", "### Key Experiments:", ""])
            for exp in self.experiments[:5]:  # Top 5
                status_icon = "✅" if exp.status == "passed" else "❌" if exp.status == "failed" else "⏳"
                lines.append(f"- {status_icon} **{exp.experiment_id}**: {exp.result_summary}")
                if exp.deployed:
                    lines.append(f"  - *Deployed on {exp.deployed_at.strftime('%Y-%m-%d')}*")
        
        lines.extend(["", "---", "", "## News Days Review", ""])
        
        if self.news_days:
            lines.append(f"| Date | Events | Severity | P&L | Effective |")
            lines.append(f"|------|--------|----------|-----|-----------|")
            for day in self.news_days:
                effective = "✅" if day.intervention_effective else "❌"
                lines.append(f"| {day.date} | {len(day.events)} | {day.max_severity} | ₹{day.pnl:,.0f} | {effective} |")
        else:
            lines.append("No significant news days this month.")
        
        lines.extend(["", "---", "", "## Recommendations for Next Month", ""])
        
        if self.recommendations:
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"### {i}. {rec.description}")
                lines.append(f"- **Priority**: {rec.priority}/10")
                lines.append(f"- **Type**: {rec.test_type}")
                lines.append(f"- **Hypothesis**: {rec.hypothesis}")
                lines.append(f"- **Expected**: {rec.expected_improvement}")
                lines.append(f"- **Risk**: {rec.risk_assessment}")
                lines.append("")
        else:
            lines.append("No specific recommendations for next month.")
        
        if self.requires_approval:
            lines.extend([
                "",
                "---",
                "",
                "## ⚠️ APPROVAL REQUIRED",
                "",
                "The following items require your approval before proceeding:",
                "",
            ])
            for item in self.approval_items:
                lines.append(f"- [ ] {item}")
            lines.extend([
                "",
                "Please review and update the approval gate file.",
            ])
        
        return "\n".join(lines)


class MonthlyReportGenerator:
    """
    Generates monthly supervisor reports.
    
    Outputs:
    - Executive summary (markdown)
    - Machine summary (JSON)
    """
    
    def __init__(
        self,
        output_dir: str = "./monthly_reports",
    ):
        """Initialize generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        month: str,
        performance: PerformanceMetrics,
        strategy_changes: List[StrategyChange] = None,
        experiments: List[ExperimentRecord] = None,
        news_days: List[NewsDayReview] = None,
        recommendations: List[BacktestRecommendation] = None,
    ) -> MonthlyReport:
        """
        Generate monthly report.
        
        Args:
            month: Month string "YYYY-MM"
            performance: Performance metrics
            strategy_changes: List of strategy changes
            experiments: List of experiments
            news_days: List of news day reviews
            recommendations: List of recommendations
            
        Returns:
            MonthlyReport object
        """
        report = MonthlyReport(
            report_id=f"report_{month}",
            month=month,
            generated_at=datetime.now(),
            performance=performance,
            strategy_changes=strategy_changes or [],
            experiments=experiments or [],
            news_days=news_days or [],
            recommendations=recommendations or [],
        )
        
        # Count experiment results
        report.experiments_passed = sum(1 for e in report.experiments if e.status == "passed")
        report.experiments_failed = sum(1 for e in report.experiments if e.status == "failed")
        
        # Determine if approval is required
        self._check_approval_requirements(report)
        
        return report
    
    def _check_approval_requirements(self, report: MonthlyReport):
        """Check if any items require approval."""
        items = []
        
        # Significant underperformance
        if report.performance.pnl_vs_baseline_pct < -10:
            items.append("Review strategy after significant underperformance (-10%+ vs baseline)")
        
        # High drawdown
        if report.performance.max_drawdown_pct > 12:
            items.append(f"Review risk parameters after {report.performance.max_drawdown_pct:.1f}% drawdown")
        
        # Many failed experiments
        if report.experiments_failed > 5:
            items.append(f"Review experiment criteria after {report.experiments_failed} failures")
        
        # Deployed changes
        deployed = [e for e in report.experiments if e.deployed]
        if deployed:
            items.append(f"Confirm {len(deployed)} deployed changes for continued use")
        
        # High priority recommendations
        high_priority = [r for r in report.recommendations if r.priority >= 8]
        if high_priority:
            items.append(f"Review {len(high_priority)} high-priority recommendations")
        
        report.requires_approval = len(items) > 0
        report.approval_items = items
    
    def save_report(self, report: MonthlyReport) -> Dict[str, Path]:
        """
        Save report as both executive summary and machine JSON.
        
        Returns:
            Dict with paths to saved files
        """
        # Executive summary (markdown)
        summary_path = self.output_dir / f"{report.month}_executive_summary.md"
        with open(summary_path, 'w') as f:
            f.write(report.to_executive_summary())
        
        # Machine summary (JSON)
        json_path = self.output_dir / f"{report.month}_machine_summary.json"
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        return {
            "executive_summary": summary_path,
            "machine_summary": json_path,
        }
    
    def generate_and_save(
        self,
        month: str,
        performance: PerformanceMetrics,
        **kwargs,
    ) -> Dict[str, Path]:
        """Generate and save report in one call."""
        report = self.generate(month, performance, **kwargs)
        return self.save_report(report)
