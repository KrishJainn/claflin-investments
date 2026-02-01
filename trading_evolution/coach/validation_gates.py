"""
Walk-Forward Backtest Gate.

Tests candidates across multiple time windows:
- Recent (30 days)
- Medium (90 days)  
- Long (180 days)

Must beat frozen baseline on risk-adjusted metrics.
Non-LLM deterministic validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from enum import Enum
import numpy as np


class GateResult(Enum):
    """Result of a gate check."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIP = "SKIP"


@dataclass
class WindowConfig:
    """Configuration for a backtest window."""
    name: str
    days: int
    weight: float  # Importance weight for final score
    min_trades: int
    
    @classmethod
    def recent(cls) -> 'WindowConfig':
        return cls(name="recent", days=30, weight=0.4, min_trades=10)
    
    @classmethod
    def medium(cls) -> 'WindowConfig':
        return cls(name="medium", days=90, weight=0.35, min_trades=25)
    
    @classmethod
    def long(cls) -> 'WindowConfig':
        return cls(name="long", days=180, weight=0.25, min_trades=40)


@dataclass
class WindowMetrics:
    """Metrics from a single backtest window."""
    window_name: str
    start_date: date
    end_date: date
    
    # Core metrics
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    volatility: float = 0.0
    
    # Additional
    avg_trade_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    
    def risk_adjusted_return(self) -> float:
        """Calculate risk-adjusted return (Sharpe-like)."""
        if self.volatility == 0:
            return 0.0
        return self.total_pnl / (self.volatility * np.sqrt(self.total_trades) + 1e-10)


@dataclass
class WalkForwardResult:
    """Result of walk-forward validation."""
    
    # Overall
    passed: bool
    overall_score: float  # 0.0 to 1.0
    
    # Window results
    baseline_metrics: Dict[str, WindowMetrics] = field(default_factory=dict)
    candidate_metrics: Dict[str, WindowMetrics] = field(default_factory=dict)
    
    # Comparison
    pnl_improvement: Dict[str, float] = field(default_factory=dict)
    sharpe_improvement: Dict[str, float] = field(default_factory=dict)
    drawdown_change: Dict[str, float] = field(default_factory=dict)
    
    # Gate details
    windows_passed: List[str] = field(default_factory=list)
    windows_failed: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "windows_passed": self.windows_passed,
            "windows_failed": self.windows_failed,
            "failure_reasons": self.failure_reasons,
            "pnl_improvement": self.pnl_improvement,
            "sharpe_improvement": self.sharpe_improvement,
        }


class WalkForwardGate:
    """
    Walk-forward backtest validation gate.
    
    Tests candidate across multiple time windows and compares
    to a frozen baseline. Must beat baseline on risk-adjusted metrics.
    """
    
    def __init__(
        self,
        windows: List[WindowConfig] = None,
        min_improvement_pct: float = 2.0,
        max_drawdown_increase_pct: float = 3.0,
        require_all_windows: bool = False,
        min_windows_passing: int = 2,
    ):
        """
        Initialize gate.
        
        Args:
            windows: Window configurations to test
            min_improvement_pct: Minimum P&L improvement required
            max_drawdown_increase_pct: Max allowed drawdown increase
            require_all_windows: If True, all windows must pass
            min_windows_passing: Minimum windows that must pass
        """
        self.windows = windows or [
            WindowConfig.recent(),
            WindowConfig.medium(),
            WindowConfig.long(),
        ]
        self.min_improvement_pct = min_improvement_pct
        self.max_drawdown_increase_pct = max_drawdown_increase_pct
        self.require_all_windows = require_all_windows
        self.min_windows_passing = min_windows_passing
    
    def validate(
        self,
        baseline_metrics: Dict[str, WindowMetrics],
        candidate_metrics: Dict[str, WindowMetrics],
    ) -> WalkForwardResult:
        """
        Validate candidate against baseline across all windows.
        
        Args:
            baseline_metrics: Metrics from baseline strategy per window
            candidate_metrics: Metrics from candidate strategy per window
            
        Returns:
            WalkForwardResult with pass/fail and details
        """
        result = WalkForwardResult(
            passed=False,
            overall_score=0.0,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
        )
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for window in self.windows:
            if window.name not in baseline_metrics or window.name not in candidate_metrics:
                result.failure_reasons.append(f"Missing metrics for window: {window.name}")
                continue
            
            baseline = baseline_metrics[window.name]
            candidate = candidate_metrics[window.name]
            
            # Calculate improvements
            pnl_imp = self._calc_improvement(baseline.total_pnl, candidate.total_pnl)
            sharpe_imp = candidate.sharpe_ratio - baseline.sharpe_ratio
            dd_change = candidate.max_drawdown_pct - baseline.max_drawdown_pct
            
            result.pnl_improvement[window.name] = pnl_imp
            result.sharpe_improvement[window.name] = sharpe_imp
            result.drawdown_change[window.name] = dd_change
            
            # Check window
            window_passed, reason = self._check_window(
                window, baseline, candidate, pnl_imp, sharpe_imp, dd_change
            )
            
            if window_passed:
                result.windows_passed.append(window.name)
                weighted_score += window.weight * 1.0
            else:
                result.windows_failed.append(window.name)
                result.failure_reasons.append(f"{window.name}: {reason}")
            
            total_weight += window.weight
        
        # Calculate overall score
        if total_weight > 0:
            result.overall_score = weighted_score / total_weight
        
        # Determine pass/fail
        if self.require_all_windows:
            result.passed = len(result.windows_failed) == 0
        else:
            result.passed = len(result.windows_passed) >= self.min_windows_passing
        
        return result
    
    def _check_window(
        self,
        window: WindowConfig,
        baseline: WindowMetrics,
        candidate: WindowMetrics,
        pnl_imp: float,
        sharpe_imp: float,
        dd_change: float,
    ) -> Tuple[bool, str]:
        """Check if candidate passes for a specific window."""
        
        # Check minimum trades
        if candidate.total_trades < window.min_trades:
            return False, f"Insufficient trades: {candidate.total_trades} < {window.min_trades}"
        
        # Check P&L improvement
        if pnl_imp < self.min_improvement_pct:
            return False, f"PnL improvement below threshold: {pnl_imp:.1f}% < {self.min_improvement_pct}%"
        
        # Check drawdown increase
        if dd_change > self.max_drawdown_increase_pct:
            return False, f"Drawdown increased too much: {dd_change:.1f}% > {self.max_drawdown_increase_pct}%"
        
        # Check Sharpe improvement (soft requirement)
        if sharpe_imp < -0.5:  # Allow small Sharpe decrease if P&L is better
            return False, f"Sharpe ratio decreased significantly: {sharpe_imp:.2f}"
        
        return True, ""
    
    def _calc_improvement(self, baseline: float, candidate: float) -> float:
        """Calculate percentage improvement."""
        if baseline == 0:
            return 100.0 if candidate > 0 else 0.0
        return ((candidate - baseline) / abs(baseline)) * 100


@dataclass
class StabilityConfig:
    """Configuration for stability gate."""
    
    # Turnover limits
    max_daily_turnover: int = 15  # Max trades per day
    max_turnover_change_pct: float = 50.0  # Can't change turnover by more than 50%
    
    # Drawdown limits
    max_drawdown_pct: float = 15.0
    max_drawdown_increase_pct: float = 3.0
    
    # Consistency requirements
    min_profitable_days_pct: float = 45.0  # At least 45% of days profitable
    max_consecutive_losses: int = 5
    
    # "Performance doesn't come from 1 day"
    single_day_pnl_cap_pct: float = 30.0  # No single day > 30% of total PnL


@dataclass
class StabilityResult:
    """Result of stability validation."""
    
    passed: bool
    score: float  # 0.0 to 1.0
    
    # Individual checks
    turnover_check: GateResult = GateResult.SKIP
    drawdown_check: GateResult = GateResult.SKIP
    consistency_check: GateResult = GateResult.SKIP
    concentration_check: GateResult = GateResult.SKIP
    
    # Details
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "turnover_check": self.turnover_check.value,
            "drawdown_check": self.drawdown_check.value,
            "consistency_check": self.consistency_check.value,
            "concentration_check": self.concentration_check.value,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
        }


class StabilityGate:
    """
    Stability validation gate.
    
    Ensures candidate strategy is stable and doesn't rely on
    lucky outliers. Checks turnover, drawdown, and consistency.
    """
    
    def __init__(self, config: StabilityConfig = None):
        """Initialize with configuration."""
        self.config = config or StabilityConfig()
    
    def validate(
        self,
        baseline_daily_pnls: List[float],
        candidate_daily_pnls: List[float],
        baseline_trades_per_day: List[int],
        candidate_trades_per_day: List[int],
    ) -> StabilityResult:
        """
        Validate stability of candidate.
        
        Args:
            baseline_daily_pnls: Daily P&L for baseline
            candidate_daily_pnls: Daily P&L for candidate
            baseline_trades_per_day: Trades per day for baseline
            candidate_trades_per_day: Trades per day for candidate
            
        Returns:
            StabilityResult with pass/fail and details
        """
        result = StabilityResult(passed=False, score=0.0)
        
        checks_total = 4
        checks_passed = 0
        
        # Check 1: Turnover
        turnover_ok, turnover_msg = self._check_turnover(
            baseline_trades_per_day, candidate_trades_per_day
        )
        if turnover_ok:
            result.turnover_check = GateResult.PASS
            result.checks_passed.append("turnover")
            checks_passed += 1
        else:
            result.turnover_check = GateResult.FAIL
            result.checks_failed.append(f"turnover: {turnover_msg}")
        
        # Check 2: Drawdown
        dd_ok, dd_msg = self._check_drawdown(candidate_daily_pnls)
        if dd_ok:
            result.drawdown_check = GateResult.PASS
            result.checks_passed.append("drawdown")
            checks_passed += 1
        else:
            result.drawdown_check = GateResult.FAIL
            result.checks_failed.append(f"drawdown: {dd_msg}")
        
        # Check 3: Consistency
        cons_ok, cons_msg = self._check_consistency(candidate_daily_pnls)
        if cons_ok:
            result.consistency_check = GateResult.PASS
            result.checks_passed.append("consistency")
            checks_passed += 1
        else:
            result.consistency_check = GateResult.FAIL
            result.checks_failed.append(f"consistency: {cons_msg}")
        
        # Check 4: Concentration ("performance doesn't come from 1 day")
        conc_ok, conc_msg = self._check_concentration(candidate_daily_pnls)
        if conc_ok:
            result.concentration_check = GateResult.PASS
            result.checks_passed.append("concentration")
            checks_passed += 1
        else:
            result.concentration_check = GateResult.FAIL
            result.checks_failed.append(f"concentration: {conc_msg}")
        
        # Calculate score and pass/fail
        result.score = checks_passed / checks_total
        result.passed = checks_passed >= 3  # Need at least 3/4 to pass
        
        return result
    
    def _check_turnover(
        self,
        baseline_trades: List[int],
        candidate_trades: List[int],
    ) -> Tuple[bool, str]:
        """Check turnover limits."""
        
        if not candidate_trades:
            return True, ""
        
        avg_baseline = np.mean(baseline_trades) if baseline_trades else 0
        avg_candidate = np.mean(candidate_trades)
        max_candidate = max(candidate_trades)
        
        # Check max daily turnover
        if max_candidate > self.config.max_daily_turnover:
            return False, f"Max daily trades {max_candidate} > {self.config.max_daily_turnover}"
        
        # Check turnover change
        if avg_baseline > 0:
            change_pct = abs((avg_candidate - avg_baseline) / avg_baseline) * 100
            if change_pct > self.config.max_turnover_change_pct:
                return False, f"Turnover changed {change_pct:.0f}% > {self.config.max_turnover_change_pct}%"
        
        return True, ""
    
    def _check_drawdown(self, daily_pnls: List[float]) -> Tuple[bool, str]:
        """Check drawdown limits."""
        
        if not daily_pnls:
            return True, ""
        
        cumulative = np.cumsum(daily_pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        
        if len(drawdowns) == 0:
            return True, ""
        
        max_dd = np.max(drawdowns)
        total_pnl = sum(daily_pnls)
        
        if total_pnl > 0:
            max_dd_pct = (max_dd / (sum(abs(p) for p in daily_pnls) / len(daily_pnls))) * 100 / len(daily_pnls)
        else:
            max_dd_pct = 100  # All losses
        
        if max_dd_pct > self.config.max_drawdown_pct:
            return False, f"Max drawdown {max_dd_pct:.1f}% > {self.config.max_drawdown_pct}%"
        
        return True, ""
    
    def _check_consistency(self, daily_pnls: List[float]) -> Tuple[bool, str]:
        """Check consistency requirements."""
        
        if not daily_pnls:
            return True, ""
        
        profitable_days = sum(1 for p in daily_pnls if p > 0)
        profitable_pct = (profitable_days / len(daily_pnls)) * 100
        
        if profitable_pct < self.config.min_profitable_days_pct:
            return False, f"Only {profitable_pct:.0f}% profitable days < {self.config.min_profitable_days_pct}%"
        
        # Check consecutive losses
        max_consecutive = 0
        current_consecutive = 0
        for p in daily_pnls:
            if p < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        if max_consecutive > self.config.max_consecutive_losses:
            return False, f"{max_consecutive} consecutive losing days > {self.config.max_consecutive_losses}"
        
        return True, ""
    
    def _check_concentration(self, daily_pnls: List[float]) -> Tuple[bool, str]:
        """Check that performance doesn't come from 1 day."""
        
        if not daily_pnls:
            return True, ""
        
        total_pnl = sum(daily_pnls)
        if total_pnl <= 0:
            return True, ""  # No concentration issue if losing
        
        max_day_pnl = max(daily_pnls)
        concentration = (max_day_pnl / total_pnl) * 100
        
        if concentration > self.config.single_day_pnl_cap_pct:
            return False, f"Single day = {concentration:.0f}% of total PnL > {self.config.single_day_pnl_cap_pct}%"
        
        return True, ""
