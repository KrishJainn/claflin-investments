"""
Canary Deployment for Strategy Changes.

Safely tests new strategies at reduced size:
- Trades at 10-20% of normal position size
- Runs for 3-5 sessions before full deployment
- Auto-rollback if underperforms baseline

Non-LLM deterministic deployment logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from enum import Enum
import json
from pathlib import Path


class DeploymentState(Enum):
    """State of canary deployment."""
    PENDING = "pending"
    ACTIVE = "active"
    PASSED = "passed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""
    
    # Sizing
    canary_size_pct: float = 0.15  # 15% of normal position size
    
    # Duration
    min_sessions: int = 3
    max_sessions: int = 5
    
    # Pass criteria
    min_canary_trades: int = 10
    max_underperformance_pct: float = 10.0  # Can underperform by max 10%
    max_canary_drawdown_pct: float = 5.0  # Max DD during canary
    
    # Auto-rollback triggers
    consecutive_losses_rollback: int = 4
    single_loss_rollback_pct: float = 3.0  # Rollback if single loss > 3% of capital


@dataclass
class CanarySession:
    """Results from a single canary session."""
    session_date: date
    session_num: int
    
    # Trading results
    trades: int = 0
    pnl: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    
    # Comparison to baseline
    baseline_pnl: float = 0.0
    relative_performance: float = 0.0  # canary - baseline
    
    # Flags
    triggered_rollback: bool = False
    rollback_reason: Optional[str] = None


@dataclass
class CanaryDeployment:
    """A complete canary deployment."""
    
    # Identification
    deployment_id: str
    patch_id: str
    started_at: datetime
    
    # State
    state: DeploymentState = DeploymentState.PENDING
    
    # Configuration
    config: CanaryConfig = field(default_factory=CanaryConfig)
    
    # Sessions
    sessions: List[CanarySession] = field(default_factory=list)
    
    # Results
    total_canary_pnl: float = 0.0
    total_baseline_pnl: float = 0.0
    total_trades: int = 0
    avg_relative_performance: float = 0.0
    
    # Final decision
    completed_at: Optional[datetime] = None
    final_decision: Optional[str] = None
    decision_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "deployment_id": self.deployment_id,
            "patch_id": self.patch_id,
            "state": self.state.value,
            "started_at": self.started_at.isoformat(),
            "sessions_completed": len(self.sessions),
            "total_canary_pnl": self.total_canary_pnl,
            "total_baseline_pnl": self.total_baseline_pnl,
            "avg_relative_performance": self.avg_relative_performance,
            "final_decision": self.final_decision,
            "decision_reason": self.decision_reason,
        }


class CanaryManager:
    """
    Manages canary deployments for strategy changes.
    
    Process:
    1. Start canary with reduced position size
    2. Run for 3-5 sessions, comparing to baseline
    3. Auto-promote if outperforms, auto-rollback if underperforms
    
    All decisions are deterministic (non-LLM).
    """
    
    def __init__(
        self,
        config: CanaryConfig = None,
        save_dir: str = "./canary_deployments",
    ):
        """
        Initialize manager.
        
        Args:
            config: Canary configuration
            save_dir: Directory to save deployment records
        """
        self.config = config or CanaryConfig()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Active deployments
        self._active: Dict[str, CanaryDeployment] = {}
        
        # Completed deployments
        self._completed: List[CanaryDeployment] = []
    
    def start_deployment(
        self,
        deployment_id: str,
        patch_id: str,
    ) -> CanaryDeployment:
        """
        Start a new canary deployment.
        
        Args:
            deployment_id: Unique deployment ID
            patch_id: Patch being deployed
            
        Returns:
            New CanaryDeployment object
        """
        deployment = CanaryDeployment(
            deployment_id=deployment_id,
            patch_id=patch_id,
            started_at=datetime.now(),
            state=DeploymentState.ACTIVE,
            config=self.config,
        )
        
        self._active[deployment_id] = deployment
        
        return deployment
    
    def record_session(
        self,
        deployment_id: str,
        session_date: date,
        trades: int,
        pnl: float,
        win_rate: float,
        max_drawdown: float,
        baseline_pnl: float,
    ) -> Tuple[CanarySession, Optional[str]]:
        """
        Record results from a canary session.
        
        Args:
            deployment_id: Deployment ID
            session_date: Date of session
            trades: Number of trades
            pnl: P&L from canary
            win_rate: Win rate
            max_drawdown: Max drawdown during session
            baseline_pnl: P&L from baseline for comparison
            
        Returns:
            Tuple of (session, action) where action is None, "rollback", or "promote"
        """
        if deployment_id not in self._active:
            raise ValueError(f"No active deployment: {deployment_id}")
        
        deployment = self._active[deployment_id]
        session_num = len(deployment.sessions) + 1
        
        # Create session record
        session = CanarySession(
            session_date=session_date,
            session_num=session_num,
            trades=trades,
            pnl=pnl,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            baseline_pnl=baseline_pnl,
            relative_performance=pnl - baseline_pnl,
        )
        
        # Check for rollback triggers
        rollback_triggered, rollback_reason = self._check_rollback_triggers(
            deployment, session
        )
        
        if rollback_triggered:
            session.triggered_rollback = True
            session.rollback_reason = rollback_reason
            deployment.sessions.append(session)
            self._rollback(deployment, rollback_reason)
            return session, "rollback"
        
        # Add session
        deployment.sessions.append(session)
        
        # Update totals
        deployment.total_canary_pnl += pnl
        deployment.total_baseline_pnl += baseline_pnl
        deployment.total_trades += trades
        
        # Check if ready for final decision
        if session_num >= self.config.min_sessions:
            action = self._evaluate_deployment(deployment)
            return session, action
        
        return session, None
    
    def _check_rollback_triggers(
        self,
        deployment: CanaryDeployment,
        session: CanarySession,
    ) -> Tuple[bool, Optional[str]]:
        """Check if any rollback triggers are hit."""
        
        # Single large loss
        if session.pnl < 0:
            # Approximate capital as 100k for % calculation
            loss_pct = abs(session.pnl) / 100000 * 100
            if loss_pct > self.config.single_loss_rollback_pct:
                return True, f"Single session loss {loss_pct:.1f}% > {self.config.single_loss_rollback_pct}%"
        
        # Max drawdown exceeded
        if session.max_drawdown > self.config.max_canary_drawdown_pct:
            return True, f"Drawdown {session.max_drawdown:.1f}% > {self.config.max_canary_drawdown_pct}%"
        
        # Consecutive losses
        recent_sessions = deployment.sessions[-self.config.consecutive_losses_rollback:]
        if len(recent_sessions) >= self.config.consecutive_losses_rollback:
            if all(s.pnl < 0 for s in recent_sessions):
                return True, f"{self.config.consecutive_losses_rollback} consecutive losing sessions"
        
        return False, None
    
    def _evaluate_deployment(self, deployment: CanaryDeployment) -> Optional[str]:
        """Evaluate if deployment should be promoted or rolled back."""
        
        sessions = deployment.sessions
        
        # Need minimum trades
        if deployment.total_trades < self.config.min_canary_trades:
            if len(sessions) >= self.config.max_sessions:
                self._rollback(deployment, "Insufficient trades after max sessions")
                return "rollback"
            return None  # Continue waiting
        
        # Calculate average relative performance
        relative_perfs = [s.relative_performance for s in sessions]
        avg_relative = sum(relative_perfs) / len(relative_perfs)
        deployment.avg_relative_performance = avg_relative
        
        # Check underperformance limit
        if deployment.total_baseline_pnl > 0:
            underperformance_pct = (
                (deployment.total_baseline_pnl - deployment.total_canary_pnl) 
                / deployment.total_baseline_pnl * 100
            )
        else:
            underperformance_pct = 0 if deployment.total_canary_pnl >= 0 else 100
        
        # Decision logic
        if underperformance_pct > self.config.max_underperformance_pct:
            self._rollback(
                deployment, 
                f"Underperformed by {underperformance_pct:.1f}% > {self.config.max_underperformance_pct}%"
            )
            return "rollback"
        
        if len(sessions) >= self.config.max_sessions:
            # Must decide - promote if not too bad
            if deployment.total_canary_pnl >= 0 or underperformance_pct <= 5:
                self._promote(deployment)
                return "promote"
            else:
                self._rollback(deployment, "Did not meet promotion criteria after max sessions")
                return "rollback"
        
        # Continue canary if outperforming or close
        if avg_relative >= 0:
            # Doing well, can promote early after min sessions
            if len(sessions) >= self.config.min_sessions:
                self._promote(deployment)
                return "promote"
        
        return None  # Continue canary
    
    def _promote(self, deployment: CanaryDeployment):
        """Promote canary to full deployment."""
        deployment.state = DeploymentState.PASSED
        deployment.completed_at = datetime.now()
        deployment.final_decision = "PROMOTE"
        deployment.decision_reason = (
            f"Passed after {len(deployment.sessions)} sessions. "
            f"Total PnL: {deployment.total_canary_pnl:,.0f}, "
            f"Avg relative: {deployment.avg_relative_performance:,.0f}"
        )
        
        self._finalize(deployment)
    
    def _rollback(self, deployment: CanaryDeployment, reason: str):
        """Rollback canary deployment."""
        deployment.state = DeploymentState.ROLLED_BACK
        deployment.completed_at = datetime.now()
        deployment.final_decision = "ROLLBACK"
        deployment.decision_reason = reason
        
        self._finalize(deployment)
    
    def _finalize(self, deployment: CanaryDeployment):
        """Finalize a deployment."""
        if deployment.deployment_id in self._active:
            del self._active[deployment.deployment_id]
        
        self._completed.append(deployment)
        self._save_deployment(deployment)
    
    def _save_deployment(self, deployment: CanaryDeployment):
        """Save deployment record to disk."""
        filepath = self.save_dir / f"{deployment.deployment_id}.json"
        
        data = {
            **deployment.to_dict(),
            "sessions": [
                {
                    "session_num": s.session_num,
                    "date": s.session_date.isoformat(),
                    "trades": s.trades,
                    "pnl": s.pnl,
                    "baseline_pnl": s.baseline_pnl,
                    "relative_performance": s.relative_performance,
                    "triggered_rollback": s.triggered_rollback,
                    "rollback_reason": s.rollback_reason,
                }
                for s in deployment.sessions
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_active_deployments(self) -> List[CanaryDeployment]:
        """Get all active canary deployments."""
        return list(self._active.values())
    
    def get_canary_position_size(self, normal_size: float) -> float:
        """Get position size for canary trading."""
        return normal_size * self.config.canary_size_pct
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict]:
        """Get status of a deployment."""
        if deployment_id in self._active:
            return self._active[deployment_id].to_dict()
        
        for d in self._completed:
            if d.deployment_id == deployment_id:
                return d.to_dict()
        
        return None
