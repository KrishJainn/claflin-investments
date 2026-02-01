"""
Posture Manager for News Interventions.

Manages trading posture based on events:
- FREEZE: No entries, flatten existing (for CRITICAL events)
- CONSERVATIVE: Lower size, fewer trades, tighter stops
- NORMAL: Standard trading parameters

All actions are deterministic and whitelisted.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path

from .event_classifier import EventSeverity, MarketEvent


class TradingPosture(Enum):
    """Trading posture levels (whitelisted actions)."""
    FREEZE = "FREEZE"           # No entries, may flatten
    CONSERVATIVE = "CONSERVATIVE"  # Reduced risk
    NORMAL = "NORMAL"           # Standard parameters


@dataclass
class PostureParameters:
    """
    Trading parameters for a posture.
    
    All values are bounded to prevent extreme changes.
    """
    
    # Position sizing (as multiplier of normal)
    position_size_mult: float = 1.0  # 0.0 to 1.0
    
    # Trade frequency
    max_trades_per_day: int = 20
    
    # Risk parameters
    stop_loss_mult: float = 1.0  # Multiplier for stop distance
    take_profit_mult: float = 1.0
    
    # Entry restrictions
    allow_new_entries: bool = True
    allow_longs: bool = True
    allow_shorts: bool = True
    
    # Exit behavior
    flatten_existing: bool = False
    
    # Bounds for validation
    MIN_SIZE_MULT = 0.0
    MAX_SIZE_MULT = 1.0
    MIN_STOP_MULT = 0.5
    MAX_STOP_MULT = 2.0
    
    def validate(self) -> List[str]:
        """Validate parameters are within bounds."""
        errors = []
        
        if not (self.MIN_SIZE_MULT <= self.position_size_mult <= self.MAX_SIZE_MULT):
            errors.append(f"position_size_mult {self.position_size_mult} out of bounds [{self.MIN_SIZE_MULT}, {self.MAX_SIZE_MULT}]")
        
        if not (self.MIN_STOP_MULT <= self.stop_loss_mult <= self.MAX_STOP_MULT):
            errors.append(f"stop_loss_mult {self.stop_loss_mult} out of bounds [{self.MIN_STOP_MULT}, {self.MAX_STOP_MULT}]")
        
        if self.max_trades_per_day < 0:
            errors.append(f"max_trades_per_day must be >= 0")
        
        return errors
    
    def is_valid(self) -> bool:
        return len(self.validate()) == 0
    
    def to_dict(self) -> Dict:
        return {
            "position_size_mult": self.position_size_mult,
            "max_trades_per_day": self.max_trades_per_day,
            "stop_loss_mult": self.stop_loss_mult,
            "take_profit_mult": self.take_profit_mult,
            "allow_new_entries": self.allow_new_entries,
            "allow_longs": self.allow_longs,
            "allow_shorts": self.allow_shorts,
            "flatten_existing": self.flatten_existing,
        }


# Pre-defined posture parameters (whitelisted)
POSTURE_PRESETS: Dict[TradingPosture, PostureParameters] = {
    TradingPosture.FREEZE: PostureParameters(
        position_size_mult=0.0,
        max_trades_per_day=0,
        allow_new_entries=False,
        allow_longs=False,
        allow_shorts=False,
        flatten_existing=True,  # Flatten on critical events
    ),
    TradingPosture.CONSERVATIVE: PostureParameters(
        position_size_mult=0.5,  # 50% of normal size
        max_trades_per_day=5,    # Reduced trades
        stop_loss_mult=0.7,      # Tighter stops
        take_profit_mult=1.2,    # Slightly wider TP
        allow_new_entries=True,
        allow_longs=True,
        allow_shorts=False,      # No shorts during uncertainty
        flatten_existing=False,
    ),
    TradingPosture.NORMAL: PostureParameters(
        position_size_mult=1.0,
        max_trades_per_day=20,
        stop_loss_mult=1.0,
        take_profit_mult=1.0,
        allow_new_entries=True,
        allow_longs=True,
        allow_shorts=True,
        flatten_existing=False,
    ),
}


@dataclass
class PostureChange:
    """Record of a posture change."""
    timestamp: datetime
    previous_posture: TradingPosture
    new_posture: TradingPosture
    trigger_event: Optional[str] = None
    trigger_severity: Optional[str] = None
    reason: str = ""
    parameters: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "previous_posture": self.previous_posture.value,
            "new_posture": self.new_posture.value,
            "trigger_event": self.trigger_event,
            "trigger_severity": self.trigger_severity,
            "reason": self.reason,
            "parameters": self.parameters,
        }


class PostureManager:
    """
    Manages trading posture based on market events.
    
    Features:
    - Deterministic posture selection based on event severity
    - Whitelisted actions only (FREEZE, CONSERVATIVE, NORMAL)
    - Bounded parameters
    - Full audit trail
    """
    
    def __init__(
        self,
        presets: Dict[TradingPosture, PostureParameters] = None,
        audit_dir: str = "./posture_audit",
    ):
        """
        Initialize manager.
        
        Args:
            presets: Custom posture presets
            audit_dir: Directory for audit logs
        """
        self.presets = presets or POSTURE_PRESETS.copy()
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Current state
        self._current_posture = TradingPosture.NORMAL
        self._current_params = self.presets[TradingPosture.NORMAL]
        self._active_events: List[MarketEvent] = []
        
        # Audit trail
        self._changes: List[PostureChange] = []
    
    def get_current_posture(self) -> Tuple[TradingPosture, PostureParameters]:
        """Get current posture and parameters."""
        return self._current_posture, self._current_params
    
    def evaluate_posture(
        self,
        active_events: List[MarketEvent],
        current_time: datetime,
    ) -> Tuple[TradingPosture, PostureParameters, Optional[str]]:
        """
        Evaluate what posture should be based on active events.
        
        Deterministic mapping:
        - CRITICAL event -> FREEZE
        - HIGH event -> CONSERVATIVE
        - MEDIUM/LOW/NORMAL -> NORMAL
        
        Args:
            active_events: Currently active market events
            current_time: Current time
            
        Returns:
            Tuple of (posture, parameters, reason)
        """
        if not active_events:
            return TradingPosture.NORMAL, self.presets[TradingPosture.NORMAL], None
        
        # Find highest severity
        max_severity = EventSeverity.NORMAL
        trigger_event = None
        
        for event in active_events:
            if self._severity_priority(event.severity) < self._severity_priority(max_severity):
                max_severity = event.severity
                trigger_event = event
        
        # Map severity to posture (deterministic)
        if max_severity == EventSeverity.CRITICAL:
            posture = TradingPosture.FREEZE
            reason = f"CRITICAL event: {trigger_event.title}"
        elif max_severity == EventSeverity.HIGH:
            posture = TradingPosture.CONSERVATIVE
            reason = f"HIGH severity event: {trigger_event.title}"
        else:
            posture = TradingPosture.NORMAL
            reason = None
        
        return posture, self.presets[posture], reason
    
    def update_posture(
        self,
        active_events: List[MarketEvent],
        current_time: datetime,
    ) -> Optional[PostureChange]:
        """
        Update posture based on active events.
        
        Args:
            active_events: Currently active events
            current_time: Current time
            
        Returns:
            PostureChange if posture changed, None otherwise
        """
        new_posture, new_params, reason = self.evaluate_posture(
            active_events, current_time
        )
        
        if new_posture == self._current_posture:
            return None
        
        # Record change
        change = PostureChange(
            timestamp=current_time,
            previous_posture=self._current_posture,
            new_posture=new_posture,
            trigger_event=active_events[0].event_id if active_events else None,
            trigger_severity=active_events[0].severity.value if active_events else None,
            reason=reason or "Event cleared",
            parameters=new_params.to_dict(),
        )
        
        self._current_posture = new_posture
        self._current_params = new_params
        self._active_events = active_events
        self._changes.append(change)
        
        return change
    
    def can_enter_trade(
        self,
        direction: str,  # "LONG" or "SHORT"
        current_time: datetime = None,
    ) -> Tuple[bool, str]:
        """
        Check if new trade entry is allowed.
        
        Args:
            direction: Trade direction
            current_time: Current time (for logging)
            
        Returns:
            Tuple of (allowed, reason)
        """
        params = self._current_params
        
        if not params.allow_new_entries:
            return False, f"Entries blocked in {self._current_posture.value} mode"
        
        if direction == "LONG" and not params.allow_longs:
            return False, "Long entries blocked"
        
        if direction == "SHORT" and not params.allow_shorts:
            return False, "Short entries blocked"
        
        return True, f"Allowed in {self._current_posture.value} mode"
    
    def get_adjusted_position_size(self, normal_size: float) -> float:
        """Get position size adjusted for current posture."""
        return normal_size * self._current_params.position_size_mult
    
    def get_adjusted_stop_loss(self, normal_stop_atr: float) -> float:
        """Get stop loss distance adjusted for current posture."""
        return normal_stop_atr * self._current_params.stop_loss_mult
    
    def should_flatten(self) -> bool:
        """Check if current posture requires flattening."""
        return self._current_params.flatten_existing
    
    def get_max_trades_today(self) -> int:
        """Get maximum trades allowed today."""
        return self._current_params.max_trades_per_day
    
    def get_change_history(self) -> List[PostureChange]:
        """Get full history of posture changes."""
        return self._changes.copy()
    
    def save_audit_log(self, filename: str = None):
        """Save audit log to file."""
        fname = filename or f"posture_audit_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = self.audit_dir / fname
        
        data = {
            "saved_at": datetime.now().isoformat(),
            "current_posture": self._current_posture.value,
            "changes": [c.to_dict() for c in self._changes],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def _severity_priority(self, severity: EventSeverity) -> int:
        """Get priority order (lower = more severe)."""
        return {
            EventSeverity.CRITICAL: 0,
            EventSeverity.HIGH: 1,
            EventSeverity.MEDIUM: 2,
            EventSeverity.LOW: 3,
            EventSeverity.NORMAL: 4,
        }.get(severity, 5)


# Convenience function for deterministic intervention
def get_intervention_action(
    event_severity: EventSeverity,
) -> Tuple[TradingPosture, str]:
    """
    Get deterministic intervention action for event severity.
    
    Whitelisted mapping only:
    - CRITICAL -> FREEZE
    - HIGH -> CONSERVATIVE  
    - MEDIUM/LOW/NORMAL -> NORMAL
    
    Args:
        event_severity: Severity of active event
        
    Returns:
        Tuple of (posture, description)
    """
    mapping = {
        EventSeverity.CRITICAL: (
            TradingPosture.FREEZE,
            "Freeze all trading - no entries, flatten existing"
        ),
        EventSeverity.HIGH: (
            TradingPosture.CONSERVATIVE,
            "Conservative mode - 50% size, no shorts, tighter stops"
        ),
        EventSeverity.MEDIUM: (
            TradingPosture.NORMAL,
            "Normal trading with monitoring"
        ),
        EventSeverity.LOW: (
            TradingPosture.NORMAL,
            "Normal trading"
        ),
        EventSeverity.NORMAL: (
            TradingPosture.NORMAL,
            "Normal trading"
        ),
    }
    
    return mapping.get(event_severity, (TradingPosture.NORMAL, "Normal trading"))
