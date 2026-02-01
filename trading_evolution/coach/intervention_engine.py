"""
News Intervention Engine.

Combines event classification and posture management for
safe handling of "big day" events.

Features:
- Event classification (CRITICAL/HIGH/MEDIUM/LOW)
- Time validity (no future info)
- Deterministic posture changes
- Full audit logging
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

from .event_classifier import (
    EventClassifier, EventSeverity, EventType, MarketEvent, INDIAN_MARKET_EVENTS
)
from .posture_manager import (
    PostureManager, TradingPosture, PostureParameters, PostureChange,
    get_intervention_action, POSTURE_PRESETS
)


@dataclass
class InterventionRecord:
    """Record of an intervention action."""
    
    # Timing
    timestamp: datetime
    
    # Trigger
    trigger_event_id: Optional[str]
    trigger_event_title: str
    trigger_severity: EventSeverity
    
    # Action taken
    action: TradingPosture
    action_description: str
    
    # Parameters applied
    parameters: Dict
    
    # Impact
    trades_blocked: int = 0
    positions_flattened: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "trigger_event_id": self.trigger_event_id,
            "trigger_event_title": self.trigger_event_title,
            "trigger_severity": self.trigger_severity.value,
            "action": self.action.value,
            "action_description": self.action_description,
            "parameters": self.parameters,
            "trades_blocked": self.trades_blocked,
            "positions_flattened": self.positions_flattened,
        }


class NewsInterventionEngine:
    """
    Main engine for news-based trading interventions.
    
    Usage:
        engine = NewsInterventionEngine()
        
        # Register known events
        engine.register_event(...)
        
        # Check posture before trading
        posture, params = engine.get_current_posture(current_time)
        if posture == TradingPosture.FREEZE:
            # Don't trade
        elif posture == TradingPosture.CONSERVATIVE:
            # Use reduced size
    
    All interventions are deterministic and whitelisted.
    """
    
    def __init__(
        self,
        audit_dir: str = "./intervention_audit",
    ):
        """Initialize engine."""
        self.classifier = EventClassifier()
        self.posture_manager = PostureManager(audit_dir=audit_dir)
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Intervention records
        self._interventions: List[InterventionRecord] = []
        
        # Stats
        self._trades_blocked = 0
        self._positions_flattened = 0
    
    def register_event(
        self,
        event_type: EventType,
        event_time: datetime,
        title: str,
        description: str = "",
        affected_symbols: List[str] = None,
        custom_severity: EventSeverity = None,
    ) -> MarketEvent:
        """
        Register a market event.
        
        Args:
            event_type: Type of event
            event_time: When event occurs
            title: Event title
            description: Event description
            affected_symbols: Affected symbols
            custom_severity: Override severity
            
        Returns:
            Classified MarketEvent
        """
        return self.classifier.classify(
            event_type=event_type,
            event_time=event_time,
            title=title,
            description=description,
            affected_symbols=affected_symbols,
            custom_severity=custom_severity,
        )
    
    def register_critical_event(
        self,
        event_time: datetime,
        title: str,
        description: str = "",
    ) -> MarketEvent:
        """Quick registration of a critical event."""
        return self.register_event(
            event_type=EventType.MAJOR_ANNOUNCEMENT,
            event_time=event_time,
            title=title,
            description=description,
            custom_severity=EventSeverity.CRITICAL,
        )
    
    def check_and_update_posture(
        self,
        current_time: datetime,
        symbol: str = None,
    ) -> Tuple[TradingPosture, PostureParameters, Optional[InterventionRecord]]:
        """
        Check for active events and update posture.
        
        Args:
            current_time: Current time
            symbol: Optional symbol to filter events
            
        Returns:
            Tuple of (posture, parameters, intervention_record)
        """
        # Get active events (respects time validity)
        active_events = self.classifier.get_active_events(current_time, symbol)
        
        # Update posture
        change = self.posture_manager.update_posture(active_events, current_time)
        
        # Get current state
        posture, params = self.posture_manager.get_current_posture()
        
        # Record intervention if posture changed
        intervention = None
        if change is not None:
            intervention = self._record_intervention(change, active_events)
        
        return posture, params, intervention
    
    def get_current_posture(
        self,
        current_time: datetime = None,
        symbol: str = None,
    ) -> Tuple[TradingPosture, PostureParameters]:
        """
        Get current trading posture.
        
        Args:
            current_time: Current time (default: now)
            symbol: Optional symbol filter
            
        Returns:
            Tuple of (posture, parameters)
        """
        now = current_time or datetime.now()
        posture, params, _ = self.check_and_update_posture(now, symbol)
        return posture, params
    
    def can_trade(
        self,
        direction: str,
        symbol: str = None,
        current_time: datetime = None,
    ) -> Tuple[bool, str]:
        """
        Check if trading is allowed.
        
        Args:
            direction: "LONG" or "SHORT"
            symbol: Symbol to trade
            current_time: Current time
            
        Returns:
            Tuple of (allowed, reason)
        """
        now = current_time or datetime.now()
        
        # Update posture first
        self.check_and_update_posture(now, symbol)
        
        # Check with posture manager
        allowed, reason = self.posture_manager.can_enter_trade(direction, now)
        
        if not allowed:
            self._trades_blocked += 1
        
        return allowed, reason
    
    def get_position_size(
        self,
        normal_size: float,
        current_time: datetime = None,
    ) -> float:
        """Get position size adjusted for current posture."""
        if current_time:
            self.check_and_update_posture(current_time)
        return self.posture_manager.get_adjusted_position_size(normal_size)
    
    def get_stop_loss(
        self,
        normal_stop_atr: float,
        current_time: datetime = None,
    ) -> float:
        """Get stop loss adjusted for current posture."""
        if current_time:
            self.check_and_update_posture(current_time)
        return self.posture_manager.get_adjusted_stop_loss(normal_stop_atr)
    
    def should_flatten_all(self, current_time: datetime = None) -> bool:
        """Check if all positions should be flattened."""
        if current_time:
            self.check_and_update_posture(current_time)
        return self.posture_manager.should_flatten()
    
    def record_flatten(self, count: int):
        """Record positions flattened due to intervention."""
        self._positions_flattened += count
        if self._interventions:
            self._interventions[-1].positions_flattened += count
    
    def get_active_events(
        self,
        current_time: datetime = None,
    ) -> List[MarketEvent]:
        """Get all active events."""
        now = current_time or datetime.now()
        return self.classifier.get_active_events(now)
    
    def get_max_severity(
        self,
        current_time: datetime = None,
        symbol: str = None,
    ) -> EventSeverity:
        """Get maximum severity of active events."""
        now = current_time or datetime.now()
        return self.classifier.get_max_severity(now, symbol)
    
    def get_intervention_history(self) -> List[InterventionRecord]:
        """Get full intervention history."""
        return self._interventions.copy()
    
    def get_stats(self) -> Dict:
        """Get intervention statistics."""
        return {
            "total_interventions": len(self._interventions),
            "trades_blocked": self._trades_blocked,
            "positions_flattened": self._positions_flattened,
            "current_posture": self.posture_manager.get_current_posture()[0].value,
        }
    
    def _record_intervention(
        self,
        change: PostureChange,
        active_events: List[MarketEvent],
    ) -> InterventionRecord:
        """Record an intervention."""
        trigger_event = active_events[0] if active_events else None
        
        action, description = get_intervention_action(
            EventSeverity(change.trigger_severity) if change.trigger_severity else EventSeverity.NORMAL
        )
        
        record = InterventionRecord(
            timestamp=change.timestamp,
            trigger_event_id=trigger_event.event_id if trigger_event else None,
            trigger_event_title=trigger_event.title if trigger_event else "Event cleared",
            trigger_severity=EventSeverity(change.trigger_severity) if change.trigger_severity else EventSeverity.NORMAL,
            action=action,
            action_description=description,
            parameters=change.parameters or {},
        )
        
        self._interventions.append(record)
        return record
    
    def save_audit_log(self, filename: str = None):
        """Save full audit log."""
        fname = filename or f"intervention_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.audit_dir / fname
        
        data = {
            "saved_at": datetime.now().isoformat(),
            "stats": self.get_stats(),
            "interventions": [i.to_dict() for i in self._interventions],
            "posture_changes": [c.to_dict() for c in self.posture_manager.get_change_history()],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load_event_calendar(self, filepath: str):
        """Load event calendar from file."""
        self.classifier.load_calendar_from_file(filepath)
    
    def setup_indian_market_defaults(self):
        """Set up default Indian market events."""
        now = datetime.now()
        
        # Register known holidays as CRITICAL
        for holiday_str in INDIAN_MARKET_EVENTS.get("known_holidays_2024", []):
            try:
                holiday_date = datetime.strptime(holiday_str, "%Y-%m-%d")
                if holiday_date > now - timedelta(days=1):  # Skip past holidays
                    self.register_event(
                        event_type=EventType.MARKET_HOLIDAY,
                        event_time=holiday_date.replace(hour=9, minute=0),
                        title=f"Market Holiday: {holiday_str}",
                        custom_severity=EventSeverity.CRITICAL,
                    )
            except ValueError:
                pass
