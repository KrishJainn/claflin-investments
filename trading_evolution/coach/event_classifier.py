"""
Event Classifier for News Interventions.

Classifies market events by severity:
- CRITICAL: RBI rate decisions, budget, market closures
- HIGH: Major earnings, geopolitical events
- MEDIUM: Sector-specific news, economic data
- LOW: Regular news, analyst upgrades/downgrades

Ensures time validity (no future info leakage).
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum
import json


class EventSeverity(Enum):
    """Event severity levels."""
    CRITICAL = "CRITICAL"  # Freeze trading
    HIGH = "HIGH"          # Conservative mode
    MEDIUM = "MEDIUM"      # Reduced size
    LOW = "LOW"            # Normal trading
    NORMAL = "NORMAL"      # No event impact


class EventType(Enum):
    """Types of market events."""
    # Monetary
    RBI_POLICY = "rbi_policy"
    FED_DECISION = "fed_decision"
    
    # Economic
    GDP_RELEASE = "gdp_release"
    INFLATION_DATA = "inflation_data"
    EMPLOYMENT_DATA = "employment_data"
    
    # Market
    BUDGET = "budget"
    MARKET_HOLIDAY = "market_holiday"
    EXPIRY_DAY = "expiry_day"
    
    # Corporate
    EARNINGS = "earnings"
    MAJOR_ANNOUNCEMENT = "major_announcement"
    
    # Geopolitical
    GEOPOLITICAL = "geopolitical"
    
    # Other
    SECTOR_NEWS = "sector_news"
    ANALYST_ACTION = "analyst_action"
    UNKNOWN = "unknown"


# Default severity mappings
EVENT_SEVERITY_MAP: Dict[EventType, EventSeverity] = {
    EventType.RBI_POLICY: EventSeverity.CRITICAL,
    EventType.FED_DECISION: EventSeverity.HIGH,
    EventType.BUDGET: EventSeverity.CRITICAL,
    EventType.MARKET_HOLIDAY: EventSeverity.CRITICAL,
    EventType.EXPIRY_DAY: EventSeverity.HIGH,
    EventType.GDP_RELEASE: EventSeverity.MEDIUM,
    EventType.INFLATION_DATA: EventSeverity.MEDIUM,
    EventType.EMPLOYMENT_DATA: EventSeverity.MEDIUM,
    EventType.EARNINGS: EventSeverity.HIGH,
    EventType.MAJOR_ANNOUNCEMENT: EventSeverity.HIGH,
    EventType.GEOPOLITICAL: EventSeverity.HIGH,
    EventType.SECTOR_NEWS: EventSeverity.MEDIUM,
    EventType.ANALYST_ACTION: EventSeverity.LOW,
    EventType.UNKNOWN: EventSeverity.LOW,
}


@dataclass
class MarketEvent:
    """A classified market event."""
    
    # Identification
    event_id: str
    event_type: EventType
    severity: EventSeverity
    
    # Timing
    event_time: datetime
    valid_from: datetime  # When the event info becomes actionable
    valid_until: datetime  # When the impact expires
    
    # Details
    title: str
    description: str = ""
    affected_symbols: List[str] = field(default_factory=list)
    affected_sectors: List[str] = field(default_factory=list)
    
    # Source
    source: str = ""
    confidence: float = 1.0  # 0.0 to 1.0
    
    def is_valid_at(self, check_time: datetime) -> bool:
        """Check if event is valid at a given time (no future info)."""
        return self.valid_from <= check_time <= self.valid_until
    
    def affects_symbol(self, symbol: str) -> bool:
        """Check if event affects a specific symbol."""
        if not self.affected_symbols:
            return True  # Affects all if no specific symbols
        return symbol in self.affected_symbols
    
    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "event_time": self.event_time.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "title": self.title,
            "description": self.description,
            "affected_symbols": self.affected_symbols,
            "affected_sectors": self.affected_sectors,
            "source": self.source,
            "confidence": self.confidence,
        }


class EventClassifier:
    """
    Classifies and manages market events.
    
    Features:
    - Severity classification
    - Time validity checking (no future info)
    - Symbol/sector filtering
    """
    
    def __init__(
        self,
        severity_map: Dict[EventType, EventSeverity] = None,
        default_validity_hours: int = 4,
    ):
        """
        Initialize classifier.
        
        Args:
            severity_map: Custom severity mappings
            default_validity_hours: Default event validity period
        """
        self.severity_map = severity_map or EVENT_SEVERITY_MAP.copy()
        self.default_validity_hours = default_validity_hours
        
        # Event storage
        self._events: Dict[str, MarketEvent] = {}
        
        # Known scheduled events (date -> list of events)
        self._calendar: Dict[date, List[MarketEvent]] = {}
    
    def classify(
        self,
        event_type: EventType,
        event_time: datetime,
        title: str,
        description: str = "",
        affected_symbols: List[str] = None,
        affected_sectors: List[str] = None,
        custom_severity: EventSeverity = None,
        validity_hours: int = None,
    ) -> MarketEvent:
        """
        Classify and register a new event.
        
        Args:
            event_type: Type of event
            event_time: When the event occurs
            title: Event title
            description: Event description
            affected_symbols: Symbols affected
            affected_sectors: Sectors affected
            custom_severity: Override default severity
            validity_hours: Override validity period
            
        Returns:
            Classified MarketEvent
        """
        # Determine severity
        severity = custom_severity or self.severity_map.get(
            event_type, EventSeverity.LOW
        )
        
        # Calculate validity window
        hours = validity_hours or self._get_validity_hours(event_type, severity)
        
        # For scheduled events, valid from announcement
        # For surprise events, valid from event_time
        valid_from = event_time - timedelta(hours=hours // 2)
        valid_until = event_time + timedelta(hours=hours)
        
        event = MarketEvent(
            event_id=f"evt_{event_time.strftime('%Y%m%d_%H%M')}_{event_type.value[:8]}",
            event_type=event_type,
            severity=severity,
            event_time=event_time,
            valid_from=valid_from,
            valid_until=valid_until,
            title=title,
            description=description,
            affected_symbols=affected_symbols or [],
            affected_sectors=affected_sectors or [],
        )
        
        self._events[event.event_id] = event
        
        # Add to calendar
        event_date = event_time.date()
        if event_date not in self._calendar:
            self._calendar[event_date] = []
        self._calendar[event_date].append(event)
        
        return event
    
    def get_active_events(
        self,
        current_time: datetime,
        symbol: str = None,
    ) -> List[MarketEvent]:
        """
        Get all active events at current time.
        
        Args:
            current_time: Current time (for time validity check)
            symbol: Filter by symbol (optional)
            
        Returns:
            List of active events
        """
        active = []
        
        for event in self._events.values():
            if not event.is_valid_at(current_time):
                continue
            
            if symbol and not event.affects_symbol(symbol):
                continue
            
            active.append(event)
        
        # Sort by severity (CRITICAL first)
        severity_order = {
            EventSeverity.CRITICAL: 0,
            EventSeverity.HIGH: 1,
            EventSeverity.MEDIUM: 2,
            EventSeverity.LOW: 3,
            EventSeverity.NORMAL: 4,
        }
        active.sort(key=lambda e: severity_order.get(e.severity, 5))
        
        return active
    
    def get_max_severity(
        self,
        current_time: datetime,
        symbol: str = None,
    ) -> EventSeverity:
        """Get the maximum severity of active events."""
        active = self.get_active_events(current_time, symbol)
        
        if not active:
            return EventSeverity.NORMAL
        
        return active[0].severity  # Already sorted by severity
    
    def get_events_for_date(self, target_date: date) -> List[MarketEvent]:
        """Get all events for a specific date."""
        return self._calendar.get(target_date, [])
    
    def register_known_events(self, events: List[Dict]):
        """
        Register known/scheduled events from external source.
        
        Args:
            events: List of event dictionaries
        """
        for evt_data in events:
            try:
                event_type = EventType(evt_data.get("type", "unknown"))
            except ValueError:
                event_type = EventType.UNKNOWN
            
            self.classify(
                event_type=event_type,
                event_time=datetime.fromisoformat(evt_data["time"]),
                title=evt_data.get("title", ""),
                description=evt_data.get("description", ""),
                affected_symbols=evt_data.get("symbols", []),
                affected_sectors=evt_data.get("sectors", []),
            )
    
    def _get_validity_hours(
        self,
        event_type: EventType,
        severity: EventSeverity,
    ) -> int:
        """Get validity hours based on event type and severity."""
        
        # Critical events have longer impact
        if severity == EventSeverity.CRITICAL:
            return 24  # Full day
        elif severity == EventSeverity.HIGH:
            return 8   # Intraday
        elif severity == EventSeverity.MEDIUM:
            return 4
        else:
            return 2
    
    def load_calendar_from_file(self, filepath: str):
        """Load event calendar from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.register_known_events(data.get("events", []))
    
    def save_calendar_to_file(self, filepath: str):
        """Save event calendar to JSON file."""
        events = [e.to_dict() for e in self._events.values()]
        
        with open(filepath, 'w') as f:
            json.dump({"events": events}, f, indent=2)


# Pre-defined Indian market events (can be expanded)
INDIAN_MARKET_EVENTS = {
    "expiry_days": {
        "weekly": [3],  # Thursday
        "monthly_last_thursday": True,
    },
    "known_holidays_2024": [
        "2024-01-26",  # Republic Day
        "2024-03-08",  # Mahashivratri
        "2024-03-25",  # Holi
        "2024-03-29",  # Good Friday
        "2024-04-11",  # Eid
        "2024-04-14",  # Dr. Ambedkar Jayanti
        "2024-04-17",  # Ram Navami
        "2024-04-21",  # Mahavir Jayanti
        "2024-05-01",  # May Day
        "2024-05-23",  # Buddha Purnima
        "2024-06-17",  # Eid ul-Adha
        "2024-07-17",  # Muharram
        "2024-08-15",  # Independence Day
        "2024-10-02",  # Gandhi Jayanti
        "2024-11-01",  # Diwali
        "2024-11-15",  # Guru Nanak Jayanti
        "2024-12-25",  # Christmas
    ],
}
