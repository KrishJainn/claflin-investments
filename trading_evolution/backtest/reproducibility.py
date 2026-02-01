"""
Reproducibility and Event Logging Framework.

Ensures backtest runs are fully reproducible and provides
comprehensive event logging for debugging and analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json
import uuid
import logging
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that can be logged."""
    
    # System events
    BACKTEST_START = "backtest_start"
    BACKTEST_END = "backtest_end"
    NEW_DAY = "new_day"
    
    # Data events
    BAR_PROCESSED = "bar_processed"
    DATA_LOADED = "data_loaded"
    
    # Signal events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_BELOW_THRESHOLD = "signal_below_threshold"
    
    # Trade events
    ENTRY_ATTEMPTED = "entry_attempted"
    ENTRY_EXECUTED = "entry_executed"
    ENTRY_REJECTED = "entry_rejected"
    EXIT_EXECUTED = "exit_executed"
    
    # Risk events
    DAILY_LIMIT_CHECK = "daily_limit_check"
    DAILY_LIMIT_HIT = "daily_limit_hit"
    POSITION_LIMIT_HIT = "position_limit_hit"
    COOLDOWN_ACTIVE = "cooldown_active"
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"
    EOD_FLATTEN = "eod_flatten"
    
    # Error events
    ERROR = "error"
    WARNING = "warning"


@dataclass
class Event:
    """A single logged event."""
    
    event_id: str
    event_type: EventType
    timestamp: datetime
    bar_index: int
    symbol: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'bar_index': self.bar_index,
            'symbol': self.symbol,
            'data': self.data,
            'reason': self.reason,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Event':
        """Create from dictionary."""
        data = data.copy()
        data['event_type'] = EventType(data['event_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class EventLogger:
    """
    Logs all trading events for analysis and debugging.
    
    Every decision point writes an event, enabling:
    - Full audit trail
    - "Why" analysis for each trade
    - Debugging failed trades
    - Performance attribution
    """
    
    def __init__(self, run_id: str = None):
        """
        Initialize event logger.
        
        Args:
            run_id: Optional run identifier
        """
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self._events: List[Event] = []
        self._current_bar_index: int = 0
    
    def log(
        self,
        event_type: EventType,
        timestamp: datetime,
        symbol: str = "",
        data: Dict[str, Any] = None,
        reason: str = "",
    ) -> Event:
        """
        Log an event.
        
        Args:
            event_type: Type of event
            timestamp: When the event occurred
            symbol: Related symbol (if any)
            data: Additional event data
            reason: Reason for the event
            
        Returns:
            The logged Event
        """
        event = Event(
            event_id=f"{self.run_id}_{len(self._events):06d}",
            event_type=event_type,
            timestamp=timestamp,
            bar_index=self._current_bar_index,
            symbol=symbol,
            data=data or {},
            reason=reason,
        )
        
        self._events.append(event)
        
        # Log to Python logger at appropriate level
        if event_type in [EventType.ERROR]:
            logger.error(f"{event_type.value}: {symbol} - {reason}")
        elif event_type in [EventType.WARNING, EventType.DAILY_LIMIT_HIT]:
            logger.warning(f"{event_type.value}: {symbol} - {reason}")
        else:
            logger.debug(f"{event_type.value}: {symbol} - {reason}")
        
        return event
    
    def set_bar_index(self, index: int):
        """Set current bar index for events."""
        self._current_bar_index = index
    
    def get_events(
        self,
        event_type: EventType = None,
        symbol: str = None,
    ) -> List[Event]:
        """
        Get filtered events.
        
        Args:
            event_type: Filter by event type
            symbol: Filter by symbol
            
        Returns:
            List of matching events
        """
        events = self._events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if symbol:
            events = [e for e in events if e.symbol == symbol]
        
        return events
    
    def get_trade_events(self, symbol: str = None) -> List[Event]:
        """Get all trade-related events."""
        trade_types = [
            EventType.ENTRY_ATTEMPTED,
            EventType.ENTRY_EXECUTED,
            EventType.ENTRY_REJECTED,
            EventType.EXIT_EXECUTED,
            EventType.STOP_LOSS_HIT,
            EventType.TAKE_PROFIT_HIT,
            EventType.EOD_FLATTEN,
        ]
        
        events = [e for e in self._events if e.event_type in trade_types]
        
        if symbol:
            events = [e for e in events if e.symbol == symbol]
        
        return events
    
    def get_rejection_reasons(self) -> Dict[str, int]:
        """Get counts of entry rejection reasons."""
        rejections = self.get_events(EventType.ENTRY_REJECTED)
        
        reasons = {}
        for e in rejections:
            reason = e.reason
            reasons[reason] = reasons.get(reason, 0) + 1
        
        return reasons
    
    def export_to_json(self, filepath: str):
        """Export all events to JSON file."""
        events_data = [e.to_dict() for e in self._events]
        
        with open(filepath, 'w') as f:
            json.dump(events_data, f, indent=2)
        
        logger.info(f"Exported {len(events_data)} events to {filepath}")
    
    def clear(self):
        """Clear all events."""
        self._events = []
        self._current_bar_index = 0


class ReproducibilityManager:
    """
    Ensures backtest runs are fully reproducible.
    
    Provides:
    - Config hashing
    - Data hashing
    - Strategy versioning
    - Run manifests
    """
    
    @staticmethod
    def generate_run_id() -> str:
        """Generate unique run identifier."""
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    @staticmethod
    def hash_config(config_dict: Dict) -> str:
        """
        Generate deterministic hash of configuration.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            8-character hash
        """
        # Sort keys for determinism
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    @staticmethod
    def hash_data(data: Dict[str, Any]) -> str:
        """
        Generate hash of data for reproducibility verification.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            
        Returns:
            8-character hash
        """
        hash_parts = []
        
        for symbol in sorted(data.keys()):
            df = data[symbol]
            if hasattr(df, 'index') and len(df) > 0:
                hash_parts.append(f"{symbol}:{len(df)}:{df.index[0]}:{df.index[-1]}")
            else:
                hash_parts.append(f"{symbol}:0")
        
        data_str = "|".join(hash_parts)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    @staticmethod
    def hash_strategy(indicator_weights: Dict[str, float]) -> str:
        """
        Generate hash of strategy (indicator weights).
        
        Args:
            indicator_weights: Dictionary of indicator -> weight
            
        Returns:
            8-character hash
        """
        weights_str = json.dumps(indicator_weights, sort_keys=True)
        return hashlib.md5(weights_str.encode()).hexdigest()[:8]
    
    @staticmethod
    def create_manifest(
        run_id: str,
        config_hash: str,
        data_hash: str,
        strategy_hash: str,
        result_summary: Dict,
    ) -> Dict:
        """
        Create a run manifest for reproducibility.
        
        Args:
            run_id: Unique run identifier
            config_hash: Hash of configuration
            data_hash: Hash of data
            strategy_hash: Hash of strategy
            result_summary: Summary of results
            
        Returns:
            Manifest dictionary
        """
        return {
            'manifest_version': '1.0',
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'hashes': {
                'config': config_hash,
                'data': data_hash,
                'strategy': strategy_hash,
                'combined': hashlib.md5(
                    f"{config_hash}{data_hash}{strategy_hash}".encode()
                ).hexdigest()[:8],
            },
            'result_summary': result_summary,
        }
    
    @staticmethod
    def verify_reproducibility(manifest1: Dict, manifest2: Dict) -> Tuple[bool, List[str]]:
        """
        Verify that two runs are reproducible.
        
        Args:
            manifest1: First run manifest
            manifest2: Second run manifest
            
        Returns:
            Tuple of (is_reproducible, list of differences)
        """
        differences = []
        
        # Compare hashes
        for hash_type in ['config', 'data', 'strategy']:
            h1 = manifest1.get('hashes', {}).get(hash_type)
            h2 = manifest2.get('hashes', {}).get(hash_type)
            
            if h1 != h2:
                differences.append(f"{hash_type} hash mismatch: {h1} vs {h2}")
        
        # Compare key results
        r1 = manifest1.get('result_summary', {})
        r2 = manifest2.get('result_summary', {})
        
        if r1.get('total_trades') != r2.get('total_trades'):
            differences.append(f"Trade count mismatch")
        
        pnl1 = r1.get('net_pnl', 0)
        pnl2 = r2.get('net_pnl', 0)
        if abs(pnl1 - pnl2) > 0.01:
            differences.append(f"P&L mismatch: {pnl1} vs {pnl2}")
        
        return len(differences) == 0, differences
    
    @staticmethod
    def save_manifest(manifest: Dict, filepath: str):
        """Save manifest to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    @staticmethod
    def load_manifest(filepath: str) -> Dict:
        """Load manifest from file."""
        with open(filepath, 'r') as f:
            return json.load(f)


# Import Tuple for type hints
