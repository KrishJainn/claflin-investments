"""
Alert System module.

Alerts when entry/exit signals trigger.
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

from .signal_generator import LiveSignal
from ..super_indicator.signals import SignalType

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class AlertType(Enum):
    """Types of alerts."""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    STOP_HIT = "stop_hit"
    TARGET_HIT = "target_hit"
    REGIME_CHANGE = "regime_change"
    CUSTOM = "custom"


@dataclass
class Alert:
    """Trading alert."""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    symbol: str
    message: str
    timestamp: datetime
    signal_value: float
    price: float
    acknowledged: bool = False
    details: Dict = None


class AlertManager:
    """
    Manages trading alerts.

    Features:
    - Generate alerts from signals
    - Priority-based alerting
    - Alert history tracking
    - Console output
    - Callback support for custom notifications
    """

    def __init__(self,
                 min_priority: AlertPriority = AlertPriority.MEDIUM,
                 callback: Callable[[Alert], None] = None):
        """
        Initialize alert manager.

        Args:
            min_priority: Minimum priority to generate alerts
            callback: Optional callback function for custom notifications
        """
        self.min_priority = min_priority
        self.callback = callback
        self._alerts: List[Alert] = []
        self._alert_counter = 0

    def process_signals(self, signals: List[LiveSignal]) -> List[Alert]:
        """
        Process signals and generate alerts.

        Args:
            signals: List of LiveSignal objects

        Returns:
            List of generated alerts
        """
        new_alerts = []

        for signal in signals:
            alert = self._signal_to_alert(signal)
            if alert and alert.priority.value <= self.min_priority.value:
                self._alerts.append(alert)
                new_alerts.append(alert)

                # Print to console
                self._print_alert(alert)

                # Call callback if provided
                if self.callback:
                    self.callback(alert)

        return new_alerts

    def _signal_to_alert(self, signal: LiveSignal) -> Optional[Alert]:
        """Convert a LiveSignal to an Alert."""
        self._alert_counter += 1
        alert_id = f"ALT-{self._alert_counter:05d}"

        # Determine alert type and priority
        if signal.signal_type == SignalType.LONG_ENTRY:
            alert_type = AlertType.ENTRY_LONG
            priority = AlertPriority.HIGH if signal.confidence > 0.8 else AlertPriority.MEDIUM
            message = (
                f"LONG ENTRY: {signal.symbol} at ${signal.entry_price:.2f} | "
                f"Signal: {signal.signal_strength:+.2f} | "
                f"Stop: ${signal.stop_price:.2f} | "
                f"Target: ${signal.target_price:.2f}"
            )

        elif signal.signal_type == SignalType.SHORT_ENTRY:
            alert_type = AlertType.ENTRY_SHORT
            priority = AlertPriority.HIGH if signal.confidence > 0.8 else AlertPriority.MEDIUM
            message = (
                f"SHORT ENTRY: {signal.symbol} at ${signal.entry_price:.2f} | "
                f"Signal: {signal.signal_strength:+.2f} | "
                f"Stop: ${signal.stop_price:.2f} | "
                f"Target: ${signal.target_price:.2f}"
            )

        elif signal.signal_type == SignalType.LONG_EXIT:
            alert_type = AlertType.EXIT_LONG
            priority = AlertPriority.MEDIUM
            message = (
                f"LONG EXIT: {signal.symbol} | "
                f"Signal weakened to {signal.signal_strength:+.2f}"
            )

        elif signal.signal_type == SignalType.SHORT_EXIT:
            alert_type = AlertType.EXIT_SHORT
            priority = AlertPriority.MEDIUM
            message = (
                f"SHORT EXIT: {signal.symbol} | "
                f"Signal strengthened to {signal.signal_strength:+.2f}"
            )

        else:
            return None

        return Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            priority=priority,
            symbol=signal.symbol,
            message=message,
            timestamp=signal.timestamp,
            signal_value=signal.signal_strength,
            price=signal.current_price,
            details={
                'confidence': signal.confidence,
                'regime': signal.market_regime.value,
                'position_size': signal.position_size,
                'risk_amount': signal.risk_amount,
                'top_contributors': signal.top_contributors
            }
        )

    def _print_alert(self, alert: Alert):
        """Print alert to console."""
        priority_str = {
            AlertPriority.CRITICAL: "!!!",
            AlertPriority.HIGH: "** ",
            AlertPriority.MEDIUM: "*  ",
            AlertPriority.LOW: "   "
        }[alert.priority]

        timestamp = alert.timestamp.strftime("%H:%M:%S")

        print(f"\n{priority_str} [{timestamp}] ALERT: {alert.message}")

        if alert.priority == AlertPriority.CRITICAL:
            print("=" * 60)

    def create_alert(self,
                     alert_type: AlertType,
                     symbol: str,
                     message: str,
                     price: float = 0,
                     signal_value: float = 0,
                     priority: AlertPriority = AlertPriority.MEDIUM,
                     details: Dict = None) -> Alert:
        """
        Create a custom alert.

        Args:
            alert_type: Type of alert
            symbol: Trading symbol
            message: Alert message
            price: Current price
            signal_value: Signal value
            priority: Alert priority
            details: Additional details

        Returns:
            Created Alert object
        """
        self._alert_counter += 1
        alert_id = f"ALT-{self._alert_counter:05d}"

        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            priority=priority,
            symbol=symbol,
            message=message,
            timestamp=datetime.now(),
            signal_value=signal_value,
            price=price,
            details=details
        )

        if priority.value <= self.min_priority.value:
            self._alerts.append(alert)
            self._print_alert(alert)

            if self.callback:
                self.callback(alert)

        return alert

    def alert_stop_hit(self, symbol: str, price: float, entry_price: float, direction: str):
        """Create a stop loss hit alert."""
        pnl_pct = ((price - entry_price) / entry_price * 100
                  if direction == 'LONG'
                  else (entry_price - price) / entry_price * 100)

        return self.create_alert(
            alert_type=AlertType.STOP_HIT,
            symbol=symbol,
            message=f"STOP HIT: {symbol} {direction} closed at ${price:.2f} ({pnl_pct:+.1f}%)",
            price=price,
            priority=AlertPriority.HIGH
        )

    def alert_target_hit(self, symbol: str, price: float, entry_price: float, direction: str):
        """Create a target hit alert."""
        pnl_pct = ((price - entry_price) / entry_price * 100
                  if direction == 'LONG'
                  else (entry_price - price) / entry_price * 100)

        return self.create_alert(
            alert_type=AlertType.TARGET_HIT,
            symbol=symbol,
            message=f"TARGET HIT: {symbol} {direction} closed at ${price:.2f} ({pnl_pct:+.1f}%)",
            price=price,
            priority=AlertPriority.MEDIUM
        )

    def alert_regime_change(self, symbol: str, old_regime: str, new_regime: str):
        """Create a regime change alert."""
        return self.create_alert(
            alert_type=AlertType.REGIME_CHANGE,
            symbol=symbol,
            message=f"REGIME CHANGE: {symbol} changed from {old_regime} to {new_regime}",
            priority=AlertPriority.LOW
        )

    def get_alerts(self,
                   symbol: str = None,
                   alert_type: AlertType = None,
                   unacknowledged_only: bool = False) -> List[Alert]:
        """
        Get alerts with optional filters.

        Args:
            symbol: Filter by symbol
            alert_type: Filter by alert type
            unacknowledged_only: Only unacknowledged alerts

        Returns:
            List of matching alerts
        """
        alerts = self._alerts.copy()

        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        return alerts

    def acknowledge(self, alert_id: str):
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return

    def acknowledge_all(self, symbol: str = None):
        """Acknowledge all alerts, optionally filtered by symbol."""
        for alert in self._alerts:
            if symbol is None or alert.symbol == symbol:
                alert.acknowledged = True

    def clear_old_alerts(self, max_age_hours: int = 24):
        """Clear alerts older than specified hours."""
        cutoff = datetime.now()
        from datetime import timedelta
        cutoff = cutoff - timedelta(hours=max_age_hours)

        self._alerts = [a for a in self._alerts if a.timestamp > cutoff]

    def get_summary(self) -> str:
        """Get alert summary."""
        if not self._alerts:
            return "No alerts"

        lines = [
            "Alert Summary",
            "=" * 50,
            f"Total alerts: {len(self._alerts)}",
            f"Unacknowledged: {len([a for a in self._alerts if not a.acknowledged])}",
            "",
            "Recent Alerts:"
        ]

        for alert in sorted(self._alerts, key=lambda a: a.timestamp, reverse=True)[:10]:
            ack = " " if alert.acknowledged else "*"
            lines.append(
                f"  {ack} [{alert.timestamp.strftime('%H:%M')}] "
                f"{alert.priority.name[:3]:3s} | {alert.symbol:6s} | {alert.alert_type.value}"
            )

        return "\n".join(lines)


def print_signal_banner(title: str, signals: List[LiveSignal]):
    """
    Print formatted signal banner to console.

    Args:
        title: Banner title
        signals: List of signals
    """
    width = 70
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

    if not signals:
        print("  No signals at this time")
    else:
        for signal in signals:
            direction = "LONG" if signal.signal_type in [SignalType.LONG_ENTRY, SignalType.LONG_EXIT] else "SHORT"
            action = "ENTRY" if signal.signal_type in [SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY] else "EXIT"

            print(
                f"  {signal.symbol:6s} {direction:5s} {action:5s} | "
                f"Signal: {signal.signal_strength:+.2f} | "
                f"Price: ${signal.current_price:.2f} | "
                f"Conf: {signal.confidence:.0%}"
            )

    print("=" * width + "\n")


def monitor_watchlist(signal_generator,
                      watchlist,
                      alert_manager: AlertManager,
                      interval_seconds: int = 60):
    """
    Continuously monitor watchlist for signals.

    Args:
        signal_generator: LiveSignalGenerator instance
        watchlist: Watchlist instance
        alert_manager: AlertManager instance
        interval_seconds: Check interval
    """
    import time

    logger.info(f"Starting watchlist monitor (interval: {interval_seconds}s)")

    try:
        while True:
            symbols = watchlist.get_symbols()
            signals = signal_generator.generate_signals(symbols)

            # Update watchlist with signal values
            for signal in signals:
                watchlist.update_signal(signal.symbol, signal.signal_strength)

            # Generate alerts
            if signals:
                alert_manager.process_signals(signals)

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")
