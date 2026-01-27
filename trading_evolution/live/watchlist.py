"""
Watchlist Management module.

Manages stocks being actively monitored for signals.
"""

from typing import Dict, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class WatchlistItem:
    """Item in the watchlist."""
    symbol: str
    added_date: datetime
    notes: str = ""
    alert_threshold_long: float = 0.7
    alert_threshold_short: float = -0.7
    last_signal_value: float = 0.0
    last_checked: datetime = None
    triggered: bool = False
    priority: int = 1  # 1 = high, 2 = medium, 3 = low


class Watchlist:
    """
    Manages a list of stocks to monitor for trading signals.

    Features:
    - Add/remove symbols
    - Set custom alert thresholds
    - Track signal history
    - Priority ordering
    - Persistence to JSON
    """

    def __init__(self, filepath: str = None):
        """
        Initialize watchlist.

        Args:
            filepath: Path for persistent storage
        """
        self.filepath = filepath
        self._items: Dict[str, WatchlistItem] = {}

        if filepath:
            self._load()

    def add(self,
            symbol: str,
            notes: str = "",
            alert_threshold_long: float = 0.7,
            alert_threshold_short: float = -0.7,
            priority: int = 1):
        """
        Add symbol to watchlist.

        Args:
            symbol: Trading symbol
            notes: Optional notes
            alert_threshold_long: Threshold for long entry alerts
            alert_threshold_short: Threshold for short entry alerts
            priority: Priority level (1=high, 2=medium, 3=low)
        """
        symbol = symbol.upper()

        if symbol in self._items:
            logger.info(f"{symbol} already in watchlist, updating")

        self._items[symbol] = WatchlistItem(
            symbol=symbol,
            added_date=datetime.now(),
            notes=notes,
            alert_threshold_long=alert_threshold_long,
            alert_threshold_short=alert_threshold_short,
            priority=priority
        )

        self._save()
        logger.info(f"Added {symbol} to watchlist")

    def remove(self, symbol: str):
        """
        Remove symbol from watchlist.

        Args:
            symbol: Trading symbol
        """
        symbol = symbol.upper()

        if symbol in self._items:
            del self._items[symbol]
            self._save()
            logger.info(f"Removed {symbol} from watchlist")
        else:
            logger.warning(f"{symbol} not in watchlist")

    def get(self, symbol: str) -> Optional[WatchlistItem]:
        """
        Get watchlist item.

        Args:
            symbol: Trading symbol

        Returns:
            WatchlistItem or None
        """
        return self._items.get(symbol.upper())

    def get_all(self) -> List[WatchlistItem]:
        """Get all watchlist items sorted by priority."""
        return sorted(self._items.values(), key=lambda x: (x.priority, x.symbol))

    def get_symbols(self) -> List[str]:
        """Get list of all symbols."""
        return list(self._items.keys())

    def get_high_priority(self) -> List[str]:
        """Get high priority symbols."""
        return [item.symbol for item in self._items.values() if item.priority == 1]

    def contains(self, symbol: str) -> bool:
        """Check if symbol is in watchlist."""
        return symbol.upper() in self._items

    def update_signal(self, symbol: str, signal_value: float):
        """
        Update last signal value for a symbol.

        Args:
            symbol: Trading symbol
            signal_value: Current Super Indicator value
        """
        symbol = symbol.upper()
        if symbol in self._items:
            item = self._items[symbol]
            item.last_signal_value = signal_value
            item.last_checked = datetime.now()

            # Check if threshold triggered
            if (signal_value >= item.alert_threshold_long or
                signal_value <= item.alert_threshold_short):
                item.triggered = True
            else:
                item.triggered = False

            self._save()

    def get_triggered(self) -> List[WatchlistItem]:
        """Get items that have triggered alerts."""
        return [item for item in self._items.values() if item.triggered]

    def reset_triggered(self, symbol: str):
        """Reset triggered status for a symbol."""
        symbol = symbol.upper()
        if symbol in self._items:
            self._items[symbol].triggered = False
            self._save()

    def set_threshold(self,
                      symbol: str,
                      long_threshold: float = None,
                      short_threshold: float = None):
        """
        Set alert thresholds for a symbol.

        Args:
            symbol: Trading symbol
            long_threshold: New long entry threshold
            short_threshold: New short entry threshold
        """
        symbol = symbol.upper()
        if symbol in self._items:
            if long_threshold is not None:
                self._items[symbol].alert_threshold_long = long_threshold
            if short_threshold is not None:
                self._items[symbol].alert_threshold_short = short_threshold
            self._save()

    def set_priority(self, symbol: str, priority: int):
        """
        Set priority for a symbol.

        Args:
            symbol: Trading symbol
            priority: Priority level (1=high, 2=medium, 3=low)
        """
        symbol = symbol.upper()
        if symbol in self._items:
            self._items[symbol].priority = priority
            self._save()

    def add_note(self, symbol: str, note: str):
        """
        Add note to a symbol.

        Args:
            symbol: Trading symbol
            note: Note text
        """
        symbol = symbol.upper()
        if symbol in self._items:
            self._items[symbol].notes = note
            self._save()

    def clear(self):
        """Clear all items from watchlist."""
        self._items = {}
        self._save()
        logger.info("Watchlist cleared")

    def _save(self):
        """Save watchlist to file."""
        if not self.filepath:
            return

        data = {}
        for symbol, item in self._items.items():
            data[symbol] = {
                'symbol': item.symbol,
                'added_date': item.added_date.isoformat(),
                'notes': item.notes,
                'alert_threshold_long': item.alert_threshold_long,
                'alert_threshold_short': item.alert_threshold_short,
                'last_signal_value': item.last_signal_value,
                'last_checked': item.last_checked.isoformat() if item.last_checked else None,
                'triggered': item.triggered,
                'priority': item.priority
            }

        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load watchlist from file."""
        if not self.filepath:
            return

        path = Path(self.filepath)
        if not path.exists():
            return

        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)

            for symbol, item_data in data.items():
                self._items[symbol] = WatchlistItem(
                    symbol=item_data['symbol'],
                    added_date=datetime.fromisoformat(item_data['added_date']),
                    notes=item_data.get('notes', ''),
                    alert_threshold_long=item_data.get('alert_threshold_long', 0.7),
                    alert_threshold_short=item_data.get('alert_threshold_short', -0.7),
                    last_signal_value=item_data.get('last_signal_value', 0),
                    last_checked=(datetime.fromisoformat(item_data['last_checked'])
                                 if item_data.get('last_checked') else None),
                    triggered=item_data.get('triggered', False),
                    priority=item_data.get('priority', 1)
                )

            logger.info(f"Loaded {len(self._items)} items from watchlist")

        except Exception as e:
            logger.warning(f"Could not load watchlist: {e}")

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if not self._items:
            return "Watchlist is empty"

        lines = [
            "Watchlist Summary",
            "=" * 50,
            f"Total symbols: {len(self._items)}",
            f"High priority: {len(self.get_high_priority())}",
            f"Triggered alerts: {len(self.get_triggered())}",
            "",
            "Symbols:"
        ]

        for item in self.get_all():
            status = "*" if item.triggered else " "
            priority_str = {1: "HIGH", 2: "MED", 3: "LOW"}[item.priority]
            lines.append(
                f"  {status} {item.symbol:6s} | {priority_str:4s} | "
                f"Signal: {item.last_signal_value:+.2f} | "
                f"Long>{item.alert_threshold_long:.1f} Short<{item.alert_threshold_short:.1f}"
            )

        return "\n".join(lines)

    def __len__(self) -> int:
        """Get number of items in watchlist."""
        return len(self._items)

    def __iter__(self):
        """Iterate over symbols."""
        return iter(self._items.keys())


def create_default_watchlist(filepath: str = None) -> Watchlist:
    """
    Create watchlist with default symbols.

    Args:
        filepath: Path for persistent storage

    Returns:
        Watchlist with default symbols
    """
    watchlist = Watchlist(filepath)

    # Default symbols from config
    default_symbols = [
        ('SPY', 1, 'S&P 500 ETF'),
        ('QQQ', 1, 'Nasdaq 100 ETF'),
        ('AAPL', 1, 'Apple Inc'),
        ('MSFT', 1, 'Microsoft'),
        ('GOOGL', 1, 'Alphabet'),
        ('AMZN', 1, 'Amazon'),
        ('NVDA', 1, 'Nvidia'),
        ('META', 1, 'Meta Platforms'),
        ('TSLA', 2, 'Tesla'),
        ('JPM', 2, 'JP Morgan'),
        ('V', 2, 'Visa'),
        ('JNJ', 2, 'Johnson & Johnson'),
        ('WMT', 2, 'Walmart'),
        ('PG', 2, 'Procter & Gamble'),
        ('MA', 2, 'Mastercard'),
        ('HD', 2, 'Home Depot'),
        ('BAC', 3, 'Bank of America'),
        ('XOM', 3, 'Exxon Mobil'),
        ('PFE', 3, 'Pfizer'),
        ('KO', 3, 'Coca-Cola'),
    ]

    for symbol, priority, notes in default_symbols:
        watchlist.add(symbol, notes=notes, priority=priority)

    logger.info(f"Created default watchlist with {len(watchlist)} symbols")
    return watchlist
