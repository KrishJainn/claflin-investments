"""
Signal Generation module.

Generates trading signals from Super Indicator values:
- LONG_ENTRY: Strong buy signal
- LONG_EXIT: Close long position
- SHORT_ENTRY: Strong sell signal
- SHORT_EXIT: Close short position
- HOLD: Maintain current position
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Types of trading signals."""
    LONG_ENTRY = "LONG_ENTRY"
    LONG_EXIT = "LONG_EXIT"
    SHORT_ENTRY = "SHORT_ENTRY"
    SHORT_EXIT = "SHORT_EXIT"
    HOLD = "HOLD"
    NEUTRAL = "NEUTRAL"


class PositionState(Enum):
    """Current position state."""
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class TradingSignal:
    """Complete trading signal with context."""
    timestamp: pd.Timestamp
    signal_type: SignalType
    super_indicator_value: float
    confidence: float
    suggested_stop_loss: Optional[float] = None
    suggested_take_profit: Optional[float] = None
    contributing_indicators: List[Dict] = None

    def __post_init__(self):
        if self.contributing_indicators is None:
            self.contributing_indicators = []


class SignalGenerator:
    """
    Generates trading signals from Super Indicator values.

    Signal Logic:
    - LONG_ENTRY: SI crosses above +0.7 from below
    - LONG_EXIT: SI drops below +0.3 while in long position
    - SHORT_ENTRY: SI crosses below -0.7 from above
    - SHORT_EXIT: SI rises above -0.3 while in short position

    Additional filters can be applied for signal quality.
    """

    def __init__(self,
                 long_entry_threshold: float = 0.7,
                 long_exit_threshold: float = 0.3,
                 short_entry_threshold: float = -0.7,
                 short_exit_threshold: float = -0.3,
                 confirmation_bars: int = 1,
                 min_signal_change: float = 0.1):
        """
        Initialize signal generator.

        Args:
            long_entry_threshold: SI value to trigger long entry (default 0.7)
            long_exit_threshold: SI value to trigger long exit (default 0.3)
            short_entry_threshold: SI value to trigger short entry (default -0.7)
            short_exit_threshold: SI value to trigger short exit (default -0.3)
            confirmation_bars: Bars to wait for signal confirmation (default 1)
            min_signal_change: Minimum SI change for signal (default 0.1)
        """
        self.long_entry_threshold = long_entry_threshold
        self.long_exit_threshold = long_exit_threshold
        self.short_entry_threshold = short_entry_threshold
        self.short_exit_threshold = short_exit_threshold
        self.confirmation_bars = confirmation_bars
        self.min_signal_change = min_signal_change

    def generate_signals(self, super_indicator: pd.Series,
                         position_state: PositionState = PositionState.FLAT
                         ) -> pd.DataFrame:
        """
        Generate trading signals from Super Indicator series.

        Args:
            super_indicator: Series of SI values [-1, 1]
            position_state: Starting position state

        Returns:
            DataFrame with columns: signal_type, si_value, confidence
        """
        signals = pd.DataFrame(index=super_indicator.index)
        signals['si_value'] = super_indicator
        signals['si_prev'] = super_indicator.shift(1)
        signals['signal_type'] = SignalType.HOLD.value
        signals['confidence'] = abs(super_indicator)

        current_state = position_state

        for i in range(1, len(signals)):
            idx = signals.index[i]
            si = signals.loc[idx, 'si_value']
            si_prev = signals.loc[idx, 'si_prev']

            if pd.isna(si) or pd.isna(si_prev):
                continue

            signal = self._determine_signal(si, si_prev, current_state)
            signals.loc[idx, 'signal_type'] = signal.value

            # Update position state
            if signal == SignalType.LONG_ENTRY:
                current_state = PositionState.LONG
            elif signal == SignalType.SHORT_ENTRY:
                current_state = PositionState.SHORT
            elif signal in (SignalType.LONG_EXIT, SignalType.SHORT_EXIT):
                current_state = PositionState.FLAT

        return signals[['signal_type', 'si_value', 'confidence']]

    def _determine_signal(self, si: float, si_prev: float,
                          position: PositionState) -> SignalType:
        """Determine signal based on SI value and current position."""

        # Entry signals (only when flat or allowing position flips)
        if position == PositionState.FLAT:
            # Long entry: SI crosses above threshold
            if si > self.long_entry_threshold and si_prev <= self.long_entry_threshold:
                return SignalType.LONG_ENTRY

            # Short entry: SI crosses below threshold
            if si < self.short_entry_threshold and si_prev >= self.short_entry_threshold:
                return SignalType.SHORT_ENTRY

        # Exit signals
        elif position == PositionState.LONG:
            # Long exit: SI drops below exit threshold
            if si < self.long_exit_threshold:
                return SignalType.LONG_EXIT

            # Flip to short if strong reversal
            if si < self.short_entry_threshold:
                return SignalType.LONG_EXIT  # Exit first, then can go short

        elif position == PositionState.SHORT:
            # Short exit: SI rises above exit threshold
            if si > self.short_exit_threshold:
                return SignalType.SHORT_EXIT

            # Flip to long if strong reversal
            if si > self.long_entry_threshold:
                return SignalType.SHORT_EXIT  # Exit first, then can go long

        return SignalType.HOLD

    def get_signal_at_timestamp(self, super_indicator: pd.Series,
                                timestamp: pd.Timestamp,
                                position: PositionState) -> TradingSignal:
        """
        Get signal at a specific timestamp.

        Args:
            super_indicator: Full SI series
            timestamp: Point in time
            position: Current position state

        Returns:
            TradingSignal with full context
        """
        if timestamp not in super_indicator.index:
            return TradingSignal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                super_indicator_value=0.0,
                confidence=0.0
            )

        idx = super_indicator.index.get_loc(timestamp)
        si = super_indicator.iloc[idx]

        if idx > 0:
            si_prev = super_indicator.iloc[idx - 1]
        else:
            si_prev = 0.0

        signal_type = self._determine_signal(si, si_prev, position)
        confidence = min(1.0, abs(si) / 0.7)

        return TradingSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            super_indicator_value=si,
            confidence=confidence
        )


class SignalFilter:
    """
    Filters trading signals for quality.

    Applies additional criteria to reduce false signals.
    """

    def __init__(self,
                 min_confidence: float = 0.5,
                 min_holding_period: int = 5,
                 max_daily_signals: int = 3):
        """
        Initialize filter.

        Args:
            min_confidence: Minimum confidence for signal (default 0.5)
            min_holding_period: Minimum bars between entry and exit (default 5)
            max_daily_signals: Maximum signals per day (default 3)
        """
        self.min_confidence = min_confidence
        self.min_holding_period = min_holding_period
        self.max_daily_signals = max_daily_signals

    def filter_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Filter signals based on quality criteria.

        Args:
            signals: DataFrame from SignalGenerator.generate_signals()

        Returns:
            Filtered DataFrame
        """
        filtered = signals.copy()

        # Filter by confidence
        entry_mask = filtered['signal_type'].isin([
            SignalType.LONG_ENTRY.value,
            SignalType.SHORT_ENTRY.value
        ])
        low_confidence = entry_mask & (filtered['confidence'] < self.min_confidence)
        filtered.loc[low_confidence, 'signal_type'] = SignalType.HOLD.value

        return filtered


def calculate_signal_metrics(signals: pd.DataFrame,
                             returns: pd.Series) -> Dict:
    """
    Calculate metrics for signal quality.

    Args:
        signals: DataFrame with signal types
        returns: Series of forward returns

    Returns:
        Dict with signal metrics
    """
    # Align
    common_idx = signals.index.intersection(returns.index)
    signals_aligned = signals.loc[common_idx]
    returns_aligned = returns.loc[common_idx]

    # Long entry performance
    long_entries = signals_aligned['signal_type'] == SignalType.LONG_ENTRY.value
    long_returns = returns_aligned[long_entries]

    # Short entry performance
    short_entries = signals_aligned['signal_type'] == SignalType.SHORT_ENTRY.value
    short_returns = returns_aligned[short_entries]

    metrics = {
        'total_signals': (long_entries | short_entries).sum(),
        'long_signals': long_entries.sum(),
        'short_signals': short_entries.sum(),
        'long_avg_return': long_returns.mean() if len(long_returns) > 0 else 0,
        'short_avg_return': -short_returns.mean() if len(short_returns) > 0 else 0,  # Negate for short
        'long_win_rate': (long_returns > 0).mean() if len(long_returns) > 0 else 0,
        'short_win_rate': (short_returns < 0).mean() if len(short_returns) > 0 else 0,  # Negative is win for short
    }

    return metrics
