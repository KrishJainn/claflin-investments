"""
Market Regime Detection module.

Detects market regimes: trending_up, trending_down, ranging, volatile.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class RegimeSnapshot:
    """Snapshot of market regime at a point in time."""
    timestamp: pd.Timestamp
    regime: MarketRegime
    confidence: float
    trend_strength: float
    volatility_percentile: float
    adx_value: float
    regime_duration: int  # Bars in current regime


class RegimeDetector:
    """
    Detects market regimes using multiple indicators.

    Uses:
    - ADX for trend strength
    - Moving average slope for direction
    - ATR percentile for volatility
    - Price range compression for ranging
    """

    def __init__(self,
                 trend_period: int = 20,
                 volatility_period: int = 20,
                 adx_threshold: float = 25.0,
                 volatility_threshold: float = 0.7):
        """
        Initialize regime detector.

        Args:
            trend_period: Period for trend detection
            volatility_period: Period for volatility calculation
            adx_threshold: ADX above this = trending
            volatility_threshold: Percentile above this = volatile
        """
        self.trend_period = trend_period
        self.volatility_period = volatility_period
        self.adx_threshold = adx_threshold
        self.volatility_threshold = volatility_threshold

    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime.

        Args:
            df: OHLCV DataFrame

        Returns:
            Current market regime
        """
        if len(df) < self.trend_period * 2:
            return MarketRegime.UNKNOWN

        # Calculate indicators
        adx = self._calculate_adx(df)
        trend_direction = self._calculate_trend_direction(df)
        volatility_pct = self._calculate_volatility_percentile(df)

        # Determine regime
        is_trending = adx > self.adx_threshold
        is_volatile = volatility_pct > self.volatility_threshold

        if is_volatile and not is_trending:
            return MarketRegime.VOLATILE
        elif is_trending and trend_direction > 0:
            return MarketRegime.TRENDING_UP
        elif is_trending and trend_direction < 0:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING

    def detect_all_regimes(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime for each bar in DataFrame.

        Args:
            df: OHLCV DataFrame

        Returns:
            Series of MarketRegime values
        """
        regimes = []
        min_period = self.trend_period * 2

        for i in range(len(df)):
            if i < min_period:
                regimes.append(MarketRegime.UNKNOWN)
            else:
                slice_df = df.iloc[:i + 1].copy()
                regime = self.detect_regime(slice_df)
                regimes.append(regime)

        return pd.Series(regimes, index=df.index)

    def get_regime_snapshot(self, df: pd.DataFrame) -> RegimeSnapshot:
        """
        Get detailed regime snapshot.

        Args:
            df: OHLCV DataFrame

        Returns:
            RegimeSnapshot with all details
        """
        if len(df) < self.trend_period * 2:
            return RegimeSnapshot(
                timestamp=df.index[-1] if len(df) > 0 else pd.Timestamp.now(),
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                trend_strength=0.0,
                volatility_percentile=0.0,
                adx_value=0.0,
                regime_duration=0
            )

        adx = self._calculate_adx(df)
        trend_direction = self._calculate_trend_direction(df)
        trend_strength = self._calculate_trend_strength(df)
        volatility_pct = self._calculate_volatility_percentile(df)
        regime = self.detect_regime(df)

        # Calculate confidence
        if regime == MarketRegime.TRENDING_UP or regime == MarketRegime.TRENDING_DOWN:
            confidence = min(1.0, adx / 50.0)  # ADX of 50+ = 100% confidence
        elif regime == MarketRegime.VOLATILE:
            confidence = min(1.0, volatility_pct)
        elif regime == MarketRegime.RANGING:
            confidence = min(1.0, (self.adx_threshold - adx) / self.adx_threshold)
        else:
            confidence = 0.0

        # Calculate regime duration
        all_regimes = self.detect_all_regimes(df)
        duration = self._calculate_regime_duration(all_regimes)

        return RegimeSnapshot(
            timestamp=df.index[-1],
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength * np.sign(trend_direction),
            volatility_percentile=volatility_pct,
            adx_value=adx,
            regime_duration=duration
        )

    def _calculate_adx(self, df: pd.DataFrame) -> float:
        """Calculate ADX (Average Directional Index)."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        period = self.trend_period

        if len(df) < period + 1:
            return 0.0

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed values using Wilder's smoothing
        atr = self._wilder_smooth(tr, period)
        plus_di = 100 * self._wilder_smooth(plus_dm, period) / (atr + 1e-10)
        minus_di = 100 * self._wilder_smooth(minus_dm, period) / (atr + 1e-10)

        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._wilder_smooth(dx, period)

        return adx[-1] if len(adx) > 0 else 0.0

    def _wilder_smooth(self, values: np.ndarray, period: int) -> np.ndarray:
        """Apply Wilder's smoothing."""
        result = np.zeros_like(values)
        result[:period] = np.nan

        if len(values) >= period:
            result[period - 1] = np.mean(values[:period])

            for i in range(period, len(values)):
                result[i] = (result[i - 1] * (period - 1) + values[i]) / period

        return result

    def _calculate_trend_direction(self, df: pd.DataFrame) -> float:
        """
        Calculate trend direction using EMA slope.

        Returns:
            Positive for uptrend, negative for downtrend
        """
        close = df['close'].values
        period = self.trend_period

        if len(close) < period:
            return 0.0

        # Calculate EMA
        ema = pd.Series(close).ewm(span=period, adjust=False).mean().values

        # Return slope direction (last EMA vs earlier)
        lookback = min(5, len(ema) - 1)
        if lookback > 0:
            return ema[-1] - ema[-lookback - 1]
        return 0.0

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength (0 to 1).

        Uses price position relative to moving averages.
        """
        close = df['close'].values
        period = self.trend_period

        if len(close) < period * 2:
            return 0.0

        # Short and long EMAs
        short_ema = pd.Series(close).ewm(span=period, adjust=False).mean().values[-1]
        long_ema = pd.Series(close).ewm(span=period * 2, adjust=False).mean().values[-1]

        # Strength = normalized distance between EMAs
        avg_price = np.mean(close[-period:])
        strength = abs(short_ema - long_ema) / (avg_price + 1e-10)

        return min(1.0, strength * 20)  # Scale to 0-1

    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        """
        Calculate current volatility as percentile of historical.

        Returns:
            Percentile from 0 to 1
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        period = self.volatility_period

        if len(df) < period * 2:
            return 0.5

        # Calculate ATR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # Rolling ATR
        atr_series = pd.Series(tr).rolling(period).mean()
        current_atr = atr_series.iloc[-1]

        # Historical ATR values
        historical_atr = atr_series.dropna().values

        if len(historical_atr) == 0:
            return 0.5

        # Percentile rank
        percentile = np.sum(historical_atr <= current_atr) / len(historical_atr)

        return percentile

    def _calculate_regime_duration(self, regimes: pd.Series) -> int:
        """Calculate how long current regime has lasted."""
        if len(regimes) == 0:
            return 0

        current = regimes.iloc[-1]
        duration = 1

        for i in range(len(regimes) - 2, -1, -1):
            if regimes.iloc[i] == current:
                duration += 1
            else:
                break

        return duration

    def get_regime_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about regime distribution.

        Args:
            df: OHLCV DataFrame

        Returns:
            Dictionary with regime statistics
        """
        regimes = self.detect_all_regimes(df)
        valid_regimes = regimes[regimes != MarketRegime.UNKNOWN]

        if len(valid_regimes) == 0:
            return {'total_bars': len(df), 'valid_bars': 0}

        regime_counts = {}
        for regime in MarketRegime:
            if regime != MarketRegime.UNKNOWN:
                count = (valid_regimes == regime).sum()
                regime_counts[regime.value] = {
                    'count': count,
                    'percentage': count / len(valid_regimes)
                }

        # Calculate average regime duration
        durations = {r: [] for r in MarketRegime if r != MarketRegime.UNKNOWN}
        current_regime = None
        current_duration = 0

        for regime in valid_regimes:
            if regime == current_regime:
                current_duration += 1
            else:
                if current_regime is not None:
                    durations[current_regime].append(current_duration)
                current_regime = regime
                current_duration = 1

        if current_regime is not None:
            durations[current_regime].append(current_duration)

        avg_durations = {
            r.value: np.mean(durations[r]) if durations[r] else 0
            for r in MarketRegime if r != MarketRegime.UNKNOWN
        }

        return {
            'total_bars': len(df),
            'valid_bars': len(valid_regimes),
            'regime_distribution': regime_counts,
            'avg_regime_duration': avg_durations,
            'current_regime': regimes.iloc[-1].value if len(regimes) > 0 else 'unknown'
        }

    def get_regime_transitions(self, df: pd.DataFrame) -> List[Dict]:
        """
        Get regime transition history.

        Args:
            df: OHLCV DataFrame

        Returns:
            List of regime transitions
        """
        regimes = self.detect_all_regimes(df)
        valid_mask = regimes != MarketRegime.UNKNOWN

        transitions = []
        prev_regime = None

        for i, (idx, regime) in enumerate(regimes.items()):
            if not valid_mask.iloc[i]:
                continue

            if regime != prev_regime:
                transitions.append({
                    'timestamp': idx,
                    'from_regime': prev_regime.value if prev_regime else None,
                    'to_regime': regime.value,
                    'bar_index': i
                })
                prev_regime = regime

        return transitions


def detect_regime_for_timestamp(df: pd.DataFrame,
                                 timestamp: pd.Timestamp,
                                 detector: RegimeDetector = None) -> MarketRegime:
    """
    Detect regime at a specific timestamp.

    Args:
        df: OHLCV DataFrame
        timestamp: Timestamp to check
        detector: Optional RegimeDetector instance

    Returns:
        Market regime at that timestamp
    """
    if detector is None:
        detector = RegimeDetector()

    # Slice data up to timestamp (no lookahead)
    if timestamp in df.index:
        slice_df = df.loc[:timestamp]
    else:
        slice_df = df[df.index <= timestamp]

    return detector.detect_regime(slice_df)


def get_regime_at_timestamps(df: pd.DataFrame,
                             timestamps: List[pd.Timestamp],
                             detector: RegimeDetector = None) -> Dict[pd.Timestamp, MarketRegime]:
    """
    Get regimes at multiple timestamps efficiently.

    Args:
        df: OHLCV DataFrame
        timestamps: List of timestamps
        detector: Optional RegimeDetector instance

    Returns:
        Dictionary mapping timestamp to regime
    """
    if detector is None:
        detector = RegimeDetector()

    # Detect all regimes once
    all_regimes = detector.detect_all_regimes(df)

    results = {}
    for ts in timestamps:
        if ts in all_regimes.index:
            results[ts] = all_regimes.loc[ts]
        elif ts < all_regimes.index[0]:
            results[ts] = MarketRegime.UNKNOWN
        else:
            # Find closest previous timestamp
            prev_idx = all_regimes.index[all_regimes.index <= ts]
            if len(prev_idx) > 0:
                results[ts] = all_regimes.loc[prev_idx[-1]]
            else:
                results[ts] = MarketRegime.UNKNOWN

    return results
