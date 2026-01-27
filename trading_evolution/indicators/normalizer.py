"""
Indicator Normalization module.

Normalizes all indicators to a consistent -1 to +1 range using various methods:
- Bounded indicators: Direct scaling based on known ranges
- Unbounded indicators: Adaptive z-score with tanh squashing
- Price-relative indicators: Percentage deviation normalization
- Volume indicators: Percentile rank normalization

CRITICAL: Uses expanding windows to prevent lookahead bias.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from scipy import stats

from .universe import IndicatorDefinition, IndicatorUniverse


class NormalizationMethod(Enum):
    """Available normalization methods."""
    BOUNDED = "bounded"       # For indicators with known min/max
    ADAPTIVE = "adaptive"     # Adaptive z-score with tanh
    PERCENTILE = "percentile" # Percentile rank
    PRICE_RELATIVE = "price_relative"  # % deviation from price


@dataclass
class NormalizationConfig:
    """Configuration for indicator normalization."""
    default_method: NormalizationMethod = NormalizationMethod.ADAPTIVE
    min_periods: int = 20  # Minimum periods before normalization
    clip_outliers: bool = True
    outlier_std: float = 3.0  # Clip at N standard deviations
    tanh_scale: float = 0.5  # Scale factor for tanh squashing
    output_range: Tuple[float, float] = (-1.0, 1.0)


class IndicatorNormalizer:
    """
    Normalizes technical indicators to a consistent -1 to +1 range.

    Key Features:
    - Uses EXPANDING windows to prevent lookahead bias
    - Automatic method selection based on indicator properties
    - Handles bounded, unbounded, and price-relative indicators
    - Smooth output via tanh squashing

    Strategy by indicator type:
    1. Bounded (RSI, Stoch): Direct linear scaling
    2. Unbounded (MACD, ROC): Adaptive z-score + tanh
    3. Overlap/MA: % deviation from close, then adaptive
    4. Volume: Percentile rank (0-1) mapped to (-1, 1)
    """

    def __init__(self, config: NormalizationConfig = None,
                 universe: IndicatorUniverse = None):
        """
        Initialize normalizer.

        Args:
            config: Normalization configuration
            universe: Indicator universe for metadata
        """
        self.config = config or NormalizationConfig()
        self.universe = universe or IndicatorUniverse()
        self.universe.load_all()

        # Cache for expanding statistics
        self._stats_cache: Dict[str, Dict] = {}

    def normalize_all(self, indicators_df: pd.DataFrame,
                      price_series: pd.Series = None) -> pd.DataFrame:
        """
        Normalize all indicators in a DataFrame.

        CRITICAL: Uses only historical data at each point (no lookahead).

        Args:
            indicators_df: DataFrame with indicator values
            price_series: Close prices (needed for price-relative normalization)

        Returns:
            DataFrame with normalized values in [-1, 1] range
        """
        normalized = pd.DataFrame(index=indicators_df.index)

        for col in indicators_df.columns:
            try:
                # Get indicator definition for metadata
                defn = self._find_definition(col)

                # Determine normalization method
                method = self._select_method(col, defn)

                # Normalize based on method
                if method == NormalizationMethod.BOUNDED and defn and defn.natural_range:
                    normalized[col] = self._normalize_bounded(
                        indicators_df[col], defn.natural_range
                    )
                elif method == NormalizationMethod.PRICE_RELATIVE and price_series is not None:
                    normalized[col] = self._normalize_price_relative(
                        indicators_df[col], price_series
                    )
                elif method == NormalizationMethod.PERCENTILE:
                    normalized[col] = self._normalize_percentile(indicators_df[col])
                else:
                    normalized[col] = self._normalize_adaptive(indicators_df[col])

            except Exception as e:
                # On error, return zeros
                normalized[col] = 0.0

        return normalized

    def normalize_single(self, series: pd.Series, indicator_name: str,
                         price_series: pd.Series = None) -> pd.Series:
        """Normalize a single indicator series."""
        defn = self._find_definition(indicator_name)
        method = self._select_method(indicator_name, defn)

        if method == NormalizationMethod.BOUNDED and defn and defn.natural_range:
            return self._normalize_bounded(series, defn.natural_range)
        elif method == NormalizationMethod.PRICE_RELATIVE and price_series is not None:
            return self._normalize_price_relative(series, price_series)
        elif method == NormalizationMethod.PERCENTILE:
            return self._normalize_percentile(series)
        else:
            return self._normalize_adaptive(series)

    def normalize_at_timestamp(self, series: pd.Series, indicator_name: str,
                               timestamp: pd.Timestamp) -> float:
        """
        Normalize indicator at a specific point in time.

        CRITICAL: Only uses data available up to timestamp (no lookahead).

        Args:
            series: Full indicator series
            indicator_name: Name of indicator
            timestamp: Point in time to normalize

        Returns:
            Normalized value in [-1, 1]
        """
        if timestamp not in series.index:
            return 0.0

        # Get historical data only
        historical = series.loc[:timestamp]

        if len(historical) < self.config.min_periods:
            return 0.0

        current_value = historical.iloc[-1]
        if pd.isna(current_value):
            return 0.0

        # Get definition for metadata
        defn = self._find_definition(indicator_name)

        # For bounded indicators, use direct scaling
        if defn and defn.natural_range:
            return self._scale_bounded_value(current_value, defn.natural_range)

        # For unbounded, use expanding statistics
        history = historical.iloc[:-1]  # Exclude current
        if len(history) < self.config.min_periods:
            return 0.0

        mean = history.mean()
        std = history.std()

        if std == 0 or pd.isna(std):
            return 0.0

        z_score = (current_value - mean) / std

        # Clip outliers
        if self.config.clip_outliers:
            z_score = np.clip(z_score, -self.config.outlier_std, self.config.outlier_std)

        # Tanh squashing
        normalized = np.tanh(z_score * self.config.tanh_scale)

        return float(np.clip(normalized, -1.0, 1.0))

    def _find_definition(self, col_name: str) -> Optional[IndicatorDefinition]:
        """Find indicator definition matching column name."""
        # Try exact match first
        defn = self.universe.get_definition(col_name)
        if defn:
            return defn

        # Try to match base name (e.g., 'RSI_14' matches 'RSI_14')
        for name in self.universe.get_all():
            if col_name.startswith(name.split('_')[0]):
                return self.universe.get_definition(name)

        return None

    def _select_method(self, col_name: str,
                       defn: Optional[IndicatorDefinition]) -> NormalizationMethod:
        """Select appropriate normalization method for indicator."""
        if defn:
            # Bounded indicators
            if defn.natural_range is not None:
                return NormalizationMethod.BOUNDED

            # Volume indicators use percentile
            if defn.category == 'volume':
                return NormalizationMethod.PERCENTILE

            # Overlap/MA indicators use price-relative
            if defn.category == 'overlap':
                return NormalizationMethod.PRICE_RELATIVE

        # Default to adaptive
        return self.config.default_method

    def _normalize_bounded(self, series: pd.Series,
                           natural_range: Tuple[float, float]) -> pd.Series:
        """
        Normalize bounded indicator to [-1, 1].

        RSI [0, 100] -> [-1, +1]
        Williams %R [-100, 0] -> [-1, +1]
        Stochastic [0, 100] -> [-1, +1]
        """
        min_val, max_val = natural_range

        # Linear scaling to [-1, 1]
        midpoint = (max_val + min_val) / 2
        half_range = (max_val - min_val) / 2

        if half_range == 0:
            return pd.Series(0.0, index=series.index)

        normalized = (series - midpoint) / half_range

        return normalized.clip(-1.0, 1.0)

    def _scale_bounded_value(self, value: float,
                             natural_range: Tuple[float, float]) -> float:
        """Scale a single bounded value."""
        min_val, max_val = natural_range
        midpoint = (max_val + min_val) / 2
        half_range = (max_val - min_val) / 2

        if half_range == 0:
            return 0.0

        normalized = (value - midpoint) / half_range
        return float(np.clip(normalized, -1.0, 1.0))

    def _normalize_adaptive(self, series: pd.Series) -> pd.Series:
        """
        Normalize unbounded indicator using adaptive z-score.

        Uses EXPANDING window to prevent lookahead bias.
        Applies tanh squashing for smooth [-1, 1] output.
        """
        # Use expanding statistics (only past data)
        expanding_mean = series.expanding(min_periods=self.config.min_periods).mean()
        expanding_std = series.expanding(min_periods=self.config.min_periods).std()

        # Avoid division by zero
        expanding_std = expanding_std.replace(0, np.nan)

        # Z-score
        z_score = (series - expanding_mean) / expanding_std

        # Clip outliers
        if self.config.clip_outliers:
            z_score = z_score.clip(-self.config.outlier_std, self.config.outlier_std)

        # Tanh squashing for smooth output
        normalized = np.tanh(z_score * self.config.tanh_scale)

        return normalized.clip(-1.0, 1.0)

    def _normalize_percentile(self, series: pd.Series) -> pd.Series:
        """
        Normalize using percentile rank (for volume indicators).

        Uses expanding window to avoid lookahead.
        Maps [0, 1] percentile to [-1, 1].
        """
        def expanding_percentile(x):
            if len(x) < self.config.min_periods:
                return 0.5  # Neutral
            # Percentile of current value vs history
            current = x.iloc[-1]
            history = x.iloc[:-1]
            if len(history) < 2:
                return 0.5
            return stats.percentileofscore(history, current) / 100

        percentiles = series.expanding(min_periods=self.config.min_periods).apply(
            expanding_percentile, raw=False
        )

        # Map [0, 1] to [-1, 1]
        normalized = (percentiles - 0.5) * 2

        return normalized.clip(-1.0, 1.0)

    def _normalize_price_relative(self, indicator_series: pd.Series,
                                   price_series: pd.Series) -> pd.Series:
        """
        Normalize price-relative indicators (moving averages).

        Calculates % deviation from price, then normalizes adaptively.
        """
        # Calculate percentage deviation from price
        pct_deviation = (indicator_series - price_series) / price_series

        # Apply adaptive normalization to the deviation
        return self._normalize_adaptive(pct_deviation)


class SignalAggregator:
    """
    Aggregates normalized indicator signals into Super Indicator output.
    """

    @staticmethod
    def weighted_average(normalized_df: pd.DataFrame,
                         weights: Dict[str, float]) -> pd.Series:
        """
        Calculate weighted average of normalized indicators.

        Args:
            normalized_df: DataFrame with normalized values [-1, 1]
            weights: Dict of indicator_name -> weight (can be negative for inverse)

        Returns:
            Series with aggregated signal values [-1, 1]
        """
        weighted_sum = pd.Series(0.0, index=normalized_df.index)
        total_weight = 0.0

        for name, weight in weights.items():
            if name not in normalized_df.columns:
                continue
            if weight == 0:
                continue

            values = normalized_df[name].fillna(0)
            weighted_sum += values * weight
            total_weight += abs(weight)

        if total_weight == 0:
            return pd.Series(0.0, index=normalized_df.index)

        # Raw weighted average
        raw_signal = weighted_sum / total_weight

        # Final tanh squashing for smooth output
        # Scale factor of 1.5 makes average signal of 0.5 map to ~0.64
        signal = np.tanh(raw_signal * 1.5)

        return signal.clip(-1.0, 1.0)

    @staticmethod
    def weighted_at_timestamp(normalized_df: pd.DataFrame,
                              weights: Dict[str, float],
                              timestamp: pd.Timestamp) -> float:
        """Calculate weighted signal at a specific timestamp."""
        if timestamp not in normalized_df.index:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for name, weight in weights.items():
            if name not in normalized_df.columns:
                continue
            if weight == 0:
                continue

            value = normalized_df.loc[timestamp, name]
            if pd.isna(value):
                continue

            weighted_sum += value * weight
            total_weight += abs(weight)

        if total_weight == 0:
            return 0.0

        raw_signal = weighted_sum / total_weight
        signal = np.tanh(raw_signal * 1.5)

        return float(np.clip(signal, -1.0, 1.0))
