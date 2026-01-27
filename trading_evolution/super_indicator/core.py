"""
Super Indicator Core module.

The SuperIndicator combines multiple technical indicators into a single
signal ranging from -1 (strong bearish) to +1 (strong bullish).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass

from .dna import SuperIndicatorDNA
from ..indicators.normalizer import IndicatorNormalizer, SignalAggregator


@dataclass
class SignalOutput:
    """Output from Super Indicator calculation."""
    value: float  # Signal value in [-1, 1]
    direction: str  # 'LONG', 'SHORT', 'NEUTRAL'
    strength: str  # 'STRONG', 'MODERATE', 'WEAK'
    confidence: float  # Confidence in signal (0-1)

    @classmethod
    def from_value(cls, value: float) -> 'SignalOutput':
        """Create SignalOutput from raw value."""
        # Direction
        if value > 0.3:
            direction = 'LONG'
        elif value < -0.3:
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'

        # Strength
        abs_val = abs(value)
        if abs_val > 0.7:
            strength = 'STRONG'
        elif abs_val > 0.4:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'

        # Confidence (how far from neutral)
        confidence = min(1.0, abs_val / 0.7)

        return cls(
            value=value,
            direction=direction,
            strength=strength,
            confidence=confidence
        )


class SuperIndicator:
    """
    Super Indicator: Combines normalized indicators into a single signal.

    The Super Indicator:
    1. Takes normalized indicator values [-1, 1]
    2. Applies learned weights to each indicator
    3. Aggregates into single signal [-1, 1]
    4. Provides trading signals (LONG/SHORT entry/exit)

    Signal Interpretation:
    - > +0.7: Strong buy signal (LONG ENTRY)
    - +0.3 to +0.7: Bullish but not entry
    - -0.3 to +0.3: Neutral
    - -0.7 to -0.3: Bearish but not entry
    - < -0.7: Strong sell signal (SHORT ENTRY)
    """

    # Signal thresholds
    LONG_ENTRY_THRESHOLD = 0.7
    LONG_EXIT_THRESHOLD = 0.3
    SHORT_ENTRY_THRESHOLD = -0.7
    SHORT_EXIT_THRESHOLD = -0.3

    def __init__(self, dna: SuperIndicatorDNA,
                 normalizer: IndicatorNormalizer = None):
        """
        Initialize Super Indicator.

        Args:
            dna: DNA configuration with indicator weights
            normalizer: Normalizer for indicator values
        """
        self.dna = dna
        self.normalizer = normalizer or IndicatorNormalizer()
        self._weights = dna.get_weights()
        self._active_indicators = dna.get_active_indicators()

    def calculate(self, normalized_indicators: pd.DataFrame) -> pd.Series:
        """
        Calculate Super Indicator signal for entire series.

        Args:
            normalized_indicators: DataFrame with normalized indicator values [-1, 1]

        Returns:
            Series with signal values [-1, 1]
        """
        return SignalAggregator.weighted_average(
            normalized_indicators,
            self._weights
        )

    def calculate_at_timestamp(self, normalized_indicators: pd.DataFrame,
                               timestamp: pd.Timestamp) -> float:
        """
        Calculate signal at a specific timestamp.

        Args:
            normalized_indicators: DataFrame with normalized values
            timestamp: Point in time to calculate

        Returns:
            Signal value [-1, 1]
        """
        return SignalAggregator.weighted_at_timestamp(
            normalized_indicators,
            self._weights,
            timestamp
        )

    def get_signal(self, normalized_indicators: pd.DataFrame,
                   timestamp: pd.Timestamp) -> SignalOutput:
        """
        Get full signal output at timestamp.

        Args:
            normalized_indicators: DataFrame with normalized values
            timestamp: Point in time

        Returns:
            SignalOutput with direction, strength, confidence
        """
        value = self.calculate_at_timestamp(normalized_indicators, timestamp)
        return SignalOutput.from_value(value)

    def calculate_from_raw(self, indicators_df: pd.DataFrame,
                           price_series: pd.Series) -> pd.Series:
        """
        Calculate from raw (unnormalized) indicator values.

        Args:
            indicators_df: Raw indicator values
            price_series: Close prices for price-relative normalization

        Returns:
            Series with signal values [-1, 1]
        """
        # Normalize first
        normalized = self.normalizer.normalize_all(
            indicators_df,
            price_series=price_series
        )

        return self.calculate(normalized)

    def get_contributing_indicators(self, normalized_indicators: pd.DataFrame,
                                    timestamp: pd.Timestamp,
                                    top_n: int = 5) -> List[Dict]:
        """
        Get top contributing indicators at a timestamp.

        Useful for understanding what's driving the signal.

        Args:
            normalized_indicators: DataFrame with normalized values
            timestamp: Point in time
            top_n: Number of top contributors to return

        Returns:
            List of dicts with indicator name, value, weight, contribution
        """
        if timestamp not in normalized_indicators.index:
            return []

        contributions = []
        for name, weight in self._weights.items():
            if name not in normalized_indicators.columns:
                continue

            value = normalized_indicators.loc[timestamp, name]
            if pd.isna(value):
                continue

            contribution = value * weight
            contributions.append({
                'indicator': name,
                'value': float(value),
                'weight': weight,
                'contribution': float(contribution),
                'direction': 'BULLISH' if contribution > 0 else 'BEARISH'
            })

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)

        return contributions[:top_n]

    def get_signal_strength_breakdown(self, normalized_indicators: pd.DataFrame,
                                      timestamp: pd.Timestamp) -> Dict:
        """
        Get breakdown of signal strength by indicator category.

        Args:
            normalized_indicators: DataFrame with normalized values
            timestamp: Point in time

        Returns:
            Dict with category -> contribution
        """
        if timestamp not in normalized_indicators.index:
            return {}

        category_contributions = {}

        for name, gene in self.dna.genes.items():
            if not gene.active or gene.weight == 0:
                continue
            if name not in normalized_indicators.columns:
                continue

            value = normalized_indicators.loc[timestamp, name]
            if pd.isna(value):
                continue

            category = gene.category or 'other'
            contribution = value * gene.weight

            if category not in category_contributions:
                category_contributions[category] = 0.0
            category_contributions[category] += contribution

        return category_contributions

    def update_dna(self, new_dna: SuperIndicatorDNA):
        """Update the DNA configuration."""
        self.dna = new_dna
        self._weights = new_dna.get_weights()
        self._active_indicators = new_dna.get_active_indicators()

    @property
    def num_active_indicators(self) -> int:
        """Number of active indicators."""
        return len(self._active_indicators)

    @property
    def active_indicators(self) -> List[str]:
        """List of active indicator names."""
        return self._active_indicators.copy()

    @property
    def weights(self) -> Dict[str, float]:
        """Current weight configuration."""
        return self._weights.copy()


def create_super_indicator(weights: Dict[str, float],
                           normalizer: IndicatorNormalizer = None) -> SuperIndicator:
    """
    Quick factory function to create Super Indicator from weights.

    Args:
        weights: Dict of indicator_name -> weight
        normalizer: Optional normalizer

    Returns:
        Configured SuperIndicator
    """
    from .dna import create_dna_from_weights

    dna = create_dna_from_weights(weights)
    return SuperIndicator(dna, normalizer)
