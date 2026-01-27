"""
Live Signal Generator module.

Generates trading signals for current market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from ..super_indicator.dna import SuperIndicatorDNA
from ..super_indicator.core import SuperIndicator
from ..super_indicator.signals import SignalGenerator, SignalType, TradingSignal
from ..indicators.calculator import IndicatorCalculator
from ..indicators.normalizer import IndicatorNormalizer
from ..data.fetcher import DataFetcher
from ..data.market_regime import RegimeDetector, MarketRegime
from ..player.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class LiveSignal:
    """Live trading signal with full context."""
    symbol: str
    signal_type: SignalType
    signal_strength: float
    timestamp: datetime
    current_price: float
    entry_price: float
    stop_price: float
    target_price: float
    position_size: int
    risk_amount: float
    market_regime: MarketRegime
    confidence: float
    top_contributors: List[Tuple[str, float]]  # (indicator, contribution)
    notes: str = ""


class LiveSignalGenerator:
    """
    Generates live trading signals using evolved DNA.

    Features:
    - Real-time signal generation
    - Risk-adjusted position sizing
    - Market regime awareness
    - Confidence scoring
    """

    def __init__(self,
                 dna: SuperIndicatorDNA,
                 data_fetcher: DataFetcher = None,
                 indicator_calculator: IndicatorCalculator = None,
                 normalizer: IndicatorNormalizer = None,
                 risk_manager: RiskManager = None,
                 portfolio_value: float = 100000):
        """
        Initialize live signal generator.

        Args:
            dna: Evolved DNA configuration
            data_fetcher: Data fetcher for market data
            indicator_calculator: Indicator calculator
            normalizer: Indicator normalizer
            risk_manager: Risk manager for position sizing
            portfolio_value: Current portfolio value
        """
        self.dna = dna
        self.data_fetcher = data_fetcher or DataFetcher()
        self.calculator = indicator_calculator or IndicatorCalculator()
        self.normalizer = normalizer or IndicatorNormalizer()
        self.risk_manager = risk_manager or RiskManager()
        self.portfolio_value = portfolio_value

        # Build Super Indicator
        self.super_indicator = SuperIndicator(dna)
        self.signal_generator = SignalGenerator()
        self.regime_detector = RegimeDetector()

        # Get active indicators
        self.active_indicators = dna.get_active_indicators()
        logger.info(f"LiveSignalGenerator initialized with {len(self.active_indicators)} active indicators")

    def generate_signals(self, symbols: List[str]) -> List[LiveSignal]:
        """
        Generate signals for all symbols.

        Args:
            symbols: List of symbols to analyze

        Returns:
            List of LiveSignal objects
        """
        signals = []

        for symbol in symbols:
            try:
                signal = self.generate_signal(symbol)
                if signal and signal.signal_type != SignalType.HOLD:
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Error generating signal for {symbol}: {e}")

        # Sort by signal strength
        signals.sort(key=lambda s: abs(s.signal_strength), reverse=True)

        return signals

    def generate_signal(self, symbol: str) -> Optional[LiveSignal]:
        """
        Generate signal for a single symbol.

        Args:
            symbol: Trading symbol

        Returns:
            LiveSignal or None
        """
        # Fetch recent data
        df = self.data_fetcher.fetch(symbol, period='6mo')

        if df is None or len(df) < 100:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        # Calculate indicators
        indicator_values = self.calculator.calculate_all(df)

        # Normalize indicators
        normalized = self.normalizer.normalize_all(
            indicator_values, df,
            include_indicators=self.active_indicators
        )

        if not normalized:
            return None

        # Get Super Indicator value
        si_value = self.super_indicator.calculate(normalized)

        # Generate signal
        signal = self.signal_generator.generate_signal(si_value)

        # Get current price
        current_price = df['close'].iloc[-1]

        # Detect market regime
        regime = self.regime_detector.detect_regime(df)

        # Calculate position sizing
        atr = self._calculate_atr(df)
        stop_distance = atr * 2

        if signal.signal_type == SignalType.LONG_ENTRY:
            stop_price = current_price - stop_distance
            target_price = current_price + (stop_distance * 2)  # 2:1 R:R
        elif signal.signal_type == SignalType.SHORT_ENTRY:
            stop_price = current_price + stop_distance
            target_price = current_price - (stop_distance * 2)
        else:
            stop_price = 0
            target_price = 0

        # Get position size from risk manager
        position = self.risk_manager.calculate_position_size(
            entry_price=current_price,
            stop_price=stop_price,
            portfolio_value=self.portfolio_value
        )

        # Calculate confidence
        confidence = self._calculate_confidence(si_value, regime, signal.signal_type)

        # Get top contributing indicators
        contributors = self._get_top_contributors(normalized)

        return LiveSignal(
            symbol=symbol,
            signal_type=signal.signal_type,
            signal_strength=si_value,
            timestamp=datetime.now(),
            current_price=current_price,
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            position_size=position.shares,
            risk_amount=position.risk_amount,
            market_regime=regime,
            confidence=confidence,
            top_contributors=contributors
        )

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        atr = np.mean(tr[-period:])
        return atr

    def _calculate_confidence(self,
                              si_value: float,
                              regime: MarketRegime,
                              signal_type: SignalType) -> float:
        """Calculate signal confidence."""
        # Base confidence from signal strength
        base_confidence = min(1.0, abs(si_value))

        # Regime adjustment
        regime_multiplier = 1.0
        if signal_type == SignalType.LONG_ENTRY:
            if regime == MarketRegime.TRENDING_UP:
                regime_multiplier = 1.2
            elif regime == MarketRegime.TRENDING_DOWN:
                regime_multiplier = 0.7
        elif signal_type == SignalType.SHORT_ENTRY:
            if regime == MarketRegime.TRENDING_DOWN:
                regime_multiplier = 1.2
            elif regime == MarketRegime.TRENDING_UP:
                regime_multiplier = 0.7

        if regime == MarketRegime.VOLATILE:
            regime_multiplier *= 0.9

        confidence = base_confidence * regime_multiplier
        return min(1.0, confidence)

    def _get_top_contributors(self,
                              normalized: Dict[str, float],
                              top_n: int = 5) -> List[Tuple[str, float]]:
        """Get top contributing indicators to the signal."""
        contributions = []

        for indicator, value in normalized.items():
            if indicator in self.dna.genes:
                weight = self.dna.genes[indicator].weight
                contribution = value * weight
                contributions.append((indicator, contribution))

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        return contributions[:top_n]

    def format_signals_report(self, signals: List[LiveSignal]) -> str:
        """
        Format signals as human-readable report.

        Args:
            signals: List of LiveSignal objects

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            f"SUPER INDICATOR SIGNALS - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 70,
            ""
        ]

        # Group by signal type
        long_entries = [s for s in signals if s.signal_type == SignalType.LONG_ENTRY]
        short_entries = [s for s in signals if s.signal_type == SignalType.SHORT_ENTRY]
        long_exits = [s for s in signals if s.signal_type == SignalType.LONG_EXIT]
        short_exits = [s for s in signals if s.signal_type == SignalType.SHORT_EXIT]

        if long_entries:
            lines.append("LONG ENTRY SIGNALS:")
            for s in long_entries:
                lines.append(
                    f"  {s.symbol:6s}: {s.signal_strength:+.2f} | "
                    f"Entry: ${s.entry_price:.2f} | "
                    f"Stop: ${s.stop_price:.2f} | "
                    f"Target: ${s.target_price:.2f} | "
                    f"Conf: {s.confidence:.0%}"
                )
            lines.append("")

        if short_entries:
            lines.append("SHORT ENTRY SIGNALS:")
            for s in short_entries:
                lines.append(
                    f"  {s.symbol:6s}: {s.signal_strength:+.2f} | "
                    f"Entry: ${s.entry_price:.2f} | "
                    f"Stop: ${s.stop_price:.2f} | "
                    f"Target: ${s.target_price:.2f} | "
                    f"Conf: {s.confidence:.0%}"
                )
            lines.append("")

        if long_exits:
            lines.append("LONG EXIT SIGNALS:")
            for s in long_exits:
                lines.append(f"  {s.symbol:6s}: Exit at {s.signal_strength:+.2f}")
            lines.append("")

        if short_exits:
            lines.append("SHORT EXIT SIGNALS:")
            for s in short_exits:
                lines.append(f"  {s.symbol:6s}: Cover at {s.signal_strength:+.2f}")
            lines.append("")

        if not any([long_entries, short_entries, long_exits, short_exits]):
            lines.append("No actionable signals at this time.")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def get_signal_details(self, signal: LiveSignal) -> str:
        """
        Get detailed breakdown of a signal.

        Args:
            signal: LiveSignal object

        Returns:
            Detailed signal information
        """
        lines = [
            f"Signal Details: {signal.symbol}",
            "-" * 40,
            f"Signal Type:      {signal.signal_type.value}",
            f"Signal Strength:  {signal.signal_strength:+.3f}",
            f"Confidence:       {signal.confidence:.1%}",
            f"Market Regime:    {signal.market_regime.value}",
            "",
            f"Entry Price:      ${signal.entry_price:.2f}",
            f"Stop Loss:        ${signal.stop_price:.2f}",
            f"Target:           ${signal.target_price:.2f}",
            f"Position Size:    {signal.position_size} shares",
            f"Risk Amount:      ${signal.risk_amount:.2f}",
            "",
            "Top Contributing Indicators:",
        ]

        for indicator, contribution in signal.top_contributors:
            direction = "+" if contribution > 0 else ""
            lines.append(f"  {indicator:30s}: {direction}{contribution:.3f}")

        return "\n".join(lines)


def generate_daily_signals(dna: SuperIndicatorDNA,
                           symbols: List[str],
                           portfolio_value: float = 100000) -> str:
    """
    Generate and format daily trading signals.

    Args:
        dna: Evolved DNA configuration
        symbols: List of symbols to analyze
        portfolio_value: Current portfolio value

    Returns:
        Formatted signals report
    """
    generator = LiveSignalGenerator(
        dna=dna,
        portfolio_value=portfolio_value
    )

    signals = generator.generate_signals(symbols)
    return generator.format_signals_report(signals)
