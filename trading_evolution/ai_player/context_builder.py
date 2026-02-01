"""
Trade Context Builder for AI Player.

Builds comprehensive context for each trade, capturing:
- Indicator values and their contributions
- Market regime information
- Recent news sentiment
- Historical performance in similar conditions
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeContext:
    """Complete context for a single trade decision."""
    
    # Basic trade info
    symbol: str
    timestamp: datetime
    direction: str  # 'long', 'short', or 'hold'
    
    # Price context
    current_price: float
    high: float
    low: float
    volume: float
    atr: float
    
    # Signal context
    super_indicator_value: float
    signal_strength: float
    signal_confidence: float
    
    # Indicator breakdown
    indicator_values: Dict[str, float] = field(default_factory=dict)
    indicator_contributions: Dict[str, float] = field(default_factory=dict)
    top_contributing_indicators: List[str] = field(default_factory=list)
    
    # Market regime
    market_regime: str = "unknown"  # trending_up, trending_down, ranging, volatile
    volatility_percentile: float = 0.5
    trend_strength: float = 0.0
    
    # News context
    news_sentiment: float = 0.0  # -1 (bearish) to 1 (bullish)
    relevant_headlines: List[str] = field(default_factory=list)
    news_impact_score: float = 0.0
    
    # Historical context
    similar_past_trades_win_rate: float = 0.5
    similar_conditions_count: int = 0
    
    # AI reasoning (filled by SignalInterpreter)
    ai_reasoning: str = ""
    ai_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'direction': self.direction,
            'current_price': self.current_price,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'atr': self.atr,
            'super_indicator_value': self.super_indicator_value,
            'signal_strength': self.signal_strength,
            'signal_confidence': self.signal_confidence,
            'indicator_values': self.indicator_values,
            'indicator_contributions': self.indicator_contributions,
            'top_contributing_indicators': self.top_contributing_indicators,
            'market_regime': self.market_regime,
            'volatility_percentile': self.volatility_percentile,
            'trend_strength': self.trend_strength,
            'news_sentiment': self.news_sentiment,
            'relevant_headlines': self.relevant_headlines,
            'news_impact_score': self.news_impact_score,
            'similar_past_trades_win_rate': self.similar_past_trades_win_rate,
            'similar_conditions_count': self.similar_conditions_count,
            'ai_reasoning': self.ai_reasoning,
            'ai_confidence': self.ai_confidence,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string for storage."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeContext':
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class TradeContextBuilder:
    """Builds comprehensive trade context from market data and indicators."""
    
    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50,
    ):
        """
        Initialize context builder.
        
        Args:
            volatility_window: Window for volatility percentile calculation
            trend_window: Window for trend strength calculation
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self._volatility_history: Dict[str, List[float]] = {}
        self._news_cache: Dict[str, List[Dict]] = {}
    
    def build_context(
        self,
        symbol: str,
        timestamp: datetime,
        ohlcv: Dict[str, float],
        super_indicator_value: float,
        indicator_snapshot: Dict[str, float],
        indicator_weights: Dict[str, float],
        atr: float = 1.0,
        news_data: Optional[List[Dict]] = None,
    ) -> TradeContext:
        """
        Build complete trade context.
        
        Args:
            symbol: Stock ticker
            timestamp: Current timestamp
            ohlcv: OHLCV data dict with keys: open, high, low, close, volume
            super_indicator_value: Current super indicator value
            indicator_snapshot: Current values of all indicators
            indicator_weights: Current weights for each indicator
            atr: Average True Range value
            news_data: Recent news items for the symbol
            
        Returns:
            Complete TradeContext object
        """
        # Determine signal direction and strength
        direction = self._determine_direction(super_indicator_value)
        signal_strength = abs(super_indicator_value)
        signal_confidence = self._calculate_confidence(super_indicator_value, indicator_snapshot)
        
        # Calculate indicator contributions
        contributions = self._calculate_contributions(indicator_snapshot, indicator_weights)
        top_indicators = sorted(contributions.keys(), key=lambda x: abs(contributions[x]), reverse=True)[:5]
        
        # Determine market regime
        regime = self._detect_market_regime(indicator_snapshot)
        volatility_pct = self._calculate_volatility_percentile(symbol, atr)
        trend_strength = self._calculate_trend_strength(indicator_snapshot)
        
        # Process news if available
        news_sentiment = 0.0
        headlines = []
        news_impact = 0.0
        if news_data:
            news_sentiment, headlines, news_impact = self._process_news(news_data)
        
        return TradeContext(
            symbol=symbol,
            timestamp=timestamp,
            direction=direction,
            current_price=ohlcv.get('close', 0.0),
            high=ohlcv.get('high', 0.0),
            low=ohlcv.get('low', 0.0),
            volume=ohlcv.get('volume', 0.0),
            atr=atr,
            super_indicator_value=super_indicator_value,
            signal_strength=signal_strength,
            signal_confidence=signal_confidence,
            indicator_values=indicator_snapshot,
            indicator_contributions=contributions,
            top_contributing_indicators=top_indicators,
            market_regime=regime,
            volatility_percentile=volatility_pct,
            trend_strength=trend_strength,
            news_sentiment=news_sentiment,
            relevant_headlines=headlines[:3],  # Top 3 headlines
            news_impact_score=news_impact,
        )
    
    def _determine_direction(self, si_value: float) -> str:
        """Determine trade direction from super indicator value."""
        if si_value > 0.3:
            return 'long'
        elif si_value < -0.3:
            return 'short'
        return 'hold'
    
    def _calculate_confidence(
        self,
        si_value: float,
        indicators: Dict[str, float]
    ) -> float:
        """Calculate signal confidence based on indicator agreement."""
        if not indicators:
            return 0.5
        
        # Count how many indicators agree with signal direction
        signal_dir = 1 if si_value > 0 else -1 if si_value < 0 else 0
        agreeing = sum(1 for v in indicators.values() if (v > 0) == (signal_dir > 0))
        agreement_ratio = agreeing / len(indicators)
        
        # Combine with signal strength
        strength_factor = min(abs(si_value), 1.0)
        confidence = (agreement_ratio * 0.6) + (strength_factor * 0.4)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _calculate_contributions(
        self,
        indicators: Dict[str, float],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate each indicator's contribution to the signal."""
        contributions = {}
        for name, value in indicators.items():
            weight = weights.get(name, 1.0)
            contributions[name] = value * weight
        return contributions
    
    def _detect_market_regime(self, indicators: Dict[str, float]) -> str:
        """Detect current market regime from indicators."""
        # Use ADX and trend indicators if available
        adx = indicators.get('ADX_14', 25)
        ema_trend = indicators.get('EMA_Trend', 0)
        
        if adx > 40:
            return 'trending_up' if ema_trend > 0 else 'trending_down'
        elif adx > 25:
            return 'mild_trend'
        else:
            return 'ranging'
    
    def _calculate_volatility_percentile(self, symbol: str, current_atr: float) -> float:
        """Calculate where current volatility sits in recent history."""
        if symbol not in self._volatility_history:
            self._volatility_history[symbol] = []
        
        history = self._volatility_history[symbol]
        history.append(current_atr)
        
        # Keep only recent history
        if len(history) > self.volatility_window:
            history.pop(0)
        
        if len(history) < 5:
            return 0.5  # Not enough data
        
        # Calculate percentile
        below = sum(1 for v in history if v < current_atr)
        return below / len(history)
    
    def _calculate_trend_strength(self, indicators: Dict[str, float]) -> float:
        """Calculate trend strength from available indicators."""
        # Use multiple trend indicators if available
        adx = indicators.get('ADX_14', 0) / 100  # Normalize to 0-1
        macd_hist = indicators.get('MACD_Histogram', 0)
        ema_diff = indicators.get('EMA_Trend', 0)
        
        # Combine signals
        trend_signals = [
            adx,
            min(abs(macd_hist) / 2, 1),  # Cap at 1
            min(abs(ema_diff) / 0.02, 1),  # Normalize
        ]
        
        return sum(trend_signals) / len(trend_signals)
    
    def _process_news(self, news_data: List[Dict]) -> tuple:
        """Process news data for sentiment and impact."""
        if not news_data:
            return 0.0, [], 0.0
        
        # Simple sentiment aggregation (will be enhanced by SentimentAnalyzer)
        sentiments = [item.get('sentiment', 0) for item in news_data]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        
        headlines = [item.get('headline', '') for item in news_data]
        
        # Calculate impact based on recency and source importance
        impacts = [item.get('impact', 0.5) for item in news_data]
        max_impact = max(impacts) if impacts else 0.0
        
        return avg_sentiment, headlines, max_impact
    
    def update_news_cache(self, symbol: str, news_items: List[Dict]):
        """Update cached news for a symbol."""
        self._news_cache[symbol] = news_items
    
    def get_cached_news(self, symbol: str) -> List[Dict]:
        """Get cached news for a symbol."""
        return self._news_cache.get(symbol, [])
