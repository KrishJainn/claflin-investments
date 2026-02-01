"""
AI Signal Interpreter for the Player-Coach Trading System.

Uses LLM to interpret trading signals when confidence is low,
providing reasoning and adjusted confidence scores.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging
import json

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .context_builder import TradeContext
from ..ai_config import AIConfig, DEFAULT_AI_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class InterpretedSignal:
    """Result of AI signal interpretation."""
    
    original_direction: str  # Original signal direction
    interpreted_direction: str  # AI-adjusted direction
    original_confidence: float
    ai_confidence: float
    reasoning: str
    should_trade: bool
    risk_assessment: str  # 'low', 'medium', 'high'
    suggested_position_size: float  # 0.0 to 1.0 multiplier


class SignalInterpreter:
    """AI-powered signal interpretation using Gemini."""
    
    INTERPRETATION_PROMPT = """You are an expert algorithmic trader analyzing a trading signal.

## Current Trade Context
- **Symbol**: {symbol}
- **Time**: {timestamp}
- **Current Price**: ${current_price:.2f}
- **Signal Direction**: {direction}
- **Signal Strength**: {signal_strength:.2f} (0-1 scale)
- **Signal Confidence**: {signal_confidence:.2f} (0-1 scale)

## Market Conditions
- **Market Regime**: {market_regime}
- **Volatility Percentile**: {volatility_percentile:.0%}
- **Trend Strength**: {trend_strength:.2f}

## Top Contributing Indicators
{indicator_summary}

## Recent News Sentiment
- **Overall Sentiment**: {news_sentiment:.2f} (-1 bearish to +1 bullish)
- **News Impact Score**: {news_impact_score:.2f}
- **Headlines**: {headlines}

## Your Task
Analyze this trading signal and provide:
1. Whether to TAKE or SKIP this trade
2. Your confidence level (0.0 to 1.0)
3. Risk assessment (low/medium/high)
4. Brief reasoning (2-3 sentences max)

Respond ONLY in this JSON format:
{{
    "should_trade": true/false,
    "adjusted_direction": "long"/"short"/"hold",
    "confidence": 0.0-1.0,
    "risk": "low"/"medium"/"high",
    "position_size_multiplier": 0.0-1.0,
    "reasoning": "Your brief reasoning here"
}}
"""
    
    def __init__(self, config: AIConfig = None):
        """
        Initialize signal interpreter.
        
        Args:
            config: AI configuration
        """
        self.config = config or DEFAULT_AI_CONFIG
        self._client = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize the LLM client."""
        if not GENAI_AVAILABLE:
            logger.warning("google-genai not installed, LLM interpretation disabled")
            return
        
        if not self.config.llm.api_key:
            logger.warning("GOOGLE_API_KEY not set, LLM interpretation disabled")
            return
        
        try:
            self._client = genai.Client(api_key=self.config.llm.api_key)
            logger.info(f"Initialized Gemini LLM: {self.config.llm.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self._client = None
    
    def interpret(self, context: TradeContext) -> InterpretedSignal:
        """
        Interpret a trading signal using AI.
        
        Args:
            context: Complete trade context
            
        Returns:
            InterpretedSignal with AI reasoning and adjusted confidence
        """
        # If LLM not available or confidence is high, use rule-based interpretation
        if (not self._client or 
            context.signal_confidence >= self.config.player.signal_confidence_threshold):
            return self._rule_based_interpretation(context)
        
        try:
            return self._llm_interpretation(context)
        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}")
            return self._rule_based_interpretation(context)
    
    def _llm_interpretation(self, context: TradeContext) -> InterpretedSignal:
        """Use LLM to interpret the signal."""
        # Build prompt
        indicator_summary = self._format_indicators(context)
        headlines = ", ".join(context.relevant_headlines) if context.relevant_headlines else "None"
        
        prompt = self.INTERPRETATION_PROMPT.format(
            symbol=context.symbol,
            timestamp=context.timestamp.strftime("%Y-%m-%d %H:%M"),
            current_price=context.current_price,
            direction=context.direction,
            signal_strength=context.signal_strength,
            signal_confidence=context.signal_confidence,
            market_regime=context.market_regime,
            volatility_percentile=context.volatility_percentile,
            trend_strength=context.trend_strength,
            indicator_summary=indicator_summary,
            news_sentiment=context.news_sentiment,
            news_impact_score=context.news_impact_score,
            headlines=headlines,
        )
        
        # Call LLM
        response = self._client.models.generate_content(
            model=self.config.llm.model_name,
            contents=prompt,
            config={
                "temperature": self.config.llm.temperature,
                "max_output_tokens": 512,
            }
        )
        
        # Parse response
        return self._parse_llm_response(response.text, context)
    
    def _parse_llm_response(
        self,
        response_text: str,
        context: TradeContext
    ) -> InterpretedSignal:
        """Parse LLM response into InterpretedSignal."""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                return InterpretedSignal(
                    original_direction=context.direction,
                    interpreted_direction=data.get('adjusted_direction', context.direction),
                    original_confidence=context.signal_confidence,
                    ai_confidence=float(data.get('confidence', 0.5)),
                    reasoning=data.get('reasoning', 'No reasoning provided'),
                    should_trade=bool(data.get('should_trade', False)),
                    risk_assessment=data.get('risk', 'medium'),
                    suggested_position_size=float(data.get('position_size_multiplier', 1.0)),
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        
        # Fallback to rule-based
        return self._rule_based_interpretation(context)
    
    def _rule_based_interpretation(self, context: TradeContext) -> InterpretedSignal:
        """Rule-based signal interpretation fallback."""
        # Simple rules based on signal strength and confidence
        should_trade = (
            context.signal_strength > 0.5 and 
            context.signal_confidence > 0.4
        )
        
        # Adjust for news
        if context.news_impact_score > 0.8:
            # High impact news - be more cautious
            should_trade = should_trade and context.signal_confidence > 0.6
        
        # Risk assessment based on volatility
        if context.volatility_percentile > 0.8:
            risk = 'high'
            position_size = 0.5
        elif context.volatility_percentile > 0.5:
            risk = 'medium'
            position_size = 0.75
        else:
            risk = 'low'
            position_size = 1.0
        
        reasoning = (
            f"Rule-based: Signal strength {context.signal_strength:.2f}, "
            f"confidence {context.signal_confidence:.2f}, "
            f"volatility {context.volatility_percentile:.0%}"
        )
        
        return InterpretedSignal(
            original_direction=context.direction,
            interpreted_direction=context.direction,
            original_confidence=context.signal_confidence,
            ai_confidence=context.signal_confidence,
            reasoning=reasoning,
            should_trade=should_trade,
            risk_assessment=risk,
            suggested_position_size=position_size,
        )
    
    def _format_indicators(self, context: TradeContext) -> str:
        """Format top indicators for prompt."""
        lines = []
        for indicator in context.top_contributing_indicators[:5]:
            value = context.indicator_values.get(indicator, 0)
            contribution = context.indicator_contributions.get(indicator, 0)
            direction = "bullish" if contribution > 0 else "bearish"
            lines.append(f"- {indicator}: {value:.3f} ({direction}, contribution: {abs(contribution):.3f})")
        return "\n".join(lines) if lines else "No indicator data available"
    
    def should_use_llm(self, context: TradeContext) -> bool:
        """Determine if LLM interpretation should be used."""
        if not self._client:
            return False
        if not self.config.player.use_llm_for_uncertain_signals:
            return False
        return context.signal_confidence < self.config.player.signal_confidence_threshold
