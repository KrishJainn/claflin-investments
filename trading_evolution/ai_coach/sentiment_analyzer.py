"""
Sentiment Analyzer for the AI Coach.

Analyzes news sentiment using Gemini LLM to determine:
- Overall sentiment (-1 to 1)
- Impact level (low/medium/high/critical)
- Relevance to specific stocks
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import json

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .news_ingester import NewsItem
from ..ai_config import AIConfig, DEFAULT_AI_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis for a news item."""
    
    news_id: str
    sentiment_score: float  # -1 (very bearish) to 1 (very bullish)
    impact_level: str  # low, medium, high, critical
    confidence: float  # 0-1 confidence in analysis
    affected_symbols: List[str]
    reasoning: str
    trading_implications: str


class SentimentAnalyzer:
    """Analyzes news sentiment using LLM."""
    
    SENTIMENT_PROMPT = """You are a financial analyst assessing market news sentiment for Indian stock trading.

## News Item
**Headline**: {headline}
**Source**: {source}
**Published**: {published}
**Summary**: {summary}

## Watchlist Symbols
{symbols}

## Task
Analyze this news and provide:
1. **Sentiment Score**: -1.0 (very bearish) to +1.0 (very bullish)
2. **Impact Level**: critical/high/medium/low
3. **Affected Symbols**: Which symbols from the watchlist are affected
4. **Trading Implications**: Brief actionable insight (1 sentence)

Impact Level Guidelines:
- CRITICAL: Fed/RBI rate decisions, major earnings miss/beat, bankruptcies
- HIGH: Earnings reports, major acquisitions, regulatory changes
- MEDIUM: Analyst upgrades/downgrades, sector news
- LOW: General market commentary, minor news

Respond ONLY in this JSON format:
{{
    "sentiment": -1.0 to 1.0,
    "impact": "critical"/"high"/"medium"/"low",
    "confidence": 0.0 to 1.0,
    "affected_symbols": ["SYMBOL1.NS", "SYMBOL2.NS"],
    "reasoning": "Brief explanation",
    "trading_implications": "Actionable insight"
}}
"""
    
    BATCH_PROMPT = """Analyze the following {count} news items for market sentiment.
For each item, provide sentiment (-1 to 1), impact level, and key takeaway.

News Items:
{news_items}

Respond with a JSON array of results:
[
    {{"id": "news_id", "sentiment": 0.5, "impact": "medium", "takeaway": "Brief insight"}},
    ...
]
"""
    
    def __init__(self, config: AIConfig = None):
        """
        Initialize sentiment analyzer.
        
        Args:
            config: AI configuration
        """
        self.config = config or DEFAULT_AI_CONFIG
        self._client = None
        self._init_llm()
        
        # Cache for analyzed items
        self._analyzed_cache: Dict[str, SentimentResult] = {}
    
    def _init_llm(self):
        """Initialize LLM client."""
        if not GENAI_AVAILABLE:
            logger.warning("google-genai not installed, LLM sentiment disabled")
            return
        
        if not self.config.llm.api_key:
            logger.warning("GOOGLE_API_KEY not set, LLM sentiment disabled")
            return
        
        try:
            self._client = genai.Client(api_key=self.config.llm.api_key)
            logger.info("Initialized Gemini for sentiment analysis")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self._client = None
    
    def analyze(
        self,
        news_item: NewsItem,
        watchlist_symbols: List[str] = None,
    ) -> SentimentResult:
        """
        Analyze sentiment of a single news item.
        
        Args:
            news_item: NewsItem to analyze
            watchlist_symbols: List of symbols to check for relevance
            
        Returns:
            SentimentResult with analysis
        """
        # Check cache
        if news_item.id in self._analyzed_cache:
            return self._analyzed_cache[news_item.id]
        
        # Use LLM if available
        if self._client:
            try:
                result = self._llm_analyze(news_item, watchlist_symbols or [])
                self._analyzed_cache[news_item.id] = result
                return result
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
        
        # Fallback to rule-based
        result = self._rule_based_analyze(news_item)
        self._analyzed_cache[news_item.id] = result
        return result
    
    def analyze_batch(
        self,
        news_items: List[NewsItem],
        watchlist_symbols: List[str] = None,
    ) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple news items.
        
        Args:
            news_items: List of NewsItems to analyze
            watchlist_symbols: List of symbols to check for relevance
            
        Returns:
            List of SentimentResults
        """
        results = []
        
        for item in news_items:
            result = self.analyze(item, watchlist_symbols)
            results.append(result)
        
        return results
    
    def _llm_analyze(
        self,
        news_item: NewsItem,
        watchlist_symbols: List[str],
    ) -> SentimentResult:
        """Analyze using LLM."""
        symbols_str = "\n".join([f"- {s}" for s in watchlist_symbols[:20]])
        
        prompt = self.SENTIMENT_PROMPT.format(
            headline=news_item.headline,
            source=news_item.source,
            published=news_item.published_at.strftime("%Y-%m-%d %H:%M"),
            summary=news_item.summary or "No summary available",
            symbols=symbols_str or "No specific symbols",
        )
        
        response = self._client.models.generate_content(
            model=self.config.llm.model_name,
            contents=prompt,
            config={
                "temperature": 0.2,  # Low for consistent analysis
                "max_output_tokens": 512,
            }
        )
        
        return self._parse_llm_response(response.text, news_item)
    
    def _parse_llm_response(
        self,
        response_text: str,
        news_item: NewsItem,
    ) -> SentimentResult:
        """Parse LLM response into SentimentResult."""
        try:
            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                return SentimentResult(
                    news_id=news_item.id,
                    sentiment_score=float(data.get('sentiment', 0)),
                    impact_level=data.get('impact', 'medium'),
                    confidence=float(data.get('confidence', 0.5)),
                    affected_symbols=data.get('affected_symbols', []),
                    reasoning=data.get('reasoning', 'No reasoning'),
                    trading_implications=data.get('trading_implications', ''),
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        
        return self._rule_based_analyze(news_item)
    
    def _rule_based_analyze(self, news_item: NewsItem) -> SentimentResult:
        """Rule-based sentiment analysis fallback."""
        headline_lower = news_item.headline.lower()
        
        # Positive keywords
        positive = ['surge', 'gain', 'rise', 'beat', 'profit', 'growth', 
                   'upgrade', 'bullish', 'record', 'strong', 'exceeds']
        
        # Negative keywords
        negative = ['fall', 'drop', 'loss', 'miss', 'decline', 'weak',
                   'downgrade', 'bearish', 'concern', 'risk', 'warning']
        
        # Critical keywords
        critical = ['crash', 'crisis', 'halt', 'suspended', 'fraud', 
                   'bankruptcy', 'rate hike', 'rate cut']
        
        # Count matches
        pos_count = sum(1 for w in positive if w in headline_lower)
        neg_count = sum(1 for w in negative if w in headline_lower)
        crit_count = sum(1 for w in critical if w in headline_lower)
        
        # Calculate sentiment
        sentiment = (pos_count - neg_count) * 0.2
        sentiment = max(-1.0, min(1.0, sentiment))
        
        # Determine impact
        if crit_count > 0:
            impact = 'critical'
        elif news_item.category == 'earnings':
            impact = 'high'
        elif news_item.category == 'macro':
            impact = 'high'
        else:
            impact = 'medium'
        
        return SentimentResult(
            news_id=news_item.id,
            sentiment_score=sentiment,
            impact_level=impact,
            confidence=0.5,  # Lower confidence for rule-based
            affected_symbols=news_item.symbols,
            reasoning=f"Rule-based: {pos_count} positive, {neg_count} negative keywords",
            trading_implications="Analyze further before trading",
        )
    
    def get_aggregate_sentiment(
        self,
        results: List[SentimentResult],
        symbol: str = None,
    ) -> Tuple[float, str]:
        """
        Get aggregate sentiment from multiple results.
        
        Args:
            results: List of SentimentResults
            symbol: Optional symbol to filter by
            
        Returns:
            Tuple of (aggregate_sentiment, dominant_impact)
        """
        if symbol:
            results = [r for r in results if symbol in r.affected_symbols]
        
        if not results:
            return 0.0, 'low'
        
        # Weight by confidence and impact
        impact_weights = {'critical': 2.0, 'high': 1.5, 'medium': 1.0, 'low': 0.5}
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for r in results:
            weight = r.confidence * impact_weights.get(r.impact_level, 1.0)
            weighted_sum += r.sentiment_score * weight
            weight_total += weight
        
        aggregate_sentiment = weighted_sum / weight_total if weight_total > 0 else 0.0
        
        # Dominant impact
        impact_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for r in results:
            impact_counts[r.impact_level] += 1
        
        dominant_impact = max(impact_counts, key=impact_counts.get)
        
        return aggregate_sentiment, dominant_impact
    
    def clear_cache(self):
        """Clear analysis cache."""
        self._analyzed_cache = {}
