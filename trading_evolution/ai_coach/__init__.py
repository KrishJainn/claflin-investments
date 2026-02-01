"""
AI Coach Module for the Player-Coach Trading System.

Phase 2 — Coach v1: LLM-powered post-market analysis (diagnosis only, no recommendations yet).
Also includes Phase 1+ components for news, sentiment, strategy updates, and feedback loop.
"""

from .ai_analyzer import AIAnalyzer
from .news_ingester import NewsIngester
from .sentiment_analyzer import SentimentAnalyzer
from .strategy_updater import StrategyUpdater
from .feedback_loop import FeedbackLoop
from .post_market_analyzer import (
    PostMarketAnalyzer,
    CoachDiagnosis,
    Mistake,
    MistakeType,
    Opportunity,
    NewsEvent,
    IndicatorScore,
    WeightRecommendation,
)

__all__ = [
    'AIAnalyzer',
    'NewsIngester',
    'SentimentAnalyzer',
    'StrategyUpdater',
    'FeedbackLoop',
    # Phase 2
    'PostMarketAnalyzer',
    'CoachDiagnosis',
    'Mistake',
    'MistakeType',
    'Opportunity',
    'NewsEvent',
    'IndicatorScore',
    'WeightRecommendation',
]
