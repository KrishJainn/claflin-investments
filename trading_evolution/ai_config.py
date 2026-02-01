"""
AI Configuration module for the Player-Coach Trading System.

Contains LLM settings, API configurations, and safety guardrails.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import os


@dataclass
class LLMConfig:
    """LLM (Gemini) configuration."""
    model_name: str = "gemini-3-flash-preview"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    temperature: float = 0.3  # Lower for more consistent trading decisions
    max_tokens: int = 2048
    timeout_seconds: int = 30


@dataclass
class NewsAPIConfig:
    """News API configuration."""
    newsapi_key: Optional[str] = field(default_factory=lambda: os.getenv("NEWS_API_KEY"))
    yahoo_finance_enabled: bool = True
    fetch_interval_minutes: int = 30
    max_articles_per_fetch: int = 50
    relevance_threshold: float = 0.5


@dataclass
class SafetyConfig:
    """Safety guardrails - NON-NEGOTIABLE limits."""
    daily_loss_limit_pct: float = 0.02  # 2% of capital
    weekly_loss_limit_pct: float = 0.05  # 5% of capital
    max_drawdown_pct: float = 0.20  # 20% max drawdown
    max_strategy_drift_pct: float = 0.30  # Max 30% weight change per update
    max_position_correlation: float = 0.80  # Correlation limit between positions
    require_human_approval: bool = False  # Set True for human-in-loop
    alert_on_breach: bool = True  # Send alerts when limits breached


@dataclass
class LearningCycleConfig:
    """Learning cycle timing configuration (IST timezone)."""
    # Post-market analysis window
    analysis_start_hour: int = 16  # 4:00 PM IST
    analysis_end_hour: int = 18  # 6:00 PM IST
    
    # Pre-market update window
    update_hour: int = 8  # 8:00 AM IST
    
    # Learning parameters
    min_trades_for_learning: int = 5  # Minimum trades before Coach learns
    lookback_days: int = 30  # Days of history for pattern analysis
    validation_holdout_pct: float = 0.20  # Holdout for validating updates


@dataclass
class AIPlayerConfig:
    """AI Player specific configuration."""
    use_llm_for_uncertain_signals: bool = True
    signal_confidence_threshold: float = 0.6  # Below this, consult LLM
    log_full_context: bool = True
    max_reasoning_depth: int = 3  # Depth of LLM reasoning chain


@dataclass
class AICoachConfig:
    """AI Coach specific configuration."""
    analysis_frequency: str = "daily"  # daily, hourly, weekly
    max_weight_adjustment_per_indicator: float = 0.10  # Max 10% per update
    min_confidence_for_update: float = 0.7
    require_backtest_validation: bool = True
    min_backtest_improvement_pct: float = 0.05  # 5% improvement required


@dataclass
class AIConfig:
    """Main AI configuration container."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    news: NewsAPIConfig = field(default_factory=NewsAPIConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    learning_cycle: LearningCycleConfig = field(default_factory=LearningCycleConfig)
    player: AIPlayerConfig = field(default_factory=AIPlayerConfig)
    coach: AICoachConfig = field(default_factory=AICoachConfig)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if not self.llm.api_key:
            issues.append("GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable not set")
        
        if not self.news.newsapi_key and not self.news.yahoo_finance_enabled:
            issues.append("No news source configured (set NEWS_API_KEY or enable Yahoo Finance)")
        
        if self.safety.daily_loss_limit_pct > 0.05:
            issues.append("Daily loss limit too high (>5%), this is risky")
        
        return issues


# Default AI configuration instance
DEFAULT_AI_CONFIG = AIConfig()
