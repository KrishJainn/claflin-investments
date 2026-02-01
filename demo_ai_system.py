#!/usr/bin/env python3
"""
Demo script for the Player-Coach AI Trading System.

This script demonstrates the core components of the AI system:
1. AI Player with context building and signal interpretation
2. AI Coach with news ingestion, sentiment analysis, and feedback loop

Run this to verify the modules are working correctly.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def demo_ai_config():
    """Demonstrate AI configuration."""
    print("\n" + "="*60)
    print("1. AI Configuration")
    print("="*60)
    
    from trading_evolution.ai_config import DEFAULT_AI_CONFIG
    
    config = DEFAULT_AI_CONFIG
    issues = config.validate()
    
    print(f"LLM Model: {config.llm.model_name}")
    print(f"Safety Limits:")
    print(f"  - Daily Loss Limit: {config.safety.daily_loss_limit_pct:.1%}")
    print(f"  - Weekly Loss Limit: {config.safety.weekly_loss_limit_pct:.1%}")
    print(f"  - Max Drawdown: {config.safety.max_drawdown_pct:.1%}")
    print(f"Learning Cycle: {config.coach.analysis_frequency}")
    
    if issues:
        print(f"\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Configuration valid")
    
    return config


def demo_context_builder():
    """Demonstrate trade context building."""
    print("\n" + "="*60)
    print("2. Trade Context Builder")
    print("="*60)
    
    from trading_evolution.ai_player.context_builder import TradeContextBuilder
    
    builder = TradeContextBuilder()
    
    # Simulate a trade context
    context = builder.build_context(
        symbol="RELIANCE.NS",
        timestamp=datetime.now(),
        ohlcv={'open': 2500, 'high': 2520, 'low': 2490, 'close': 2510, 'volume': 1000000},
        super_indicator_value=0.65,
        indicator_snapshot={
            'RSI_14': 0.6,
            'MACD_Histogram': 0.02,
            'ADX_14': 35,
            'EMA_Trend': 0.01,
        },
        indicator_weights={
            'RSI_14': 1.2,
            'MACD_Histogram': 0.8,
            'ADX_14': 1.0,
            'EMA_Trend': 1.1,
        },
        atr=25.0,
    )
    
    print(f"Symbol: {context.symbol}")
    print(f"Direction: {context.direction}")
    print(f"Signal Strength: {context.signal_strength:.2f}")
    print(f"Signal Confidence: {context.signal_confidence:.2f}")
    print(f"Market Regime: {context.market_regime}")
    print(f"Top Indicators: {context.top_contributing_indicators}")
    
    print("\n✅ Context builder working")
    return context


def demo_signal_interpreter(context):
    """Demonstrate signal interpretation."""
    print("\n" + "="*60)
    print("3. Signal Interpreter")
    print("="*60)
    
    from trading_evolution.ai_player.signal_interpreter import SignalInterpreter
    from trading_evolution.ai_config import AIConfig
    
    # Create config without requiring API key for demo
    config = AIConfig()
    interpreter = SignalInterpreter(config)
    
    # Interpret the signal (will use rule-based fallback if no API key)
    result = interpreter.interpret(context)
    
    print(f"Original Direction: {result.original_direction}")
    print(f"Interpreted Direction: {result.interpreted_direction}")
    print(f"Should Trade: {result.should_trade}")
    print(f"AI Confidence: {result.ai_confidence:.2f}")
    print(f"Risk Assessment: {result.risk_assessment}")
    print(f"Position Size: {result.suggested_position_size:.2f}")
    print(f"Reasoning: {result.reasoning}")
    
    print("\n✅ Signal interpreter working")
    return result


def demo_news_ingester():
    """Demonstrate news ingestion."""
    print("\n" + "="*60)
    print("4. News Ingester")
    print("="*60)
    
    from trading_evolution.ai_coach.news_ingester import NewsIngester, NewsItem
    from trading_evolution.ai_config import AIConfig
    
    config = AIConfig()
    ingester = NewsIngester(config)
    
    # Create sample news item (don't actually fetch to avoid API calls)
    sample_news = NewsItem(
        id="demo123",
        headline="Reliance Industries reports strong quarterly earnings, beats estimates",
        source="Demo Source",
        published_at=datetime.now(),
        url="https://example.com/news",
        summary="Reliance Industries reported quarterly profits that exceeded analyst expectations...",
        symbols=["RELIANCE.NS"],
        category="earnings",
    )
    
    print(f"Sample News:")
    print(f"  Headline: {sample_news.headline[:60]}...")
    print(f"  Source: {sample_news.source}")
    print(f"  Category: {sample_news.category}")
    print(f"  Symbols: {sample_news.symbols}")
    
    # Test categorization
    test_headlines = [
        "RBI announces interest rate decision",
        "TCS shares surge on strong earnings",
        "Indian IT sector outlook remains positive",
    ]
    
    print(f"\nCategorization test:")
    for headline in test_headlines:
        category = ingester._categorize_news(headline)
        print(f"  '{headline[:40]}...' -> {category}")
    
    print("\n✅ News ingester working")
    return sample_news


def demo_sentiment_analyzer(news_item):
    """Demonstrate sentiment analysis."""
    print("\n" + "="*60)
    print("5. Sentiment Analyzer")
    print("="*60)
    
    from trading_evolution.ai_coach.sentiment_analyzer import SentimentAnalyzer
    from trading_evolution.ai_config import AIConfig
    
    config = AIConfig()
    analyzer = SentimentAnalyzer(config)
    
    # Analyze the sample news (will use rule-based fallback)
    result = analyzer.analyze(news_item, watchlist_symbols=["RELIANCE.NS", "TCS.NS"])
    
    print(f"News ID: {result.news_id}")
    print(f"Sentiment Score: {result.sentiment_score:.2f}")
    print(f"Impact Level: {result.impact_level}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Affected Symbols: {result.affected_symbols}")
    print(f"Reasoning: {result.reasoning}")
    
    print("\n✅ Sentiment analyzer working")
    return result


def demo_strategy_updater():
    """Demonstrate strategy updates."""
    print("\n" + "="*60)
    print("6. Strategy Updater")
    print("="*60)
    
    from trading_evolution.ai_coach.strategy_updater import StrategyUpdater
    from trading_evolution.ai_coach.ai_analyzer import TradeAnalysis
    from trading_evolution.ai_config import AIConfig
    
    config = AIConfig()
    updater = StrategyUpdater(config)
    
    # Initialize with sample weights
    initial_weights = {
        'RSI_14': 1.0,
        'MACD_Histogram': 1.0,
        'ADX_14': 1.0,
        'EMA_Trend': 1.0,
        'Bollinger_Band': 0.8,
    }
    
    version = updater.initialize_strategy(
        indicator_weights=initial_weights,
        entry_threshold=0.7,
        exit_threshold=0.3,
    )
    
    print(f"Initial Version: {version.version_id}")
    print(f"Entry Threshold: {version.entry_threshold}")
    print(f"Exit Threshold: {version.exit_threshold}")
    print(f"Indicators: {len(version.indicator_weights)}")
    
    # Simulate an analysis with recommendations
    analysis = TradeAnalysis(
        period_start=datetime.now() - timedelta(days=1),
        period_end=datetime.now(),
        total_trades=20,
        winning_trades=12,
        losing_trades=8,
        total_pnl=15000,
        win_rate=0.6,
        key_insights=["RSI working well", "Consider tightening entry"],
        recommended_adjustments={
            'entry_threshold_change': 0.02,
            'indicator_weight_changes': {'RSI_14': 0.05, 'MACD_Histogram': -0.03}
        },
        confidence_score=0.75,
    )
    
    # Generate update
    update = updater.generate_update(analysis)
    
    if update:
        print(f"\nUpdate Generated:")
        print(f"  Confidence: {update.confidence:.2f}")
        print(f"  Changes Capped: {update.changes_capped}")
        print(f"  Proposed Changes: {list(update.proposed_changes.keys())}")
        
        # Apply update
        new_version = updater.apply_update(update)
        print(f"\nNew Version: {new_version.version_id}")
        print(f"  Entry Threshold: {new_version.entry_threshold}")
    
    print("\n✅ Strategy updater working")
    return updater


def demo_feedback_loop():
    """Demonstrate feedback loop."""
    print("\n" + "="*60)
    print("7. Feedback Loop")
    print("="*60)
    
    from trading_evolution.ai_coach.feedback_loop import FeedbackLoop
    from trading_evolution.ai_config import AIConfig
    import tempfile
    
    config = AIConfig()
    
    # Use temp directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        loop = FeedbackLoop(config, data_dir=Path(tmpdir))
        
        # Initialize with sample strategy
        initial_weights = {'RSI_14': 1.0, 'MACD_Histogram': 1.0}
        loop.initialize_strategy(initial_weights)
        
        status = loop.get_status()
        
        print(f"Feedback Loop Status:")
        print(f"  Running: {status['is_running']}")
        print(f"  Current Version: {status['current_version_id']}")
        print(f"  Total Versions: {status['total_versions']}")
        
        strategy = loop.get_current_strategy()
        print(f"\nCurrent Strategy:")
        print(f"  Version ID: {strategy['version_id']}")
        print(f"  Entry Threshold: {strategy['entry_threshold']}")
        print(f"  Indicators: {len(strategy['indicator_weights'])}")
    
    print("\n✅ Feedback loop working")


def main():
    """Run all demos."""
    print("="*60)
    print("PLAYER-COACH AI TRADING SYSTEM - DEMO")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run demos
        config = demo_ai_config()
        context = demo_context_builder()
        signal = demo_signal_interpreter(context)
        news = demo_news_ingester()
        sentiment = demo_sentiment_analyzer(news)
        updater = demo_strategy_updater()
        demo_feedback_loop()
        
        print("\n" + "="*60)
        print("ALL DEMOS PASSED ✅")
        print("="*60)
        print("\nThe Player-Coach AI Trading System is ready!")
        print("\nNext steps:")
        print("1. Set GOOGLE_API_KEY environment variable for LLM features")
        print("2. Optionally set NEWS_API_KEY for real news fetching")
        print("3. Integrate with existing trading system via FeedbackLoop callbacks")
        print("4. Run daily learning cycles to improve strategy")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
