"""
Feedback Loop Orchestrator for the AI Coach.

Coordinates the daily learning cycle:
- Post-market: Analyze day's trades
- Overnight: Generate strategy updates
- Pre-market: Validate and deploy new strategy
"""

from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
import json
import threading
from pathlib import Path

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False

from .ai_analyzer import AIAnalyzer, TradeAnalysis
from .news_ingester import NewsIngester, NewsItem
from .sentiment_analyzer import SentimentAnalyzer, SentimentResult
from .strategy_updater import StrategyUpdater, StrategyUpdate, StrategyVersion
from ..ai_config import AIConfig, DEFAULT_AI_CONFIG

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """
    Orchestrates the continuous feedback loop between Player and Coach.
    
    Daily Cycle:
    1. Post-market (4-6 PM): Collect trades, fetch news, analyze performance
    2. Overnight: Generate strategy updates
    3. Pre-market (8-9 AM): Validate and deploy updates
    """
    
    def __init__(
        self,
        config: AIConfig = None,
        data_dir: Path = None,
    ):
        """
        Initialize feedback loop orchestrator.
        
        Args:
            config: AI configuration
            data_dir: Directory for storing state and results
        """
        self.config = config or DEFAULT_AI_CONFIG
        self.data_dir = Path(data_dir) if data_dir else Path("ai_coach_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.analyzer = AIAnalyzer(self.config)
        self.news_ingester = NewsIngester(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.strategy_updater = StrategyUpdater(self.config)
        
        # State
        self._is_running = False
        self._last_analysis_time: Optional[datetime] = None
        self._last_update_time: Optional[datetime] = None
        self._pending_update: Optional[StrategyUpdate] = None
        
        # Callbacks
        self._on_analysis_complete: Optional[Callable[[TradeAnalysis], None]] = None
        self._on_update_deployed: Optional[Callable[[StrategyVersion], None]] = None
        self._get_trade_contexts: Optional[Callable[[], List[Dict]]] = None
        self._validate_strategy: Optional[Callable[[Dict], Dict]] = None
        
        # Daily state cache
        self._daily_trades: List[Dict] = []
        self._daily_news: List[NewsItem] = []
        self._daily_sentiment: List[SentimentResult] = []
        
        logger.info("FeedbackLoop initialized")
    
    def set_callbacks(
        self,
        get_trade_contexts: Callable[[], List[Dict]] = None,
        validate_strategy: Callable[[Dict], Dict] = None,
        on_analysis_complete: Callable[[TradeAnalysis], None] = None,
        on_update_deployed: Callable[[StrategyVersion], None] = None,
    ):
        """
        Set callback functions for integration with trading system.
        
        Args:
            get_trade_contexts: Function to get trade contexts from Player
            validate_strategy: Function to backtest a strategy
            on_analysis_complete: Called when daily analysis is complete
            on_update_deployed: Called when new strategy is deployed
        """
        self._get_trade_contexts = get_trade_contexts
        self._validate_strategy = validate_strategy
        self._on_analysis_complete = on_analysis_complete
        self._on_update_deployed = on_update_deployed
    
    def initialize_strategy(
        self,
        indicator_weights: Dict[str, float],
        entry_threshold: float = 0.7,
        exit_threshold: float = 0.3,
        stop_loss_multiplier: float = 2.0,
    ) -> StrategyVersion:
        """Initialize the first strategy version."""
        version = self.strategy_updater.initialize_strategy(
            indicator_weights=indicator_weights,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss_multiplier=stop_loss_multiplier,
        )
        self._save_state()
        return version
    
    def run_daily_cycle(
        self,
        trade_contexts: List[Dict] = None,
        watchlist_symbols: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete daily learning cycle.
        
        This is the main entry point for daily learning.
        Can be called manually or scheduled.
        
        Args:
            trade_contexts: Trade contexts from Player (or uses callback)
            watchlist_symbols: Symbols to fetch news for
            
        Returns:
            Summary of the cycle results
        """
        cycle_start = datetime.now()
        results = {
            'cycle_start': cycle_start.isoformat(),
            'trades_analyzed': 0,
            'news_items': 0,
            'analysis_complete': False,
            'update_generated': False,
            'update_deployed': False,
            'errors': [],
        }
        
        try:
            # Phase 1: Collect data
            logger.info("Phase 1: Collecting trade data and news")
            
            # Get trade contexts
            if trade_contexts is None and self._get_trade_contexts:
                trade_contexts = self._get_trade_contexts()
            
            if not trade_contexts:
                logger.warning("No trade contexts available for analysis")
                results['errors'].append("No trade contexts")
                return results
            
            results['trades_analyzed'] = len(trade_contexts)
            self._daily_trades = trade_contexts
            
            # Fetch news
            if watchlist_symbols:
                self._daily_news = self.news_ingester.fetch_all(
                    symbols=watchlist_symbols,
                    hours_back=24,
                )
                results['news_items'] = len(self._daily_news)
                
                # Analyze sentiment
                self._daily_sentiment = self.sentiment_analyzer.analyze_batch(
                    self._daily_news,
                    watchlist_symbols,
                )
            
            # Phase 2: Analyze performance
            logger.info("Phase 2: Analyzing trading performance")
            
            news_summary = self._build_news_summary()
            current_weights = self.strategy_updater.get_current_weights()
            
            analysis = self.analyzer.analyze_trades(
                trade_contexts=trade_contexts,
                news_summary=news_summary,
                current_weights=current_weights,
            )
            
            results['analysis_complete'] = True
            self._last_analysis_time = datetime.now()
            
            if self._on_analysis_complete:
                self._on_analysis_complete(analysis)
            
            # Phase 3: Generate strategy update
            logger.info("Phase 3: Generating strategy update")
            
            update = self.strategy_updater.generate_update(analysis)
            
            if update:
                results['update_generated'] = True
                self._pending_update = update
                
                # Phase 4: Validate and deploy
                if self.config.coach.require_backtest_validation and self._validate_strategy:
                    logger.info("Phase 4: Validating strategy update")
                    
                    # Build strategy for validation
                    test_strategy = {
                        'indicator_weights': update.proposed_changes.get(
                            'indicator_weights',
                            current_weights
                        ),
                        'entry_threshold': update.proposed_changes.get(
                            'entry_threshold',
                            self.strategy_updater.get_current_version().entry_threshold
                        ),
                        'exit_threshold': update.proposed_changes.get(
                            'exit_threshold',
                            self.strategy_updater.get_current_version().exit_threshold
                        ),
                    }
                    
                    backtest_results = self._validate_strategy(test_strategy)
                    
                    # Check if improvement meets threshold
                    current_version = self.strategy_updater.get_current_version()
                    if current_version and current_version.backtest_sharpe:
                        old_sharpe = current_version.backtest_sharpe
                        new_sharpe = backtest_results.get('sharpe_ratio', 0)
                        improvement = (new_sharpe - old_sharpe) / abs(old_sharpe) if old_sharpe else 0
                        
                        if improvement < self.config.coach.min_backtest_improvement_pct:
                            logger.info(
                                f"Update improvement {improvement:.1%} below threshold "
                                f"{self.config.coach.min_backtest_improvement_pct:.1%}, skipping"
                            )
                            self._pending_update = None
                            return results
                else:
                    backtest_results = None
                
                # Deploy update
                if self._pending_update:
                    new_version = self.strategy_updater.apply_update(
                        self._pending_update,
                        backtest_results,
                    )
                    results['update_deployed'] = True
                    self._last_update_time = datetime.now()
                    self._pending_update = None
                    
                    if self._on_update_deployed:
                        self._on_update_deployed(new_version)
                    
                    logger.info(f"Deployed strategy version {new_version.version_id}")
            
            # Save state
            self._save_state()
            self._save_cycle_results(results, analysis)
            
        except Exception as e:
            logger.error(f"Daily cycle error: {e}")
            results['errors'].append(str(e))
        
        results['cycle_end'] = datetime.now().isoformat()
        results['duration_seconds'] = (datetime.now() - cycle_start).total_seconds()
        
        logger.info(
            f"Daily cycle complete: {results['trades_analyzed']} trades, "
            f"update_deployed={results['update_deployed']}"
        )
        
        return results
    
    def _build_news_summary(self) -> str:
        """Build news summary for analysis."""
        if not self._daily_sentiment:
            return "No news data available"
        
        # Get aggregate sentiment
        agg_sentiment, dominant_impact = self.sentiment_analyzer.get_aggregate_sentiment(
            self._daily_sentiment
        )
        
        # Get high impact items
        high_impact = [s for s in self._daily_sentiment if s.impact_level in ['high', 'critical']]
        
        summary_parts = [
            f"Overall market sentiment: {agg_sentiment:.2f} ({dominant_impact} impact)",
            f"Total news items: {len(self._daily_news)}",
        ]
        
        if high_impact:
            summary_parts.append("High impact news:")
            for item in high_impact[:3]:
                summary_parts.append(f"- {item.reasoning[:100]}")
        
        return "\n".join(summary_parts)
    
    def _save_state(self):
        """Save current state to disk."""
        state = {
            'last_analysis_time': self._last_analysis_time.isoformat() if self._last_analysis_time else None,
            'last_update_time': self._last_update_time.isoformat() if self._last_update_time else None,
            'current_version_id': self.strategy_updater.get_current_version().version_id if self.strategy_updater.get_current_version() else None,
        }
        
        state_file = self.data_dir / 'feedback_loop_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Also save strategy versions
        versions_file = self.data_dir / 'strategy_versions.json'
        self.strategy_updater.export_versions(str(versions_file))
    
    def _save_cycle_results(self, results: Dict, analysis: TradeAnalysis):
        """Save cycle results for historical tracking."""
        results_dir = self.data_dir / 'cycle_results'
        results_dir.mkdir(exist_ok=True)
        
        filename = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = results_dir / filename
        
        data = {
            'results': results,
            'analysis': analysis.to_dict() if analysis else None,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_state(self):
        """Load state from disk."""
        state_file = self.data_dir / 'feedback_loop_state.json'
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            if state.get('last_analysis_time'):
                self._last_analysis_time = datetime.fromisoformat(state['last_analysis_time'])
            if state.get('last_update_time'):
                self._last_update_time = datetime.fromisoformat(state['last_update_time'])
        
        # Load strategy versions
        versions_file = self.data_dir / 'strategy_versions.json'
        if versions_file.exists():
            self.strategy_updater.import_versions(str(versions_file))
    
    def start_scheduled(self):
        """Start the scheduled feedback loop (runs in background)."""
        if not SCHEDULE_AVAILABLE:
            logger.error("schedule library not installed, cannot run scheduled loop")
            return
        
        if self._is_running:
            logger.warning("Feedback loop already running")
            return
        
        # Schedule post-market analysis
        analysis_time = f"{self.config.learning_cycle.analysis_start_hour}:00"
        schedule.every().day.at(analysis_time).do(self._scheduled_analysis)
        
        self._is_running = True
        logger.info(f"Scheduled feedback loop started (analysis at {analysis_time})")
        
        # Run scheduler in background thread
        def run_scheduler():
            while self._is_running:
                schedule.run_pending()
                import time
                time.sleep(60)
        
        thread = threading.Thread(target=run_scheduler, daemon=True)
        thread.start()
    
    def stop_scheduled(self):
        """Stop the scheduled feedback loop."""
        self._is_running = False
        schedule.clear()
        logger.info("Scheduled feedback loop stopped")
    
    def _scheduled_analysis(self):
        """Called by scheduler to run daily cycle."""
        logger.info("Running scheduled daily analysis")
        self.run_daily_cycle()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current feedback loop status."""
        current_version = self.strategy_updater.get_current_version()
        
        return {
            'is_running': self._is_running,
            'last_analysis': self._last_analysis_time.isoformat() if self._last_analysis_time else None,
            'last_update': self._last_update_time.isoformat() if self._last_update_time else None,
            'current_version_id': current_version.version_id if current_version else None,
            'total_versions': len(self.strategy_updater.get_version_history()),
            'pending_update': self._pending_update is not None,
        }
    
    def get_current_strategy(self) -> Optional[Dict[str, Any]]:
        """Get current strategy configuration for Player."""
        version = self.strategy_updater.get_current_version()
        if not version:
            return None
        
        return {
            'version_id': version.version_id,
            'indicator_weights': version.indicator_weights,
            'entry_threshold': version.entry_threshold,
            'exit_threshold': version.exit_threshold,
            'stop_loss_multiplier': version.stop_loss_multiplier,
        }
    
    def force_rollback(self, to_version_id: int = None) -> StrategyVersion:
        """Force rollback to a previous strategy version."""
        version = self.strategy_updater.rollback(to_version_id)
        self._save_state()
        return version
