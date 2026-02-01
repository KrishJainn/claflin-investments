"""
AI Analyzer for the Coach.

Analyzes trading performance using LLM to:
- Identify winning/losing patterns
- Correlate trades with market conditions and news
- Generate insights for strategy improvement
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from ..ai_config import AIConfig, DEFAULT_AI_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class TradeAnalysis:
    """Analysis result for a set of trades."""
    
    period_start: datetime
    period_end: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate: float
    
    # Pattern insights
    winning_patterns: List[str] = field(default_factory=list)
    losing_patterns: List[str] = field(default_factory=list)
    
    # Indicator insights
    top_performing_indicators: List[str] = field(default_factory=list)
    underperforming_indicators: List[str] = field(default_factory=list)
    
    # Market condition insights
    best_market_regime: str = ""
    worst_market_regime: str = ""
    
    # News correlation
    news_impact_assessment: str = ""
    
    # Recommendations
    key_insights: List[str] = field(default_factory=list)
    recommended_adjustments: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'win_rate': self.win_rate,
            'winning_patterns': self.winning_patterns,
            'losing_patterns': self.losing_patterns,
            'top_performing_indicators': self.top_performing_indicators,
            'underperforming_indicators': self.underperforming_indicators,
            'best_market_regime': self.best_market_regime,
            'worst_market_regime': self.worst_market_regime,
            'news_impact_assessment': self.news_impact_assessment,
            'key_insights': self.key_insights,
            'recommended_adjustments': self.recommended_adjustments,
            'confidence_score': self.confidence_score,
        }


class AIAnalyzer:
    """LLM-powered trade performance analyzer."""
    
    ANALYSIS_PROMPT = """You are an expert quantitative analyst reviewing trading performance.

## Performance Summary
- **Period**: {period_start} to {period_end}
- **Total Trades**: {total_trades}
- **Win Rate**: {win_rate:.1%}
- **Total P&L**: ₹{total_pnl:,.0f}
- **Average Win**: ₹{avg_win:,.0f}
- **Average Loss**: ₹{avg_loss:,.0f}

## Sample Winning Trades
{winning_samples}

## Sample Losing Trades
{losing_samples}

## Indicator Contribution Analysis
{indicator_analysis}

## Market Regime Performance
{regime_analysis}

## Recent News Context
{news_context}

## Your Task
Analyze this trading data and provide:
1. **Winning Patterns**: What conditions led to profitable trades?
2. **Losing Patterns**: What conditions led to losses?
3. **Top Indicators**: Which indicators are working well?
4. **Underperforming Indicators**: Which should be reduced?
5. **Key Insights**: 3-5 actionable insights
6. **Recommended Adjustments**: Specific parameter changes (be conservative)

Respond in this JSON format:
{{
    "winning_patterns": ["pattern1", "pattern2"],
    "losing_patterns": ["pattern1", "pattern2"],
    "top_indicators": ["IND1", "IND2"],
    "underperforming_indicators": ["IND3", "IND4"],
    "best_regime": "trending_up/trending_down/ranging/volatile",
    "worst_regime": "trending_up/trending_down/ranging/volatile",
    "news_assessment": "Brief news impact assessment",
    "key_insights": ["insight1", "insight2", "insight3"],
    "adjustments": {{
        "entry_threshold_change": -0.05 to 0.05,
        "exit_threshold_change": -0.05 to 0.05,
        "indicator_weight_changes": {{"INDICATOR": change_value}},
        "stop_loss_multiplier_change": -0.2 to 0.2
    }},
    "confidence": 0.0 to 1.0
}}
"""
    
    def __init__(self, config: AIConfig = None):
        """
        Initialize AI Analyzer.
        
        Args:
            config: AI configuration
        """
        self.config = config or DEFAULT_AI_CONFIG
        self._client = None
        self._init_llm()
        
        self._analysis_history: List[TradeAnalysis] = []
    
    def _init_llm(self):
        """Initialize LLM client."""
        if not GENAI_AVAILABLE:
            logger.warning("google-genai not installed, LLM analysis disabled")
            return
        
        if not self.config.llm.api_key:
            logger.warning("GOOGLE_API_KEY not set, LLM analysis disabled")
            return
        
        try:
            self._client = genai.Client(api_key=self.config.llm.api_key)
            logger.info("Initialized Gemini for trade analysis")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self._client = None
    
    def analyze_trades(
        self,
        trade_contexts: List[Dict[str, Any]],
        news_summary: str = "",
        current_weights: Dict[str, float] = None,
    ) -> TradeAnalysis:
        """
        Analyze a set of trades.
        
        Args:
            trade_contexts: List of trade context dictionaries
            news_summary: Summary of relevant news
            current_weights: Current indicator weights
            
        Returns:
            TradeAnalysis with insights and recommendations
        """
        if not trade_contexts:
            return self._empty_analysis()
        
        # Calculate basic metrics
        metrics = self._calculate_metrics(trade_contexts)
        
        # Prepare analysis data
        winning_samples = self._format_trade_samples(
            [t for t in trade_contexts if t.get('pnl', 0) > 0][:5]
        )
        losing_samples = self._format_trade_samples(
            [t for t in trade_contexts if t.get('pnl', 0) < 0][:5]
        )
        indicator_analysis = self._analyze_indicators(trade_contexts, current_weights or {})
        regime_analysis = self._analyze_regimes(trade_contexts)
        
        # Use LLM if available
        if self._client:
            try:
                analysis = self._llm_analyze(
                    metrics, winning_samples, losing_samples,
                    indicator_analysis, regime_analysis, news_summary
                )
                self._analysis_history.append(analysis)
                return analysis
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
        
        # Fallback to rule-based
        analysis = self._rule_based_analyze(metrics, trade_contexts, current_weights or {})
        self._analysis_history.append(analysis)
        return analysis
    
    def _calculate_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate basic performance metrics."""
        pnls = [t.get('pnl', 0) for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]
        
        timestamps = [t.get('timestamp') for t in trades if t.get('timestamp')]
        if timestamps:
            if isinstance(timestamps[0], str):
                timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
            period_start = min(timestamps)
            period_end = max(timestamps)
        else:
            period_end = datetime.now()
            period_start = period_end - timedelta(days=1)
        
        return {
            'period_start': period_start,
            'period_end': period_end,
            'total_trades': len(trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'total_pnl': sum(pnls),
            'win_rate': len(winning) / len(trades) if trades else 0,
            'avg_win': sum(winning) / len(winning) if winning else 0,
            'avg_loss': abs(sum(losing) / len(losing)) if losing else 0,
        }
    
    def _format_trade_samples(self, trades: List[Dict]) -> str:
        """Format trade samples for prompt."""
        if not trades:
            return "No trades in this category"
        
        lines = []
        for t in trades:
            line = (
                f"- {t.get('symbol', 'N/A')}: {t.get('direction', 'N/A')} "
                f"P&L: ₹{t.get('pnl', 0):,.0f}, "
                f"SI: {t.get('super_indicator_value', 0):.2f}, "
                f"Regime: {t.get('market_regime', 'unknown')}"
            )
            lines.append(line)
        
        return "\n".join(lines)
    
    def _analyze_indicators(
        self,
        trades: List[Dict],
        weights: Dict[str, float],
    ) -> str:
        """Analyze indicator contributions across trades."""
        # Aggregate indicator contributions for winning vs losing trades
        indicator_wins = {}
        indicator_losses = {}
        
        for trade in trades:
            contributions = trade.get('indicator_contributions', {})
            pnl = trade.get('pnl', 0)
            
            for ind, contrib in contributions.items():
                if pnl > 0:
                    indicator_wins[ind] = indicator_wins.get(ind, 0) + contrib
                else:
                    indicator_losses[ind] = indicator_losses.get(ind, 0) + contrib
        
        lines = []
        all_indicators = set(indicator_wins.keys()) | set(indicator_losses.keys())
        
        for ind in list(all_indicators)[:10]:
            win_contrib = indicator_wins.get(ind, 0)
            loss_contrib = indicator_losses.get(ind, 0)
            weight = weights.get(ind, 1.0)
            net = win_contrib - abs(loss_contrib)
            
            lines.append(
                f"- {ind}: Weight={weight:.2f}, "
                f"Win Contrib={win_contrib:.2f}, Loss Contrib={loss_contrib:.2f}, "
                f"Net={net:.2f}"
            )
        
        return "\n".join(lines) if lines else "No indicator data available"
    
    def _analyze_regimes(self, trades: List[Dict]) -> str:
        """Analyze performance by market regime."""
        regime_stats = {}
        
        for trade in trades:
            regime = trade.get('market_regime', 'unknown')
            pnl = trade.get('pnl', 0)
            
            if regime not in regime_stats:
                regime_stats[regime] = {'wins': 0, 'losses': 0, 'pnl': 0}
            
            regime_stats[regime]['pnl'] += pnl
            if pnl > 0:
                regime_stats[regime]['wins'] += 1
            else:
                regime_stats[regime]['losses'] += 1
        
        lines = []
        for regime, stats in regime_stats.items():
            total = stats['wins'] + stats['losses']
            wr = stats['wins'] / total if total > 0 else 0
            lines.append(
                f"- {regime}: {total} trades, "
                f"Win Rate: {wr:.1%}, P&L: ₹{stats['pnl']:,.0f}"
            )
        
        return "\n".join(lines) if lines else "No regime data available"
    
    def _llm_analyze(
        self,
        metrics: Dict[str, Any],
        winning_samples: str,
        losing_samples: str,
        indicator_analysis: str,
        regime_analysis: str,
        news_context: str,
    ) -> TradeAnalysis:
        """Perform LLM analysis."""
        prompt = self.ANALYSIS_PROMPT.format(
            period_start=metrics['period_start'].strftime("%Y-%m-%d"),
            period_end=metrics['period_end'].strftime("%Y-%m-%d"),
            total_trades=metrics['total_trades'],
            win_rate=metrics['win_rate'],
            total_pnl=metrics['total_pnl'],
            avg_win=metrics['avg_win'],
            avg_loss=metrics['avg_loss'],
            winning_samples=winning_samples,
            losing_samples=losing_samples,
            indicator_analysis=indicator_analysis,
            regime_analysis=regime_analysis,
            news_context=news_context or "No relevant news",
        )
        
        response = self._client.models.generate_content(
            model=self.config.llm.model_name,
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 1024,
            }
        )
        
        return self._parse_llm_response(response.text, metrics)
    
    def _parse_llm_response(
        self,
        response_text: str,
        metrics: Dict[str, Any],
    ) -> TradeAnalysis:
        """Parse LLM response into TradeAnalysis."""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                return TradeAnalysis(
                    period_start=metrics['period_start'],
                    period_end=metrics['period_end'],
                    total_trades=metrics['total_trades'],
                    winning_trades=metrics['winning_trades'],
                    losing_trades=metrics['losing_trades'],
                    total_pnl=metrics['total_pnl'],
                    win_rate=metrics['win_rate'],
                    winning_patterns=data.get('winning_patterns', []),
                    losing_patterns=data.get('losing_patterns', []),
                    top_performing_indicators=data.get('top_indicators', []),
                    underperforming_indicators=data.get('underperforming_indicators', []),
                    best_market_regime=data.get('best_regime', ''),
                    worst_market_regime=data.get('worst_regime', ''),
                    news_impact_assessment=data.get('news_assessment', ''),
                    key_insights=data.get('key_insights', []),
                    recommended_adjustments=data.get('adjustments', {}),
                    confidence_score=float(data.get('confidence', 0.5)),
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        
        return self._rule_based_analyze(metrics, [], {})
    
    def _rule_based_analyze(
        self,
        metrics: Dict[str, Any],
        trades: List[Dict],
        weights: Dict[str, float],
    ) -> TradeAnalysis:
        """Rule-based analysis fallback."""
        insights = []
        adjustments = {}
        
        # Win rate analysis
        if metrics['win_rate'] < 0.4:
            insights.append("Win rate is low - consider tightening entry criteria")
            adjustments['entry_threshold_change'] = 0.02
        elif metrics['win_rate'] > 0.6:
            insights.append("Strong win rate - strategy is working well")
        
        # Risk/reward analysis
        if metrics['avg_loss'] > 0 and metrics['avg_win'] > 0:
            rr = metrics['avg_win'] / metrics['avg_loss']
            if rr < 1.0:
                insights.append("Risk-reward ratio below 1:1 - consider widening targets or tightening stops")
                adjustments['stop_loss_multiplier_change'] = -0.1
        
        return TradeAnalysis(
            period_start=metrics['period_start'],
            period_end=metrics['period_end'],
            total_trades=metrics['total_trades'],
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            total_pnl=metrics['total_pnl'],
            win_rate=metrics['win_rate'],
            key_insights=insights,
            recommended_adjustments=adjustments,
            confidence_score=0.4,
        )
    
    def _empty_analysis(self) -> TradeAnalysis:
        """Return empty analysis when no trades."""
        now = datetime.now()
        return TradeAnalysis(
            period_start=now,
            period_end=now,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_pnl=0.0,
            win_rate=0.0,
            key_insights=["No trades to analyze"],
            confidence_score=0.0,
        )
    
    def get_analysis_history(self) -> List[TradeAnalysis]:
        """Get history of analyses."""
        return self._analysis_history.copy()
