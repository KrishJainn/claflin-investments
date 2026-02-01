"""
AI-Enhanced Trader for the Player-Coach Trading System.

Extends the existing Player class with AI-powered decision making,
comprehensive context logging, and integration with the Coach.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import json

from ..player.trader import Player
from ..player.portfolio import Portfolio
from ..player.risk_manager import RiskManager
from ..player.execution import ExecutionEngine
from ..super_indicator.core import SuperIndicator
from ..super_indicator.signals import SignalType, PositionState
from ..ai_config import AIConfig, DEFAULT_AI_CONFIG
from .context_builder import TradeContextBuilder, TradeContext
from .signal_interpreter import SignalInterpreter, InterpretedSignal

logger = logging.getLogger(__name__)


class AITrader(Player):
    """
    AI-Enhanced Trader that extends the base Player.
    
    Enhancements:
    - Builds comprehensive trade context for each decision
    - Uses LLM to interpret uncertain signals
    - Logs full context for Coach analysis
    - Integrates news sentiment into decisions
    """
    
    def __init__(
        self,
        portfolio: Portfolio = None,
        risk_manager: RiskManager = None,
        execution: ExecutionEngine = None,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.001,
        allow_short: bool = True,
        ai_config: AIConfig = None,
    ):
        """
        Initialize AI Trader.
        
        Args:
            portfolio: Portfolio manager
            risk_manager: Risk manager
            execution: Execution engine
            initial_capital: Starting capital
            slippage_pct: Slippage percentage
            allow_short: Allow short positions
            ai_config: AI-specific configuration
        """
        super().__init__(
            portfolio=portfolio,
            risk_manager=risk_manager,
            execution=execution,
            initial_capital=initial_capital,
            slippage_pct=slippage_pct,
            allow_short=allow_short,
        )
        
        self.ai_config = ai_config or DEFAULT_AI_CONFIG
        self.context_builder = TradeContextBuilder()
        self.signal_interpreter = SignalInterpreter(self.ai_config)
        
        # Track all trade contexts for Coach analysis
        self._trade_contexts: List[TradeContext] = []
        self._interpreted_signals: List[InterpretedSignal] = []
        
        # Current indicator weights (updated by Coach)
        self._indicator_weights: Dict[str, float] = {}
        
        # News cache
        self._current_news: Dict[str, List[Dict]] = {}
        
        # Safety tracking
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._trading_halted: bool = False
        self._halt_reason: str = ""
        
        logger.info("AITrader initialized with AI enhancements")
    
    def process_bar(
        self,
        symbol: str,
        timestamp: datetime,
        ohlcv: Dict[str, float],
        super_indicator_value: float,
        atr: float,
        indicator_snapshot: Dict[str, float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a bar with AI-enhanced decision making.
        
        Overrides base Player.process_bar to add:
        - Context building
        - AI signal interpretation
        - Full context logging
        
        Args:
            symbol: Stock ticker
            timestamp: Bar timestamp
            ohlcv: OHLCV dict
            super_indicator_value: Super indicator value
            atr: ATR value
            indicator_snapshot: Current indicator values
            
        Returns:
            Trade result dict or None
        """
        # Check if trading is halted
        if self._trading_halted:
            logger.debug(f"Trading halted: {self._halt_reason}")
            return None
        
        # Build comprehensive context
        context = self.context_builder.build_context(
            symbol=symbol,
            timestamp=timestamp,
            ohlcv=ohlcv,
            super_indicator_value=super_indicator_value,
            indicator_snapshot=indicator_snapshot or {},
            indicator_weights=self._indicator_weights,
            atr=atr,
            news_data=self._current_news.get(symbol, []),
        )
        
        # Determine if we need AI interpretation
        use_ai = self.signal_interpreter.should_use_llm(context)
        
        if use_ai:
            # Use AI to interpret uncertain signal
            interpreted = self.signal_interpreter.interpret(context)
            self._interpreted_signals.append(interpreted)
            
            # Update context with AI reasoning
            context.ai_reasoning = interpreted.reasoning
            context.ai_confidence = interpreted.ai_confidence
            
            # Apply AI decision
            if not interpreted.should_trade:
                logger.info(
                    f"AI skipped trade on {symbol}: {interpreted.reasoning}"
                )
                self._trade_contexts.append(context)
                return None
            
            # Adjust signal based on AI interpretation
            adjusted_si_value = self._adjust_signal_for_ai(
                super_indicator_value,
                interpreted
            )
        else:
            adjusted_si_value = super_indicator_value
            interpreted = None
        
        # Store context
        self._trade_contexts.append(context)
        
        # Call base implementation with potentially adjusted signal
        result = super().process_bar(
            symbol=symbol,
            timestamp=timestamp,
            ohlcv=ohlcv,
            super_indicator_value=adjusted_si_value,
            atr=atr,
            indicator_snapshot=indicator_snapshot,
        )
        
        # If trade was made, enrich result with context
        if result:
            result['trade_context'] = context.to_dict()
            if interpreted:
                result['ai_interpretation'] = {
                    'reasoning': interpreted.reasoning,
                    'confidence': interpreted.ai_confidence,
                    'risk': interpreted.risk_assessment,
                }
            
            # Update daily PnL tracking for safety limits
            self._update_pnl_tracking(result)
        
        return result
    
    def _adjust_signal_for_ai(
        self,
        original_si_value: float,
        interpreted: InterpretedSignal
    ) -> float:
        """Adjust signal value based on AI interpretation."""
        # If AI changed direction, flip the signal
        if interpreted.interpreted_direction != interpreted.original_direction:
            if interpreted.interpreted_direction == 'hold':
                return 0.0
            elif interpreted.interpreted_direction == 'long' and original_si_value < 0:
                return abs(original_si_value)
            elif interpreted.interpreted_direction == 'short' and original_si_value > 0:
                return -abs(original_si_value)
        
        # Scale by AI confidence
        confidence_scale = interpreted.ai_confidence / max(interpreted.original_confidence, 0.1)
        return original_si_value * min(confidence_scale, 1.5)  # Cap at 1.5x
    
    def _update_pnl_tracking(self, trade_result: Dict[str, Any]):
        """Update PnL tracking and check safety limits."""
        pnl = trade_result.get('pnl', 0)
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        
        # Check safety limits
        capital = self.portfolio.total_value if self.portfolio else 100_000
        
        daily_limit = capital * self.ai_config.safety.daily_loss_limit_pct
        weekly_limit = capital * self.ai_config.safety.weekly_loss_limit_pct
        
        if self._daily_pnl < -daily_limit:
            self._halt_trading(f"Daily loss limit breached: ₹{abs(self._daily_pnl):,.0f}")
        elif self._weekly_pnl < -weekly_limit:
            self._halt_trading(f"Weekly loss limit breached: ₹{abs(self._weekly_pnl):,.0f}")
    
    def _halt_trading(self, reason: str):
        """Halt trading due to safety limit breach."""
        self._trading_halted = True
        self._halt_reason = reason
        logger.warning(f"TRADING HALTED: {reason}")
        
        if self.ai_config.safety.alert_on_breach:
            # In production, this would send an alert
            logger.critical(f"ALERT: Trading halted - {reason}")
    
    def resume_trading(self):
        """Resume trading after halt (requires manual intervention)."""
        if self.ai_config.safety.require_human_approval:
            logger.warning("Cannot resume automatically - human approval required")
            return False
        
        self._trading_halted = False
        self._halt_reason = ""
        logger.info("Trading resumed")
        return True
    
    def reset_daily_pnl(self):
        """Reset daily PnL counter (call at start of trading day)."""
        self._daily_pnl = 0.0
        self._trading_halted = False
        self._halt_reason = ""
    
    def reset_weekly_pnl(self):
        """Reset weekly PnL counter (call at start of trading week)."""
        self._weekly_pnl = 0.0
    
    def update_news(self, symbol: str, news_items: List[Dict]):
        """Update news cache for a symbol."""
        self._current_news[symbol] = news_items
        self.context_builder.update_news_cache(symbol, news_items)
    
    def update_indicator_weights(self, weights: Dict[str, float]):
        """Update indicator weights (called by Coach)."""
        self._indicator_weights = weights
        logger.info(f"Updated indicator weights: {len(weights)} indicators")
    
    def get_trade_contexts(self) -> List[TradeContext]:
        """Get all trade contexts for Coach analysis."""
        return self._trade_contexts.copy()
    
    def get_interpreted_signals(self) -> List[InterpretedSignal]:
        """Get all AI-interpreted signals for analysis."""
        return self._interpreted_signals.copy()
    
    def clear_contexts(self):
        """Clear stored contexts after Coach has processed them."""
        self._trade_contexts = []
        self._interpreted_signals = []
    
    def get_status(self) -> Dict[str, Any]:
        """Get current AI Trader status."""
        return {
            'trading_halted': self._trading_halted,
            'halt_reason': self._halt_reason,
            'daily_pnl': self._daily_pnl,
            'weekly_pnl': self._weekly_pnl,
            'contexts_collected': len(self._trade_contexts),
            'ai_interpretations': len(self._interpreted_signals),
            'llm_enabled': self.signal_interpreter._client is not None,
        }
    
    def export_contexts_for_coach(self) -> List[Dict[str, Any]]:
        """Export all contexts in a format ready for Coach analysis."""
        return [ctx.to_dict() for ctx in self._trade_contexts]
