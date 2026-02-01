"""
Paper Trading Risk Manager.

Enforces risk limits in code:
- Daily loss stop
- Max position per symbol
- Max concurrent positions
- Flatten by close
- Cooldown after losses
"""

from dataclasses import dataclass
from datetime import datetime, date, time
from typing import Dict, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Actions the risk manager can take."""
    ALLOW = "allow"
    BLOCK_DAILY_LOSS = "blocked_daily_loss_limit"
    BLOCK_MAX_POSITIONS = "blocked_max_positions"
    BLOCK_POSITION_SIZE = "blocked_position_size"
    BLOCK_COOLDOWN = "blocked_cooldown"
    BLOCK_MARKET_CLOSED = "blocked_market_closed"
    FLATTEN_EOD = "flatten_end_of_day"


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    
    # Daily limits
    daily_loss_limit: float = 5000.0  # Max daily loss in INR
    max_trades_per_day: int = 20
    
    # Position limits
    max_position_value: float = 500_000.0  # Max value per position
    max_position_pct: float = 0.15  # Max % of capital per position
    max_concurrent_positions: int = 5
    
    # Timing
    market_open: time = None
    market_close: time = None
    flatten_before_close_mins: int = 15  # Flatten 15 mins before close
    
    # Cooldown
    cooldown_after_losses: int = 3  # Consecutive losses before cooldown
    cooldown_duration_mins: int = 30
    
    def __post_init__(self):
        if self.market_open is None:
            self.market_open = time(9, 15)
        if self.market_close is None:
            self.market_close = time(15, 30)


@dataclass
class RiskState:
    """Current risk state."""
    
    # Daily tracking
    daily_pnl: float = 0.0
    trades_today: int = 0
    
    # Consecutive losses
    consecutive_losses: int = 0
    in_cooldown: bool = False
    cooldown_until: Optional[datetime] = None
    
    # Open positions
    open_position_count: int = 0
    open_position_value: float = 0.0


class PaperRiskManager:
    """
    Risk manager for paper trading.
    
    Checks and enforces all risk limits before allowing trades.
    """
    
    def __init__(
        self,
        limits: RiskLimits = None,
        capital: float = 1_000_000.0,
    ):
        """
        Initialize risk manager.
        
        Args:
            limits: Risk limit configuration
            capital: Trading capital
        """
        self.limits = limits or RiskLimits()
        self.capital = capital
        
        # Current state
        self._state = RiskState()
        self._last_reset_date = date.today()
        
        # Position tracking
        self._positions: Dict[str, float] = {}  # symbol -> value
    
    def check_entry(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        current_time: datetime = None,
    ) -> tuple[bool, RiskAction, str]:
        """
        Check if a new entry is allowed.
        
        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            quantity: Number of shares
            price: Entry price
            current_time: Current time
            
        Returns:
            Tuple of (allowed, action, reason)
        """
        now = current_time or datetime.now()
        self._check_daily_reset(now.date())
        
        # Check market hours
        if not self._is_market_open(now):
            return False, RiskAction.BLOCK_MARKET_CLOSED, "Market is closed"
        
        # Check if should flatten
        if self._should_flatten(now):
            return False, RiskAction.FLATTEN_EOD, "Too close to market close"
        
        # Check daily loss limit
        if self._state.daily_pnl <= -self.limits.daily_loss_limit:
            return False, RiskAction.BLOCK_DAILY_LOSS, f"Daily loss limit reached: ₹{self._state.daily_pnl:,.0f}"
        
        # Check max trades per day
        if self._state.trades_today >= self.limits.max_trades_per_day:
            return False, RiskAction.BLOCK_DAILY_LOSS, f"Max trades per day reached: {self._state.trades_today}"
        
        # Check cooldown
        if self._state.in_cooldown:
            if now < self._state.cooldown_until:
                remaining = (self._state.cooldown_until - now).seconds // 60
                return False, RiskAction.BLOCK_COOLDOWN, f"In cooldown, {remaining} mins remaining"
            else:
                self._state.in_cooldown = False
                self._state.cooldown_until = None
        
        # Check max concurrent positions
        if symbol not in self._positions:
            if self._state.open_position_count >= self.limits.max_concurrent_positions:
                return False, RiskAction.BLOCK_MAX_POSITIONS, f"Max positions reached: {self._state.open_position_count}"
        
        # Check position size
        position_value = quantity * price
        
        if position_value > self.limits.max_position_value:
            return False, RiskAction.BLOCK_POSITION_SIZE, f"Position too large: ₹{position_value:,.0f} > ₹{self.limits.max_position_value:,.0f}"
        
        max_by_pct = self.capital * self.limits.max_position_pct
        if position_value > max_by_pct:
            return False, RiskAction.BLOCK_POSITION_SIZE, f"Position exceeds {self.limits.max_position_pct*100}% of capital"
        
        return True, RiskAction.ALLOW, "Entry allowed"
    
    def record_entry(
        self,
        symbol: str,
        quantity: int,
        price: float,
    ):
        """Record a new position entry."""
        value = quantity * price
        
        if symbol not in self._positions:
            self._state.open_position_count += 1
        
        self._positions[symbol] = self._positions.get(symbol, 0) + value
        self._state.open_position_value += value
        self._state.trades_today += 1
    
    def record_exit(
        self,
        symbol: str,
        pnl: float,
    ):
        """Record a position exit and update state."""
        if symbol in self._positions:
            self._state.open_position_value -= self._positions[symbol]
            del self._positions[symbol]
            self._state.open_position_count -= 1
        
        # Update daily P&L
        self._state.daily_pnl += pnl
        
        # Track consecutive losses
        if pnl < 0:
            self._state.consecutive_losses += 1
            
            if self._state.consecutive_losses >= self.limits.cooldown_after_losses:
                self._state.in_cooldown = True
                self._state.cooldown_until = datetime.now().replace(
                    minute=datetime.now().minute + self.limits.cooldown_duration_mins
                )
                logger.warning(f"Entering cooldown after {self._state.consecutive_losses} consecutive losses")
        else:
            self._state.consecutive_losses = 0
    
    def should_flatten_all(self, current_time: datetime = None) -> bool:
        """Check if all positions should be flattened."""
        now = current_time or datetime.now()
        return self._should_flatten(now)
    
    def get_positions_to_flatten(self) -> List[str]:
        """Get list of symbols that should be flattened."""
        return list(self._positions.keys())
    
    def get_state(self) -> RiskState:
        """Get current risk state."""
        return self._state
    
    def get_daily_remaining(self) -> Dict:
        """Get remaining limits for today."""
        return {
            'pnl_remaining': self.limits.daily_loss_limit + self._state.daily_pnl,
            'trades_remaining': self.limits.max_trades_per_day - self._state.trades_today,
            'positions_remaining': self.limits.max_concurrent_positions - self._state.open_position_count,
        }
    
    def _is_market_open(self, now: datetime) -> bool:
        """Check if market is open."""
        current_time = now.time()
        
        # Check weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        return self.limits.market_open <= current_time <= self.limits.market_close
    
    def _should_flatten(self, now: datetime) -> bool:
        """Check if we should flatten positions."""
        if not self._is_market_open(now):
            return True
        
        # Check if close to market close
        close_dt = now.replace(
            hour=self.limits.market_close.hour,
            minute=self.limits.market_close.minute,
            second=0,
            microsecond=0
        )
        
        mins_to_close = (close_dt - now).total_seconds() / 60
        
        return mins_to_close <= self.limits.flatten_before_close_mins
    
    def _check_daily_reset(self, today: date):
        """Reset daily counters if new day."""
        if today > self._last_reset_date:
            self._state.daily_pnl = 0.0
            self._state.trades_today = 0
            self._last_reset_date = today
            logger.info(f"Daily reset for {today}")
    
    def reset(self):
        """Reset all state."""
        self._state = RiskState()
        self._positions.clear()
        self._last_reset_date = date.today()
