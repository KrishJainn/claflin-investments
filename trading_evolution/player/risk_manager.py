"""
Risk Management module.

Handles:
- Position sizing (max 2% risk per trade)
- Stop loss calculation (ATR-based)
- Risk/reward validation
- Maximum concurrent positions
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from decimal import Decimal, ROUND_DOWN
import numpy as np


@dataclass
class RiskParameters:
    """Risk management parameters."""
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    max_concurrent_positions: int = 5
    atr_stop_multiplier: float = 2.0  # Stop loss at 2x ATR
    min_risk_reward_ratio: float = 1.5  # Minimum 1.5:1 R:R
    max_position_pct: float = 0.20  # Max 20% of portfolio in single position
    min_position_size: int = 1  # Minimum shares to trade


class RiskManager:
    """
    Manages trading risk and position sizing.

    Key functions:
    - Calculate position size based on risk tolerance
    - Calculate stop loss levels
    - Validate trades against risk limits
    - Track current risk exposure
    """

    def __init__(self, params: RiskParameters = None):
        """
        Initialize risk manager.

        Args:
            params: Risk parameters (uses defaults if None)
        """
        self.params = params or RiskParameters()
        self.current_positions = 0
        self.current_risk_exposure = 0.0

    def calculate_position_size(self,
                                portfolio_value: float,
                                entry_price: float,
                                stop_loss_price: float,
                                direction: str = 'LONG') -> int:
        """
        Calculate position size based on risk per trade.

        Uses: Position Size = (Risk Amount) / (Price - Stop Loss)

        Args:
            portfolio_value: Current portfolio value
            entry_price: Intended entry price
            stop_loss_price: Stop loss price
            direction: 'LONG' or 'SHORT'

        Returns:
            Number of shares to trade
        """
        # Calculate risk per share
        if direction == 'LONG':
            risk_per_share = entry_price - stop_loss_price
        else:
            risk_per_share = stop_loss_price - entry_price

        if risk_per_share <= 0:
            return 0

        # Calculate max risk amount
        max_risk_amount = portfolio_value * self.params.max_risk_per_trade

        # Position size based on risk
        position_size_risk = int(max_risk_amount / risk_per_share)

        # Also limit by max position percentage
        max_position_value = portfolio_value * self.params.max_position_pct
        position_size_value = int(max_position_value / entry_price)

        # Take the smaller of the two limits
        position_size = min(position_size_risk, position_size_value)

        # Ensure minimum size
        return max(0, position_size)

    def calculate_stop_loss(self,
                            entry_price: float,
                            atr: float,
                            direction: str = 'LONG') -> float:
        """
        Calculate stop loss price based on ATR.

        Args:
            entry_price: Entry price
            atr: Average True Range value
            direction: 'LONG' or 'SHORT'

        Returns:
            Stop loss price
        """
        stop_distance = atr * self.params.atr_stop_multiplier

        if direction == 'LONG':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        return stop_loss

    def calculate_take_profit(self,
                              entry_price: float,
                              stop_loss_price: float,
                              direction: str = 'LONG',
                              risk_reward_ratio: float = None) -> float:
        """
        Calculate take profit price based on risk/reward ratio.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            direction: 'LONG' or 'SHORT'
            risk_reward_ratio: Target R:R (uses min if None)

        Returns:
            Take profit price
        """
        rr_ratio = risk_reward_ratio or self.params.min_risk_reward_ratio

        # Calculate risk distance
        if direction == 'LONG':
            risk_distance = entry_price - stop_loss_price
            take_profit = entry_price + (risk_distance * rr_ratio)
        else:
            risk_distance = stop_loss_price - entry_price
            take_profit = entry_price - (risk_distance * rr_ratio)

        return take_profit

    def can_open_position(self, current_positions: int = None) -> Tuple[bool, str]:
        """
        Check if we can open a new position.

        Args:
            current_positions: Current number of positions

        Returns:
            Tuple of (can_open, reason)
        """
        positions = current_positions if current_positions is not None else self.current_positions

        if positions >= self.params.max_concurrent_positions:
            return False, f"Max positions ({self.params.max_concurrent_positions}) reached"

        return True, "OK"

    def validate_trade(self,
                       portfolio_value: float,
                       entry_price: float,
                       stop_loss_price: float,
                       position_size: int,
                       direction: str = 'LONG') -> Tuple[bool, str]:
        """
        Validate a proposed trade against risk limits.

        Args:
            portfolio_value: Current portfolio value
            entry_price: Entry price
            stop_loss_price: Stop loss price
            position_size: Proposed position size
            direction: 'LONG' or 'SHORT'

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check position count
        can_open, reason = self.can_open_position()
        if not can_open:
            return False, reason

        # Check position value doesn't exceed max
        position_value = entry_price * position_size
        max_value = portfolio_value * self.params.max_position_pct
        if position_value > max_value:
            return False, f"Position value {position_value:.2f} exceeds max {max_value:.2f}"

        # Check risk per trade
        if direction == 'LONG':
            risk_per_share = entry_price - stop_loss_price
        else:
            risk_per_share = stop_loss_price - entry_price

        total_risk = risk_per_share * position_size
        max_risk = portfolio_value * self.params.max_risk_per_trade
        if total_risk > max_risk * 1.1:  # 10% tolerance
            return False, f"Risk {total_risk:.2f} exceeds max {max_risk:.2f}"

        return True, "OK"

    def get_risk_metrics(self,
                         positions: list,
                         portfolio_value: float) -> dict:
        """
        Calculate current risk metrics.

        Args:
            positions: List of current positions
            portfolio_value: Current portfolio value

        Returns:
            Dict with risk metrics
        """
        total_at_risk = 0.0
        total_exposure = 0.0

        for pos in positions:
            entry = pos.get('entry_price', 0)
            stop = pos.get('stop_loss', 0)
            qty = pos.get('quantity', 0)
            direction = pos.get('direction', 'LONG')

            if direction == 'LONG':
                risk = (entry - stop) * qty
            else:
                risk = (stop - entry) * qty

            total_at_risk += max(0, risk)
            total_exposure += entry * qty

        return {
            'num_positions': len(positions),
            'total_at_risk': total_at_risk,
            'risk_pct': total_at_risk / portfolio_value if portfolio_value > 0 else 0,
            'total_exposure': total_exposure,
            'exposure_pct': total_exposure / portfolio_value if portfolio_value > 0 else 0,
            'available_positions': self.params.max_concurrent_positions - len(positions)
        }

    def adjust_for_volatility(self,
                              position_size: int,
                              current_atr: float,
                              baseline_atr: float) -> int:
        """
        Adjust position size for current volatility.

        Reduces size when volatility is high, increases when low.

        Args:
            position_size: Base position size
            current_atr: Current ATR
            baseline_atr: Normal/baseline ATR

        Returns:
            Adjusted position size
        """
        if baseline_atr <= 0 or current_atr <= 0:
            return position_size

        # Volatility ratio
        vol_ratio = baseline_atr / current_atr

        # Adjust (cap at 0.5x to 2x)
        vol_ratio = max(0.5, min(2.0, vol_ratio))

        return int(position_size * vol_ratio)
