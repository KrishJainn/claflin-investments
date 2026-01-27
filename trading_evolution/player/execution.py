"""
Trade Execution module.

Simulates realistic trade execution with:
- Slippage
- Commissions
- Order fills
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class ExecutionResult:
    """Result of trade execution."""
    executed: bool
    fill_price: float
    quantity: int
    slippage_cost: float
    commission: float
    total_cost: float
    reason: str = ""


class ExecutionEngine:
    """
    Simulates trade execution with realistic costs.

    Models:
    - Slippage: Price impact from market orders
    - Commission: Fixed or percentage based
    - Partial fills: Not implemented (always full fill for simplicity)
    """

    def __init__(self,
                 slippage_pct: float = 0.001,
                 commission_pct: float = 0.0,
                 commission_per_share: float = 0.0,
                 min_commission: float = 0.0):
        """
        Initialize execution engine.

        Args:
            slippage_pct: Slippage as percentage of price (default 0.1%)
            commission_pct: Commission as percentage of trade value
            commission_per_share: Commission per share
            min_commission: Minimum commission per trade
        """
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

    def execute_market_order(self,
                             side: OrderSide,
                             price: float,
                             quantity: int,
                             volatility: float = None) -> ExecutionResult:
        """
        Execute a market order with slippage.

        Args:
            side: BUY or SELL
            price: Current market price
            quantity: Number of shares
            volatility: Optional volatility for dynamic slippage

        Returns:
            ExecutionResult with fill details
        """
        if quantity <= 0:
            return ExecutionResult(
                executed=False,
                fill_price=0,
                quantity=0,
                slippage_cost=0,
                commission=0,
                total_cost=0,
                reason="Invalid quantity"
            )

        # Calculate slippage
        slippage_factor = self._calculate_slippage(side, volatility)

        if side == OrderSide.BUY:
            fill_price = price * (1 + slippage_factor)
        else:
            fill_price = price * (1 - slippage_factor)

        # Calculate slippage cost
        slippage_cost = abs(fill_price - price) * quantity

        # Calculate commission
        commission = self._calculate_commission(fill_price, quantity)

        # Total cost
        trade_value = fill_price * quantity
        total_cost = trade_value + commission

        return ExecutionResult(
            executed=True,
            fill_price=fill_price,
            quantity=quantity,
            slippage_cost=slippage_cost,
            commission=commission,
            total_cost=total_cost
        )

    def execute_entry(self,
                      direction: str,
                      price: float,
                      quantity: int) -> ExecutionResult:
        """
        Execute entry trade.

        Args:
            direction: 'LONG' or 'SHORT'
            price: Entry price
            quantity: Position size

        Returns:
            ExecutionResult
        """
        if direction == 'LONG':
            return self.execute_market_order(OrderSide.BUY, price, quantity)
        else:
            return self.execute_market_order(OrderSide.SELL, price, quantity)

    def execute_exit(self,
                     direction: str,
                     price: float,
                     quantity: int) -> ExecutionResult:
        """
        Execute exit trade.

        Args:
            direction: 'LONG' or 'SHORT' (position being closed)
            price: Exit price
            quantity: Position size

        Returns:
            ExecutionResult
        """
        if direction == 'LONG':
            # Closing long = selling
            return self.execute_market_order(OrderSide.SELL, price, quantity)
        else:
            # Closing short = buying to cover
            return self.execute_market_order(OrderSide.BUY, price, quantity)

    def _calculate_slippage(self, side: OrderSide, volatility: float = None) -> float:
        """
        Calculate slippage factor.

        Can be enhanced with:
        - Volatility-based slippage
        - Volume-based slippage
        - Time-of-day effects
        """
        base_slippage = self.slippage_pct

        # Optionally adjust for volatility
        if volatility is not None and volatility > 0:
            # Higher volatility = more slippage
            volatility_multiplier = min(2.0, max(0.5, volatility / 0.02))
            base_slippage *= volatility_multiplier

        return base_slippage

    def _calculate_commission(self, price: float, quantity: int) -> float:
        """Calculate commission for trade."""
        # Percentage-based commission
        pct_commission = price * quantity * self.commission_pct

        # Per-share commission
        per_share_commission = quantity * self.commission_per_share

        # Total
        total = pct_commission + per_share_commission

        # Apply minimum
        return max(total, self.min_commission)

    def estimate_costs(self,
                       price: float,
                       quantity: int,
                       is_round_trip: bool = True) -> dict:
        """
        Estimate execution costs for a trade.

        Args:
            price: Trade price
            quantity: Number of shares
            is_round_trip: Include exit costs

        Returns:
            Dict with cost breakdown
        """
        trade_value = price * quantity

        # Entry costs
        entry_slippage = trade_value * self.slippage_pct
        entry_commission = self._calculate_commission(price, quantity)

        if is_round_trip:
            # Exit costs (assume similar)
            exit_slippage = entry_slippage
            exit_commission = entry_commission
            total_slippage = entry_slippage + exit_slippage
            total_commission = entry_commission + exit_commission
        else:
            total_slippage = entry_slippage
            total_commission = entry_commission

        return {
            'trade_value': trade_value,
            'total_slippage': total_slippage,
            'total_commission': total_commission,
            'total_costs': total_slippage + total_commission,
            'cost_pct': (total_slippage + total_commission) / trade_value
        }


def simulate_fill_price(base_price: float,
                        side: str,
                        slippage_pct: float = 0.001) -> float:
    """
    Simple function to simulate fill price with slippage.

    Args:
        base_price: Market price
        side: 'BUY' or 'SELL'
        slippage_pct: Slippage percentage

    Returns:
        Simulated fill price
    """
    if side.upper() == 'BUY':
        return base_price * (1 + slippage_pct)
    else:
        return base_price * (1 - slippage_pct)
