"""
Paper Execution Simulator.

Simulates live trade execution with realistic fills:
- Bid-ask spread modeling
- Volatility-based slippage
- Partial fill simulation (optional)
- Full cost modeling using Indian market costs
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import uuid

from ..backtest.indian_costs import IndianCostModel, CostBreakdown

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Order submitted for execution."""
    
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    order_id: str = ""  # Auto-generated if empty
    limit_price: Optional[float] = None
    
    # Metadata
    strategy_version: str = ""
    signal_value: float = 0.0
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0


@dataclass
class PaperFill:
    """Simulated order fill."""
    
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    
    # Prices
    requested_price: float
    fill_price: float
    slippage: float
    
    # Quantity
    quantity: int
    
    # Costs
    cost_breakdown: CostBreakdown
    total_cost: float
    
    # Timing
    timestamp: datetime
    
    # Metadata
    strategy_version: str = ""
    reason: str = ""  # Why this trade was taken
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'requested_price': self.requested_price,
            'fill_price': self.fill_price,
            'slippage': self.slippage,
            'quantity': self.quantity,
            'total_cost': self.total_cost,
            'timestamp': self.timestamp.isoformat(),
            'strategy_version': self.strategy_version,
            'reason': self.reason,
        }


class PaperExecutor:
    """
    Paper execution simulator with realistic fills.
    
    Features:
    - Spread-based fill prices (mid ± spread/2)
    - Volatility-adjusted slippage
    - Indian market cost model
    - Partial fills (optional)
    - Full audit trail
    """
    
    def __init__(
        self,
        cost_model: IndianCostModel = None,
        base_spread_pct: float = 0.001,  # 0.1% default spread
        volatility_slippage_mult: float = 1.5,
        enable_partial_fills: bool = False,
    ):
        """
        Initialize paper executor.
        
        Args:
            cost_model: Cost model (defaults to Indian intraday)
            base_spread_pct: Base bid-ask spread percentage
            volatility_slippage_mult: Multiplier for volatility-based slippage
            enable_partial_fills: Whether to simulate partial fills
        """
        self.cost_model = cost_model or IndianCostModel.for_intraday()
        self.base_spread_pct = base_spread_pct
        self.volatility_slippage_mult = volatility_slippage_mult
        self.enable_partial_fills = enable_partial_fills
        
        # Order tracking
        self._pending_orders: Dict[str, Order] = {}
        self._fills: List[PaperFill] = []
        self._order_history: List[Order] = []
    
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float = None,
        strategy_version: str = "",
        signal_value: float = 0.0,
        reason: str = "",
    ) -> Order:
        """
        Submit an order for execution.
        
        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Number of shares
            order_type: MARKET or LIMIT
            limit_price: Limit price (for limit orders)
            strategy_version: Version of strategy placing order
            signal_value: Super Indicator value at order time
            reason: Reason for the trade
            
        Returns:
            Order object
        """
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            strategy_version=strategy_version,
            signal_value=signal_value,
            reason=reason,
            timestamp=datetime.now(),
        )
        
        self._pending_orders[order.order_id] = order
        self._order_history.append(order)
        
        logger.debug(f"Order submitted: {order.order_id} {side.value} {quantity} {symbol}")
        
        return order
    
    def execute_order(
        self,
        order: Order,
        current_price: float,
        volatility: float = None,
        bid: float = None,
        ask: float = None,
    ) -> PaperFill:
        """
        Execute an order at current price.
        
        Args:
            order: Order to execute
            current_price: Current market price
            volatility: Optional volatility for slippage scaling (ATR / price)
            bid: Optional bid price
            ask: Optional ask price
            
        Returns:
            PaperFill with execution details
        """
        if order.order_id in self._pending_orders:
            del self._pending_orders[order.order_id]
        
        # Calculate spread-based prices if not provided
        if bid is None or ask is None:
            spread_pct = self._calculate_spread(current_price, volatility)
            half_spread = current_price * spread_pct / 2
            bid = current_price - half_spread
            ask = current_price + half_spread
        
        # Calculate fill price based on side
        if order.side == OrderSide.BUY:
            # Buy at ask + slippage
            base_fill = ask
            slippage_pct = self._calculate_slippage(volatility, is_buy=True)
            fill_price = base_fill * (1 + slippage_pct)
        else:
            # Sell at bid - slippage
            base_fill = bid
            slippage_pct = self._calculate_slippage(volatility, is_buy=False)
            fill_price = base_fill * (1 - slippage_pct)
        
        slippage_amount = abs(fill_price - current_price) * order.quantity
        
        # Calculate costs
        if order.side == OrderSide.BUY:
            cost_breakdown = self.cost_model.calculate_entry_cost(
                fill_price, order.quantity, volatility
            )
        else:
            cost_breakdown = self.cost_model.calculate_exit_cost(
                fill_price, order.quantity, volatility
            )
        
        # Create fill
        fill = PaperFill(
            fill_id=str(uuid.uuid4())[:8],
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            requested_price=current_price,
            fill_price=fill_price,
            slippage=slippage_amount,
            quantity=order.quantity,
            cost_breakdown=cost_breakdown,
            total_cost=cost_breakdown.total_cost,
            timestamp=datetime.now(),
            strategy_version=order.strategy_version,
            reason=order.reason,
        )
        
        self._fills.append(fill)
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        
        logger.info(
            f"Order filled: {order.order_id} {order.side.value} {order.quantity} {order.symbol} "
            f"@ ₹{fill_price:.2f} (slippage: ₹{slippage_amount:.2f})"
        )
        
        return fill
    
    def execute(
        self,
        order: Order,
        current_price: float,
        volatility: float = None,
    ) -> PaperFill:
        """
        Convenience alias for execute_order.
        
        If order doesn't have an order_id, generates one.
        """
        if not order.order_id:
            order.order_id = str(uuid.uuid4())[:8]
        return self.execute_order(order, current_price, volatility)
    
    def execute_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        current_price: float,
        volatility: float = None,
        strategy_version: str = "",
        signal_value: float = 0.0,
        reason: str = "",
    ) -> PaperFill:
        """
        Submit and immediately execute a market order.
        
        Convenience method combining submit_order and execute_order.
        """
        order = self.submit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            strategy_version=strategy_version,
            signal_value=signal_value,
            reason=reason,
        )
        
        return self.execute_order(order, current_price, volatility)
    
    def _calculate_spread(
        self,
        price: float,
        volatility: float = None,
    ) -> float:
        """Calculate bid-ask spread percentage."""
        # Base spread varies by price level (liquidity proxy)
        if price < 100:
            base_spread = 0.003  # 0.3%
        elif price < 500:
            base_spread = 0.002  # 0.2%
        elif price < 2000:
            base_spread = 0.001  # 0.1%
        else:
            base_spread = 0.0005  # 0.05%
        
        # Adjust for volatility
        if volatility is not None and volatility > 0:
            # Higher volatility = wider spreads
            vol_mult = min(2.0, max(0.5, volatility / 0.015))  # Baseline 1.5% vol
            base_spread *= vol_mult
        
        return max(base_spread, self.base_spread_pct)
    
    def _calculate_slippage(
        self,
        volatility: float = None,
        is_buy: bool = True,
    ) -> float:
        """Calculate slippage percentage."""
        base_slippage = 0.0005  # 0.05% base
        
        if volatility is not None and volatility > 0:
            # Higher volatility = more slippage
            vol_mult = volatility / 0.015 * self.volatility_slippage_mult
            base_slippage *= max(0.5, min(3.0, vol_mult))
        
        return base_slippage
    
    def get_fills(self, symbol: str = None) -> List[PaperFill]:
        """Get all fills, optionally filtered by symbol."""
        if symbol:
            return [f for f in self._fills if f.symbol == symbol]
        return self._fills.copy()
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return list(self._pending_orders.values())
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self._pending_orders:
            order = self._pending_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            logger.info(f"Order cancelled: {order_id}")
            return True
        return False
    
    def cancel_all_pending(self) -> int:
        """Cancel all pending orders."""
        count = len(self._pending_orders)
        for order in self._pending_orders.values():
            order.status = OrderStatus.CANCELLED
        self._pending_orders.clear()
        return count
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics."""
        if not self._fills:
            return {'total_fills': 0}
        
        total_slippage = sum(f.slippage for f in self._fills)
        total_costs = sum(f.total_cost for f in self._fills)
        
        return {
            'total_fills': len(self._fills),
            'total_slippage': total_slippage,
            'avg_slippage': total_slippage / len(self._fills),
            'total_costs': total_costs,
            'avg_cost_per_trade': total_costs / len(self._fills),
        }
    
    def reset(self):
        """Reset executor state."""
        self._pending_orders.clear()
        self._fills.clear()
        self._order_history.clear()
