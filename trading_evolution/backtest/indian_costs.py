"""
Indian Market Cost Model.

Complete transaction cost model for NSE equity trading:
- Brokerage (broker-specific)
- STT (Securities Transaction Tax)
- Exchange Transaction Charges
- SEBI Turnover Fee
- GST (on brokerage + exchange charges)
- Stamp Duty (state-specific)
- Slippage (market impact)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TradeType(Enum):
    """Trade type affects STT calculation."""
    DELIVERY = "delivery"      # Higher STT
    INTRADAY = "intraday"      # Lower STT
    FUTURES = "futures"
    OPTIONS = "options"


@dataclass
class CostBreakdown:
    """Detailed breakdown of transaction costs."""
    
    # Trade info
    trade_value: float
    quantity: int
    price: float
    side: str  # 'BUY' or 'SELL'
    
    # Individual costs
    brokerage: float = 0.0
    stt: float = 0.0
    exchange_charges: float = 0.0
    sebi_fee: float = 0.0
    gst: float = 0.0
    stamp_duty: float = 0.0
    slippage: float = 0.0
    
    # Totals
    total_statutory: float = 0.0
    total_cost: float = 0.0
    cost_pct: float = 0.0
    
    # Fill price after slippage
    fill_price: float = 0.0
    
    def __post_init__(self):
        """Calculate totals."""
        self.total_statutory = (
            self.stt + 
            self.exchange_charges + 
            self.sebi_fee + 
            self.gst + 
            self.stamp_duty
        )
        self.total_cost = self.brokerage + self.total_statutory + self.slippage
        if self.trade_value > 0:
            self.cost_pct = self.total_cost / self.trade_value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'trade_value': self.trade_value,
            'quantity': self.quantity,
            'price': self.price,
            'side': self.side,
            'brokerage': self.brokerage,
            'stt': self.stt,
            'exchange_charges': self.exchange_charges,
            'sebi_fee': self.sebi_fee,
            'gst': self.gst,
            'stamp_duty': self.stamp_duty,
            'slippage': self.slippage,
            'total_statutory': self.total_statutory,
            'total_cost': self.total_cost,
            'cost_pct': self.cost_pct,
            'fill_price': self.fill_price,
        }


@dataclass
class IndianCostModel:
    """
    Complete Indian equity market transaction costs (NSE).
    
    Default values are for a typical discount broker (Zerodha-like).
    All percentages are expressed as decimals (0.01 = 1%).
    
    Reference: https://zerodha.com/brokerage-calculator
    """
    
    # ===== Brokerage =====
    # Discount broker: typically 0.03% or ₹20 flat (whichever is lower for intraday)
    brokerage_pct: float = 0.0003  # 0.03%
    brokerage_flat: float = 20.0   # ₹20 flat per order (Zerodha)
    use_flat_brokerage: bool = True  # Use flat for intraday
    
    # ===== STT (Securities Transaction Tax) =====
    # Delivery: 0.1% on both buy and sell
    # Intraday: 0.025% on sell only
    stt_delivery_pct: float = 0.001    # 0.1%
    stt_intraday_pct: float = 0.00025  # 0.025% (sell only)
    
    # ===== Exchange Transaction Charges (NSE) =====
    # NSE: 0.00345% of turnover
    exchange_txn_pct: float = 0.0000345
    
    # ===== SEBI Turnover Fee =====
    # ₹10 per crore (0.0001%)
    sebi_fee_pct: float = 0.000001
    
    # ===== GST =====
    # 18% on (brokerage + exchange charges)
    gst_pct: float = 0.18
    
    # ===== Stamp Duty =====
    # Varies by state, charged on buy side only
    # Maharashtra: 0.01%, Karnataka: 0.015%, etc.
    stamp_duty_pct: float = 0.00015  # 0.015% (avg)
    stamp_duty_on_buy_only: bool = True
    
    # ===== Slippage Model =====
    base_slippage_pct: float = 0.0005  # 0.05% base
    volatility_slippage_mult: float = 1.0  # Multiplier for high vol
    volume_impact_enabled: bool = False  # Not implemented yet
    
    # ===== Trade Type =====
    trade_type: TradeType = TradeType.INTRADAY
    
    def calculate_entry_cost(
        self,
        price: float,
        quantity: int,
        volatility: float = None,
    ) -> CostBreakdown:
        """
        Calculate costs for entering a position (BUY).
        
        Args:
            price: Entry price per share
            quantity: Number of shares
            volatility: Optional ATR or volatility for slippage scaling
            
        Returns:
            CostBreakdown with all costs itemized
        """
        trade_value = price * quantity
        
        # Brokerage
        if self.use_flat_brokerage:
            brokerage = self.brokerage_flat
        else:
            brokerage = trade_value * self.brokerage_pct
        
        # STT - on buy only for delivery
        if self.trade_type == TradeType.DELIVERY:
            stt = trade_value * self.stt_delivery_pct
        else:
            stt = 0.0  # Intraday STT only on sell
        
        # Exchange charges
        exchange_charges = trade_value * self.exchange_txn_pct
        
        # SEBI fee
        sebi_fee = trade_value * self.sebi_fee_pct
        
        # GST on brokerage + exchange charges
        gst = (brokerage + exchange_charges) * self.gst_pct
        
        # Stamp duty (on buy only)
        stamp_duty = trade_value * self.stamp_duty_pct if self.stamp_duty_on_buy_only else 0.0
        
        # Slippage (buy at higher price)
        slippage_pct = self._calculate_slippage(volatility)
        slippage = trade_value * slippage_pct
        fill_price = price * (1 + slippage_pct)
        
        return CostBreakdown(
            trade_value=trade_value,
            quantity=quantity,
            price=price,
            side='BUY',
            brokerage=brokerage,
            stt=stt,
            exchange_charges=exchange_charges,
            sebi_fee=sebi_fee,
            gst=gst,
            stamp_duty=stamp_duty,
            slippage=slippage,
            fill_price=fill_price,
        )
    
    def calculate_exit_cost(
        self,
        price: float,
        quantity: int,
        volatility: float = None,
    ) -> CostBreakdown:
        """
        Calculate costs for exiting a position (SELL).
        
        Args:
            price: Exit price per share
            quantity: Number of shares
            volatility: Optional ATR or volatility for slippage scaling
            
        Returns:
            CostBreakdown with all costs itemized
        """
        trade_value = price * quantity
        
        # Brokerage
        if self.use_flat_brokerage:
            brokerage = self.brokerage_flat
        else:
            brokerage = trade_value * self.brokerage_pct
        
        # STT - always on sell
        if self.trade_type == TradeType.DELIVERY:
            stt = trade_value * self.stt_delivery_pct
        else:
            stt = trade_value * self.stt_intraday_pct
        
        # Exchange charges
        exchange_charges = trade_value * self.exchange_txn_pct
        
        # SEBI fee
        sebi_fee = trade_value * self.sebi_fee_pct
        
        # GST on brokerage + exchange charges
        gst = (brokerage + exchange_charges) * self.gst_pct
        
        # No stamp duty on sell
        stamp_duty = 0.0 if self.stamp_duty_on_buy_only else trade_value * self.stamp_duty_pct
        
        # Slippage (sell at lower price)
        slippage_pct = self._calculate_slippage(volatility)
        slippage = trade_value * slippage_pct
        fill_price = price * (1 - slippage_pct)
        
        return CostBreakdown(
            trade_value=trade_value,
            quantity=quantity,
            price=price,
            side='SELL',
            brokerage=brokerage,
            stt=stt,
            exchange_charges=exchange_charges,
            sebi_fee=sebi_fee,
            gst=gst,
            stamp_duty=stamp_duty,
            slippage=slippage,
            fill_price=fill_price,
        )
    
    def calculate_round_trip(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        volatility: float = None,
    ) -> Dict:
        """
        Calculate total costs for a round-trip trade.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Number of shares
            volatility: Optional volatility for slippage
            
        Returns:
            Dictionary with entry, exit, and total costs
        """
        entry_cost = self.calculate_entry_cost(entry_price, quantity, volatility)
        exit_cost = self.calculate_exit_cost(exit_price, quantity, volatility)
        
        # Gross P&L (before costs)
        gross_pnl = (exit_price - entry_price) * quantity
        
        # Net P&L (after costs)
        total_costs = entry_cost.total_cost + exit_cost.total_cost
        net_pnl = gross_pnl - total_costs
        
        return {
            'entry': entry_cost.to_dict(),
            'exit': exit_cost.to_dict(),
            'gross_pnl': gross_pnl,
            'total_costs': total_costs,
            'net_pnl': net_pnl,
            'cost_impact_pct': total_costs / (entry_price * quantity),
            'actual_entry_price': entry_cost.fill_price,
            'actual_exit_price': exit_cost.fill_price,
        }
    
    def _calculate_slippage(self, volatility: float = None) -> float:
        """Calculate slippage percentage."""
        slippage = self.base_slippage_pct
        
        if volatility is not None and volatility > 0:
            # Scale slippage with volatility
            # Assume baseline volatility of ~1.5% daily
            vol_ratio = volatility / 0.015
            slippage *= max(0.5, min(2.0, vol_ratio)) * self.volatility_slippage_mult
        
        return slippage
    
    def get_config(self) -> Dict:
        """Get cost model configuration as dictionary."""
        return {
            'trade_type': self.trade_type.value,
            'brokerage_pct': self.brokerage_pct,
            'brokerage_flat': self.brokerage_flat,
            'use_flat_brokerage': self.use_flat_brokerage,
            'stt_delivery_pct': self.stt_delivery_pct,
            'stt_intraday_pct': self.stt_intraday_pct,
            'exchange_txn_pct': self.exchange_txn_pct,
            'sebi_fee_pct': self.sebi_fee_pct,
            'gst_pct': self.gst_pct,
            'stamp_duty_pct': self.stamp_duty_pct,
            'base_slippage_pct': self.base_slippage_pct,
        }
    
    @classmethod
    def for_delivery(cls) -> 'IndianCostModel':
        """Create cost model for delivery trades."""
        return cls(
            trade_type=TradeType.DELIVERY,
            use_flat_brokerage=False,  # Delivery often percentage-based
        )
    
    @classmethod
    def for_intraday(cls) -> 'IndianCostModel':
        """Create cost model for intraday trades."""
        return cls(
            trade_type=TradeType.INTRADAY,
            use_flat_brokerage=True,
        )


# Convenience function
def calculate_trade_costs(
    entry_price: float,
    exit_price: float,
    quantity: int,
    trade_type: str = "intraday",
) -> Dict:
    """
    Quick function to calculate round-trip costs.
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        quantity: Number of shares
        trade_type: 'intraday' or 'delivery'
        
    Returns:
        Cost breakdown dictionary
    """
    if trade_type.lower() == "delivery":
        model = IndianCostModel.for_delivery()
    else:
        model = IndianCostModel.for_intraday()
    
    return model.calculate_round_trip(entry_price, exit_price, quantity)
