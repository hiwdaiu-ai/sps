"""
Trade execution engine with slippage modeling
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np


@dataclass
class Order:
    """Represents a trade order"""
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'limit' or 'market'
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Fill:
    """Represents a filled order"""
    order: Order
    fill_price: float
    fill_quantity: float
    commission: float
    slippage: float
    timestamp: str


class ExecutionEngine:
    """Handle order execution with slippage and commission modeling"""
    
    def __init__(self, use_limit_orders: bool = True, 
                 slippage_estimate: float = 0.001,
                 commission_rate: float = 0.0005):
        self.use_limit_orders = use_limit_orders
        self.slippage_estimate = slippage_estimate
        self.commission_rate = commission_rate
        self.fills = []
    
    def execute_order(self, order: Order, market_price: float) -> Fill:
        """
        Execute an order with simulated slippage and commission
        
        Args:
            order: Order to execute
            market_price: Current market price
            
        Returns:
            Fill object with execution details
        """
        # Simulate slippage
        if order.order_type == 'market':
            # Market orders get slippage
            slippage_factor = np.random.normal(self.slippage_estimate, 
                                              self.slippage_estimate * 0.5)
            if order.side == 'buy':
                fill_price = market_price * (1 + abs(slippage_factor))
            else:
                fill_price = market_price * (1 - abs(slippage_factor))
        else:
            # Limit orders execute at limit price (if filled)
            fill_price = order.price if order.price else market_price
        
        # Calculate commission
        commission = order.quantity * fill_price * self.commission_rate
        
        # Create fill
        fill = Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=abs(fill_price - market_price),
            timestamp=str(np.datetime64('now'))
        )
        
        self.fills.append(fill)
        return fill
    
    def place_order_with_risk_controls(self, symbol: str, side: str, 
                                      quantity: float, entry_price: float,
                                      stop_loss: float, take_profit: float,
                                      market_price: float) -> Fill:
        """
        Place order with stop-loss and take-profit
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Position size
            entry_price: Desired entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            market_price: Current market price
            
        Returns:
            Fill object
        """
        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            order_type='limit' if self.use_limit_orders else 'market',
            quantity=quantity,
            price=entry_price if self.use_limit_orders else None,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Execute
        return self.execute_order(order, market_price)
    
    def get_fills_summary(self) -> Dict:
        """Get summary of all fills"""
        if not self.fills:
            return {
                'total_fills': 0,
                'total_commission': 0,
                'avg_slippage': 0
            }
        
        return {
            'total_fills': len(self.fills),
            'total_commission': sum(f.commission for f in self.fills),
            'avg_slippage': np.mean([f.slippage for f in self.fills])
        }
