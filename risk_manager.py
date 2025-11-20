"""
Risk management and position sizing
"""

import numpy as np
from typing import Dict, Optional


class RiskManager:
    """Handle position sizing, stop-loss, and risk controls"""
    
    def __init__(self, max_position_size: float = 0.02, 
                 stop_loss_atr_mult: float = 2.0,
                 take_profit_atr_mult: float = 3.0,
                 max_daily_loss: float = 0.05,
                 max_concurrent: int = 3):
        self.max_position_size = max_position_size
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.max_daily_loss = max_daily_loss
        self.max_concurrent = max_concurrent
        
        self.daily_pnl = 0.0
        self.active_positions = 0
    
    def calculate_position_size(self, capital: float, entry_price: float, 
                               stop_loss: float) -> float:
        """
        Calculate position size using fixed fractional method
        
        Args:
            capital: Total capital
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Position size in base currency
        """
        # Risk amount per trade
        risk_amount = capital * self.max_position_size
        
        # Risk per unit
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        # Position size
        position_size = risk_amount / price_risk
        
        return position_size
    
    def calculate_stops(self, entry_price: float, atr: float, 
                       is_long: bool) -> Dict[str, float]:
        """
        Calculate stop-loss and take-profit levels
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            is_long: True for long position, False for short
            
        Returns:
            Dictionary with stop_loss and take_profit prices
        """
        if is_long:
            stop_loss = entry_price - (atr * self.stop_loss_atr_mult)
            take_profit = entry_price + (atr * self.take_profit_atr_mult)
        else:
            stop_loss = entry_price + (atr * self.stop_loss_atr_mult)
            take_profit = entry_price - (atr * self.take_profit_atr_mult)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def can_trade(self, capital: float) -> bool:
        """
        Check if trading is allowed based on risk limits
        
        Args:
            capital: Current capital
            
        Returns:
            True if trading is allowed
        """
        # Check daily loss limit
        if abs(self.daily_pnl) >= capital * self.max_daily_loss:
            return False
        
        # Check concurrent position limit
        if self.active_positions >= self.max_concurrent:
            return False
        
        return True
    
    def update_pnl(self, pnl: float):
        """Update daily P&L"""
        self.daily_pnl += pnl
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_pnl = 0.0
    
    def open_position(self):
        """Increment active position count"""
        self.active_positions += 1
    
    def close_position(self):
        """Decrement active position count"""
        self.active_positions = max(0, self.active_positions - 1)
