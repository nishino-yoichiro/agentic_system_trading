"""
Risk Manager
============

Centralized risk management for the trading system.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_size: float = 0.1  # Max 10% of capital per position
    max_total_exposure: float = 0.5  # Max 50% of capital across all positions
    max_daily_loss: float = 0.02  # Max 2% daily loss
    max_leverage: float = 1.0  # 1x leverage (no margin)
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit


class RiskManager:
    """
    Risk management for trading operations.
    
    Ensures all trades comply with risk limits.
    """
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.positions = {}
        
    def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        capital: float
    ) -> tuple[bool, str]:
        """
        Validate an order against risk limits.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Order price
            capital: Available capital
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check position size
        position_value = quantity * price
        position_pct = position_value / capital if capital > 0 else 0
        
        if position_pct > self.risk_limits.max_position_size:
            return False, f"Position size {position_pct:.2%} exceeds limit {self.risk_limits.max_position_size:.2%}"
        
        # Check total exposure
        total_exposure = sum(
            pos['quantity'] * pos['price']
            for pos in self.positions.values()
        )
        total_exposure_pct = total_exposure / capital if capital > 0 else 0
        
        if (total_exposure + position_value) / capital > self.risk_limits.max_total_exposure:
            return False, f"Total exposure would exceed limit {self.risk_limits.max_total_exposure:.2%}"
        
        # Check daily loss
        if self.daily_pnl < -self.risk_limits.max_daily_loss * capital:
            return False, "Daily loss limit exceeded"
        
        return True, "Order passed risk checks"
    
    def record_trade(self, symbol: str, side: str, quantity: float, price: float):
        """Record a completed trade"""
        key = f"{symbol}_{side}"
        
        if key in self.positions:
            self.positions[key]['quantity'] += quantity
        else:
            self.positions[key] = {
                'quantity': quantity,
                'price': price
            }
        
        self.daily_trades += 1
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L"""
        self.daily_pnl += pnl
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.positions = {}
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary"""
        return {
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'positions': len(self.positions),
            'limits': self.risk_limits
        }
