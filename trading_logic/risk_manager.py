"""
Risk Management System

Manages portfolio risk and position sizing
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger
from datetime import datetime, timedelta


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_size: float = 0.1  # 10% max position
    max_portfolio_risk: float = 0.05  # 5% max portfolio risk
    max_drawdown: float = 0.15  # 15% max drawdown
    max_correlation: float = 0.7  # 70% max correlation
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit


@dataclass
class Position:
    """Position information"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    side: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    risk_amount: float


class RiskManager:
    """Manage portfolio risk and position sizing"""
    
    def __init__(self, risk_limits: RiskLimits = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.positions = {}
        self.portfolio_value = 0.0
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, 
                              risk_amount: float) -> float:
        """Calculate position size based on risk"""
        try:
            if stop_loss <= 0 or entry_price <= 0:
                return 0.0
            
            # Risk per share
            risk_per_share = abs(entry_price - stop_loss)
            
            # Position size based on risk amount
            position_size = risk_amount / risk_per_share
            
            # Apply maximum position size limit
            max_position_value = self.portfolio_value * self.risk_limits.max_position_size
            max_position_size = max_position_value / entry_price
            
            position_size = min(position_size, max_position_size)
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def add_position(self, position: Position):
        """Add a new position"""
        self.positions[position.symbol] = position
        logger.info(f"Added position: {position.symbol} - {position.quantity} shares")
    
    def update_position(self, symbol: str, current_price: float):
        """Update position with current price"""
        if symbol in self.positions:
            self.positions[symbol].current_price = current_price
    
    def check_stop_loss(self, symbol: str) -> bool:
        """Check if position should be closed due to stop loss"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        if position.side == 'long':
            return position.current_price <= position.stop_loss
        else:  # short
            return position.current_price >= position.stop_loss
    
    def check_take_profit(self, symbol: str) -> bool:
        """Check if position should be closed due to take profit"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        if position.side == 'long':
            return position.current_price >= position.take_profit
        else:  # short
            return position.current_price <= position.take_profit
    
    def calculate_portfolio_risk(self) -> Dict[str, float]:
        """Calculate current portfolio risk metrics"""
        if not self.positions:
            return {'total_risk': 0, 'max_position_risk': 0, 'correlation_risk': 0}
        
        total_risk = 0
        max_position_risk = 0
        
        for position in self.positions.values():
            # Position risk
            position_value = position.quantity * position.current_price
            position_risk = position_value / self.portfolio_value
            
            total_risk += position_risk
            max_position_risk = max(max_position_risk, position_risk)
        
        return {
            'total_risk': total_risk,
            'max_position_risk': max_position_risk,
            'correlation_risk': 0  # Placeholder
        }
    
    def check_risk_limits(self) -> List[str]:
        """Check if any risk limits are exceeded"""
        violations = []
        
        portfolio_risk = self.calculate_portfolio_risk()
        
        # Check total portfolio risk
        if portfolio_risk['total_risk'] > self.risk_limits.max_portfolio_risk:
            violations.append(f"Portfolio risk {portfolio_risk['total_risk']:.2%} exceeds limit {self.risk_limits.max_portfolio_risk:.2%}")
        
        # Check maximum position size
        if portfolio_risk['max_position_risk'] > self.risk_limits.max_position_size:
            violations.append(f"Max position size {portfolio_risk['max_position_risk']:.2%} exceeds limit {self.risk_limits.max_position_size:.2%}")
        
        return violations
    
    def get_position_recommendations(self, symbol: str, current_price: float, 
                                   signal_strength: float) -> Dict[str, Any]:
        """Get position recommendations based on risk management"""
        try:
            # Calculate position size
            stop_loss = current_price * (1 - self.risk_limits.stop_loss_pct)
            risk_amount = self.portfolio_value * 0.01  # 1% risk per trade
            
            position_size = self.calculate_position_size(symbol, current_price, stop_loss, risk_amount)
            
            # Calculate take profit
            take_profit = current_price * (1 + self.risk_limits.take_profit_pct)
            
            # Determine if position should be taken
            can_take_position = (
                position_size > 0 and
                signal_strength > 0.5 and
                len(self.positions) < 10  # Max 10 positions
            )
            
            return {
                'can_take_position': can_take_position,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': risk_amount
            }
            
        except Exception as e:
            logger.error(f"Error getting position recommendations: {e}")
            return {
                'can_take_position': False,
                'position_size': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'risk_amount': 0
            }

