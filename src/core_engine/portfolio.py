"""
Portfolio Manager
=================

Manages portfolio state and position tracking.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis"""
        return self.quantity * self.avg_price


class Portfolio:
    """
    Portfolio manager for tracking positions and performance.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
    def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float
    ) -> bool:
        """
        Add a new position or update existing.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to add (positive for long, negative for short)
            price: Execution price
            
        Returns:
            bool: True if position added successfully
        """
        cost = quantity * price
        
        if cost > self.cash:
            logger.warning(f"Insufficient cash to add position: {cost:.2f} > {self.cash:.2f}")
            return False
        
        if symbol in self.positions:
            # Update existing position
            pos = self.positions[symbol]
            total_qty = pos.quantity + quantity
            total_cost = (pos.quantity * pos.avg_price) + cost
            
            if abs(total_qty) < 1e-6:  # Position closed
                del self.positions[symbol]
            else:
                pos.quantity = total_qty
                pos.avg_price = total_cost / total_qty if total_qty != 0 else 0
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price
            )
        
        # Update cash
        self.cash -= cost
        
        # Record trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'cost': cost
        })
        
        logger.info(f"Added position: {quantity:.4f} {symbol} @ {price:.2f}")
        return True
    
    def close_position(self, symbol: str, price: float) -> Optional[float]:
        """
        Close a position completely.
        
        Args:
            symbol: Trading symbol
            price: Exit price
            
        Returns:
            Realized P&L, or None if position not found
        """
        if symbol not in self.positions:
            logger.warning(f"Position not found for {symbol}")
            return None
        
        pos = self.positions[symbol]
        pnl = (price - pos.avg_price) * pos.quantity
        
        self.cash += price * pos.quantity
        
        # Record trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': -pos.quantity,  # Negative to close
            'price': price,
            'pnl': pnl
        })
        
        del self.positions[symbol]
        
        logger.info(f"Closed position: {symbol}, P&L: {pnl:.2f}")
        return pnl
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all positions.
        
        Args:
            prices: Dictionary of symbol -> current price
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
                position.unrealized_pnl = (
                    prices[symbol] - position.avg_price
                ) * position.quantity
    
    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        positions_value = sum(
            pos.market_value for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_realized_pnl(self) -> float:
        """Get total realized P&L from closed positions"""
        return sum(
            trade.get('pnl', 0) for trade in self.trade_history
            if 'pnl' in trade
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'positions_value': sum(pos.market_value for pos in self.positions.values()),
            'total_value': self.get_total_value(),
            'unrealized_pnl': self.get_unrealized_pnl(),
            'realized_pnl': self.get_realized_pnl(),
            'total_pnl': self.get_total_value() - self.initial_capital,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history)
        }
