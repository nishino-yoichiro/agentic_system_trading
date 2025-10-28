"""
Alpaca Broker
=============

Order execution adapter for Alpaca equities trading.
Handles order placement, status checking, and fills.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest
    )
    from alpaca.trading.enums import (
        OrderSide as AlpacaOrderSide,
        OrderType as AlpacaOrderType,
        TimeInForce,
        OrderStatus as AlpacaOrderStatus
    )
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Fix relative import issue
try:
    from ..core_engine.market_adapter import Order, OrderType, OrderSide, OrderStatus
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core_engine.market_adapter import Order, OrderType, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


class AlpacaBroker:
    """
    Broker interface for Alpaca order execution.
    
    Handles:
    - Market/Limit/Stop orders
    - Fill notifications
    - Order status tracking
    - Account balance monitoring
    """
    
    def __init__(self, paper: bool = True):
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py not installed. Install with: pip install alpaca-py"
            )
        
        self.paper = paper
        
        # API credentials
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca credentials required. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )
        
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Initialize trading client
        self.trading_client = TradingClient(
            self.api_key,
            self.secret_key,
            paper=paper
        )
        
        logger.info(f"Initialized Alpaca broker (paper={paper})")
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> Order:
        """
        Place an order on Alpaca.
        
        Args:
            symbol: Trading symbol (e.g., 'SPY')
            side: Order side (buy or sell)
            quantity: Number of shares
            order_type: Type of order (market, limit, stop, etc.)
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Order time-in-force
            
        Returns:
            Order: Order object with ID and status
        """
        try:
            # Convert to Alpaca types
            alpaca_side = AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL
            
            # Create order request based on type
            if order_type == OrderType.MARKET:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=time_in_force
                )
                
            elif order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Limit order requires price parameter")
                
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    limit_price=price,
                    time_in_force=time_in_force
                )
                
            elif order_type == OrderType.STOP:
                if stop_price is None:
                    raise ValueError("Stop order requires stop_price parameter")
                
                order_request = StopOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    stop_price=stop_price,
                    time_in_force=time_in_force
                )
                
            elif order_type == OrderType.STOP_LIMIT:
                if price is None or stop_price is None:
                    raise ValueError(
                        "Stop-limit order requires both price and stop_price parameters"
                    )
                
                order_request = StopLimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    limit_price=price,
                    stop_price=stop_price,
                    time_in_force=time_in_force
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            # Submit order
            alpaca_order = self.trading_client.submit_order(order_request)
            
            # Convert to our Order object
            order = Order(
                order_id=alpaca_order.client_order_id or str(alpaca_order.id),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=float(alpaca_order.qty),
                price=float(alpaca_order.limit_price) if hasattr(alpaca_order, 'limit_price') else None,
                stop_price=float(alpaca_order.stop_price) if hasattr(alpaca_order, 'stop_price') else None,
                status=self._convert_order_status(alpaca_order.status),
                timestamp=alpaca_order.created_at
            )
            
            logger.info(
                f"Placed {order_type.value} order: {side.value} {quantity} {symbol} "
                f"(ID: {order.order_id})"
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            OrderStatus: Current order status
        """
        try:
            alpaca_order = self.trading_client.get_order_by_client_id(order_id)
            return self._convert_order_status(alpaca_order.status)
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return OrderStatus.REJECTED
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            bool: True if cancellation successful
        """
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Cancelled order: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_all_orders(
        self,
        status_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all orders.
        
        Args:
            status_filter: Filter by status ('all', 'open', 'closed')
            limit: Maximum number of orders to return
            
        Returns:
            List of order dictionaries
        """
        try:
            params = {'limit': limit}
            
            if status_filter:
                params['status'] = status_filter
            
            orders = self.trading_client.get_orders(params)
            
            return [
                {
                    'order_id': str(order.id),
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'qty': float(order.qty),
                    'status': order.status.value,
                    'filled_qty': float(order.filled_qty),
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                    'created_at': order.created_at.isoformat()
                }
                for order in orders
            ]
            
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List of position dictionaries
        """
        try:
            positions = self.trading_client.get_all_positions()
            
            return [
                {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl) if hasattr(pos, 'unrealized_pl') else 0.0
                }
                for pos in positions
            ]
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account info
        """
        try:
            account = self.trading_client.get_account()
            
            return {
                'account_number': account.account_number,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'equity': float(account.equity),
                'day_trading_buying_power': float(account.daytrading_buying_power),
                'pattern_day_trader': account.pattern_day_trader,
                'portfolio_value': float(account.portfolio_value),
                'status': account.status
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def _convert_order_status(self, alpaca_status: AlpacaOrderStatus) -> OrderStatus:
        """
        Convert Alpaca order status to our OrderStatus enum.
        
        Args:
            alpaca_status: Alpaca order status
            
        Returns:
            OrderStatus: Converted status
        """
        status_map = {
            'new': OrderStatus.PENDING,
            'pending_new': OrderStatus.PENDING,
            'accepted': OrderStatus.PENDING,
            'pending_replace': OrderStatus.PENDING,
            'partially_filled': OrderStatus.PARTIAL,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.CANCELLED,
            'replaced': OrderStatus.CANCELLED,
            'pending_cancel': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
            'cancelled': OrderStatus.CANCELLED
        }
        
        return status_map.get(alpaca_status.value.lower(), OrderStatus.REJECTED)
