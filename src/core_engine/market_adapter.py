"""
Market Adapter Interface
========================

Abstract base class defining the interface that all market adapters must implement.
This allows the same core engine to work with different exchanges (Coinbase, Alpaca, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Optional, Dict, List, Any
from enum import Enum
import pandas as pd


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass
class OHLCVBar:
    """
    Unified OHLCV bar structure for cross-market compatibility
    """
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = "unknown"  # 'coinbase', 'alpaca', etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'source': self.source
        }
    
    def to_series(self) -> pd.Series:
        """Convert to pandas Series"""
        return pd.Series(self.to_dict())


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class MarketAdapter(ABC):
    """
    Abstract base class for market adapters.
    
    Each adapter handles exchange-specific details while providing
    a unified interface to the core trading engine.
    """
    
    def __init__(self, symbol: str, **kwargs):
        self.symbol = symbol
        self.config = kwargs
        self.is_connected = False
        self._last_bar: Optional[OHLCVBar] = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the market data source.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the market data source"""
        pass
    
    @abstractmethod
    async def stream_data(self, interval_seconds: int = 60) -> AsyncIterator[OHLCVBar]:
        """
        Stream OHLCV bars from the market.
        
        Args:
            interval_seconds: Bar interval in seconds (e.g., 60 for 1-minute bars)
            
        Yields:
            OHLCVBar: OHLCV bar data
        """
        pass
    
    @abstractmethod
    async def place_order(
        self,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """
        Place an order on the exchange.
        
        Args:
            side: Order side (buy or sell)
            quantity: Number of shares/units
            order_type: Type of order (market, limit, etc.)
            price: Limit price (if applicable)
            stop_price: Stop price (if applicable)
            
        Returns:
            Order: Order object with status
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get the current status of an order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            OrderStatus: Current order status
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            bool: True if cancellation successful
        """
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information (balance, buying power, etc.)
        
        Returns:
            Dict with account information
        """
        pass
    
    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        For crypto: always returns True
        For equities: checks if within trading hours
        
        Returns:
            bool: True if market is open
        """
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    def _create_bar(
        self,
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float
    ) -> OHLCVBar:
        """Helper method to create OHLCV bars"""
        return OHLCVBar(
            timestamp=timestamp,
            symbol=self.symbol,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            source=self.__class__.__name__.lower().replace('adapter', '')
        )
    
    def validate_bar(self, bar: OHLCVBar) -> bool:
        """
        Validate that a bar has reasonable values.
        
        Args:
            bar: OHLCVBar to validate
            
        Returns:
            bool: True if bar is valid
        """
        if not (0 < bar.open < 1e10 and
                0 < bar.high < 1e10 and
                0 < bar.low < 1e10 and
                0 < bar.close < 1e10 and
                0 <= bar.volume < 1e12):
            return False
        
        if not (bar.low <= bar.open <= bar.high and
                bar.low <= bar.close <= bar.high):
            return False
        
        if not (bar.low <= bar.high):
            return False
        
        return True
