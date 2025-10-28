"""
Core Engine Interfaces
=====================

Abstract base classes for market adapters and trading infrastructure.
These interfaces allow the same strategy engine to work across different markets.
"""

from .market_adapter import MarketAdapter, OHLCVBar, OrderType, OrderStatus, OrderSide
from .signal_engine import SignalEngine
from .risk_manager import RiskManager, RiskLimits
from .portfolio import Portfolio, Position

__all__ = [
    'MarketAdapter',
    'OHLCVBar',
    'OrderType',
    'OrderStatus',
    'OrderSide',
    'SignalEngine',
    'RiskManager',
    'RiskLimits',
    'Portfolio',
    'Position'
]
