"""
Enhanced Crypto Trading Pipeline - Trading Logic Module

This module handles intelligent trading decisions and portfolio management:
- Multi-source signal fusion
- Portfolio optimization and allocation
- Risk management and position sizing
- Backtesting and strategy validation
- Performance monitoring and reporting
"""

from .signal_generator import SignalGenerator, SignalFusion
from .portfolio_optimizer import PortfolioOptimizer
from .risk_manager import RiskManager
from .backtester import Backtester

__all__ = [
    'SignalGenerator',
    'SignalFusion',
    'PortfolioOptimizer',
    'RiskManager',
    'Backtester'
]

