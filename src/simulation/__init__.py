"""
Enhanced Crypto Trading Pipeline - Simulation Module

This module handles advanced Monte Carlo simulations and portfolio optimization:
- Monte Carlo simulations with correlation modeling
- Portfolio optimization using modern portfolio theory
- Risk metrics calculation (VaR, CVaR, drawdown)
- Market regime detection and adaptation
- Scenario analysis and stress testing
"""

from .portfolio_simulator import PortfolioSimulator, SimulationResult
from .correlation_engine import CorrelationEngine
from .regime_detector import RegimeDetector
from .risk_metrics import RiskCalculator

__all__ = [
    'PortfolioSimulator',
    'SimulationResult',
    'CorrelationEngine', 
    'RegimeDetector',
    'RiskCalculator'
]

