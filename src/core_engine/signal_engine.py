"""
Signal Engine
=============

Core signal generation engine that works across markets.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import logging

from .market_adapter import OHLCVBar

logger = logging.getLogger(__name__)


class SignalEngine:
    """
    Core signal generation engine.
    
    This class is market-agnostic and operates on OHLCV data.
    Strategies can be plugged in to generate signals from any market.
    """
    
    def __init__(self):
        self.signals_history = []
        self.strategies = {}
    
    def register_strategy(self, name: str, strategy_func):
        """
        Register a signal generation strategy.
        
        Args:
            name: Strategy name
            strategy_func: Function that takes OHLCVBar and returns signal dict
        """
        self.strategies[name] = strategy_func
        logger.info(f"Registered strategy: {name}")
    
    def generate_signal(
        self,
        bar: OHLCVBar,
        strategy_name: Optional[str] = None,
        historical_bars: Optional[List[OHLCVBar]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate signals from a bar.
        
        Args:
            bar: Current OHLCV bar
            strategy_name: Specific strategy to run (None for all)
            historical_bars: Recent bars for context
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        strategies_to_run = [strategy_name] if strategy_name else self.strategies.keys()
        
        for strategy in strategies_to_run:
            if strategy not in self.strategies:
                continue
            
            try:
                strategy_func = self.strategies[strategy]
                signal = strategy_func(bar, historical_bars or [])
                
                if signal:
                    signal['strategy'] = strategy
                    signal['timestamp'] = bar.timestamp
                    signal['symbol'] = bar.symbol
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error running strategy {strategy}: {e}")
        
        # Store in history
        for signal in signals:
            self.signals_history.append(signal)
        
        return signals
    
    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent signal history.
        
        Args:
            symbol: Filter by symbol
            limit: Maximum number of signals
            
        Returns:
            List of signal dictionaries
        """
        history = self.signals_history
        
        if symbol:
            history = [s for s in history if s.get('symbol') == symbol]
        
        return history[-limit:]
