#!/usr/bin/env python3
"""
Professional BTC NY Session Strategy
====================================

Demonstrates the new metadata-driven approach with constant-time complexity.
"""

import pandas as pd
from datetime import datetime
from typing import Optional
import logging

from crypto_signal_framework import BaseStrategy, StrategyMetadata, StrategyType, Signal, SignalType

logger = logging.getLogger(__name__)

class BTCNYSessionStrategy(BaseStrategy):
    """
    BTC NY Session Strategy - Professional implementation
    
    This strategy demonstrates the new metadata-driven approach:
    - Declares it only needs current price/time (CONSTANT_TIME)
    - Framework only processes relevant timestamps (NY open/close)
    - O(1) complexity regardless of data size
    """
    
    def __init__(self):
        # Declare strategy requirements upfront
        metadata = StrategyMetadata(
            lookback=0,  # No historical data needed
            fields_required=["close", "timestamp"],  # Only need current price and time
            strategy_type=StrategyType.CONSTANT_TIME,  # Constant-time strategy
            batch_mode=False,  # Process one timestamp at a time
            min_confidence=0.8,  # High confidence threshold
            vol_target=0.10  # 10% volatility target
        )
        
        super().__init__("btc_ny_session", metadata)
        
        # Strategy parameters
        self.ny_open_hour = 9
        self.ny_open_minute = 30
        self.ny_close_hour = 16
        self.ny_close_minute = 30
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.05  # 5% take profit
    
    def generate_signal(self, current_row: pd.Series, history: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        """
        Generate signal based on current price and time only
        
        Args:
            current_row: Current market data (only needs 'close' and timestamp)
            history: Not used for this strategy (None)
            
        Returns:
            Signal object or None
        """
        try:
            current_time = current_row.name  # Timestamp from index
            current_price = current_row['close']
            
            # NY market hours (9:30 AM - 4:30 PM ET)
            ny_open_time = current_time.replace(
                hour=self.ny_open_hour, 
                minute=self.ny_open_minute, 
                second=0, 
                microsecond=0
            )
            ny_close_time = current_time.replace(
                hour=self.ny_close_hour, 
                minute=self.ny_close_minute, 
                second=0, 
                microsecond=0
            )
            
            # Check if we're in NY session
            if ny_open_time <= current_time <= ny_close_time:
                # Buy signal at NY open
                if (current_time.hour == self.ny_open_hour and 
                    current_time.minute == self.ny_open_minute):
                    
                    return Signal(
                        signal_type=SignalType.LONG,
                        confidence=0.80,
                        entry_price=current_price,
                        stop_loss=current_price * (1 - self.stop_loss_pct),
                        take_profit=current_price * (1 + self.take_profit_pct),
                        reason="NY session open - Buy signal",
                        timestamp=current_time,
                        strategy_name=self.name
                    )
                
                # Sell signal at NY close
                elif (current_time.hour == self.ny_close_hour and 
                      current_time.minute == self.ny_close_minute):
                    
                    return Signal(
                        signal_type=SignalType.SHORT,
                        confidence=0.80,
                        entry_price=current_price,
                        stop_loss=current_price * (1 + self.stop_loss_pct),
                        take_profit=current_price * (1 - self.take_profit_pct),
                        reason="NY session close - Sell signal",
                        timestamp=current_time,
                        strategy_name=self.name
                    )
            
            # No signal outside NY session times
            return None
            
        except Exception as e:
            logger.error(f"Error generating BTC NY session signal: {e}")
            return None

# Example usage and testing
if __name__ == "__main__":
    # Create strategy instance
    strategy = BTCNYSessionStrategy()
    
    # Display strategy metadata
    metadata = strategy.get_data_requirements()
    print(f"Strategy: {strategy.name}")
    print(f"Type: {metadata.strategy_type.name}")
    print(f"Lookback: {metadata.lookback}")
    print(f"Fields required: {metadata.fields_required}")
    print(f"Min confidence: {metadata.min_confidence}")
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'close': [50000, 50100, 50200],
        'volume': [1000, 1100, 1200]
    }, index=[
        datetime(2024, 1, 1, 9, 30),  # NY open
        datetime(2024, 1, 1, 12, 0),  # Mid session
        datetime(2024, 1, 1, 16, 30)  # NY close
    ])
    
    print("\nTesting with sample data:")
    for timestamp, row in sample_data.iterrows():
        signal = strategy.generate_signal(row)
        if signal:
            print(f"{timestamp}: {signal.signal_type.name} at ${signal.entry_price:.0f} - {signal.reason}")
        else:
            print(f"{timestamp}: No signal")
