#!/usr/bin/env python3
"""
Test Every Minute Strategy
=========================

A test strategy that generates a BUY signal every minute for manual testing.
This should be removed after testing is complete.

Author: Quantitative Strategy Designer
Date: 2025-10-14
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Optional
import logging

from crypto_signal_framework import BaseStrategy, StrategyMetadata, StrategyType, Signal, SignalType

logger = logging.getLogger(__name__)

class TestEveryMinuteStrategy(BaseStrategy):
    """
    Test strategy that generates a BUY signal every minute
    """
    
    def __init__(self):
        metadata = StrategyMetadata(
            lookback=0,
            fields_required=["close", "timestamp"],
            strategy_type=StrategyType.CONSTANT_TIME,
            batch_mode=False,
            min_confidence=0.5,
            vol_target=0.10
        )
        
        super().__init__("test_every_minute", metadata)
    
    def generate_signal(self, current_row: pd.Series, history: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        """Generate a BUY signal every minute"""
        try:
            current_time = current_row.name
            current_price = current_row['close']
            
            # Always generate a BUY signal
            return Signal(
                signal_type=SignalType.LONG,
                confidence=0.9,
                entry_price=current_price,
                stop_loss=current_price * 0.95,
                take_profit=current_price * 1.05,
                reason=f"Test strategy - every minute signal at {current_time}",
                timestamp=current_time,
                strategy_name=self.name
            )
            
        except Exception as e:
            logger.error(f"Error generating test signal: {e}")
            return None

# Example usage and testing
if __name__ == "__main__":
    # Create strategy instance
    strategy = TestEveryMinuteStrategy()
    
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
        datetime.now(timezone.utc),
        datetime.now(timezone.utc),
        datetime.now(timezone.utc)
    ])
    
    print("\nTesting with sample data:")
    for timestamp, row in sample_data.iterrows():
        signal = strategy.generate_signal(row)
        if signal:
            print(f"{timestamp}: {signal.signal_type.name} at ${signal.entry_price:.0f} - {signal.reason}")
        else:
            print(f"{timestamp}: No signal")
