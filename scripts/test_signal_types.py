#!/usr/bin/env python3
"""
Test Signal Types Script
========================

Quick test to verify signal types are working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_signal_types():
    """Test signal type definitions and mappings"""
    print("Testing signal types...")
    
    try:
        from src.crypto_signal_framework import SignalType, Signal
        from src.test_every_minute_strategy import TestEveryMinuteStrategy
        import pandas as pd
        from datetime import datetime, timezone
        
        print(f"Available signal types: {[t.name for t in SignalType]}")
        print(f"Signal type values: {[t.value for t in SignalType]}")
        
        # Test the test strategy
        strategy = TestEveryMinuteStrategy()
        print(f"Test strategy name: {strategy.name}")
        
        # Create test data
        test_data = pd.DataFrame({
            'close': [50000],
            'volume': [1000]
        }, index=[datetime.now(timezone.utc)])
        
        # Generate a test signal
        signal = strategy.generate_signal(test_data.iloc[0])
        
        if signal:
            print(f"Generated signal: {signal.signal.name}")
            print(f"Signal type: {signal.signal}")
            print(f"Signal value: {signal.signal.value}")
            print(f"Confidence: {signal.confidence}")
            print(f"Entry price: {signal.entry_price}")
            print(f"Reason: {signal.reason}")
            
            # Test signal type checks
            print(f"\nSignal type checks:")
            print(f"signal.signal == SignalType.LONG: {signal.signal == SignalType.LONG}")
            print(f"signal.signal == SignalType.SHORT: {signal.signal == SignalType.SHORT}")
            print(f"signal.signal == SignalType.FLAT: {signal.signal == SignalType.FLAT}")
            print(f"signal.signal != SignalType.FLAT: {signal.signal != SignalType.FLAT}")
            
        else:
            print("No signal generated!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_signal_types()
