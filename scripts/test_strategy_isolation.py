#!/usr/bin/env python3
"""
Quick Test Strategy Test
========================

Test the test_every_minute strategy in isolation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_strategy_isolation():
    """Test the test strategy in isolation"""
    print("Testing test_every_minute strategy in isolation...")
    
    try:
        from src.test_every_minute_strategy import TestEveryMinuteStrategy
        from src.crypto_signal_framework import SignalType
        import pandas as pd
        from datetime import datetime, timezone
        
        # Create strategy
        strategy = TestEveryMinuteStrategy()
        print(f"Strategy created: {strategy.name}")
        
        # Create test data
        test_data = pd.DataFrame({
            'close': [50000, 50100],
            'volume': [1000, 1100]
        }, index=[
            datetime.now(timezone.utc),
            datetime.now(timezone.utc)
        ])
        
        print(f"Test data shape: {test_data.shape}")
        print(f"Test data columns: {list(test_data.columns)}")
        
        # Test signal generation
        for i, (timestamp, row) in enumerate(test_data.iterrows()):
            print(f"\nTesting row {i}:")
            print(f"  Timestamp: {timestamp}")
            print(f"  Close price: {row['close']}")
            
            signal = strategy.generate_signal(row)
            
            if signal:
                print(f"  ✅ Signal generated!")
                print(f"    Type: {signal.signal.name}")
                print(f"    Confidence: {signal.confidence}")
                print(f"    Entry price: {signal.entry_price}")
                print(f"    Reason: {signal.reason}")
                
                # Test signal type check
                print(f"    Signal != FLAT: {signal.signal != SignalType.FLAT}")
            else:
                print(f"  ❌ No signal generated")
        
        print("\nStrategy test completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_strategy_isolation()
