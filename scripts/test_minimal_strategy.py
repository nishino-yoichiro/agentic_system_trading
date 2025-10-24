#!/usr/bin/env python3
"""
Minimal Test Strategy Test
==========================

Test just the test strategy with minimal setup.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_minimal():
    """Test the test strategy with minimal setup"""
    print("Testing test_every_minute strategy with minimal setup...")
    
    try:
        # Import required modules
        from src.test_every_minute_strategy import TestEveryMinuteStrategy
        from src.crypto_signal_framework import SignalFramework, SignalType
        import pandas as pd
        from datetime import datetime, timezone
        
        print("✅ Modules imported successfully")
        
        # Create strategy
        strategy = TestEveryMinuteStrategy()
        print(f"✅ Strategy created: {strategy.name}")
        
        # Create signal framework
        framework = SignalFramework()
        framework.add_strategy(strategy)
        print(f"✅ Framework created with {len(framework.strategies)} strategies")
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': [datetime.now(timezone.utc)],
            'open': [50000],
            'high': [50100],
            'low': [49900],
            'close': [50050],
            'volume': [1000]
        })
        print(f"✅ Test data created: {test_data.shape}")
        
        # Test signal generation
        data_dict = {'BTC': test_data}
        print(f"✅ Data dict created: {list(data_dict.keys())}")
        
        signals = framework.generate_signals(data_dict, strategies=['test_every_minute'])
        print(f"✅ Signal generation completed: {len(signals)} signals")
        
        for i, signal in enumerate(signals):
            if signal:
                print(f"  Signal {i}: {signal.signal.name} - {signal.strategy_name}")
                print(f"    Confidence: {signal.confidence}")
                print(f"    Entry price: {signal.entry_price}")
                print(f"    Reason: {signal.reason}")
            else:
                print(f"  Signal {i}: None")
        
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal()
