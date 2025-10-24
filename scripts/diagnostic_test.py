#!/usr/bin/env python3
"""
Simple Diagnostic Test
======================

Check if the test strategy works with the actual data format.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def check_data_and_strategy():
    """Check if data exists and strategy works"""
    print("=== DIAGNOSTIC TEST ===")
    
    # Check if data files exist
    data_files = [
        Path("data/BTC_1m_historical.parquet"),
        Path("data/crypto_db/BTC_historical.parquet")
    ]
    
    print("1. Checking data files:")
    for file_path in data_files:
        if file_path.exists():
            print(f"   ✅ Found: {file_path}")
            try:
                # Try to read the file
                import pandas as pd
                df = pd.read_parquet(file_path)
                print(f"      Shape: {df.shape}")
                print(f"      Columns: {list(df.columns)}")
                print(f"      Index type: {type(df.index)}")
                if not df.empty:
                    print(f"      Latest timestamp: {df.index.max()}")
            except Exception as e:
                print(f"      ❌ Error reading: {e}")
        else:
            print(f"   ❌ Missing: {file_path}")
    
    print("\n2. Testing strategy import:")
    try:
        from src.test_every_minute_strategy import TestEveryMinuteStrategy
        strategy = TestEveryMinuteStrategy()
        print(f"   ✅ Strategy imported: {strategy.name}")
        print(f"   ✅ Strategy type: {strategy.metadata.strategy_type.name}")
        print(f"   ✅ Strategy lookback: {strategy.metadata.lookback}")
    except Exception as e:
        print(f"   ❌ Error importing strategy: {e}")
    
    print("\n3. Testing signal framework:")
    try:
        from src.crypto_signal_framework import SignalFramework
        framework = SignalFramework()
        framework.add_strategy(strategy)
        print(f"   ✅ Framework created with {len(framework.strategies)} strategies")
        print(f"   ✅ Available strategies: {list(framework.strategies.keys())}")
    except Exception as e:
        print(f"   ❌ Error with framework: {e}")
    
    print("\n=== DIAGNOSTIC COMPLETE ===")

if __name__ == "__main__":
    check_data_and_strategy()
