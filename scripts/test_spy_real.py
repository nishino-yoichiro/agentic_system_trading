#!/usr/bin/env python3
"""Test SPY adapter with real Alpaca API keys"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adapters import AlpacaMarketAdapter
import logging

logging.basicConfig(level=logging.INFO)  # Show details
logger = logging.getLogger(__name__)


async def test():
    """Test SPY adapter with real keys"""
    
    print("=" * 80)
    print("Testing SPY Adapter with REAL Alpaca API Keys")
    print("=" * 80)
    
    adapter = AlpacaMarketAdapter(symbol="SPY", paper=True)
    
    try:
        # Connect
        print("\n[1] Connecting to Alpaca...")
        result = await adapter.connect()
        print(f"    Result: {'SUCCESS' if result else 'FAILED'}")
        print(f"    Historical client initialized: {adapter.historical_client is not None}")
        
        if not adapter.historical_client:
            print("    ERROR: Historical client not initialized")
            await adapter.disconnect()
            return
        
        # Test historical data
        print("\n[2] Fetching SPY historical data (7 days)...")
        print("    Note: Free plan has 15-minute delay")
        
        historical_df = await adapter.load_historical_data(days=7)
        
        if historical_df is None or historical_df.empty:
            print("    ❌ No data returned")
        else:
            print(f"    ✅ SUCCESS! Got {len(historical_df):,} bars")
            print(f"    Date range: {historical_df.index[0].date()} to {historical_df.index[-1].date()}")
            print(f"    Latest close: ${historical_df['close'].iloc[-1]:.2f}")
            print(f"    Sample data:")
            print(historical_df.tail(3).to_string())
        
        await adapter.disconnect()
        
        print("\n" + "=" * 80)
        if not historical_df.empty:
            print("✅ SPY ADAPTER WORKS!")
        else:
            print("⚠️  No data returned (check API keys or subscription)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test())


