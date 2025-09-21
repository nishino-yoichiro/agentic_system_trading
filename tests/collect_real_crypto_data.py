#!/usr/bin/env python3
"""
Collect Real Crypto Historical Data

This script replaces synthetic data with real historical data from Coinbase Advanced Trade API.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_ingestion.coinbase_advanced_client import create_coinbase_advanced_client

async def collect_real_crypto_data():
    """Collect real historical crypto data from Coinbase"""
    print("🚀 Collecting Real Crypto Historical Data")
    print("=" * 50)
    
    # Create client
    client = create_coinbase_advanced_client()
    
    if not client:
        print("❌ Failed to create Coinbase Advanced client")
        print("   Make sure COINBASE_API_KEY and COINBASE_API_SECRET are set in .env")
        return
    
    # Collect historical data for BTC and ETH
    symbols = ['BTC', 'ETH']
    
    print(f"📊 Collecting 30 days of 1-minute data for {symbols}...")
    
    # Collect data
    data = await client.collect_historical_data(symbols, days=30, granularity=60)
    
    if not data:
        print("❌ No data collected")
        return
    
    # Save data
    data_dir = Path("data")
    await client.save_historical_data(data, data_dir)
    
    print(f"\n🎉 Real crypto data collection complete!")
    print(f"   Collected data for {len(data)} symbols")
    
    for symbol, df in data.items():
        print(f"   {symbol}: {len(df):,} data points")
        if not df.empty:
            print(f"      Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            print(f"      Current price: ${df['close'].iloc[-1]:.2f}")
            print(f"      Date range: {df.index.min()} to {df.index.max()}")

if __name__ == "__main__":
    asyncio.run(collect_real_crypto_data())

