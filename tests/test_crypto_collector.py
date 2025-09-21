#!/usr/bin/env python3
"""
Test Crypto Data Collector

This script tests the crypto data collector with Coinbase API.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_ingestion.crypto_collector import CryptoDataCollector

async def test_crypto_collector():
    """Test crypto data collector"""
    print("🚀 Testing Crypto Data Collector")
    print("=" * 50)
    
    collector = CryptoDataCollector()
    
    # Test with BTC and ETH
    symbols = ['BTC', 'ETH']
    
    print(f"Collecting data for {symbols}...")
    data = await collector.collect_crypto_data(symbols)
    
    print(f"\n✅ Collected data for {len(data)} symbols:")
    for symbol, df in data.items():
        if not df.empty:
            price = df['close'].iloc[-1]
            print(f"   {symbol}: ${price:,.2f}")
        else:
            print(f"   {symbol}: No data")
    
    # Test saving data
    print(f"\n💾 Saving data...")
    await collector.save_crypto_data(data, Path("data"))
    
    # Test summary
    print(f"\n📊 Getting summary...")
    summary = await collector.get_crypto_summary(symbols)
    print(f"   Total symbols: {summary['total_symbols']}")
    print(f"   Price range: ${summary['price_range']['min']:,.2f} - ${summary['price_range']['max']:,.2f}")
    
    print(f"\n🎉 Crypto collector test complete!")

if __name__ == "__main__":
    asyncio.run(test_crypto_collector())

