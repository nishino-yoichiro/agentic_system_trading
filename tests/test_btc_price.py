#!/usr/bin/env python3
"""
Test BTC Price from Coinbase

Simple test to get current BTC price and verify it's working.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_ingestion.coinbase_rest_client import create_coinbase_rest_client

async def main():
    """Get current BTC price"""
    print("🚀 Testing BTC Price from Coinbase")
    print("=" * 40)
    
    client = create_coinbase_rest_client()
    
    # Get current BTC price
    btc_price = await client.get_current_price("BTC")
    
    if btc_price:
        print(f"✅ Current BTC Price: ${btc_price:,.2f}")
        
        # Get a few more crypto prices
        symbols = ["ETH", "ADA", "SOL"]
        print(f"\n📊 Other Crypto Prices:")
        for symbol in symbols:
            price = await client.get_current_price(symbol)
            if price:
                print(f"   {symbol}: ${price:,.2f}")
        
        print(f"\n🎉 Coinbase API is working! Ready to replace CoinGecko.")
    else:
        print("❌ Failed to get BTC price")

if __name__ == "__main__":
    asyncio.run(main())

