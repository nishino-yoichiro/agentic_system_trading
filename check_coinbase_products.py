#!/usr/bin/env python3
"""
Check which crypto products are supported by Coinbase Advanced Trade API
"""

import asyncio
import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient

async def check_supported_products():
    """Check which crypto products are supported by Coinbase"""
    load_dotenv()
    
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ API keys not found in .env file")
        return
    
    try:
        client = RESTClient(api_key=api_key, api_secret=api_secret)
        
        # Get all products
        products = client.get_products()
        
        # Filter for crypto-USD pairs
        crypto_products = []
        for product in products.products:
            if product.quote_currency_id == 'USD' and product.status == 'online':
                crypto_products.append(product.base_currency_id)
        
        print(f"✅ Found {len(crypto_products)} supported crypto-USD pairs:")
        for crypto in sorted(crypto_products):
            print(f"  - {crypto}")
        
        # Check our specific symbols
        our_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']
        print(f"\n🔍 Checking our symbols:")
        for symbol in our_symbols:
            if symbol in crypto_products:
                print(f"  ✅ {symbol} - SUPPORTED")
            else:
                print(f"  ❌ {symbol} - NOT SUPPORTED")
        
        return crypto_products
        
    except Exception as e:
        print(f"❌ Error checking products: {e}")
        return []

if __name__ == "__main__":
    asyncio.run(check_supported_products())
