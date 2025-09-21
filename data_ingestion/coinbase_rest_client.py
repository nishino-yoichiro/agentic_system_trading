#!/usr/bin/env python3
"""
Coinbase REST API Client for Crypto Data Collection

This module provides a simple REST client for collecting cryptocurrency data
from Coinbase using direct HTTP requests instead of the SDK.

Features:
- Current price data (no auth required)
- Historical price data (with auth)
- Simple and reliable
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import time
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class CoinbaseRESTClient:
    """Simple REST client for Coinbase API"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coinbase.com/v2"
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol (no auth required)"""
        try:
            url = f"{self.base_url}/prices/{symbol}-USD/spot"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = float(data["data"]["amount"])
                        logger.info(f"Current {symbol} price: ${price}")
                        return price
                    else:
                        logger.error(f"Error getting price for {symbol}: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def get_historical_prices(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical prices for a symbol (no auth required)"""
        try:
            # Coinbase public API has limited historical data
            # We'll get daily data for the specified period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.base_url}/prices/{symbol}-USD/historic"
            params = {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "data" in data and "prices" in data["data"]:
                            prices = data["data"]["prices"]
                            
                            # Convert to DataFrame
                            df_data = []
                            for price_point in prices:
                                try:
                                    # Handle timestamp conversion
                                    timestamp = price_point['time']
                                    if isinstance(timestamp, (int, float)):
                                        # Convert Unix timestamp
                                        timestamp = pd.to_datetime(timestamp, unit='s')
                                    else:
                                        timestamp = pd.to_datetime(timestamp)
                                    
                                    df_data.append({
                                        'timestamp': timestamp,
                                        'price': float(price_point['price'])
                                    })
                                except Exception as e:
                                    logger.warning(f"Error parsing price point: {e}")
                                    continue
                            
                            df = pd.DataFrame(df_data)
                            df = df.set_index('timestamp')
                            df = df.sort_index()
                            
                            logger.info(f"Retrieved {len(df)} historical prices for {symbol}")
                            return df
                        else:
                            logger.warning(f"No historical data found for {symbol}")
                            return pd.DataFrame()
                    else:
                        logger.error(f"Error getting historical prices for {symbol}: {response.status}")
                        return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting historical prices for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                price = await self.get_current_price(symbol)
                if price:
                    results[symbol] = price
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
                continue
        
        return results
    
    async def test_connection(self) -> bool:
        """Test API connection"""
        try:
            price = await self.get_current_price("BTC")
            return price is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

def create_coinbase_rest_client() -> CoinbaseRESTClient:
    """Create Coinbase REST client"""
    return CoinbaseRESTClient()

# Example usage
async def main():
    """Example usage"""
    client = create_coinbase_rest_client()
    
    # Test connection
    if await client.test_connection():
        print("✅ Coinbase API connection successful")
        
        # Get current BTC price
        btc_price = await client.get_current_price("BTC")
        print(f"Current BTC price: ${btc_price}")
        
        # Get historical data
        historical_data = await client.get_historical_prices("BTC", days=7)
        if not historical_data.empty:
            print(f"Historical data shape: {historical_data.shape}")
            print(f"Date range: {historical_data.index.min()} to {historical_data.index.max()}")
        
        # Get multiple prices
        symbols = ["BTC", "ETH", "ADA"]
        prices = await client.get_multiple_prices(symbols)
        for symbol, price in prices.items():
            print(f"{symbol}: ${price}")
    else:
        print("❌ Coinbase API connection failed")

if __name__ == "__main__":
    asyncio.run(main())
