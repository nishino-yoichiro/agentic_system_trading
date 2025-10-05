#!/usr/bin/env python3
"""
Real-time Price Collector using Coinbase Free API
Gets current spot prices without authentication
"""

import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class RealtimePriceCollector:
    """Collects real-time prices using Coinbase's free API"""
    
    def __init__(self):
        self.base_url = "https://api.coinbase.com/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoAgent/1.0'
        })
    
    def get_spot_price(self, currency_pair: str = 'BTC-USD') -> Optional[Dict]:
        """Get current spot price for a currency pair"""
        try:
            url = f"{self.base_url}/prices/{currency_pair}/spot"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data:
                return {
                    'currency_pair': currency_pair,
                    'price': float(data['data']['amount']),
                    'timestamp': datetime.now(timezone.utc),
                    'base': data['data']['base'],
                    'currency': data['data']['currency']
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting spot price for {currency_pair}: {e}")
            return None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        prices = {}
        
        for symbol in symbols:
            try:
                # Convert symbol to Coinbase format (e.g., BTC -> BTC-USD)
                currency_pair = f"{symbol}-USD"
                price_data = self.get_spot_price(currency_pair)
                
                if price_data:
                    prices[symbol] = price_data['price']
                    logger.debug(f"Got price for {symbol}: ${price_data['price']}")
                else:
                    logger.warning(f"Failed to get price for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
        
        return prices
    
    def collect_and_save_prices(self, symbols: List[str], data_dir: Path = None) -> Dict[str, Dict]:
        """Collect current prices and save to database"""
        if data_dir is None:
            data_dir = Path("data/crypto_db")
        
        data_dir.mkdir(exist_ok=True)
        
        collected_data = {}
        current_time = datetime.now(timezone.utc)
        
        for symbol in symbols:
            try:
                currency_pair = f"{symbol}-USD"
                price_data = self.get_spot_price(currency_pair)
                
                if price_data:
                    # Create a DataFrame with the current price
                    df = pd.DataFrame({
                        'timestamp': [current_time],
                        'open': [price_data['price']],
                        'high': [price_data['price']],
                        'low': [price_data['price']],
                        'close': [price_data['price']],
                        'volume': [0.0]  # Volume not available from spot price
                    })
                    df.set_index('timestamp', inplace=True)
                    
                    # Load existing data
                    db_file = data_dir / f"{symbol}_historical.parquet"
                    if db_file.exists():
                        existing_df = pd.read_parquet(db_file)
                        existing_df.index = pd.to_datetime(existing_df.index)
                        
                        # Append new data
                        combined_df = pd.concat([existing_df, df])
                        
                        # Remove duplicates (keep last)
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                        
                        # Sort by timestamp
                        combined_df = combined_df.sort_index()
                        
                        # Keep only last 7 days to prevent file from growing too large
                        cutoff = current_time - pd.Timedelta(days=7)
                        combined_df = combined_df[combined_df.index >= cutoff]
                        
                    else:
                        combined_df = df
                    
                    # Save updated data
                    combined_df.to_parquet(db_file)
                    logger.info(f"Updated {symbol} with current price ${price_data['price']:.2f}")
                    
                    collected_data[symbol] = {
                        'price': price_data['price'],
                        'timestamp': current_time,
                        'data_points': len(combined_df)
                    }
                    
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
        
        return collected_data

def main():
    """Test the real-time price collector"""
    collector = RealtimePriceCollector()
    
    # Test single price
    print("Testing single price...")
    btc_price = collector.get_spot_price('BTC-USD')
    print(f"BTC Price: {btc_price}")
    
    # Test multiple prices
    print("\nTesting multiple prices...")
    symbols = ['BTC', 'ETH', 'ADA']
    prices = collector.get_multiple_prices(symbols)
    print(f"Prices: {prices}")
    
    # Test collection and saving
    print("\nTesting collection and saving...")
    data_dir = Path("data/crypto_db")
    collected = collector.collect_and_save_prices(['BTC'], data_dir)
    print(f"Collected: {collected}")

if __name__ == "__main__":
    main()
