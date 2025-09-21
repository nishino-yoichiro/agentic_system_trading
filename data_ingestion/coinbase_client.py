#!/usr/bin/env python3
"""
Coinbase API Client for Crypto Data Collection

This module provides a comprehensive client for collecting cryptocurrency data
from Coinbase Advanced Trade API with proper rate limiting and error handling.

Features:
- Historical price data (minute-level precision)
- Real-time price data
- Multiple cryptocurrencies support
- Rate limiting and error handling
- Data validation and quality checks
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import os
from dataclasses import dataclass

from coinbase.rest import RESTClient

logger = logging.getLogger(__name__)

@dataclass
class CoinbaseConfig:
    """Configuration for Coinbase API client"""
    api_key: str
    api_secret: str
    passphrase: str = ""
    sandbox: bool = False
    rate_limit_delay: float = 0.1  # 100ms between requests
    max_retries: int = 3
    retry_delay: float = 1.0

class CoinbaseClient:
    """Coinbase API client for crypto data collection"""
    
    def __init__(self, config: CoinbaseConfig):
        self.config = config
        self.client = RESTClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            base_url='api.coinbase.com' if not config.sandbox else 'api-public.sandbox.exchange.coinbase.com'
        )
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_reset = time.time()
        
    def _rate_limit(self):
        """Implement rate limiting to respect API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    async def _make_request(self, func, *args, **kwargs):
        """Make API request with retry logic and rate limiting"""
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_str = str(e).lower()
                if '429' in error_str or 'rate limit' in error_str:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit exceeded, waiting {wait_time}s (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"API error: {e}")
                    if attempt == self.config.max_retries - 1:
                        raise
                    await asyncio.sleep(self.config.retry_delay)
        
        raise Exception("Max retries exceeded")
    
    async def get_available_products(self) -> List[Dict[str, Any]]:
        """Get list of available trading products"""
        try:
            products = await self._make_request(self.client.get_products)
            # Filter for USD pairs only
            usd_products = [p for p in products if p.get('quote_currency_id') == 'USD']
            logger.info(f"Found {len(usd_products)} USD trading pairs")
            return usd_products
        except Exception as e:
            logger.error(f"Error getting products: {e}")
            return []
    
    async def get_historical_candles(
        self, 
        product_id: str, 
        start_time: datetime, 
        end_time: datetime,
        granularity: int = 60  # 60 seconds = 1 minute
    ) -> pd.DataFrame:
        """
        Get historical candle data for a product
        
        Args:
            product_id: Trading pair (e.g., 'BTC-USD')
            start_time: Start datetime
            end_time: End datetime
            granularity: Candle granularity in seconds (60, 300, 900, 3600, 21600, 86400)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert datetime to ISO format
            start_iso = start_time.isoformat()
            end_iso = end_time.isoformat()
            
            logger.info(f"Fetching candles for {product_id} from {start_iso} to {end_iso}")
            
            # Get candles data
            candles = await self._make_request(
                self.client.get_candles,
                product_id=product_id,
                start=start_iso,
                end=end_iso,
                granularity=granularity
            )
            
            if not candles:
                logger.warning(f"No candles data for {product_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            # Rename columns to standard format
            column_mapping = {
                'start': 'timestamp',
                'low': 'low',
                'high': 'high',
                'open': 'open',
                'close': 'close',
                'volume': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Sort by timestamp
            df = df.sort_index()
            
            logger.info(f"Retrieved {len(df)} candles for {product_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting candles for {product_id}: {e}")
            return pd.DataFrame()
    
    async def get_current_price(self, product_id: str) -> Optional[float]:
        """Get current price for a product"""
        try:
            ticker = await self._make_request(
                self.client.get_product_ticker,
                product_id=product_id
            )
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error getting current price for {product_id}: {e}")
            return None
    
    async def get_24h_stats(self, product_id: str) -> Dict[str, Any]:
        """Get 24-hour statistics for a product"""
        try:
            stats = await self._make_request(
                self.client.get_product_ticker,
                product_id=product_id
            )
            return {
                'price': float(stats['price']),
                'volume_24h': float(stats.get('volume', 0)),
                'volume_30d': float(stats.get('volume_30d', 0)),
                'price_change_24h': float(stats.get('price_change_24h', 0)),
                'price_change_percent_24h': float(stats.get('price_change_percent_24h', 0))
            }
        except Exception as e:
            logger.error(f"Error getting 24h stats for {product_id}: {e}")
            return {}
    
    async def collect_crypto_data(
        self, 
        symbols: List[str], 
        days_back: int = 30,
        granularity: int = 60
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect historical data for multiple crypto symbols
        
        Args:
            symbols: List of crypto symbols (e.g., ['BTC', 'ETH', 'ADA'])
            days_back: Number of days of historical data
            granularity: Candle granularity in seconds
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        logger.info(f"Collecting {days_back} days of data for {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                product_id = f"{symbol}-USD"
                logger.info(f"Collecting data for {product_id}")
                
                df = await self.get_historical_candles(
                    product_id=product_id,
                    start_time=start_time,
                    end_time=end_time,
                    granularity=granularity
                )
                
                if not df.empty:
                    results[symbol] = df
                    logger.info(f"✅ Collected {len(df)} data points for {symbol}")
                else:
                    logger.warning(f"❌ No data collected for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error collecting data for {symbol}: {e}")
                continue
        
        logger.info(f"Data collection complete: {len(results)} symbols with data")
        return results
    
    async def validate_data_quality(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Validate data quality and return metrics"""
        if df.empty:
            return {'valid': False, 'issues': ['No data']}
        
        issues = []
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.any():
            issues.append(f"Missing values: {missing_data[missing_data > 0].to_dict()}")
        
        # Check for duplicate timestamps
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate timestamps: {duplicates}")
        
        # Check for extreme price changes
        if 'close' in df.columns:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = price_changes > 0.2  # 20% change
            if extreme_changes.any():
                issues.append(f"Extreme price changes: {extreme_changes.sum()}")
        
        # Check data frequency
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            if not time_diffs.empty:
                expected_interval = pd.Timedelta(seconds=60)  # 1 minute
                inconsistent = (time_diffs != expected_interval).sum()
                if inconsistent > 0:
                    issues.append(f"Inconsistent intervals: {inconsistent}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'data_points': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}" if not df.empty else "No data"
        }

def create_coinbase_client() -> Optional[CoinbaseClient]:
    """Create Coinbase client from environment variables"""
    try:
        api_key = os.getenv('COINBASE_API_KEY')
        api_secret = os.getenv('COINBASE_API_SECRET')
        
        if not api_key or not api_secret:
            logger.error("COINBASE_API_KEY and COINBASE_API_SECRET must be set in .env file")
            return None
        
        config = CoinbaseConfig(
            api_key=api_key,
            api_secret=api_secret,
            sandbox=False  # Set to True for testing
        )
        
        return CoinbaseClient(config)
        
    except Exception as e:
        logger.error(f"Error creating Coinbase client: {e}")
        return None

# Example usage
async def main():
    """Example usage of Coinbase client"""
    client = create_coinbase_client()
    if not client:
        return
    
    # Test with BTC
    symbols = ['BTC', 'ETH', 'ADA']
    data = await client.collect_crypto_data(symbols, days_back=7, granularity=60)
    
    for symbol, df in data.items():
        print(f"\n{symbol} Data:")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        if not df.empty:
            print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

if __name__ == "__main__":
    asyncio.run(main())
