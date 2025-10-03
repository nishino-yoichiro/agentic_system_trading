#!/usr/bin/env python3
"""
Coinbase Advanced Trade API Client for Real Historical Data

This module provides access to real historical candle data using the
Coinbase Advanced Trade API with proper authentication.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    print("⚠️ python-dotenv not installed. Environment variables may not load from .env file")

try:
    from coinbase.rest import RESTClient
except ImportError:
    print("❌ coinbase-advanced-py not installed. Run: pip install coinbase-advanced-py")
    RESTClient = None

logger = logging.getLogger(__name__)

class CoinbaseAdvancedClient:
    """Coinbase Advanced Trade API client for real historical data"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.getenv('COINBASE_API_KEY')
        self.api_secret = api_secret or os.getenv('COINBASE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("COINBASE_API_KEY and COINBASE_API_SECRET must be set")
        
        if RESTClient is None:
            raise ImportError("coinbase-advanced-py package not installed")
        
        # Format the API secret correctly
        # Check if secret is already in PEM format
        if self.api_secret.startswith('-----BEGIN EC PRIVATE KEY-----'):
            # Convert literal \n to actual newlines
            formatted_secret = self.api_secret.replace('\\n', '\n')
        else:
            # Format as PEM
            formatted_secret = f"-----BEGIN EC PRIVATE KEY-----\n{self.api_secret}\n-----END EC PRIVATE KEY-----\n"
        
        # Initialize REST client
        self.client = RESTClient(api_key=self.api_key, api_secret=formatted_secret)
        
        # Rate limiting
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
    
    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = asyncio.get_event_loop().time()
    
    def get_candles(self, ticker: str, days: int = 30, granularity: int = 60, start_time: str = None, end_time: str = None) -> Optional[List[Dict]]:
        """
        Get real candle data from Coinbase Advanced Trade API
        
        Args:
            ticker (str): Product ID (e.g., "BTC-USD", "ETH-USD")
            days (int): Number of days back to fetch data (used if start_time/end_time not provided)
            granularity (int): Granularity in seconds (60=1min, 300=5min, 900=15min, 3600=1hr, 21600=6hr, 86400=1day)
            start_time (str): Start time in ISO 8601 format (optional)
            end_time (str): End time in ISO 8601 format (optional)
        
        Returns:
            List of candle data or None if error
        """
        try:
            # Validate granularity
            valid_granularities = [60, 300, 900, 3600, 21600, 86400]
            if granularity not in valid_granularities:
                logger.error(f"Invalid granularity {granularity}. Must be one of: {valid_granularities}")
                return None
            
            # Calculate time range - limit to 350 candles max
            # Use UTC time and avoid current minute (may be incomplete)
            from datetime import timezone
            end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)
            
            # Calculate max days based on granularity to stay under 350 candles
            max_candles = 350
            if granularity == 60:  # 1 minute
                max_days = max_candles / (24 * 60)  # ~2.4 days
            elif granularity == 300:  # 5 minutes
                max_days = max_candles / (24 * 12)  # ~29 days
            elif granularity == 900:  # 15 minutes
                max_days = max_candles / (24 * 4)  # ~3.6 days
            elif granularity == 3600:  # 1 hour
                max_days = max_candles / 24  # ~14.6 days
            else:
                max_days = 1  # Default to 1 day
            
            # Only calculate timestamps if not provided
            if not start_time or not end_time:
                # Use the smaller of requested days or max days
                actual_days = min(days, max_days)
                start_time = end_time - timedelta(days=actual_days)
                
                logger.info(f"Adjusted time range to {actual_days:.1f} days to stay under {max_candles} candles limit")
            
            # Convert to Unix timestamps as strings (required by Coinbase Advanced Trade API)
            start_timestamp = str(int(start_time.timestamp()))
            end_timestamp = str(int(end_time.timestamp()))
            
            # Debug: Log the actual timestamps being sent
            logger.debug(f"Start timestamp: {start_timestamp}")
            logger.debug(f"End timestamp: {end_timestamp}")
            
            logger.info(f"Fetching {ticker} candles: {days} days, {granularity}s granularity")
            logger.info(f"Time range: {start_time} to {end_time}")
            
            # Convert granularity to Coinbase API string format
            granularity_map = {
                60: "ONE_MINUTE",
                300: "FIVE_MINUTE", 
                900: "FIFTEEN_MINUTE",
                3600: "ONE_HOUR",
                21600: "SIX_HOUR",
                86400: "ONE_DAY"
            }
            
            granularity_str = granularity_map.get(granularity, "ONE_MINUTE")
            
            # Try get_public_candles first (no authentication needed for historical data)
            try:
                logger.info(f"Trying get_public_candles with granularity: {granularity_str}")
                # Use provided timestamps if available, otherwise use calculated ones
                if start_time and end_time:
                    # Convert provided times to Unix timestamps as strings
                    start_unix = str(int(start_time.timestamp()))
                    end_unix = str(int(end_time.timestamp()))
                    candles_response = self.client.get_public_candles(
                        product_id=ticker,
                        start=start_unix,
                        end=end_unix,
                        granularity=granularity_str
                    )
                else:
                    # Use calculated timestamps (Unix format)
                    candles_response = self.client.get_public_candles(
                        product_id=ticker,
                        start=start_timestamp,
                        end=end_timestamp,
                        granularity=granularity_str
                    )
            except Exception as e:
                logger.warning(f"get_public_candles failed: {e}")
                # Try get_candles as fallback
                try:
                    logger.info(f"Trying get_candles with granularity: {granularity_str}")
                    if start_time and end_time:
                        # Convert provided times to Unix timestamps as strings
                        start_unix = str(int(start_time.timestamp()))
                        end_unix = str(int(end_time.timestamp()))
                        candles_response = self.client.get_candles(
                            product_id=ticker,
                            start=start_unix,
                            end=end_unix,
                            granularity=granularity_str
                        )
                    else:
                        # Use calculated timestamps (Unix format)
                        candles_response = self.client.get_candles(
                            product_id=ticker,
                            start=start_timestamp,
                            end=end_timestamp,
                            granularity=granularity_str
                        )
                except Exception as e2:
                    logger.error(f"Both methods failed: {e2}")
                    raise e2
            
            # Handle the response object properly
            if hasattr(candles_response, 'candles'):
                candles = candles_response.candles
            elif hasattr(candles_response, 'data'):
                candles = candles_response.data
            elif isinstance(candles_response, list):
                candles = candles_response
            else:
                # Try to convert to dict and get candles
                try:
                    candles_dict = candles_response.__dict__ if hasattr(candles_response, '__dict__') else {}
                    candles = candles_dict.get('candles', [])
                except:
                    candles = []
            
            logger.info(f"Retrieved {len(candles)} candles for {ticker}")
            
            return candles
            
        except Exception as e:
            logger.error(f"Error fetching candles for {ticker}: {e}")
            return None
    
    def candles_to_dataframe(self, candles: List[Dict], ticker: str) -> pd.DataFrame:
        """Convert candles data to pandas DataFrame"""
        if not candles:
            return pd.DataFrame()
        
        data = []
        for candle in candles:
            try:
                # Convert timestamp to datetime
                timestamp = datetime.fromtimestamp(int(candle['start']))
                
                data.append({
                    'timestamp': timestamp,
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle['volume'])
                })
            except (ValueError, KeyError) as e:
                logger.warning(f"Error parsing candle data: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index()
        
        logger.info(f"Converted {len(df)} candles to DataFrame for {ticker}")
        return df
    
    async def collect_historical_data(self, symbols: List[str], days: int = 30, granularity: int = 60) -> Dict[str, pd.DataFrame]:
        """
        Collect historical data for multiple symbols
        
        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            days: Number of days back
            granularity: Granularity in seconds
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Convert symbol to Coinbase format
                ticker = f"{symbol}-USD"
                
                # Apply rate limiting
                await self._rate_limit()
                
                # Get candles
                candles = self.get_candles(ticker, days, granularity)
                
                if candles:
                    # Convert to DataFrame
                    df = self.candles_to_dataframe(candles, symbol)
                    
                    if not df.empty:
                        results[symbol] = df
                        logger.info(f"✅ Collected {len(df)} data points for {symbol}")
                        logger.info(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                        logger.info(f"   Date range: {df.index.min()} to {df.index.max()}")
                    else:
                        logger.warning(f"❌ No data for {symbol}")
                else:
                    logger.warning(f"❌ Failed to get candles for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error collecting data for {symbol}: {e}")
                continue
        
        return results
    
    async def save_historical_data(self, data: Dict[str, pd.DataFrame], data_dir: Path):
        """Save historical data to parquet files"""
        raw_dir = data_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        
        for symbol, df in data.items():
            if not df.empty:
                file_path = raw_dir / f"prices_{symbol}.parquet"
                df.to_parquet(file_path)
                logger.info(f"💾 Saved {symbol} data to {file_path}")

def create_coinbase_advanced_client() -> Optional[CoinbaseAdvancedClient]:
    """Create Coinbase Advanced client from environment variables"""
    try:
        return CoinbaseAdvancedClient()
    except Exception as e:
        logger.error(f"Error creating Coinbase Advanced client: {e}")
        return None

# Example usage
async def main():
    """Example usage"""
    client = create_coinbase_advanced_client()
    
    if not client:
        print("❌ Failed to create Coinbase Advanced client")
        return
    
    # Collect historical data for BTC and ETH
    symbols = ['BTC', 'ETH']
    data = await client.collect_historical_data(symbols, days=30, granularity=60)
    
    print(f"\n📊 Collected data for {len(data)} symbols:")
    for symbol, df in data.items():
        print(f"   {symbol}: {len(df)} data points")
        if not df.empty:
            print(f"      Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            print(f"      Current price: ${df['close'].iloc[-1]:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
