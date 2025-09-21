"""
High-Frequency Data Collection APIs
Minute-level and 5-minute granularity for better trading analysis
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import time

logger = logging.getLogger(__name__)

class HighFrequencyPolygonClient:
    """Polygon.io client for minute-level stock data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.rate_limit = 5  # calls per minute
        self.last_call_time = 0
    
    async def _rate_limit_wait(self):
        """Wait if rate limit is reached"""
        now = time.time()
        time_since_last = now - self.last_call_time
        
        # For free tier, wait 15 seconds between requests to be safe
        if time_since_last < 15:
            wait_time = 15 - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        
        self.last_call_time = time.time()
    
    async def get_minute_data(self, symbol: str, days_back: int = 30, cache: 'IncrementalCache' = None) -> pd.DataFrame:
        """Get minute-level data for a symbol with proper pagination using next_url"""
        try:
            # Check if we have a checkpoint to resume from
            checkpoint_timestamp = None
            if cache:
                checkpoint_timestamp = cache.get_checkpoint(symbol)
                if checkpoint_timestamp:
                    logger.info(f"Resuming {symbol} from checkpoint: {checkpoint_timestamp}")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # If we have a checkpoint, start from there instead of the beginning
            if checkpoint_timestamp and checkpoint_timestamp > start_date:
                start_date = checkpoint_timestamp
                logger.info(f"Starting {symbol} from checkpoint: {start_date}")
            
            all_data = []
            next_url = None
            
            # Initial request
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/minute/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'apikey': self.api_key
            }
            
            request_count = 0
            max_requests = 50  # Safety limit to prevent infinite loops
            
            while request_count < max_requests:
                await self._rate_limit_wait()
                request_count += 1
                
                if next_url:
                    # For next_url, we need to add the API key as a parameter
                    current_url = next_url
                    current_params = {'apikey': self.api_key}
                else:
                    current_url = url
                    current_params = params
                
                logger.info(f"Fetching {symbol} data (request {request_count})")
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(current_url, params=current_params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if data.get('status') in ['OK', 'DELAYED'] and data.get('results'):
                                results = data['results']
                                all_data.extend(results)
                                logger.info(f"Fetched {len(results)} points for {symbol} (total: {len(all_data)})")
                                
                                # Save checkpoint every 5000 points
                                if cache and len(all_data) % 5000 == 0 and all_data:
                                    last_timestamp = datetime.fromtimestamp(all_data[-1]['t'] / 1000)
                                    cache.save_checkpoint(symbol, last_timestamp, len(all_data))
                                
                                # Check for next_url for pagination
                                next_url = data.get('next_url')
                                if not next_url:
                                    logger.info(f"Pagination complete for {symbol}")
                                    break
                            else:
                                logger.warning(f"No data returned for {symbol} - status: {data.get('status')}")
                                break
                        else:
                            logger.error(f"API error for {symbol}: {response.status}")
                            break
                
                # Rate limiting is handled in _rate_limit_wait()
            
            if all_data:
                df = pd.DataFrame(all_data)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                
                # Rename columns to standard format
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high', 
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume',
                    'vw': 'volume_weighted_avg_price',
                    'n': 'transactions'
                })
                
                # Select relevant columns
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_weighted_avg_price']]
                
                # Sort by timestamp
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"Fetched {len(df)} minute-level data points for {symbol} (spanning {days_back} days)")
                return df
            else:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
                        
        except Exception as e:
            logger.error(f"Error fetching minute data for {symbol}: {e}")
            return pd.DataFrame()

class HighFrequencyCoinGeckoClient:
    """CoinGecko client for 5-minute crypto data"""
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.rate_limit = 50  # calls per minute
        self.last_call_time = 0
    
    async def _rate_limit_wait(self):
        """Wait if rate limit is reached"""
        now = time.time()
        time_since_last = now - self.last_call_time
        if time_since_last < 60 / self.rate_limit:
            wait_time = (60 / self.rate_limit) - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        self.last_call_time = time.time()
    
    async def get_5min_data(self, coin_id: str, days_back: int = 30) -> pd.DataFrame:
        """Get 5-minute data for a crypto coin"""
        await self._rate_limit_wait()
        
        try:
            # CoinGecko coin ID mapping
            coin_mapping = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum', 
                'BNB': 'binancecoin',
                'ADA': 'cardano',
                'SOL': 'solana',
                'DOT': 'polkadot',
                'AVAX': 'avalanche-2',
                'MATIC': 'matic-network',
                'LINK': 'chainlink',
                'UNI': 'uniswap'
            }
            
            coin_id = coin_mapping.get(coin_id, coin_id.lower())
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            
            params = {
                'vs_currency': 'usd',
                'from': int(start_date.timestamp()),
                'to': int(end_date.timestamp()),
                'interval': '5m'  # 5-minute intervals
            }
            
            if self.api_key:
                params['x_cg_demo_api_key'] = self.api_key
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'prices' in data and data['prices']:
                            # Convert to DataFrame
                            prices = data['prices']
                            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                            
                            # Convert timestamp to datetime
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            
                            # For 5-minute data, we only get price, so we'll use it for OHLC
                            df['open'] = df['price']
                            df['high'] = df['price'] 
                            df['low'] = df['price']
                            df['close'] = df['price']
                            df['volume'] = 0  # Not available in this endpoint
                            
                            # Select relevant columns
                            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                            
                            logger.info(f"Fetched {len(df)} 5-minute data points for {coin_id}")
                            return df
                        else:
                            logger.warning(f"No 5-minute data returned for {coin_id}")
                            return pd.DataFrame()
                    else:
                        logger.error(f"API error for {coin_id}: {response.status}")
                        return pd.DataFrame()
                        
        except Exception as e:
            logger.error(f"Error fetching 5-minute data for {coin_id}: {e}")
            return pd.DataFrame()

class BinanceClient:
    """Binance API client for 1-minute crypto data"""
    
    def __init__(self, api_key: str = "", secret_key: str = ""):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.binance.com/api/v3"
        self.rate_limit = 1200  # calls per minute
        self.last_call_time = 0
    
    async def _rate_limit_wait(self):
        """Wait if rate limit is reached"""
        now = time.time()
        time_since_last = now - self.last_call_time
        if time_since_last < 60 / self.rate_limit:
            wait_time = (60 / self.rate_limit) - time_since_last
            await asyncio.sleep(wait_time)
        self.last_call_time = time.time()
    
    async def get_1min_data(self, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """Get 1-minute data for a crypto symbol"""
        await self._rate_limit_wait()
        
        try:
            # Convert symbol to Binance format
            binance_symbol = f"{symbol}USDT"
            
            # Calculate time range
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            
            url = f"{self.base_url}/klines"
            
            params = {
                'symbol': binance_symbol,
                'interval': '1m',  # 1-minute intervals
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000  # Max 1000 candles per request
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data:
                            # Convert to DataFrame
                            df = pd.DataFrame(data, columns=[
                                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                                'close_time', 'quote_asset_volume', 'number_of_trades',
                                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                            ])
                            
                            # Convert timestamp to datetime
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            
                            # Convert price columns to float
                            price_columns = ['open', 'high', 'low', 'close', 'volume']
                            for col in price_columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            # Select relevant columns
                            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                            
                            logger.info(f"Fetched {len(df)} 1-minute data points for {symbol}")
                            return df
                        else:
                            logger.warning(f"No 1-minute data returned for {symbol}")
                            return pd.DataFrame()
                    else:
                        logger.error(f"API error for {symbol}: {response.status}")
                        return pd.DataFrame()
                        
        except Exception as e:
            logger.error(f"Error fetching 1-minute data for {symbol}: {e}")
            return pd.DataFrame()

async def collect_high_frequency_data(symbols: List[str], days_back: int = 30, api_keys: Dict[str, str] = None) -> Dict[str, pd.DataFrame]:
    """Collect high-frequency data for all symbols"""
    logger.info(f"Collecting high-frequency data for {len(symbols)} symbols")
    
    if not api_keys:
        api_keys = {}
    
    results = {}
    
    try:
        # Initialize clients
        polygon_client = HighFrequencyPolygonClient(api_keys.get('polygon', ''))
        coingecko_client = HighFrequencyCoinGeckoClient(api_keys.get('coingecko', ''))
        binance_client = BinanceClient(api_keys.get('binance', ''), api_keys.get('binance_secret', ''))
        
        # Separate crypto and stock symbols
        crypto_symbols = [s for s in symbols if s in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']]
        stock_symbols = [s for s in symbols if s not in crypto_symbols]
        
        # Collect stock data (1-minute)
        for symbol in stock_symbols:
            logger.info(f"Collecting 1-minute data for {symbol}")
            df = await polygon_client.get_minute_data(symbol, days_back)
            if not df.empty:
                results[symbol] = df
        
        # Collect crypto data (1-minute from Binance, 5-minute from CoinGecko as fallback)
        for symbol in crypto_symbols:
            logger.info(f"Collecting high-frequency data for {symbol}")
            
            # Try Binance first (1-minute)
            df = await binance_client.get_1min_data(symbol, days_back)
            
            # Fallback to CoinGecko (5-minute) if Binance fails
            if df.empty:
                logger.info(f"Binance failed for {symbol}, trying CoinGecko 5-minute data")
                df = await coingecko_client.get_5min_data(symbol, days_back)
            
            if not df.empty:
                results[symbol] = df
        
        logger.info(f"High-frequency data collection completed: {len(results)} symbols")
        return results
        
    except Exception as e:
        logger.error(f"Error in high-frequency data collection: {e}")
        return results
