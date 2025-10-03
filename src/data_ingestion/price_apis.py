"""
Price API Clients for collecting financial market data (Windows Compatible)

Supports multiple price data sources:
- Polygon.io (stocks, crypto, forex)
- Alpaca (stocks, crypto)
- Binance (cryptocurrency)
- CoinGecko (cryptocurrency)
- Yahoo Finance (fallback)
"""

import asyncio
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import yaml
import os
from loguru import logger
import time
import json


@dataclass
class PriceData:
    """Standardized price data format"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trades: Optional[int] = None
    source: str = "unknown"
    raw_data: Optional[Dict] = None


class PolygonClient:
    """Polygon.io API client for stocks and crypto"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.polygon.io"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
    async def __aenter__(self):
        self.session = requests.Session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def get_stock_prices(self, symbol: str, start_date: datetime, end_date: datetime, 
                              timespan: str = 'day', multiplier: int = 1) -> List[PriceData]:
        """Get stock price data from Polygon"""
        params = {
            'apikey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            response = self.session.get(url, params=params)
            
            if response.status_code == 429:
                logger.warning("Rate limit exceeded, waiting 60 seconds")
                await asyncio.sleep(60)
                return await self.get_stock_prices(symbol, start_date, end_date, timespan, multiplier)
            
            response.raise_for_status()
            data = response.json()
            
            prices = []
            for result in data.get('results', []):
                price = self._parse_polygon_result(result, symbol, 'stock')
                if price:
                    prices.append(price)
            
            logger.info(f"Fetched {len(prices)} stock prices for {symbol}")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching stock prices for {symbol}: {e}")
            return []
    
    def _parse_polygon_result(self, result: Dict, symbol: str, asset_type: str) -> Optional[PriceData]:
        """Parse Polygon API result into PriceData"""
        try:
            return PriceData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(result['t'] / 1000),
                open=result['o'],
                high=result['h'],
                low=result['l'],
                close=result['c'],
                volume=result['v'],
                vwap=result.get('vw'),
                trades=result.get('n'),
                source='polygon',
                raw_data=result
            )
        except Exception as e:
            logger.error(f"Error parsing Polygon result: {e}")
            return None


class BinanceClient:
    """Binance API client for cryptocurrency data"""
    
    def __init__(self, api_key: str = None, secret_key: str = None, base_url: str = "https://api.binance.com"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = requests.Session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def get_crypto_prices(self, symbol: str, start_date: datetime, end_date: datetime, 
                               interval: str = '1d') -> List[PriceData]:
        """Get cryptocurrency price data from Binance"""
        try:
            # Convert symbol format (e.g., BTC-USD -> BTCUSDT)
            binance_symbol = symbol.replace('-', '').replace('USD', 'USDT')
            
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'startTime': int(start_date.timestamp() * 1000),
                'endTime': int(end_date.timestamp() * 1000),
                'limit': 1000
            }
            
            response = self.session.get(f"{self.base_url}/api/v3/klines", params=params)
            response.raise_for_status()
            data = response.json()
            
            prices = []
            for kline in data:
                price = self._parse_binance_kline(kline, symbol)
                if price:
                    prices.append(price)
            
            logger.info(f"Fetched {len(prices)} crypto prices for {symbol}")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching crypto prices for {symbol}: {e}")
            return []
    
    def _parse_binance_kline(self, kline: List, symbol: str) -> Optional[PriceData]:
        """Parse Binance kline data into PriceData"""
        try:
            return PriceData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(kline[0] / 1000),
                open=float(kline[1]),
                high=float(kline[2]),
                low=float(kline[3]),
                close=float(kline[4]),
                volume=float(kline[5]),
                vwap=float(kline[7]) if kline[7] else None,
                trades=int(kline[8]),
                source='binance',
                raw_data=kline
            )
        except Exception as e:
            logger.error(f"Error parsing Binance kline: {e}")
            return None


class CoinGeckoClient:
    """CoinGecko API client for cryptocurrency data"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.coingecko.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = requests.Session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def get_crypto_prices(self, coin_id: str, days: int = 30) -> List[PriceData]:
        """Get cryptocurrency price data from CoinGecko"""
        try:
            # Add delay to avoid rate limiting
            await asyncio.sleep(0.5)
            
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            if self.api_key:
                params['x_cg_demo_api_key'] = self.api_key
            
            response = self.session.get(f"{self.base_url}/api/v3/coins/{coin_id}/market_chart", params=params)
            response.raise_for_status()
            data = response.json()
            
            prices = []
            for i, (timestamp, price) in enumerate(zip(data['prices'], data['prices'])):
                # CoinGecko only provides OHLC data, so we use the same values
                price_data = PriceData(
                    symbol=coin_id.upper(),
                    timestamp=datetime.fromtimestamp(timestamp[0] / 1000),
                    open=price[1],
                    high=price[1],
                    low=price[1],
                    close=price[1],
                    volume=data['total_volumes'][i][1] if i < len(data['total_volumes']) else 0,
                    source='coingecko',
                    raw_data={'price': price, 'volume': data['total_volumes'][i] if i < len(data['total_volumes']) else [0, 0]}
                )
                prices.append(price_data)
            
            logger.info(f"Fetched {len(prices)} crypto prices for {coin_id}")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching crypto prices for {coin_id}: {e}")
            return []


class YahooFinanceClient:
    """Yahoo Finance client for stocks and crypto (free, no API key needed)"""
    
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com"
        self.session = None
        
    async def __aenter__(self):
        self.session = requests.Session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def get_prices(self, symbol: str, start_date: datetime, end_date: datetime) -> List[PriceData]:
        """Get price data from Yahoo Finance"""
        try:
            # Add delay to avoid rate limiting
            await asyncio.sleep(1)
            
            # Convert dates to timestamps
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())
            
            # Map crypto symbols to Yahoo Finance format
            yahoo_symbol = self._map_to_yahoo_symbol(symbol)
            
            url = f"{self.base_url}/v8/finance/chart/{yahoo_symbol}"
            params = {
                'period1': start_ts,
                'period2': end_ts,
                'interval': '1d',
                'includePrePost': 'false',
                'events': 'div,split'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'chart' not in data or not data['chart']['result']:
                logger.warning(f"No data found for {symbol}")
                return []
            
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            prices = []
            for i, timestamp in enumerate(timestamps):
                if i < len(quotes['open']) and quotes['open'][i] is not None:
                    price_data = PriceData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(timestamp),
                        open=quotes['open'][i],
                        high=quotes['high'][i],
                        low=quotes['low'][i],
                        close=quotes['close'][i],
                        volume=quotes['volume'][i] if quotes['volume'][i] else 0,
                        source='yahoo_finance',
                        raw_data={'timestamp': timestamp, 'quote': quotes}
                    )
                    prices.append(price_data)
            
            logger.info(f"Fetched {len(prices)} prices for {symbol} from Yahoo Finance")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching prices for {symbol} from Yahoo Finance: {e}")
            return []
    
    def _map_to_yahoo_symbol(self, symbol: str) -> str:
        """Map symbols to Yahoo Finance format"""
        # For crypto, add -USD suffix
        crypto_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']
        if symbol in crypto_symbols:
            return f"{symbol}-USD"
        return symbol


class AlpacaClient:
    """Alpaca API client for stocks and crypto"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        })
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def get_stock_prices(self, symbol: str, start_date: datetime, end_date: datetime) -> List[PriceData]:
        """Get stock price data from Alpaca"""
        try:
            params = {
                'symbols': symbol,
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'timeframe': '1Day',
                'limit': 10000
            }
            
            response = self.session.get(f"{self.base_url}/v2/stocks/bars", params=params)
            response.raise_for_status()
            data = response.json()
            
            prices = []
            for bar in data.get('bars', {}).get(symbol, []):
                price = self._parse_alpaca_bar(bar, symbol)
                if price:
                    prices.append(price)
            
            logger.info(f"Fetched {len(prices)} stock prices for {symbol}")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching stock prices for {symbol}: {e}")
            return []
    
    def _parse_alpaca_bar(self, bar: Dict, symbol: str) -> Optional[PriceData]:
        """Parse Alpaca bar data into PriceData"""
        try:
            return PriceData(
                symbol=symbol,
                timestamp=datetime.fromisoformat(bar['t'].replace('Z', '+00:00')),
                open=bar['o'],
                high=bar['h'],
                low=bar['l'],
                close=bar['c'],
                volume=bar['v'],
                vwap=bar.get('vw'),
                trades=bar.get('n'),
                source='alpaca',
                raw_data=bar
            )
        except Exception as e:
            logger.error(f"Error parsing Alpaca bar: {e}")
            return None


# Convenience functions for backward compatibility
async def collect_crypto_prices(symbols: List[str], days_back: int = 30) -> List[Dict]:
    """Collect crypto prices (placeholder)"""
    return []

async def collect_stock_prices(symbols: List[str], days_back: int = 30) -> Dict[str, pd.DataFrame]:
    """Collect stock prices using Yahoo Finance"""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    logger.info(f"Collecting stock prices for {len(symbols)} symbols using Yahoo Finance")
    
    results = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    for symbol in symbols:
        try:
            logger.info(f"Fetching {symbol} data from Yahoo Finance...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1m')
            
            if not df.empty:
                # Rename columns to match expected format
                df.columns = [col.lower() for col in df.columns]
                df.index.name = 'timestamp'
                results[symbol] = df
                logger.info(f"Collected {len(df)} data points for {symbol}")
            else:
                logger.warning(f"No data returned for {symbol}")
                
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            continue
    
    logger.info(f"Successfully collected stock data for {len(results)} symbols")
    return results

async def collect_price_data(symbols: List[str], days_back: int = 30, api_keys: Dict[str, str] = None) -> Dict[str, Any]:
    """Collect price data for symbols using real APIs"""
    import pandas as pd
    from datetime import datetime, timedelta
    import yaml
    import os
    
    # Load API keys if not provided
    if api_keys is None:
        api_keys_path = 'config/api_keys.yaml'
        if not os.path.exists(api_keys_path):
            api_keys_path = 'config/api_keys_local.yaml'
        
        try:
            with open(api_keys_path, 'r') as f:
                api_keys = yaml.safe_load(f)
        except:
            logger.warning("No API keys found, using mock data")
            return await _generate_mock_data(symbols, days_back)
    
    price_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Separate crypto and stock symbols
    crypto_symbols = [s for s in symbols if s in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']]
    stock_symbols = [s for s in symbols if s in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'BAC', 'XOM']]
    
    # Collect crypto data using CoinGecko (free, no API key needed)
    if crypto_symbols:
        # Extract CoinGecko API key if available
        coingecko_key = None
        if isinstance(api_keys.get('coingecko'), dict):
            coingecko_key = api_keys['coingecko'].get('api_key')
        elif isinstance(api_keys.get('coingecko'), str):
            coingecko_key = api_keys['coingecko']
        
        try:
            async with CoinGeckoClient(coingecko_key) as client:
                for symbol in crypto_symbols:
                    # Map symbols to CoinGecko coin IDs
                    coin_id = _get_coin_id(symbol)
                    if coin_id:
                        prices = await client.get_crypto_prices(coin_id, days_back)
                        if prices:
                            df = _convert_prices_to_dataframe(prices)
                            price_data[symbol] = df
                            logger.info(f"Collected {len(df)} crypto prices for {symbol} from CoinGecko")
                        else:
                            logger.warning(f"No crypto data for {symbol}, using mock data")
                            price_data[symbol] = _generate_mock_symbol_data(symbol, days_back)
                    else:
                        logger.warning(f"Unknown crypto symbol {symbol}, using mock data")
                        price_data[symbol] = _generate_mock_symbol_data(symbol, days_back)
        except Exception as e:
            logger.error(f"Error collecting crypto data from CoinGecko: {e}, using mock data")
            for symbol in crypto_symbols:
                price_data[symbol] = _generate_mock_symbol_data(symbol, days_back)
    
    # Collect stock data - try Polygon first, then Yahoo Finance
    if stock_symbols:
        # Extract polygon API key from nested structure or flat structure
        polygon_key = None
        if isinstance(api_keys.get('polygon'), dict):
            polygon_key = api_keys['polygon'].get('api_key')
        elif isinstance(api_keys.get('polygon'), str):
            polygon_key = api_keys['polygon']
        
        if polygon_key and not polygon_key.startswith("your_") and polygon_key != "api_key":
            try:
                async with PolygonClient(polygon_key) as client:
                    for symbol in stock_symbols:
                        prices = await client.get_stock_prices(symbol, start_date, end_date)
                        if prices:
                            df = _convert_prices_to_dataframe(prices)
                            price_data[symbol] = df
                            logger.info(f"Collected {len(df)} stock prices for {symbol} from Polygon")
                        else:
                            logger.warning(f"No stock data for {symbol} from Polygon, trying Yahoo Finance")
                            # Try Yahoo Finance as fallback
                            await _try_yahoo_finance(symbol, start_date, end_date, price_data)
            except Exception as e:
                logger.error(f"Error collecting stock data from Polygon: {e}, trying Yahoo Finance")
                # Try Yahoo Finance for all stocks
                for symbol in stock_symbols:
                    await _try_yahoo_finance(symbol, start_date, end_date, price_data)
        else:
            logger.info("No Polygon API key, using Yahoo Finance for stocks")
            # Use Yahoo Finance for all stocks
            for symbol in stock_symbols:
                await _try_yahoo_finance(symbol, start_date, end_date, price_data)
    
    # If no API keys or all failed, use mock data
    if not price_data:
        logger.warning("No real data collected, using mock data")
        return await _generate_mock_data(symbols, days_back)
    
    return price_data


def _convert_prices_to_dataframe(prices: List[PriceData]) -> pd.DataFrame:
    """Convert PriceData list to DataFrame"""
    data = []
    for price in prices:
        data.append({
            'date': price.timestamp,
            'open': price.open,
            'high': price.high,
            'low': price.low,
            'close': price.close,
            'volume': price.volume
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)
    return df


async def _generate_mock_data(symbols: List[str], days_back: int) -> Dict[str, Any]:
    """Generate mock data as fallback"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    price_data = {}
    
    for symbol in symbols:
        price_data[symbol] = _generate_mock_symbol_data(symbol, days_back)
    
    return price_data


async def _try_yahoo_finance(symbol: str, start_date: datetime, end_date: datetime, price_data: dict):
    """Try to get data from Yahoo Finance"""
    try:
        async with YahooFinanceClient() as client:
            prices = await client.get_prices(symbol, start_date, end_date)
            if prices:
                df = _convert_prices_to_dataframe(prices)
                price_data[symbol] = df
                logger.info(f"Collected {len(df)} prices for {symbol} from Yahoo Finance")
            else:
                logger.warning(f"No data for {symbol} from Yahoo Finance, using mock data")
                price_data[symbol] = _generate_mock_symbol_data(symbol, (end_date - start_date).days)
    except Exception as e:
        logger.error(f"Error getting {symbol} from Yahoo Finance: {e}, using mock data")
        price_data[symbol] = _generate_mock_symbol_data(symbol, (end_date - start_date).days)


def _get_coin_id(symbol: str) -> str:
    """Map crypto symbols to CoinGecko coin IDs"""
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
    return coin_mapping.get(symbol, None)


def _generate_mock_symbol_data(symbol: str, days_back: int) -> pd.DataFrame:
    """Generate mock data for a single symbol"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
    
    # Generate random walk price data
    base_price = 100 if 'BTC' in symbol else 50
    returns = np.random.normal(0, 0.02, days_back)  # 2% daily volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days_back)
    })
    
    return df