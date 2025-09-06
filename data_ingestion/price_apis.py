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
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.coingecko.com/api/v3"):
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
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            if self.api_key:
                params['x_cg_demo_api_key'] = self.api_key
            
            response = self.session.get(f"{self.base_url}/coins/{coin_id}/market_chart", params=params)
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

async def collect_stock_prices(symbols: List[str], days_back: int = 30) -> List[Dict]:
    """Collect stock prices (placeholder)"""
    return []

async def collect_price_data(symbols: List[str], days_back: int = 30) -> Dict[str, Any]:
    """Collect price data for symbols (mock implementation)"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate mock price data for testing
    price_data = {}
    
    for symbol in symbols:
        # Generate 30 days of mock OHLCV data
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
        
        price_data[symbol] = df
    
    return price_data