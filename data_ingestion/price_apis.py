"""
Price API Clients for collecting financial market data

Supports multiple price data sources:
- Polygon.io (stocks, crypto, forex)
- Alpaca (stocks, crypto)
- Binance (cryptocurrency)
- CoinGecko (cryptocurrency)
- Yahoo Finance (fallback)
"""

import asyncio
import aiohttp
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


@dataclass
class MarketMetrics:
    """Market-wide metrics"""
    timestamp: datetime
    total_market_cap: Optional[float] = None
    fear_greed_index: Optional[float] = None
    vix: Optional[float] = None
    dxy: Optional[float] = None  # Dollar Index
    gold_price: Optional[float] = None
    oil_price: Optional[float] = None
    source: str = "unknown"


class PolygonClient:
    """Polygon.io client for stocks, crypto, and forex data"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.polygon.io"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def get_stock_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timespan: str = "day",
        multiplier: int = 1
    ) -> List[PriceData]:
        """Get stock price data from Polygon"""
        await self._rate_limit()
        
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        params = {
            'apikey': self.api_key,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'timespan': timespan,
            'multiplier': multiplier,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            async with self.session.get(url, params=params) as response:
                if response.status == 429:
                    logger.warning("Rate limit exceeded, waiting 60 seconds")
                    await asyncio.sleep(60)
                    return await self.get_stock_prices(symbol, start_date, end_date, timespan, multiplier)
                
                response.raise_for_status()
                data = await response.json()
                
                prices = []
                for result in data.get('results', []):
                    price = self._parse_polygon_result(result, symbol, 'stock')
                    if price:
                        prices.append(price)
                
                logger.info(f"Fetched {len(prices)} price points for {symbol}")
                return prices
                
        except Exception as e:
            logger.error(f"Error fetching stock prices for {symbol}: {e}")
            return []
    
    async def get_crypto_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timespan: str = "day"
    ) -> List[PriceData]:
        """Get cryptocurrency price data from Polygon"""
        await self._rate_limit()
        
        # Convert symbol format (BTC -> X:BTCUSD)
        polygon_symbol = f"X:{symbol}USD" if not symbol.startswith('X:') else symbol
        
        params = {
            'apikey': self.api_key,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'timespan': timespan,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{polygon_symbol}/range/1/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            async with self.session.get(url, params=params) as response:
                if response.status == 429:
                    logger.warning("Rate limit exceeded, waiting 60 seconds")
                    await asyncio.sleep(60)
                    return await self.get_crypto_prices(symbol, start_date, end_date, timespan)
                
                response.raise_for_status()
                data = await response.json()
                
                prices = []
                for result in data.get('results', []):
                    price = self._parse_polygon_result(result, symbol, 'crypto')
                    if price:
                        prices.append(price)
                
                logger.info(f"Fetched {len(prices)} price points for {symbol}")
                return prices
                
        except Exception as e:
            logger.error(f"Error fetching crypto prices for {symbol}: {e}")
            return []
    
    def _parse_polygon_result(self, result: Dict, symbol: str, asset_type: str) -> Optional[PriceData]:
        """Parse Polygon API result into PriceData"""
        try:
            timestamp = datetime.fromtimestamp(result['t'] / 1000)
            
            return PriceData(
                symbol=symbol,
                timestamp=timestamp,
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
    """Binance client for cryptocurrency data"""
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.binance.com"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1d",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[PriceData]:
        """Get kline/candlestick data from Binance"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000)
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        try:
            async with self.session.get(f"{self.base_url}/api/v3/klines", params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                prices = []
                for kline in data:
                    price = self._parse_binance_kline(kline, symbol)
                    if price:
                        prices.append(price)
                
                logger.info(f"Fetched {len(prices)} klines for {symbol}")
                return prices
                
        except Exception as e:
            logger.error(f"Error fetching Binance klines for {symbol}: {e}")
            return []
    
    def _parse_binance_kline(self, kline: List, symbol: str) -> Optional[PriceData]:
        """Parse Binance kline data"""
        try:
            return PriceData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(int(kline[0]) / 1000),
                open=float(kline[1]),
                high=float(kline[2]),
                low=float(kline[3]),
                close=float(kline[4]),
                volume=float(kline[5]),
                trades=int(kline[8]),
                source='binance',
                raw_data=kline
            )
        except Exception as e:
            logger.error(f"Error parsing Binance kline: {e}")
            return None
    
    async def get_ticker_24hr(self, symbol: str) -> Optional[Dict]:
        """Get 24hr ticker price change statistics"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            async with self.session.get(f"{self.base_url}/api/v3/ticker/24hr", params={'symbol': symbol}) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error fetching 24hr ticker for {symbol}: {e}")
            return None


class CoinGeckoClient:
    """CoinGecko client for cryptocurrency data"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = None
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def get_coin_market_data(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 30
    ) -> Optional[Dict]:
        """Get market data for a specific coin"""
        await self._rate_limit()
        
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': 'daily' if days > 1 else 'hourly'
        }
        
        if self.api_key:
            params['x_cg_demo_api_key'] = self.api_key
        
        try:
            async with self.session.get(f"{self.base_url}/coins/{coin_id}/market_chart", params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error fetching market data for {coin_id}: {e}")
            return None
    
    async def get_fear_greed_index(self) -> Optional[float]:
        """Get current Fear & Greed Index"""
        await self._rate_limit()
        
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            async with self.session.get(f"{self.base_url}/fear-greed-index") as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('data', [{}])[0].get('value')
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return None


class AlpacaClient:
    """Alpaca client for stocks and crypto data"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        }
        self.session = aiohttp.ClientSession(headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Day"
    ) -> List[PriceData]:
        """Get historical bars from Alpaca"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        params = {
            'symbols': symbol,
            'start': start.isoformat(),
            'end': end.isoformat(),
            'timeframe': timeframe,
            'asof': None,
            'feed': None,
            'page_token': None,
            'sort': 'asc',
            'limit': 10000
        }
        
        try:
            async with self.session.get(f"{self.base_url}/v2/stocks/bars", params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                prices = []
                for bar in data.get('bars', {}).get(symbol, []):
                    price = self._parse_alpaca_bar(bar, symbol)
                    if price:
                        prices.append(price)
                
                logger.info(f"Fetched {len(prices)} bars for {symbol}")
                return prices
                
        except Exception as e:
            logger.error(f"Error fetching Alpaca bars for {symbol}: {e}")
            return []
    
    def _parse_alpaca_bar(self, bar: Dict, symbol: str) -> Optional[PriceData]:
        """Parse Alpaca bar data"""
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


async def collect_price_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    polygon_key: Optional[str] = None,
    binance_key: Optional[str] = None,
    coingecko_key: Optional[str] = None
) -> Dict[str, List[PriceData]]:
    """
    Collect price data from multiple sources
    
    Args:
        symbols: List of symbols to collect data for
        start_date: Start date for data collection
        end_date: End date for data collection
        polygon_key: Polygon.io API key
        binance_key: Binance API key (optional)
        coingecko_key: CoinGecko API key (optional)
    """
    all_data = {}
    
    # Determine which symbols are crypto vs stocks
    crypto_symbols = []
    stock_symbols = []
    
    for symbol in symbols:
        if symbol.upper() in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']:
            crypto_symbols.append(symbol)
        else:
            stock_symbols.append(symbol)
    
    # Collect crypto data
    if crypto_symbols and (binance_key or coingecko_key):
        if binance_key:
            async with BinanceClient(binance_key) as client:
                for symbol in crypto_symbols:
                    try:
                        data = await client.get_klines(f"{symbol}USDT", "1d", start_date, end_date)
                        all_data[symbol] = data
                    except Exception as e:
                        logger.error(f"Error collecting Binance data for {symbol}: {e}")
        
        if coingecko_key:
            async with CoinGeckoClient(coingecko_key) as client:
                for symbol in crypto_symbols:
                    try:
                        # Map symbol to CoinGecko ID
                        coin_id = symbol.lower()
                        data = await client.get_coin_market_data(coin_id, "usd", (end_date - start_date).days)
                        if data:
                            prices = _parse_coingecko_data(data, symbol)
                            if symbol not in all_data:
                                all_data[symbol] = prices
                    except Exception as e:
                        logger.error(f"Error collecting CoinGecko data for {symbol}: {e}")
    
    # Collect stock data
    if stock_symbols and polygon_key:
        async with PolygonClient(polygon_key) as client:
            for symbol in stock_symbols:
                try:
                    data = await client.get_stock_prices(symbol, start_date, end_date)
                    all_data[symbol] = data
                except Exception as e:
                    logger.error(f"Error collecting Polygon data for {symbol}: {e}")
    
    logger.info(f"Collected price data for {len(all_data)} symbols")
    return all_data


def _parse_coingecko_data(data: Dict, symbol: str) -> List[PriceData]:
    """Parse CoinGecko market chart data"""
    prices = []
    
    try:
        price_data = data.get('prices', [])
        volume_data = data.get('total_volumes', [])
        
        for i, (timestamp, price) in enumerate(price_data):
            volume = volume_data[i][1] if i < len(volume_data) else 0
            
            prices.append(PriceData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(timestamp / 1000),
                open=price,  # CoinGecko doesn't provide OHLC, use close as all values
                high=price,
                low=price,
                close=price,
                volume=volume,
                source='coingecko',
                raw_data={'timestamp': timestamp, 'price': price, 'volume': volume}
            ))
    except Exception as e:
        logger.error(f"Error parsing CoinGecko data: {e}")
    
    return prices


if __name__ == "__main__":
    # Example usage
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def main():
        symbols = ['BTC', 'ETH', 'AAPL', 'TSLA']
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        polygon_key = os.getenv('POLYGON_API_KEY')
        binance_key = os.getenv('BINANCE_API_KEY')
        coingecko_key = os.getenv('COINGECKO_API_KEY')
        
        data = await collect_price_data(symbols, start_date, end_date, polygon_key, binance_key, coingecko_key)
        
        for symbol, prices in data.items():
            print(f"{symbol}: {len(prices)} price points")
    
    asyncio.run(main())
