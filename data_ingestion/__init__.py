"""
Enhanced Crypto Trading Pipeline - Data Ingestion Module

This module handles all data collection from various sources:
- News APIs (NewsAPI, RavenPack)
- Price APIs (Polygon, Alpaca, Binance, CoinGecko)
- Social Media (Reddit, Twitter)
- Web Scraping (News sites, forums)

Key Features:
- Robust error handling and retry logic
- Rate limiting and API quota management
- Data validation and quality assurance
- Real-time and historical data collection
- Unified data format across all sources
"""

from .news_apis import NewsAPIClient, RavenPackClient
from .price_apis import PolygonClient, AlpacaClient, BinanceClient, CoinGeckoClient
from .coinbase_client import CoinbaseClient, CoinbaseConfig, create_coinbase_client
from .coinbase_rest_client import CoinbaseRESTClient, create_coinbase_rest_client
from .social_apis import RedditClient, TwitterClient
from .data_validator import DataValidator
from .unified_collector import UnifiedDataCollector

__all__ = [
    'NewsAPIClient',
    'RavenPackClient', 
    'PolygonClient',
    'AlpacaClient',
    'BinanceClient',
    'CoinGeckoClient',
    'CoinbaseClient',
    'CoinbaseConfig',
    'create_coinbase_client',
    'CoinbaseRESTClient',
    'create_coinbase_rest_client',
    'RedditClient',
    'TwitterClient',
    'DataValidator',
    'UnifiedDataCollector'
]

