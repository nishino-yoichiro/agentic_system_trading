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
from .data_validator import DataValidator
from .unified_collector import UnifiedDataCollector

# Optional imports for exchange clients (may not be available in all environments)
try:
    from .coinbase_client import CoinbaseClient, CoinbaseConfig, create_coinbase_client
    from .coinbase_rest_client import CoinbaseRESTClient, create_coinbase_rest_client
    from .social_apis import RedditClient, TwitterClient
    _HAS_EXCHANGE_CLIENTS = True
except ImportError:
    _HAS_EXCHANGE_CLIENTS = False
    # Create dummy classes for when exchange clients are not available
    class CoinbaseClient:
        def __init__(self, *args, **kwargs):
            raise ImportError("Coinbase client not available")
    
    class CoinbaseConfig:
        pass
    
    def create_coinbase_client(*args, **kwargs):
        raise ImportError("Coinbase client not available")
    
    class CoinbaseRESTClient:
        def __init__(self, *args, **kwargs):
            raise ImportError("Coinbase REST client not available")
    
    def create_coinbase_rest_client(*args, **kwargs):
        raise ImportError("Coinbase REST client not available")
    
    class RedditClient:
        def __init__(self, *args, **kwargs):
            raise ImportError("Reddit client not available")
    
    class TwitterClient:
        def __init__(self, *args, **kwargs):
            raise ImportError("Twitter client not available")

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

