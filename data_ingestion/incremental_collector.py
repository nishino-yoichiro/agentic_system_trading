"""
Incremental Data Collection System
Handles bulk historical data loading and real-time incremental updates
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum

from .price_apis import collect_crypto_prices, collect_stock_prices
from .news_apis import collect_crypto_news, collect_stock_news
from .high_frequency_apis import collect_high_frequency_data

logger = logging.getLogger(__name__)

class DataType(Enum):
    NEWS = "news"
    PRICE = "price"

class RefreshStrategy(Enum):
    BULK_HISTORICAL = "bulk_historical"  # One-time bulk load
    INCREMENTAL = "incremental"          # Real-time updates
    DAILY = "daily"                      # Daily refresh

@dataclass
class DataCollectionConfig:
    """Configuration for data collection strategies"""
    data_type: DataType
    refresh_strategy: RefreshStrategy
    bulk_days_back: int = 365  # For historical data
    incremental_interval: int = 20  # seconds
    max_historical_points: int = 500000  # Polygon limit
    cache_ttl_hours: int = 24  # For news
    price_cache_ttl_minutes: int = 1  # For price data
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class IncrementalDataCollector:
    """Handles incremental data collection with different strategies per data type"""
    
    def __init__(self, data_dir: Path, api_keys: Dict[str, str]):
        self.data_dir = data_dir
        self.api_keys = api_keys
        self.raw_dir = data_dir / 'raw'
        self.incremental_dir = data_dir / 'incremental'
        self.metadata_file = data_dir / 'collection_metadata.json'
        
        # Create directories
        self.raw_dir.mkdir(exist_ok=True)
        self.incremental_dir.mkdir(exist_ok=True)
        
        # Load collection metadata
        self.metadata = self._load_metadata()
        
        # Collection configs
        self.configs = {
            DataType.NEWS: DataCollectionConfig(
                data_type=DataType.NEWS,
                refresh_strategy=RefreshStrategy.DAILY,
                cache_ttl_hours=24
            ),
            DataType.PRICE: DataCollectionConfig(
                data_type=DataType.PRICE,
                refresh_strategy=RefreshStrategy.INCREMENTAL,
                incremental_interval=20,
                price_cache_ttl_minutes=1
            )
        }
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load collection metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metadata: {e}")
        
        return {
            'last_bulk_collection': {},
            'last_incremental_update': {},
            'collection_stats': {},
            'api_usage': {}
        }
    
    def _save_metadata(self):
        """Save collection metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _get_data_age(self, file_path: Path, ttl_hours: int = 24) -> Tuple[bool, float]:
        """Check if data is fresh enough based on TTL"""
        if not file_path.exists():
            return False, 0
        
        file_age_hours = (datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)).total_seconds() / 3600
        is_fresh = file_age_hours < ttl_hours
        
        return is_fresh, file_age_hours
    
    async def collect_bulk_historical_data(self, symbols: List[str], days_back: int = 365, use_high_frequency: bool = True) -> Dict[str, Any]:
        """Collect bulk historical data for initial setup"""
        logger.info(f"Starting bulk historical data collection for {len(symbols)} symbols, {days_back} days back")
        logger.info(f"High-frequency mode: {use_high_frequency}")
        
        start_time = datetime.now()
        results = {
            'crypto_data': {},
            'stock_data': {},
            'news_data': [],
            'collection_time': start_time,
            'symbols_processed': 0,
            'api_calls_made': 0,
            'data_granularity': 'high_frequency' if use_high_frequency else 'daily'
        }
        
        try:
            if use_high_frequency:
                # Use high-frequency APIs for minute-level data
                logger.info("Using high-frequency data collection (minute-level)")
                hf_data = await collect_high_frequency_data(symbols, days_back, self.api_keys)
                
                # Separate crypto and stock data
                crypto_symbols = [s for s in symbols if s in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']]
                stock_symbols = [s for s in symbols if s not in crypto_symbols]
                
                for symbol, df in hf_data.items():
                    if symbol in crypto_symbols:
                        results['crypto_data'][symbol] = df
                    else:
                        results['stock_data'][symbol] = df
                
                results['api_calls_made'] = len(symbols)
                results['symbols_processed'] = len(hf_data)
                
            else:
                # Use standard daily APIs
                logger.info("Using standard daily data collection")
                crypto_symbols = [s for s in symbols if s in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']]
                if crypto_symbols:
                    logger.info(f"Collecting crypto data for {crypto_symbols}")
                    crypto_data = await collect_crypto_prices(crypto_symbols, days_back, self.api_keys)
                    results['crypto_data'] = crypto_data
                    results['api_calls_made'] += len(crypto_symbols)
                
                stock_symbols = [s for s in symbols if s not in crypto_symbols]
                if stock_symbols:
                    logger.info(f"Collecting stock data for {stock_symbols}")
                    stock_data = await collect_stock_prices(stock_symbols, days_back, self.api_keys)
                    results['stock_data'] = stock_data
                    results['api_calls_made'] += len(stock_symbols)
            
            # Collect news data
            logger.info("Collecting historical news data")
            crypto_news = await collect_crypto_news(self.api_keys.get('newsapi', ''), days_back)
            stock_news = await collect_stock_news(self.api_keys.get('newsapi', ''), days_back)
            results['news_data'] = crypto_news + stock_news
            results['api_calls_made'] += 2  # 2 news API calls
            
            # Save bulk data
            await self._save_bulk_data(results)
            
            # Update metadata
            self.metadata['last_bulk_collection'] = {
                'timestamp': start_time.isoformat(),
                'symbols': symbols,
                'days_back': days_back,
                'api_calls': results['api_calls_made'],
                'high_frequency': use_high_frequency,
                'data_granularity': results['data_granularity']
            }
            self._save_metadata()
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Bulk collection completed in {duration:.2f} seconds, {results['api_calls_made']} API calls made")
            logger.info(f"Data granularity: {results['data_granularity']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk historical collection: {e}")
            raise
    
    async def _save_bulk_data(self, data: Dict[str, Any]):
        """Save bulk historical data"""
        try:
            # Save crypto data
            for symbol, df in data['crypto_data'].items():
                file_path = self.raw_dir / f'prices_{symbol}.parquet'
                df.to_parquet(file_path)
                logger.info(f"Saved {len(df)} crypto price points for {symbol}")
            
            # Save stock data
            for symbol, df in data['stock_data'].items():
                file_path = self.raw_dir / f'prices_{symbol}.parquet'
                df.to_parquet(file_path)
                logger.info(f"Saved {len(df)} stock price points for {symbol}")
            
            # Save news data
            if data['news_data']:
                news_df = pd.DataFrame([article.__dict__ for article in data['news_data']])
                news_file = self.raw_dir / 'news.parquet'
                news_df.to_parquet(news_file)
                logger.info(f"Saved {len(data['news_data'])} news articles")
                
        except Exception as e:
            logger.error(f"Error saving bulk data: {e}")
            raise
    
    async def collect_incremental_price_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Collect incremental price data (real-time updates)"""
        logger.info(f"Collecting incremental price data for {symbols}")
        
        start_time = datetime.now()
        results = {
            'price_updates': {},
            'collection_time': start_time,
            'symbols_updated': 0,
            'api_calls_made': 0
        }
        
        try:
            # Collect latest price data (1 day back for incremental)
            crypto_symbols = [s for s in symbols if s in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']]
            stock_symbols = [s for s in symbols if s not in crypto_symbols]
            
            # Get crypto updates
            if crypto_symbols:
                crypto_data = await collect_crypto_prices(crypto_symbols, 1, self.api_keys)
                results['price_updates'].update(crypto_data)
                results['api_calls_made'] += len(crypto_symbols)
            
            # Get stock updates
            if stock_symbols:
                stock_data = await collect_stock_prices(stock_symbols, 1)
                # Convert DataFrame to dict format for consistency
                for symbol, df in stock_data.items():
                    results['price_updates'][symbol] = df
                results['api_calls_made'] += len(stock_symbols)
            
            # Merge with existing data
            await self._merge_incremental_data(results['price_updates'])
            
            # Update metadata
            self.metadata['last_incremental_update'] = {
                'timestamp': start_time.isoformat(),
                'symbols': symbols,
                'api_calls': results['api_calls_made']
            }
            self._save_metadata()
            
            results['symbols_updated'] = len(symbols)
            logger.info(f"Incremental update completed: {results['symbols_updated']} symbols, {results['api_calls_made']} API calls")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in incremental collection: {e}")
            raise
    
    async def _merge_incremental_data(self, new_data: Dict[str, pd.DataFrame]):
        """Merge new incremental data with existing historical data"""
        for symbol, new_df in new_data.items():
            if new_df.empty:
                continue
                
            # Load existing data
            existing_file = self.raw_dir / f'prices_{symbol}.parquet'
            if existing_file.exists():
                existing_df = pd.read_parquet(existing_file)
                
                # Find new data points (not in existing data)
                if 'timestamp' in new_df.columns and 'timestamp' in existing_df.columns:
                    # Convert to datetime if needed
                    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                    
                    # Find new timestamps
                    existing_timestamps = set(existing_df['timestamp'].dt.floor('T'))  # Round to minute
                    new_timestamps = set(new_df['timestamp'].dt.floor('T'))
                    truly_new = new_timestamps - existing_timestamps
                    
                    if truly_new:
                        # Filter to only truly new data
                        new_df_filtered = new_df[new_df['timestamp'].dt.floor('T').isin(truly_new)]
                        
                        if not new_df_filtered.empty:
                            # Append new data
                            combined_df = pd.concat([existing_df, new_df_filtered], ignore_index=True)
                            combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
                            
                            # Save updated data
                            combined_df.to_parquet(existing_file)
                            logger.info(f"Added {len(new_df_filtered)} new data points for {symbol}")
                        else:
                            logger.info(f"No new data points for {symbol}")
                    else:
                        logger.info(f"No new timestamps for {symbol}")
                else:
                    # If no timestamp column, just append
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df.to_parquet(existing_file)
                    logger.info(f"Appended {len(new_df)} data points for {symbol}")
            else:
                # No existing data, save as new
                new_df.to_parquet(existing_file)
                logger.info(f"Created new price file for {symbol} with {len(new_df)} points")
    
    async def collect_news_data(self, hours_back: int = 24) -> Dict[str, Any]:
        """Collect news data (daily refresh)"""
        logger.info(f"Collecting news data for last {hours_back} hours")
        
        start_time = datetime.now()
        results = {
            'news_data': [],
            'collection_time': start_time,
            'articles_collected': 0
        }
        
        try:
            # Check if news data is fresh enough
            news_file = self.raw_dir / 'news.parquet'
            is_fresh, age_hours = self._get_data_age(news_file, self.configs[DataType.NEWS].cache_ttl_hours)
            
            if is_fresh:
                logger.info(f"News data is fresh ({age_hours:.1f} hours old), skipping collection")
                if news_file.exists():
                    news_df = pd.read_parquet(news_file)
                    results['articles_collected'] = len(news_df)
                return results
            
            # Collect fresh news data
            crypto_news = await collect_crypto_news(self.api_keys.get('newsapi', ''), hours_back // 24 + 1)
            stock_news = await collect_stock_news(self.api_keys.get('newsapi', ''), hours_back // 24 + 1)
            
            all_news = crypto_news + stock_news
            results['news_data'] = all_news
            results['articles_collected'] = len(all_news)
            
            # Save news data
            if all_news:
                news_df = pd.DataFrame([article.__dict__ for article in all_news])
                news_df.to_parquet(news_file)
                logger.info(f"Saved {len(all_news)} news articles")
            
            return results
            
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            raise
    
    async def get_consolidated_data(self, symbols: List[str], days_back: int = 7) -> Dict[str, Any]:
        """Get consolidated data (historical + incremental) for analysis"""
        logger.info(f"Consolidating data for {len(symbols)} symbols (last {days_back} days)")
        
        consolidated = {
            'price_data': {},
            'news_data': [],
            'data_ages': {},
            'total_data_points': 0
        }
        
        # Calculate cutoff date (timezone-aware)
        from datetime import timezone
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        try:
            # Load price data
            for symbol in symbols:
                price_file = self.raw_dir / f'prices_{symbol}.parquet'
                if price_file.exists():
                    df = pd.read_parquet(price_file)
                    
                    # Filter to last N days
                    if not df.empty:
                        # Ensure we have a datetime index
                        if not isinstance(df.index, pd.DatetimeIndex):
                            if 'timestamp' in df.columns:
                                # Convert timestamp column to datetime and set as index
                                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                                df = df.set_index('timestamp')
                            elif df.index.name == 'timestamp':
                                # Index is already named 'timestamp', just convert to datetime
                                df.index = pd.to_datetime(df.index, utc=True)
                            else:
                                logger.warning(f"No timestamp column found for {symbol}")
                                continue
                        
                        # Ensure the index is timezone-aware
                        if df.index.tz is None:
                            df.index = df.index.tz_localize('UTC')
                        elif hasattr(cutoff_date, 'tz') and df.index.tz != cutoff_date.tz:
                            df.index = df.index.tz_convert('UTC')
                        
                        # Filter to last N days
                        try:
                            recent_df = df[df.index >= cutoff_date]
                        except Exception as e:
                            logger.error(f"Error filtering {symbol} data: {e}")
                            logger.error(f"Index type: {type(df.index)}, cutoff_date type: {type(cutoff_date)}")
                            logger.error(f"Index timezone: {getattr(df.index, 'tz', 'None')}")
                            continue
                        logger.info(f"{symbol}: {len(df)} total points, {len(recent_df)} points in last {days_back} days")
                        
                        consolidated['price_data'][symbol] = recent_df
                        consolidated['total_data_points'] += len(recent_df)
                        
                        # Check data age
                        is_fresh, age_hours = self._get_data_age(price_file, 1)  # 1 hour for price data
                        consolidated['data_ages'][symbol] = {
                            'age_hours': age_hours,
                            'is_fresh': is_fresh,
                            'data_points': len(recent_df),
                            'total_points': len(df)
                        }
                    else:
                        logger.warning(f"No valid data found for {symbol}")
                else:
                    logger.warning(f"No price data found for {symbol}")
            
            # Load news data
            news_file = self.raw_dir / 'news.parquet'
            if news_file.exists():
                news_df = pd.read_parquet(news_file)
                consolidated['news_data'] = news_df.to_dict('records')
                
                is_fresh, age_hours = self._get_data_age(news_file, 24)  # 24 hours for news
                consolidated['data_ages']['news'] = {
                    'age_hours': age_hours,
                    'is_fresh': is_fresh,
                    'articles': len(news_df)
                }
            
            logger.info(f"Consolidated data: {consolidated['total_data_points']} price points, {len(consolidated['news_data'])} news articles")
            return consolidated
            
        except Exception as e:
            logger.error(f"Error consolidating data: {e}")
            raise
    
    async def start_continuous_collection(self, symbols: List[str], interval_seconds: int = 20):
        """Start continuous incremental data collection"""
        logger.info(f"Starting continuous collection for {symbols} every {interval_seconds} seconds")
        
        try:
            while True:
                try:
                    await self.collect_incremental_price_data(symbols)
                    await asyncio.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Error in continuous collection: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute on error
                    
        except KeyboardInterrupt:
            logger.info("Continuous collection stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in continuous collection: {e}")
            raise
