#!/usr/bin/env python3
"""
Continuous News Collection Service
Runs in background to collect news data every hour for all crypto symbols
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time
from dataclasses import dataclass

from data_ingestion.news_collector import NewsCollector
from data_ingestion.news_apis import collect_crypto_news, collect_stock_news
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class NewsCollectionStats:
    """Statistics for continuous news collection"""
    start_time: datetime
    total_collections: int = 0
    successful_collections: int = 0
    failed_collections: int = 0
    articles_collected: int = 0
    last_collection_time: Optional[datetime] = None
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time.isoformat(),
            'total_collections': self.total_collections,
            'successful_collections': self.successful_collections,
            'failed_collections': self.failed_collections,
            'articles_collected': self.articles_collected,
            'last_collection_time': self.last_collection_time.isoformat() if self.last_collection_time else None,
            'last_error': self.last_error,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'success_rate': self.successful_collections / max(self.total_collections, 1) * 100
        }

class ContinuousNewsCollector:
    """Runs continuous news collection in background"""
    
    def __init__(self, data_dir: Path, api_keys: Dict[str, str], symbols: List[str]):
        self.data_dir = data_dir
        self.api_keys = api_keys
        self.symbols = symbols
        self.news_collector = NewsCollector(data_dir, api_keys)
        
        # Stats and control
        self.stats = NewsCollectionStats(start_time=datetime.now())
        self.running = False
        self.interval_hours = 1  # Collect news every hour
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def _collect_news_data(self) -> Optional[Dict[str, Any]]:
        """Collect news data for all symbols"""
        try:
            logger.info(f"Starting news collection for {len(self.symbols)} symbols")
            
            # Collect crypto news
            crypto_news = await collect_crypto_news(
                self.api_keys.get('newsapi', ''), 
                hours_back=24,  # Collect last 24 hours
                max_articles=500
            )
            
            # Collect stock news (if any stock symbols)
            stock_symbols = [s for s in self.symbols if s not in ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']]
            stock_news = []
            if stock_symbols:
                stock_news = await collect_stock_news(
                    self.api_keys.get('newsapi', ''), 
                    hours_back=24,
                    max_articles=200
                )
            
            all_news = crypto_news + stock_news
            
            # Save to parquet file
            if all_news:
                news_df = pd.DataFrame([article.__dict__ for article in all_news])
                news_file = self.data_dir / 'raw' / 'news.parquet'
                news_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Append to existing data or create new
                if news_file.exists():
                    existing_df = pd.read_parquet(news_file)
                    # Remove duplicates based on title and timestamp
                    combined_df = pd.concat([existing_df, news_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['title', 'timestamp'], keep='last')
                    combined_df.to_parquet(news_file, index=False)
                else:
                    news_df.to_parquet(news_file, index=False)
                
                logger.info(f"Saved {len(all_news)} news articles to {news_file}")
            
            return {
                'articles_collected': len(all_news),
                'crypto_articles': len(crypto_news),
                'stock_articles': len(stock_news),
                'collection_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            raise
    
    async def start_continuous_collection(self, interval_hours: int = 1):
        """Start continuous news collection"""
        self.running = True
        self.interval_hours = interval_hours
        
        logger.info(f"Starting continuous news collection every {interval_hours} hours")
        logger.info(f"Collecting news for symbols: {', '.join(self.symbols)}")
        
        try:
            while self.running:
                collection_start = datetime.now()
                self.stats.total_collections += 1
                
                try:
                    # Collect news data
                    result = await self._collect_news_data()
                    
                    if result:
                        self.stats.successful_collections += 1
                        self.stats.articles_collected += result['articles_collected']
                        self.stats.last_collection_time = collection_start
                        
                        logger.info(f"Collection #{self.stats.total_collections}: "
                                  f"{result['articles_collected']} articles "
                                  f"(Crypto: {result['crypto_articles']}, Stock: {result['stock_articles']})")
                    else:
                        self.stats.failed_collections += 1
                        logger.warning(f"Collection #{self.stats.total_collections}: No data collected")
                    
                except Exception as e:
                    self.stats.failed_collections += 1
                    self.stats.last_error = str(e)
                    logger.error(f"Collection #{self.stats.total_collections} failed: {e}")
                
                # Calculate sleep time
                collection_duration = (datetime.now() - collection_start).total_seconds()
                sleep_time = max(0, self.interval_hours * 3600 - collection_duration)
                
                if sleep_time > 0:
                    logger.info(f"Sleeping for {sleep_time/3600:.1f} hours until next collection")
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"Collection took {collection_duration/3600:.2f} hours, longer than interval {self.interval_hours} hours")
                
        except KeyboardInterrupt:
            logger.info("Continuous news collection stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in continuous news collection: {e}")
            raise
        finally:
            self.running = False
            await self._save_final_stats()
    
    async def _save_final_stats(self):
        """Save final collection statistics"""
        stats_file = self.data_dir / 'logs' / 'news_collection_stats.json'
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        
        logger.info(f"Final stats saved to {stats_file}")
        logger.info(f"Total collections: {self.stats.total_collections}")
        logger.info(f"Success rate: {self.stats.successful_collections / max(self.stats.total_collections, 1) * 100:.1f}%")
        logger.info(f"Total articles collected: {self.stats.articles_collected}")

async def run_continuous_news_collection(data_dir: str, symbols: List[str], interval_hours: int = 1):
    """Run continuous news collection service"""
    data_path = Path(data_dir)
    
    # Load API keys from environment variables
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_keys = {
        'newsapi': os.getenv('NEWSAPI_KEY', ''),
        'cryptocompare': os.getenv('CRYPTOCOMPARE_KEY', ''),
        'reddit_client_id': os.getenv('REDDIT_CLIENT_ID', ''),
        'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET', ''),
        'reddit_user_agent': os.getenv('REDDIT_USER_AGENT', ''),
    }
    
    # Start collection
    collector = ContinuousNewsCollector(data_path, api_keys, symbols)
    await collector.start_continuous_collection(interval_hours)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous News Collection Service")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--symbols", nargs="+", 
                       default=["BTC", "ETH", "ADA", "SOL", "AVAX", "DOT", "LINK", "MATIC", "UNI"],
                       help="Symbols to collect news for")
    parser.add_argument("--interval", type=float, default=1.0, help="Collection interval in hours")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler('logs/continuous_news_collection.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run continuous news collection
    asyncio.run(run_continuous_news_collection(args.data_dir, args.symbols, args.interval))
