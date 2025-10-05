"""
Continuous Data Collection Service
Runs in background to collect incremental price data every 20 seconds
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

from .incremental_collector import IncrementalDataCollector, DataType, RefreshStrategy
from .crypto_collector import CryptoDataCollector
from .realtime_price_collector import RealtimePriceCollector

logger = logging.getLogger(__name__)

@dataclass
class CollectionStats:
    """Statistics for continuous collection"""
    start_time: datetime
    total_updates: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    api_calls_made: int = 0
    last_update_time: Optional[datetime] = None
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time.isoformat(),
            'total_updates': self.total_updates,
            'successful_updates': self.successful_updates,
            'failed_updates': self.failed_updates,
            'api_calls_made': self.api_calls_made,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'last_error': self.last_error,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'success_rate': self.successful_updates / max(self.total_updates, 1) * 100
        }

class ContinuousDataCollector:
    """Runs continuous data collection in background"""
    
    def __init__(self, data_dir: Path, api_keys: Dict[str, str], symbols: List[str]):
        self.data_dir = data_dir
        self.api_keys = api_keys
        self.symbols = symbols
        self.collector = IncrementalDataCollector(data_dir, api_keys)
        self.crypto_collector = CryptoDataCollector(api_keys)
        self.realtime_collector = RealtimePriceCollector()
        
        # Stats and control
        self.stats = CollectionStats(start_time=datetime.now())
        self.running = False
        self.interval_seconds = 20
        
        # Rate limiting
        self.api_limits = {
            'polygon': {'calls_per_minute': 0, 'last_reset': datetime.now()},
            'coingecko': {'calls_per_minute': 0, 'last_reset': datetime.now()},
            'coinbase': {'calls_per_minute': 0, 'last_reset': datetime.now()}
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _check_rate_limits(self) -> bool:
        """Check if we can make API calls within rate limits"""
        now = datetime.now()
        
        # Determine which APIs we'll actually use
        crypto_symbols = [s for s in self.symbols if s in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']]
        stock_symbols = [s for s in self.symbols if s not in crypto_symbols]
        
        # Check only the APIs we'll actually use
        apis_to_check = []
        if crypto_symbols:
            apis_to_check.append('coinbase')
        if stock_symbols:
            apis_to_check.append('polygon')
        
        for api in apis_to_check:
            limits = self.api_limits[api]
            # Reset counter if minute has passed
            if (now - limits['last_reset']).total_seconds() >= 60:
                limits['calls_per_minute'] = 0
                limits['last_reset'] = now
            
            # Check if we're at limit
            if api == 'polygon' and limits['calls_per_minute'] >= 5:
                logger.warning("Polygon rate limit reached, waiting...")
                return False
            elif api == 'coinbase' and limits['calls_per_minute'] >= 10:
                logger.warning("Coinbase rate limit reached, waiting...")
                return False
        
        return True
    
    def _record_api_call(self, api: str):
        """Record an API call for rate limiting"""
        if api in self.api_limits:
            self.api_limits[api]['calls_per_minute'] += 1
            self.stats.api_calls_made += 1
    
    async def _collect_with_rate_limiting(self) -> Optional[Dict[str, Any]]:
        """Collect data with rate limiting"""
        if not self._check_rate_limits():
            return None
        
        try:
            # Determine which APIs we'll use
            crypto_symbols = [s for s in self.symbols if s in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']]
            stock_symbols = [s for s in self.symbols if s not in crypto_symbols]
            
            results = {
                'price_updates': {},
                'collection_time': datetime.now(),
                'symbols_updated': 0,
                'api_calls_made': 0
            }
            
            # Collect crypto data using hybrid approach
            if crypto_symbols:
                logger.info(f"Collecting crypto data for {crypto_symbols} using hybrid approach")
                try:
                    # First, try to get historical data with proper OHLCV
                    crypto_result = await self.crypto_collector.collect_crypto_data(
                        symbols=crypto_symbols,
                        days_back=1
                    )
                    if crypto_result:
                        results['price_updates'].update(crypto_result)
                        results['api_calls_made'] += len(crypto_symbols)
                        self._record_api_call('coinbase')
                        logger.info(f"Updated {len(crypto_result)} crypto symbols with historical data")
                    
                    # Then, supplement with real-time price updates for current minute
                    try:
                        realtime_result = self.realtime_collector.collect_and_save_prices(
                            symbols=crypto_symbols,
                            data_dir=self.data_dir
                        )
                        if realtime_result:
                            logger.info(f"Supplemented with real-time prices for {len(realtime_result)} symbols")
                    except Exception as e:
                        logger.warning(f"Could not supplement with real-time data: {e}")
                        
                except Exception as e:
                    logger.error(f"Error collecting crypto data: {e}")
                    # Fallback to real-time only if historical fails
                    try:
                        realtime_result = self.realtime_collector.collect_and_save_prices(
                            symbols=crypto_symbols,
                            data_dir=self.data_dir
                        )
                        if realtime_result:
                            price_updates = {symbol: data['price'] for symbol, data in realtime_result.items()}
                            results['price_updates'].update(price_updates)
                            results['api_calls_made'] += len(crypto_symbols)
                            self._record_api_call('coinbase')
                            logger.info(f"Fallback: Updated {len(realtime_result)} crypto symbols with real-time prices only")
                    except Exception as e2:
                        logger.error(f"Both historical and real-time collection failed: {e2}")
            
            # Collect stock data using existing method
            if stock_symbols:
                stock_result = await self.collector.collect_incremental_price_data(stock_symbols)
                if stock_result and 'price_updates' in stock_result:
                    results['price_updates'].update(stock_result['price_updates'])
                    results['api_calls_made'] += stock_result.get('api_calls_made', 0)
                    self._record_api_call('polygon')
            
            # Merge with existing data
            if results['price_updates']:
                await self.collector._merge_incremental_data(results['price_updates'])
                results['symbols_updated'] = len(results['price_updates'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in rate-limited collection: {e}")
            self.stats.last_error = str(e)
            return None
    
    async def start_continuous_collection(self, interval_seconds: int = 20):
        """Start continuous data collection"""
        self.interval_seconds = interval_seconds
        self.running = True
        
        logger.info(f"Starting continuous collection for {self.symbols}")
        logger.info(f"Update interval: {interval_seconds} seconds")
        logger.info(f"Rate limits: Polygon=5/min, CoinGecko=50/min, Coinbase=10/min")
        
        try:
            while self.running:
                update_start = datetime.now()
                self.stats.total_updates += 1
                
                try:
                    # Collect incremental data
                    result = await self._collect_with_rate_limiting()
                    
                    if result:
                        self.stats.successful_updates += 1
                        self.stats.last_update_time = update_start
                        
                        logger.info(f"Update #{self.stats.total_updates}: "
                                  f"{result.get('symbols_updated', 0)} symbols, "
                                  f"{result.get('api_calls_made', 0)} API calls")
                    else:
                        self.stats.failed_updates += 1
                        logger.warning(f"Update #{self.stats.total_updates}: Skipped due to rate limits")
                    
                except Exception as e:
                    self.stats.failed_updates += 1
                    self.stats.last_error = str(e)
                    logger.error(f"Update #{self.stats.total_updates} failed: {e}")
                
                # Calculate sleep time
                update_duration = (datetime.now() - update_start).total_seconds()
                sleep_time = max(0, self.interval_seconds - update_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"Update took {update_duration:.2f}s, longer than interval {self.interval_seconds}s")
                
        except KeyboardInterrupt:
            logger.info("Continuous collection stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in continuous collection: {e}")
            raise
        finally:
            self.running = False
            await self._save_final_stats()
    
    async def _save_final_stats(self):
        """Save final collection statistics"""
        try:
            stats_file = self.data_dir / 'collection_stats.json'
            with open(stats_file, 'w') as f:
                json.dump(self.stats.to_dict(), f, indent=2)
            
            logger.info("Final collection statistics:")
            logger.info(f"  Total updates: {self.stats.total_updates}")
            logger.info(f"  Successful: {self.stats.successful_updates}")
            logger.info(f"  Failed: {self.stats.failed_updates}")
            logger.info(f"  Success rate: {self.stats.successful_updates / max(self.stats.total_updates, 1) * 100:.1f}%")
            logger.info(f"  API calls made: {self.stats.api_calls_made}")
            logger.info(f"  Uptime: {(datetime.now() - self.stats.start_time).total_seconds() / 3600:.1f} hours")
            
        except Exception as e:
            logger.error(f"Error saving final stats: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current collection statistics"""
        return self.stats.to_dict()
    
    def stop(self):
        """Stop continuous collection"""
        logger.info("Stopping continuous collection...")
        self.running = False

async def run_continuous_collection(data_dir: str, symbols: List[str], interval_seconds: int = 20):
    """Run continuous collection as standalone service"""
    from .api_utils import load_api_keys
    
    # Load API keys
    api_keys = load_api_keys()
    
    # Create collector
    collector = ContinuousDataCollector(Path(data_dir), api_keys, symbols)
    
    # Start collection
    await collector.start_continuous_collection(interval_seconds)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Data Collection Service")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--symbols", nargs="+", 
                       default=["BTC", "ETH", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
                       help="Symbols to collect")
    parser.add_argument("--interval", type=int, default=20, help="Update interval in seconds")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler('logs/continuous_collection.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run continuous collection
    asyncio.run(run_continuous_collection(args.data_dir, args.symbols, args.interval))
