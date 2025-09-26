"""
BTC-Only Continuous Data Collection
Simple script to collect only BTC data continuously
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_ingestion.continuous_collector import ContinuousDataCollector
from data_ingestion.crypto_collector import CryptoDataCollector
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/btc_collection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Start BTC-only continuous collection"""
    
    # Load API keys
    try:
        with open('config/api_keys.yaml', 'r') as f:
            api_keys = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("API keys file not found. Please run setup_api_keys.py first.")
        return
    
    # BTC-only configuration
    symbols = ['BTC']
    data_dir = Path('data')
    
    logger.info("Starting BTC-only continuous data collection...")
    logger.info(f"Target ticker: {symbols[0]}")
    logger.info("Data source: Coinbase Advanced API")
    logger.info("Collection interval: 20 seconds")
    logger.info("Press Ctrl+C to stop")
    
    # Create continuous collector
    collector = ContinuousDataCollector(data_dir, api_keys, symbols)
    
    try:
        # Start continuous collection
        await collector.start_continuous_collection(interval_seconds=20)
    except KeyboardInterrupt:
        logger.info("Collection stopped by user")
    except Exception as e:
        logger.error(f"Collection error: {e}")
    finally:
        # Print final stats
        stats = collector.get_stats()
        logger.info("Final Collection Stats:")
        logger.info(f"  Total updates: {stats['total_updates']}")
        logger.info(f"  Successful: {stats['successful_updates']}")
        logger.info(f"  Failed: {stats['failed_updates']}")
        logger.info(f"  Success rate: {stats['success_rate']:.1f}%")
        logger.info(f"  Uptime: {stats['uptime_hours']:.1f} hours")

if __name__ == "__main__":
    asyncio.run(main())
