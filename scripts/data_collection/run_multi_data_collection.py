"""
Multi-Symbol Data Collection Runner
Interactive script to collect data for multiple crypto symbols
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_ingestion.continuous_collector import ContinuousDataCollector
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/multi_data_collection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run multi-symbol data collection with user input"""
    
    # Available symbols
    available_symbols = ['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI']
    
    print("🚀 Multi-Symbol Crypto Data Collection")
    print("=" * 50)
    print(f"Available symbols: {', '.join(available_symbols)}")
    print()
    
    # Get user input for symbols
    symbols_input = input("Enter symbols to collect (space-separated, default: all): ").strip()
    if not symbols_input:
        symbols = available_symbols  # Default to all symbols
    else:
        symbols = [s.upper().strip() for s in symbols_input.split()]
    
    # Validate symbols
    invalid_symbols = [s for s in symbols if s not in available_symbols]
    if invalid_symbols:
        print(f"Invalid symbols: {', '.join(invalid_symbols)}")
        print(f"Available: {', '.join(available_symbols)}")
        return
    
    # Get collection interval
    interval_input = input("Collection interval in seconds (default 20): ").strip()
    interval = int(interval_input) if interval_input.isdigit() else 20
    
    # Get data directory
    data_dir_input = input("Data directory (default 'data'): ").strip()
    data_dir = Path(data_dir_input) if data_dir_input else Path('data')
    
    print(f"\nCollecting data for: {', '.join(symbols)}")
    print(f"Interval: {interval} seconds")
    print(f"Data directory: {data_dir}")
    print("Press Ctrl+C to stop")
    print()
    
    # Run the collection
    asyncio.run(run_collection(symbols, data_dir, interval))

async def run_collection(symbols, data_dir, interval):
    """Run the actual data collection"""
    
    # Load API keys
    try:
        with open('config/api_keys.yaml', 'r') as f:
            api_keys = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("API keys file not found. Please run setup_api_keys.py first.")
        return
    
    logger.info(f"Starting multi-symbol data collection for: {', '.join(symbols)}")
    logger.info(f"Data source: Coinbase Advanced API")
    logger.info(f"Collection interval: {interval} seconds")
    
    # Create continuous collector
    collector = ContinuousDataCollector(data_dir, api_keys, symbols)
    
    try:
        # Start continuous collection
        await collector.start_continuous_collection(interval_seconds=interval)
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
    main()
