#!/usr/bin/env python3
"""
Initialize Historical Data
Collects initial historical data to ensure sufficient data for technical indicators
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_ingestion.crypto_collector import CryptoDataCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/initialize_historical.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def initialize_historical_data():
    """Initialize historical data for all symbols"""
    print("🔄 Initializing Historical Data")
    print("=" * 50)
    print("This will collect historical data to ensure sufficient data for technical indicators")
    print("=" * 50)
    
    # Initialize collector
    collector = CryptoDataCollector()
    
    # Symbols to initialize
    symbols = ['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI']
    
    print(f"Collecting historical data for: {', '.join(symbols)}")
    print("This may take a few minutes...")
    
    try:
        # Collect 7 days of historical data
        results = await collector.collect_crypto_data(symbols, days_back=7)
        
        print("\n📊 Collection Results:")
        print("=" * 30)
        
        for symbol in symbols:
            if symbol in results:
                df = results[symbol]
                print(f"✅ {symbol}: {len(df)} data points")
                if len(df) >= 50:
                    print(f"   ✅ Sufficient for technical indicators")
                else:
                    print(f"   ⚠️  May need more data for technical indicators")
            else:
                print(f"❌ {symbol}: No data collected")
        
        print("\n🎉 Historical data initialization complete!")
        print("You can now run the WebSocket collection system.")
        
    except Exception as e:
        logger.error(f"Error initializing historical data: {e}")
        print(f"❌ Error: {e}")

def main():
    """Main function"""
    asyncio.run(initialize_historical_data())

if __name__ == "__main__":
    main()
