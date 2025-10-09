#!/usr/bin/env python3
"""
Collect Historical Data for Backtesting
Simple script to collect historical data with the updated 30-day setting
"""

import asyncio
import logging
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append('src')

from data_ingestion.crypto_collector import CryptoDataCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    """Collect historical data for backtesting"""
    
    # Load API keys
    try:
        with open('config/api_keys.yaml', 'r') as f:
            api_keys = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("API keys file not found. Please run setup_api_keys.py first.")
        return
    
    # Create collector
    collector = CryptoDataCollector(api_keys)
    
    # Collect BTC data for 30 days
    symbols = ['BTC']
    days_back = 30
    
    logger.info(f"Collecting {days_back} days of historical data for {symbols}")
    
    try:
        results = await collector.collect_crypto_data(symbols=symbols, days_back=days_back)
        
        for symbol, df in results.items():
            logger.info(f"Collected {len(df)} data points for {symbol}")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"Days span: {(df.index[-1] - df.index[0]).days} days")
        
        logger.info("Historical data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Error collecting historical data: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
