#!/usr/bin/env python3
"""
Crypto Data Repair Script

This script checks for missing data in crypto parquet files and repairs any gaps
by making targeted API calls to fill in missing minutes.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
from data_ingestion.crypto_collector import CryptoDataCollector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def repair_crypto_data(symbols: list = None, days_back: int = 7, min_completeness: float = 95.0):
    """Repair missing crypto data for specified symbols"""
    
    if symbols is None:
        # Get symbols from config
        import yaml
        with open('config/pipeline_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        symbols = config['assets']['crypto']
    
    logger.info(f"Starting crypto data repair for {len(symbols)} symbols")
    
    collector = CryptoDataCollector()
    
    for symbol in symbols:
        logger.info(f"\n🔍 Checking {symbol}...")
        
        # Check data completeness
        completeness = await collector.check_data_completeness(symbol, days_back)
        
        if completeness['status'] == 'no_data':
            logger.warning(f"❌ {symbol}: No data file found")
            continue
        elif completeness['status'] == 'error':
            logger.error(f"❌ {symbol}: Error checking completeness - {completeness.get('error', 'Unknown error')}")
            continue
        elif completeness['status'] == 'complete':
            logger.info(f"✅ {symbol}: Data is complete ({completeness.get('completeness_pct', 100):.1f}%)")
            continue
        elif completeness['status'] == 'incomplete':
            completeness_pct = completeness.get('completeness_pct', 0)
            missing_count = completeness.get('missing_count', 0)
            missing_periods = completeness.get('missing_periods', [])
            
            logger.info(f"⚠️  {symbol}: Incomplete data ({completeness_pct:.1f}% complete, {missing_count} missing minutes)")
            
            if completeness_pct >= min_completeness:
                logger.info(f"✅ {symbol}: Above minimum completeness threshold ({min_completeness}%), skipping repair")
                continue
            
            # Repair missing periods
            logger.info(f"🔧 Repairing {symbol}...")
            repaired_count = 0
            
            for start_time, end_time in missing_periods:
                logger.info(f"  Repairing period: {start_time} to {end_time}")
                
                # Get repair data
                repair_df = await collector.repair_missing_data(symbol, start_time, end_time)
                
                if not repair_df.empty:
                    # Load existing data
                    data_file = Path("data/raw") / f"prices_{symbol}.parquet"
                    existing_df = pd.read_parquet(data_file)
                    
                    # Ensure timestamp index
                    if 'timestamp' in existing_df.columns:
                        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], utc=True)
                        existing_df.set_index('timestamp', inplace=True)
                    
                    # Combine data
                    combined_df = pd.concat([existing_df, repair_df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]  # Remove duplicates
                    combined_df.sort_index(inplace=True)
                    
                    # Save repaired data
                    combined_df.to_parquet(data_file)
                    repaired_count += len(repair_df)
                    
                    logger.info(f"  ✅ Repaired {len(repair_df)} minutes")
                else:
                    logger.warning(f"  ⚠️  No data returned for repair period")
            
            if repaired_count > 0:
                logger.info(f"✅ {symbol}: Repaired {repaired_count} total minutes")
                
                # Verify repair
                new_completeness = await collector.check_data_completeness(symbol, days_back)
                new_pct = new_completeness.get('completeness_pct', 0)
                logger.info(f"📊 {symbol}: New completeness: {new_pct:.1f}%")
            else:
                logger.warning(f"⚠️  {symbol}: No data could be repaired")

async def main():
    """Main repair function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Repair missing crypto data')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to repair (default: all from config)')
    parser.add_argument('--days', type=int, default=7, help='Days of data to check (default: 7)')
    parser.add_argument('--min-completeness', type=float, default=95.0, help='Minimum completeness threshold (default: 95.0%)')
    
    args = parser.parse_args()
    
    await repair_crypto_data(
        symbols=args.symbols,
        days_back=args.days,
        min_completeness=args.min_completeness
    )

if __name__ == "__main__":
    asyncio.run(main())
