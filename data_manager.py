#!/usr/bin/env python3
"""
Crypto Data Manager

A comprehensive tool for managing crypto data collection with different modes:
1. Auto-append recent data (detects what's missing)
2. Historical data collection (specify number of days)
3. Gap filling (repair missing data)
4. Data validation (check completeness)

Usage:
    python data_manager.py --mode auto                    # Auto-append recent data
    python data_manager.py --mode historical --days 365   # Collect 1 year of data
    python data_manager.py --mode gaps                    # Fill gaps only
    python data_manager.py --mode validate                # Check data completeness
    python data_manager.py --mode repair --symbol BTC     # Repair specific symbol
"""

import asyncio
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

from data_ingestion.crypto_collector import CryptoDataCollector
from data_ingestion.coinbase_advanced_client import CoinbaseAdvancedClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class CryptoDataManager:
    """Comprehensive crypto data management system"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.collector = CryptoDataCollector(api_keys)
        self.advanced_client = CoinbaseAdvancedClient(api_keys)
        self.db_dir = Path("data/crypto_db")
        self.db_dir.mkdir(exist_ok=True)
        
        # Supported symbols
        self.symbols = [
            'BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI'
        ]
    
    async def auto_append_recent(self, max_days: int = 7) -> Dict[str, int]:
        """Auto-append recent data, detecting what's missing automatically"""
        logger.info(f"🔄 Auto-appending recent data (max {max_days} days)")
        
        results = {}
        for symbol in self.symbols:
            try:
                # Check existing data
                existing_df = self._get_existing_data(symbol)
                if existing_df is None or existing_df.empty:
                    logger.info(f"📊 {symbol}: No existing data, collecting {max_days} days")
                    days_to_collect = max_days
                else:
                    # Calculate how many days we have
                    last_date = existing_df.index.max()
                    days_old = (datetime.now(timezone.utc) - last_date).days
                    
                    if days_old >= max_days:
                        logger.info(f"📊 {symbol}: Data is {days_old} days old, collecting {max_days} days")
                        days_to_collect = max_days
                    else:
                        days_needed = max_days - days_old
                        logger.info(f"📊 {symbol}: Data is {days_old} days old, collecting {days_needed} more days")
                        days_to_collect = days_needed
                
                # Collect data
                new_data = await self._collect_historical_data(symbol, days_to_collect)
                if new_data is not None:
                    results[symbol] = len(new_data)
                    logger.info(f"✅ {symbol}: Added {len(new_data)} new data points")
                else:
                    results[symbol] = 0
                    logger.warning(f"⚠️ {symbol}: No new data collected")
                    
            except Exception as e:
                logger.error(f"❌ {symbol}: Error - {e}")
                results[symbol] = 0
        
        return results
    
    async def collect_historical(self, days: int, symbols: List[str] = None) -> Dict[str, int]:
        """Collect historical data for specified number of days"""
        if symbols is None:
            symbols = self.symbols
            
        logger.info(f"📚 Collecting {days} days of historical data for {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            try:
                # Check if we already have enough data
                existing_df = self._get_existing_data(symbol)
                if existing_df is not None and not existing_df.empty:
                    existing_days = (existing_df.index.max() - existing_df.index.min()).days
                    if existing_days >= days:
                        logger.info(f"📊 {symbol}: Already has {existing_days} days of data")
                        results[symbol] = 0
                        continue
                
                # Collect historical data
                new_data = await self._collect_historical_data(symbol, days)
                if new_data is not None:
                    results[symbol] = len(new_data)
                    logger.info(f"✅ {symbol}: Collected {len(new_data)} data points")
                else:
                    results[symbol] = 0
                    logger.warning(f"⚠️ {symbol}: No data collected")
                    
            except Exception as e:
                logger.error(f"❌ {symbol}: Error - {e}")
                results[symbol] = 0
        
        return results
    
    async def fill_gaps(self, symbols: List[str] = None) -> Dict[str, int]:
        """Fill gaps in existing data"""
        if symbols is None:
            symbols = self.symbols
            
        logger.info(f"🔧 Filling gaps for {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            try:
                # Use the existing gap filling logic
                existing_df = self._get_existing_data(symbol)
                if existing_df is None or existing_df.empty:
                    logger.info(f"📊 {symbol}: No existing data to fill gaps")
                    results[symbol] = 0
                    continue
                
                # Find gaps
                gaps = self._find_data_gaps(symbol, existing_df.index.min(), existing_df.index.max())
                if not gaps:
                    logger.info(f"✅ {symbol}: No gaps found")
                    results[symbol] = 0
                    continue
                
                # Fill gaps
                total_filled = 0
                for gap_start, gap_end in gaps:
                    gap_data = await self._collect_data_for_period(symbol, gap_start, gap_end)
                    if gap_data is not None:
                        total_filled += len(gap_data)
                        logger.info(f"🔧 {symbol}: Filled gap {gap_start} to {gap_end} ({len(gap_data)} points)")
                
                results[symbol] = total_filled
                logger.info(f"✅ {symbol}: Filled {total_filled} data points")
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Error - {e}")
                results[symbol] = 0
        
        return results
    
    async def validate_data(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Validate data completeness and quality"""
        if symbols is None:
            symbols = self.symbols
            
        logger.info(f"🔍 Validating data for {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            try:
                df = self._get_existing_data(symbol)
                if df is None or df.empty:
                    results[symbol] = {
                        'status': 'no_data',
                        'points': 0,
                        'completeness': 0.0,
                        'date_range': None
                    }
                    continue
                
                # Calculate completeness
                date_range = (df.index.max() - df.index.min()).days
                expected_points = date_range * 24 * 60  # 1 minute intervals
                actual_points = len(df)
                completeness = (actual_points / expected_points) * 100 if expected_points > 0 else 0
                
                # Check for gaps
                gaps = self._find_data_gaps(symbol, df.index.min(), df.index.max())
                
                results[symbol] = {
                    'status': 'complete' if completeness > 80 else 'incomplete',
                    'points': actual_points,
                    'completeness': completeness,
                    'date_range': f"{df.index.min()} to {df.index.max()}",
                    'gaps': len(gaps),
                    'gap_details': gaps[:5] if gaps else []  # First 5 gaps
                }
                
                logger.info(f"📊 {symbol}: {actual_points:,} points, {completeness:.1f}% complete, {len(gaps)} gaps")
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Error - {e}")
                results[symbol] = {'status': 'error', 'error': str(e)}
        
        return results
    
    async def repair_symbol(self, symbol: str, days: int = 7) -> int:
        """Repair data for a specific symbol"""
        logger.info(f"🔧 Repairing {symbol} data ({days} days)")
        
        try:
            # Collect fresh data
            new_data = await self._collect_historical_data(symbol, days)
            if new_data is not None:
                logger.info(f"✅ {symbol}: Repaired with {len(new_data)} data points")
                return len(new_data)
            else:
                logger.warning(f"⚠️ {symbol}: No data collected")
                return 0
        except Exception as e:
            logger.error(f"❌ {symbol}: Error - {e}")
            return 0
    
    def _get_existing_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load existing historical data for a symbol"""
        try:
            db_file = self.db_dir / f"{symbol}_historical.parquet"
            if db_file.exists():
                df = pd.read_parquet(db_file)
                df.index = pd.to_datetime(df.index, utc=True)
                return df
            return None
        except Exception as e:
            logger.error(f"Error loading existing data for {symbol}: {e}")
            return None
    
    def _find_data_gaps(self, symbol: str, start_time: datetime, end_time: datetime) -> List[tuple]:
        """Find gaps in historical data"""
        try:
            existing_df = self._get_existing_data(symbol)
            if existing_df is None or existing_df.empty:
                return [(start_time, end_time)]
            
            gaps = []
            current_start = start_time
            
            # Check for gaps in the time range
            for i in range(len(existing_df.index) - 1):
                current_end = existing_df.index[i]
                next_start = existing_df.index[i + 1]
                
                # Check if there's a significant gap (more than 10 minutes but less than 24 hours)
                gap_minutes = (next_start - current_end).total_seconds() / 60
                if 10 < gap_minutes < 1440:  # Only fill gaps between 10 minutes and 24 hours
                    gap_start = current_end + timedelta(minutes=1)
                    gap_end = next_start - timedelta(minutes=1)
                    if gap_start < end_time and gap_end > start_time:
                        gaps.append((max(gap_start, start_time), min(gap_end, end_time)))
            
            # Check for gap at the end (most recent data)
            if existing_df.index.max() < end_time:
                gap_start = existing_df.index.max() + timedelta(minutes=1)
                if gap_start < end_time:
                    gaps.append((gap_start, end_time))
            
            # Check for gap at the beginning (oldest data)
            if existing_df.index.min() > start_time:
                gap_end = existing_df.index.min() - timedelta(minutes=1)
                if gap_end > start_time:
                    gaps.append((start_time, gap_end))
            
            if gaps:
                # Limit the number of gaps to prevent excessive API calls
                max_gaps = 20
                if len(gaps) > max_gaps:
                    logger.warning(f"Found {len(gaps)} gaps for {symbol}, limiting to {max_gaps} largest gaps")
                    gaps.sort(key=lambda x: (x[1] - x[0]).total_seconds(), reverse=True)
                    gaps = gaps[:max_gaps]
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error finding data gaps for {symbol}: {e}")
            return [(start_time, end_time)]
    
    async def _collect_historical_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Collect historical data for a symbol"""
        try:
            end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)
            start_time = end_time - timedelta(days=days)
            
            # Use the existing collector logic
            result = await self.collector.collect_crypto_data([symbol], days_back=days)
            if symbol in result:
                return result[symbol]
            return None
            
        except Exception as e:
            logger.error(f"Error collecting historical data for {symbol}: {e}")
            return None
    
    async def _collect_data_for_period(self, symbol: str, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Collect data for a specific time period"""
        try:
            # Calculate days needed
            days_needed = (end_time - start_time).days + 1
            
            # Use the existing collector logic
            result = await self.collector.collect_crypto_data([symbol], days_back=days_needed)
            if symbol in result:
                # Filter to the specific time period
                df = result[symbol]
                filtered_df = df[(df.index >= start_time) & (df.index <= end_time)]
                return filtered_df
            return None
            
        except Exception as e:
            logger.error(f"Error collecting data for period {symbol}: {e}")
            return None

async def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Crypto Data Manager')
    parser.add_argument('--mode', choices=['auto', 'historical', 'gaps', 'validate', 'repair'], 
                       required=True, help='Operation mode')
    parser.add_argument('--days', type=int, default=7, help='Number of days to collect')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to process')
    parser.add_argument('--symbol', help='Single symbol for repair mode')
    
    args = parser.parse_args()
    
    # Initialize data manager
    manager = CryptoDataManager()
    
    if args.mode == 'auto':
        logger.info("🚀 Starting auto-append mode")
        results = await manager.auto_append_recent(args.days)
        logger.info(f"✅ Auto-append completed: {sum(results.values())} total new points")
        
    elif args.mode == 'historical':
        logger.info(f"📚 Starting historical collection mode ({args.days} days)")
        results = await manager.collect_historical(args.days, args.symbols)
        logger.info(f"✅ Historical collection completed: {sum(results.values())} total new points")
        
    elif args.mode == 'gaps':
        logger.info("🔧 Starting gap filling mode")
        results = await manager.fill_gaps(args.symbols)
        logger.info(f"✅ Gap filling completed: {sum(results.values())} total new points")
        
    elif args.mode == 'validate':
        logger.info("🔍 Starting data validation mode")
        results = await manager.validate_data(args.symbols)
        
        # Print validation results
        print("\n" + "="*60)
        print("DATA VALIDATION RESULTS")
        print("="*60)
        for symbol, data in results.items():
            if data['status'] == 'error':
                print(f"❌ {symbol}: ERROR - {data['error']}")
            elif data['status'] == 'no_data':
                print(f"📊 {symbol}: NO DATA")
            else:
                status_icon = "✅" if data['status'] == 'complete' else "⚠️"
                print(f"{status_icon} {symbol}: {data['points']:,} points, {data['completeness']:.1f}% complete, {data['gaps']} gaps")
                if data['gap_details']:
                    print(f"   Gap details: {data['gap_details']}")
        print("="*60)
        
    elif args.mode == 'repair':
        if not args.symbol:
            logger.error("❌ Repair mode requires --symbol argument")
            return
        logger.info(f"🔧 Starting repair mode for {args.symbol}")
        result = await manager.repair_symbol(args.symbol, args.days)
        logger.info(f"✅ Repair completed: {result} new points")

if __name__ == "__main__":
    asyncio.run(main())
