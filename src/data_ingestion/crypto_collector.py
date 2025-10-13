#!/usr/bin/env python3
"""
Crypto Data Collector using Coinbase Advanced API

This module provides crypto data collection using Coinbase Advanced Trade API
for high-quality historical data with minute-level granularity.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import time

from .coinbase_advanced_client import CoinbaseAdvancedClient
from .coinbase_rest_client import create_coinbase_rest_client

logger = logging.getLogger(__name__)

class CryptoDataCollector:
    """Crypto data collector using Coinbase Advanced API"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        # Initialize both clients
        self.advanced_client = CoinbaseAdvancedClient(api_keys)
        self.rest_client = create_coinbase_rest_client()
        
        self.crypto_symbols = [
            'BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI'
        ]
        
        # Rate limiting for Coinbase Advanced API
        self.rate_limit_delay = 0.1  # 100ms between requests (10 req/sec max)
        self.last_request_time = 0
        
        # Pagination settings for 350 candle limit
        self.candles_per_request = 350
        self.minute_granularity = 60  # 1 minute candles
        
        # Database settings for historical data
        self.db_dir = Path("data/crypto_db")
        self.db_dir.mkdir(exist_ok=True)
        
        # Cache for technical indicators (separate from price data)
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    async def _rate_limit_wait(self):
        """Wait to respect rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _get_existing_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load existing historical data for a symbol"""
        try:
            db_file = self.db_dir / f"{symbol}_historical.parquet"
            if not db_file.exists():
                logger.info(f"No existing data found for {symbol}")
                return None
            
            df = pd.read_parquet(db_file)
            if df.empty:
                return None
            
            # Ensure proper datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    df = df.set_index('timestamp')
                else:
                    df.index = pd.to_datetime(df.index, utc=True)
            
            # Ensure timezone-aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            logger.info(f"Loaded existing data for {symbol}: {len(df)} points from {df.index.min()} to {df.index.max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading existing data for {symbol}: {e}")
            return None
    
    def _save_historical_data(self, symbol: str, df: pd.DataFrame):
        """Save historical data to database (append-only)"""
        try:
            db_file = self.db_dir / f"{symbol}_historical.parquet"
            
            if db_file.exists():
                # Load existing data and merge
                existing_df = pd.read_parquet(db_file)
                if not existing_df.empty:
                    # Ensure proper datetime index for existing data
                    if not isinstance(existing_df.index, pd.DatetimeIndex):
                        if 'timestamp' in existing_df.columns:
                            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], utc=True)
                            existing_df = existing_df.set_index('timestamp')
                        else:
                            existing_df.index = pd.to_datetime(existing_df.index, utc=True)
                    
                    # Ensure timezone-aware
                    if existing_df.index.tz is None:
                        existing_df.index = existing_df.index.tz_localize('UTC')
                    
                    # Combine and remove duplicates
                    combined_df = pd.concat([existing_df, df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df = combined_df.sort_index()
                    
                    logger.info(f"Merged data for {symbol}: {len(existing_df)} existing + {len(df)} new = {len(combined_df)} total")
                    df = combined_df
            
            # Save to database
            df.to_parquet(db_file)
            logger.info(f"Saved {len(df)} data points for {symbol} to historical database")
            
        except Exception as e:
            logger.error(f"Error saving historical data for {symbol}: {e}")
    
    def _save_gap_candles(self, symbol: str, candles: list):
        """Save candles incrementally during collection"""
        try:
            if not candles:
                return
            
            # Convert Candle objects to proper data structure
            data = []
            for candle in candles:
                # Handle different candle object types
                if hasattr(candle, 'to_dict'):
                    candle_dict = candle.to_dict()
                elif hasattr(candle, '__dict__'):
                    candle_dict = candle.__dict__
                else:
                    # If it's already a dict, use it directly
                    candle_dict = candle
                
                # Extract values safely
                data.append({
                    'timestamp': candle_dict.get('start', candle_dict.get('timestamp', 0)),
                    'open': float(candle_dict.get('open', 0)),
                    'high': float(candle_dict.get('high', 0)),
                    'low': float(candle_dict.get('low', 0)),
                    'close': float(candle_dict.get('close', 0)),
                    'volume': float(candle_dict.get('volume', 0))
                })
            
            # Convert to DataFrame
            gap_df = pd.DataFrame(data)
            
            # Convert timestamp to datetime (it's currently a Unix timestamp string)
            gap_df['timestamp'] = pd.to_datetime(gap_df['timestamp'], unit='s', utc=True)
            gap_df = gap_df.set_index('timestamp')
            gap_df = gap_df.sort_index()
            
            # Remove duplicates
            gap_df = gap_df[~gap_df.index.duplicated(keep='first')]
            
            # Save incrementally
            self._save_historical_data(symbol, gap_df)
            logger.info(f"✅ Incremental save: {len(gap_df)} candles for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving gap candles for {symbol}: {e}")
    
    def _find_data_gaps(self, symbol: str, start_time: datetime, end_time: datetime) -> List[Tuple[datetime, datetime]]:
        """Find gaps in historical data that need to be filled"""
        try:
            existing_df = self._get_existing_data(symbol)
            if existing_df is None or existing_df.empty:
                return [(start_time, end_time)]
            
            gaps = []
            
            # First check: Does existing data already cover the entire requested range?
            existing_start = existing_df.index.min()
            existing_end = existing_df.index.max()
            
            # If existing data covers the entire requested range, no gaps needed
            if existing_start <= start_time and existing_end >= end_time:
                logger.info(f"[COVERAGE CHECK] {symbol} data already covers entire requested range ({start_time} to {end_time})")
                return []
            
            # Check for gap at the beginning (oldest data)
            if existing_start > start_time:
                gap_end = existing_start - timedelta(minutes=1)
                if gap_end > start_time:
                    gaps.append((start_time, gap_end))
                    logger.info(f"[GAP] {symbol} missing data from {start_time} to {gap_end}")
            
            # Check for gaps in the middle of the time range
            for i in range(len(existing_df.index) - 1):
                current_end = existing_df.index[i]
                next_start = existing_df.index[i + 1]
                
                # Check if there's a significant gap (more than 10 minutes but less than 24 hours)
                # Small gaps are normal in crypto markets during low-volume periods
                # Very large gaps might be due to market closures or data issues
                gap_minutes = (next_start - current_end).total_seconds() / 60
                if 10 < gap_minutes < 1440:  # Only fill gaps between 10 minutes and 24 hours
                    gap_start = current_end + timedelta(minutes=1)
                    gap_end = next_start - timedelta(minutes=1)
                    if gap_start < end_time and gap_end > start_time:
                        gaps.append((max(gap_start, start_time), min(gap_end, end_time)))
                        logger.info(f"[GAP] {symbol} missing data from {gap_start} to {gap_end}")
            
            # Check for gap at the end (most recent data)
            # Only fetch recent data if the latest data is more than 5 minutes old
            latest_data_time = existing_df.index.max()
            current_time = datetime.now(tz=latest_data_time.tzinfo if latest_data_time.tzinfo else None)
            minutes_behind = (current_time - latest_data_time).total_seconds() / 60
            
            if existing_end < end_time:
                gap_start = existing_end + timedelta(minutes=1)
                if gap_start < end_time:
                    gaps.append((gap_start, end_time))
                    logger.info(f"[GAP] {symbol} missing data from {gap_start} to {end_time}")
            elif minutes_behind > 5:
                # Force a recent data fetch if data is stale
                recent_start = current_time - timedelta(minutes=10)  # Get last 10 minutes
                gaps.append((recent_start, current_time))
                logger.info(f"[STALE DATA] {symbol} data is {minutes_behind:.1f} minutes old, fetching recent data")
            
            if gaps:
                # Limit the number of gaps to prevent excessive API calls
                max_gaps = 20  # Maximum 20 gaps per symbol
                if len(gaps) > max_gaps:
                    logger.warning(f"Found {len(gaps)} gaps for {symbol}, limiting to {max_gaps} largest gaps")
                    # Sort by gap size (largest first) and take the first max_gaps
                    gaps.sort(key=lambda x: (x[1] - x[0]).total_seconds(), reverse=True)
                    gaps = gaps[:max_gaps]
                
                logger.info(f"Found {len(gaps)} data gaps for {symbol}")
                for i, (gap_start, gap_end) in enumerate(gaps):
                    gap_hours = (gap_end - gap_start).total_seconds() / 3600
                    logger.info(f"  Gap {i+1}: {gap_start} to {gap_end} ({gap_hours:.1f} hours)")
            else:
                logger.info(f"No data gaps found for {symbol}")
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error finding data gaps for {symbol}: {e}")
            return [(start_time, end_time)]
    
    async def collect_crypto_prices(self, symbols: List[str] = None) -> Dict[str, float]:
        """Collect current crypto prices"""
        if symbols is None:
            symbols = self.crypto_symbols
        
        logger.info(f"Collecting crypto prices for {len(symbols)} symbols")
        
        prices = await self.rest_client.get_multiple_prices(symbols)
        
        logger.info(f"Collected prices for {len(prices)} symbols")
        return prices
    
    async def collect_crypto_data(self, symbols: List[str] = None, days_back: int = 7) -> Dict[str, pd.DataFrame]:
        """
        Collect crypto data using incremental database approach - only fetch missing data
        
        Args:
            symbols: List of crypto symbols to collect
            days_back: Number of days to collect (default 7 for efficiency)
        """
        if symbols is None:
            symbols = self.crypto_symbols
        
        logger.info(f"Collecting {days_back} days of minute-level crypto data for {len(symbols)} symbols")
        
        results = {}
        total_requests = 0
        no_gaps_count = 0
        
        # Calculate time range
        from datetime import timezone
        end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)
        start_time = end_time - timedelta(days=days_back)
        
        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"Processing {symbol} ({i}/{len(symbols)})")
                
                # Load existing historical data
                existing_data = self._get_existing_data(symbol)
                
                # Find gaps that need to be filled
                gaps = self._find_data_gaps(symbol, start_time, end_time)
                
                if not gaps:
                    # No gaps found, no new data to return
                    if existing_data is not None:
                        no_gaps_count += 1
                        logger.info(f"[NO GAPS] {symbol} data is up to date, no new data needed")
                        # For bulk collection, still return existing data
                        filtered_data = existing_data[(existing_data.index >= start_time) & (existing_data.index <= end_time)]
                        results[symbol] = filtered_data
                        continue
                    else:
                        # No existing data, need to collect everything
                        gaps = [(start_time, end_time)]
                
                # Collect data for each gap
                all_new_candles = []
                
                for gap_idx, (gap_start, gap_end) in enumerate(gaps):
                    logger.info(f"Filling gap {gap_idx+1}/{len(gaps)} for {symbol}: {gap_start} to {gap_end}")
                    
                    # Calculate requests needed for this gap
                    gap_minutes = (gap_end - gap_start).total_seconds() / 60
                    gap_requests = int((gap_minutes + self.candles_per_request - 1) // self.candles_per_request)
                    
                    logger.info(f"Gap {gap_idx+1}: {gap_minutes:.1f} minutes, {gap_requests} requests needed")
                    
                    # Collect data for this gap
                    current_end = gap_end
                    gap_candles = []  # Store candles for this gap only
                    requests_since_save = 0
                    save_interval = 100  # Save every 100 requests
                    saved_candles_count = 0  # Track how many candles we've already saved
                    
                    for request_count in range(gap_requests):
                        await self._rate_limit_wait()
                        
                        # Each call goes back exactly 350 minutes
                        current_start = current_end - timedelta(minutes=self.candles_per_request)
                        
                        # Don't go before gap start
                        if current_start < gap_start:
                            current_start = gap_start
                        
                        logger.info(f"  Request {request_count+1}/{gap_requests}: {current_start} to {current_end}")
                        
                        # Convert to Unix timestamps as strings
                        start_unix = str(int(current_start.timestamp()))
                        end_unix = str(int(current_end.timestamp()))
                        
                        # Get candles for this specific time range
                        try:
                            from coinbase.rest import RESTClient
                            import os
                            from dotenv import load_dotenv
                            load_dotenv()
                            
                            api_key = os.getenv('COINBASE_API_KEY')
                            api_secret = os.getenv('COINBASE_API_SECRET')
                            
                            if api_key and api_secret:
                                client = RESTClient(api_key=api_key, api_secret=api_secret)
                                response = client.get_public_candles(
                                    product_id=f"{symbol}-USD",
                                    start=start_unix,
                                    end=end_unix,
                                    granularity="ONE_MINUTE"
                                )
                                candles = response.candles if hasattr(response, 'candles') else []
                            else:
                                candles = []
                        except Exception as e:
                            error_msg = str(e)
                            if "product_id" in error_msg.lower() or "not found" in error_msg.lower():
                                logger.warning(f"Symbol {symbol} not supported by Coinbase, skipping...")
                                break  # Skip this symbol entirely
                            else:
                                logger.error(f"API call failed for {symbol}: {e}")
                                candles = []
                        
                        if candles:
                            gap_candles.extend(candles)
                            logger.info(f"  Got {len(candles)} candles, gap total so far: {len(gap_candles)}")
                        else:
                            logger.warning(f"  No candles returned for request {request_count+1}")
                        
                        total_requests += 1
                        requests_since_save += 1
                        
                        # Save incrementally every 100 requests
                        if requests_since_save >= save_interval and gap_candles:
                            # Only save new candles since last save
                            new_candles = gap_candles[saved_candles_count:]
                            if new_candles:
                                logger.info(f"💾 Saving progress after {requests_since_save} requests ({len(new_candles)} new candles)...")
                                self._save_gap_candles(symbol, new_candles)
                                saved_candles_count = len(gap_candles)
                            requests_since_save = 0
                        
                        # Move to next time window
                        current_end = current_start
                        
                        # Stop if we've reached the gap start
                        if current_end <= gap_start:
                            break
                    
                    # Save any remaining candles from this gap
                    if gap_candles:
                        # Only save candles that haven't been saved yet
                        remaining_candles = gap_candles[saved_candles_count:]
                        if remaining_candles:
                            logger.info(f"💾 Saving final gap {gap_idx+1} data for {symbol}: {len(remaining_candles)} remaining candles")
                            self._save_gap_candles(symbol, remaining_candles)
                        
                        # Add to all candles for final processing
                        all_new_candles.extend(gap_candles)
                
                if all_new_candles:
                    # Data has already been saved incrementally after each gap
                    # Just get the complete dataset for the requested time range
                    complete_data = self._get_existing_data(symbol)
                    if complete_data is not None:
                        # Filter to requested time range
                        filtered_data = complete_data[(complete_data.index >= start_time) & (complete_data.index <= end_time)]
                        results[symbol] = filtered_data
                        
                        # Validate data completeness
                        expected_minutes = days_back * 24 * 60
                        actual_minutes = len(filtered_data)
                        completeness = (actual_minutes / expected_minutes) * 100
                        
                        logger.info(f"Data completeness for {symbol}: {actual_minutes}/{expected_minutes} minutes ({completeness:.1f}%)")
                        
                        # Crypto markets don't trade every single minute - 80%+ is good
                        if completeness < 80:
                            logger.warning(f"[WARNING] {symbol} data is only {completeness:.1f}% complete - may have significant gaps")
                        else:
                            logger.info(f"[SUCCESS] {symbol} data is {completeness:.1f}% complete")
                        
                        logger.info(f"[SUCCESS] Collected {len(all_new_candles)} new data points for {symbol}, total: {len(filtered_data)} points")
                    else:
                        logger.warning(f"[WARNING] Failed to load complete data for {symbol}")
                else:
                    logger.warning(f"[WARNING] No new data collected for {symbol}")
                    # For bulk collection, still return existing data if available
                    if existing_data is not None:
                        filtered_data = existing_data[(existing_data.index >= start_time) & (existing_data.index <= end_time)]
                        results[symbol] = filtered_data
                        logger.info(f"[FALLBACK] Returning existing data for {symbol}: {len(filtered_data)} points")
                    
            except Exception as e:
                logger.error(f"[ERROR] Error collecting data for {symbol}: {e}")
                continue
        
        logger.info(f"Crypto data collection complete: {len(results)} symbols with data, {no_gaps_count} no gaps, {total_requests} new requests")
        return results
    
    async def save_crypto_data(self, data: Dict[str, pd.DataFrame], data_dir: Path):
        """Save crypto data to parquet files (consolidated to crypto_db directory)"""
        # Use the same directory as historical data for consistency
        crypto_db_dir = data_dir / "crypto_db"
        crypto_db_dir.mkdir(exist_ok=True)
        
        saved_count = 0
        for symbol, df in data.items():
            if not df.empty:
                file_path = crypto_db_dir / f"{symbol}_historical.parquet"
                df.to_parquet(file_path)
                logger.info(f"Saved {symbol} data to {file_path} ({len(df)} points)")
                saved_count += 1
            else:
                logger.warning(f"Skipping empty data for {symbol}")
        
        logger.info(f"Successfully saved data for {saved_count}/{len(data)} symbols")
    
    async def get_crypto_summary(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Get a summary of crypto data"""
        if symbols is None:
            symbols = self.crypto_symbols
        
        prices = await self.collect_crypto_prices(symbols)
        
        if not prices:
            return {"error": "No crypto data available"}
        
        # Calculate basic statistics
        price_values = list(prices.values())
        
        summary = {
            "total_symbols": len(prices),
            "price_range": {
                "min": min(price_values),
                "max": max(price_values),
                "avg": np.mean(price_values)
            },
            "prices": prices,
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    async def repair_missing_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Repair missing data for a specific symbol and time range"""
        logger.info(f"Repairing missing data for {symbol} from {start_time} to {end_time}")
        
        try:
            # Convert to Unix timestamps as strings
            start_unix = str(int(start_time.timestamp()))
            end_unix = str(int(end_time.timestamp()))
            
            # Make direct API call for the specific time range
            from coinbase.rest import RESTClient
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv('COINBASE_API_KEY')
            api_secret = os.getenv('COINBASE_API_SECRET')
            
            if not api_key or not api_secret:
                logger.error("API keys not found for repair")
                return pd.DataFrame()
            
            client = RESTClient(api_key=api_key, api_secret=api_secret)
            response = client.get_public_candles(
                product_id=f"{symbol}-USD",
                start=start_unix,
                end=end_unix,
                granularity="ONE_MINUTE"
            )
            
            candles = response.candles if hasattr(response, 'candles') else []
            
            if not candles:
                logger.warning(f"No data returned for {symbol} repair")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for candle in candles:
                data.append({
                    'timestamp': pd.to_datetime(candle.start, unit='s', utc=True),
                    'open': float(candle.open),
                    'high': float(candle.high),
                    'low': float(candle.low),
                    'close': float(candle.close),
                    'volume': float(candle.volume)
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                logger.info(f"Repaired {len(df)} data points for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error repairing data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def check_data_completeness(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """Check data completeness and identify missing periods"""
        logger.info(f"Checking data completeness for {symbol}")
        
        try:
            # Load existing data
            data_file = Path("data/raw") / f"prices_{symbol}.parquet"
            if not data_file.exists():
                return {"status": "no_data", "missing_periods": []}
            
            df = pd.read_parquet(data_file)
            if df.empty:
                return {"status": "empty_data", "missing_periods": []}
            
            # Ensure timestamp index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df.set_index('timestamp', inplace=True)
            
            # Calculate expected time range
            end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)
            start_time = end_time - timedelta(days=days_back)
            
            # Create expected minute range
            expected_minutes = pd.date_range(start=start_time, end=end_time, freq='1min')
            
            # Find missing minutes
            missing_minutes = expected_minutes.difference(df.index)
            
            if len(missing_minutes) == 0:
                return {"status": "complete", "missing_periods": []}
            
            # Group consecutive missing periods
            missing_periods = []
            current_start = None
            current_end = None
            
            for minute in missing_minutes:
                if current_start is None:
                    current_start = minute
                    current_end = minute
                elif minute == current_end + timedelta(minutes=1):
                    current_end = minute
                else:
                    missing_periods.append((current_start, current_end))
                    current_start = minute
                    current_end = minute
            
            if current_start is not None:
                missing_periods.append((current_start, current_end))
            
            return {
                "status": "incomplete",
                "total_expected": len(expected_minutes),
                "total_actual": len(df),
                "missing_count": len(missing_minutes),
                "missing_periods": missing_periods,
                "completeness_pct": (len(df) / len(expected_minutes)) * 100
            }
            
        except Exception as e:
            logger.error(f"Error checking completeness for {symbol}: {e}")
            return {"status": "error", "error": str(e)}

# Example usage
async def main():
    """Example usage of crypto data collector"""
    collector = CryptoDataCollector()
    
    # Test with a few symbols
    symbols = ['BTC', 'ETH', 'ADA']
    
    # Collect data
    data = await collector.collect_crypto_data(symbols)
    
    print(f"Collected data for {len(data)} symbols:")
    for symbol, df in data.items():
        print(f"  {symbol}: {len(df)} data points")
        if not df.empty:
            print(f"    Current price: ${df['close'].iloc[-1]:,.2f}")
    
    # Get summary
    summary = await collector.get_crypto_summary(symbols)
    print(f"\nSummary: {summary['total_symbols']} symbols")
    print(f"Price range: ${summary['price_range']['min']:,.2f} - ${summary['price_range']['max']:,.2f}")

if __name__ == "__main__":
    asyncio.run(main())

