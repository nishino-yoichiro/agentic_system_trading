#!/usr/bin/env python3
"""
Unified Data Manager
Combines historical data fetching with real-time price updates
Provides a single source of truth for all price data
"""

import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import asyncio

from .crypto_collector import CryptoDataCollector
from .realtime_price_collector import RealtimePriceCollector

logger = logging.getLogger(__name__)

class UnifiedDataManager:
    """Manages both historical and real-time data for a unified data source"""
    
    def __init__(self, api_keys: Dict[str, str], data_dir: Path = None):
        self.api_keys = api_keys
        self.data_dir = data_dir or Path("data/crypto_db")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize collectors
        self.historical_collector = CryptoDataCollector(api_keys)
        self.realtime_collector = RealtimePriceCollector()
        
        # Cache for recent data to avoid repeated API calls
        self._data_cache = {}
        self._cache_expiry = {}
        self.cache_duration_minutes = 1  # Cache real-time data for 1 minute
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data for analysis and backtesting"""
        try:
            # Load from existing historical database
            db_file = self.data_dir / f"{symbol}_historical.parquet"
            if db_file.exists():
                df = pd.read_parquet(db_file)
                df.index = pd.to_datetime(df.index)
                
                # Filter to last N days
                cutoff_date = datetime.now() - timedelta(days=days)
                cutoff_date = pd.Timestamp(cutoff_date, tz='UTC')
                df = df[df.index >= cutoff_date]
                
                logger.info(f"Loaded {len(df)} historical data points for {symbol}")
                return df
            else:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_price(self, symbol: str) -> Optional[Dict]:
        """Get current real-time price"""
        try:
            # Check cache first
            cache_key = f"{symbol}_realtime"
            now = datetime.now()
            
            if (cache_key in self._data_cache and 
                cache_key in self._cache_expiry and 
                now < self._cache_expiry[cache_key]):
                return self._data_cache[cache_key]
            
            # Fetch fresh data
            currency_pair = f"{symbol}-USD"
            price_data = self.realtime_collector.get_spot_price(currency_pair)
            
            if price_data:
                # Cache the result
                self._data_cache[cache_key] = price_data
                self._cache_expiry[cache_key] = now + timedelta(minutes=self.cache_duration_minutes)
                
                logger.debug(f"Got real-time price for {symbol}: ${price_data['price']:.2f}")
                return price_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting real-time price for {symbol}: {e}")
            return None
    
    def get_combined_data(self, symbol: str, days: int = 30, include_realtime: bool = True) -> pd.DataFrame:
        """Get combined historical + real-time data for live trading"""
        try:
            # Get historical data (with proper OHLCV)
            historical_df = self.get_historical_data(symbol, days)
            
            if not include_realtime or historical_df.empty:
                return historical_df
            
            # Get current real-time price
            realtime_data = self.get_realtime_price(symbol)
            if not realtime_data:
                logger.warning(f"Could not get real-time data for {symbol}, returning historical only")
                return historical_df
            
            # For live trading, we'll use historical data for technical analysis
            # but update the most recent close price with real-time data
            current_time = realtime_data['timestamp']
            current_price = realtime_data['price']
            
            if not historical_df.empty:
                latest_time = historical_df.index.max()
                time_diff = (current_time - latest_time).total_seconds()
                
                # If real-time data is within the current minute, update the close price
                if time_diff < 60:  # Less than 1 minute difference
                    # Update the close price with real-time data
                    historical_df.loc[latest_time, 'close'] = current_price
                    # Update high/low if needed
                    historical_df.loc[latest_time, 'high'] = max(historical_df.loc[latest_time, 'high'], current_price)
                    historical_df.loc[latest_time, 'low'] = min(historical_df.loc[latest_time, 'low'], current_price)
                    # Keep the original volume (don't overwrite with 0)
                    
                    logger.info(f"Updated {symbol} close price to ${current_price:.2f} (realtime)")
                else:
                    # Real-time data is for a new minute, but we don't have volume
                    # So we'll just use historical data for technical analysis
                    logger.info(f"Real-time data for {symbol} is {time_diff:.0f}s ahead, using historical data for analysis")
            
            logger.info(f"Combined data for {symbol}: {len(historical_df)} points, latest close: ${historical_df['close'].iloc[-1]:.2f}")
            return historical_df
            
        except Exception as e:
            logger.error(f"Error getting combined data for {symbol}: {e}")
            return historical_df if 'historical_df' in locals() else pd.DataFrame()
    
    async def update_historical_data(self, symbols: List[str], days_back: int = 7) -> Dict[str, int]:
        """Update historical data for symbols (run periodically)"""
        try:
            logger.info(f"Updating historical data for {symbols}")
            results = await self.historical_collector.collect_crypto_data(
                symbols=symbols,
                days_back=days_back
            )
            return results or {}
        except Exception as e:
            logger.error(f"Error updating historical data: {e}")
            return {}
    
    def update_realtime_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Update real-time data for symbols (run frequently)"""
        try:
            logger.info(f"Updating real-time data for {symbols}")
            results = self.realtime_collector.collect_and_save_prices(
                symbols=symbols,
                data_dir=self.data_dir
            )
            return results
        except Exception as e:
            logger.error(f"Error updating real-time data: {e}")
            return {}
    
    def get_data_freshness(self, symbol: str) -> Tuple[datetime, int]:
        """Get the timestamp of the most recent data and minutes behind current time"""
        try:
            db_file = self.data_dir / f"{symbol}_historical.parquet"
            if not db_file.exists():
                return None, None
            
            df = pd.read_parquet(db_file)
            if df.empty:
                return None, None
            
            df.index = pd.to_datetime(df.index)
            latest_time = df.index.max()
            current_time = datetime.now(timezone.utc)
            minutes_behind = (current_time - latest_time).total_seconds() / 60
            
            return latest_time, minutes_behind
            
        except Exception as e:
            logger.error(f"Error checking data freshness for {symbol}: {e}")
            return None, None

def main():
    """Test the unified data manager"""
    import yaml
    
    # Load API keys
    with open('config/api_keys.yaml', 'r') as f:
        api_keys = yaml.safe_load(f)
    
    manager = UnifiedDataManager(api_keys)
    
    # Test historical data
    print("Testing historical data...")
    btc_historical = manager.get_historical_data('BTC', days=7)
    print(f"Historical BTC data: {len(btc_historical)} points")
    if not btc_historical.empty:
        print(f"Latest historical: {btc_historical.index[-1]} - ${btc_historical['close'].iloc[-1]:.2f}")
    
    # Test real-time data
    print("\nTesting real-time data...")
    btc_realtime = manager.get_realtime_price('BTC')
    print(f"Real-time BTC: {btc_realtime}")
    
    # Test combined data
    print("\nTesting combined data...")
    btc_combined = manager.get_combined_data('BTC', days=7, include_realtime=True)
    print(f"Combined BTC data: {len(btc_combined)} points")
    if not btc_combined.empty:
        print(f"Latest combined: {btc_combined.index[-1]} - ${btc_combined['close'].iloc[-1]:.2f}")
    
    # Test data freshness
    print("\nTesting data freshness...")
    latest_time, minutes_behind = manager.get_data_freshness('BTC')
    print(f"Latest data: {latest_time}, Minutes behind: {minutes_behind:.1f}")

if __name__ == "__main__":
    main()
