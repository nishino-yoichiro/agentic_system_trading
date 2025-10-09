#!/usr/bin/env python3
"""
Live Data Collector
Continuously collects real-time price data for live trading
Updates every few seconds with current spot prices
"""

import pandas as pd
import requests
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import threading
import json

logger = logging.getLogger(__name__)

class LiveDataCollector:
    """Continuously collects real-time price data for live trading"""
    
    def __init__(self, data_dir: Path = None, update_interval: int = 5):
        self.data_dir = data_dir or Path("data/crypto_db")
        self.data_dir.mkdir(exist_ok=True)
        self.update_interval = update_interval  # seconds
        self.running = False
        self.symbols = ['BTC']  # Default to BTC for testing
        
        # Real-time price API
        self.base_url = "https://api.coinbase.com/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoAgent/1.0'
        })
        
        # Thread for continuous collection
        self.collection_thread = None
        
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current spot price for a symbol"""
        try:
            currency_pair = f"{symbol}-USD"
            url = f"{self.base_url}/prices/{currency_pair}/spot"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data:
                return {
                    'symbol': symbol,
                    'price': float(data['data']['amount']),
                    'timestamp': datetime.now(timezone.utc),
                    'base': data['data']['base'],
                    'currency': data['data']['currency']
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def update_price_data(self, symbol: str) -> bool:
        """Update the price data file with current price"""
        try:
            # Get current price
            price_data = self.get_current_price(symbol)
            if not price_data:
                return False
            
            current_time = price_data['timestamp']
            current_price = price_data['price']
            
            # Load existing data
            db_file = self.data_dir / f"{symbol}_historical.parquet"
            if db_file.exists():
                df = pd.read_parquet(db_file)
                df.index = pd.to_datetime(df.index)
                
                # Check if we need to update the latest minute or create a new one
                if not df.empty:
                    latest_time = df.index.max()
                    time_diff = (current_time - latest_time).total_seconds()
                    
                    if time_diff < 60:  # Within the same minute
                        # Update the current minute's data
                        df.loc[latest_time, 'close'] = current_price
                        df.loc[latest_time, 'high'] = max(df.loc[latest_time, 'high'], current_price)
                        df.loc[latest_time, 'low'] = min(df.loc[latest_time, 'low'], current_price)
                        # Keep original volume
                    else:
                        # Create new minute data
                        new_row = pd.DataFrame({
                            'open': [current_price],
                            'high': [current_price],
                            'low': [current_price],
                            'close': [current_price],
                            'volume': [0.0]  # No volume data from spot price
                        }, index=[current_time])
                        df = pd.concat([df, new_row])
                else:
                    # No existing data, create new
                    df = pd.DataFrame({
                        'open': [current_price],
                        'high': [current_price],
                        'low': [current_price],
                        'close': [current_price],
                        'volume': [0.0]
                    }, index=[current_time])
            else:
                # No file exists, create new
                df = pd.DataFrame({
                    'open': [current_price],
                    'high': [current_price],
                    'low': [current_price],
                    'close': [current_price],
                    'volume': [0.0]
                }, index=[current_time])
            
            # Keep only last 7 days to prevent file from growing too large
            cutoff = current_time - timedelta(days=7)
            df = df[df.index >= cutoff]
            
            # Save updated data
            df.to_parquet(db_file)
            
            logger.debug(f"Updated {symbol}: ${current_price:.2f} at {current_time.strftime('%H:%M:%S')}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating price data for {symbol}: {e}")
            return False
    
    def _collection_loop(self):
        """Main collection loop that runs in a separate thread"""
        logger.info(f"Starting live data collection for {self.symbols} (every {self.update_interval}s)")
        
        while self.running:
            try:
                for symbol in self.symbols:
                    self.update_price_data(symbol)
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(self.update_interval)
    
    def start_collection(self, symbols: List[str] = None):
        """Start continuous data collection"""
        if symbols:
            self.symbols = symbols
        
        if self.running:
            logger.warning("Collection already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info(f"Started live data collection for {self.symbols}")
    
    def stop_collection(self):
        """Stop continuous data collection"""
        if not self.running:
            return
        
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        logger.info("Stopped live data collection")
    
    def get_latest_data(self, symbol: str, minutes: int = 1) -> pd.DataFrame:
        """Get the latest N minutes of data"""
        try:
            db_file = self.data_dir / f"{symbol}_historical.parquet"
            if not db_file.exists():
                return pd.DataFrame()
            
            df = pd.read_parquet(db_file)
            df.index = pd.to_datetime(df.index)
            
            # Get last N minutes
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            recent_df = df[df.index >= cutoff]
            
            return recent_df
            
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {e}")
            return pd.DataFrame()

def main():
    """Test the live data collector"""
    collector = LiveDataCollector(update_interval=5)
    
    print("Testing live data collector...")
    
    # Test single update
    print("Testing single update...")
    success = collector.update_price_data('BTC')
    print(f"Update successful: {success}")
    
    # Test latest data
    print("Testing latest data retrieval...")
    latest = collector.get_latest_data('BTC', minutes=5)
    print(f"Latest data points: {len(latest)}")
    if not latest.empty:
        print(f"Latest price: ${latest['close'].iloc[-1]:.2f} at {latest.index[-1]}")
    
    # Test continuous collection (run for 30 seconds)
    print("Testing continuous collection for 30 seconds...")
    collector.start_collection(['BTC'])
    
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    
    collector.stop_collection()
    print("Test completed")

if __name__ == "__main__":
    main()

