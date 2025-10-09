#!/usr/bin/env python3
"""
Data Collection Only
====================

Runs only the data collection component to build 1-minute candles
from real-time WebSocket data. This can be run separately from
signal generation and trading.

Author: Quantitative Strategy Designer
Date: 2025-01-28
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import pandas as pd
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_ingestion.websocket_price_feed import WebSocketPriceFeed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCollectionOnly:
    """
    Data collection system that builds 1-minute candles from real-time data
    """
    
    def __init__(self, data_dir: str = "data", symbols: List[str] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.symbols = symbols or ['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI']
        
        # WebSocket feed
        ws_symbols = [f"{s}-USD" for s in self.symbols]
        self.ws_feed = WebSocketPriceFeed(ws_symbols, self._on_price_update)
        
        # Current minute data
        self.current_minute_data = {}
        self.running = False
        
        # Load existing data
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing historical data for all symbols"""
        logger.info("Loading existing historical data...")
        
        for symbol in self.symbols:
            try:
                file_path = self.data_dir / f"{symbol}_1m_historical.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    df.index = pd.to_datetime(df.index)
                    logger.info(f"✅ {symbol}: {len(df)} data points available")
                else:
                    logger.warning(f"❌ {symbol}: No data file found")
                    
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
    
    def _on_price_update(self, symbol: str, price_data: dict):
        """Handle real-time price updates and build 1-minute candles"""
        try:
            symbol = symbol.replace('-USD', '')
            if symbol not in self.symbols:
                return
                
            current_time = price_data['timestamp']
            price = price_data['price']
            
            # Round to current minute
            minute_time = current_time.replace(second=0, microsecond=0)
            
            # Initialize current minute data if needed
            if symbol not in self.current_minute_data:
                self.current_minute_data[symbol] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': 0.0,
                    'count': 1,
                    'minute': minute_time
                }
            else:
                # Update current minute data
                data = self.current_minute_data[symbol]
                
                # Check if we're in a new minute
                if minute_time != data['minute']:
                    # Save previous minute data
                    self._save_minute_data(symbol, data)
                    
                    # Start new minute
                    self.current_minute_data[symbol] = {
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': 0.0,
                        'count': 1,
                        'minute': minute_time
                    }
                else:
                    # Update current minute
                    data['high'] = max(data['high'], price)
                    data['low'] = min(data['low'], price)
                    data['close'] = price
                    data['count'] += 1
                    # Estimate volume based on tick count
                    data['volume'] = data['count'] * 0.1
            
            logger.debug(f"Updated {symbol}: ${price:.2f} at {current_time.strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Error handling price update for {symbol}: {e}")
    
    def _save_minute_data(self, symbol: str, data: dict):
        """Save completed minute data to file"""
        try:
            # Create DataFrame for this minute
            df = pd.DataFrame({
                'open': [data['open']],
                'high': [data['high']],
                'low': [data['low']],
                'close': [data['close']],
                'volume': [data['volume']]
            }, index=[data['minute']])
            
            # File path
            file_path = self.data_dir / f"{symbol}_1m_historical.parquet"
            
            # Load existing data
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                existing_df.index = pd.to_datetime(existing_df.index)
                
                # Combine and remove duplicates
                combined_df = pd.concat([existing_df, df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.sort_index()
                
                # Keep only last 7 days
                cutoff = datetime.now() - timedelta(days=7)
                combined_df = combined_df[combined_df.index >= cutoff]
                
                # Save
                combined_df.to_parquet(file_path)
                logger.info(f"💾 {symbol}: {data['minute'].strftime('%H:%M')} - ${data['close']:.2f}")
            else:
                # First time - create new file
                df.to_parquet(file_path)
                logger.info(f"📁 {symbol}: Created new file - {data['minute'].strftime('%H:%M')} - ${data['close']:.2f}")
                
        except Exception as e:
            logger.error(f"Error saving minute data for {symbol}: {e}")
    
    def start(self):
        """Start data collection"""
        if self.running:
            logger.warning("Data collection already running")
            return
        
        logger.info("📡 Starting Data Collection")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info("This will collect real-time data and build 1-minute candles")
        logger.info("Press Ctrl+C to stop")
        
        self.running = True
        
        try:
            self.ws_feed.start()
            
            # Keep running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Stopping data collection...")
            self.stop()
    
    def stop(self):
        """Stop data collection"""
        self.running = False
        
        if self.ws_feed:
            self.ws_feed.stop()
        
        # Save any remaining minute data
        for symbol, data in self.current_minute_data.items():
            self._save_minute_data(symbol, data)
        
        logger.info("✅ Data collection stopped")
    
    def get_data_summary(self) -> Dict:
        """Get summary of collected data"""
        summary = {}
        
        for symbol in self.symbols:
            try:
                file_path = self.data_dir / f"{symbol}_1m_historical.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    df.index = pd.to_datetime(df.index)
                    
                    summary[symbol] = {
                        'data_points': len(df),
                        'latest_time': df.index.max().isoformat() if not df.empty else None,
                        'latest_price': df['close'].iloc[-1] if not df.empty else None,
                        'file_size_mb': file_path.stat().st_size / (1024 * 1024)
                    }
                else:
                    summary[symbol] = {'data_points': 0, 'latest_time': None, 'latest_price': None, 'file_size_mb': 0}
                    
            except Exception as e:
                logger.error(f"Error getting summary for {symbol}: {e}")
                summary[symbol] = {'error': str(e)}
        
        return summary

def main():
    """Main function"""
    print("📡 Data Collection Only")
    print("=" * 50)
    print("This collects real-time data via WebSocket")
    print("and builds 1-minute candles for analysis")
    print("=" * 50)
    
    # Initialize data collection
    collector = DataCollectionOnly()
    
    # Show initial data summary
    summary = collector.get_data_summary()
    print(f"\n📊 Current Data Status:")
    for symbol, data in summary.items():
        if 'error' in data:
            print(f"   {symbol}: Error - {data['error']}")
        else:
            print(f"   {symbol}: {data['data_points']} points, Latest: ${data['latest_price']:.2f} ({data['latest_time']})")
    print()
    
    # Start data collection
    collector.start()

if __name__ == "__main__":
    main()
