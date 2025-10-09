#!/usr/bin/env python3
"""
WebSocket-Only Data Collection
Collects real-time data via WebSocket and saves to 1-minute files
No historical bulk collection - just real-time updates
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_ingestion.websocket_price_feed import WebSocketPriceFeed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/websocket_collection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class WebSocketOnlyCollector:
    """Collects only real-time data via WebSocket"""
    
    def __init__(self, data_dir: Path = None, symbols: list = None):
        self.data_dir = data_dir or Path("data/crypto_db")
        self.data_dir.mkdir(exist_ok=True)
        self.symbols = symbols if symbols is not None else ['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI']
        
        # WebSocket feed
        ws_symbols = [f"{s}-USD" for s in self.symbols]
        self.ws_feed = WebSocketPriceFeed(ws_symbols, self._on_price_update)
        
        # Current minute data
        self.current_minute_data = {}
        
        # Load existing historical data for each symbol
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing historical data for all symbols"""
        logger.info("Loading existing historical data...")
        
        for symbol in self.symbols:
            try:
                # Try to load from both possible locations
                file_paths = [
                    self.data_dir / f"{symbol}_1m_historical.parquet",
                    Path("data") / f"{symbol}_1m_historical.parquet"
                ]
                
                existing_data = None
                for file_path in file_paths:
                    if file_path.exists():
                        existing_data = pd.read_parquet(file_path)
                        existing_data.index = pd.to_datetime(existing_data.index)
                        logger.info(f"Loaded {len(existing_data)} existing data points for {symbol}")
                        break
                
                if existing_data is not None and len(existing_data) > 0:
                    # Ensure we have enough data for technical indicators (at least 50 periods)
                    if len(existing_data) < 50:
                        logger.warning(f"Only {len(existing_data)} data points for {symbol} - may not be enough for technical indicators")
                    else:
                        logger.info(f"✅ {symbol} has {len(existing_data)} data points - sufficient for technical analysis")
                else:
                    logger.warning(f"No existing data found for {symbol} - will start fresh")
                    
            except Exception as e:
                logger.error(f"Error loading existing data for {symbol}: {e}")
        
    def _on_price_update(self, symbol: str, price_data: dict):
        """Handle real-time price updates"""
        try:
            symbol = symbol.replace('-USD', '')
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
                cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=7)
                combined_df = combined_df[combined_df.index >= cutoff]
                
                # Save
                combined_df.to_parquet(file_path)
                logger.info(f"Saved {symbol} minute data: {data['minute'].strftime('%H:%M')} - ${data['close']:.2f}")
            else:
                # First time - create new file
                df.to_parquet(file_path)
                logger.info(f"Created new minute file for {symbol}: {data['minute'].strftime('%H:%M')} - ${data['close']:.2f}")
                
        except Exception as e:
            logger.error(f"Error saving minute data for {symbol}: {e}")
    
    def _ensure_sufficient_data(self):
        """Ensure we have enough historical data for technical indicators"""
        logger.info("Checking data sufficiency for technical indicators...")
        
        for symbol in self.symbols:
            try:
                file_path = self.data_dir / f"{symbol}_1m_historical.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    df.index = pd.to_datetime(df.index)
                    
                    if len(df) < 50:
                        logger.warning(f"⚠️  {symbol}: Only {len(df)} data points available")
                        logger.warning(f"   Technical indicators may not work properly")
                        logger.warning(f"   Consider running historical data collection first")
                    else:
                        logger.info(f"✅ {symbol}: {len(df)} data points - sufficient for technical analysis")
                else:
                    logger.warning(f"❌ {symbol}: No data file found")
                    
            except Exception as e:
                logger.error(f"Error checking data for {symbol}: {e}")
    
    def start(self):
        """Start WebSocket collection"""
        logger.info(f"Starting WebSocket-only collection for {self.symbols}")
        logger.info("This will collect real-time data and save 1-minute candles")
        
        # Check data sufficiency
        self._ensure_sufficient_data()
        
        logger.info("Press Ctrl+C to stop")
        
        try:
            self.ws_feed.start()
            
            # Keep running
            while True:
                import time
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Stopping WebSocket collection...")
            self.ws_feed.stop()
            
            # Save any remaining minute data
            for symbol, data in self.current_minute_data.items():
                self._save_minute_data(symbol, data)
            
            logger.info("WebSocket collection stopped")

def main():
    """Main function"""
    print("🚀 WebSocket-Only Data Collection")
    print("=" * 50)
    print("This collects real-time data via WebSocket")
    print("No historical bulk collection - just live updates")
    print("=" * 50)
    
    collector = WebSocketOnlyCollector()
    collector.start()

if __name__ == "__main__":
    main()
