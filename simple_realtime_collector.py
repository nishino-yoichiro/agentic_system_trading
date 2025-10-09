#!/usr/bin/env python3
"""
Simple Real-Time Data Collector
Collects WebSocket data and builds 1-minute candles, appending to existing historical data
"""

import asyncio
import pandas as pd
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_ingestion.websocket_price_feed import WebSocketPriceFeed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

class SimpleRealtimeCollector:
    def __init__(self, symbols=['BTC']):
        self.symbols = symbols
        self.data_dir = Path("data/crypto_db")  # Match where live trading looks
        self.data_dir.mkdir(exist_ok=True)
        
        # Current minute data being built
        self.current_minute = {}
        
        # WebSocket feed
        ws_symbols = [f"{s}-USD" for s in symbols]
        self.ws_feed = WebSocketPriceFeed(ws_symbols, self._on_price_update)
        
    def _load_existing_data(self, symbol):
        """Load existing historical data"""
        file_path = self.data_dir / f"{symbol}_1m_historical.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df.index = pd.to_datetime(df.index)
            logger.info(f"Loaded {len(df)} existing data points for {symbol}")
            return df
        else:
            logger.info(f"No existing data for {symbol}, starting fresh")
            return pd.DataFrame()
    
    def _save_data(self, symbol, df):
        """Save data to file"""
        file_path = self.data_dir / f"{symbol}_1m_historical.parquet"
        df.to_parquet(file_path)
        logger.info(f"Saved {len(df)} data points for {symbol}")
    
    def _on_price_update(self, symbol, price_data):
        """Handle real-time price updates"""
        try:
            symbol = symbol.replace('-USD', '')
            current_time = price_data['timestamp']
            price = price_data['price']
            
            # Round to current minute
            minute_time = current_time.replace(second=0, microsecond=0)
            
            # Initialize or update current minute data
            if symbol not in self.current_minute:
                self.current_minute[symbol] = {
                    'minute': minute_time,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': 0.0,
                    'count': 1
                }
            else:
                data = self.current_minute[symbol]
                
                # Check if we're in a new minute
                if minute_time != data['minute']:
                    # Save previous minute data
                    self._finalize_minute(symbol, data)
                    
                    # Start new minute
                    self.current_minute[symbol] = {
                        'minute': minute_time,
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': 0.0,
                        'count': 1
                    }
                else:
                    # Update current minute
                    data['high'] = max(data['high'], price)
                    data['low'] = min(data['low'], price)
                    data['close'] = price
                    data['count'] += 1
                    data['volume'] = data['count'] * 0.1  # Estimate volume
            
        except Exception as e:
            logger.error(f"Error handling price update for {symbol}: {e}")
    
    def _finalize_minute(self, symbol, data):
        """Finalize and save completed minute data"""
        try:
            # Load existing data
            existing_df = self._load_existing_data(symbol)
            
            # Create new minute data
            new_data = pd.DataFrame({
                'open': [data['open']],
                'high': [data['high']],
                'low': [data['low']],
                'close': [data['close']],
                'volume': [data['volume']]
            }, index=[data['minute']])
            
            # Combine with existing data
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_data])
                # Remove duplicates and sort
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.sort_index()
            else:
                combined_df = new_data
            
            # Keep only last 7 days to prevent file from growing too large
            cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=7)
            combined_df = combined_df[combined_df.index >= cutoff]
            
            # Save data
            self._save_data(symbol, combined_df)
            
            logger.info(f"Finalized {symbol} minute: {data['minute'].strftime('%H:%M')} - ${data['close']:.2f}")
            
        except Exception as e:
            logger.error(f"Error finalizing minute for {symbol}: {e}")
    
    def start(self):
        """Start collecting data"""
        logger.info(f"Starting real-time collection for {self.symbols}")
        
        # Load existing data for each symbol
        for symbol in self.symbols:
            existing_df = self._load_existing_data(symbol)
            if len(existing_df) >= 50:
                logger.info(f"✅ {symbol}: {len(existing_df)} data points - sufficient for technical analysis")
            else:
                logger.warning(f"⚠️  {symbol}: {len(existing_df)} data points - may need more for technical indicators")
        
        logger.info("Press Ctrl+C to stop")
        
        try:
            self.ws_feed.start()
            
            # Keep running
            while True:
                import time
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Stopping collection...")
            self.ws_feed.stop()
            
            # Save any remaining minute data
            for symbol, data in self.current_minute.items():
                self._finalize_minute(symbol, data)
            
            logger.info("Collection stopped")

def main():
    """Main function"""
    print("🚀 Simple Real-Time Data Collector")
    print("=" * 40)
    print("Collects WebSocket data and builds 1-minute candles")
    print("Appends to existing historical data")
    print("=" * 40)
    
    collector = SimpleRealtimeCollector(['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI'])
    collector.start()

if __name__ == "__main__":
    main()
