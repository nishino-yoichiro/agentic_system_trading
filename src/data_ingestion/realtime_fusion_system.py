#!/usr/bin/env python3
"""
Real-Time Fusion System
Combines historical OHLCV data with live WebSocket ticks
Creates pseudo-candles for real-time signal generation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable
import logging
from pathlib import Path
import threading
import time

from .websocket_price_feed import WebSocketPriceFeed

logger = logging.getLogger(__name__)

class PseudoCandle:
    """Represents a synthetic candle being built from live ticks"""
    
    def __init__(self, symbol: str, start_time: datetime, initial_price: float):
        self.symbol = symbol
        self.start_time = start_time
        self.open = initial_price
        self.high = initial_price
        self.low = initial_price
        self.close = initial_price
        self.volume = 0.0
        self.tick_count = 1
        self.last_update = start_time
        
    def update(self, price: float, timestamp: datetime):
        """Update pseudo-candle with new tick"""
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.tick_count += 1
        self.last_update = timestamp
        
        # Estimate volume based on tick frequency and price movement
        price_change = abs(price - self.open) / self.open
        self.volume = self.tick_count * (1 + price_change) * 0.1  # Rough estimation
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame row"""
        return pd.DataFrame({
            'open': [self.open],
            'high': [self.high],
            'low': [self.low],
            'close': [self.close],
            'volume': [self.volume]
        }, index=[self.start_time])
    
    def is_complete(self, current_time: datetime) -> bool:
        """Check if candle should be finalized (new minute started)"""
        return current_time.minute != self.start_time.minute

class RealTimeFusionSystem:
    """Main fusion system that combines historical and live data"""
    
    def __init__(self, data_dir: Path = None, symbols: List[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/crypto_db")
        self.symbols = symbols or ['BTC']
        self.data_dir.mkdir(exist_ok=True)
        
        # Historical data buffers (multi-timeframe)
        self.historical_data = {
            '1h': {},   # 1-hour candles
            '15m': {},  # 15-minute candles
            '5m': {},   # 5-minute candles
            '1m': {}    # 1-minute candles
        }
        
        # Real-time data
        self.pseudo_candles = {}  # Current pseudo-candles being built
        self.latest_prices = {}   # Latest prices from WebSocket
        
        # WebSocket feed
        self.ws_feed = None
        self.running = False
        
        # Signal tracking
        self.last_signal_time = {}  # Track last signal time per symbol
        
        # Callbacks
        self.signal_callback = None
        self.update_interval = 60  # Generate signals every minute
        
    def start(self, signal_callback: Callable = None):
        """Start the fusion system"""
        if self.running:
            return
        
        self.signal_callback = signal_callback
        self.running = True
        
        # Load historical data
        self._load_historical_data()
        
        # Start WebSocket feed
        ws_symbols = [f"{s}-USD" for s in self.symbols]
        self.ws_feed = WebSocketPriceFeed(ws_symbols, self._on_price_update)
        self.ws_feed.start()
        
        # Start signal generation loop
        self.signal_thread = threading.Thread(target=self._signal_loop, daemon=True)
        self.signal_thread.start()
        
        logger.info("Started Real-Time Fusion System")
    
    def stop(self):
        """Stop the fusion system"""
        self.running = False
        if self.ws_feed:
            self.ws_feed.stop()
        logger.info("Stopped Real-Time Fusion System")
    
    def _load_historical_data(self):
        """Load historical data for all timeframes"""
        for symbol in self.symbols:
            for timeframe in self.historical_data.keys():
                file_path = self.data_dir / f"{symbol}_{timeframe}_historical.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    df.index = pd.to_datetime(df.index)
                    self.historical_data[timeframe][symbol] = df
                    logger.info(f"Loaded {len(df)} {timeframe} candles for {symbol}")
                else:
                    # Create empty DataFrame
                    self.historical_data[timeframe][symbol] = pd.DataFrame()
    
    def _on_price_update(self, symbol: str, price_data: Dict):
        """Handle real-time price updates from WebSocket"""
        try:
            symbol = symbol.replace('-USD', '')
            self.latest_prices[symbol] = price_data
            
            current_time = price_data['timestamp']
            price = price_data['price']
            
            # Update or create pseudo-candle
            if symbol not in self.pseudo_candles:
                self.pseudo_candles[symbol] = PseudoCandle(symbol, current_time, price)
            else:
                pseudo_candle = self.pseudo_candles[symbol]
                
                # Check if we need to finalize current candle
                if pseudo_candle.is_complete(current_time):
                    self._finalize_pseudo_candle(symbol)
                    # Create new pseudo-candle
                    self.pseudo_candles[symbol] = PseudoCandle(symbol, current_time, price)
                else:
                    # Update existing pseudo-candle
                    pseudo_candle.update(price, current_time)
            
        except Exception as e:
            logger.error(f"Error handling price update for {symbol}: {e}")
    
    def _finalize_pseudo_candle(self, symbol: str):
        """Finalize pseudo-candle and add to historical data"""
        if symbol not in self.pseudo_candles:
            return
        
        pseudo_candle = self.pseudo_candles[symbol]
        
        # Add to 1-minute data
        if symbol in self.historical_data['1m']:
            new_candle = pseudo_candle.to_dataframe()
            self.historical_data['1m'][symbol] = pd.concat([
                self.historical_data['1m'][symbol], 
                new_candle
            ])
            
            # Keep only last 7 days
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            self.historical_data['1m'][symbol] = self.historical_data['1m'][symbol][
                self.historical_data['1m'][symbol].index >= cutoff
            ]
            
            # Save to file
            self._save_historical_data(symbol, '1m')
            
            logger.debug(f"Finalized pseudo-candle for {symbol}: ${pseudo_candle.close:.2f}")
    
    def _save_historical_data(self, symbol: str, timeframe: str):
        """Save historical data to file"""
        try:
            file_path = self.data_dir / f"{symbol}_{timeframe}_historical.parquet"
            self.historical_data[timeframe][symbol].to_parquet(file_path)
        except Exception as e:
            logger.error(f"Error saving {timeframe} data for {symbol}: {e}")
    
    def _signal_loop(self):
        """Main signal generation loop"""
        while self.running:
            try:
                # Wait until the next minute boundary
                now = datetime.now()
                seconds_until_next_minute = 60 - now.second
                time.sleep(seconds_until_next_minute)
                
                # Generate signals for all symbols
                for symbol in self.symbols:
                    self._generate_signal(symbol)
                
            except Exception as e:
                logger.error(f"Error in signal loop: {e}")
                time.sleep(5)
    
    def _generate_signal(self, symbol: str):
        """Generate signal for a symbol"""
        try:
            # Check if we already generated a signal for this minute
            current_time = datetime.now()
            current_minute = current_time.replace(second=0, microsecond=0)
            
            if symbol in self.last_signal_time and self.last_signal_time[symbol] == current_minute:
                logger.debug(f"Signal already generated for {symbol} this minute")
                return
            
            # Get multi-timeframe context
            context = self._get_signal_context(symbol)
            
            if not context:
                return
            
            # Generate signal based on context
            signal = self._create_signal(symbol, context)
            
            # Call signal callback if provided
            if self.signal_callback and signal:
                self.last_signal_time[symbol] = current_minute
                self.signal_callback(symbol, signal)
                
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
    
    def _get_signal_context(self, symbol: str) -> Dict:
        """Get multi-timeframe context for signal generation"""
        context = {}
        
        # Get historical context for each timeframe
        for timeframe in ['1h', '15m', '5m', '1m']:
            if symbol in self.historical_data[timeframe]:
                df = self.historical_data[timeframe][symbol]
                if not df.empty:
                    context[timeframe] = df.tail(50)  # Last 50 candles
        
        # Add current pseudo-candle if available
        if symbol in self.pseudo_candles:
            pseudo_candle = self.pseudo_candles[symbol]
            context['current'] = pseudo_candle.to_dataframe()
        
        # Add latest price
        if symbol in self.latest_prices:
            context['latest_price'] = self.latest_prices[symbol]
        
        return context
    
    def _create_signal(self, symbol: str, context: Dict) -> Optional[Dict]:
        """Create trading signal based on context"""
        try:
            # Simple alternating strategy for testing
            current_time = datetime.now()
            minute = current_time.minute
            
            # Determine signal type
            if minute % 2 == 1:  # Odd minutes - LONG
                signal_type = 'LONG'
                confidence = 0.6
                reason = f"Test alternating LONG signal - minute {minute}"
            else:  # Even minutes - SHORT
                signal_type = 'SHORT'
                confidence = 0.6
                reason = f"Test alternating SHORT signal - minute {minute}"
            
            logger.info(f"DEBUG: Generated signal for {symbol}: {signal_type} at minute {minute}")
            
            # Get current price
            if 'latest_price' in context:
                current_price = context['latest_price']['price']
            elif 'current' in context:
                current_price = context['current']['close'].iloc[-1]
            else:
                return None
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'price': current_price,
                'reason': reason,
                'timestamp': current_time.isoformat(),
                'context': {
                    '1m_candles': len(context.get('1m', [])),
                    '5m_candles': len(context.get('5m', [])),
                    '15m_candles': len(context.get('15m', [])),
                    '1h_candles': len(context.get('1h', []))
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating signal for {symbol}: {e}")
            return None
    
    def get_latest_data(self, symbol: str, timeframe: str = '1m', minutes: int = 10) -> pd.DataFrame:
        """Get latest data for a symbol and timeframe"""
        try:
            if symbol not in self.historical_data[timeframe]:
                return pd.DataFrame()
            
            df = self.historical_data[timeframe][symbol]
            if df.empty:
                return df
            
            # Get last N minutes
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            recent_df = df[df.index >= cutoff]
            
            # Add current pseudo-candle if available
            if symbol in self.pseudo_candles:
                pseudo_candle = self.pseudo_candles[symbol]
                pseudo_df = pseudo_candle.to_dataframe()
                recent_df = pd.concat([recent_df, pseudo_df])
            
            return recent_df
            
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {e}")
            return pd.DataFrame()

def main():
    """Test the fusion system"""
    def signal_callback(symbol, signal):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol} Signal: {signal['signal_type']} @ ${signal['price']:.2f}")
    
    fusion = RealTimeFusionSystem(symbols=['BTC'])
    
    try:
        fusion.start(signal_callback)
        time.sleep(60)  # Run for 1 minute
    except KeyboardInterrupt:
        pass
    finally:
        fusion.stop()

if __name__ == "__main__":
    main()
