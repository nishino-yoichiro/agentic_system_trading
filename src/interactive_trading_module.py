#!/usr/bin/env python3
"""
Interactive Trading Module
=========================

Interactive trading system that allows manual trading based on generated signals.
Features:
- Strategy and ticker selection
- Configurable signal generation intervals
- Sound notifications for signals
- Desktop notifications
- Minute-by-minute signal summaries
- Real-time websocket data integration

Author: Quantitative Strategy Designer
Date: 2025-01-28
"""

import asyncio
import logging
import sys
import time
import threading
import json
import os
import platform
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
import yaml
from collections import defaultdict, deque

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion.websocket_price_feed import WebSocketPriceFeed
from src.crypto_signal_integration import CryptoSignalIntegration
from src.crypto_analysis_engine import CryptoAnalysisEngine
from src.crypto_signal_framework import Signal, SignalType

# Sound and notification imports
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Sound notifications will be disabled.")

try:
    from plyer import notification
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False
    print("Warning: plyer not available. Desktop notifications will be disabled.")

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,  # Temporarily enable debug logging
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/interactive_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SoundNotificationSystem:
    """Handles sound notifications for trading signals"""
    
    def __init__(self):
        self.enabled = PYGAME_AVAILABLE
        self.sounds = {}
        
        if self.enabled:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self._load_sounds()
            except Exception as e:
                logger.warning(f"Failed to initialize pygame mixer: {e}")
                self.enabled = False
    
    def _load_sounds(self):
        """Load different sounds for different signal types"""
        try:
            # Create simple beep sounds programmatically
            self.sounds = {
                'buy': self._create_beep_sound(800, 0.3),  # Higher pitch for buy
                'sell': self._create_beep_sound(400, 0.3),  # Lower pitch for sell
                'strong_buy': self._create_beep_sound(1000, 0.5),  # Even higher for strong buy
                'strong_sell': self._create_beep_sound(300, 0.5),  # Even lower for strong sell
                'default': self._create_beep_sound(600, 0.2)  # Default sound
            }
        except Exception as e:
            logger.warning(f"Failed to create sounds: {e}")
            self.enabled = False
    
    def _create_beep_sound(self, frequency: int, duration: float):
        """Create a simple beep sound"""
        sample_rate = 22050
        frames = int(duration * sample_rate)
        arr = np.zeros((frames, 2))
        
        for i in range(frames):
            wave = 4096 * np.sin(2 * np.pi * frequency * i / sample_rate)
            arr[i][0] = wave  # Left channel
            arr[i][1] = wave  # Right channel
        
        sound = pygame.sndarray.make_sound(arr.astype(np.int16))
        return sound
    
    def play_signal_sound(self, signal: Signal):
        """Play appropriate sound for signal type"""
        if not self.enabled:
            return
        
        try:
            sound_key = 'default'
            
            if signal.signal_type == SignalType.LONG:
                sound_key = 'buy'
            elif signal.signal_type == SignalType.SHORT:
                sound_key = 'sell'
            
            if sound_key in self.sounds:
                self.sounds[sound_key].play()
                
        except Exception as e:
            logger.warning(f"Failed to play sound: {e}")

class DesktopNotificationSystem:
    """Handles desktop notifications for trading signals"""
    
    def __init__(self):
        self.enabled = False  # Disable notifications to avoid Windows TclNotifier errors
    
    def send_signal_notification(self, signal: Signal, symbol: str):
        """Send desktop notification for signal"""
        if not self.enabled:
            return
        
        try:
            # Determine notification title and message
            signal_emoji = {
                SignalType.LONG: "🟢",
                SignalType.SHORT: "🔴", 
                SignalType.FLAT: "📊"
            }.get(signal.signal_type, "📊")
            
            title = f"{signal_emoji} {signal.signal_type.name.upper()} Signal - {symbol}"
            message = f"Strategy: {signal.strategy_name}\nConfidence: {signal.confidence:.2f}\nReason: {signal.reason}"
            
            # Send notification
            notification.notify(
                title=title,
                message=message,
                timeout=10,  # Show for 10 seconds
                app_name="Crypto Trading Signals"
            )
            
        except Exception as e:
            # Disable notifications if they fail
            logger.warning(f"Desktop notifications disabled due to error: {e}")
            self.enabled = False

class InteractiveTradingModule:
    """
    Interactive trading module with real-time signal generation and notifications
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.signal_integration = None
        self.ws_feed = None
        self.sound_system = SoundNotificationSystem()
        self.notification_system = DesktopNotificationSystem()
        
        # Trading state
        self.running = False
        self.selected_strategies = []
        self.selected_symbols = []
        self.signal_interval = 60  # Default 1 minute
        self.current_minute_data = {}
        
        # Signal tracking
        self.signals_this_minute = defaultdict(list)
        self.total_signals_today = 0
        self.signal_history = deque(maxlen=1000)  # Keep last 1000 signals
        
        # Data collection
        self.historical_data = {}
        self.latest_prices = {}
        
        # WebSocket candle building
        self.current_candles = {}  # symbol -> current minute candle data
        
        # Threading
        self.signal_thread = None
        self.data_thread = None
        
    def setup_interactive_session(self):
        """Interactive setup for strategies, symbols, and intervals"""
        print("\n" + "="*60)
        print("🚀 INTERACTIVE TRADING MODULE SETUP")
        print("="*60)
        
        # Load available strategies
        try:
            integration = CryptoSignalIntegration()
            available_strategies = list(integration.framework.strategies.keys())
        except Exception as e:
            logger.warning(f"Could not load strategies: {e}")
            available_strategies = []
        
        # Strategy selection
        print(f"\n📊 Available Strategies:")
        for i, strategy in enumerate(available_strategies, 1):
            print(f"  {i}. {strategy}")
        
        print(f"\nEnter strategy numbers (comma-separated) or 'all' for all strategies:")
        strategy_input = input("Strategies: ").strip()
        
        if strategy_input.lower() == 'all':
            self.selected_strategies = available_strategies
        else:
            try:
                strategy_indices = [int(x.strip()) - 1 for x in strategy_input.split(',')]
                self.selected_strategies = [available_strategies[i] for i in strategy_indices if 0 <= i < len(available_strategies)]
            except (ValueError, IndexError):
                print("Invalid input. Using all strategies.")
                self.selected_strategies = available_strategies
        
        # Symbol selection
        default_symbols = ['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI']
        print(f"\n💰 Available Symbols: {', '.join(default_symbols)}")
        symbol_input = input("Enter symbols (comma-separated) or 'all' for all: ").strip()
        
        if symbol_input.lower() == 'all' or not symbol_input:
            self.selected_symbols = default_symbols
        else:
            self.selected_symbols = [s.strip().upper() for s in symbol_input.split(',') if s.strip()]
        
        # Signal interval selection
        print(f"\n⏱️  Signal Generation Interval:")
        print("  1. Every 10 seconds (high frequency)")
        print("  2. Every 20 seconds")
        print("  3. Every 30 seconds")
        print("  4. Every 60 seconds (1 minute) - default")
        print("  5. Every 2 minutes")
        print("  6. Every 5 minutes")
        
        interval_choice = input("Choose interval (1-6) [4]: ").strip()
        
        interval_map = {
            '1': 10, '2': 20, '3': 30, '4': 60, '5': 120, '6': 300
        }
        
        self.signal_interval = interval_map.get(interval_choice, 60)
        
        # Notification preferences
        print(f"\n🔔 Notification Settings:")
        sound_enabled = input("Enable sound notifications? (Y/n): ").strip().lower()
        if sound_enabled in ['n', 'no']:
            self.sound_system.enabled = False
        
        desktop_enabled = input("Enable desktop notifications? (Y/n): ").strip().lower()
        if desktop_enabled in ['n', 'no']:
            self.notification_system.enabled = False
        
        print(f"\n✅ Configuration Complete!")
        print(f"   Strategies: {', '.join(self.selected_strategies)}")
        print(f"   Symbols: {', '.join(self.selected_symbols)}")
        print(f"   Signal Interval: {self.signal_interval} seconds")
        print(f"   Sound Notifications: {'Enabled' if self.sound_system.enabled else 'Disabled'}")
        print(f"   Desktop Notifications: {'Enabled' if self.notification_system.enabled else 'Disabled'}")
        
        return True
    
    def _load_historical_data(self):
        """Load historical data for selected symbols and check for gaps"""
        logger.info("Loading historical data and checking for gaps...")
        
        for symbol in self.selected_symbols:
            try:
                # Try to load data from multiple possible locations
                data_files = [
                    self.data_dir / f"{symbol}_1m_historical.parquet",
                    self.data_dir / "crypto_db" / f"{symbol}_historical.parquet"
                ]
                
                data_loaded = False
                for data_file in data_files:
                    if data_file.exists():
                        df = pd.read_parquet(data_file)
                        if not df.empty:
                            # Ensure timestamp is properly handled
                            df = self._ensure_timestamp_column(df)
                            
                            self.historical_data[symbol] = df
                            logger.info(f"Loaded {len(df)} historical records for {symbol} from {data_file}")
                            
                            # Check for data gaps
                            self._check_data_gaps(symbol, df)
                            data_loaded = True
                            break
                
                if not data_loaded:
                    logger.warning(f"No historical data found for {symbol} in any location")
                    print(f"⚠️  WARNING: No historical data found for {symbol}")
                    print(f"   This will prevent signal generation!")
                    print(f"   Run: python main.py data-collection --symbols {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to load historical data for {symbol}: {e}")
    
    def _check_data_gaps(self, symbol: str, df: pd.DataFrame):
        """Check for data gaps and fill them if needed"""
        try:
            if df.empty:
                return
            
            # Get the latest timestamp
            if 'timestamp' in df.columns:
                latest_time = df['timestamp'].max()
            else:
                latest_time = df.index.max()
            
            # Ensure timezone awareness
            if latest_time.tz is None:
                latest_time = latest_time.tz_localize('UTC')
            
            current_time = datetime.now(timezone.utc)
            time_diff = (current_time - latest_time).total_seconds() / 60  # minutes
            
            # If data is more than 5 minutes old, suggest gap filling
            if time_diff > 5:
                logger.warning(f"Data for {symbol} is {time_diff:.1f} minutes old. Consider running gap filling:")
                logger.warning(f"  python main.py data-collection --symbols {symbol}")
                
                # Ask user if they want to fill gaps
                response = input(f"Fill gaps for {symbol}? (y/N): ").strip().lower()
                if response in ['y', 'yes']:
                    try:
                        self._fill_data_gaps(symbol)
                    except Exception as e:
                        logger.error(f"Gap filling failed: {e}")
                        logger.info(f"You can manually run: python main.py data-collection --symbols {symbol}")
                else:
                    logger.info(f"Skipping gap filling for {symbol}. You can run it later with:")
                    logger.info(f"  python main.py data-collection --symbols {symbol}")
            
        except Exception as e:
            logger.error(f"Error checking data gaps for {symbol}: {e}")
    
    def _fill_data_gaps(self, symbol: str):
        """Fill data gaps for a symbol"""
        try:
            logger.info(f"Filling data gaps for {symbol}...")
            
            # Import the data collector
            from src.data_ingestion.crypto_collector import CryptoDataCollector
            
            # CryptoDataCollector doesn't accept data_dir parameter
            collector = CryptoDataCollector()
            
            # Collect recent data (last 1 day) - this is async, so we need to run it
            import asyncio
            result = asyncio.run(collector.collect_crypto_data([symbol], days_back=1))
            
            if result and symbol in result:
                logger.info(f"Successfully filled gaps for {symbol}")
                # Reload the data
                self._reload_symbol_data(symbol)
            else:
                logger.warning(f"Failed to fill gaps for {symbol}")
                
        except Exception as e:
            logger.error(f"Error filling data gaps for {symbol}: {e}")
    
    def _reload_symbol_data(self, symbol: str):
        """Reload data for a specific symbol after gap filling"""
        try:
            # Check both possible data locations
            data_files = [
                self.data_dir / f"{symbol}_1m_historical.parquet",
                self.data_dir / "crypto_db" / f"{symbol}_historical.parquet"
            ]
            
            for data_file in data_files:
                if data_file.exists():
                    df = pd.read_parquet(data_file)
                    if not df.empty:
                        # Ensure timestamp is properly handled
                        df = self._ensure_timestamp_column(df)
                        
                        self.historical_data[symbol] = df
                        logger.info(f"Reloaded {len(df)} records for {symbol} from {data_file}")
                        return
            
            logger.warning(f"No updated data found for {symbol} after gap filling")
            
        except Exception as e:
            logger.error(f"Error reloading data for {symbol}: {e}")
    
    def _initialize_signal_integration(self):
        """Initialize signal integration with selected strategies"""
        try:
            self.signal_integration = CryptoSignalIntegration(
                data_dir=str(self.data_dir),
                selected_strategies=self.selected_strategies
            )
            logger.info(f"Initialized signal integration with {len(self.selected_strategies)} strategies")
            
            # Debug: Log available strategies in the framework
            available_strategies = list(self.signal_integration.framework.strategies.keys())
            logger.debug(f"Available strategies in framework: {available_strategies}")
            
            # Debug: Check if test_every_minute is in the framework
            if 'test_every_minute' in available_strategies:
                logger.debug("test_every_minute strategy is available in framework")
            else:
                logger.warning("test_every_minute strategy is NOT available in framework")
                
        except Exception as e:
            logger.error(f"Failed to initialize signal integration: {e}")
            raise
    
    def _on_price_update(self, symbol: str, price_data: dict):
        """Handle real-time price updates from WebSocket"""
        try:
            # Extract symbol without -USD suffix
            base_symbol = symbol.replace('-USD', '')
            
            if base_symbol not in self.selected_symbols:
                return
            
            # Store latest price
            self.latest_prices[base_symbol] = price_data
            
            logger.debug(f"WebSocket update for {base_symbol}: ${price_data['price']:.2f}")
            
            # Update current minute data
            current_time = datetime.now(timezone.utc)
            minute_key = current_time.strftime('%Y-%m-%d %H:%M')
            
            # Ensure current_minute_data is initialized
            if self.current_minute_data is None:
                self.current_minute_data = {}
            
            if base_symbol not in self.current_minute_data:
                self.current_minute_data[base_symbol] = {}
            
            if minute_key not in self.current_minute_data[base_symbol]:
                self.current_minute_data[base_symbol][minute_key] = {
                    'open': price_data['price'],
                    'high': price_data['price'],
                    'low': price_data['price'],
                    'close': price_data['price'],
                    'volume': price_data.get('volume', 0),
                    'timestamp': current_time
                }
            else:
                # Update OHLCV for current minute
                minute_data = self.current_minute_data[base_symbol][minute_key]
                minute_data['high'] = max(minute_data['high'], price_data['price'])
                minute_data['low'] = min(minute_data['low'], price_data['price'])
                minute_data['close'] = price_data['price']
                minute_data['volume'] += price_data.get('volume', 0)
                
        except Exception as e:
            logger.error(f"Error processing price update for {symbol}: {e}")
    
    def _signal_generation_loop(self):
        """Main signal generation loop running in background thread"""
        logger.info(f"Starting signal generation loop (interval: {self.signal_interval}s)")
        
        # Log strategy information once at startup
        logger.info(f"Active strategies: {', '.join(self.selected_strategies)}")
        logger.info(f"Monitoring symbols: {', '.join(self.selected_symbols)}")
        
        while self.running:
            try:
                current_time = datetime.now(timezone.utc)
                minute_key = current_time.strftime('%Y-%m-%d %H:%M')
                
                # Only show loop status when signals are generated
                signals_generated = 0
                
                for symbol in self.selected_symbols:
                    try:
                        # Get integrated data (historical + current WebSocket candle)
                        integrated_data = self._get_integrated_data(symbol)
                        
                        if integrated_data is not None and not integrated_data.empty:
                            # Generate signals using integrated data
                            signals = self.signal_integration.framework.generate_signals(
                                {symbol: integrated_data},
                                strategies=self.selected_strategies,
                                live_mode=True
                            )
                            
                            # Process generated signals
                            for signal in signals:
                                if signal and signal.signal_type != SignalType.FLAT:
                                    # Convert Signal object to dict format
                                    signal_dict = {
                                        'signal_type': signal.signal_type.name,
                                        'strategy': signal.strategy_name,
                                        'confidence': signal.confidence,
                                        'price': signal.entry_price,
                                        'reason': signal.reason
                                    }
                                    self._process_signal_from_dict(signal_dict, symbol, current_time)
                                    signals_generated += 1
                            
                    except Exception as e:
                        logger.error(f"Error generating signals for {symbol}: {e}")
                
                # Show interval summary (every 10 seconds)
                price_info = []
                for symbol in self.selected_symbols:
                    if symbol in self.latest_prices:
                        price = self.latest_prices[symbol]['price']
                        price_info.append(f"{symbol}: ${price:.2f}")
                
                prices_str = " | ".join(price_info) if price_info else "No prices"
                logger.info(f"🔄 {current_time.strftime('%H:%M:%S')}: {signals_generated} signals | {prices_str}")
                
                # Check if we need to show minute summary
                current_minute = current_time.strftime('%Y-%m-%d %H:%M')
                if current_minute != getattr(self, '_last_minute_summary', None):
                    self._last_minute_summary = current_minute
                    self._show_minute_summary(current_minute)
                
                # Wait for next interval
                time.sleep(self.signal_interval)
                
            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")
                time.sleep(self.signal_interval)
    
    def _prepare_signal_data(self, symbol: str, minute_key: str) -> Optional[pd.DataFrame]:
        """Prepare data for signal generation by combining historical and real-time data"""
        try:
            logger.debug(f"Preparing signal data for {symbol} at {minute_key}")
            
            # Start with historical data
            if symbol in self.historical_data:
                df = self.historical_data[symbol].copy()
                logger.debug(f"Historical data for {symbol}: {len(df)} rows")
            else:
                df = pd.DataFrame()
                logger.debug(f"No historical data for {symbol}")
            
            # Ensure timestamp column exists and is properly formatted
            if not df.empty:
                df = self._ensure_timestamp_column(df)
                logger.debug(f"After timestamp processing: {len(df)} rows, columns: {list(df.columns)}")
            
            # Add current minute data if available
            if (self.current_minute_data is not None and 
                symbol in self.current_minute_data and 
                minute_key in self.current_minute_data[symbol]):
                current_data = self.current_minute_data[symbol][minute_key]
                logger.debug(f"Adding current minute data for {symbol}: {current_data}")
                
                # Create a row for current minute
                current_row = pd.DataFrame([{
                    'timestamp': current_data['timestamp'],
                    'open': current_data['open'],
                    'high': current_data['high'],
                    'low': current_data['low'],
                    'close': current_data['close'],
                    'volume': current_data['volume']
                }])
                
                # Append to historical data
                if not df.empty:
                    df = pd.concat([df, current_row], ignore_index=True)
                else:
                    df = current_row
            else:
                logger.debug(f"No current minute data for {symbol} at {minute_key}")
            
            # Ensure we have enough data - reduced for test strategy
            if len(df) < 1:  # Minimum data requirement (reduced for testing)
                logger.debug(f"Not enough data for {symbol}: {len(df)} rows (need 1+)")
                return None
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            logger.debug(f"Final data for {symbol}: {len(df)} rows, columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing signal data for {symbol}: {e}")
            return None
    
    def _ensure_timestamp_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the dataframe has a proper timestamp column"""
        try:
            # If timestamp is already a column, use it
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            
            # If the index is datetime-like, use it as timestamp (most common case)
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.reset_index()
                df['timestamp'] = df['index']
                df = df.drop('index', axis=1)
                return df
            
            # If the index is not datetime but has a name, try to convert it
            if df.index.name and ('time' in str(df.index.name).lower() or 'date' in str(df.index.name).lower()):
                df = df.reset_index()
                df['timestamp'] = pd.to_datetime(df[df.index.name])
                df = df.drop(df.index.name, axis=1)
                return df
            
            # If timestamp is the index name, move it to a column
            if df.index.name == 'timestamp' or 'timestamp' in str(df.index.name):
                df = df.reset_index()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            
            # Look for other time-related columns
            time_columns = [col for col in df.columns if any(time_word in col.lower() for time_word in ['time', 'date', 'datetime', 'ts'])]
            if time_columns:
                df['timestamp'] = pd.to_datetime(df[time_columns[0]])
                return df
            
            # If no timestamp found, create one based on row number (fallback)
            logger.warning(f"No timestamp column found, using row index as fallback")
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='1min')
            return df
            
        except Exception as e:
            logger.error(f"Error ensuring timestamp column: {e}")
            return df
    
    def _process_signal(self, signal: Signal, symbol: str, timestamp: datetime):
        """Process a generated signal"""
        try:
            # Add to signal history
            signal_info = {
                'timestamp': timestamp,
                'symbol': symbol,
                'strategy': signal.strategy_name,
                'signal_type': signal.signal.name,  # Use name instead of value
                'confidence': signal.confidence,
                'reason': signal.reason,
                'price': signal.entry_price if hasattr(signal, 'entry_price') else None
            }
            
            self.signal_history.append(signal_info)
            self.signals_this_minute[timestamp.strftime('%Y-%m-%d %H:%M')].append(signal_info)
            self.total_signals_today += 1
            
            # Play sound notification
            self.sound_system.play_signal_sound(signal)
            
            # Send desktop notification
            self.notification_system.send_signal_notification(signal, symbol)
            
            # Log signal
            logger.info(f"SIGNAL: {signal.signal.name.upper()} {symbol} | {signal.strategy_name} | Confidence: {signal.confidence:.2f} | {signal.reason}")
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def _process_signal_from_dict(self, signal_dict: Dict, symbol: str, timestamp: datetime):
        """Process a signal from generate_live_signals (returns dict format)"""
        try:
            signal_type = signal_dict.get('signal_type', 'FLAT')
            strategy_name = signal_dict.get('strategy', 'Unknown')
            confidence = signal_dict.get('confidence', 0.0)
            price = signal_dict.get('price', 0.0)
            
            # Add to signal history
            signal_info = {
                'timestamp': timestamp,
                'symbol': symbol,
                'strategy': strategy_name,
                'signal_type': signal_type,
                'confidence': confidence,
                'reason': signal_dict.get('reason', 'Live signal'),
                'price': price
            }
            
            self.signal_history.append(signal_info)
            self.signals_this_minute[timestamp.strftime('%Y-%m-%d %H:%M')].append(signal_info)
            self.total_signals_today += 1
            
            # Play sound notification
            if self.sound_system:
                # Create a mock signal object for the sound system
                mock_signal = type('MockSignal', (), {
                    'signal_type': type('SignalType', (), {'name': signal_type})()
                })()
                self.sound_system.play_signal_sound(mock_signal)
            
            # Send desktop notification
            if self.notification_system:
                if signal_type in ['LONG', 'SHORT']:
                    # Create a mock signal object for the notification system
                    mock_signal = type('MockSignal', (), {
                        'signal_type': type('SignalType', (), {'name': signal_type})(),
                        'strategy_name': strategy_name,
                        'confidence': confidence,
                        'reason': signal_dict.get('reason', 'Live signal')
                    })()
                    self.notification_system.send_signal_notification(mock_signal, symbol)
            
            # Log signal
            logger.info(f"🔔 Signal: {signal_type} {symbol} | Strategy: {strategy_name} | Confidence: {confidence:.2f} | Price: ${price:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing signal from dict: {e}")
    
    def _get_integrated_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get integrated data combining historical data with current WebSocket candle"""
        try:
            # Load historical data
            hist_data = self.signal_integration.analysis_engine.load_symbol_data(symbol, days=30)
            if hist_data is None or hist_data.empty:
                return None
            
            # Ensure timestamp column exists
            hist_data = self._ensure_timestamp_column(hist_data)
            
            # Add current WebSocket candle if available
            current_time = datetime.now(timezone.utc)
            minute_key = current_time.strftime('%Y-%m-%d %H:%M')
            
            if (self.current_minute_data is not None and 
                symbol in self.current_minute_data and 
                minute_key in self.current_minute_data[symbol]):
                
                # Get current WebSocket candle
                ws_candle = self.current_minute_data[symbol][minute_key]
                
                # Create DataFrame row for current candle
                current_row = pd.DataFrame([{
                    'timestamp': ws_candle['timestamp'],
                    'open': ws_candle['open'],
                    'high': ws_candle['high'],
                    'low': ws_candle['low'],
                    'close': ws_candle['close'],
                    'volume': ws_candle['volume']
                }])
                
                # Append to historical data
                integrated_data = pd.concat([hist_data, current_row], ignore_index=True)
                integrated_data = integrated_data.sort_values('timestamp').reset_index(drop=True)
                
                # Set timestamp as index for signal framework
                integrated_data = integrated_data.set_index('timestamp')
                
                logger.debug(f"Integrated data for {symbol}: {len(hist_data)} hist + 1 live = {len(integrated_data)} total")
                return integrated_data
            
            # If no WebSocket data, return historical data with timestamp as index
            hist_data = hist_data.set_index('timestamp')
            return hist_data
            
        except Exception as e:
            logger.error(f"Error getting integrated data for {symbol}: {e}")
            return None
    
    def _show_minute_summary(self, minute_key: str):
        """Show total trades generated in the previous minute"""
        try:
            # Get signals for the previous minute
            previous_minute_signals = self.signals_this_minute.get(minute_key, [])
            total_signals = len(previous_minute_signals)
            
            if total_signals > 0:
                logger.info(f"📊 MINUTE SUMMARY ({minute_key}): {total_signals} total signals generated")
                # Show breakdown by symbol
                symbol_counts = {}
                for signal in previous_minute_signals:
                    symbol = signal['symbol']
                    symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                
                for symbol, count in symbol_counts.items():
                    logger.info(f"   {symbol}: {count} signals")
            else:
                logger.info(f"📊 MINUTE SUMMARY ({minute_key}): No signals generated")
                
        except Exception as e:
            logger.error(f"Error showing minute summary: {e}")
    
    def _display_minute_summary(self):
        """Display minute-by-minute signal summary"""
        try:
            current_time = datetime.now(timezone.utc)
            current_minute = current_time.strftime('%Y-%m-%d %H:%M')
            
            # Get signals for current minute
            current_signals = self.signals_this_minute.get(current_minute, [])
            
            if current_signals:
                print(f"\nSIGNALS THIS MINUTE ({current_minute}): {len(current_signals)} signals")
                print("-" * 60)
                
                for signal in current_signals:
                    print(f"  {signal['signal_type'].upper():12} {signal['symbol']:6} | {signal['strategy']:25} | Conf: {signal['confidence']:.2f}")
                    print(f"    Reason: {signal['reason']}")
                    print()
            else:
                # Only show status if no signals this minute
                print(f"\nStatus: No signals this minute | Total today: {self.total_signals_today} | Next check: {self.signal_interval}s | Time: {current_time.strftime('%H:%M:%S UTC')}")
            
        except Exception as e:
            logger.error(f"Error displaying minute summary: {e}")
    
    def start(self):
        """Start the interactive trading module"""
        if self.running:
            logger.warning("Interactive trading module already running")
            return
        
        # Setup interactive session
        if not self.setup_interactive_session():
            logger.error("Failed to setup interactive session")
            return
        
        logger.info("🚀 Starting Interactive Trading Module")
        
        # Load historical data
        self._load_historical_data()
        
        # Check if we have data for signal generation
        print(f"📊 Historical data loaded for symbols: {list(self.historical_data.keys())}")
        for symbol, df in self.historical_data.items():
            print(f"   {symbol}: {len(df)} rows")
        
        # Initialize signal integration
        self._initialize_signal_integration()
        
        # Initialize WebSocket feed
        ws_symbols = [f"{s}-USD" for s in self.selected_symbols]
        self.ws_feed = WebSocketPriceFeed(ws_symbols, self._on_price_update)
        
        # Start data collection
        self.running = True
        self.ws_feed.start()
        
        # Start signal generation in background
        self.signal_thread = threading.Thread(target=self._signal_generation_loop, daemon=True)
        self.signal_thread.start()
        
        logger.info("✅ Interactive Trading Module started")
        logger.info("Press Ctrl+C to stop")
        
        try:
            # Main display loop
            while self.running:
                self._display_minute_summary()
                time.sleep(60)  # Update display every minute
                
        except KeyboardInterrupt:
            logger.info("Stopping Interactive Trading Module...")
            self.stop()
    
    def stop(self):
        """Stop the interactive trading module"""
        self.running = False
        
        if self.ws_feed:
            self.ws_feed.stop()
        
        logger.info("✅ Interactive Trading Module stopped")

def main():
    """Main entry point for interactive trading module"""
    try:
        module = InteractiveTradingModule()
        module.start()
    except Exception as e:
        logger.error(f"Failed to start interactive trading module: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
