#!/usr/bin/env python3
"""
Integrated Live Trading System
==============================

Combines real-time data collection with sophisticated signal generation
and live trade execution. This is the main live trading system that should
be used instead of the separate components.

Author: Quantitative Strategy Designer
Date: 2025-01-28
"""

import asyncio
import logging
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
import yaml
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_ingestion.websocket_price_feed import WebSocketPriceFeed
from src.crypto_signal_integration import CryptoSignalIntegration
from src.crypto_analysis_engine import CryptoAnalysisEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/integrated_live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegratedLiveTrading:
    """
    Integrated live trading system that combines:
    1. Real-time data collection (1-minute candles)
    2. Sophisticated signal generation
    3. Live trade execution and logging
    """
    
    def __init__(self, data_dir: str = "data", symbols: List[str] = None, api_keys: Dict = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.symbols = symbols or ['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI']
        
        # Initialize components
        self.api_keys = api_keys or {}
        self.ws_feed = None
        self.signal_integration = None
        self.analysis_engine = CryptoAnalysisEngine()
        
        # Trading state
        self.portfolio_state = {
            "current_positions": {},  # symbol -> position_size
            "cash_balance": 100000.0,
            "total_value": 100000.0,
            "cumulative_pnl": 0.0,
            "last_update": datetime.now().isoformat()
        }
        
        # Data collection state
        self.current_minute_data = {}
        self.running = False
        self.signal_generation_interval = 60  # 1 minute
        
        # Trade logging
        self.trades_log = self.data_dir / "live_trades.csv"
        self.portfolio_file = self.data_dir / "portfolio_state.json"
        
        # Load existing state
        self._load_portfolio_state()
        self._initialize_trades_log()
        
    def _load_portfolio_state(self):
        """Load existing portfolio state"""
        if self.portfolio_file.exists():
            try:
                with open(self.portfolio_file, 'r') as f:
                    loaded_state = json.load(f)
                
                # Ensure all required keys are present
                default_state = {
                    "current_positions": {},
                    "cash_balance": 100000.0,
                    "total_value": 100000.0,
                    "cumulative_pnl": 0.0,
                    "last_update": datetime.now().isoformat()
                }
                
                # Merge loaded state with defaults
                for key, default_value in default_state.items():
                    if key not in loaded_state:
                        loaded_state[key] = default_value
                
                self.portfolio_state = loaded_state
                logger.info("Loaded existing portfolio state")
            except Exception as e:
                logger.warning(f"Could not load portfolio state: {e}")
                # Keep the default state if loading fails
    
    def _save_portfolio_state(self):
        """Save current portfolio state"""
        try:
            self.portfolio_state["last_update"] = datetime.now().isoformat()
            with open(self.portfolio_file, 'w') as f:
                json.dump(self.portfolio_state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save portfolio state: {e}")
    
    def _initialize_trades_log(self):
        """Initialize trades log file"""
        if not self.trades_log.exists():
            df = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'signal_type', 'price', 'confidence', 
                'reason', 'strategy', 'simulated_pnl', 'cumulative_pnl', 
                'position_size', 'trade_id', 'action'
            ])
            df.to_csv(self.trades_log, index=False)
            logger.info(f"Initialized trades log: {self.trades_log}")
    
    def _on_price_update(self, symbol: str, price_data: dict):
        """Handle real-time price updates and build 1-minute candles"""
        try:
            symbol = symbol.replace('-USD', '')
            if symbol not in self.symbols:
                return
                
            current_time = price_data['timestamp']
            price = price_data['price']
            
            # Round to current minute and ensure timezone consistency
            minute_time = current_time.replace(second=0, microsecond=0)
            # Ensure timezone is UTC
            if minute_time.tzinfo is None:
                minute_time = minute_time.replace(tzinfo=timezone.utc)
            elif minute_time.tzinfo != timezone.utc:
                minute_time = minute_time.astimezone(timezone.utc)
            
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
            
            # Ensure the index is timezone-aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            # File path
            file_path = self.data_dir / f"{symbol}_1m_historical.parquet"
            
            # Load existing data
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                existing_df.index = pd.to_datetime(existing_df.index)
                # Ensure timezone consistency
                if existing_df.index.tz is None:
                    existing_df.index = existing_df.index.tz_localize('UTC')
                
                # Combine and remove duplicates
                combined_df = pd.concat([existing_df, df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.sort_index()
                
                # Keep only last 7 days (handle timezone properly)
                cutoff = datetime.now(timezone.utc) - timedelta(days=7)
                # Ensure cutoff is timezone-aware
                if combined_df.index.tz is None:
                    combined_df.index = combined_df.index.tz_localize('UTC')
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
    
    def _initialize_signal_integration(self):
        """Initialize signal integration with available strategies"""
        try:
            # Get available strategies
            from src.crypto_trading_strategies import CryptoTradingStrategies
            strategies = CryptoTradingStrategies()
            available_strategies = list(strategies.strategies.keys())
            
            # Use all available strategies for live trading
            selected_strategies = available_strategies
            logger.info(f"Using all {len(selected_strategies)} strategies for live trading: {', '.join(selected_strategies)}")
            
            # Initialize signal integration
            self.signal_integration = CryptoSignalIntegration(
                data_dir=str(self.data_dir),
                selected_strategies=selected_strategies
            )
            
            logger.info("Signal integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing signal integration: {e}")
            self.signal_integration = None
    
    def _generate_and_execute_signals(self):
        """Generate signals and execute trades"""
        try:
            if not self.signal_integration:
                logger.warning("Signal integration not initialized")
                return
            
            # Generate signals for all symbols
            logger.info("Generating signals...")
            signals = self.signal_integration.generate_signals(
                symbols=self.symbols,
                days=1095  # Use 3 years of data for maximum strategy effectiveness
            )
            
            if not signals:
                logger.info("No signals generated")
                return
            
            # Process each signal
            for signal in signals:
                self._execute_signal(signal)
                
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
    
    def _execute_signal(self, signal: Dict):
        """Execute a trading signal"""
        try:
            symbol = signal['symbol']
            signal_type = signal['signal_type']
            confidence = signal['confidence']
            price = signal['entry_price']
            strategy = signal['strategy']
            reason = signal['reason']
            
            # Generate trade ID
            trade_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{strategy}"
            
            # Determine trade action based on signal and current position
            current_position = self.portfolio_state["current_positions"].get(symbol, 0.0)
            trade_action = 'HOLD'
            position_size = 0.0
            simulated_pnl = 0.0
            
            if signal_type == 'LONG':
                if current_position <= 0:  # Not currently long
                    trade_action = 'BUY'
                    # Calculate position size (10% of portfolio, scaled by confidence)
                    position_value = self.portfolio_state["cash_balance"] * 0.1 * confidence
                    position_size = position_value / price
                else:
                    trade_action = 'HOLD'  # Already long
                    
            elif signal_type == 'SHORT':
                if current_position > 0:  # Currently long, need to sell
                    trade_action = 'SELL'
                    position_size = current_position
                    # Calculate P&L
                    cost_basis = current_position * self.portfolio_state.get(f"{symbol}_avg_price", price)
                    simulated_pnl = position_size * price - cost_basis
                else:
                    trade_action = 'HOLD'  # No position to close
            
            # Execute trade if action is not HOLD
            if trade_action != 'HOLD':
                self._execute_trade(
                    symbol=symbol,
                    action=trade_action,
                    price=price,
                    position_size=position_size,
                    confidence=confidence,
                    strategy=strategy,
                    reason=reason,
                    trade_id=trade_id,
                    simulated_pnl=simulated_pnl
                )
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    def _execute_trade(self, symbol: str, action: str, price: float, position_size: float, 
                      confidence: float, strategy: str, reason: str, trade_id: str, 
                      simulated_pnl: float = 0.0):
        """Execute a trade and update portfolio"""
        try:
            current_position = self.portfolio_state["current_positions"].get(symbol, 0.0)
            
            if action == 'BUY' and position_size > 0:
                # Buy position
                cost = position_size * price
                if cost <= self.portfolio_state["cash_balance"]:
                    self.portfolio_state["current_positions"][symbol] = current_position + position_size
                    self.portfolio_state["cash_balance"] -= cost
                    self.portfolio_state[f"{symbol}_avg_price"] = price
                    logger.info(f"BUY {symbol}: {position_size:.6f} @ ${price:.2f} (${cost:.2f})")
                else:
                    logger.warning(f"Insufficient cash for {symbol} trade: ${cost:.2f} > ${self.portfolio_state['cash_balance']:.2f}")
                    return
                    
            elif action == 'SELL' and current_position > 0:
                # Sell position
                proceeds = current_position * price
                self.portfolio_state["current_positions"][symbol] = 0.0
                self.portfolio_state["cash_balance"] += proceeds
                self.portfolio_state["cumulative_pnl"] += simulated_pnl
                logger.info(f"SELL {symbol}: {current_position:.6f} @ ${price:.2f} (${proceeds:.2f}) P&L: ${simulated_pnl:.2f}")
            
            # Update total portfolio value
            self._update_portfolio_value()
            
            # Log trade
            self._log_trade({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal_type': action,
                'price': price,
                'confidence': confidence,
                'reason': reason,
                'strategy': strategy,
                'simulated_pnl': simulated_pnl,
                'cumulative_pnl': self.portfolio_state["cumulative_pnl"],
                'position_size': position_size,
                'trade_id': trade_id,
                'action': action
            })
            
            # Save portfolio state
            self._save_portfolio_state()
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _update_portfolio_value(self):
        """Update total portfolio value"""
        try:
            total_value = self.portfolio_state["cash_balance"]
            
            # Add value of all positions
            for symbol, position_size in self.portfolio_state["current_positions"].items():
                if position_size > 0:
                    # Get current price from latest data
                    try:
                        file_path = self.data_dir / f"{symbol}_1m_historical.parquet"
                        if file_path.exists():
                            df = pd.read_parquet(file_path)
                            if not df.empty:
                                current_price = df['close'].iloc[-1]
                                total_value += position_size * current_price
                    except Exception as e:
                        logger.warning(f"Could not get current price for {symbol}: {e}")
            
            self.portfolio_state["total_value"] = total_value
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def _log_trade(self, trade_data: Dict):
        """Log trade to CSV file"""
        try:
            df = pd.DataFrame([trade_data])
            
            # Append to existing file
            if self.trades_log.exists():
                df.to_csv(self.trades_log, mode='a', header=False, index=False)
            else:
                df.to_csv(self.trades_log, index=False)
            
            logger.info(f"Logged trade: {trade_data['action']} {trade_data['symbol']} @ ${trade_data['price']:.2f}")
            
        except Exception as e:
            logger.error(f"Could not log trade: {e}")
    
    def _signal_generation_loop(self):
        """Background loop for signal generation"""
        while self.running:
            try:
                time.sleep(self.signal_generation_interval)
                if self.running:
                    logger.info("🔄 Generating signals...")
                    self._generate_and_execute_signals()
            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")
                time.sleep(30)  # Wait 30 seconds on error
    
    def start(self):
        """Start the integrated live trading system"""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("🚀 Starting Integrated Live Trading System")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Data directory: {self.data_dir}")
        
        # Initialize signal integration
        self._initialize_signal_integration()
        
        # Initialize WebSocket feed
        ws_symbols = [f"{s}-USD" for s in self.symbols]
        self.ws_feed = WebSocketPriceFeed(ws_symbols, self._on_price_update)
        
        # Start data collection
        self.running = True
        self.ws_feed.start()
        
        # Start signal generation in background
        signal_thread = threading.Thread(target=self._signal_generation_loop, daemon=True)
        signal_thread.start()
        
        logger.info("✅ Integrated Live Trading System started")
        logger.info(f"Signal generation interval: {self.signal_generation_interval} seconds")
        logger.info("Press Ctrl+C to stop")
        
        try:
            # Keep running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping Integrated Live Trading System...")
            self.stop()
    
    def stop(self):
        """Stop the integrated live trading system"""
        self.running = False
        
        if self.ws_feed:
            self.ws_feed.stop()
        
        # Save any remaining minute data
        for symbol, data in self.current_minute_data.items():
            self._save_minute_data(symbol, data)
        
        # Save final portfolio state
        self._save_portfolio_state()
        
        logger.info("✅ Integrated Live Trading System stopped")
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        self._update_portfolio_value()
        
        # Ensure all required keys exist
        if "current_positions" not in self.portfolio_state:
            self.portfolio_state["current_positions"] = {}
        if "cash_balance" not in self.portfolio_state:
            self.portfolio_state["cash_balance"] = 100000.0
        if "total_value" not in self.portfolio_state:
            self.portfolio_state["total_value"] = 100000.0
        if "cumulative_pnl" not in self.portfolio_state:
            self.portfolio_state["cumulative_pnl"] = 0.0
        if "last_update" not in self.portfolio_state:
            self.portfolio_state["last_update"] = datetime.now().isoformat()
        
        return {
            "cash_balance": self.portfolio_state["cash_balance"],
            "total_value": self.portfolio_state["total_value"],
            "cumulative_pnl": self.portfolio_state["cumulative_pnl"],
            "positions": self.portfolio_state["current_positions"],
            "last_update": self.portfolio_state["last_update"]
        }
    
    def get_recent_trades(self, limit: int = 10) -> pd.DataFrame:
        """Get recent trades from log"""
        if not self.trades_log.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.trades_log)
            return df.tail(limit)
        except Exception as e:
            logger.error(f"Could not read trades: {e}")
            return pd.DataFrame()

def main():
    """Main function"""
    print("🚀 Integrated Live Trading System")
    print("=" * 50)
    print("This system combines:")
    print("  • Real-time data collection (1-minute candles)")
    print("  • Sophisticated signal generation (every 1 minute)")
    print("  • Live trade execution and logging")
    print("=" * 50)
    
    # Load API keys
    try:
        with open('config/api_keys.yaml', 'r') as f:
            api_keys = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load API keys: {e}")
        api_keys = {}
    
    # Initialize and start system
    trading_system = IntegratedLiveTrading(api_keys=api_keys)
    
    # Show initial portfolio state
    portfolio = trading_system.get_portfolio_summary()
    print(f"\n💰 Initial Portfolio:")
    print(f"   Cash: ${portfolio['cash_balance']:,.2f}")
    print(f"   Total Value: ${portfolio['total_value']:,.2f}")
    print(f"   Cumulative P&L: ${portfolio['cumulative_pnl']:,.2f}")
    print(f"   Positions: {portfolio['positions']}")
    print()
    
    # Start the system
    trading_system.start()

if __name__ == "__main__":
    main()
