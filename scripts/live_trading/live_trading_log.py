"""
Live Trading Log System
Automatically tracks BTC signals and simulated PnL from continuous data collection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import os

from crypto_signal_integration import CryptoSignalIntegration
from crypto_analysis_engine import CryptoAnalysisEngine
from data_ingestion.realtime_fusion_system import RealTimeFusionSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    timestamp: str
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    price: float
    confidence: float
    reason: str
    strategy: str
    simulated_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    position_size: float = 0.0
    trade_id: str = ""

class LiveTradingLog:
    def __init__(self, data_dir: str = "data", log_file: str = "live_trades.csv", api_keys: Dict = None):
        self.data_dir = Path(data_dir)
        self.log_file = self.data_dir / log_file
        self.portfolio_file = self.data_dir / "portfolio_state.json"
        
        # Initialize signal integration with available strategies
        from src.crypto_trading_strategies import CryptoTradingStrategies
        strategies = CryptoTradingStrategies()
        available_strategies = list(strategies.strategies.keys())
        print(f"Available strategies for live trading: {', '.join(available_strategies)}")
        
        # Use first two strategies by default
        selected_strategies = available_strategies[:2] if len(available_strategies) >= 2 else available_strategies
        print(f"Using strategies: {', '.join(selected_strategies)}")
        
        self.signal_integration = CryptoSignalIntegration(selected_strategies=selected_strategies)
        self.analysis_engine = CryptoAnalysisEngine()
        
        # Portfolio state
        self.portfolio_state = {
            "current_position": 0.0,  # BTC amount held
            "cash_balance": 100000.0,  # Starting cash
            "total_value": 100000.0,
            "cumulative_pnl": 0.0,
            "last_trade_price": 0.0,
            "last_signal": "HOLD",
            "last_update": datetime.now().isoformat()
        }
        
        # Load existing portfolio state
        self._load_portfolio_state()
        
        # Initialize real-time fusion system (for signal generation only)
        self.fusion_system = RealTimeFusionSystem(data_dir=self.data_dir, symbols=['BTC'])
        
        # Start real-time fusion system (this will only generate signals, not collect data)
        self.fusion_system.start(self._on_signal_generated)
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize log file if it doesn't exist
        self._initialize_log_file()
    
    def _on_signal_generated(self, symbol: str, signal: Dict):
        """Callback for when fusion system generates a signal"""
        try:
            # Execute trade based on signal
            trade = self._execute_trade(signal, signal['price'])
            
            # Log the trade
            self._log_trade(trade)
            
            # Update portfolio state
            self._save_portfolio_state()
            
            logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Signal: {trade.signal_type} @ ${trade.price:.2f} | PnL: ${trade.simulated_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error handling signal for {symbol}: {e}")
    
    def _initialize_log_file(self):
        """Initialize CSV log file with headers if it doesn't exist"""
        if not self.log_file.exists():
            df = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'signal_type', 'price', 'confidence', 
                'reason', 'strategy', 'simulated_pnl', 'cumulative_pnl', 
                'position_size', 'trade_id'
            ])
            df.to_csv(self.log_file, index=False)
            logger.info(f"Initialized trading log: {self.log_file}")
    
    def _load_portfolio_state(self):
        """Load existing portfolio state from file"""
        if self.portfolio_file.exists():
            try:
                with open(self.portfolio_file, 'r') as f:
                    self.portfolio_state = json.load(f)
                logger.info("Loaded existing portfolio state")
            except Exception as e:
                logger.warning(f"Could not load portfolio state: {e}")
    
    def _save_portfolio_state(self):
        """Save current portfolio state to file"""
        try:
            with open(self.portfolio_file, 'w') as f:
                json.dump(self.portfolio_state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save portfolio state: {e}")
    
    def _calculate_position_size(self, signal_confidence: float, current_price: float) -> float:
        """Calculate position size based on confidence and available cash"""
        # Simple position sizing: use 10% of portfolio per trade, scaled by confidence
        base_allocation = 0.1  # 10% of portfolio
        confidence_multiplier = signal_confidence
        max_position_value = self.portfolio_state["cash_balance"] * base_allocation * confidence_multiplier
        
        # Convert to BTC amount
        position_size = max_position_value / current_price
        return position_size
    
    def _execute_trade(self, signal_data: Dict, current_price: float) -> Trade:
        """Execute a simulated trade based on signal"""
        signal_type = signal_data.get('signal_type', 'HOLD')
        confidence = signal_data.get('confidence', 0.0)
        reason = signal_data.get('reason', '')
        strategy = signal_data.get('strategy', '')
        
        # Generate trade ID
        trade_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{strategy}"
        
        # Convert LONG/SHORT signals to BUY/SELL actions
        trade_action = 'HOLD'
        position_size = 0.0
        simulated_pnl = 0.0
        
        if signal_type == 'LONG':
            if self.portfolio_state["current_position"] <= 0:  # Not currently long
                trade_action = 'BUY'
                position_size = self._calculate_position_size(confidence, current_price)
            else:
                trade_action = 'HOLD'  # Already long, just hold
                
        elif signal_type == 'SHORT':
            # For testing: SHORT signals always execute SELL trades
            trade_action = 'SELL'
            if self.portfolio_state["current_position"] > 0:  # Currently long, need to sell
                # Calculate profit/loss from current position
                position_value = self.portfolio_state["current_position"] * current_price
                cost_basis = self.portfolio_state["current_position"] * self.portfolio_state["last_trade_price"]
                simulated_pnl = position_value - cost_basis
                position_size = self.portfolio_state["current_position"]  # Sell all current position
            else:
                # If we're flat, just log the SELL signal (no actual position change)
                position_size = 0.0
        
        # Update portfolio state based on trade action
        if trade_action == 'BUY' and position_size > 0:
            cost = position_size * current_price
            self.portfolio_state["current_position"] += position_size
            self.portfolio_state["cash_balance"] -= cost
            self.portfolio_state["last_trade_price"] = current_price
            
        elif trade_action == 'SELL' and self.portfolio_state["current_position"] > 0:
            proceeds = self.portfolio_state["current_position"] * current_price
            self.portfolio_state["cash_balance"] += proceeds
            self.portfolio_state["current_position"] = 0.0
        
        # Update cumulative PnL
        self.portfolio_state["cumulative_pnl"] += simulated_pnl
        self.portfolio_state["total_value"] = (
            self.portfolio_state["cash_balance"] + 
            self.portfolio_state["current_position"] * current_price
        )
        self.portfolio_state["last_signal"] = signal_type
        self.portfolio_state["last_update"] = datetime.now().isoformat()
        
        # Create trade record
        trade = Trade(
            timestamp=datetime.now().isoformat(),
            symbol="BTC",
            signal_type=trade_action,  # Use the actual trade action
            price=current_price,
            confidence=confidence,
            reason=reason,
            strategy=strategy,
            simulated_pnl=simulated_pnl,
            cumulative_pnl=self.portfolio_state["cumulative_pnl"],
            position_size=position_size,
            trade_id=trade_id
        )
        
        return trade
    
    def _log_trade(self, trade: Trade):
        """Log trade to CSV file"""
        try:
            # Convert trade to dict and append to CSV
            trade_dict = asdict(trade)
            df = pd.DataFrame([trade_dict])
            
            # Append to existing file
            if self.log_file.exists():
                df.to_csv(self.log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.log_file, index=False)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[{timestamp}] Logged trade: {trade.signal_type} @ ${trade.price:.2f} | PnL: ${trade.simulated_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Could not log trade: {e}")
    
    def generate_and_log_signals(self, days: int = 1) -> List[Trade]:
        """Generate signals and log any trades using fusion system"""
        try:
            # The fusion system now handles signal generation automatically
            # This method is kept for compatibility but signals are generated via callback
            logger.info("Signal generation is now handled by the fusion system")
            return []
        
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        return {
            "cash_balance": self.portfolio_state["cash_balance"],
            "btc_position": self.portfolio_state["current_position"],
            "total_value": self.portfolio_state["total_value"],
            "cumulative_pnl": self.portfolio_state["cumulative_pnl"],
            "last_signal": self.portfolio_state["last_signal"],
            "last_update": self.portfolio_state["last_update"]
        }
    
    def get_recent_trades(self, limit: int = 10) -> pd.DataFrame:
        """Get recent trades from log"""
        if not self.log_file.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.log_file)
            return df.tail(limit)
        except Exception as e:
            logger.error(f"Could not read trades: {e}")
            return pd.DataFrame()
    
    def get_daily_summary(self, date: str = None) -> Dict:
        """Get daily trading summary"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        if not self.log_file.exists():
            return {"date": date, "trades": 0, "pnl": 0.0}
        
        try:
            df = pd.read_csv(self.log_file)
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            target_date = pd.to_datetime(date).date()
            
            daily_trades = df[df['date'] == target_date]
            
            return {
                "date": date,
                "trades": len(daily_trades),
                "pnl": daily_trades['simulated_pnl'].sum() if not daily_trades.empty else 0.0,
                "signals": daily_trades['signal_type'].tolist() if not daily_trades.empty else []
            }
        except Exception as e:
            logger.error(f"Could not generate daily summary: {e}")
            return {"date": date, "trades": 0, "pnl": 0.0}

def main():
    """Main function to run live trading log"""
    print("🚀 Starting Live Trading Log System")
    print("=" * 50)
    
    # Initialize trading log
    trading_log = LiveTradingLog()
    
    # Show current portfolio state
    portfolio = trading_log.get_portfolio_summary()
    print(f"💰 Portfolio Status:")
    print(f"   Cash: ${portfolio['cash_balance']:,.2f}")
    print(f"   BTC Position: {portfolio['btc_position']:.6f} BTC")
    print(f"   Total Value: ${portfolio['total_value']:,.2f}")
    print(f"   Cumulative PnL: ${portfolio['cumulative_pnl']:,.2f}")
    print(f"   Last Signal: {portfolio['last_signal']}")
    print()
    
    # Generate and log signals
    print("📊 Generating signals...")
    trades = trading_log.generate_and_log_signals()
    
    if trades:
        print(f"✅ Executed {len(trades)} trades:")
        for trade in trades:
            print(f"   {trade.signal_type} @ ${trade.price:.2f} | PnL: ${trade.simulated_pnl:.2f}")
    else:
        print("ℹ️  No new trades executed")
    
    print()
    
    # Show recent trades
    recent_trades = trading_log.get_recent_trades(5)
    if not recent_trades.empty:
        print("📈 Recent Trades:")
        for _, trade in recent_trades.iterrows():
            print(f"   {trade['timestamp'][:19]} | {trade['signal_type']} @ ${trade['price']:.2f} | PnL: ${trade['simulated_pnl']:.2f}")
    
    print()
    print("✅ Live trading log updated successfully!")
    print(f"📁 Log file: {trading_log.log_file}")
    print(f"💼 Portfolio state: {trading_log.portfolio_file}")

if __name__ == "__main__":
    main()
