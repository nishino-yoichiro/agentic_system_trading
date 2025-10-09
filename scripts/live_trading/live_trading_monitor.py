#!/usr/bin/env python3
"""
Live Trading Monitor
===================

Real-time monitoring dashboard for the integrated live trading system.
Shows data collection, signal generation, and detailed signal analysis.

Author: Quantitative Strategy Designer
Date: 2025-01-28
"""

import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.crypto_signal_integration import CryptoSignalIntegration
from src.crypto_analysis_engine import CryptoAnalysisEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveTradingMonitor:
    """
    Real-time monitor for the live trading system
    """
    
    def __init__(self, data_dir: str = "data", symbols: List[str] = None):
        self.data_dir = Path(data_dir)
        self.symbols = symbols or ['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI']
        
        # Initialize components
        self.signal_integration = None
        self.analysis_engine = CryptoAnalysisEngine()
        
        # Monitoring state
        self.last_data_check = {}
        self.last_signals = {}
        self.monitoring_active = True
        
        # Initialize signal integration
        self._initialize_signal_integration()
    
    def _initialize_signal_integration(self):
        """Initialize signal integration for monitoring"""
        try:
            # Get available strategies
            from src.crypto_trading_strategies import CryptoTradingStrategies
            strategies = CryptoTradingStrategies()
            available_strategies = list(strategies.strategies.keys())
            
            # Use all available strategies for monitoring
            selected_strategies = available_strategies
            logger.info(f"Monitoring all {len(selected_strategies)} strategies: {', '.join(selected_strategies)}")
            
            # Initialize signal integration
            self.signal_integration = CryptoSignalIntegration(
                data_dir=str(self.data_dir),
                selected_strategies=selected_strategies
            )
            
            logger.info("Signal integration initialized for monitoring")
            
        except Exception as e:
            logger.error(f"Error initializing signal integration: {e}")
            self.signal_integration = None
    
    def get_latest_data_summary(self) -> Dict:
        """Get summary of latest data for all symbols"""
        summary = {}
        
        for symbol in self.symbols:
            try:
                file_path = self.data_dir / f"{symbol}_1m_historical.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    df.index = pd.to_datetime(df.index)
                    
                    if not df.empty:
                        latest_time = df.index[-1]
                        latest_price = df['close'].iloc[-1]
                        data_age_minutes = (datetime.now(timezone.utc) - latest_time).total_seconds() / 60
                        
                        summary[symbol] = {
                            'latest_time': latest_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                            'latest_price': latest_price,
                            'data_age_minutes': round(data_age_minutes, 1),
                            'total_points': len(df),
                            'last_5_prices': df['close'].tail(5).tolist(),
                            'last_5_times': [t.strftime('%H:%M:%S') for t in df.tail(5).index],
                            'is_fresh': data_age_minutes < 2  # Fresh if less than 2 minutes old
                        }
                    else:
                        summary[symbol] = {'error': 'No data available'}
                else:
                    summary[symbol] = {'error': 'File not found'}
                    
            except Exception as e:
                summary[symbol] = {'error': str(e)}
        
        return summary
    
    def get_detailed_signals(self) -> Dict:
        """Get detailed signal analysis for all symbols"""
        if not self.signal_integration:
            return {'error': 'Signal integration not available'}
        
        try:
            # Generate signals for all symbols
            signals = self.signal_integration.generate_signals(
                symbols=self.symbols,
                days=1095  # Use 3 years of data for maximum strategy effectiveness
            )
            
            # Group signals by symbol and strategy
            signal_analysis = {}
            
            for symbol in self.symbols:
                symbol_signals = [s for s in signals if s['symbol'] == symbol]
                
                if symbol_signals:
                    signal_analysis[symbol] = {
                        'has_signals': True,
                        'signals': symbol_signals,
                        'signal_count': len(symbol_signals)
                    }
                else:
                    # Analyze why no signals were generated
                    signal_analysis[symbol] = {
                        'has_signals': False,
                        'reason': self._analyze_no_signals(symbol)
                    }
            
            return signal_analysis
            
        except Exception as e:
            return {'error': f'Error generating signals: {e}'}
    
    def _analyze_no_signals(self, symbol: str) -> str:
        """Analyze why no signals were generated for a symbol"""
        try:
            # Check if we have enough data
            file_path = self.data_dir / f"{symbol}_1m_historical.parquet"
            if not file_path.exists():
                return "No data file found"
            
            df = pd.read_parquet(file_path)
            if df.empty:
                return "Data file is empty"
            
            if len(df) < 50:
                return f"Insufficient data ({len(df)} points, need 50+)"
            
            # Check data freshness
            latest_time = df.index[-1]
            data_age_minutes = (datetime.now(timezone.utc) - latest_time).total_seconds() / 60
            
            if data_age_minutes > 10:
                return f"Data too old ({data_age_minutes:.1f} minutes)"
            
            # Check if strategies are configured
            if not self.signal_integration:
                return "Signal integration not available"
            
            # Check strategy-specific reasons
            strategy_reasons = []
            for strategy_name in self.signal_integration.framework.strategies.keys():
                if symbol.lower() in strategy_name.lower():
                    strategy = self.signal_integration.framework.strategies[strategy_name]
                    config = strategy['config']
                    
                    # Check confidence threshold
                    if strategy.get('sharpe', 0) < 0:
                        strategy_reasons.append(f"{strategy_name}: Negative Sharpe ratio")
                    
                    # Check regime filters
                    if config.regime_filters:
                        # This would need more detailed analysis
                        strategy_reasons.append(f"{strategy_name}: Regime filters not met")
            
            if strategy_reasons:
                return "; ".join(strategy_reasons)
            
            return "No specific reason identified - may be normal market conditions"
            
        except Exception as e:
            return f"Analysis error: {e}"
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        try:
            portfolio_file = self.data_dir / "portfolio_state.json"
            if portfolio_file.exists():
                with open(portfolio_file, 'r') as f:
                    portfolio = json.load(f)
                
                # Ensure all required keys exist
                default_keys = {
                    "current_positions": {},
                    "cash_balance": 100000.0,
                    "total_value": 100000.0,
                    "cumulative_pnl": 0.0,
                    "last_update": datetime.now().isoformat()
                }
                
                for key, default_value in default_keys.items():
                    if key not in portfolio:
                        portfolio[key] = default_value
                
                return portfolio
            else:
                return {
                    "current_positions": {},
                    "cash_balance": 100000.0,
                    "total_value": 100000.0,
                    "cumulative_pnl": 0.0,
                    "last_update": "No portfolio file found"
                }
        except Exception as e:
            return {"error": f"Could not load portfolio: {e}"}
    
    def get_recent_trades(self, limit: int = 5) -> List[Dict]:
        """Get recent trades"""
        try:
            trades_file = self.data_dir / "live_trades.csv"
            if trades_file.exists():
                # Try to read with error handling for malformed CSV
                try:
                    df = pd.read_csv(trades_file)
                except pd.errors.ParserError as e:
                    logger.warning(f"CSV parsing error: {e}")
                    # Try to read with more flexible parsing
                    df = pd.read_csv(trades_file, on_bad_lines='skip')
                
                if not df.empty:
                    # Convert to list of dicts and get recent trades
                    recent_trades = df.tail(limit).to_dict('records')
                    return recent_trades
            return []
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    def display_monitor(self):
        """Display the monitoring dashboard"""
        print("\n" + "="*80)
        print("🚀 LIVE TRADING MONITOR")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()
        
        # Data Collection Status
        print("📊 DATA COLLECTION STATUS")
        print("-" * 40)
        data_summary = self.get_latest_data_summary()
        
        for symbol, data in data_summary.items():
            if 'error' in data:
                print(f"❌ {symbol}: {data['error']}")
            else:
                status = "🟢 FRESH" if data['is_fresh'] else "🟡 STALE"
                print(f"{status} {symbol}: ${data['latest_price']:.2f} ({data['data_age_minutes']}m ago)")
                print(f"    Latest: {data['latest_time']} | Points: {data['total_points']}")
                if data['last_5_prices']:
                    prices_str = " → ".join([f"${p:.2f}" for p in data['last_5_prices']])
                    print(f"    Last 5 prices: {prices_str}")
        print()
        
        # Signal Generation Status
        print("🎯 SIGNAL GENERATION STATUS")
        print("-" * 40)
        signal_analysis = self.get_detailed_signals()
        
        if 'error' in signal_analysis:
            print(f"❌ Error: {signal_analysis['error']}")
        else:
            for symbol, analysis in signal_analysis.items():
                if analysis['has_signals']:
                    print(f"🟢 {symbol}: {analysis['signal_count']} signals generated")
                    for signal in analysis['signals']:
                        print(f"    📈 {signal['strategy']}: {signal['signal_type']} @ ${signal['entry_price']:.2f}")
                        print(f"        Confidence: {signal['confidence']:.2f} | Reason: {signal['reason']}")
                else:
                    print(f"🔴 {symbol}: No signals")
                    print(f"    Reason: {analysis['reason']}")
        print()
        
        # Portfolio Status
        print("💰 PORTFOLIO STATUS")
        print("-" * 40)
        portfolio = self.get_portfolio_status()
        
        if 'error' in portfolio:
            print(f"❌ Error: {portfolio['error']}")
        else:
            print(f"💵 Cash: ${portfolio['cash_balance']:,.2f}")
            print(f"📈 Total Value: ${portfolio['total_value']:,.2f}")
            print(f"📊 P&L: ${portfolio['cumulative_pnl']:,.2f}")
            print(f"🕒 Last Update: {portfolio['last_update']}")
            
            if portfolio['current_positions']:
                print("📋 Current Positions:")
                for symbol, size in portfolio['current_positions'].items():
                    if size > 0:
                        print(f"    {symbol}: {size:.6f}")
        print()
        
        # Recent Trades
        print("📋 RECENT TRADES")
        print("-" * 40)
        recent_trades = self.get_recent_trades(3)
        
        if recent_trades:
            for trade in recent_trades:
                timestamp = trade.get('timestamp', 'Unknown')[:19]
                print(f"🔄 {timestamp} | {trade.get('symbol', 'N/A')} | {trade.get('action', 'N/A')} @ ${trade.get('price', 0):.2f}")
                print(f"    Strategy: {trade.get('strategy', 'N/A')} | P&L: ${trade.get('simulated_pnl', 0):.2f}")
        else:
            print("No recent trades found")
        print()
        
        print("="*80)
        print("Press Ctrl+C to stop monitoring")
        print("="*80)
    
    def start_monitoring(self, refresh_interval: int = 30):
        """Start continuous monitoring"""
        print("🚀 Starting Live Trading Monitor")
        print(f"Refresh interval: {refresh_interval} seconds")
        print("Press Ctrl+C to stop")
        
        try:
            while self.monitoring_active:
                self.display_monitor()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            self.monitoring_active = False

def main():
    """Main function"""
    print("🚀 Live Trading Monitor")
    print("=" * 50)
    print("This monitor shows:")
    print("  • Real-time data collection status")
    print("  • Signal generation analysis")
    print("  • Portfolio status")
    print("  • Recent trades")
    print("=" * 50)
    
    # Initialize monitor
    monitor = LiveTradingMonitor()
    
    # Start monitoring
    monitor.start_monitoring(refresh_interval=30)  # Refresh every 30 seconds

if __name__ == "__main__":
    main()
