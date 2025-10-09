#!/usr/bin/env python3
"""
Unified Crypto Trading Pipeline
==============================

Clean, modular system that uses underlying functions instead of duplicate scripts.
All functionality is accessed through a single main.py file.

Usage:
    python main_unified.py data-collection    # Start real-time data collection
    python main_unified.py live-trading       # Start live trading with signals
    python main_unified.py backtest           # Run backtests
    python main_unified.py dashboard          # Start dashboard
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import core modules
from data_ingestion.realtime_fusion_system import RealTimeFusionSystem
from crypto_signal_integration import CryptoSignalIntegration
from crypto_analysis_engine import CryptoAnalysisEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedCryptoPipeline:
    """Unified crypto trading pipeline"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Core components
        self.fusion_system = None
        self.signal_integration = None
        self.analysis_engine = None
        
    def start_data_collection(self, symbols: list = None):
        """Start real-time data collection"""
        print("📡 Starting Real-Time Data Collection")
        print("=" * 50)
        
        symbols = symbols or ['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI']
        
        # Initialize fusion system for data collection
        self.fusion_system = RealTimeFusionSystem(
            data_dir=self.data_dir / "crypto_db",
            symbols=symbols
        )
        
        # Start data collection
        self.fusion_system.start()
        
        print(f"Collecting data for: {', '.join(symbols)}")
        print("Press Ctrl+C to stop")
        
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping data collection...")
            self.fusion_system.stop()
            print("Data collection stopped")
    
    def start_live_trading(self, strategies: list = None):
        """Start live trading with signal generation"""
        print("🚀 Starting Live Trading System")
        print("=" * 50)
        
        strategies = strategies or ['btc_ny_session', 'liquidity_sweep_reversal']
        
        # Initialize signal integration
        self.signal_integration = CryptoSignalIntegration(
            data_dir=self.data_dir,
            selected_strategies=strategies
        )
        
        # Initialize analysis engine
        self.analysis_engine = CryptoAnalysisEngine()
        
        # Start live trading
        self._run_live_trading_loop()
    
    def run_backtest(self, symbols: list = None, days: int = 30):
        """Run backtest on historical data"""
        print("📈 Starting Backtest")
        print("=" * 50)
        
        symbols = symbols or ['BTC']
        
        # Initialize signal integration for backtesting
        self.signal_integration = CryptoSignalIntegration(
            data_dir=self.data_dir,
            selected_strategies=['btc_ny_session', 'liquidity_sweep_reversal']
        )
        
        # Load historical data
        historical_data = self.signal_integration.load_crypto_data(symbols, days)
        
        # Run backtest
        results = self._run_backtest_analysis(historical_data)
        
        # Display results
        self._display_backtest_results(results)
    
    def start_dashboard(self):
        """Start Streamlit dashboard"""
        print("📊 Starting Dashboard")
        print("=" * 50)
        
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "scripts/dashboards/streamlit_app.py"
        ])
    
    def _run_live_trading_loop(self):
        """Run the live trading loop"""
        print("Live trading started. Press Ctrl+C to stop")
        
        try:
            import time
            while True:
                # Generate signals
                signals = self.signal_integration.generate_signals(['BTC'])
                
                # Process signals
                for signal in signals:
                    if signal:
                        print(f"Signal: {signal['signal_type']} @ ${signal['price']:.2f}")
                
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping live trading...")
            print("Live trading stopped")
    
    def _run_backtest_analysis(self, historical_data):
        """Run backtest analysis"""
        results = {}
        
        for symbol, data in historical_data.items():
            if data.empty:
                continue
                
            # Generate signals for historical data
            signals = self.signal_integration.generate_signals([symbol], data)
            
            # Calculate performance
            performance = self._calculate_performance(data, signals)
            results[symbol] = performance
        
        return results
    
    def _calculate_performance(self, data, signals):
        """Calculate backtest performance"""
        if not signals:
            return {"total_return": 0, "sharpe_ratio": 0, "max_drawdown": 0}
        
        # Simple performance calculation
        returns = data['close'].pct_change().dropna()
        total_return = (1 + returns).prod() - 1
        sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": returns.cumsum().expanding().max().sub(returns.cumsum()).max()
        }
    
    def _display_backtest_results(self, results):
        """Display backtest results"""
        print("\n📊 Backtest Results:")
        print("=" * 30)
        
        for symbol, perf in results.items():
            print(f"{symbol}:")
            print(f"  Total Return: {perf['total_return']:.2%}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {perf['max_drawdown']:.2%}")
            print()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Unified Crypto Trading Pipeline")
    parser.add_argument("command", choices=[
        "data-collection", "live-trading", "backtest", "dashboard"
    ], help="Command to run")
    
    parser.add_argument("--symbols", nargs="+", 
                       default=['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI'],
                       help="Symbols to trade")
    
    parser.add_argument("--strategies", nargs="+",
                       default=['btc_ny_session', 'liquidity_sweep_reversal'],
                       help="Trading strategies to use")
    
    parser.add_argument("--days", type=int, default=30,
                       help="Days of historical data for backtest")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = UnifiedCryptoPipeline()
    
    # Execute command
    if args.command == "data-collection":
        pipeline.start_data_collection(args.symbols)
    elif args.command == "live-trading":
        pipeline.start_live_trading(args.strategies)
    elif args.command == "backtest":
        pipeline.run_backtest(args.symbols, args.days)
    elif args.command == "dashboard":
        pipeline.start_dashboard()

if __name__ == "__main__":
    main()
