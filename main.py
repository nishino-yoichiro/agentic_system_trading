#!/usr/bin/env python3
"""
Enhanced Crypto Trading Pipeline
================================

Production-ready crypto trading system with live signal generation, backtesting, and portfolio management.

Usage:
    python main.py data-collection    # Collect historical data with gap filling
    python main.py live-trading       # Start live trading with real-time data
    python main.py backtest           # Run standard backtests
    python main.py backtest --backtest-type parameter-sweep  # Run parameter optimization
    python main.py dashboard          # Start dashboard
    python main.py initialize-data    # Initialize historical data (legacy)
    python main.py monitor            # Start live trading monitor
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Core modules will be imported as needed by individual commands

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
        
    def start_data_collection(self, symbols: list = None, days: int = 7):
        """Start historical data collection with gap filling and incremental saving"""
        print("📡 Starting Historical Data Collection")
        print("=" * 50)
        print("This collects historical data with gap filling and saves incrementally.")
        print("If you Ctrl+C, progress is saved after each gap.")
        print("=" * 50)
        
        symbols = symbols or ['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI']
        
        # Use the original crypto collector for historical data with gap filling
        import asyncio
        from src.data_ingestion.crypto_collector import CryptoDataCollector
        
        async def collect_data():
            collector = CryptoDataCollector()
            # This will fill gaps incrementally and save incrementally
            results = await collector.collect_crypto_data(symbols, days_back=days)
            
            print("\n📊 Collection Results:")
            print("=" * 30)
            for symbol in symbols:
                if symbol in results:
                    df = results[symbol]
                    print(f"✅ {symbol}: {len(df)} data points")
                else:
                    print(f"❌ {symbol}: No data collected")
        
        asyncio.run(collect_data())
    
    def start_live_trading(self, strategies: list = None):
        """Start live trading with signal generation"""
        print("🚀 Starting Live Trading System")
        print("=" * 50)
        
        # Use integrated live trading system
        from scripts.live_trading.integrated_live_trading import main as live_main
        live_main()
    
    def run_backtest(self, symbols: list = None, days: int = 30, backtest_type: str = "normal"):
        """Run backtest on historical data"""
        print("📈 Starting Backtest")
        print("=" * 50)
        
        if backtest_type == "parameter-sweep":
            print("🔬 Running Parameter Sweep Backtest")
            print("This will test different parameter combinations to find optimal settings.")
            print("=" * 50)
            
            # Use parameter sweep backtester
            from scripts.backtesting.parameter_sweep_backtester import ParameterSweepBacktester
            
            backtester = ParameterSweepBacktester()
            
            # Run parameter sweep
            symbols = symbols or ['BTC', 'ETH']
            results = backtester.run_parameter_sweep(
                symbols=symbols,
                days=days,
                output_dir="backtests/results"
            )
            
            print(f"\n📊 Parameter Sweep Complete!")
            print(f"Tested {len(results)} parameter combinations")
            print("Results saved to backtests/results/")
            
        else:
            print("📊 Running Standard Multi-Symbol Backtest")
            print("=" * 50)
            
            # Use the wrapper for interactive backtesting
            from scripts.backtesting.run_multi_backtest import main as backtest_main
            backtest_main()
    
    def initialize_data(self, symbols: list = None):
        """Initialize historical data"""
        print("🔄 Initializing Historical Data")
        print("=" * 50)
        
        # Use historical data initializer
        from scripts.data_collection.initialize_historical_data import main as init_main
        init_main()
    
    def start_dashboard(self):
        """Start Streamlit dashboard"""
        print("📊 Starting Dashboard")
        print("=" * 50)
        
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "scripts/dashboards/streamlit_app.py"
        ])
    

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Unified Crypto Trading Pipeline")
    parser.add_argument("command", choices=[
        "data-collection", "live-trading", "backtest", "dashboard", "initialize-data", "monitor"
    ], help="Command to run")
    
    parser.add_argument("--symbols", nargs="+", 
                       default=['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI'],
                       help="Symbols to trade")
    
    parser.add_argument("--strategies", nargs="+",
                       default=['btc_ny_session', 'liquidity_sweep_reversal'],
                       help="Trading strategies to use")
    
    parser.add_argument("--days", type=int, default=30,
                       help="Days of historical data for backtest")
    
    parser.add_argument("--backtest-type", choices=["normal", "parameter-sweep"], 
                       default="normal", help="Type of backtest to run")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = UnifiedCryptoPipeline()
    
    # Execute command
    if args.command == "data-collection":
        pipeline.start_data_collection(args.symbols, args.days)
    elif args.command == "live-trading":
        pipeline.start_live_trading(args.strategies)
    elif args.command == "backtest":
        pipeline.run_backtest(args.symbols, args.days, args.backtest_type)
    elif args.command == "dashboard":
        pipeline.start_dashboard()
    elif args.command == "initialize-data":
        pipeline.initialize_data(args.symbols)
    elif args.command == "monitor":
        pipeline.start_monitor()

if __name__ == "__main__":
    main()
