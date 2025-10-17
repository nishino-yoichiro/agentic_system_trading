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
    
    def start_live_trading(self, strategies: list = None, symbols: list = None,
                           fresh_portfolio: bool = False, initial_capital: float = 100000.0,
                           prefill_days: int = 0, auto_fill_gaps: bool = True):
        """Start live trading with signal generation"""
        print("🚀 Starting Live Trading System")
        print("=" * 50)
        
        # Wire directly to IntegratedLiveTrading with options
        from scripts.live_trading.integrated_live_trading import IntegratedLiveTrading
        import yaml
        
        # Load API keys (optional)
        try:
            with open('config/api_keys.yaml', 'r') as f:
                api_keys = yaml.safe_load(f)
        except Exception:
            api_keys = {}
        
        trading_system = IntegratedLiveTrading(
            api_keys=api_keys,
            symbols=symbols or ['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI'],
            initial_capital=initial_capital,
            fresh_portfolio=fresh_portfolio,
            selected_strategies=strategies,
            prefill_days=prefill_days,
            auto_fill_gaps=auto_fill_gaps
        )
        
        portfolio = trading_system.get_portfolio_summary()
        print(f"\n💰 Portfolio:")
        print(f"   Cash: ${portfolio['cash_balance']:,.2f}")
        print(f"   Total Value: ${portfolio['total_value']:,.2f}")
        print(f"   Cumulative P&L: ${portfolio['cumulative_pnl']:,.2f}")
        print(f"   Positions: {portfolio['positions']}")
        print()
        
        trading_system.start()
    
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
    
    def start_monitor(self, refresh_interval: int = 30):
        """Start live trading terminal monitor"""
        print("🖥️  Starting Live Trading Monitor")
        print("=" * 50)
        
        from scripts.live_trading.live_trading_monitor import LiveTradingMonitor
        monitor = LiveTradingMonitor()
        monitor.start_monitoring(refresh_interval=refresh_interval)


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
                       default=None,
                       help="Trading strategies to use (default: all)")
    parser.add_argument("--fresh-portfolio", action="store_true",
                       help="Start with a fresh portfolio and reset to initial capital")
    parser.add_argument("--initial-capital", type=float, default=100000.0,
                       help="Initial capital for fresh portfolio")
    
    parser.add_argument("--days", type=int, default=30,
                       help="Days of historical data for backtest")
    
    parser.add_argument("--backtest-type", choices=["normal", "parameter-sweep"], 
                       default="normal", help="Type of backtest to run")
    
    # Live-trading interactive selection
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive prompts to choose symbols, strategies, and portfolio options")
    parser.add_argument("--prefill-days", type=int, default=0,
                       help="Prefill N days of historical data before live trading (0 = disabled)")
    parser.add_argument("--no-prefill", action="store_true",
                       help="Disable automatic gap fill check before starting live trading")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = UnifiedCryptoPipeline()
    
    # Execute command
    if args.command == "data-collection":
        pipeline.start_data_collection(args.symbols, args.days)
    elif args.command == "live-trading":
        if args.interactive:
            # Build interactive selections
            default_symbols = args.symbols
            try:
                from src.crypto_signal_integration import CryptoSignalIntegration
                integration = CryptoSignalIntegration()
                available_strategies = list(integration.framework.strategies.keys())
            except Exception:
                available_strategies = []
            
            print("\n=== Live Trading Interactive Setup ===")
            # Fresh portfolio
            resp = input("Start with a fresh portfolio? (y/N): ").strip().lower()
            fresh_portfolio = resp in ("y", "yes")
            # Initial capital
            cap_in = input(f"Initial capital [{args.initial_capital}]: ").strip()
            try:
                initial_capital = float(cap_in) if cap_in else args.initial_capital
            except ValueError:
                initial_capital = args.initial_capital
            # Symbols
            print(f"Available symbols (default): {', '.join(default_symbols)}")
            sym_in = input("Enter symbols comma-separated or 'all' for default: ").strip()
            if not sym_in or sym_in.lower() == 'all':
                symbols = default_symbols
            else:
                symbols = [s.strip().upper() for s in sym_in.split(',') if s.strip()]
            # Strategies
            if available_strategies:
                print(f"Available strategies: {', '.join(available_strategies)}")
            strat_in = input("Enter strategies comma-separated or 'all' for all: ").strip()
            if not strat_in or strat_in.lower() == 'all':
                strategies = None  # None means use all
            else:
                strategies = [s.strip() for s in strat_in.split(',') if s.strip()]
            
            pipeline.start_live_trading(
                strategies=strategies,
                symbols=symbols,
                fresh_portfolio=fresh_portfolio,
                initial_capital=initial_capital,
                prefill_days=args.prefill_days,
                auto_fill_gaps=not args.no_prefill
            )
        else:
            pipeline.start_live_trading(
                strategies=args.strategies,
                symbols=args.symbols,
                fresh_portfolio=args.fresh_portfolio,
                initial_capital=args.initial_capital,
                prefill_days=args.prefill_days,
                auto_fill_gaps=not args.no_prefill
            )
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
