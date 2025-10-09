"""
Enhanced Crypto Trading Pipeline - Main Entry Point
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    parser = argparse.ArgumentParser(description="Enhanced Crypto Trading Pipeline")
    parser.add_argument("command", choices=[
        "live-trading", "dashboard", "backtest", "data-collection", "advanced-portfolio", "data-only", "monitor"
    ], help="Command to run")
    parser.add_argument("--args", nargs="*", help="Additional arguments")
    
    args = parser.parse_args()
    
    if args.command == "live-trading":
        print("🚀 Starting Live Trading System")
        from scripts.live_trading.integrated_live_trading import main as live_main
        live_main()
    
    elif args.command == "dashboard":
        print("📊 Starting Dashboard")
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "streamlit", "run", "scripts/dashboards/streamlit_app.py", "--server.port", "8501"])
    
    elif args.command == "backtest":
        print("📈 Starting Backtest")
        from scripts.backtesting.run_multi_backtest import main as backtest_main
        backtest_main()
    
    elif args.command == "data-collection":
        print("📡 Starting WebSocket Data Collection")
        from scripts.data_collection.websocket_only_collection import main as data_main
        data_main()
    
    elif args.command == "advanced-portfolio":
        print("💼 Starting Advanced Portfolio")
        from scripts.run_advanced_portfolio import main as portfolio_main
        portfolio_main()
    
    elif args.command == "data-only":
        print("📡 Starting Data Collection Only")
        from scripts.live_trading.run_data_collection import main as data_main
        data_main()
    
    elif args.command == "monitor":
        print("📊 Starting Live Trading Monitor")
        from scripts.live_trading.live_trading_monitor import main as monitor_main
        monitor_main()

if __name__ == "__main__":
    main()
