"""
Run Live Trading Log System
Continuous signal generation and trade logging
"""

import time
import schedule
import logging
import sys
import yaml
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from live_trading_log import LiveTradingLog

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global trading log instance
trading_log = None

def run_trading_update():
    """Run trading log update"""
    global trading_log
    
    try:
        logger.info("🔄 Starting trading log update...")
        
        # The fusion system now handles signal generation automatically
        # This function just shows portfolio status
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}] ℹ️  No new trades executed (handled by fusion system)")
        
        # Show portfolio summary
        if trading_log:
            portfolio = trading_log.get_portfolio_summary()
            logger.info(f"[{timestamp}] 💰 Portfolio: ${portfolio['total_value']:,.2f} | PnL: ${portfolio['cumulative_pnl']:,.2f}")
        
    except Exception as e:
        logger.error(f"❌ Error in trading update: {e}")

def main():
    """Main function to run continuous trading log"""
    global trading_log
    
    print("🚀 Starting Continuous Live Trading Log")
    print("=" * 50)
    print("This will run trading updates every 1 minute")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    # Load API keys
    try:
        with open('config/api_keys.yaml', 'r') as f:
            api_keys = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load API keys: {e}")
        api_keys = {}
    
    # Initialize trading log with API keys (this starts the fusion system)
    trading_log = LiveTradingLog(api_keys=api_keys)
    
    # Schedule trading updates every 1 minute (just for portfolio status)
    schedule.every(1).minutes.do(run_trading_update)
    
    # Run initial update
    run_trading_update()
    
    # Keep running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n🛑 Stopping live trading log...")
        logger.info("Live trading log stopped by user")
        if trading_log:
            trading_log.print_portfolio_summary()

if __name__ == "__main__":
    main()
