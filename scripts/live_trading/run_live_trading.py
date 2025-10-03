"""
Run Live Trading Log System
Continuous signal generation and trade logging
"""

import time
import schedule
import logging
from datetime import datetime
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

def run_trading_update():
    """Run trading log update"""
    try:
        logger.info("🔄 Starting trading log update...")
        
        trading_log = LiveTradingLog()
        
        # Generate and log signals
        trades = trading_log.generate_and_log_signals()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if trades:
            logger.info(f"[{timestamp}] ✅ Executed {len(trades)} trades")
            for trade in trades:
                logger.info(f"[{timestamp}]    {trade.signal_type} @ ${trade.price:.2f} | PnL: ${trade.simulated_pnl:.2f}")
        else:
            logger.info(f"[{timestamp}] ℹ️  No new trades executed")
        
        # Show portfolio summary
        portfolio = trading_log.get_portfolio_summary()
        logger.info(f"[{timestamp}] 💰 Portfolio: ${portfolio['total_value']:,.2f} | PnL: ${portfolio['cumulative_pnl']:,.2f}")
        
    except Exception as e:
        logger.error(f"❌ Error in trading update: {e}")

def main():
    """Main function to run continuous trading log"""
    print("🚀 Starting Continuous Live Trading Log")
    print("=" * 50)
    print("This will run trading updates every 1 minute")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    # Schedule trading updates every 1 minute
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

if __name__ == "__main__":
    main()
