#!/usr/bin/env python3
"""
Debug Interactive Trading
=========================

Run the interactive trading module with debug logging to see what's happening.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/debug_interactive_trading.log'),
        logging.StreamHandler()
    ]
)

def main():
    """Run interactive trading with debug logging"""
    print("Starting interactive trading with debug logging...")
    
    try:
        from src.interactive_trading_module import InteractiveTradingModule
        
        # Create module
        module = InteractiveTradingModule()
        
        # Set up minimal test configuration
        module.selected_strategies = ['test_every_minute']
        module.selected_symbols = ['BTC']
        module.signal_interval = 10  # 10 seconds for faster testing
        
        print(f"Selected strategies: {module.selected_strategies}")
        print(f"Selected symbols: {module.selected_symbols}")
        print(f"Signal interval: {module.signal_interval}")
        
        # Load historical data
        module._load_historical_data()
        
        # Initialize signal integration
        module._initialize_signal_integration()
        
        print("Setup complete. Starting signal generation test...")
        
        # Test signal generation once
        import pandas as pd
        from datetime import datetime, timezone
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': [datetime.now(timezone.utc)],
            'open': [50000],
            'high': [50100],
            'low': [49900],
            'close': [50050],
            'volume': [1000]
        })
        
        print(f"Test data: {test_data}")
        
        # Test signal generation
        data_dict = {'BTC': test_data}
        signals = module.signal_integration.framework.generate_signals(
            data_dict, 
            strategies=['test_every_minute']
        )
        
        print(f"Generated {len(signals)} signals")
        for i, signal in enumerate(signals):
            print(f"Signal {i}: {signal.signal.name if signal else 'None'} - {signal.strategy_name if signal else 'None'}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
