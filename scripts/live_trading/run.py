"""
Live Trading System Runner
Simple script to run the live trading log system
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from run_live_trading import main

if __name__ == "__main__":
    print("🚀 Starting Live Trading System")
    print("This will generate signals every 1 minute")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    main()
