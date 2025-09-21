"""
Check Collection Status
Shows current progress of historical data collection
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_ingestion.incremental_cache import IncrementalCache

def main():
    """Check and display collection status"""
    cache = IncrementalCache(Path("data"))
    status = cache.get_collection_status()
    
    print("=" * 60)
    print("COLLECTION STATUS")
    print("=" * 60)
    print(f"Total symbols: {status['total_symbols']}")
    print(f"Completed: {status['completed']}")
    print(f"In progress: {status['in_progress']}")
    print(f"Remaining: {status['remaining']}")
    print(f"Progress: {status['progress_percentage']:.1f}%")
    print()
    
    if status['collection_start_time']:
        start_time = datetime.fromisoformat(status['collection_start_time'])
        print(f"Collection started: {start_time}")
    
    if status['last_update']:
        last_update = datetime.fromisoformat(status['last_update'])
        print(f"Last update: {last_update}")
    
    print()
    
    # Show completed symbols
    if cache.progress.get('symbols_completed'):
        print("Completed symbols:")
        for symbol in cache.progress['symbols_completed']:
            print(f"  ✅ {symbol}")
    
    # Show in-progress symbols
    if cache.progress.get('symbols_in_progress'):
        print("\nIn progress symbols:")
        for symbol, info in cache.progress['symbols_in_progress'].items():
            started_at = datetime.fromisoformat(info['started_at'])
            print(f"  🔄 {symbol} (started: {started_at})")
    
    # Show remaining symbols
    remaining = cache.get_symbols_to_collect(
        cache.progress.get('symbols_completed', []) + 
        list(cache.progress.get('symbols_in_progress', {}).keys()) +
        [f"SYMBOL_{i}" for i in range(status['total_symbols'])]
    )
    
    if remaining:
        print(f"\nRemaining symbols: {len(remaining)}")
        for symbol in remaining[:10]:  # Show first 10
            print(f"  ⏳ {symbol}")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more")
    
    print("\n" + "=" * 60)
    
    if status['remaining'] > 0:
        print("To resume collection, run: python setup_historical_incremental.py")
    else:
        print("Collection complete! All symbols processed.")

if __name__ == "__main__":
    main()


