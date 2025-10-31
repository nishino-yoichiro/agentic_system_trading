#!/usr/bin/env python3
"""
Adapter Testing Script
======================

Tests both BTC (Coinbase) and SPY (Alpaca) adapters for:
1. Historical data loading with gap detection
2. WebSocket streaming
3. Parquet storage operations

Usage:
    python scripts/test_adapters.py --symbol BTC  --market crypto
    python scripts/test_adapters.py --symbol SPY  --market equities
    python scripts/test_adapters.py --all  # Test both
"""

import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adapters import CoinbaseMarketAdapter, AlpacaMarketAdapter, SessionController

async def test_historical_data(symbol: str, market: str, days: int = 7):
    """Test historical data loading"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Historical Data: {symbol} ({market})")
    logger.info(f"{'='*60}")
    
    try:
        # Create appropriate adapter
        if market == 'crypto':
            adapter = CoinbaseMarketAdapter(symbol=symbol)
        elif market == 'equities':
            adapter = AlpacaMarketAdapter(symbol=symbol, paper=True)
            # Check if market is open
            session = SessionController(exchange='NYSE')
            if not session.is_market_open():
                logger.warning("Market is closed. Historical data should still work.")
        else:
            raise ValueError(f"Unknown market: {market}")
        
        # Load historical data (auto gap-fills)
        logger.info(f"Loading {days} days of historical data...")
        df = adapter.load_historical_data(days=days, fill_gaps=True)
        
        if df is not None and len(df) > 0:
            logger.info(f"✅ Successfully loaded {len(df)} bars")
            logger.info(f"   Date range: {df.index.min()} to {df.index.max()}")
            logger.info(f"   Latest close: ${df['close'].iloc[-1]:.2f}")
            
            # Check Parquet file exists
            storage = adapter.storage
            parquet_file = storage.db_dir / f"{symbol}_historical.parquet"
            if parquet_file.exists():
                logger.info(f"✅ Parquet file exists: {parquet_file}")
                
                # Check file size
                size_mb = parquet_file.stat().st_size / (1024 * 1024)
                logger.info(f"   File size: {size_mb:.2f} MB")
            
            return True
        else:
            logger.error(f"❌ No data loaded")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error loading historical data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_websocket_streaming(symbol: str, market: str, max_bars: int = 3):
    """Test WebSocket streaming"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing WebSocket Streaming: {symbol} ({market})")
    logger.info(f"{'='*60}")
    
    try:
        # Create appropriate adapter
        if market == 'crypto':
            adapter = CoinbaseMarketAdapter(symbol=symbol)
        elif market == 'equities':
            adapter = AlpacaMarketAdapter(symbol=symbol, paper=True)
            # Check if market is open
            session = SessionController(exchange='NYSE')
            if not session.is_market_open():
                logger.warning("⚠️ Market is closed. WebSocket will wait for next open.")
                return False
        else:
            raise ValueError(f"Unknown market: {market}")
        
        # Connect
        logger.info("Connecting to market data stream...")
        await adapter.connect()
        
        if not adapter.is_connected:
            logger.error("❌ Failed to connect")
            return False
        
        logger.info("✅ Connected successfully!")
        logger.info(f"Streaming {max_bars} bars...")
        
        bar_count = 0
        async for bar in adapter.stream_data(interval_seconds=60):
            bar_count += 1
            logger.info(
                f"  Bar {bar_count}: {bar.timestamp} | "
                f"${bar.close:.2f} | Vol: {bar.volume:.0f}"
            )
            
            if bar_count >= max_bars:
                logger.info(f"✅ Received {bar_count} bars successfully!")
                break
            
            # Timeout after 60 seconds
            if bar_count == 0:
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("Streaming timeout")
        
        # Disconnect
        await adapter.disconnect()
        logger.info("✅ Disconnected")
        
        # Check live bar file
        storage = adapter.storage
        live_file = storage.db_dir / f"{symbol}_1m.parquet"
        if live_file.exists():
            logger.info(f"✅ Live bars saved to: {live_file}")
        
        return True
        
    except TimeoutError:
        logger.error("❌ Streaming timeout - no data received")
        return False
    except Exception as e:
        logger.error(f"❌ Error streaming data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_storage_operations(symbol: str, market: str):
    """Test storage operations"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Storage Operations: {symbol} ({market})")
    logger.info(f"{'='*60}")
    
    try:
        if market == 'crypto':
            adapter = CoinbaseMarketAdapter(symbol=symbol)
        elif market == 'equities':
            adapter = AlpacaMarketAdapter(symbol=symbol, paper=True)
        else:
            raise ValueError(f"Unknown market: {market}")
        
        storage = adapter.storage
        logger.info(f"Storage directory: {storage.db_dir}")
        
        # Test historical
        historical = storage.load_historical(symbol)
        if historical is not None and len(historical) > 0:
            logger.info(f"✅ Historical data loaded: {len(historical)} bars")
        else:
            logger.warning("⚠️ No historical data found")
        
        # Test live bars
        live = storage.load_live_bars(symbol)
        if live is not None and len(live) > 0:
            logger.info(f"✅ Live bars found: {len(live)} bars")
        else:
            logger.info("ℹ️  No live bars yet (normal if not streaming)")
        
        # Test combined data
        combined = storage.get_combined_data(symbol, days=7)
        if combined is not None and len(combined) > 0:
            logger.info(f"✅ Combined data: {len(combined)} bars")
        else:
            logger.warning("⚠️ No combined data")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing storage: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def run_full_test(symbol: str, market: str):
    """Run all tests for a symbol"""
    logger.info(f"\n{'#'*60}")
    logger.info(f"  FULL TEST: {symbol} ({market})")
    logger.info(f"{'#'*60}\n")
    
    results = {
        'historical': False,
        'streaming': False,
        'storage': False
    }
    
    # Test 1: Historical data
    results['historical'] = await test_historical_data(symbol, market, days=7)
    
    # Test 2: Storage operations
    results['storage'] = await test_storage_operations(symbol, market)
    
    # Test 3: WebSocket streaming (skip if market closed for equities)
    if market == 'equities':
        session = SessionController(exchange='NYSE')
        if not session.is_market_open():
            logger.warning("⚠️ Skipping WebSocket test (market closed)")
            results['streaming'] = None
        else:
            results['streaming'] = await test_websocket_streaming(symbol, market, max_bars=3)
    else:
        results['streaming'] = await test_websocket_streaming(symbol, market, max_bars=3)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST SUMMARY: {symbol} ({market})")
    logger.info(f"{'='*60}")
    logger.info(f"Historical: {'✅ PASS' if results['historical'] else '❌ FAIL'}")
    logger.info(f"Storage:    {'✅ PASS' if results['storage'] else '❌ FAIL'}")
    logger.info(f"Streaming:  {'✅ PASS' if results['streaming'] else '⚠️ SKIP' if results['streaming'] is None else '❌ FAIL'}")
    
    all_passed = all(v for v in results.values() if v is not None)
    logger.info(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test adapters")
    parser.add_argument('--symbol', help='Symbol to test (BTC, SPY, etc.)')
    parser.add_argument('--market', choices=['crypto', 'equities'], 
                       help='Market type')
    parser.add_argument('--all', action='store_true', 
                       help='Test both BTC and SPY')
    parser.add_argument('--test', choices=['historical', 'streaming', 'storage', 'all'],
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    if args.all:
        # Test both
        logger.info("Running full test suite for both markets")
        
        results = []
        results.append(await run_full_test('BTC', 'crypto'))
        
        # Check if equities market is open
        session = SessionController(exchange='NYSE')
        logger.info(f"\nEquities market status: {session.format_market_status()}")
        
        results.append(await run_full_test('SPY', 'equities'))
        
        all_passed = all(results)
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL RESULT: {'✅ ALL TESTS PASSED' if all_passed else '⚠️ SOME TESTS FAILED'}")
        
    elif args.symbol and args.market:
        # Test specific symbol
        await run_full_test(args.symbol, args.market)
    
    else:
        logger.error("Specify --symbol and --market, or use --all")
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())


