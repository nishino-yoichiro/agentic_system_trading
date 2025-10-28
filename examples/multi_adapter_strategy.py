"""
Multi-Adapter Strategy Example
==============================

Example showing how to use the same core engine with different market adapters
(Crypto via Coinbase, Equities via Alpaca).

This demonstrates:
1. Same strategy running on both crypto and equities
2. Session-aware trading for equities
3. Unified signal generation across markets
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core components
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core_engine import MarketAdapter, OHLCVBar, OrderSide, OrderType, SignalEngine, RiskManager, RiskLimits, Portfolio
from adapters import AlpacaMarketAdapter, AlpacaBroker, CoinbaseMarketAdapter, SessionController


class SimpleMovingAverageStrategy:
    """
    Simple moving average crossover strategy.
    
    This strategy can run on any market adapter (crypto or equities).
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.price_history = []
    
    def generate_signal(self, bar: OHLCVBar, historical: List[OHLCVBar]) -> Dict[str, Any]:
        """
        Generate trading signal based on moving average crossover.
        
        Args:
            bar: Current OHLCV bar
            historical: Recent bars for context
            
        Returns:
            Signal dictionary or None
        """
        # Add current bar to history
        if historical:
            self.price_history = [b.close for b in historical]
        
        self.price_history.append(bar.close)
        
        # Keep only needed history
        if len(self.price_history) > self.slow_period:
            self.price_history = self.price_history[-self.slow_period:]
        
        if len(self.price_history) < self.slow_period:
            return None
        
        # Calculate moving averages
        fast_ma = sum(self.price_history[-self.fast_period:]) / self.fast_period
        slow_ma = sum(self.price_history[-self.slow_period:]) / self.slow_period
        
        # Generate signal
        signal = None
        
        if fast_ma > slow_ma:
            # Golden cross - bullish
            signal = {
                'signal_type': 'buy',
                'confidence': 0.7,
                'entry_price': bar.close,
                'reason': f'Fast MA ({fast_ma:.2f}) crossed above Slow MA ({slow_ma:.2f})'
            }
        
        elif fast_ma < slow_ma:
            # Death cross - bearish
            signal = {
                'signal_type': 'sell',
                'confidence': 0.7,
                'entry_price': bar.close,
                'reason': f'Fast MA ({fast_ma:.2f}) crossed below Slow MA ({slow_ma:.2f})'
            }
        
        return signal


async def run_trading_bot(
    adapter: MarketAdapter,
    symbol: str,
    strategy_name: str,
    capital: float = 10000.0,
    max_positions: int = 5
):
    """
    Run a trading bot with a specific adapter.
    
    Args:
        adapter: MarketAdapter instance
        symbol: Trading symbol
        strategy_name: Strategy name
        capital: Starting capital
        max_positions: Maximum number of positions
    """
    logger.info(f"Starting trading bot for {symbol} with {adapter.__class__.__name__}")
    
    # Initialize components
    signal_engine = SignalEngine()
    risk_manager = RiskManager(RiskLimits(
        max_position_size=0.1,  # Max 10% per position
        max_total_exposure=0.5,  # Max 50% total
        max_daily_loss=0.02  # Max 2% daily loss
    ))
    portfolio = Portfolio(initial_capital=capital)
    
    # Setup strategy
    sma_strategy = SimpleMovingAverageStrategy(fast_period=10, slow_period=20)
    signal_engine.register_strategy(
        strategy_name,
        lambda bar, hist: sma_strategy.generate_signal(bar, hist)
    )
    
    # Connect to market
    await adapter.connect()
    
    try:
        historical_bars = []
        
        async for bar in adapter.stream_data(interval_seconds=60):
            logger.info(
                f"Received bar: {symbol} @ {bar.timestamp} "
                f"${bar.close:.2f} vol={bar.volume:.2f}"
            )
            
            # Add to history
            historical_bars.append(bar)
            if len(historical_bars) > 50:  # Keep last 50 bars
                historical_bars = historical_bars[-50:]
            
            # Generate signals
            signals = signal_engine.generate_signal(bar, strategy_name, historical_bars)
            
            for signal in signals:
                logger.info(
                    f"Signal: {signal['signal_type']} @ ${signal['entry_price']:.2f} "
                    f"| confidence: {signal['confidence']:.2f}"
                )
                
                # In production, you would:
                # 1. Validate signal against risk limits
                # 2. Place order via broker
                # 3. Track position
                # 4. Manage stop-loss/take-profit
                
        logger.info("Data stream ended")
        
    finally:
        await adapter.disconnect()
        logger.info("Trading bot stopped")


async def main():
    """Main entry point - demonstrates running strategies on different markets."""
    """
    Main entry point - demonstrates running strategies on different markets.
    """
    logger.info("Multi-Adapter Trading Example")
    logger.info("=" * 60)
    
    # Example 3: Session controller demonstration (run first)
    logger.info("\n1. Session Controller Demo")
    logger.info("   Checking market hours for NYSE...")
    
    try:
        session_controller = SessionController(exchange='NYSE')
        current_status = session_controller.format_market_status()
        logger.info(f"Market status: {current_status}")
        
        next_open = session_controller.get_next_open()
        logger.info(f"Next market open: {next_open}")
        
        wait_seconds = session_controller.wait_until_open()
        logger.info(f"Seconds until market opens: {wait_seconds:.0f}")
    except Exception as e:
        logger.error(f"Error in session controller: {e}")
    
    # Example 1: Run strategy on Alpaca (SPY)
    logger.info("\n2. Running strategy on Alpaca (SPY)")
    logger.info("   (Market hours: 9:30-16:00 ET)")
    logger.info("   Note: Requires ALPACA_API_KEY and ALPACA_SECRET_KEY")
    
    try:
        alpaca_adapter = AlpacaMarketAdapter(symbol="SPY", paper=True)
        # Only run if during market hours or close to open
        logger.info("Starting Alpaca adapter...")
        logger.info("Run with: python -c 'import asyncio; from adapters import AlpacaMarketAdapter; asyncio.run(AlpacaMarketAdapter(\"SPY\").connect())'")
    except Exception as e:
        logger.error(f"Error initializing Alpaca: {e}")
    
    # Example 2: Run same strategy on Coinbase (BTC)
    logger.info("\n3. Running same strategy on Coinbase (BTC)")
    logger.info("   (24/7 continuous trading)")
    
    try:
        coinbase_adapter = CoinbaseMarketAdapter(symbol="BTC")
        logger.info("Testing BTC connection...")
        
        # Actually connect to test
        try:
            result = await coinbase_adapter.connect()
            if result:
                logger.info("✅ BTC adapter connected successfully!")
                logger.info("✅ Your Coinbase keys are working!")
                
                # Try to get historical data
                logger.info("Fetching recent BTC data...")
                end = datetime.now()
                start = end - timedelta(days=1)
                df = coinbase_adapter.get_market_data("BTC", start, end)
                
                if df is not None and len(df) > 0:
                    logger.info(f"✅ Got {len(df)} bars of BTC data!")
                    logger.info(f"   Last close: ${df['close'].iloc[-1]:.2f}")
                    
                    # Test streaming
                    logger.info("Testing streaming (will fetch one bar)...")
                    await coinbase_adapter.disconnect()
                else:
                    logger.warning("⚠️ No BTC data returned (might be API issue)")
            else:
                logger.warning("⚠️ BTC adapter failed to connect (check API keys)")
        except Exception as e:
            logger.error(f"❌ Error connecting to BTC: {e}")
            logger.info("This is expected if COINBASE_API_KEY/SECRET not set")
            
    except Exception as e:
        logger.error(f"Error initializing Coinbase: {e}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
