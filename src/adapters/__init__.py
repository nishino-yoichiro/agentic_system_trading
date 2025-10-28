"""
Market Adapters
===============

Implementations of MarketAdapter for different exchanges:
- Coinbase (crypto, 24/7)
- Alpaca (equities, 9:30-16:00 ET)
- Binance (optional, future)

Usage:
    # Create adapter for SPY (equities)
    adapter = AlpacaMarketAdapter(symbol="SPY", paper=True)
    await adapter.connect()
    
    async for bar in adapter.stream_data(interval_seconds=60):
        if adapter.is_market_open():
            # Trade during market hours
            signal = strategy.generate_signal(bar)
    
    # Create adapter for BTC (crypto)
    adapter = CoinbaseMarketAdapter(symbol="BTC")
    await adapter.connect()
    
    async for bar in adapter.stream_data(interval_seconds=60):
        # Always can trade (crypto never closes)
        signal = strategy.generate_signal(bar)
"""

from .alpaca_adapter import AlpacaMarketAdapter
from .alpaca_broker import AlpacaBroker
from .session_controller import SessionController
from .coinbase_adapter import CoinbaseMarketAdapter

__all__ = [
    'AlpacaMarketAdapter',
    'AlpacaBroker',
    'SessionController',
    'CoinbaseMarketAdapter'
]