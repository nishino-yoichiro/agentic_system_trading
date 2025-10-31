# Multi-Adapter Architecture Guide

## Overview

This document explains the multi-adapter architecture that allows your trading system to work seamlessly across different markets: crypto (Coinbase) and equities (Alpaca).

## Why This Architecture?

### Problem Statement

Crypto and equities operate in fundamentally different market microstructures:

| Aspect | Coinbase (Crypto) | Alpaca (Equities) |
|--------|------------------|-------------------|
| **Market Hours** | 24/7 continuous | 9:30-16:00 ET (session-based) |
| **Order Book** | Full L2 depth | Top-of-book only (NBBO) |
| **Tick Size** | Arbitrary decimal | Fixed $0.01 (for SPY) |
| **Data Model** | Always-on streaming | Discrete sessions with halts |
| **Execution** | Instant fills | Routed via brokers, may reject |
| **Fee Model** | Spread-based | SEC/FINRA fees + routing |

If you tried to use crypto-specific code for equities (or vice versa), you'd hit issues like:
- Assuming markets are always open (crypto thinking) vs. handling session boundaries (equities)
- Expecting L2 order book depth when only top-of-book is available
- Missing DST transitions, OPEX days, trading halts
- Incorrect timestamp handling (UTC everywhere vs. timezone-aware ET for equities)

### Solution: Adapter Pattern

We solve this by **isolating market-specific logic** behind a unified interface. Your core strategy engine never knows whether it's trading BTC or SPY - it just receives OHLCV bars and emits signals.

```
Core Engine (market-agnostic)
    ↓
    Receives: OHLCVBar {timestamp, open, high, low, close, volume, source}
    Emits: Signal {signal_type, confidence, entry_price}
    ↓
Market Adapters (market-specific)
    ↓
Coinbase Adapter      Alpaca Adapter
(24/7 crypto)       (9:30-16:00 ET)
```

## Architecture Components

### 1. Core Engine (`src/core_engine/`)

#### `MarketAdapter` (Abstract Interface)

```python
class MarketAdapter(ABC):
    async def stream_data(interval_seconds: int) -> AsyncIterator[OHLCVBar]
    async def place_order(...) -> Order
    async def get_order_status(order_id: str) -> OrderStatus
    def is_market_open() -> bool
    def get_market_data(...) -> pd.DataFrame
```

**Key Data Structure: OHLCVBar**

```python
@dataclass
class OHLCVBar:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str  # 'coinbase', 'alpaca', etc.
```

This unified bar format allows your strategies to work identically regardless of market.

#### `SignalEngine`

Market-agnostic signal generation. Strategies receive OHLCV bars and return signal dictionaries.

#### `RiskManager`

Centralized risk limits (position size, daily loss, total exposure).

#### `Portfolio`

Tracks positions, P&L, and trade history across all markets.

### 2. Adapters (`src/adapters/`)

#### `AlpacaMarketAdapter`

Handles US equities:
- WebSocket streaming via Alpaca v2/sip or v2/iex
- Trade aggregation into 1-minute bars
- **Session management**: Only processes trades during 9:30-16:00 ET
- Uses `pandas-market-calendars` for trading days, DST, holidays
- Respects market halts, OPEX days

#### `AlpacaBroker`

Separate class for order execution:
- Market/Limit/Stop orders
- Fill tracking
- Account balance monitoring

#### `CoinbaseMarketAdapter`

Handles crypto:
- 24/7 continuous streaming
- No session boundaries
- Direct trade data aggregation

#### `SessionController`

Market hours management for equities:
```python
controller = SessionController(exchange='NYSE')
if controller.is_market_open():
    # Run strategy
else:
    # Sleep until next open
    wait_seconds = controller.wait_until_open()
```

### 3. Usage Pattern

```python
# Run same strategy on both markets
strategy = MovingAverageStrategy(fast=10, slow=20)

# Equities (session-aware)
alpaca_adapter = AlpacaMarketAdapter(symbol="SPY")
await alpaca_adapter.connect()

async for bar in alpaca_adapter.stream_data(interval_seconds=60):
    if alpaca_adapter.is_market_open():
        signal = strategy.generate_signal(bar)
        # Trade if signal
    else:
        # Wait until market opens
        await asyncio.sleep(controller.wait_until_open())

# Crypto (24/7)
coinbase_adapter = CoinbaseMarketAdapter(symbol="BTC")
await coinbase_adapter.connect()

async for bar in coinbase_adapter.stream_data(interval_seconds=60):
    signal = strategy.generate_signal(bar)
    # Always can trade (crypto never closes)
```

## Migration Strategy

### Current State (Crypto-Only)

Your existing code has crypto-specific logic scattered throughout:
- `src/crypto_signal_integration.py`
- `src/crypto_analysis_engine.py`
- `src/data_ingestion/coinbase_*.py`

### Target State (Multi-Adapter)

1. **Keep existing code working** - No breaking changes to crypto functionality
2. **Add adapter layer** - New `src/adapters/` and `src/core_engine/` directories
3. **Gradually refactor** - Over time, extract common logic into core engine
4. **Parallel implementation** - Both crypto and equities can coexist

### Step-by-Step Migration

#### Phase 1: Setup (✅ Done)
- [x] Create `core_engine` directory with base interfaces
- [x] Create `adapters` directory with Alpaca and Coinbase implementations
- [x] Add session controller for market hours
- [x] Add broker interface for order execution

#### Phase 2: Test in Parallel (Next Steps)
- [ ] Create example that runs the same strategy on both BTC and SPY
- [ ] Validate that crypto adapters still work
- [ ] Test session controller edge cases (DST, holidays, halts)

#### Phase 3: Refactor (Future)
- [ ] Extract signal generation from crypto-specific code
- [ ] Move to core signal engine
- [ ] Update existing strategies to use new interface

#### Phase 4: Multi-Asset Backtesting
- [ ] Unified backtester that handles both markets
- [ ] Cross-market correlation analysis
- [ ] Portfolio allocation across asset classes

## Benefits

### Code Reuse
```python
# Same strategy code
class MovingAverageStrategy:
    def generate_signal(self, bar: OHLCVBar):
        # This works for ANY market
        ...
```

### Faster Development
- Test strategy on crypto (fast iteration, 24/7 data)
- Deploy to equities with minimal changes
- Single signal library for all markets

### Risk Isolation
- Market-specific bugs stay in adapters
- Session handling doesn't affect core logic
- Exchange outages only impact that adapter

### Portfolio Expansion
- Trade both crypto and equities from same codebase
- Unified risk management
- Cross-asset strategies

## Example: Complete Trading Loop

```python
import asyncio
from adapters import AlpacaMarketAdapter, SessionController
from core_engine import SignalEngine, Portfolio, RiskManager

# Setup
alpaca = AlpacaMarketAdapter(symbol="SPY", paper=True)
session = SessionController(exchange='NYSE')
engine = SignalEngine()
portfolio = Portfolio(initial_capital=10000)
risk_manager = RiskManager()

# Register strategy
engine.register_strategy("ma_crossover", moving_average_strategy)

await alpaca.connect()

async for bar in alpaca.stream_data(interval_seconds=60):
    # Only trade during market hours
    if not session.is_market_open():
        logger.info("Market closed, skipping...")
        continue
    
    # Generate signals
    signals = engine.generate_signal(bar, strategy_name="ma_crossover")
    
    for signal in signals:
        # Validate against risk limits
        is_valid, reason = risk_manager.validate_order(
            symbol=bar.symbol,
            side=signal['signal_type'],
            quantity=signal['quantity'],
            price=signal['entry_price'],
            capital=portfolio.get_total_value()
        )
        
        if is_valid:
            # Place order via AlpacaBroker
            order = await alpaca_broker.place_order(...)
            portfolio.add_position(symbol=bar.symbol, ...)
            logger.info(f"Order placed: {order.order_id}")
        else:
            logger.warning(f"Order rejected: {reason}")
```

## Configuration

Add to `.env`:

```bash
# Alpaca (equities)
ALPACA_API_KEY=pk_test_...
ALPACA_SECRET_KEY=...

# Coinbase (crypto)
COINBASE_API_KEY=...
COINBASE_SECRET_KEY=...
```

## Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run example
python examples/multi_adapter_strategy.py

# Should see:
# 1. SPY trading during market hours only
# 2. BTC trading 24/7
# 3. Same strategy generating signals for both
```

## Next Steps

1. ✅ Core adapter interfaces created
2. ✅ Alpaca adapter implemented
3. ✅ Session controller for market hours
4. ⏳ Test with live paper trading
5. ⏳ Integrate with existing strategies
6. ⏳ Add backtesting support
7. ⏳ Create unified dashboard

## File Structure

```
src/
├── core_engine/          # Market-agnostic core
│   ├── market_adapter.py    # Abstract interface
│   ├── signal_engine.py     # Signal generation
│   ├── risk_manager.py      # Risk limits
│   └── portfolio.py         # Position tracking
│
└── adapters/             # Market-specific implementations
    ├── alpaca_adapter.py    # US equities (SPY)
    ├── alpaca_broker.py     # Order execution
    ├── coinbase_adapter.py  # Crypto (BTC)
    └── session_controller.py # Market hours
```

## Conclusion

This architecture gives you:
- ✅ **One engine, many markets** - Same code works for crypto and equities
- ✅ **Session awareness** - Automatic handling of market hours for equities
- ✅ **Risk isolation** - Market-specific issues stay in adapters
- ✅ **Future-proof** - Easy to add new exchanges (Binance, IBKR, etc.)

