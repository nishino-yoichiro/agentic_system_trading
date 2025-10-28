# Async vs Sync Code - Explained Simply

## Quick Answer

**Synchronous (sync)**: One task at a time, must wait for each to finish
**Asynchronous (async)**: Can start multiple tasks and handle them as they complete

---

## Visual Comparison

### Synchronous Code (Blocking)
```
Task 1: ──────────────────> DONE
Task 2:  ─────────────────────> DONE
Task 3:   ───────────────────────> DONE

Time: │────────────────────────────────────────┤
```
**Problem**: If Task 1 takes 5 seconds, you must wait 5 seconds before Task 2 even starts.

### Asynchronous Code (Non-Blocking)
```
Task 1: ──────────────────> DONE
Task 2: ───────> DONE
Task 3: ─────────> DONE

Time: │────────────────────────┤
```
**Benefit**: All 3 tasks can run at the same time. Total time = longest task, not sum of all tasks.

---

## Real Example from Your Code

### The Problem We Kept Hitting

```python
# ❌ BAD: Trying to call async from sync
def get_historical_data_sync(self):  # This is NOT async
    df = await self.load_data()  # ERROR: Can't use 'await' in non-async function!
```

### The Solutions

#### Option 1: Make Everything Async
```python
# ✅ GOOD: Both are async
async def get_historical_data(self):
    df = await self.load_data()  # Works!
```

#### Option 2: Run Async from Sync
```python
# ✅ GOOD: Wrapper that runs async
def get_historical_data_sync(self):
    import asyncio
    df = asyncio.run(self.load_data())  # Convert async to sync
```

#### Option 3: Await if Already in Async Context
```python
# ✅ GOOD: If caller is async
async def some_main_function():
    df = await self.load_data()  # Works!
```

---

## Why Your WebSocket Code Uses Async

### The Problem WebSockets Solve

When you connect to Coinbase/Alpaca WebSocket, you're waiting for **messages to arrive**. 

**Synchronous approach**:
```python
# ❌ BLOCKS the entire program
message = websocket.receive()  # Wait here... 1 second... 2 seconds...
process(message)  # Can't do anything else while waiting!
```

**Asynchronous approach**:
```python
# ✅ Can do other things while waiting
async for message in websocket.stream():
    process(message)  # Only handles message when it arrives
```

### Why This Matters for Trading

Your trading bot needs to:
1. Monitor live market data (WebSocket)
2. Update signals (calculations)
3. Check risk limits
4. Place orders

**Sync**: These happen one at a time (slow, might miss data)
**Async**: All happen concurrently (fast, catches everything)

---

## Why We Kept Running Into Issues

### Issue 1: Mixing Sync and Async Methods

```python
class CoinbaseAdapter:
    # ❌ Started as sync method
    def load_historical_data(self):
        df = await fetch_from_api()  # ERROR!
        
    # ✅ Fixed: Made async
    async def load_historical_data(self):
        df = await fetch_from_api()  # Works!
```

### Issue 2: Calling Async from Sync Context

```python
# ❌ In example script that runs at top level
def main():
    df = await adapter.load_historical_data()  # ERROR!

# ✅ Fixed: Wrap in asyncio.run()
async def main():
    df = await adapter.load_historical_data()

if __name__ == '__main__':
    asyncio.run(main())
```

### Issue 3: Double-Wrapping

```python
# ❌ Already in async, trying to run again
async def fetch_data(self):
    result = asyncio.run(other_async_function())  # ERROR!
    # You're already in an async context!

# ✅ Fixed: Just await it
async def fetch_data(self):
    result = await other_async_function()  # Works!
```

---

## How to Tell What's Async

### Methods/Functions

**Synchronous**:
```python
def calculate_ma(df):
    return df.rolling(20).mean()

# Call it normally:
result = calculate_ma(df)
```

**Asynchronous**:
```python
async def fetch_live_data():
    async for bar in websocket.stream():
        yield bar

# Must use await:
result = await fetch_live_data()
# OR use async for:
async for bar in fetch_live_data():
    process(bar)
```

### Libraries

**Sync APIs** (Alpaca historical client):
```python
from alpaca.data import StockHistoricalDataClient

client = StockHistoricalDataClient()
bars = client.get_stock_bars(request)  # BLOCKS until done
```

**Async APIs** (Alpaca data stream):
```python
from alpaca.data.live import StockDataStream

stream = StockDataStream()
async for trade in stream.subscribe_trades():  # Non-blocking
    process(trade)
```

---

## Common Patterns in Your Code

### Pattern 1: WebSocket Streaming

```python
async def stream_data(self):
    async for bar in websocket:  # Wait for bars without blocking
        yield bar  # Yield when available
```

### Pattern 2: Converting Sync to Async

```python
async def _fetch_data(self):
    # Alpaca client is sync-only, so wrap it
    def sync_call():
        return self.client.get_data()
    
    # Run sync function in thread pool (non-blocking)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, sync_call)
    return result
```

### Pattern 3: Main Entry Points

```python
# ✅ Good pattern
async def main():
    adapter = CoinbaseAdapter()
    await adapter.connect()
    async for bar in adapter.stream_data():
        process(bar)

if __name__ == '__main__':
    asyncio.run(main())  # Entry point
```

---

## When to Use What

### Use Synchronous When:
- Simple calculations (math, pandas operations)
- Reading from local files
- Working with data that's already loaded
- Any operation that completes instantly

### Use Asynchronous When:
- Network operations (API calls, WebSockets)
- Database queries
- File I/O (reading/writing large files)
- Any operation that might "wait" for something

---

## The Rules of Async

### Rule 1: You Can't Await in Sync Context
```python
# ❌ Won't work
def my_function():
    result = await some_async_function()  # ERROR!
```

### Rule 2: You Can Call Sync from Async (But Don't Block!)
```python
# ✅ OK
async def my_async_function():
    result = some_sync_function()  # Works, but blocks!

# ✅ BETTER
async def my_async_function():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, some_sync_function)
```

### Rule 3: Any Function That Uses `await` Must Be `async`
```python
async def fetch_data(self):  # Must be async
    data = await api_call()  # Now can use await
```

---

## Why Your Project Uses Async

### Crypto/Stock Data Streaming

```python
# Real-time data comes continuously
# You need to:
1. Receive updates from WebSocket (async)
2. Aggregate into bars (can be sync once data received)
3. Update signals (can be sync)
4. Make trading decisions (can be sync)
5. Place orders via API (async)

# Async lets you do #1 and #5 while #2,3,4 run!
```

### The Alternative (Synchronous Hell)

```python
def main():
    while True:
        bar = websocket.receive()  # Blocks here waiting...
        process(bar)  # Finally got here
        # If 10 seconds pass without data, you've wasted 10 seconds!
```

---

## Summary

**Synchronous (sync)**:
- One thing at a time
- Can't use `await`
- Use for: simple operations, calculations

**Asynchronous (async)**:
- Can handle multiple things concurrently
- Must use `await` to wait for results
- Use for: network calls, WebSockets, I/O

**The Mixing Problem**:
- Can't use `await` in sync functions
- Can't call async without `await` or `asyncio.run()`
- WebSockets require async (continuous data stream)

**Your Project**:
- Adapters stream live data → Need async
- Backtester works on loaded data → Can be sync
- Signal generation works on dataframe → Can be sync
- Order placement makes API calls → Need async

