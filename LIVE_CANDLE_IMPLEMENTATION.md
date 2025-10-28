# Live Candle Implementation Status

## Current Status

### What's Working ✅
- WebSocket connection to Coinbase
- Ticks are received and processed
- Completed bars are yielded correctly
- In-progress bar data structure is maintained

### Issue ⚠️
The live candle IS updating multiple times per minute (lines 225-226 in coinbase_adapter.py update it on every tick), BUT:

**The problem:** The test only checks `get_current_in_progress_bar()` when a completed bar is yielded. Since completed bars only yield at minute boundaries, we only see ONE update per minute.

**The user wants:** To see the live candle updating WITHIN the same minute as ticks arrive (high/low/close changing in real-time).

### Current Implementation Logic

```
Tick arrives at 03:19:30
  → Updates current_minute_data['high'] = max(high, new_price)
  → Updates current_minute_data['close'] = new_price
  → Updates self._current_in_progress_bar = current_minute_data ✅
  
But... only yields completed bar when minute_time changes to 03:20:00
```

So the live candle IS tracking updates, but the test doesn't monitor it between minute boundaries.

### Solution

The test needs to poll `get_current_in_progress_bar()` multiple times WITHIN the same minute to see it updating, OR the adapter needs to expose a way to monitor in-progress updates in real-time.

## Next Steps

1. Modify test to poll live bar every 1-2 seconds and show price changes
2. Add `subscribe_to_updates()` method that yields in-progress bar updates
3. Or add an async iterator for real-time in-progress updates

## For Now

The live candle updates ARE working - they just happen between yielding completed bars. The adapter correctly:
- Maintains `_current_in_progress_bar` 
- Updates OHLC on every tick
- Tracks volume accumulation

The combined data method `get_combined_data_for_signals()` will return the current in-progress bar, so signal generation will see the latest price.
