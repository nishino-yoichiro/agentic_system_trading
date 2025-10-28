# Live In-Progress Bar Feature

## Problem
Previous WebSocket integration had a 1-minute lag because signals only used **completed** candles. When you call at 10:02, you get candle for 10:01 (last completed minute).

## Solution
The adapter now provides **3 layers of data**:

### 1. Historical Data (`{symbol}_historical.parquet`)
- Bulk historical OHLCV bars
- Loaded from Parquet files
- Used for backtesting and historical analysis

### 2. Completed Recent Bars (`{symbol}_1m.parquet`)
- Recently completed 1-minute bars
- Updated via WebSocket streaming
- Written to Parquet when minute completes

### 3. **Live In-Progress Bar** (`get_current_in_progress_bar()`)
- Current minute being built in real-time
- Updates with every WebSocket tick
- **This eliminates the 1-minute lag**

## Data Flow

```
WebSocket Ticks → Aggregate → Current In-Progress Bar
                                      ↓
                                   Update continuously
                                      ↓
                              signal generation uses:
                              Historical + Completed + In-Progress
```

## Methods

### `stream_data()` - Completed bars
- Yields completed 1-minute bars (lagging by ~1 minute)
- Used for writing completed bars to storage

### `get_current_in_progress_bar()` - Live bar
- Returns current in-progress OHLCVBar (no lag)
- Returns None if not streaming

### `get_combined_data_for_signals()` - **For signal generation**
- Returns: Historical + Completed + In-Progress (all 3 layers)
- **This is what should be used for generating live signals**
- No 1-minute lag - real-time signals

## Usage Example

```python
adapter = CoinbaseMarketAdapter("BTC")

# Start streaming (accumulates completed bars)
async for completed_bar in adapter.stream_data():
    print(f"Completed: {completed_bar.timestamp}")
    
    # Get current in-progress bar (zero lag)
    live_bar = adapter.get_current_in_progress_bar()
    if live_bar:
        print(f"Live: {live_bar.timestamp} - ${live_bar.close}")
    
    # For signal generation: use combined data
    combined_data = adapter.get_combined_data_for_signals(days=30)
    signals = generate_signals(combined_data)  # Real-time, no lag!
```

## When to Use

- **Backtesting**: Use `load_historical_data()` (historical only)
- **Live Trading**: Use `get_combined_data_for_signals()` (all 3 layers)
- **Monitoring**: Use `get_current_in_progress_bar()` for display

## Next Steps for Integration

When integrating into `CryptoSignalIntegration.generate_live_signals()`:

```python
# OLD (1-minute lag):
data = adapter.load_historical_data(days=30) + adapter.get_completed_bars()

# NEW (real-time):
data = adapter.get_combined_data_for_signals(days=30)  # Includes in-progress!
signals = framework.generate_signals({'BTC': data}, live_mode=True)
```

This change will be made when integrating adapters into the signal generation system.
