import asyncio
from pathlib import Path
import sys
sys.path.insert(0, str(Path('src')))
from adapters import AlpacaMarketAdapter
from data_ingestion.unified_data_storage import UnifiedDataStorage
from datetime import timezone

async def main():
    a = AlpacaMarketAdapter('SPY', paper=True)
    await a.connect()

    print('\n[1] HISTORICAL (14 days)')
    df_hist = await a.load_historical_data(days=14)
    print('Loaded:', 0 if df_hist is None else len(df_hist))

    print('\n[2] STREAM COMPLETED 1-MINUTE BARS (2 bars)')
    count=0
    async def consume():
        nonlocal count
        async for bar in a.stream_data():
            count+=1
            print('Completed bar', count, bar.timestamp, bar.close)
            if count>=2:
                break
    task = asyncio.create_task(consume())

    print('\n[3] LIVE IN-PROGRESS (check every 20s x3)')
    for i in range(3):
        await asyncio.sleep(20)
        live = a.get_current_in_progress_bar()
        if live:
            print('Live', i+1, live.timestamp, live.open, live.high, live.low, live.close)
        else:
            print('Live', i+1, 'None')

    task.cancel()
    try:
        await task
    except:
        pass

    print('\n[4] COMBINED')
    combined = a.get_combined_data_for_signals(days=14)
    print('Combined len:', 0 if combined is None else len(combined))

    await a.disconnect()

asyncio.run(main())
