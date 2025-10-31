import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adapters import AlpacaMarketAdapter


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


async def main():
    setup_logging()
    adapter = AlpacaMarketAdapter('SPY', paper=True)
    await adapter.connect()

    print("\n[TEST] Fetching 20 days of SPY historical data...")
    df = await adapter.load_historical_data(days=20)

    if df is not None and not df.empty:
        print(f"Loaded bars: {len(df):,}")
        print(f"Range: {df.index[0]} -> {df.index[-1]}")
    else:
        print('No data returned')

    await adapter.disconnect()


if __name__ == "__main__":
    asyncio.run(main())



