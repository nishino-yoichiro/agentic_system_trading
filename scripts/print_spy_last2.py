import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_ingestion.unified_data_storage import UnifiedDataStorage


def main():
    storage = UnifiedDataStorage(market_type='equities')
    df = storage.load_historical('SPY')
    if df is None or df.empty:
        print("No SPY historical data found.")
        return
    print(df.tail(2).to_string())


if __name__ == "__main__":
    main()



