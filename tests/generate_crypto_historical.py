#!/usr/bin/env python3
"""
Generate Historical Crypto Data for Testing

This script generates realistic minute-level historical data for BTC and ETH
to test technical indicators properly.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_ingestion.coinbase_rest_client import create_coinbase_rest_client

def generate_minute_data(symbol: str, current_price: float, days: int = 30) -> pd.DataFrame:
    """Generate realistic minute-level data for testing"""
    
    # Generate timestamps for the last N days, minute by minute
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Generate realistic price movements
    n_points = len(timestamps)
    
    # Start with current price and work backwards
    prices = np.zeros(n_points)
    prices[-1] = current_price  # Most recent price
    
    # Generate random walk with drift (crypto tends to have upward bias)
    # Use realistic volatility for crypto (higher than stocks)
    volatility = 0.02 if symbol == 'BTC' else 0.025  # 2-2.5% per minute volatility
    drift = 0.0001  # Slight upward bias
    
    # Generate returns (log returns)
    returns = np.random.normal(drift, volatility, n_points-1)
    
    # Work backwards from current price
    for i in range(n_points-2, -1, -1):
        prices[i] = prices[i+1] * np.exp(-returns[i])
    
    # Generate OHLC data
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        # Generate realistic OHLC from the price
        # Add some intraday volatility
        intraday_vol = price * 0.001  # 0.1% intraday volatility
        
        high = price + np.random.exponential(intraday_vol)
        low = price - np.random.exponential(intraday_vol)
        
        # Ensure OHLC relationships are correct
        high = max(high, price)
        low = min(low, price)
        
        # Open is previous close (or current price for first point)
        open_price = prices[i-1] if i > 0 else price
        
        # Generate volume (random but realistic)
        volume = np.random.exponential(1000) * (1 + np.random.normal(0, 0.5))
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': max(0, volume)
        })
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    return df

async def main():
    """Generate historical crypto data"""
    print("🚀 Generating Historical Crypto Data")
    print("=" * 50)
    
    # Get current prices from Coinbase
    client = create_coinbase_rest_client()
    
    symbols = ['BTC', 'ETH']
    current_prices = await client.get_multiple_prices(symbols)
    
    if not current_prices:
        print("❌ Failed to get current prices")
        return
    
    print(f"📊 Current prices:")
    for symbol, price in current_prices.items():
        print(f"   {symbol}: ${price:,.2f}")
    
    # Generate historical data
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📈 Generating 30 days of minute-level data...")
    
    for symbol, current_price in current_prices.items():
        print(f"   Generating data for {symbol}...")
        
        # Generate 30 days of minute data
        df = generate_minute_data(symbol, current_price, days=30)
        
        # Save to parquet
        file_path = data_dir / f"prices_{symbol}.parquet"
        df.to_parquet(file_path)
        
        print(f"   ✅ Saved {len(df):,} data points to {file_path}")
        print(f"   📊 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
    
    print(f"\n🎉 Historical data generation complete!")
    print(f"   Files saved in: {data_dir}")
    print(f"   Ready for technical indicator testing!")

if __name__ == "__main__":
    asyncio.run(main())

