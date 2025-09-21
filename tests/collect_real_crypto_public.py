#!/usr/bin/env python3
"""
Collect Real Crypto Historical Data using Public APIs

This script uses public APIs to collect real historical data without authentication.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def get_coinbase_historical_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Get historical data from Coinbase public API"""
    
    # Coinbase public API for historical data
    url = f"https://api.coinbase.com/v2/prices/{symbol}-USD/historic"
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        "start": start_date.isoformat(),
        "end": end_date.isoformat()
    }
    
    try:
        print(f"📊 Fetching historical data for {symbol} from Coinbase public API...")
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if "data" in data and "prices" in data["data"]:
                prices = data["data"]["prices"]
                
                # Convert to DataFrame
                df_data = []
                for price_point in prices:
                    try:
                        timestamp = pd.to_datetime(price_point['time'])
                        df_data.append({
                            'timestamp': timestamp,
                            'open': float(price_point['price']),
                            'high': float(price_point['price']),
                            'low': float(price_point['price']),
                            'close': float(price_point['price']),
                            'volume': 0  # Volume not available from public API
                        })
                    except (ValueError, KeyError) as e:
                        print(f"⚠️ Error parsing price point: {e}")
                        continue
                
                df = pd.DataFrame(df_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                df = df.sort_index()
                
                print(f"✅ Retrieved {len(df)} data points for {symbol}")
                return df
            else:
                print(f"❌ No price data found for {symbol}")
                return pd.DataFrame()
        else:
            print(f"❌ API error for {symbol}: {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

async def get_yahoo_finance_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Get historical data from Yahoo Finance (alternative source)"""
    try:
        import yfinance as yf
        
        print(f"📊 Fetching historical data for {symbol} from Yahoo Finance...")
        
        # Create ticker object
        ticker = yf.Ticker(f"{symbol}-USD")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        hist = ticker.history(start=start_date, end=end_date, interval="1m")
        
        if not hist.empty:
            # Rename columns to match our format
            hist = hist.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Reset index to get timestamp as column
            hist = hist.reset_index()
            hist = hist.rename(columns={'Datetime': 'timestamp'})
            hist['timestamp'] = pd.to_datetime(hist['timestamp'])
            hist = hist.set_index('timestamp')
            
            print(f"✅ Retrieved {len(hist)} data points for {symbol} from Yahoo Finance")
            return hist
        else:
            print(f"❌ No data found for {symbol} on Yahoo Finance")
            return pd.DataFrame()
            
    except ImportError:
        print("❌ yfinance not installed. Install with: pip install yfinance")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error fetching Yahoo Finance data for {symbol}: {e}")
        return pd.DataFrame()

async def collect_real_crypto_data():
    """Collect real historical crypto data"""
    print("🚀 Collecting Real Crypto Historical Data")
    print("=" * 50)
    
    symbols = ['BTC', 'ETH']
    data = {}
    
    for symbol in symbols:
        print(f"\n📈 Collecting data for {symbol}...")
        
        # Try Coinbase first
        df = await get_coinbase_historical_data(symbol, days=30)
        
        # If Coinbase fails, try Yahoo Finance
        if df.empty:
            print(f"   Coinbase failed, trying Yahoo Finance...")
            df = await get_yahoo_finance_data(symbol, days=30)
        
        if not df.empty:
            data[symbol] = df
            print(f"   ✅ {symbol}: {len(df)} data points")
            print(f"      Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            print(f"      Date range: {df.index.min()} to {df.index.max()}")
        else:
            print(f"   ❌ No data collected for {symbol}")
    
    if data:
        # Save data
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving data...")
        for symbol, df in data.items():
            file_path = data_dir / f"prices_{symbol}.parquet"
            df.to_parquet(file_path)
            print(f"   Saved {symbol} to {file_path}")
        
        print(f"\n🎉 Real crypto data collection complete!")
        print(f"   Collected data for {len(data)} symbols")
    else:
        print(f"\n❌ No data collected")

if __name__ == "__main__":
    asyncio.run(collect_real_crypto_data())

