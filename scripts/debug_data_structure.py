#!/usr/bin/env python3
"""
Debug Data Structure Script
===========================

Quick script to check the structure of historical data files.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def check_data_structure(symbol: str = "BTC"):
    """Check the structure of historical data files"""
    print(f"Checking data structure for {symbol}...")
    
    # Check 1-minute data
    data_file = Path(f"data/{symbol}_1m_historical.parquet")
    if data_file.exists():
        print(f"\n1-minute data file exists: {data_file}")
        try:
            df = pd.read_parquet(data_file)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Index name: {df.index.name}")
            print(f"Index type: {type(df.index)}")
            print(f"Index dtype: {df.index.dtype}")
            
            if not df.empty:
                print(f"\nFirst 3 rows:")
                print(df.head(3))
                
                print(f"\nIndex values (first 3):")
                print(df.index[:3])
                
                # Check if index is datetime
                if pd.api.types.is_datetime64_any_dtype(df.index):
                    print(f"Index is datetime: True")
                    print(f"Index timezone: {df.index.tz}")
                else:
                    print(f"Index is datetime: False")
            else:
                print("DataFrame is empty")
                
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print(f"1-minute data file not found: {data_file}")
    
    # Check regular historical data
    data_file2 = Path(f"data/crypto_db/{symbol}_historical.parquet")
    if data_file2.exists():
        print(f"\nRegular historical data file exists: {data_file2}")
        try:
            df2 = pd.read_parquet(data_file2)
            print(f"Shape: {df2.shape}")
            print(f"Columns: {list(df2.columns)}")
            print(f"Index name: {df2.index.name}")
            print(f"Index type: {type(df2.index)}")
            print(f"Index dtype: {df2.index.dtype}")
            
            if not df2.empty:
                print(f"\nFirst 3 rows:")
                print(df2.head(3))
                
                print(f"\nIndex values (first 3):")
                print(df2.index[:3])
                
                # Check if index is datetime
                if pd.api.types.is_datetime64_any_dtype(df2.index):
                    print(f"Index is datetime: True")
                    print(f"Index timezone: {df2.index.tz}")
                else:
                    print(f"Index is datetime: False")
            else:
                print("DataFrame is empty")
                
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print(f"Regular historical data file not found: {data_file2}")

if __name__ == "__main__":
    check_data_structure("BTC")
