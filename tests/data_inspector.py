#!/usr/bin/env python3
"""
Data Inspector - Price Data Analysis and Visualization Tool

This tool allows you to inspect, analyze, and visualize price data from the pipeline
to verify accuracy against online sources.

Usage:
    python tests/data_inspector.py --symbol BTC --action display
    python tests/data_inspector.py --symbol BTC --action chart
    python tests/data_inspector.py --symbol BTC --action range --start 2024-01-01 --end 2024-01-31
    python tests/data_inspector.py --symbol BTC --action validate
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class DataInspector:
    """Data inspector for price data analysis and visualization"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        
    def load_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load price data for a symbol"""
        file_path = self.raw_data_dir / f"prices_{symbol}.parquet"
        
        if not file_path.exists():
            print(f"❌ No data found for symbol: {symbol}")
            print(f"   Expected file: {file_path}")
            print(f"   Available files:")
            for f in self.raw_data_dir.glob("prices_*.parquet"):
                print(f"   - {f.name}")
            return None
            
        try:
            df = pd.read_parquet(file_path)
            
            # Set date/timestamp column as index if it exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            # Handle Coinbase data format (single row with current price)
            if len(df) == 1 and 'close' in df.columns:
                print(f"✅ Loaded current price data for {symbol}: ${df['close'].iloc[0]:,.2f}")
            else:
                print(f"✅ Loaded {len(df):,} data points for {symbol}")
            
            return df
        except Exception as e:
            print(f"❌ Error loading data for {symbol}: {e}")
            return None
    
    def display_data_info(self, df: pd.DataFrame, symbol: str):
        """Display basic information about the data"""
        print(f"\n📊 DATA INFO FOR {symbol}")
        print("=" * 50)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Time span: {df.index.max() - df.index.min()}")
        print(f"Data frequency: {self._get_frequency(df)}")
        
        print(f"\n📈 PRICE STATISTICS")
        print("-" * 30)
        if 'close' in df.columns:
            print(f"Close price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            print(f"Current close: ${df['close'].iloc[-1]:.2f}")
            print(f"Price change: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
        
        print(f"\n📋 SAMPLE DATA (Last 5 rows)")
        print("-" * 30)
        print(df.tail())
        
    def display_data_range(self, df: pd.DataFrame, symbol: str, start_date: str, end_date: str):
        """Display data for a specific date range"""
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter data
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            filtered_df = df[mask]
            
            if filtered_df.empty:
                print(f"❌ No data found for {symbol} between {start_date} and {end_date}")
                return
                
            print(f"\n📊 DATA FOR {symbol} ({start_date} to {end_date})")
            print("=" * 60)
            print(f"Data points: {len(filtered_df):,}")
            print(f"Date range: {filtered_df.index.min()} to {filtered_df.index.max()}")
            
            if 'close' in filtered_df.columns:
                print(f"Close range: ${filtered_df['close'].min():.2f} - ${filtered_df['close'].max():.2f}")
                print(f"Price change: {((filtered_df['close'].iloc[-1] / filtered_df['close'].iloc[0]) - 1) * 100:.2f}%")
            
            print(f"\n📋 DATA")
            print("-" * 30)
            print(filtered_df)
            
        except Exception as e:
            print(f"❌ Error filtering data: {e}")
    
    def create_chart(self, df: pd.DataFrame, symbol: str, chart_type: str = "price"):
        """Create price charts"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{symbol} Price Data Analysis', fontsize=16, fontweight='bold')
        
        if 'close' in df.columns:
            # Price chart
            axes[0, 0].plot(df.index, df['close'], linewidth=1, alpha=0.8)
            axes[0, 0].set_title(f'{symbol} Close Price')
            axes[0, 0].set_ylabel('Price ($)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Volume chart (if available)
            if 'volume' in df.columns:
                axes[0, 1].bar(df.index, df['volume'], alpha=0.7, width=0.8)
                axes[0, 1].set_title(f'{symbol} Volume')
                axes[0, 1].set_ylabel('Volume')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'No volume data available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title(f'{symbol} Volume (N/A)')
            
            # Price distribution
            axes[1, 0].hist(df['close'], bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title(f'{symbol} Price Distribution')
            axes[1, 0].set_xlabel('Price ($)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Daily returns
            returns = df['close'].pct_change().dropna()
            axes[1, 1].plot(df.index[1:], returns, linewidth=0.8, alpha=0.8)
            axes[1, 1].set_title(f'{symbol} Daily Returns')
            axes[1, 1].set_ylabel('Returns')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = f"tests/{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved: {chart_path}")
        
        plt.show()
    
    def validate_data_quality(self, df: pd.DataFrame, symbol: str):
        """Validate data quality and identify potential issues"""
        print(f"\n🔍 DATA QUALITY VALIDATION FOR {symbol}")
        print("=" * 50)
        
        issues = []
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.any():
            print(f"⚠️  Missing values found:")
            for col, count in missing_data[missing_data > 0].items():
                print(f"   {col}: {count} missing values")
                issues.append(f"Missing values in {col}")
        else:
            print("✅ No missing values")
        
        # Check for duplicate timestamps
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            print(f"⚠️  {duplicates} duplicate timestamps found")
            issues.append(f"{duplicates} duplicate timestamps")
        else:
            print("✅ No duplicate timestamps")
        
        # Check for price anomalies
        if 'close' in df.columns:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = price_changes > 0.2  # 20% change
            if extreme_changes.any():
                extreme_count = extreme_changes.sum()
                print(f"⚠️  {extreme_count} extreme price changes (>20%)")
                issues.append(f"{extreme_count} extreme price changes")
            else:
                print("✅ No extreme price changes")
        
        # Check data frequency consistency
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            most_common_freq = time_diffs.mode().iloc[0] if not time_diffs.empty else None
            if most_common_freq:
                inconsistent = (time_diffs != most_common_freq).sum()
                if inconsistent > 0:
                    print(f"⚠️  {inconsistent} inconsistent time intervals")
                    issues.append(f"{inconsistent} inconsistent time intervals")
                else:
                    print("✅ Consistent time intervals")
        
        # Summary
        if issues:
            print(f"\n❌ Found {len(issues)} data quality issues:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        else:
            print(f"\n✅ Data quality looks good!")
    
    def compare_with_online(self, df: pd.DataFrame, symbol: str):
        """Compare data with online sources (placeholder for future implementation)"""
        print(f"\n🌐 ONLINE COMPARISON FOR {symbol}")
        print("=" * 50)
        print("📝 To verify data accuracy, compare with:")
        print(f"   • CoinGecko: https://www.coingecko.com/en/coins/{symbol.lower()}")
        print(f"   • Yahoo Finance: https://finance.yahoo.com/quote/{symbol}-USD")
        print(f"   • TradingView: https://www.tradingview.com/symbols/{symbol}USD/")
        
        if 'close' in df.columns:
            latest_price = df['close'].iloc[-1]
            latest_time = df.index[-1]
            print(f"\n📊 Latest data point:")
            print(f"   Price: ${latest_price:.2f}")
            print(f"   Time: {latest_time}")
            print(f"   Compare this with online sources above")
    
    def calculate_indicators(self, df: pd.DataFrame, symbol: str):
        """Calculate and display technical indicators"""
        print(f"\n📈 TECHNICAL INDICATORS FOR {symbol}")
        print("=" * 50)
        
        if df.empty:
            print("❌ No data to analyze")
            return
        
        if len(df) < 20:
            print(f"❌ Insufficient data: {len(df)} points (minimum: 20)")
            return
        
        try:
            # Import the technical indicators calculator
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            sys.path.insert(0, str(project_root))
            
            from feature_engineering.technical_indicators_ta import IndicatorCalculator
            
            # Calculate indicators
            calculator = IndicatorCalculator()
            indicators = calculator.calculate_all_indicators(df)
            
            print(f"✅ Calculated technical indicators successfully")
            print(f"\n📊 INDICATOR SUMMARY:")
            print("-" * 30)
            
            # Display key indicators
            if hasattr(indicators, 'rsi'):
                print(f"RSI (14): {indicators.rsi:.2f}")
            if hasattr(indicators, 'macd'):
                print(f"MACD: {indicators.macd:.4f}")
            if hasattr(indicators, 'macd_signal'):
                print(f"MACD Signal: {indicators.macd_signal:.4f}")
            if hasattr(indicators, 'macd_histogram'):
                print(f"MACD Histogram: {indicators.macd_histogram:.4f}")
            if hasattr(indicators, 'bb_upper'):
                print(f"Bollinger Upper: {indicators.bb_upper:.2f}")
            if hasattr(indicators, 'bb_lower'):
                print(f"Bollinger Lower: {indicators.bb_lower:.2f}")
            if hasattr(indicators, 'sma_20'):
                print(f"SMA (20): {indicators.sma_20:.2f}")
            if hasattr(indicators, 'ema_12'):
                print(f"EMA (12): {indicators.ema_12:.2f}")
            if hasattr(indicators, 'ema_26'):
                print(f"EMA (26): {indicators.ema_26:.2f}")
            
            print(f"\n🎉 Technical analysis complete!")
            
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_frequency(self, df: pd.DataFrame) -> str:
        """Determine data frequency"""
        if len(df) < 2:
            return "Unknown"
        
        try:
            time_diffs = df.index.to_series().diff().dropna()
            if time_diffs.empty:
                return "Unknown"
                
            # Convert to timedelta if not already
            if not isinstance(time_diffs.iloc[0], pd.Timedelta):
                time_diffs = pd.to_timedelta(time_diffs)
            
            most_common = time_diffs.mode().iloc[0] if not time_diffs.empty else None
            
            if most_common:
                if most_common <= pd.Timedelta(minutes=1):
                    return "1-minute"
                elif most_common <= pd.Timedelta(minutes=5):
                    return "5-minute"
                elif most_common <= pd.Timedelta(hours=1):
                    return "1-hour"
                elif most_common <= pd.Timedelta(days=1):
                    return "Daily"
                else:
                    return f"{most_common}"
        except Exception:
            pass
            
        return "Unknown"

def main():
    parser = argparse.ArgumentParser(description="Data Inspector - Price Data Analysis Tool")
    parser.add_argument('--symbol', required=True, help='Symbol to analyze (e.g., BTC, ETH, AAPL)')
    parser.add_argument('--action', choices=['display', 'chart', 'range', 'validate', 'compare', 'indicators'], 
                       default='display', help='Action to perform')
    parser.add_argument('--start', help='Start date for range analysis (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date for range analysis (YYYY-MM-DD)')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    
    args = parser.parse_args()
    
    # Initialize inspector
    inspector = DataInspector(args.data_dir)
    
    # Load data
    df = inspector.load_price_data(args.symbol)
    if df is None:
        return
    
    # Perform requested action
    if args.action == 'display':
        inspector.display_data_info(df, args.symbol)
        
    elif args.action == 'chart':
        inspector.create_chart(df, args.symbol)
        
    elif args.action == 'range':
        if not args.start or not args.end:
            print("❌ --start and --end dates are required for range analysis")
            return
        inspector.display_data_range(df, args.symbol, args.start, args.end)
        
    elif args.action == 'validate':
        inspector.validate_data_quality(df, args.symbol)
        
    elif args.action == 'compare':
        inspector.compare_with_online(df, args.symbol)
        
    elif args.action == 'indicators':
        inspector.calculate_indicators(df, args.symbol)

if __name__ == "__main__":
    main()
