#!/usr/bin/env python3
"""
Test script to verify ta library is working correctly on Windows
"""

import pandas as pd
import numpy as np
from feature_engineering.technical_indicators import IndicatorCalculator

def test_ta_library():
    """Test ta library functionality"""
    print("🧪 Testing ta library on Windows...")
    
    try:
        # Test basic import
        import ta
        print("✅ ta library imported successfully")
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        df = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + np.random.rand(100) * 2,
            'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - np.random.rand(100) * 2,
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        print("✅ Sample data created")
        
        # Test individual indicators
        print("\n📊 Testing individual indicators:")
        
        # RSI
        rsi = ta.momentum.rsi(df['close'])
        print(f"  RSI: {rsi.iloc[-1]:.2f}")
        
        # MACD
        macd = ta.trend.macd(df['close'])
        macd_signal = ta.trend.macd_signal(df['close'])
        print(f"  MACD: {macd.iloc[-1]:.4f}")
        print(f"  MACD Signal: {macd_signal.iloc[-1]:.4f}")
        
        # Bollinger Bands
        bb_upper = ta.volatility.bollinger_hband(df['close'])
        bb_middle = ta.volatility.bollinger_mavg(df['close'])
        bb_lower = ta.volatility.bollinger_lband(df['close'])
        print(f"  Bollinger Upper: {bb_upper.iloc[-1]:.2f}")
        print(f"  Bollinger Middle: {bb_middle.iloc[-1]:.2f}")
        print(f"  Bollinger Lower: {bb_lower.iloc[-1]:.2f}")
        
        # Moving Averages
        sma_20 = ta.trend.sma_indicator(df['close'], window=20)
        ema_12 = ta.trend.ema_indicator(df['close'], window=12)
        print(f"  SMA 20: {sma_20.iloc[-1]:.2f}")
        print(f"  EMA 12: {ema_12.iloc[-1]:.2f}")
        
        # Volume indicators
        obv = ta.volume.on_balance_volume(df['close'], df['volume'])
        mfi = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        print(f"  OBV: {obv.iloc[-1]:.0f}")
        print(f"  MFI: {mfi.iloc[-1]:.2f}")
        
        print("\n✅ All individual indicators working!")
        
        # Test our custom calculator
        print("\n🔧 Testing custom IndicatorCalculator:")
        calculator = IndicatorCalculator()
        indicators = calculator.calculate_all_indicators(df)
        
        print(f"  RSI: {indicators.rsi:.2f}")
        print(f"  MACD: {indicators.macd:.4f}")
        print(f"  Bollinger Position: {indicators.bollinger_position:.2f}")
        print(f"  Market Regime: {indicators.market_regime}")
        print(f"  Trend Strength: {indicators.trend_strength:.2f}")
        
        print("\n✅ Custom IndicatorCalculator working!")
        
        # Test signal generation
        print("\n📈 Testing signal generation:")
        from feature_engineering.technical_indicators import TechnicalSignalGenerator
        
        signal_generator = TechnicalSignalGenerator()
        signals = signal_generator.generate_signals(df)
        
        print(f"  Overall Signal: {signals['overall_signal']}")
        print(f"  Momentum Signals: {len(signals['momentum_signals'])}")
        print(f"  Trend Signals: {len(signals['trend_signals'])}")
        
        print("\n✅ Signal generation working!")
        
        print("\n🎉 All tests passed! ta library is working perfectly on Windows!")
        return True
        
    except ImportError as e:
        print(f"❌ Error importing ta library: {e}")
        print("💡 Try: pip install ta")
        return False
        
    except Exception as e:
        print(f"❌ Error testing ta library: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Enhanced Crypto Trading Pipeline - Windows ta Library Test")
    print("=" * 60)
    
    success = test_ta_library()
    
    if success:
        print("\n✅ SUCCESS: ta library is working correctly!")
        print("🚀 You can now run the full pipeline:")
        print("   python run_pipeline.py --mode full")
    else:
        print("\n❌ FAILED: ta library is not working")
        print("💡 Try running: pip install ta")
        print("💡 Or run: python setup_windows.py")

if __name__ == "__main__":
    main()
