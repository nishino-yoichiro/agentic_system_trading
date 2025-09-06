#!/usr/bin/env python3
"""
Simple script to run the Enhanced Crypto Trading Pipeline (Windows Compatible)
Uses lightweight dependencies only
"""

import asyncio
import pandas as pd
import numpy as np
from feature_engineering.nlp_processor_simple import SimpleNLPProcessor
from feature_engineering.technical_indicators import IndicatorCalculator
import yaml
from loguru import logger

async def test_simple_pipeline():
    """Test the simple pipeline components"""
    print("🚀 Testing Simple Enhanced Crypto Trading Pipeline\n")
    
    # Test NLP Processor
    print("1. Testing Simple NLP Processor...")
    nlp = SimpleNLPProcessor()
    await nlp.initialize()
    
    test_text = "Bitcoin is surging to new all-time highs! The crypto market is showing strong bullish momentum."
    analysis = await nlp.process_text(test_text)
    
    print(f"   ✓ Sentiment: {analysis.sentiment_label} ({analysis.sentiment_score:.2f})")
    print(f"   ✓ VADER: {analysis.vader_compound:.2f}")
    print(f"   ✓ Crypto mentions: {analysis.crypto_mentions}")
    
    # Test Technical Indicators
    print("\n2. Testing Technical Indicators (ta library)...")
    calculator = IndicatorCalculator()
    
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
    
    indicators = calculator.calculate_all_indicators(df)
    
    print(f"   ✓ RSI: {indicators.rsi:.2f}")
    print(f"   ✓ MACD: {indicators.macd:.4f}")
    print(f"   ✓ Bollinger Position: {indicators.bollinger_position:.2f}")
    print(f"   ✓ Market Regime: {indicators.market_regime}")
    
    # Test Signal Generation
    print("\n3. Testing Signal Generation...")
    from feature_engineering.technical_indicators import TechnicalSignalGenerator
    
    signal_generator = TechnicalSignalGenerator()
    signals = signal_generator.generate_signals(df)
    
    print(f"   ✓ Overall Signal: {signals['overall_signal']}")
    print(f"   ✓ Momentum Signals: {len(signals['momentum_signals'])}")
    
    print("\n✅ Simple pipeline test completed successfully!")
    print("\n📝 This version uses:")
    print("   - ta library (instead of talib)")
    print("   - TextBlob + VADER (instead of spaCy)")
    print("   - Simple dependencies (no C compilation)")
    print("   - SQLite database (instead of MongoDB)")

if __name__ == "__main__":
    asyncio.run(test_simple_pipeline())
