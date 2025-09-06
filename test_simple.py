#!/usr/bin/env python3
"""
Test simple pipeline components (Windows Compatible)
"""

import asyncio
import pandas as pd
import numpy as np
from feature_engineering.nlp_processor_simple import SimpleNLPProcessor, SimpleSentimentAnalyzer
from feature_engineering.technical_indicators import IndicatorCalculator
from loguru import logger

async def test_nlp():
    print("Testing Simple NLP Processor...")
    try:
        nlp = SimpleNLPProcessor()
        await nlp.initialize()
        
        # Test different types of text
        test_texts = [
            "Bitcoin is surging to new all-time highs!",
            "The crypto market is crashing hard today.",
            "Tesla stock is showing mixed signals.",
            "Market volatility is increasing significantly."
        ]
        
        for text in test_texts:
            analysis = await nlp.process_text(text)
            print(f"  '{text[:30]}...' -> {analysis.sentiment_label} ({analysis.sentiment_score:.2f})")
        
        print("✅ NLP Processor working!")
        return True
    except Exception as e:
        print(f"❌ NLP Processor failed: {e}")
        return False

def test_technical_indicators():
    print("Testing Technical Indicators...")
    try:
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
        
        print(f"  RSI: {indicators.rsi:.2f}")
        print(f"  MACD: {indicators.macd:.4f}")
        print(f"  Bollinger Position: {indicators.bollinger_position:.2f}")
        print(f"  Market Regime: {indicators.market_regime}")
        
        print("✅ Technical Indicators working!")
        return True
    except Exception as e:
        print(f"❌ Technical Indicators failed: {e}")
        return False

async def test_sentiment_analyzer():
    print("Testing Sentiment Analyzer...")
    try:
        analyzer = SimpleSentimentAnalyzer()
        
        test_text = "Bitcoin is surging to new all-time highs! The crypto market is showing strong bullish momentum."
        result = await analyzer.analyze_sentiment(test_text)
        
        print(f"  Direction: {result['direction']}")
        print(f"  Strength: {result['strength']:.2f}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Crypto mentions: {result['crypto_mentions']}")
        
        print("✅ Sentiment Analyzer working!")
        return True
    except Exception as e:
        print(f"❌ Sentiment Analyzer failed: {e}")
        return False

async def main():
    print("🧪 Testing Simple Enhanced Crypto Trading Pipeline Components\n")
    
    # Test components
    nlp_success = await test_nlp()
    tech_success = test_technical_indicators()
    sentiment_success = await test_sentiment_analyzer()
    
    print("\n" + "="*50)
    if nlp_success and tech_success and sentiment_success:
        print("🎉 ALL TESTS PASSED! Simple pipeline is working correctly!")
        print("\n🚀 You can now run:")
        print("   python run_simple.py")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("\n💡 Try running: python setup_simple.py")

if __name__ == "__main__":
    asyncio.run(main())
