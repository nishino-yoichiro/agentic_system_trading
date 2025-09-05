#!/usr/bin/env python3
"""
Enhanced Crypto Trading Pipeline - Simple Windows Setup (No C Compilation)

This script sets up the enhanced crypto trading pipeline on Windows with
minimal dependencies that don't require C compilation.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml
from loguru import logger


def setup_environment():
    """Set up the Python environment with simple dependencies"""
    logger.info("Setting up Python environment with simple dependencies...")
    
    try:
        # Install simple requirements (no C compilation needed)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_simple.txt"])
        logger.info("✓ Simple dependencies installed successfully")
        
        # Install Playwright browsers (if needed)
        try:
            subprocess.check_call([sys.executable, "-m", "playwright", "install"])
            logger.info("✓ Playwright browsers installed")
        except Exception as e:
            logger.warning(f"Playwright installation failed: {e}")
            logger.info("You can install Playwright later if needed for web scraping")
        
        # Test key libraries
        test_libraries = [
            ("pandas", "import pandas as pd"),
            ("numpy", "import numpy as np"),
            ("ta", "import ta"),
            ("textblob", "from textblob import TextBlob"),
            ("vaderSentiment", "from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer"),
            ("requests", "import requests"),
            ("beautifulsoup4", "from bs4 import BeautifulSoup"),
            ("matplotlib", "import matplotlib.pyplot as plt"),
            ("fpdf2", "from fpdf import FPDF")
        ]
        
        for lib_name, import_statement in test_libraries:
            try:
                exec(import_statement)
                logger.info(f"✓ {lib_name} working correctly")
            except ImportError as e:
                logger.error(f"❌ {lib_name} not working: {e}")
                return False
        
        logger.info("✅ All simple dependencies working correctly!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    logger.info("Creating directory structure...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "reports",
        "charts",
        "logs",
        "config",
        "templates"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")
    
    return True


def setup_configuration():
    """Set up configuration files"""
    logger.info("Setting up configuration files...")
    
    # Create local API keys file if it doesn't exist
    api_keys_local = Path("config/api_keys_local.yaml")
    if not api_keys_local.exists():
        api_keys_template = Path("config/api_keys.yaml")
        if api_keys_template.exists():
            shutil.copy(api_keys_template, api_keys_local)
            logger.info("✓ Created config/api_keys_local.yaml (edit with your API keys)")
        else:
            # Create a simple template
            simple_api_keys = {
                'newsapi': {'api_key': 'your_newsapi_key_here'},
                'polygon': {'api_key': 'your_polygon_key_here'},
                'binance': {'api_key': 'your_binance_key_here', 'secret_key': 'your_binance_secret_here'},
                'coingecko': {'api_key': 'your_coingecko_key_here'}
            }
            
            with open(api_keys_local, 'w') as f:
                yaml.dump(simple_api_keys, f, default_flow_style=False)
            logger.info("✓ Created simple API keys template")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Enhanced Crypto Trading Pipeline Environment Variables (Simple Version)
# Copy this file and add your actual API keys

# News APIs
NEWSAPI_KEY=your_newsapi_key_here

# Financial Data APIs
POLYGON_API_KEY=your_polygon_key_here

# Crypto APIs
BINANCE_API_KEY=your_binance_key_here
BINANCE_SECRET_KEY=your_binance_secret_here
COINGECKO_API_KEY=your_coingecko_key_here

# Database (SQLite - no setup needed)
DATABASE_URL=sqlite:///crypto_trading.db

# Email (for reports)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        logger.info("✓ Created .env file (edit with your API keys)")
    
    return True


def create_simple_scripts():
    """Create simple scripts for common tasks"""
    logger.info("Creating simple scripts...")
    
    # Create run_simple.py
    simple_script = """#!/usr/bin/env python3
\"\"\"
Simple script to run the Enhanced Crypto Trading Pipeline (Windows Compatible)
Uses lightweight dependencies only
\"\"\"

import asyncio
import pandas as pd
import numpy as np
from feature_engineering.nlp_processor_simple import SimpleNLPProcessor
from feature_engineering.technical_indicators import IndicatorCalculator
import yaml
from loguru import logger

async def test_simple_pipeline():
    \"\"\"Test the simple pipeline components\"\"\"
    print("🚀 Testing Simple Enhanced Crypto Trading Pipeline\\n")
    
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
    print("\\n2. Testing Technical Indicators (ta library)...")
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
    print("\\n3. Testing Signal Generation...")
    from feature_engineering.technical_indicators import TechnicalSignalGenerator
    
    signal_generator = TechnicalSignalGenerator()
    signals = signal_generator.generate_signals(df)
    
    print(f"   ✓ Overall Signal: {signals['overall_signal']}")
    print(f"   ✓ Momentum Signals: {len(signals['momentum_signals'])}")
    
    print("\\n✅ Simple pipeline test completed successfully!")
    print("\\n📝 This version uses:")
    print("   - ta library (instead of talib)")
    print("   - TextBlob + VADER (instead of spaCy)")
    print("   - Simple dependencies (no C compilation)")
    print("   - SQLite database (instead of MongoDB)")

if __name__ == "__main__":
    asyncio.run(test_simple_pipeline())
"""
    
    with open("run_simple.py", "w") as f:
        f.write(simple_script)
    
    # Create test_simple.py
    test_script = """#!/usr/bin/env python3
\"\"\"
Test simple pipeline components (Windows Compatible)
\"\"\"

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
    print("🧪 Testing Simple Enhanced Crypto Trading Pipeline Components\\n")
    
    # Test components
    nlp_success = await test_nlp()
    tech_success = test_technical_indicators()
    sentiment_success = await test_sentiment_analyzer()
    
    print("\\n" + "="*50)
    if nlp_success and tech_success and sentiment_success:
        print("🎉 ALL TESTS PASSED! Simple pipeline is working correctly!")
        print("\\n🚀 You can now run:")
        print("   python run_simple.py")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("\\n💡 Try running: python setup_simple.py")

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    with open("test_simple.py", "w") as f:
        f.write(test_script)
    
    logger.info("✓ Created simple test scripts")
    return True


def create_windows_batch_files():
    """Create Windows batch files for easy execution"""
    logger.info("Creating Windows batch files...")
    
    # Create run_simple.bat
    batch_content = """@echo off
echo Starting Simple Enhanced Crypto Trading Pipeline...
python run_simple.py
pause
"""
    
    with open("run_simple.bat", "w") as f:
        f.write(batch_content)
    
    # Create test_simple.bat
    test_batch = """@echo off
echo Testing Simple Pipeline Components...
python test_simple.py
pause
"""
    
    with open("test_simple.bat", "w") as f:
        f.write(test_batch)
    
    logger.info("✓ Created Windows batch files")
    return True


def main():
    """Main setup function for simple Windows installation"""
    print("🚀 Setting up Simple Enhanced Crypto Trading Pipeline for Windows\\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    
    logger.info(f"✓ Python {sys.version.split()[0]} detected")
    
    # Run setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Setting up configuration", setup_configuration),
        ("Installing simple dependencies", setup_environment),
        ("Creating simple scripts", create_simple_scripts),
        ("Creating Windows batch files", create_windows_batch_files)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\\n{step_name}...")
        if not step_func():
            logger.error(f"❌ {step_name} failed")
            sys.exit(1)
        logger.info(f"✅ {step_name} completed")
    
    print("\\n🎉 Simple Windows setup completed successfully!")
    print("\\n📋 Next steps:")
    print("1. Edit config/api_keys_local.yaml with your API keys")
    print("2. Edit .env with your environment variables")
    print("3. Double-click test_simple.bat (to test components)")
    print("4. Double-click run_simple.bat (to run simple pipeline)")
    print("5. Or run: python test_simple.py")
    
    print("\\n📚 What's different in this simple version:")
    print("- Uses ta library instead of talib (no C compilation)")
    print("- Uses TextBlob + VADER instead of spaCy (no C compilation)")
    print("- Uses SQLite instead of MongoDB (no setup needed)")
    print("- Minimal dependencies (faster installation)")
    print("- Same functionality, lighter footprint")
    
    print("\\n⚠️  Important:")
    print("- This is for educational purposes only")
    print("- Trading involves substantial risk")
    print("- Always do your own research")
    print("- No C compilation required!")


if __name__ == "__main__":
    main()
