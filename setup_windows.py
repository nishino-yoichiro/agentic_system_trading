#!/usr/bin/env python3
"""
Enhanced Crypto Trading Pipeline - Windows Setup Script

This script sets up the enhanced crypto trading pipeline on Windows with
all dependencies and initial configuration, avoiding TA-Lib issues.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml
from loguru import logger


def setup_environment():
    """Set up the Python environment and install dependencies"""
    logger.info("Setting up Python environment for Windows...")
    
    try:
        # Install requirements (excluding talib)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("✓ Dependencies installed successfully")
        
        # Install additional dependencies
        additional_deps = [
            "python -m spacy download en_core_web_sm",
            "python -c 'import nltk; nltk.download(\"vader_lexicon\"); nltk.download(\"punkt\")'"
        ]
        
        for dep in additional_deps:
            subprocess.check_call(dep, shell=True)
        
        logger.info("✓ Additional dependencies installed")
        
        # Test ta library installation
        try:
            import ta
            logger.info("✓ ta library working correctly")
        except ImportError as e:
            logger.error(f"ta library not working: {e}")
            return False
        
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
            logger.warning("API keys template not found")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Enhanced Crypto Trading Pipeline Environment Variables
# Copy this file and add your actual API keys

# News APIs
NEWSAPI_KEY=your_newsapi_key_here

# Financial Data APIs
POLYGON_API_KEY=your_polygon_key_here
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
FINNHUB_API_KEY=your_finnhub_key_here

# Crypto APIs
BINANCE_API_KEY=your_binance_key_here
BINANCE_SECRET_KEY=your_binance_secret_here
COINGECKO_API_KEY=your_coingecko_key_here

# Social Media APIs
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Database
MONGODB_URI=mongodb://localhost:27017/crypto_trading
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

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


def create_sample_scripts():
    """Create sample scripts for common tasks"""
    logger.info("Creating sample scripts...")
    
    # Create run_example.py
    example_script = """#!/usr/bin/env python3
\"\"\"
Example script to run the Enhanced Crypto Trading Pipeline (Windows)
\"\"\"

import asyncio
from run_pipeline import EnhancedCryptoPipeline

async def main():
    # Initialize pipeline
    pipeline = EnhancedCryptoPipeline()
    
    # Run complete pipeline
    results = await pipeline.run_full_pipeline(hours_back=24)
    
    if results['success']:
        print("✅ Pipeline completed successfully!")
        print(f"📊 Generated {results['recommendations_count']} recommendations")
        print(f"📄 Report saved to: {results['report_path']}")
    else:
        print("❌ Pipeline failed:")
        print(f"Error: {results['error']}")

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    with open("run_example.py", "w") as f:
        f.write(example_script)
    
    logger.info("✓ Created run_example.py")
    
    # Create test_components.py
    test_script = """#!/usr/bin/env python3
\"\"\"
Test individual pipeline components (Windows Compatible)
\"\"\"

import asyncio
from data_ingestion.news_apis import collect_crypto_news
from data_ingestion.price_apis import collect_price_data
from feature_engineering.nlp_processor import NLPProcessor
from feature_engineering.technical_indicators import IndicatorCalculator
import pandas as pd
import numpy as np

async def test_news_collection():
    print("Testing news collection...")
    try:
        # This will fail without API key, but shows the structure
        news = await collect_crypto_news("test_key", hours_back=1)
        print(f"✓ News collection: {len(news)} articles")
    except Exception as e:
        print(f"⚠ News collection: {e}")

async def test_nlp_processing():
    print("Testing NLP processing...")
    try:
        nlp = NLPProcessor()
        await nlp.initialize()
        
        sample_text = "Bitcoin is surging to new all-time highs!"
        result = await nlp.process_text(sample_text)
        print(f"✓ NLP processing: {result.sentiment_label} ({result.sentiment_score:.2f})")
    except Exception as e:
        print(f"⚠ NLP processing: {e}")

def test_technical_indicators():
    print("Testing technical indicators (ta library)...")
    try:
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
        
        calculator = IndicatorCalculator()
        indicators = calculator.calculate_all_indicators(df)
        print(f"✓ Technical indicators: RSI={indicators.rsi:.1f}, MACD={indicators.macd:.4f}")
        print(f"✓ Using ta library (Windows compatible)")
    except Exception as e:
        print(f"⚠ Technical indicators: {e}")

def test_ta_library():
    print("Testing ta library specifically...")
    try:
        import ta
        print("✓ ta library imported successfully")
        
        # Test basic functionality
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        rsi = ta.momentum.rsi(df['close'])
        print(f"✓ RSI calculation: {rsi.iloc[-1]:.2f}")
        
    except Exception as e:
        print(f"⚠ ta library test: {e}")

async def main():
    print("🧪 Testing Enhanced Crypto Trading Pipeline Components (Windows)\\n")
    
    test_ta_library()
    await test_news_collection()
    await test_nlp_processing()
    test_technical_indicators()
    
    print("\\n✅ Component testing completed!")
    print("\\n📝 Note: This version uses the 'ta' library instead of 'talib' for Windows compatibility")

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    with open("test_components.py", "w") as f:
        f.write(test_script)
    
    logger.info("✓ Created test_components.py")
    
    return True


def create_windows_batch_files():
    """Create Windows batch files for easy execution"""
    logger.info("Creating Windows batch files...")
    
    # Create run_pipeline.bat
    batch_content = """@echo off
echo Starting Enhanced Crypto Trading Pipeline...
python run_pipeline.py --mode full
pause
"""
    
    with open("run_pipeline.bat", "w") as f:
        f.write(batch_content)
    
    # Create test_components.bat
    test_batch = """@echo off
echo Testing Pipeline Components...
python test_components.py
pause
"""
    
    with open("test_components.bat", "w") as f:
        f.write(test_batch)
    
    logger.info("✓ Created Windows batch files")
    return True


def main():
    """Main setup function for Windows"""
    print("🚀 Setting up Enhanced Crypto Trading Pipeline for Windows\\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    
    logger.info(f"✓ Python {sys.version.split()[0]} detected")
    
    # Run setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Setting up configuration", setup_configuration),
        ("Installing dependencies", setup_environment),
        ("Creating sample scripts", create_sample_scripts),
        ("Creating Windows batch files", create_windows_batch_files)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\\n{step_name}...")
        if not step_func():
            logger.error(f"❌ {step_name} failed")
            sys.exit(1)
        logger.info(f"✅ {step_name} completed")
    
    print("\\n🎉 Windows setup completed successfully!")
    print("\\n📋 Next steps:")
    print("1. Edit config/api_keys_local.yaml with your API keys")
    print("2. Edit .env with your environment variables")
    print("3. Double-click test_components.bat (to test components)")
    print("4. Double-click run_pipeline.bat (to run full pipeline)")
    print("5. Or run: python run_pipeline.py --mode full")
    
    print("\\n📚 Documentation:")
    print("- README.md: Complete usage guide")
    print("- config/: Configuration files")
    print("- This version uses 'ta' library (Windows compatible)")
    
    print("\\n⚠️  Important:")
    print("- This is for educational purposes only")
    print("- Trading involves substantial risk")
    print("- Always do your own research")
    print("- Uses 'ta' library instead of 'talib' for Windows compatibility")


if __name__ == "__main__":
    main()
