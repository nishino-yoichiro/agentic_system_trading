# 🚀 Enhanced Crypto Trading Pipeline - Windows Setup

This guide is specifically for Windows users who encounter issues with TA-Lib installation.

## 🎯 **Quick Windows Setup**

### **1. Prerequisites**
- Python 3.8 or higher
- Git (optional, for cloning)
- Windows 10/11

### **2. Installation**

```cmd
# Navigate to the pipeline directory
cd enhanced_crypto_pipeline

# Run Windows-specific setup
python setup_windows.py
```

### **3. Alternative Manual Setup**

If the automated setup fails:

```cmd
# Install dependencies (excluding talib)
pip install -r requirements.txt

# Install additional dependencies
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

# Test the ta library
python -c "import ta; print('ta library working!')"
```

## 🔧 **Windows-Specific Features**

### **Batch Files for Easy Execution**
- `run_pipeline.bat` - Double-click to run the full pipeline
- `test_components.bat` - Double-click to test components

### **Technical Indicators**
This Windows version uses the `ta` library instead of `talib`:
- ✅ **ta library**: Pure Python, Windows compatible
- ❌ **talib**: Requires C++ compilation on Windows

### **Supported Indicators**
The `ta` library provides all the same indicators:
- RSI, MACD, Bollinger Bands
- Stochastic, Williams %R, CCI
- ADX, Parabolic SAR, Ichimoku
- Volume indicators (OBV, MFI, A/D Line)
- And many more!

## 🚀 **Quick Start (Windows)**

### **1. Test Installation**
```cmd
# Double-click this file or run:
test_components.bat

# Or run directly:
python test_components.py
```

### **2. Configure API Keys**
Edit `config/api_keys_local.yaml`:
```yaml
newsapi:
  api_key: "your_newsapi_key_here"
polygon:
  api_key: "your_polygon_key_here"
# ... add your other API keys
```

### **3. Run the Pipeline**
```cmd
# Double-click this file or run:
run_pipeline.bat

# Or run directly:
python run_pipeline.py --mode full
```

## 📊 **What's Different on Windows**

### **Technical Analysis Library**
- **Instead of**: `talib` (requires C++ compilation)
- **We use**: `ta` (pure Python, Windows compatible)
- **Same functionality**: All indicators work identically

### **Installation Process**
- **Automated**: `python setup_windows.py`
- **Manual**: Standard pip install (no special steps needed)
- **No C++**: No need for Visual Studio Build Tools

### **Performance**
- **Slightly slower**: ta library is pure Python vs C++ talib
- **Still fast enough**: For most trading applications
- **Same accuracy**: Identical mathematical calculations

## 🔍 **Troubleshooting Windows Issues**

### **Common Problems**

#### **1. "No module named 'talib'"**
```cmd
# This is expected! We use 'ta' instead
python -c "import ta; print('Using ta library')"
```

#### **2. "Microsoft Visual C++ 14.0 is required"**
```cmd
# This won't happen with our setup
# We avoid talib which requires C++ compilation
```

#### **3. "Permission denied" errors**
```cmd
# Run Command Prompt as Administrator
# Or use --user flag:
pip install --user -r requirements.txt
```

#### **4. "ta library not working"**
```cmd
# Reinstall ta library
pip uninstall ta
pip install ta

# Test it
python -c "import ta; print('ta library working!')"
```

### **Performance Optimization**

#### **1. Use Virtual Environment**
```cmd
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### **2. Memory Management**
```cmd
# For large datasets, increase memory limit
python -X dev run_pipeline.py --mode full
```

## 📈 **Example Usage (Windows)**

### **Basic Pipeline Run**
```python
import asyncio
from run_pipeline import EnhancedCryptoPipeline

async def main():
    pipeline = EnhancedCryptoPipeline()
    results = await pipeline.run_full_pipeline(hours_back=24)
    
    if results['success']:
        print(f"✅ Generated {results['recommendations_count']} recommendations")
        print(f"📄 Report: {results['report_path']}")
    else:
        print(f"❌ Error: {results['error']}")

# Run it
asyncio.run(main())
```

### **Test Technical Indicators**
```python
import pandas as pd
import numpy as np
from feature_engineering.technical_indicators import IndicatorCalculator

# Create sample data
df = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [101, 102, 103, 104, 105],
    'low': [99, 100, 101, 102, 103],
    'close': [100, 101, 102, 103, 104],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# Calculate indicators
calculator = IndicatorCalculator()
indicators = calculator.calculate_all_indicators(df)

print(f"RSI: {indicators.rsi:.2f}")
print(f"MACD: {indicators.macd:.4f}")
print(f"Bollinger Position: {indicators.bollinger_position:.2f}")
```

## 🎯 **Windows-Specific Tips**

### **1. Use Windows Terminal**
- Install Windows Terminal for better experience
- Supports multiple tabs and better colors

### **2. PowerShell vs Command Prompt**
- Both work fine
- PowerShell has better error messages

### **3. File Paths**
- Use forward slashes in Python code: `"data/raw"`
- Windows handles both `\` and `/` automatically

### **4. Long Paths**
- Enable long path support in Windows 10/11
- Or keep project in short path like `C:\crypto\`

## 🔧 **Advanced Windows Setup**

### **1. Using Conda (Alternative)**
```cmd
# Create conda environment
conda create -n crypto_trading python=3.9
conda activate crypto_trading

# Install dependencies
pip install -r requirements.txt
```

### **2. Using WSL (Windows Subsystem for Linux)**
```bash
# In WSL terminal
sudo apt update
sudo apt install python3-pip
pip install -r requirements.txt
# talib might work in WSL, but ta is still recommended
```

### **3. Docker (Advanced)**
```dockerfile
# Create Dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
# ta library works perfectly in Docker
```

## 📚 **Additional Resources**

### **ta Library Documentation**
- [ta-lib GitHub](https://github.com/bukosabino/ta)
- [ta-lib Documentation](https://technical-analysis-library-in-python.readthedocs.io/)

### **Windows Python Development**
- [Python on Windows](https://docs.python.org/3/using/windows.html)
- [Windows Terminal](https://github.com/microsoft/terminal)

## ⚠️ **Important Notes**

1. **No TA-Lib**: This version deliberately avoids talib for Windows compatibility
2. **Same Functionality**: All indicators work identically
3. **Performance**: Slightly slower than talib, but still very fast
4. **Maintenance**: ta library is actively maintained and updated

## 🎉 **Success!**

If you see this message, your Windows setup is complete:

```
✅ Component testing completed!
✓ ta library working correctly
✓ Technical indicators: RSI=45.2, MACD=0.0123
✓ Using ta library (Windows compatible)
```

**Ready to start trading analysis on Windows!** 🚀

Run `python run_pipeline.py --mode full` to begin!
