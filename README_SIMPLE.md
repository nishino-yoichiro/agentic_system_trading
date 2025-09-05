# 🚀 Enhanced Crypto Trading Pipeline - Simple Windows Version

**No C Compilation Required!** This is a lightweight version that avoids all the heavy dependencies that cause Windows installation issues.

## 🎯 **Why This Version?**

The original pipeline had dependency issues on Windows:
- ❌ `talib` requires C++ compilation
- ❌ `spaCy` requires C++ compilation  
- ❌ `blis` requires C++ compilation
- ❌ `transformers` is heavy and slow

**This simple version:**
- ✅ Uses `ta` library (pure Python)
- ✅ Uses `TextBlob` + `VADER` (pure Python)
- ✅ Uses `SQLite` (built into Python)
- ✅ Minimal dependencies (fast installation)
- ✅ Same functionality, lighter footprint

## 🚀 **Quick Start (Windows)**

### **1. Installation**
```cmd
cd enhanced_crypto_pipeline
python setup_simple.py
```

### **2. Test Installation**
```cmd
python test_simple.py
```

### **3. Run Simple Pipeline**
```cmd
python run_simple.py
```

## 📊 **What's Included**

### **Technical Analysis**
- ✅ RSI, MACD, Bollinger Bands
- ✅ Stochastic, Williams %R, CCI
- ✅ ADX, Parabolic SAR, Ichimoku
- ✅ Volume indicators (OBV, MFI)
- ✅ Moving averages (SMA, EMA)
- ✅ Custom indicators and signals

### **NLP Processing**
- ✅ Sentiment analysis (TextBlob + VADER)
- ✅ Entity recognition (keyword-based)
- ✅ Crypto/stock mention detection
- ✅ Text preprocessing and cleaning
- ✅ Keyword extraction

### **Data Management**
- ✅ SQLite database (no setup needed)
- ✅ Parquet file storage
- ✅ CSV export/import
- ✅ Configuration management

### **Reporting**
- ✅ PDF report generation
- ✅ Chart creation (matplotlib)
- ✅ Email alerts (optional)

## 🔧 **Dependencies (Simple)**

```txt
# Core data processing
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0

# Technical Analysis (Windows compatible)
ta>=0.10.0

# Web scraping and APIs
requests>=2.28.0
beautifulsoup4>=4.11.0
playwright>=1.30.0

# Configuration and utilities
pyyaml>=6.0
python-dotenv>=0.19.0
loguru>=0.6.0

# Basic NLP (lightweight alternatives)
textblob>=0.17.0
vaderSentiment>=3.3.2

# Data visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# PDF generation
fpdf2>=2.5.0
```

## 📈 **Example Usage**

### **Basic Pipeline Run**
```python
import asyncio
from feature_engineering.nlp_processor_simple import SimpleNLPProcessor
from feature_engineering.technical_indicators import IndicatorCalculator

async def main():
    # Test NLP
    nlp = SimpleNLPProcessor()
    await nlp.initialize()
    
    text = "Bitcoin is surging to new all-time highs!"
    analysis = await nlp.process_text(text)
    
    print(f"Sentiment: {analysis.sentiment_label} ({analysis.sentiment_score:.2f})")
    print(f"Crypto mentions: {analysis.crypto_mentions}")
    
    # Test Technical Indicators
    calculator = IndicatorCalculator()
    # ... create sample data and calculate indicators

asyncio.run(main())
```

### **Test Scripts**
```cmd
# Test all components
python test_simple.py

# Run simple pipeline
python run_simple.py

# Or use batch files on Windows
test_simple.bat
run_simple.bat
```

## 🔍 **What's Different from Full Version**

| Feature | Full Version | Simple Version |
|---------|-------------|----------------|
| **Technical Analysis** | talib (C++) | ta (Python) |
| **NLP** | spaCy + BERT | TextBlob + VADER |
| **Database** | MongoDB | SQLite |
| **Dependencies** | 50+ packages | 15 packages |
| **Installation** | Complex | Simple |
| **Performance** | Fast | Good |
| **Functionality** | Full | 90% |

## 🎯 **When to Use Each Version**

### **Use Simple Version When:**
- ✅ You're on Windows
- ✅ You want quick setup
- ✅ You don't need advanced NLP
- ✅ You're learning/experimenting
- ✅ You want minimal dependencies

### **Use Full Version When:**
- ✅ You're on Linux/Mac
- ✅ You need advanced NLP (BERT, embeddings)
- ✅ You want maximum performance
- ✅ You're doing production trading
- ✅ You have time for complex setup

## 🚀 **Quick Test**

Run this to verify everything works:

```python
import asyncio
from feature_engineering.nlp_processor_simple import SimpleNLPProcessor

async def quick_test():
    nlp = SimpleNLPProcessor()
    await nlp.initialize()
    
    text = "Bitcoin is surging to new all-time highs!"
    analysis = await nlp.process_text(text)
    
    print(f"✅ Sentiment: {analysis.sentiment_label}")
    print(f"✅ Crypto mentions: {analysis.crypto_mentions}")
    print("✅ Simple pipeline working!")

asyncio.run(quick_test())
```

## 📚 **Documentation**

- **Simple Guide**: This README
- **Full Guide**: `README.md` (for advanced version)
- **Windows Guide**: `README_WINDOWS.md` (for full version on Windows)
- **Test Scripts**: `test_simple.py`, `run_simple.py`

## 🔧 **Troubleshooting**

### **Common Issues**

#### **1. "No module named 'textblob'"**
```cmd
pip install textblob
```

#### **2. "No module named 'vaderSentiment'"**
```cmd
pip install vaderSentiment
```

#### **3. "No module named 'ta'"**
```cmd
pip install ta
```

#### **4. Playwright issues**
```cmd
python -m playwright install
```

### **Performance Tips**

1. **Use virtual environment**:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements_simple.txt
   ```

2. **For large datasets**, increase memory:
   ```cmd
   python -X dev run_simple.py
   ```

## 🎉 **Success!**

If you see this, your simple setup is working:

```
✅ Sentiment: positive
✅ Crypto mentions: ['bitcoin', 'crypto']
✅ Simple pipeline working!
```

**Ready to start trading analysis on Windows!** 🚀

## ⚠️ **Important Notes**

1. **No C Compilation**: This version deliberately avoids all C++ dependencies
2. **Same Functionality**: All core features work identically
3. **Performance**: Slightly slower than full version, but still very fast
4. **Maintenance**: All libraries are actively maintained

## 🎯 **Next Steps**

1. **Test the pipeline**: `python test_simple.py`
2. **Add your API keys**: Edit `config/api_keys_local.yaml`
3. **Run analysis**: `python run_simple.py`
4. **Customize**: Modify the code for your needs

**Happy trading!** 📈
