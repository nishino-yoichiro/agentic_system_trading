# 🧪 Test Utilities

This folder contains testing and data inspection utilities for the Enhanced Crypto Trading Pipeline.

## 📊 Data Inspector

The `data_inspector.py` tool allows you to analyze, visualize, and validate price data to ensure accuracy.

### Usage

#### Command Line
```bash
# Display basic data info
python tests/data_inspector.py --symbol BTC --action display

# Create price charts
python tests/data_inspector.py --symbol BTC --action chart

# Show data for specific date range
python tests/data_inspector.py --symbol BTC --action range --start 2024-01-01 --end 2024-01-31

# Validate data quality
python tests/data_inspector.py --symbol BTC --action validate

# Get online comparison links
python tests/data_inspector.py --symbol BTC --action compare
```

#### Windows Batch File
```cmd
# Display data info
test_data.bat BTC display

# Create charts
test_data.bat BTC chart

# Validate data
test_data.bat BTC validate
```

### Features

- **Data Display**: Show basic statistics, date ranges, and sample data
- **Charting**: Create comprehensive price charts with multiple views
- **Date Range Analysis**: Filter and analyze specific time periods
- **Data Validation**: Check for missing values, duplicates, and anomalies
- **Online Comparison**: Get links to verify data against online sources

### Available Symbols

Check the `data/raw/` folder for available price data files:
- Crypto: BTC, ETH, BNB, ADA, SOL, DOT, AVAX, MATIC, LINK, UNI
- Stocks: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, BAC, XOM

### Chart Output

Charts are saved as PNG files in the `tests/` folder with timestamps for easy comparison.

