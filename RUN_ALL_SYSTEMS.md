# 🚀 Running All Three Systems

## **Terminal Setup (3 Terminals)**

### **Terminal 1: Continuous Price Collection**
```powershell
# Activate virtual environment
.\enh_venv\Scripts\activate

# Start continuous BTC price collection (every 20 seconds)
python start_btc_collection.py
```
**What you'll see:**
- BTC price updates every 20 seconds
- Data being saved to `data/crypto_db/BTC_historical.parquet`
- Collection statistics

### **Terminal 2: Live Trading Log**
```powershell
# Activate virtual environment
.\enh_venv\Scripts\activate

# Start live trading log (every 1 minute)
python run_live_trading.py
```
**What you'll see:**
- Signal generation every minute
- BUY/SELL/HOLD decisions
- Simulated PnL updates
- Portfolio value changes

### **Terminal 3: Trading Dashboard**
```powershell
# Activate virtual environment
.\enh_venv\Scripts\activate

# Start trading dashboard
streamlit run trading_log_dashboard.py --server.port 8504
```
**What you'll see:**
- Web dashboard at `http://localhost:8504`
- Real-time charts of trades
- Portfolio performance
- Recent trade history

## **🔄 Data Flow:**

1. **Price Collection** → Updates BTC price data every 20 seconds
2. **Trading Log** → Reads fresh price data → Generates signals every 1 minute
3. **Dashboard** → Shows live results from trading log

## **📊 Expected Output:**

**Terminal 1 (Price Collection):**
```
2025-10-02 17:30:00 | INFO | BTC price: $43,250.50
2025-10-02 17:30:20 | INFO | BTC price: $43,255.75
2025-10-02 17:30:40 | INFO | BTC price: $43,248.25
```

**Terminal 2 (Trading Log):**
```
2025-10-02 17:30:00 | INFO | ✅ Executed 1 trades
2025-10-02 17:30:00 | INFO |    BUY @ $43,250.50 | PnL: $0.00
2025-10-02 17:30:00 | INFO | 💰 Portfolio: $100,000.00 | PnL: $0.00
```

**Terminal 3 (Dashboard):**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8504
```

## **🛑 Stopping Systems:**

Press `Ctrl+C` in each terminal to stop the respective system.

## **📁 Files Created:**

- `data/crypto_db/BTC_historical.parquet` - Updated price data
- `data/live_trades.csv` - Trading log
- `data/portfolio_state.json` - Portfolio state
- `live_trading.log` - System logs

## **🎯 Quick Start:**

1. Open 3 PowerShell terminals
2. Run each command above in separate terminals
3. Watch the data flow in real-time!
4. View dashboard at `http://localhost:8504`
