# 🚀 Enhanced Crypto Trading Pipeline

A sophisticated, modular trading pipeline that combines real-time data collection, advanced NLP processing, technical analysis, Monte Carlo simulations, and AI-powered recommendations for cryptocurrency and stock markets.

## 🎯 **What This Pipeline Does**

This isn't just another trading bot. It's a comprehensive system that:

1. **Collects Data Intelligently**: Real-time news, price feeds, and social sentiment from multiple APIs
2. **Processes with Advanced NLP**: Uses sentence transformers and LLM embeddings for sentiment analysis
3. **Calculates Technical Indicators**: 50+ technical indicators including RSI, MACD, Bollinger Bands, Ichimoku
4. **Runs Sophisticated Simulations**: Monte Carlo with correlation modeling and regime detection
5. **Generates Smart Recommendations**: AI-powered trade signals with confidence scores and risk management
6. **Creates Professional Reports**: PDF reports with charts, analysis, and actionable insights

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Intelligence    │    │  Decision Layer │
│                 │    │     Layer        │    │                 │
│ • News APIs     │───▶│ • NLP Processing │───▶│ • Signal Fusion │
│ • Price APIs    │    │ • Technical      │    │ • Portfolio     │
│ • Social APIs   │    │   Indicators     │    │   Optimization  │
│ • Web Scraping  │    │ • Alpha Factors  │    │ • Risk Mgmt     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Storage       │    │   Simulation     │    │   Reporting     │
│                 │    │     Engine       │    │                 │
│ • Raw Data      │    │ • Monte Carlo    │    │ • PDF Reports   │
│ • Processed     │    │ • Correlation    │    │ • Visualizations│
│ • Features      │    │ • Regime Detection│   │ • Email Alerts  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 **Quick Start**

### 1. **Installation**

#### **For Windows Users (Recommended)**
```cmd
# Navigate to the pipeline directory
cd enhanced_crypto_pipeline

# Run Windows-specific setup (avoids TA-Lib issues)
python setup_windows.py

# Test the installation
python test_ta_library.py
```

#### **For Linux/Mac Users**
```bash
# Clone the repository
git clone <your-repo-url>
cd enhanced_crypto_pipeline

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies
python -m spacy download en_core_web_sm
```

#### **Manual Installation (All Platforms)**
```bash
# Install dependencies
pip install -r requirements.txt

# Install additional dependencies
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

### 2. **Configuration**

```bash
# Copy and edit configuration files
cp config/api_keys.yaml config/api_keys_local.yaml
# Edit config/api_keys_local.yaml with your API keys

# Edit asset universe
nano config/assets.yaml

# Edit trading strategies
nano config/strategies.yaml
```

### 3. **Run the Pipeline**

```bash
# Run complete pipeline
python run_pipeline.py --mode full

# Run only data collection
python run_pipeline.py --mode data --hours-back 48

# Run only analysis on existing data
python run_pipeline.py --mode analysis

# Generate recommendations only
python run_pipeline.py --mode recommendations
```

## 📊 **Key Features**

### **Data Ingestion**
- **News APIs**: NewsAPI, RavenPack, RSS feeds
- **Price Data**: Polygon.io, Alpaca, Binance, CoinGecko
- **Social Sentiment**: Reddit, Twitter analysis
- **Real-time Feeds**: WebSocket connections for live data
- **Data Validation**: Quality assurance and error handling

### **Feature Engineering**
- **Advanced NLP**: Sentence transformers, sentiment analysis, entity extraction
- **Technical Indicators**: 50+ indicators including custom combinations
- **Alpha Factors**: Combined NLP + price action signals
- **Time Series**: Sentiment trends, volatility regimes
- **Feature Store**: Vectorized storage for ML models

### **Simulation Engine**
- **Monte Carlo**: 10,000+ scenario simulations
- **Correlation Modeling**: Dynamic correlation matrices
- **Regime Detection**: Bull/bear/sideways/volatile markets
- **Risk Metrics**: VaR, CVaR, drawdown analysis
- **Portfolio Optimization**: Mean-variance, Black-Litterman

### **Trading Logic**
- **Signal Fusion**: Multi-source signal combination
- **Portfolio Optimization**: Dynamic allocation strategies
- **Risk Management**: Position sizing, stop losses
- **Backtesting**: Historical strategy validation
- **Performance Tracking**: Real-time P&L monitoring

### **Reporting System**
- **Daily Reports**: Comprehensive PDF reports
- **Visualizations**: Charts, heatmaps, dashboards
- **Email Alerts**: Automated distribution
- **Performance Metrics**: Sharpe ratio, drawdown, win rate
- **Risk Analysis**: Portfolio risk breakdown

## 🔧 **Configuration**

### **API Keys Setup**

Create `config/api_keys_local.yaml`:

```yaml
# News APIs
newsapi:
  api_key: "your_newsapi_key_here"
  
# Financial Data
polygon:
  api_key: "your_polygon_key_here"
alpaca:
  api_key: "your_alpaca_key_here"
  secret_key: "your_alpaca_secret_here"
  
# Crypto APIs
binance:
  api_key: "your_binance_key_here"
  secret_key: "your_binance_secret_here"
coingecko:
  api_key: "your_coingecko_key_here"  # Optional
```

### **Asset Universe**

Edit `config/assets.yaml` to define your trading universe:

```yaml
crypto_assets:
  major_coins:
    - symbol: "BTC"
      name: "Bitcoin"
      weight: 0.4
    - symbol: "ETH"
      name: "Ethereum"
      weight: 0.3

stock_assets:
  tech_growth:
    - symbol: "AAPL"
      name: "Apple Inc."
      weight: 0.15
```

### **Trading Strategies**

Configure strategies in `config/strategies.yaml`:

```yaml
strategies:
  momentum:
    name: "Momentum Strategy"
    enabled: true
    weight: 0.3
    parameters:
      lookback_period: 20
      momentum_threshold: 0.05
      
  sentiment:
    name: "Sentiment Strategy"
    enabled: true
    weight: 0.2
    parameters:
      sentiment_threshold: 0.3
      news_weight: 0.6
```

## 📈 **Usage Examples**

### **Basic Usage**

```python
from enhanced_crypto_pipeline import EnhancedCryptoPipeline

# Initialize pipeline
pipeline = EnhancedCryptoPipeline("config/pipeline_config.yaml")
await pipeline.initialize()

# Run complete pipeline
results = await pipeline.run_full_pipeline(hours_back=24)

# Check results
print(f"Generated {results['recommendations_count']} recommendations")
print(f"Report saved to: {results['report_path']}")
```

### **Custom Data Collection**

```python
from data_ingestion.news_apis import collect_crypto_news
from data_ingestion.price_apis import collect_price_data

# Collect crypto news
news = await collect_crypto_news("your_newsapi_key", hours_back=48)

# Collect price data
symbols = ['BTC', 'ETH', 'AAPL', 'TSLA']
prices = await collect_price_data(symbols, start_date, end_date, polygon_key="your_key")
```

### **Feature Engineering**

```python
from feature_engineering.nlp_processor import NLPProcessor
from feature_engineering.technical_indicators import IndicatorCalculator

# Initialize NLP processor
nlp = NLPProcessor()
await nlp.initialize()

# Process news articles
results = await nlp.process_articles(articles)
sentiment_metrics = nlp.calculate_sentiment_metrics(results)

# Calculate technical indicators
calculator = IndicatorCalculator()
indicators = calculator.calculate_all_indicators(price_df)
```

### **Portfolio Simulation**

```python
from simulation.portfolio_simulator import PortfolioSimulator

simulator = PortfolioSimulator()
await simulator.initialize()

# Run simulation
results = await simulator.run_portfolio_simulation(
    assets_data,
    time_horizon=30,
    num_simulations=10000,
    risk_tolerance='medium'
)
```

## 📊 **Understanding the Output**

### **Daily Report Structure**

1. **Executive Summary**
   - Market overview and key metrics
   - Top opportunities and risk alerts
   - Portfolio performance summary

2. **Market Analysis**
   - Sentiment analysis and news highlights
   - Technical indicator summary
   - Volatility and trend analysis

3. **Trading Recommendations**
   - Buy/sell/hold signals with confidence scores
   - Price targets and stop losses
   - Position sizing recommendations

4. **Risk Analysis**
   - Portfolio VaR and stress testing
   - Correlation analysis
   - Drawdown scenarios

5. **Simulation Results**
   - Monte Carlo projections
   - Probability distributions
   - Scenario analysis

### **Signal Interpretation**

- **Confidence Score**: 0-1 scale, higher = more confident
- **Signal Strength**: weak/medium/strong
- **Time Horizon**: short/medium/long term
- **Risk Level**: low/medium/high

### **Portfolio Recommendations**

- **Asset Allocation**: Suggested weights for each asset
- **Rebalancing**: When and how to rebalance
- **Risk Adjustment**: Dynamic risk management
- **Performance Targets**: Expected returns and risk metrics

## 🔍 **Advanced Features**

### **Custom Indicators**

```python
# Add custom technical indicators
def custom_momentum_indicator(prices, period=14):
    returns = prices.pct_change()
    momentum = returns.rolling(period).sum()
    return momentum

# Register with calculator
calculator.add_custom_indicator('custom_momentum', custom_momentum_indicator)
```

### **Strategy Backtesting**

```python
# Run backtesting
backtest_results = await pipeline.run_backtest(
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=100000,
    strategy='momentum'
)

# Analyze results
print(f"Total Return: {backtest_results['total_return']:.2%}")
print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
```

### **Real-time Monitoring**

```python
# Set up real-time monitoring
async def monitor_pipeline():
    while True:
        # Check for new data
        new_data = await pipeline.check_for_updates()
        
        # Process if new data available
        if new_data:
            await pipeline.process_incremental_data()
        
        # Wait before next check
        await asyncio.sleep(300)  # 5 minutes

# Start monitoring
asyncio.create_task(monitor_pipeline())
```

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **API Rate Limits**
   ```bash
   # Check rate limit status in logs
   tail -f logs/pipeline.log | grep "rate limit"
   
   # Adjust rate limits in config
   nano config/api_keys_local.yaml
   ```

2. **Insufficient Data**
   ```python
   # Check data availability
   print(f"Price data points: {len(price_df)}")
   print(f"News articles: {len(news_data)}")
   
   # Increase collection period
   results = await pipeline.run_full_pipeline(hours_back=72)
   ```

3. **Memory Issues**
   ```bash
   # Monitor memory usage
   python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
   
   # Reduce simulation runs
   # Edit config/strategies.yaml: simulation_runs: 5000
   ```

4. **Model Loading Errors**
   ```bash
   # Install required models
   python -m spacy download en_core_web_sm
   pip install --upgrade sentence-transformers
   ```

### **Performance Optimization**

1. **Database Indexing**
   ```python
   # Add indexes for better performance
   db.prices.create_index([("symbol", 1), ("timestamp", -1)])
   db.news.create_index([("published_at", -1)])
   ```

2. **Caching**
   ```python
   # Enable Redis caching
   from data_ingestion.cache import RedisCache
   cache = RedisCache()
   await cache.set("price_data", data, ttl=3600)
   ```

3. **Parallel Processing**
   ```python
   # Process multiple symbols in parallel
   import asyncio
   
   tasks = [process_symbol(symbol) for symbol in symbols]
   results = await asyncio.gather(*tasks)
   ```

## 📚 **API Reference**

### **Main Pipeline Class**

```python
class EnhancedCryptoPipeline:
    async def initialize(self) -> None
    async def collect_data(self, hours_back: int) -> Dict[str, Any]
    async def engineer_features(self) -> Dict[str, Any]
    async def run_simulations(self) -> Dict[str, Any]
    async def generate_recommendations(self) -> List[Dict[str, Any]]
    async def generate_report(self) -> str
    async def run_full_pipeline(self, hours_back: int) -> Dict[str, Any]
```

### **Data Collection**

```python
# News APIs
async def collect_crypto_news(api_key: str, hours_back: int) -> List[NewsArticle]
async def collect_stock_news(api_key: str, hours_back: int) -> List[NewsArticle]

# Price APIs
async def collect_price_data(symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, List[PriceData]]
```

### **Feature Engineering**

```python
# NLP Processing
class NLPProcessor:
    async def process_text(self, text: str) -> SentimentResult
    async def process_articles(self, articles: List[Dict]) -> List[SentimentResult]
    def calculate_sentiment_metrics(self, results: List[SentimentResult]) -> Dict[str, float]

# Technical Indicators
class IndicatorCalculator:
    def calculate_all_indicators(self, df: pd.DataFrame) -> TechnicalIndicators
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ **Disclaimer**

This software is for educational and research purposes only. Trading cryptocurrencies and stocks involves substantial risk of loss. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 🙏 **Acknowledgments**

- **Data Providers**: NewsAPI, Polygon.io, Binance, CoinGecko
- **Libraries**: pandas, numpy, scikit-learn, transformers, talib
- **Community**: Open source contributors and financial data enthusiasts

---

**Ready to build your enhanced trading pipeline?** 🚀

Start with `python run_pipeline.py --mode full` and watch the magic happen!
