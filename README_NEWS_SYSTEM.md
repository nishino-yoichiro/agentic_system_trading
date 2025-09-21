# 📰 Crypto News Collection System

A comprehensive news collection system for crypto trading pipeline that gathers news from multiple sources, analyzes sentiment, and stores data efficiently.

## 🚀 Features

- **Multi-Source Collection**: NewsAPI.org, CryptoCompare, CoinDesk RSS, Reddit
- **Rate Limiting**: Respects API limits and prevents overuse
- **Sentiment Analysis**: Basic word-count based sentiment scoring
- **Data Storage**: SQLite database + Parquet files for analysis
- **Deduplication**: Removes duplicate articles across sources
- **Historical Backfill**: Collect historical news data
- **Real-time Collection**: Get latest news for analysis

## 📊 Data Sources

### 1. NewsAPI.org
- **Free Tier**: 1,000 requests per day
- **Coverage**: General financial news
- **Setup**: Get API key from [newsapi.org](https://newsapi.org/register)

### 2. CryptoCompare
- **Free Tier**: 100,000 calls per month
- **Coverage**: Crypto-specific news
- **Setup**: Get API key from [cryptocompare.com](https://min-api.cryptocompare.com/)

### 3. CoinDesk RSS
- **Free**: No API key required
- **Coverage**: High-quality crypto news
- **Setup**: No setup required

### 4. Reddit API
- **Free Tier**: 100 requests per minute
- **Coverage**: Community sentiment
- **Setup**: Get credentials from [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)

## 🛠️ Setup

### 1. Install Dependencies
```bash
pip install feedparser aiohttp
```

### 2. Configure API Keys
Edit `config/news_api_keys.yaml`:
```yaml
newsapi:
  api_key: "your_newsapi_key_here"

cryptocompare:
  api_key: "your_cryptocompare_key_here"
```

### 3. Test Collection
```bash
python test_news_collection.py
```

## 📈 Usage

### News Manager Script
```bash
# Collect recent news
python news_manager.py --mode collect --tickers BTC ETH --days 1

# Backfill historical data
python news_manager.py --mode backfill --tickers BTC ETH --days 30

# Validate data quality
python news_manager.py --mode validate --tickers BTC ETH --days 7

# Show summary
python news_manager.py --mode summary --tickers BTC ETH --days 7
```

### Programmatic Usage
```python
from data_ingestion.news_collector import NewsCollector

# Initialize collector
collector = NewsCollector(api_keys={'newsapi': 'your_key'})

# Collect news
result = await collector.collect_news(['BTC', 'ETH'], days_back=1)

# Get recent news
recent = collector.get_recent_news('BTC', hours_back=24, limit=10)
```

## 📊 Data Structure

### NewsArticle
```python
@dataclass
class NewsArticle:
    timestamp: datetime
    ticker: str
    source: str
    headline: str
    url: str
    content: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    article_id: Optional[str] = None
```

### Database Schema
```sql
CREATE TABLE news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id TEXT UNIQUE,
    timestamp DATETIME,
    ticker TEXT,
    source TEXT,
    headline TEXT,
    url TEXT,
    content TEXT,
    sentiment_score REAL,
    sentiment_label TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🎯 Sentiment Analysis

The system uses a basic word-count approach:

### Positive Words
- bullish, surge, rally, moon, pump, breakthrough, adoption, institutional
- partnership, upgrade, launch, success, growth, profit, gain, rise
- positive, optimistic, strong, robust, solid, excellent, outstanding

### Negative Words
- bearish, crash, dump, plunge, decline, fall, drop, sell-off
- regulation, ban, hack, scam, fraud, loss, negative, pessimistic
- weak, concern, risk, volatile, uncertain, fear, panic, doubt

### Scoring
- **Positive**: Score > 0.1
- **Negative**: Score < -0.1
- **Neutral**: Score between -0.1 and 0.1

## 📁 File Structure

```
data_ingestion/
├── news_collector.py          # Main news collection class
├── news_manager.py            # CLI management script
└── test_news_collection.py    # Test script

config/
└── news_api_keys.yaml         # API keys configuration

data/
├── news.db                    # SQLite database
├── news_BTC_20250919.parquet # Daily Parquet files
└── news_ETH_20250919.parquet
```

## 🔧 Configuration

### Rate Limits
```python
rate_limits = {
    'newsapi': {'max_calls': 1000, 'window': 86400},      # 1000/day
    'cryptocompare': {'max_calls': 100000, 'window': 2592000},  # 100k/month
    'reddit': {'max_calls': 100, 'window': 60},           # 100/min
}
```

### Collection Settings
```yaml
collection:
  default_days_back: 1
  max_articles_per_ticker: 100
  sentiment_threshold: 0.1
  sources:
    - "newsapi"
    - "cryptocompare"
    - "coindesk"
```

## 📊 Performance

### Collection Speed
- **CoinDesk RSS**: ~2-3 seconds per ticker
- **CryptoCompare**: ~1-2 seconds per ticker
- **NewsAPI**: ~2-3 seconds per ticker (with API key)

### Storage Efficiency
- **SQLite**: Fast queries, good for real-time access
- **Parquet**: Efficient storage, good for analysis
- **Deduplication**: Prevents duplicate storage

## 🚨 Error Handling

- **API Rate Limits**: Automatic rate limiting and retry logic
- **Network Errors**: Graceful handling of connection issues
- **Data Validation**: Ensures data quality before storage
- **Duplicate Prevention**: Prevents duplicate articles

## 🔮 Future Enhancements

1. **Advanced Sentiment Analysis**: Use NLP models (BERT, RoBERTa)
2. **More Sources**: Add Twitter, Telegram, Discord
3. **Real-time Streaming**: WebSocket connections for live updates
4. **Machine Learning**: Train custom sentiment models
5. **News Clustering**: Group related articles
6. **Impact Scoring**: Measure news impact on prices

## 📝 Example Output

```
============================================================
NEWS COLLECTION RESULTS
============================================================
✅ Total articles collected: 30
💾 Articles saved to database: 22
📰 Sources used: CryptoCompare, CoinDesk

📊 Sentiment Analysis:
  Average score: 0.010
  📈 Positive articles: 5
  📉 Negative articles: 3
  ➡️ Neutral articles: 22

📈 Ticker Statistics:
  BTC: 11 articles, 11 saved
  ETH: 19 articles, 11 saved
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your news source
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details
