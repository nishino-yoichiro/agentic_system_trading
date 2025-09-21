#!/usr/bin/env python3
"""
Advanced News Collection System for Crypto Trading Pipeline

Collects news from multiple sources:
- NewsAPI.org (general financial news)
- CryptoCompare (crypto-specific news)
- CoinDesk RSS (crypto news)
- Reddit API (community sentiment)

Features:
- Rate limiting and API key management
- Data deduplication and cleaning
- Sentiment analysis (basic word count)
- Historical backfill capability
- SQLite/Parquet storage
"""

import asyncio
import aiohttp
import logging
import json
import sqlite3
import pandas as pd
import feedparser
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import hashlib
import time

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Structured news article data"""
    timestamp: datetime
    ticker: str
    source: str
    headline: str
    url: str
    content: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    article_id: Optional[str] = None

class NewsCollector:
    """Advanced news collection system"""
    
    def __init__(self, data_dir: Path = Path("data"), api_keys: Dict[str, str] = None):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        
        # Load API keys from environment variables
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        self.api_keys = api_keys or {
            'newsapi': os.getenv('NEWSAPI_KEY', ''),
            'cryptocompare': os.getenv('CRYPTOCOMPARE_API_KEY', '')
        }
        
        # Database setup
        self.db_path = self.data_dir / "news.db"
        self._init_database()
        
        # Rate limiting
        self.rate_limits = {
            'newsapi': {'calls': 0, 'reset_time': time.time(), 'max_calls': 1000, 'window': 86400},  # 1000/day
            'cryptocompare': {'calls': 0, 'reset_time': time.time(), 'max_calls': 100000, 'window': 2592000},  # 100k/month
            'reddit': {'calls': 0, 'reset_time': time.time(), 'max_calls': 100, 'window': 60},  # 100/min
        }
        
        # Sentiment words (basic implementation)
        self.positive_words = {
            'bullish', 'surge', 'rally', 'moon', 'pump', 'breakthrough', 'adoption', 'institutional',
            'partnership', 'upgrade', 'launch', 'success', 'growth', 'profit', 'gain', 'rise',
            'positive', 'optimistic', 'strong', 'robust', 'solid', 'excellent', 'outstanding'
        }
        
        self.negative_words = {
            'bearish', 'crash', 'dump', 'plunge', 'decline', 'fall', 'drop', 'sell-off',
            'regulation', 'ban', 'hack', 'scam', 'fraud', 'loss', 'negative', 'pessimistic',
            'weak', 'concern', 'risk', 'volatile', 'uncertain', 'fear', 'panic', 'doubt'
        }
        
        # Crypto tickers to track
        self.crypto_tickers = ['BTC', 'ETH', 'UNI', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK']
        
    def _init_database(self):
        """Initialize SQLite database for news storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
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
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON news_articles(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker ON news_articles(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON news_articles(source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment ON news_articles(sentiment_score)')
        
        conn.commit()
        conn.close()
    
    async def _check_rate_limit(self, api_name: str) -> bool:
        """Check if we can make an API call within rate limits"""
        now = time.time()
        limits = self.rate_limits[api_name]
        
        # Reset counter if window has passed
        if now - limits['reset_time'] > limits['window']:
            limits['calls'] = 0
            limits['reset_time'] = now
        
        if limits['calls'] >= limits['max_calls']:
            wait_time = limits['window'] - (now - limits['reset_time'])
            logger.warning(f"Rate limit reached for {api_name}. Waiting {wait_time:.0f} seconds.")
            return False
        
        limits['calls'] += 1
        return True
    
    def _calculate_sentiment(self, text: str) -> Tuple[float, str]:
        """Basic sentiment analysis using word count"""
        if not text:
            return 0.0, 'neutral'
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return 0.0, 'neutral'
        
        # Calculate sentiment score (-1 to 1)
        sentiment_score = (positive_count - negative_count) / total_words
        
        if sentiment_score > 0.1:
            label = 'positive'
        elif sentiment_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return sentiment_score, label
    
    def _generate_article_id(self, headline: str, source: str, timestamp: datetime) -> str:
        """Generate unique article ID"""
        content = f"{headline}_{source}_{timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _fetch_newsapi_news(self, ticker: str, days_back: int = 1) -> List[NewsArticle]:
        """Fetch news from NewsAPI.org"""
        if not self.api_keys.get('newsapi'):
            logger.warning("NewsAPI key not found, skipping NewsAPI collection")
            return []
        
        if not await self._check_rate_limit('newsapi'):
            return []
        
        articles = []
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        # Search terms for crypto news
        search_terms = {
            'BTC': 'bitcoin OR btc',
            'ETH': 'ethereum OR eth',
            'UNI': 'uniswap OR uni',
            'ADA': 'cardano OR ada',
            'SOL': 'solana OR sol',
            'DOT': 'polkadot OR dot',
            'AVAX': 'avalanche OR avax',
            'MATIC': 'polygon OR matic',
            'LINK': 'chainlink OR link'
        }
        
        query = search_terms.get(ticker, ticker.lower())
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f"{query} AND (crypto OR cryptocurrency OR blockchain)",
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'sortBy': 'publishedAt',
                    'pageSize': 100,
                    'apiKey': self.api_keys['newsapi']
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for article_data in data.get('articles', []):
                            if article_data.get('title') and article_data.get('url'):
                                timestamp = datetime.fromisoformat(
                                    article_data['publishedAt'].replace('Z', '+00:00')
                                )
                                
                                article = NewsArticle(
                                    timestamp=timestamp,
                                    ticker=ticker,
                                    source=article_data.get('source', {}).get('name', 'NewsAPI'),
                                    headline=article_data['title'],
                                    url=article_data['url'],
                                    content=article_data.get('description', ''),
                                    article_id=self._generate_article_id(
                                        article_data['title'], 
                                        'NewsAPI', 
                                        timestamp
                                    )
                                )
                                
                                # Calculate sentiment
                                sentiment_text = f"{article.headline} {article.content or ''}"
                                article.sentiment_score, article.sentiment_label = self._calculate_sentiment(sentiment_text)
                                
                                articles.append(article)
                    else:
                        logger.error(f"NewsAPI error: {response.status} - {await response.text()}")
        
        except Exception as e:
            logger.error(f"Error fetching NewsAPI data: {e}")
        
        return articles
    
    async def _fetch_cryptocompare_news(self, ticker: str, days_back: int = 1) -> List[NewsArticle]:
        """Fetch news from CryptoCompare API"""
        if not self.api_keys.get('cryptocompare'):
            logger.warning("CryptoCompare API key not found, skipping CryptoCompare collection")
            return []
        
        if not await self._check_rate_limit('cryptocompare'):
            return []
        
        articles = []
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://min-api.cryptocompare.com/data/v2/news/"
                params = {
                    'lang': 'EN',
                    'sortOrder': 'latest',
                    'limit': 100
                }
                
                headers = {
                    'authorization': f"Apikey {self.api_keys['cryptocompare']}"
                }
                
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for article_data in data.get('Data', []):
                            # Check if article is related to our ticker
                            title = article_data.get('title', '').lower()
                            body = article_data.get('body', '').lower()
                            
                            if ticker.lower() in title or ticker.lower() in body:
                                timestamp = datetime.fromtimestamp(article_data['published_on'])
                                
                                article = NewsArticle(
                                    timestamp=timestamp,
                                    ticker=ticker,
                                    source=article_data.get('source', 'CryptoCompare'),
                                    headline=article_data['title'],
                                    url=article_data.get('url', ''),
                                    content=article_data.get('body', ''),
                                    article_id=self._generate_article_id(
                                        article_data['title'], 
                                        'CryptoCompare', 
                                        timestamp
                                    )
                                )
                                
                                # Calculate sentiment
                                sentiment_text = f"{article.headline} {article.content or ''}"
                                article.sentiment_score, article.sentiment_label = self._calculate_sentiment(sentiment_text)
                                
                                articles.append(article)
                    else:
                        logger.error(f"CryptoCompare error: {response.status} - {await response.text()}")
        
        except Exception as e:
            logger.error(f"Error fetching CryptoCompare data: {e}")
        
        return articles
    
    async def _fetch_coindesk_rss(self, ticker: str, days_back: int = 1) -> List[NewsArticle]:
        """Fetch news from CoinDesk RSS feed"""
        articles = []
        
        try:
            # CoinDesk RSS feeds
            rss_urls = [
                "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
                "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml&tag=bitcoin",
                "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml&tag=ethereum"
            ]
            
            for rss_url in rss_urls:
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries:
                    # Check if article is recent enough
                    pub_date = datetime(*entry.published_parsed[:6])
                    if pub_date < datetime.now() - timedelta(days=days_back):
                        continue
                    
                    # Check if article is related to our ticker
                    title = entry.title.lower()
                    summary = getattr(entry, 'summary', '').lower()
                    
                    if ticker.lower() in title or ticker.lower() in summary:
                        article = NewsArticle(
                            timestamp=pub_date,
                            ticker=ticker,
                            source='CoinDesk',
                            headline=entry.title,
                            url=entry.link,
                            content=getattr(entry, 'summary', ''),
                            article_id=self._generate_article_id(
                                entry.title, 
                                'CoinDesk', 
                                pub_date
                            )
                        )
                        
                        # Calculate sentiment
                        sentiment_text = f"{article.headline} {article.content or ''}"
                        article.sentiment_score, article.sentiment_label = self._calculate_sentiment(sentiment_text)
                        
                        articles.append(article)
        
        except Exception as e:
            logger.error(f"Error fetching CoinDesk RSS: {e}")
        
        return articles
    
    async def _fetch_reddit_news(self, ticker: str, days_back: int = 1) -> List[NewsArticle]:
        """Fetch news from Reddit (requires Reddit API setup)"""
        # This is a placeholder - Reddit API requires OAuth setup
        # For now, we'll skip Reddit to keep it simple
        logger.info("Reddit news collection not implemented yet")
        return []
    
    def _save_articles_to_db(self, articles: List[NewsArticle]) -> int:
        """Save articles to SQLite database"""
        if not articles:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        
        for article in articles:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO news_articles 
                    (article_id, timestamp, ticker, source, headline, url, content, sentiment_score, sentiment_label)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.article_id,
                    article.timestamp,
                    article.ticker,
                    article.source,
                    article.headline,
                    article.url,
                    article.content,
                    article.sentiment_score,
                    article.sentiment_label
                ))
                
                if cursor.rowcount > 0:
                    saved_count += 1
                    
            except Exception as e:
                logger.error(f"Error saving article {article.article_id}: {e}")
        
        conn.commit()
        conn.close()
        
        return saved_count
    
    def _save_articles_to_parquet(self, articles: List[NewsArticle], ticker: str) -> Path:
        """Save articles to Parquet file for analysis"""
        if not articles:
            return None
        
        # Convert to DataFrame
        data = []
        for article in articles:
            data.append({
                'timestamp': article.timestamp,
                'ticker': article.ticker,
                'source': article.source,
                'headline': article.headline,
                'url': article.url,
                'content': article.content,
                'sentiment_score': article.sentiment_score,
                'sentiment_label': article.sentiment_label,
                'article_id': article.article_id
            })
        
        df = pd.DataFrame(data)
        
        # Save to Parquet
        parquet_path = self.data_dir / f"news_{ticker}_{datetime.now().strftime('%Y%m%d')}.parquet"
        df.to_parquet(parquet_path, index=False)
        
        return parquet_path
    
    def _check_recent_news_cache(self, ticker: str, hours_back: int = 6) -> List[NewsArticle]:
        """Check if we have recent news in cache to avoid API calls"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check for recent articles
        cursor.execute('''
            SELECT timestamp, ticker, source, headline, url, content, sentiment_score, sentiment_label, article_id
            FROM news_articles
            WHERE ticker = ? AND timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours_back), (ticker,))
        
        rows = cursor.fetchall()
        conn.close()
        
        articles = []
        for row in rows:
            articles.append(NewsArticle(
                timestamp=datetime.fromisoformat(row[0]),
                ticker=row[1],
                source=row[2],
                headline=row[3],
                url=row[4],
                content=row[5],
                sentiment_score=row[6],
                sentiment_label=row[7],
                article_id=row[8]
            ))
        
        return articles
    
    def _should_collect_news(self, ticker: str, hours_back: int = 6) -> bool:
        """Check if we should collect news or use cached data"""
        recent_articles = self._check_recent_news_cache(ticker, hours_back)
        
        # If we have recent articles, don't collect new ones
        if recent_articles:
            logger.info(f"  Using cached news for {ticker} ({len(recent_articles)} recent articles)")
            return False
        
        return True
    
    async def collect_news(self, tickers: List[str] = None, days_back: int = 1, force_refresh: bool = False) -> Dict[str, Any]:
        """Collect news for specified tickers with intelligent caching"""
        if tickers is None:
            tickers = self.crypto_tickers
        
        logger.info(f"Collecting news for {len(tickers)} tickers over {days_back} days")
        
        all_articles = []
        collection_stats = {
            'total_articles': 0,
            'saved_articles': 0,
            'sources_used': [],
            'ticker_stats': {},
            'cached_articles': 0,
            'api_calls_made': 0
        }
        
        for ticker in tickers:
            logger.info(f"Collecting news for {ticker}")
            ticker_articles = []
            
            # Check if we should use cached data
            if not force_refresh:
                # Check for recent cached data
                cached_articles = self._check_recent_news_cache(ticker, hours_back=6)
                if cached_articles:
                    ticker_articles = cached_articles
                    collection_stats['cached_articles'] += len(cached_articles)
                    logger.info(f"  Using {len(cached_articles)} cached articles")
                else:
                    # No recent cached data, collect from APIs
                    logger.info(f"  No recent cached data, collecting from APIs...")
                    ticker_articles = await self._collect_from_apis(ticker, days_back, collection_stats)
            else:
                # Force refresh
                logger.info(f"  Force refresh enabled, collecting fresh data from APIs...")
                ticker_articles = await self._collect_from_apis(ticker, days_back, collection_stats)
            
            # Save to database (only new articles)
            saved_count = self._save_articles_to_db(ticker_articles)
            
            # Save to Parquet
            if ticker_articles:
                parquet_path = self._save_articles_to_parquet(ticker_articles, ticker)
                logger.info(f"  Saved {len(ticker_articles)} articles to {parquet_path}")
            
            # Update stats
            collection_stats['ticker_stats'][ticker] = {
                'total': len(ticker_articles),
                'saved': saved_count,
                'sources': len(set(article.source for article in ticker_articles)),
                'cached': len(ticker_articles) if not force_refresh and self._should_collect_news(ticker, hours_back=6) else 0
            }
            
            all_articles.extend(ticker_articles)
            collection_stats['total_articles'] += len(ticker_articles)
            collection_stats['saved_articles'] += saved_count
        
        # Overall sentiment analysis
        if all_articles:
            sentiment_scores = [a.sentiment_score for a in all_articles if a.sentiment_score is not None]
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                positive_count = sum(1 for s in sentiment_scores if s > 0.1)
                negative_count = sum(1 for s in sentiment_scores if s < -0.1)
                
                collection_stats['sentiment'] = {
                    'average_score': avg_sentiment,
                    'positive_articles': positive_count,
                    'negative_articles': negative_count,
                    'neutral_articles': len(sentiment_scores) - positive_count - negative_count
                }
        
        logger.info(f"News collection complete: {collection_stats['total_articles']} total articles, "
                   f"{collection_stats['saved_articles']} saved to database, "
                   f"{collection_stats['cached_articles']} from cache, "
                   f"{collection_stats['api_calls_made']} API calls made")
        
        return {
            'articles': all_articles,
            'stats': collection_stats,
            'collection_time': datetime.now()
        }
    
    async def _collect_from_apis(self, ticker: str, days_back: int, collection_stats: Dict) -> List[NewsArticle]:
        """Collect news from APIs for a specific ticker"""
        ticker_articles = []
        
        # Collect from all sources
        sources = [
            ('NewsAPI', self._fetch_newsapi_news(ticker, days_back)),
            ('CryptoCompare', self._fetch_cryptocompare_news(ticker, days_back)),
            ('CoinDesk', self._fetch_coindesk_rss(ticker, days_back)),
            ('Reddit', self._fetch_reddit_news(ticker, days_back))
        ]
        
        for source_name, source_coroutine in sources:
            try:
                articles = await source_coroutine
                if articles:
                    ticker_articles.extend(articles)
                    if source_name not in collection_stats['sources_used']:
                        collection_stats['sources_used'].append(source_name)
                    logger.info(f"  {source_name}: {len(articles)} articles")
                    collection_stats['api_calls_made'] += 1
            except Exception as e:
                logger.error(f"Error collecting from {source_name}: {e}")
        
        # Remove duplicates based on article_id
        unique_articles = {}
        for article in ticker_articles:
            if article.article_id not in unique_articles:
                unique_articles[article.article_id] = article
        
        return list(unique_articles.values())
    
    def get_recent_news(self, ticker: str = None, hours_back: int = 24, limit: int = 100) -> List[Dict]:
        """Get recent news from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT timestamp, ticker, source, headline, url, sentiment_score, sentiment_label
            FROM news_articles
            WHERE timestamp >= datetime('now', '-{} hours')
        '''.format(hours_back)
        
        params = []
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        articles = []
        for row in rows:
            articles.append({
                'timestamp': row[0],
                'ticker': row[1],
                'source': row[2],
                'headline': row[3],
                'url': row[4],
                'sentiment_score': row[5],
                'sentiment_label': row[6]
            })
        
        conn.close()
        return articles
    
    def get_news_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get news collection summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total articles
        cursor.execute('SELECT COUNT(*) FROM news_articles')
        total_articles = cursor.fetchone()[0]
        
        # Get articles by ticker
        cursor.execute('''
            SELECT ticker, COUNT(*) as count, AVG(sentiment_score) as avg_sentiment
            FROM news_articles
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY ticker
            ORDER BY count DESC
        '''.format(days_back))
        
        ticker_stats = {}
        for row in cursor.fetchall():
            ticker_stats[row[0]] = {
                'count': row[1],
                'avg_sentiment': row[2] if row[2] else 0.0
            }
        
        # Get articles by source
        cursor.execute('''
            SELECT source, COUNT(*) as count
            FROM news_articles
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY source
            ORDER BY count DESC
        '''.format(days_back))
        
        source_stats = {}
        for row in cursor.fetchall():
            source_stats[row[0]] = row[1]
        
        conn.close()
        
        return {
            'total_articles': total_articles,
            'ticker_stats': ticker_stats,
            'source_stats': source_stats,
            'days_analyzed': days_back
        }

# Example usage
async def main():
    """Example usage of NewsCollector"""
    # API keys (you'll need to get these)
    api_keys = {
        'newsapi': 'your_newsapi_key_here',
        'cryptocompare': 'your_cryptocompare_key_here'
    }
    
    collector = NewsCollector(api_keys=api_keys)
    
    # Collect news for BTC and ETH
    result = await collector.collect_news(['BTC', 'ETH'], days_back=1)
    
    print(f"Collected {result['stats']['total_articles']} articles")
    print(f"Sources used: {result['stats']['sources_used']}")
    
    # Get recent news
    recent_news = collector.get_recent_news('BTC', hours_back=24, limit=10)
    for article in recent_news:
        print(f"{article['timestamp']} | {article['source']} | {article['headline']}")

if __name__ == "__main__":
    asyncio.run(main())
