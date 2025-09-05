"""
News API Clients for collecting financial news data

Supports multiple news sources:
- NewsAPI (free tier available)
- RavenPack (premium financial news)
- Custom RSS feeds
- Web scraping fallbacks
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import yaml
import os
from loguru import logger
import time
import hashlib


@dataclass
class NewsArticle:
    """Standardized news article format"""
    id: str
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    category: str
    sentiment_score: Optional[float] = None
    entities: Optional[List[str]] = None
    symbols: Optional[List[str]] = None
    raw_data: Optional[Dict] = None


class NewsAPIClient:
    """NewsAPI.org client for general news collection"""
    
    def __init__(self, api_key: str, base_url: str = "https://newsapi.org/v2"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def get_articles(
        self, 
        query: str,
        sources: Optional[List[str]] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 100
    ) -> List[NewsArticle]:
        """
        Fetch articles from NewsAPI
        
        Args:
            query: Search query (e.g., "bitcoin", "cryptocurrency")
            sources: List of source IDs to filter by
            from_date: Start date for articles
            to_date: End date for articles
            language: Language code (default: "en")
            sort_by: Sort order ("publishedAt", "relevancy", "popularity")
            page_size: Number of articles per page (max 100)
        """
        await self._rate_limit()
        
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        params = {
            'apiKey': self.api_key,
            'q': query,
            'language': language,
            'sortBy': sort_by,
            'pageSize': min(page_size, 100)
        }
        
        if sources:
            params['sources'] = ','.join(sources)
        if from_date:
            params['from'] = from_date.strftime('%Y-%m-%d')
        if to_date:
            params['to'] = to_date.strftime('%Y-%m-%d')
        
        try:
            async with self.session.get(f"{self.base_url}/everything", params=params) as response:
                if response.status == 429:
                    logger.warning("Rate limit exceeded, waiting 60 seconds")
                    await asyncio.sleep(60)
                    return await self.get_articles(query, sources, from_date, to_date, language, sort_by, page_size)
                
                response.raise_for_status()
                data = await response.json()
                
                articles = []
                for article_data in data.get('articles', []):
                    article = self._parse_article(article_data, query)
                    if article:
                        articles.append(article)
                
                logger.info(f"Fetched {len(articles)} articles for query: {query}")
                return articles
                
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching articles: {e}")
            return []
    
    def _parse_article(self, article_data: Dict, query: str) -> Optional[NewsArticle]:
        """Parse NewsAPI article data into standardized format"""
        try:
            # Generate unique ID
            article_id = hashlib.md5(
                f"{article_data.get('url', '')}{article_data.get('publishedAt', '')}".encode()
            ).hexdigest()
            
            # Parse published date
            published_at = datetime.fromisoformat(
                article_data.get('publishedAt', '').replace('Z', '+00:00')
            )
            
            return NewsArticle(
                id=article_id,
                title=article_data.get('title', ''),
                content=article_data.get('content', '') or article_data.get('description', ''),
                url=article_data.get('url', ''),
                source=article_data.get('source', {}).get('name', 'Unknown'),
                published_at=published_at,
                category=self._categorize_article(article_data.get('title', ''), query),
                raw_data=article_data
            )
        except Exception as e:
            logger.error(f"Error parsing article: {e}")
            return None
    
    def _categorize_article(self, title: str, query: str) -> str:
        """Simple categorization based on title and query"""
        title_lower = title.lower()
        query_lower = query.lower()
        
        if any(word in title_lower for word in ['bitcoin', 'btc', 'cryptocurrency', 'crypto']):
            return 'crypto'
        elif any(word in title_lower for word in ['stock', 'equity', 'nasdaq', 'nyse', 's&p']):
            return 'stocks'
        elif any(word in title_lower for word in ['forex', 'fx', 'currency', 'dollar', 'euro']):
            return 'forex'
        elif any(word in title_lower for word in ['commodity', 'gold', 'oil', 'silver']):
            return 'commodities'
        else:
            return 'general'
    
    async def get_sources(self, category: str = "business", language: str = "en") -> List[Dict]:
        """Get available news sources"""
        await self._rate_limit()
        
        params = {
            'apiKey': self.api_key,
            'category': category,
            'language': language
        }
        
        try:
            async with self.session.get(f"{self.base_url}/sources", params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('sources', [])
        except Exception as e:
            logger.error(f"Error fetching sources: {e}")
            return []


class RavenPackClient:
    """RavenPack client for premium financial news (requires subscription)"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.ravenpack.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_news(
        self,
        start_date: datetime,
        end_date: datetime,
        entities: Optional[List[str]] = None,
        topics: Optional[List[str]] = None
    ) -> List[NewsArticle]:
        """
        Fetch news from RavenPack
        
        Args:
            start_date: Start date for news
            end_date: End date for news
            entities: List of entity names to filter by
            topics: List of topics to filter by
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        # This is a simplified implementation
        # Real RavenPack API would have more complex query parameters
        params = {
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d'),
            'format': 'json'
        }
        
        if entities:
            params['entities'] = ','.join(entities)
        if topics:
            params['topics'] = ','.join(topics)
        
        try:
            async with self.session.get(f"{self.base_url}/news", params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                articles = []
                for item in data.get('data', []):
                    article = self._parse_ravenpack_article(item)
                    if article:
                        articles.append(article)
                
                logger.info(f"Fetched {len(articles)} articles from RavenPack")
                return articles
                
        except Exception as e:
            logger.error(f"Error fetching RavenPack news: {e}")
            return []
    
    def _parse_ravenpack_article(self, item: Dict) -> Optional[NewsArticle]:
        """Parse RavenPack article data"""
        try:
            article_id = item.get('rp_id', '')
            published_at = datetime.fromisoformat(item.get('timestamp', '').replace('Z', '+00:00'))
            
            return NewsArticle(
                id=article_id,
                title=item.get('title', ''),
                content=item.get('body', ''),
                url=item.get('url', ''),
                source=item.get('source', 'RavenPack'),
                published_at=published_at,
                category='financial',
                sentiment_score=item.get('sentiment_score'),
                entities=item.get('entities', []),
                symbols=item.get('symbols', []),
                raw_data=item
            )
        except Exception as e:
            logger.error(f"Error parsing RavenPack article: {e}")
            return None


class RSSNewsClient:
    """RSS feed client for news collection"""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_rss_articles(self, rss_url: str, category: str = "general") -> List[NewsArticle]:
        """Fetch articles from RSS feed"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            async with self.session.get(rss_url) as response:
                response.raise_for_status()
                content = await response.text()
                
                # Parse RSS XML (simplified - would use feedparser in real implementation)
                articles = self._parse_rss_content(content, category)
                
                logger.info(f"Fetched {len(articles)} articles from RSS: {rss_url}")
                return articles
                
        except Exception as e:
            logger.error(f"Error fetching RSS articles: {e}")
            return []
    
    def _parse_rss_content(self, content: str, category: str) -> List[NewsArticle]:
        """Parse RSS XML content (simplified implementation)"""
        # This would use feedparser library in a real implementation
        # For now, return empty list
        return []


async def collect_crypto_news(
    newsapi_key: str,
    hours_back: int = 24,
    max_articles: int = 1000
) -> List[NewsArticle]:
    """
    Collect cryptocurrency news from multiple sources
    
    Args:
        newsapi_key: NewsAPI API key
        hours_back: How many hours back to collect news
        max_articles: Maximum number of articles to collect
    """
    crypto_queries = [
        "bitcoin", "ethereum", "cryptocurrency", "crypto", "blockchain",
        "defi", "nft", "altcoin", "binance", "coinbase"
    ]
    
    all_articles = []
    start_time = datetime.now() - timedelta(hours=hours_back)
    
    async with NewsAPIClient(newsapi_key) as client:
        for query in crypto_queries:
            try:
                articles = await client.get_articles(
                    query=query,
                    from_date=start_time,
                    page_size=100
                )
                all_articles.extend(articles)
                
                if len(all_articles) >= max_articles:
                    break
                    
            except Exception as e:
                logger.error(f"Error collecting news for query '{query}': {e}")
                continue
    
    # Remove duplicates based on URL
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        if article.url not in seen_urls:
            seen_urls.add(article.url)
            unique_articles.append(article)
    
    logger.info(f"Collected {len(unique_articles)} unique crypto news articles")
    return unique_articles[:max_articles]


async def collect_stock_news(
    newsapi_key: str,
    hours_back: int = 24,
    max_articles: int = 1000
) -> List[NewsArticle]:
    """
    Collect stock market news from multiple sources
    """
    stock_queries = [
        "stock market", "nasdaq", "nyse", "s&p 500", "dow jones",
        "earnings", "ipo", "merger", "acquisition", "dividend"
    ]
    
    all_articles = []
    start_time = datetime.now() - timedelta(hours=hours_back)
    
    async with NewsAPIClient(newsapi_key) as client:
        for query in stock_queries:
            try:
                articles = await client.get_articles(
                    query=query,
                    from_date=start_time,
                    page_size=100
                )
                all_articles.extend(articles)
                
                if len(all_articles) >= max_articles:
                    break
                    
            except Exception as e:
                logger.error(f"Error collecting news for query '{query}': {e}")
                continue
    
    # Remove duplicates
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        if article.url not in seen_urls:
            seen_urls.add(article.url)
            unique_articles.append(article)
    
    logger.info(f"Collected {len(unique_articles)} unique stock news articles")
    return unique_articles[:max_articles]


if __name__ == "__main__":
    # Example usage
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def main():
        newsapi_key = os.getenv('NEWSAPI_KEY')
        if not newsapi_key:
            print("Please set NEWSAPI_KEY environment variable")
            return
        
        # Collect crypto news
        crypto_articles = await collect_crypto_news(newsapi_key, hours_back=24)
        print(f"Collected {len(crypto_articles)} crypto articles")
        
        # Collect stock news
        stock_articles = await collect_stock_news(newsapi_key, hours_back=24)
        print(f"Collected {len(stock_articles)} stock articles")
    
    asyncio.run(main())
