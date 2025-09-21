#!/usr/bin/env python3
"""
Test script for news collection system
"""

import asyncio
import logging
import yaml
from pathlib import Path
from data_ingestion.news_collector import NewsCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)

async def test_news_collection():
    """Test the news collection system"""
    
    # Load API keys from environment variables
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_keys = {
        'newsapi': os.getenv('NEWSAPI_KEY', ''),
        'cryptocompare': os.getenv('CRYPTOCOMPARE_API_KEY', '')
    }
    
    # Filter out empty values
    api_keys = {k: v for k, v in api_keys.items() if v and v.strip()}
    
    if not api_keys:
        logger.warning("No API keys found in environment variables. Using free sources only (CoinDesk RSS)")
    
    # Initialize collector
    collector = NewsCollector(api_keys=api_keys)
    
    # Test with BTC and ETH
    tickers = ['BTC', 'ETH']
    
    logger.info(f"Testing news collection for {tickers}")
    
    # Collect news
    result = await collector.collect_news(tickers, days_back=1)
    
    # Print results
    stats = result['stats']
    print("\n" + "="*60)
    print("NEWS COLLECTION RESULTS")
    print("="*60)
    print(f"Total articles collected: {stats['total_articles']}")
    print(f"Articles saved to database: {stats['saved_articles']}")
    print(f"Sources used: {', '.join(stats['sources_used'])}")
    
    if 'sentiment' in stats:
        sentiment = stats['sentiment']
        print(f"\nSentiment Analysis:")
        print(f"  Average score: {sentiment['average_score']:.3f}")
        print(f"  Positive articles: {sentiment['positive_articles']}")
        print(f"  Negative articles: {sentiment['negative_articles']}")
        print(f"  Neutral articles: {sentiment['neutral_articles']}")
    
    print(f"\nTicker Statistics:")
    for ticker, ticker_stats in stats['ticker_stats'].items():
        print(f"  {ticker}: {ticker_stats['total']} articles, {ticker_stats['saved']} saved")
    
    # Show recent news
    print(f"\nRecent News (last 24 hours):")
    recent_news = collector.get_recent_news(hours_back=24, limit=5)
    
    for i, article in enumerate(recent_news, 1):
        sentiment_icon = "📈" if article['sentiment_score'] and article['sentiment_score'] > 0.1 else "📉" if article['sentiment_score'] and article['sentiment_score'] < -0.1 else "➡️"
        print(f"  {i}. {sentiment_icon} [{article['ticker']}] {article['headline']}")
        print(f"     Source: {article['source']} | Sentiment: {article['sentiment_label']} ({article['sentiment_score']:.3f})")
        print(f"     Time: {article['timestamp']}")
        print()
    
    # Get summary
    summary = collector.get_news_summary(days_back=7)
    print(f"\n7-Day Summary:")
    print(f"  Total articles in database: {summary['total_articles']}")
    print(f"  Articles by ticker: {summary['ticker_stats']}")
    print(f"  Articles by source: {summary['source_stats']}")

if __name__ == "__main__":
    asyncio.run(test_news_collection())
