#!/usr/bin/env python3
"""
News Management System

A comprehensive script for managing crypto news collection with different modes:
- collect: Collect news for specified tickers
- backfill: Collect historical news data
- validate: Check news data quality
- summary: Show news collection statistics
"""

import asyncio
import argparse
import logging
import yaml
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from data_ingestion.news_collector import NewsCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

def load_api_keys():
    """Load API keys from environment variables"""
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
    
    return api_keys

async def collect_news_mode(tickers, days_back, sources, force_refresh=False):
    """Collect news for specified tickers"""
    logger.info(f"🚀 Starting news collection mode")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Days back: {days_back}")
    logger.info(f"Sources: {sources}")
    logger.info(f"Force refresh: {force_refresh}")
    
    api_keys = load_api_keys()
    collector = NewsCollector(api_keys=api_keys)
    
    # Collect news
    result = await collector.collect_news(tickers, days_back=days_back, force_refresh=force_refresh)
    
    # Print results
    stats = result['stats']
    print("\n" + "="*60)
    print("NEWS COLLECTION RESULTS")
    print("="*60)
    print(f"✅ Total articles collected: {stats['total_articles']}")
    print(f"💾 Articles saved to database: {stats['saved_articles']}")
    print(f"📰 Sources used: {', '.join(stats['sources_used'])}")
    print(f"🔄 Cached articles: {stats.get('cached_articles', 0)}")
    print(f"🌐 API calls made: {stats.get('api_calls_made', 0)}")
    
    if 'sentiment' in stats:
        sentiment = stats['sentiment']
        print(f"\n📊 Sentiment Analysis:")
        print(f"  Average score: {sentiment['average_score']:.3f}")
        print(f"  📈 Positive articles: {sentiment['positive_articles']}")
        print(f"  📉 Negative articles: {sentiment['negative_articles']}")
        print(f"  ➡️ Neutral articles: {sentiment['neutral_articles']}")
    
    print(f"\n📈 Ticker Statistics:")
    for ticker, ticker_stats in stats['ticker_stats'].items():
        cached_info = f", {ticker_stats.get('cached', 0)} cached" if ticker_stats.get('cached', 0) > 0 else ""
        print(f"  {ticker}: {ticker_stats['total']} articles, {ticker_stats['saved']} saved{cached_info}")
    
    return result

async def backfill_mode(tickers, days_back, sources):
    """Backfill historical news data"""
    logger.info(f"📚 Starting news backfill mode")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Days back: {days_back}")
    
    api_keys = load_api_keys()
    collector = NewsCollector(api_keys=api_keys)
    
    # Backfill in chunks to avoid rate limits
    chunk_size = 7  # Process 7 days at a time
    total_chunks = (days_back + chunk_size - 1) // chunk_size
    
    logger.info(f"Processing {days_back} days in {total_chunks} chunks of {chunk_size} days each")
    
    all_results = []
    
    for chunk in range(total_chunks):
        start_day = chunk * chunk_size
        end_day = min((chunk + 1) * chunk_size, days_back)
        
        logger.info(f"Processing chunk {chunk + 1}/{total_chunks}: days {start_day + 1}-{end_day}")
        
        # Collect news for this chunk
        result = await collector.collect_news(tickers, days_back=end_day)
        all_results.append(result)
        
        # Rate limiting delay
        if chunk < total_chunks - 1:
            logger.info("Waiting 30 seconds before next chunk...")
            await asyncio.sleep(30)
    
    # Aggregate results
    total_articles = sum(r['stats']['total_articles'] for r in all_results)
    total_saved = sum(r['stats']['saved_articles'] for r in all_results)
    
    print("\n" + "="*60)
    print("NEWS BACKFILL RESULTS")
    print("="*60)
    print(f"✅ Total articles collected: {total_articles}")
    print(f"💾 Articles saved to database: {total_saved}")
    print(f"📅 Period: {days_back} days")
    
    return all_results

async def validate_mode(tickers, days_back):
    """Validate news data quality"""
    logger.info(f"🔍 Starting news validation mode")
    
    api_keys = load_api_keys()
    collector = NewsCollector(api_keys=api_keys)
    
    # Get news summary
    summary = collector.get_news_summary(days_back=days_back)
    
    print("\n" + "="*60)
    print("NEWS DATA VALIDATION")
    print("="*60)
    print(f"📊 Total articles in database: {summary['total_articles']}")
    print(f"📅 Period analyzed: {days_back} days")
    
    print(f"\n📈 Articles by Ticker:")
    for ticker, stats in summary['ticker_stats'].items():
        avg_sentiment = stats['avg_sentiment']
        sentiment_icon = "📈" if avg_sentiment > 0.1 else "📉" if avg_sentiment < -0.1 else "➡️"
        print(f"  {sentiment_icon} {ticker}: {stats['count']} articles (avg sentiment: {avg_sentiment:.3f})")
    
    print(f"\n📰 Articles by Source:")
    for source, count in summary['source_stats'].items():
        print(f"  {source}: {count} articles")
    
    # Data quality checks
    print(f"\n🔍 Data Quality Checks:")
    
    # Check for recent data
    recent_news = collector.get_recent_news(hours_back=24, limit=1)
    if recent_news:
        print(f"  ✅ Recent data available (last 24h)")
    else:
        print(f"  ⚠️ No recent data (last 24h)")
    
    # Check for sentiment data
    total_with_sentiment = sum(1 for ticker_stats in summary['ticker_stats'].values() 
                              if ticker_stats['avg_sentiment'] is not None)
    if total_with_sentiment > 0:
        print(f"  ✅ Sentiment analysis available for {total_with_sentiment} tickers")
    else:
        print(f"  ⚠️ No sentiment analysis data")
    
    return summary

async def summary_mode(tickers, days_back):
    """Show news collection statistics"""
    logger.info(f"📊 Starting news summary mode")
    
    api_keys = load_api_keys()
    collector = NewsCollector(api_keys=api_keys)
    
    # Get recent news
    recent_news = collector.get_recent_news(hours_back=24, limit=10)
    
    print("\n" + "="*60)
    print("RECENT NEWS SUMMARY")
    print("="*60)
    print(f"📰 Last 24 hours: {len(recent_news)} articles")
    
    if recent_news:
        print(f"\n🔥 Top Headlines:")
        for i, article in enumerate(recent_news[:5], 1):
            sentiment_icon = "📈" if article['sentiment_score'] and article['sentiment_score'] > 0.1 else "📉" if article['sentiment_score'] and article['sentiment_score'] < -0.1 else "➡️"
            print(f"  {i}. {sentiment_icon} [{article['ticker']}] {article['headline']}")
            print(f"     Source: {article['source']} | Sentiment: {article['sentiment_label']} ({article['sentiment_score']:.3f})")
            print(f"     Time: {article['timestamp']}")
            print()
    
    # Get overall summary
    summary = collector.get_news_summary(days_back=days_back)
    
    print(f"\n📊 {days_back}-Day Summary:")
    print(f"  Total articles: {summary['total_articles']}")
    print(f"  Articles by ticker: {summary['ticker_stats']}")
    print(f"  Articles by source: {summary['source_stats']}")
    
    return summary

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="News Management System")
    parser.add_argument("--mode", type=str, default="collect",
                       choices=["collect", "backfill", "validate", "summary"],
                       help="Operation mode")
    parser.add_argument("--tickers", type=str, nargs='*', default=['BTC', 'ETH'],
                       help="Tickers to process (e.g., BTC ETH UNI)")
    parser.add_argument("--days", type=int, default=1,
                       help="Number of days for collection/validation")
    parser.add_argument("--sources", type=str, nargs='*',
                       choices=["newsapi", "cryptocompare", "coindesk", "reddit"],
                       help="News sources to use")
    parser.add_argument("--force-refresh", action='store_true',
                       help="Force refresh news data (ignore cache)")
    
    args = parser.parse_args()
    
    # Default sources if not specified
    if not args.sources:
        args.sources = ["newsapi", "cryptocompare", "coindesk"]
    
    try:
        if args.mode == "collect":
            await collect_news_mode(args.tickers, args.days, args.sources, args.force_refresh)
        elif args.mode == "backfill":
            await backfill_mode(args.tickers, args.days, args.sources)
        elif args.mode == "validate":
            await validate_mode(args.tickers, args.days)
        elif args.mode == "summary":
            await summary_mode(args.tickers, args.days)
        
        print("\n✅ Operation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
