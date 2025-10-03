"""
Social Media APIs for collecting financial sentiment data

Supports:
- Reddit API (via PRAW)
- Twitter API (via Tweepy)
- Basic web scraping fallbacks
"""

import asyncio
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import yaml
import os
from loguru import logger
import time
import hashlib
import json


@dataclass
class SocialPost:
    """Standardized social media post format"""
    id: str
    content: str
    author: str
    platform: str
    created_at: datetime
    score: int
    url: str
    sentiment_score: Optional[float] = None
    entities: Optional[List[str]] = None
    symbols: Optional[List[str]] = None
    raw_data: Optional[Dict] = None


class RedditClient:
    """Reddit API client using PRAW"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str = "crypto_trading_bot"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.reddit = None
        
    async def initialize(self):
        """Initialize Reddit client"""
        try:
            import praw
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            logger.info("Reddit client initialized successfully")
        except ImportError:
            logger.warning("PRAW not installed. Reddit functionality disabled.")
            self.reddit = None
        except Exception as e:
            logger.error(f"Error initializing Reddit client: {e}")
            self.reddit = None
    
    async def get_posts(self, subreddit: str, limit: int = 100, time_filter: str = 'day') -> List[SocialPost]:
        """Get posts from a subreddit"""
        if not self.reddit:
            return []
        
        try:
            posts = []
            subreddit_obj = self.reddit.subreddit(subreddit)
            
            for submission in subreddit_obj.hot(limit=limit):
                post = SocialPost(
                    id=submission.id,
                    content=f"{submission.title} {submission.selftext}",
                    author=str(submission.author),
                    platform='reddit',
                    created_at=datetime.fromtimestamp(submission.created_utc),
                    score=submission.score,
                    url=f"https://reddit.com{submission.permalink}",
                    raw_data={
                        'subreddit': subreddit,
                        'upvote_ratio': submission.upvote_ratio,
                        'num_comments': submission.num_comments
                    }
                )
                posts.append(post)
            
            logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching Reddit posts: {e}")
            return []
    
    async def search_posts(self, query: str, subreddit: str = None, limit: int = 100) -> List[SocialPost]:
        """Search for posts containing query"""
        if not self.reddit:
            return []
        
        try:
            posts = []
            if subreddit:
                subreddit_obj = self.reddit.subreddit(subreddit)
                search_results = subreddit_obj.search(query, limit=limit)
            else:
                search_results = self.reddit.subreddit('all').search(query, limit=limit)
            
            for submission in search_results:
                post = SocialPost(
                    id=submission.id,
                    content=f"{submission.title} {submission.selftext}",
                    author=str(submission.author),
                    platform='reddit',
                    created_at=datetime.fromtimestamp(submission.created_utc),
                    score=submission.score,
                    url=f"https://reddit.com{submission.permalink}",
                    raw_data={
                        'subreddit': str(submission.subreddit),
                        'upvote_ratio': submission.upvote_ratio,
                        'num_comments': submission.num_comments
                    }
                )
                posts.append(post)
            
            logger.info(f"Found {len(posts)} posts for query: {query}")
            return posts
            
        except Exception as e:
            logger.error(f"Error searching Reddit posts: {e}")
            return []


class TwitterClient:
    """Twitter API client using Tweepy"""
    
    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token
        self.api = None
        
    async def initialize(self):
        """Initialize Twitter client"""
        try:
            import tweepy
            self.api = tweepy.Client(bearer_token=self.bearer_token)
            logger.info("Twitter client initialized successfully")
        except ImportError:
            logger.warning("Tweepy not installed. Twitter functionality disabled.")
            self.api = None
        except Exception as e:
            logger.error(f"Error initializing Twitter client: {e}")
            self.api = None
    
    async def search_tweets(self, query: str, max_results: int = 100) -> List[SocialPost]:
        """Search for tweets containing query"""
        if not self.api:
            return []
        
        try:
            tweets = self.api.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                tweet_fields=['created_at', 'public_metrics', 'author_id']
            )
            
            posts = []
            if tweets.data:
                for tweet in tweets.data:
                    post = SocialPost(
                        id=tweet.id,
                        content=tweet.text,
                        author=str(tweet.author_id),
                        platform='twitter',
                        created_at=tweet.created_at,
                        score=tweet.public_metrics.get('like_count', 0),
                        url=f"https://twitter.com/user/status/{tweet.id}",
                        raw_data={
                            'retweet_count': tweet.public_metrics.get('retweet_count', 0),
                            'reply_count': tweet.public_metrics.get('reply_count', 0),
                            'quote_count': tweet.public_metrics.get('quote_count', 0)
                        }
                    )
                    posts.append(post)
            
            logger.info(f"Found {len(posts)} tweets for query: {query}")
            return posts
            
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []
    
    async def get_user_tweets(self, username: str, max_results: int = 100) -> List[SocialPost]:
        """Get tweets from a specific user"""
        if not self.api:
            return []
        
        try:
            user = self.api.get_user(username=username)
            if not user.data:
                return []
            
            tweets = self.api.get_users_tweets(
                id=user.data.id,
                max_results=min(max_results, 100),
                tweet_fields=['created_at', 'public_metrics']
            )
            
            posts = []
            if tweets.data:
                for tweet in tweets.data:
                    post = SocialPost(
                        id=tweet.id,
                        content=tweet.text,
                        author=username,
                        platform='twitter',
                        created_at=tweet.created_at,
                        score=tweet.public_metrics.get('like_count', 0),
                        url=f"https://twitter.com/{username}/status/{tweet.id}",
                        raw_data={
                            'retweet_count': tweet.public_metrics.get('retweet_count', 0),
                            'reply_count': tweet.public_metrics.get('reply_count', 0),
                            'quote_count': tweet.public_metrics.get('quote_count', 0)
                        }
                    )
                    posts.append(post)
            
            logger.info(f"Found {len(posts)} tweets from @{username}")
            return posts
            
        except Exception as e:
            logger.error(f"Error getting user tweets: {e}")
            return []


class SocialMediaCollector:
    """Main class for collecting social media data"""
    
    def __init__(self, api_keys_path: str = 'config/api_keys.yaml'):
        self.api_keys_path = api_keys_path
        self.reddit_client = None
        self.twitter_client = None
        self.api_keys = self._load_api_keys()
        
    def _load_api_keys(self) -> Dict[str, Any]:
        """Load API keys from configuration file"""
        try:
            with open(self.api_keys_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            return {}
    
    async def initialize(self):
        """Initialize all social media clients"""
        # Initialize Reddit
        reddit_config = self.api_keys.get('reddit', {})
        if reddit_config.get('client_id') and reddit_config.get('client_secret'):
            self.reddit_client = RedditClient(
                client_id=reddit_config['client_id'],
                client_secret=reddit_config['client_secret']
            )
            await self.reddit_client.initialize()
        
        # Initialize Twitter
        twitter_config = self.api_keys.get('twitter', {})
        if twitter_config.get('bearer_token'):
            self.twitter_client = TwitterClient(
                bearer_token=twitter_config['bearer_token']
            )
            await self.twitter_client.initialize()
    
    async def collect_crypto_sentiment(self, hours_back: int = 24) -> List[SocialPost]:
        """Collect crypto-related social media posts"""
        posts = []
        
        # Reddit crypto subreddits
        crypto_subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'cryptomarkets']
        for subreddit in crypto_subreddits:
            if self.reddit_client:
                subreddit_posts = await self.reddit_client.get_posts(subreddit, limit=50)
                posts.extend(subreddit_posts)
        
        # Twitter crypto search
        crypto_queries = ['bitcoin', 'crypto', 'ethereum', 'cryptocurrency']
        for query in crypto_queries:
            if self.twitter_client:
                twitter_posts = await self.twitter_client.search_tweets(query, max_results=25)
                posts.extend(twitter_posts)
        
        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_posts = [post for post in posts if post.created_at >= cutoff_time]
        
        logger.info(f"Collected {len(recent_posts)} crypto-related social media posts")
        return recent_posts
    
    async def collect_stock_sentiment(self, hours_back: int = 24) -> List[SocialPost]:
        """Collect stock-related social media posts"""
        posts = []
        
        # Reddit stock subreddits
        stock_subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting']
        for subreddit in stock_subreddits:
            if self.reddit_client:
                subreddit_posts = await self.reddit_client.get_posts(subreddit, limit=50)
                posts.extend(subreddit_posts)
        
        # Twitter stock search
        stock_queries = ['stocks', 'investing', 'trading', 'nasdaq', 'nyse']
        for query in stock_queries:
            if self.twitter_client:
                twitter_posts = await self.twitter_client.search_tweets(query, max_results=25)
                posts.extend(twitter_posts)
        
        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_posts = [post for post in posts if post.created_at >= cutoff_time]
        
        logger.info(f"Collected {len(recent_posts)} stock-related social media posts")
        return recent_posts
    
    def posts_to_dataframe(self, posts: List[SocialPost]) -> pd.DataFrame:
        """Convert social media posts to DataFrame"""
        if not posts:
            return pd.DataFrame()
        
        data = []
        for post in posts:
            data.append({
                'id': post.id,
                'content': post.content,
                'author': post.author,
                'platform': post.platform,
                'created_at': post.created_at,
                'score': post.score,
                'url': post.url,
                'sentiment_score': post.sentiment_score,
                'entities': '|'.join(post.entities) if post.entities else '',
                'symbols': '|'.join(post.symbols) if post.symbols else '',
                'raw_data': json.dumps(post.raw_data) if post.raw_data else ''
            })
        
        return pd.DataFrame(data)


# Convenience functions for backward compatibility
async def collect_crypto_social_sentiment(api_keys_path: str = 'config/api_keys.yaml', hours_back: int = 24) -> List[Dict]:
    """Collect crypto social media sentiment (placeholder)"""
    collector = SocialMediaCollector(api_keys_path)
    await collector.initialize()
    posts = await collector.collect_crypto_sentiment(hours_back)
    return collector.posts_to_dataframe(posts).to_dict('records')

async def collect_stock_social_sentiment(api_keys_path: str = 'config/api_keys.yaml', hours_back: int = 24) -> List[Dict]:
    """Collect stock social media sentiment (placeholder)"""
    collector = SocialMediaCollector(api_keys_path)
    await collector.initialize()
    posts = await collector.collect_stock_sentiment(hours_back)
    return collector.posts_to_dataframe(posts).to_dict('records')

