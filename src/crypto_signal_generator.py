#!/usr/bin/env python3
"""
Unified Crypto Signal Generator
Multi-symbol signal generation with sentiment integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
from loguru import logger

try:
    from .crypto_analysis_engine import CryptoAnalysisEngine
    from .data_ingestion.news_collector import NewsCollector
except ImportError:
    # Fallback for when running as script
    from crypto_analysis_engine import CryptoAnalysisEngine
    from data_ingestion.news_collector import NewsCollector

class CryptoSentimentGenerator:
    """Unified sentiment-enhanced signal generator for all crypto assets"""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.analysis_engine = CryptoAnalysisEngine()
        self.news_collector = NewsCollector()
        
    def load_news_data(self, symbol: str) -> pd.DataFrame:
        """Load news data for any symbol"""
        try:
            # Get recent news from NewsCollector
            news_articles = self.news_collector.get_recent_news(ticker=symbol, hours_back=168, limit=1000)  # 7 days
            
            if not news_articles:
                logger.warning(f"No news data available for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            news_data = []
            for article in news_articles:
                news_data.append({
                    'timestamp': pd.to_datetime(article['timestamp']),
                    'ticker': article['ticker'],
                    'source': article['source'],
                    'headline': article['headline'],
                    'url': article['url'],
                    'content': article.get('content', ''),
                    'sentiment_score': article.get('sentiment_score', 0.0),
                    'sentiment_label': article.get('sentiment_label', 'neutral'),
                    'article_id': article.get('article_id', '')
                })
            
            news_df = pd.DataFrame(news_data)
            
            # Ensure timestamp is timezone-aware
            if 'timestamp' in news_df.columns:
                news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
                if news_df['timestamp'].dt.tz is None:
                    news_df['timestamp'] = news_df['timestamp'].dt.tz_localize('UTC')
                else:
                    news_df['timestamp'] = news_df['timestamp'].dt.tz_convert('UTC')
            
            logger.info(f"Loaded {len(news_df)} news articles for {symbol}")
            return news_df
            
        except Exception as e:
            logger.error(f"Failed to load news data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_sentiment_for_timestamp(self, timestamp: datetime, news_df: pd.DataFrame, 
                                  window_minutes: int = 240) -> float:
        """Get average sentiment for headlines within ±window_minutes of timestamp"""
        if news_df.empty:
            return 0.0
            
        # Convert timestamp to pandas Timestamp in UTC for comparison with news data
        if isinstance(timestamp, pd.Timestamp):
            if timestamp.tz is None:
                timestamp = timestamp.tz_localize('UTC')
            else:
                timestamp = timestamp.tz_convert('UTC')
        else:
            # Convert datetime to pandas Timestamp
            timestamp = pd.Timestamp(timestamp, tz='UTC')
        
        # Define time window
        start_time = timestamp - timedelta(minutes=window_minutes)
        end_time = timestamp + timedelta(minutes=window_minutes)
        
        # Filter news within time window
        mask = (news_df['timestamp'] >= start_time) & (news_df['timestamp'] <= end_time)
        window_news = news_df[mask]
        
        if len(window_news) == 0:
            return 0.0
            
        # Calculate mean sentiment
        mean_sentiment = window_news['sentiment_score'].mean()
        
        # Normalize sentiment to [-1, 1] range
        normalized_sentiment = np.clip(mean_sentiment * 10, -1, 1)
        
        return normalized_sentiment
    
    def generate_enhanced_signals(self, price_df: pd.DataFrame, 
                                indicators_df: Optional[pd.DataFrame] = None,
                                symbol: str = "BTC") -> Dict[str, any]:
        """Generate sentiment-enhanced signals for any symbol"""
        try:
            if price_df.empty:
                raise ValueError("No price data available")
            
            # Calculate technical indicators if not provided
            if indicators_df is None:
                indicators_df = self.analysis_engine.calculate_technical_indicators(price_df)
            
            # Generate base technical signals
            base_signals = self.analysis_engine.generate_signals(indicators_df, symbol)
            
            # Load news data
            news_df = self.load_news_data(symbol)
            
            # Get current timestamp
            current_timestamp = price_df.index[-1]
            
            # Convert to pandas Timestamp in UTC for comparison with news data
            if isinstance(current_timestamp, pd.Timestamp):
                if current_timestamp.tz is None:
                    current_timestamp = current_timestamp.tz_localize('UTC')
                else:
                    current_timestamp = current_timestamp.tz_convert('UTC')
            else:
                # Convert datetime to pandas Timestamp
                current_timestamp = pd.Timestamp(current_timestamp, tz='UTC')
            
            # Get sentiment for current time (use wider window for recent news)
            sentiment_score = self.get_sentiment_for_timestamp(current_timestamp, news_df, window_minutes=240)
            
            # Calculate sentiment multiplier
            sentiment_multiplier = 1 + (self.alpha * sentiment_score)
            
            # Apply sentiment enhancement to base signal
            enhanced_strength = base_signals['signal_strength'] * sentiment_multiplier
            
            # Determine enhanced signal type
            if enhanced_strength > 0.3:
                enhanced_signal_type = "BUY"
            elif enhanced_strength < -0.3:
                enhanced_signal_type = "SELL"
            else:
                enhanced_signal_type = "HOLD"
            
            # Calculate enhanced confidence
            enhanced_confidence = min(abs(enhanced_strength), 1.0)
            
            # Add sentiment reasoning
            sentiment_reasons = []
            if abs(sentiment_score) > 0.1:
                sentiment_reasons.append(f"Sentiment: {sentiment_score:.3f} (α={self.alpha})")
                sentiment_reasons.append(f"Multiplier: {sentiment_multiplier:.3f}")
            
            # Combine all reasons
            all_reasons = base_signals['reasons'] + sentiment_reasons
            
            return {
                'symbol': symbol,
                'base_signal_type': base_signals['signal_type'],
                'enhanced_signal_type': enhanced_signal_type,
                'base_signal_strength': base_signals['signal_strength'],
                'enhanced_signal_strength': enhanced_strength,
                'base_confidence': base_signals['confidence'],
                'enhanced_confidence': enhanced_confidence,
                'sentiment_score': sentiment_score,
                'sentiment_multiplier': sentiment_multiplier,
                'alpha': self.alpha,
                'current_price': base_signals['current_price'],
                'current_volume': base_signals['current_volume'],
                'reasons': all_reasons,
                'sentiment_reasons': sentiment_reasons,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced signals for {symbol}: {e}")
            return {
                'symbol': symbol,
                'base_signal_type': "ERROR",
                'enhanced_signal_type': "ERROR",
                'base_signal_strength': 0.0,
                'enhanced_signal_strength': 0.0,
                'base_confidence': 0.0,
                'enhanced_confidence': 0.0,
                'sentiment_score': 0.0,
                'sentiment_multiplier': 1.0,
                'alpha': self.alpha,
                'current_price': 0.0,
                'current_volume': 0.0,
                'reasons': [f"Error: {str(e)}"],
                'sentiment_reasons': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def set_alpha(self, alpha: float):
        """Update sentiment weight parameter"""
        self.alpha = alpha
        logger.info(f"Updated sentiment alpha to: {alpha}")
