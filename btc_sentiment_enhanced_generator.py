"""
BTC Sentiment-Enhanced Signal Generator
Combines technical signals with news sentiment analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import glob


class BTCSentimentEnhancedGenerator:
    """BTC signal generator enhanced with sentiment analysis"""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha  # Sentiment multiplier weight
        self.symbol = "BTC"
        
    def load_news_data(self, symbol: str = "BTC") -> pd.DataFrame:
        """Load all available news data for the symbol"""
        try:
            news_files = glob.glob(f"data/news_{symbol}_*.parquet")
            if not news_files:
                logger.warning(f"No news files found for {symbol}")
                return pd.DataFrame()
            
            all_news = []
            for file in news_files:
                df = pd.read_parquet(file)
                all_news.append(df)
            
            if not all_news:
                return pd.DataFrame()
                
            combined_news = pd.concat(all_news, ignore_index=True)
            combined_news['timestamp'] = pd.to_datetime(combined_news['timestamp'], utc=True)
            combined_news = combined_news.sort_values('timestamp')
            
            logger.info(f"Loaded {len(combined_news)} news articles for {symbol}")
            return combined_news
            
        except Exception as e:
            logger.error(f"Error loading news data: {e}")
            return pd.DataFrame()
    
    def get_sentiment_for_timestamp(self, timestamp: datetime, news_df: pd.DataFrame, 
                                  window_minutes: int = 30) -> float:
        """Get average sentiment for headlines within ±window_minutes of timestamp"""
        if news_df.empty:
            return 0.0
            
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
        # Assuming sentiment scores are roughly in [-0.1, 0.1] range
        normalized_sentiment = np.clip(mean_sentiment * 10, -1, 1)
        
        return normalized_sentiment
    
    def generate_enhanced_signals(self, price_df: pd.DataFrame, 
                                indicators_df: Optional[pd.DataFrame] = None,
                                symbol: str = "BTC") -> Dict[str, Any]:
        """Generate BTC signals enhanced with sentiment analysis"""
        
        if price_df.empty:
            return self._create_hold_signal("No price data available")
        
        try:
            # Load news data
            news_df = self.load_news_data(symbol)
            
            # Get current timestamp
            current_timestamp = price_df.index[-1]
            
            # Get sentiment for current time
            sentiment_score = self.get_sentiment_for_timestamp(current_timestamp, news_df)
            
            # Generate base technical signal (reuse existing logic)
            from btc_signal_generator import BTCSignalGenerator
            base_generator = BTCSignalGenerator()
            base_signal = base_generator.generate_btc_signals(price_df, indicators_df)
            
            # Extract base signal score
            base_score = base_signal.get('signals', {}).get('signal_score', 0)
            
            # Apply sentiment enhancement
            # Multiply signal by (1 + α * sentiment)
            sentiment_multiplier = 1 + (self.alpha * sentiment_score)
            enhanced_score = base_score * sentiment_multiplier
            
            # Convert enhanced score back to signal
            if enhanced_score >= 4:
                action = "STRONG_BUY"
                confidence = min(0.95, 0.7 + enhanced_score * 0.05)
                strength = "strong"
            elif enhanced_score >= 2:
                action = "BUY"
                confidence = min(0.9, 0.5 + enhanced_score * 0.1)
                strength = "medium"
            elif enhanced_score <= -4:
                action = "STRONG_SELL"
                confidence = min(0.95, 0.7 + abs(enhanced_score) * 0.05)
                strength = "strong"
            elif enhanced_score <= -2:
                action = "SELL"
                confidence = min(0.9, 0.5 + abs(enhanced_score) * 0.1)
                strength = "medium"
            else:
                action = "HOLD"
                confidence = 0.4
                strength = "weak"
            
            # Update reasoning to include sentiment
            original_reasoning = base_signal.get('reasoning', '')
            sentiment_reasoning = f"Sentiment: {sentiment_score:.3f} (multiplier: {sentiment_multiplier:.3f})"
            enhanced_reasoning = f"{original_reasoning}; {sentiment_reasoning}"
            
            # Calculate enhanced price targets
            current_price = base_signal.get('current_price', 0)
            if action in ["BUY", "STRONG_BUY"]:
                price_target = current_price * (1 + abs(enhanced_score) * 0.02)
                stop_loss = current_price * (1 - 0.05)
            elif action in ["SELL", "STRONG_SELL"]:
                price_target = current_price * (1 - abs(enhanced_score) * 0.02)
                stop_loss = current_price * (1 + 0.05)
            else:
                price_target = current_price
                stop_loss = current_price * 0.95
            
            return {
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'strength': strength,
                'current_price': current_price,
                'price_change_24h': base_signal.get('price_change_24h', 0),
                'price_target': price_target,
                'stop_loss': stop_loss,
                'reasoning': enhanced_reasoning,
                'signals': {
                    'base_score': base_score,
                    'enhanced_score': enhanced_score,
                    'sentiment_score': sentiment_score,
                    'sentiment_multiplier': sentiment_multiplier,
                    'alpha': self.alpha
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating enhanced signals: {e}")
            return self._create_hold_signal(f"Error: {str(e)}")
    
    def _create_hold_signal(self, reason: str) -> Dict[str, Any]:
        """Create a default HOLD signal"""
        return {
            'symbol': self.symbol,
            'action': 'HOLD',
            'confidence': 0.3,
            'strength': 'weak',
            'current_price': 0,
            'price_change_24h': 0,
            'price_target': 0,
            'stop_loss': 0,
            'reasoning': reason,
            'signals': {},
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test the sentiment-enhanced generator
    generator = BTCSentimentEnhancedGenerator(alpha=0.5)
    
    # Create test data
    test_data = pd.DataFrame({
        'close': [50000, 51000, 52000, 53000, 54000],
        'volume': [1000, 2000, 1500, 3000, 2500]
    }, index=pd.date_range('2024-01-01', periods=5, freq='1H'))
    
    signals = generator.generate_enhanced_signals(test_data, symbol="BTC")
    print("Sentiment-Enhanced BTC Signal Test:")
    print(f"Action: {signals['action']}")
    print(f"Confidence: {signals['confidence']:.2f}")
    print(f"Reasoning: {signals['reasoning']}")
    print(f"Sentiment Score: {signals['signals'].get('sentiment_score', 0):.3f}")
    print(f"Sentiment Multiplier: {signals['signals'].get('sentiment_multiplier', 1):.3f}")
