"""
NLP Processing for Financial Text Analysis (Windows Compatible)

Features:
- TextBlob + VADER sentiment analysis
- Basic entity extraction
- Topic modeling and clustering
- Sentiment time series analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import requests
from dataclasses import dataclass
import json
import re
from loguru import logger

# NLP Libraries (Windows compatible)
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')


@dataclass
class NLPAnalysis:
    """Container for NLP analysis results"""
    # Sentiment analysis
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    sentiment_score: float  # -1 to 1
    vader_compound: float  # VADER compound score
    vader_positive: float  # VADER positive score
    vader_negative: float  # VADER negative score
    vader_neutral: float  # VADER neutral score
    
    # Text analysis
    word_count: int
    char_count: int
    sentence_count: int
    readability_score: float  # Simple readability metric
    
    # Entity recognition (basic)
    entities: List[str]  # Basic entities found
    crypto_mentions: List[str]  # Crypto-related terms
    stock_mentions: List[str]  # Stock-related terms
    
    # Text preprocessing
    cleaned_text: str
    lemmatized_text: str
    keywords: List[str]  # Top keywords


class NLPProcessor:
    """NLP processor using Windows-compatible libraries"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.sentiment_analyzer = None
        self.vectorizer = None
        self.lda_model = None
        self.is_initialized = False
        
        # Crypto and stock keywords for entity recognition
        self.crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'blockchain', 'defi', 'nft', 'altcoin', 'dogecoin', 'doge',
            'litecoin', 'ltc', 'ripple', 'xrp', 'cardano', 'ada',
            'polkadot', 'dot', 'chainlink', 'link', 'uniswap', 'uni',
            'solana', 'sol', 'avalanche', 'avax', 'polygon', 'matic',
            'binance', 'coinbase', 'kraken', 'crypto.com', 'metamask'
        ]
        
        self.stock_keywords = [
            'stock', 'stocks', 'equity', 'equities', 'nasdaq', 'nyse',
            'dow', 's&p', 'sp500', 'apple', 'aapl', 'microsoft', 'msft',
            'google', 'googl', 'amazon', 'amzn', 'tesla', 'tsla',
            'nvidia', 'nvda', 'meta', 'facebook', 'fb', 'netflix', 'nflx',
            'uber', 'lyft', 'airbnb', 'spotify', 'zoom', 'zm',
            'robinhood', 'webull', 'td ameritrade', 'charles schwab'
        ]
    
    async def initialize(self):
        """Initialize NLP models using Windows-compatible libraries"""
        try:
            logger.info("Initializing NLP processor...")
            
            # Initialize VADER sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Download NLTK data
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            self.is_initialized = True
            logger.info("NLP processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP processor: {e}")
            raise
    
    async def process_text(self, text: str) -> NLPAnalysis:
        """Process a single text using Windows-compatible NLP"""
        try:
            if not text or pd.isna(text):
                return self._create_empty_analysis()
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Basic text statistics
            word_count = len(cleaned_text.split())
            char_count = len(cleaned_text)
            sentence_count = len([s for s in cleaned_text.split('.') if s.strip()])
            
            # Sentiment analysis using TextBlob
            blob = TextBlob(cleaned_text)
            sentiment_score = blob.sentiment.polarity
            
            if sentiment_score > 0.1:
                sentiment_label = 'positive'
            elif sentiment_score < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            # VADER sentiment analysis
            vader_scores = self.sentiment_analyzer.polarity_scores(cleaned_text)
            
            # Entity recognition (basic keyword matching)
            entities = self._extract_entities(cleaned_text)
            crypto_mentions = self._extract_crypto_mentions(cleaned_text)
            stock_mentions = self._extract_stock_mentions(cleaned_text)
            
            # Text preprocessing
            lemmatized_text = self._lemmatize_text(cleaned_text)
            keywords = self._extract_keywords(lemmatized_text)
            
            # Simple readability score
            readability_score = self._calculate_readability(cleaned_text)
            
            return NLPAnalysis(
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                vader_compound=vader_scores['compound'],
                vader_positive=vader_scores['pos'],
                vader_negative=vader_scores['neg'],
                vader_neutral=vader_scores['neu'],
                word_count=word_count,
                char_count=char_count,
                sentence_count=sentence_count,
                readability_score=readability_score,
                entities=entities,
                crypto_mentions=crypto_mentions,
                stock_mentions=stock_mentions,
                cleaned_text=cleaned_text,
                lemmatized_text=lemmatized_text,
                keywords=keywords
            )
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return self._create_empty_analysis()
    
    async def process_news_batch(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of news articles"""
        logger.info(f"Processing {len(news_df)} news articles with Windows-compatible NLP")
        
        results = []
        
        for idx, row in news_df.iterrows():
            try:
                # Combine title and content for analysis
                text = f"{row.get('title', '')} {row.get('content', '')}"
                
                # Process text
                analysis = await self.process_text(text)
                
                # Add results to dataframe
                result_row = row.to_dict()
                result_row.update({
                    'sentiment_label': analysis.sentiment_label,
                    'sentiment_score': analysis.sentiment_score,
                    'vader_compound': analysis.vader_compound,
                    'vader_positive': analysis.vader_positive,
                    'vader_negative': analysis.vader_negative,
                    'vader_neutral': analysis.vader_neutral,
                    'word_count': analysis.word_count,
                    'char_count': analysis.char_count,
                    'sentence_count': analysis.sentence_count,
                    'readability_score': analysis.readability_score,
                    'entities': '|'.join(analysis.entities),
                    'crypto_mentions': '|'.join(analysis.crypto_mentions),
                    'stock_mentions': '|'.join(analysis.stock_mentions),
                    'cleaned_text': analysis.cleaned_text,
                    'keywords': '|'.join(analysis.keywords)
                })
                
                results.append(result_row)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                # Add empty analysis for failed rows
                result_row = row.to_dict()
                result_row.update({
                    'sentiment_label': 'neutral',
                    'sentiment_score': 0.0,
                    'vader_compound': 0.0,
                    'vader_positive': 0.0,
                    'vader_negative': 0.0,
                    'vader_neutral': 1.0,
                    'word_count': 0,
                    'char_count': 0,
                    'sentence_count': 0,
                    'readability_score': 0.0,
                    'entities': '',
                    'crypto_mentions': '',
                    'stock_mentions': '',
                    'cleaned_text': '',
                    'keywords': ''
                })
                results.append(result_row)
        
        return pd.DataFrame(results)
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
        
        return text.strip()
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract basic entities using keyword matching"""
        entities = []
        
        # Look for capitalized words (simple NER)
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_crypto_mentions(self, text: str) -> List[str]:
        """Extract crypto-related mentions"""
        mentions = []
        for keyword in self.crypto_keywords:
            if keyword in text:
                mentions.append(keyword)
        return mentions
    
    def _extract_stock_mentions(self, text: str) -> List[str]:
        """Extract stock-related mentions"""
        mentions = []
        for keyword in self.stock_keywords:
            if keyword in text:
                mentions.append(keyword)
        return mentions
    
    def _lemmatize_text(self, text: str) -> str:
        """Simple lemmatization using TextBlob"""
        try:
            blob = TextBlob(text)
            lemmatized = ' '.join([word.lemmatize() for word in blob.words])
            return lemmatized
        except:
            return text
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using simple frequency analysis"""
        try:
            blob = TextBlob(text)
            words = [word.lower() for word in blob.words if len(word) > 3]
            
            # Count word frequencies
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and return top keywords
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, freq in sorted_words[:10] if freq > 1]
            
            return keywords
        except:
            return []
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate simple readability score"""
        try:
            sentences = [s for s in text.split('.') if s.strip()]
            words = text.split()
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.0
            
            # Simple Flesch Reading Ease approximation
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = self._count_syllables(text) / len(words)
            
            readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            return max(0, min(100, readability))  # Clamp between 0 and 100
        except:
            return 50.0  # Default middle score
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simple approximation)"""
        vowels = 'aeiouy'
        count = 0
        for word in text.split():
            word = word.lower()
            if word:
                count += sum(1 for char in word if char in vowels)
        return count
    
    def _create_empty_analysis(self) -> NLPAnalysis:
        """Create empty analysis for error cases"""
        return NLPAnalysis(
            sentiment_label='neutral',
            sentiment_score=0.0,
            vader_compound=0.0,
            vader_positive=0.0,
            vader_negative=0.0,
            vader_neutral=1.0,
            word_count=0,
            char_count=0,
            sentence_count=0,
            readability_score=50.0,
            entities=[],
            crypto_mentions=[],
            stock_mentions=[],
            cleaned_text='',
            lemmatized_text='',
            keywords=[]
        )


# Convenience functions for backward compatibility
async def collect_crypto_news(api_key: str, hours_back: int = 24) -> List[Dict]:
    """Collect crypto news (placeholder)"""
    return []

async def collect_stock_news(api_key: str, hours_back: int = 24) -> List[Dict]:
    """Collect stock news (placeholder)"""
    return []