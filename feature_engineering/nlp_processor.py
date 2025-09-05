"""
Advanced NLP Processing for Financial Text Analysis

Features:
- Sentence transformers for semantic embeddings
- LLM-based sentiment analysis
- Entity extraction and NER
- Topic modeling and clustering
- Sentiment time series analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass
import json
import re
from loguru import logger

# NLP Libraries
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModel
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import torch

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    text: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    confidence: float
    emotions: Optional[Dict[str, float]] = None
    entities: Optional[List[Dict]] = None
    topics: Optional[List[str]] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class EntityMention:
    """Named entity mention"""
    text: str
    label: str  # 'PERSON', 'ORG', 'GPE', 'MONEY', 'PERCENT', etc.
    start: int
    end: int
    confidence: float


class SentimentAnalyzer:
    """Advanced sentiment analysis using multiple models"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.sentiment_pipeline = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self):
        """Initialize the sentiment analysis model"""
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info(f"Initialized sentiment analyzer with {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            # Fallback to VADER
            self.sentiment_pipeline = None
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of text using multiple methods"""
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # VADER sentiment
        vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
        vader_sentiment = vader_scores['compound']
        
        # Transformer-based sentiment
        transformer_sentiment = 0.0
        transformer_confidence = 0.0
        sentiment_label = "neutral"
        
        if self.sentiment_pipeline:
            try:
                result = self.sentiment_pipeline(cleaned_text)
                if result:
                    sentiment_label = result[0]['label'].lower()
                    transformer_confidence = result[0]['score']
                    
                    # Convert to -1 to 1 scale
                    if sentiment_label == 'positive':
                        transformer_sentiment = transformer_confidence
                    elif sentiment_label == 'negative':
                        transformer_sentiment = -transformer_confidence
                    else:
                        transformer_sentiment = 0.0
            except Exception as e:
                logger.error(f"Error in transformer sentiment analysis: {e}")
        
        # Combine results (weighted average)
        final_sentiment = 0.7 * transformer_sentiment + 0.3 * vader_sentiment
        final_confidence = max(transformer_confidence, abs(vader_sentiment))
        
        # Determine final label
        if final_sentiment > 0.1:
            sentiment_label = "positive"
        elif final_sentiment < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        return SentimentResult(
            text=text,
            sentiment_score=final_sentiment,
            sentiment_label=sentiment_label,
            confidence=final_confidence
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()


class EntityExtractor:
    """Named Entity Recognition for financial text"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        
    async def initialize(self):
        """Initialize the NER model"""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Initialized NER model: {self.model_name}")
        except OSError:
            logger.warning(f"Model {self.model_name} not found. Install with: python -m spacy download {self.model_name}")
            # Try to load a basic model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error("No spaCy model available. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
    
    def extract_entities(self, text: str) -> List[EntityMention]:
        """Extract named entities from text"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Filter for relevant financial entities
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'MONEY', 'PERCENT', 'CARDINAL']:
                entities.append(EntityMention(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0  # spaCy doesn't provide confidence scores
                ))
        
        return entities
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract financial-specific entities"""
        entities = self.extract_entities(text)
        
        financial_entities = {
            'companies': [],
            'people': [],
            'locations': [],
            'currencies': [],
            'percentages': [],
            'numbers': []
        }
        
        for entity in entities:
            if entity.label_ == 'ORG':
                financial_entities['companies'].append(entity.text)
            elif entity.label_ == 'PERSON':
                financial_entities['people'].append(entity.text)
            elif entity.label_ == 'GPE':
                financial_entities['locations'].append(entity.text)
            elif entity.label_ == 'MONEY':
                financial_entities['currencies'].append(entity.text)
            elif entity.label_ == 'PERCENT':
                financial_entities['percentages'].append(entity.text)
            elif entity.label_ == 'CARDINAL':
                financial_entities['numbers'].append(entity.text)
        
        return financial_entities


class NLPProcessor:
    """Main NLP processing class that combines all components"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
                 ner_model: str = "en_core_web_sm"):
        self.embedding_model_name = embedding_model
        self.sentiment_model_name = sentiment_model
        self.ner_model_name = ner_model
        
        self.embedding_model = None
        self.sentiment_analyzer = SentimentAnalyzer(sentiment_model)
        self.entity_extractor = EntityExtractor(ner_model)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    async def initialize(self):
        """Initialize all NLP components"""
        logger.info("Initializing NLP processor...")
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
        
        # Initialize other components
        await self.sentiment_analyzer.initialize()
        await self.entity_extractor.initialize()
        
        logger.info("NLP processor initialized successfully")
    
    async def process_text(self, text: str) -> SentimentResult:
        """Process a single text with all NLP components"""
        # Sentiment analysis
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(text)
        
        # Entity extraction
        entities = self.entity_extractor.extract_entities(text)
        sentiment_result.entities = [
            {
                'text': entity.text,
                'label': entity.label,
                'confidence': entity.confidence
            }
            for entity in entities
        ]
        
        # Generate embedding
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text)
                sentiment_result.embedding = embedding
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
        
        return sentiment_result
    
    async def process_articles(self, articles: List[Dict]) -> List[SentimentResult]:
        """Process a batch of articles"""
        results = []
        
        for article in articles:
            try:
                # Combine title and content
                text = f"{article.get('title', '')} {article.get('content', '')}"
                
                result = await self.process_text(text)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing article: {e}")
                continue
        
        logger.info(f"Processed {len(results)} articles")
        return results
    
    def extract_topics(self, texts: List[str], n_topics: int = 10) -> List[List[str]]:
        """Extract topics using LDA"""
        try:
            # Prepare texts
            cleaned_texts = [self.sentiment_analyzer._clean_text(text) for text in texts]
            
            # TF-IDF vectorization
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_texts)
            
            # LDA topic modeling
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(tfidf_matrix)
            
            # Get top words for each topic
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append(top_words)
            
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    def cluster_articles(self, articles: List[Dict], n_clusters: int = 5) -> List[int]:
        """Cluster articles based on content similarity"""
        try:
            # Extract embeddings
            embeddings = []
            for article in articles:
                text = f"{article.get('title', '')} {article.get('content', '')}"
                if self.embedding_model:
                    embedding = self.embedding_model.encode(text)
                    embeddings.append(embedding)
            
            if not embeddings:
                return [0] * len(articles)
            
            # K-means clustering
            embeddings_array = np.array(embeddings)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            return cluster_labels.tolist()
            
        except Exception as e:
            logger.error(f"Error clustering articles: {e}")
            return [0] * len(articles)
    
    def calculate_sentiment_metrics(self, results: List[SentimentResult]) -> Dict[str, float]:
        """Calculate aggregate sentiment metrics"""
        if not results:
            return {}
        
        scores = [r.sentiment_score for r in results]
        confidences = [r.confidence for r in results]
        
        # Weighted average by confidence
        weighted_scores = [s * c for s, c in zip(scores, confidences)]
        total_confidence = sum(confidences)
        
        if total_confidence > 0:
            avg_sentiment = sum(weighted_scores) / total_confidence
        else:
            avg_sentiment = np.mean(scores)
        
        return {
            'average_sentiment': avg_sentiment,
            'sentiment_std': np.std(scores),
            'positive_ratio': sum(1 for s in scores if s > 0.1) / len(scores),
            'negative_ratio': sum(1 for s in scores if s < -0.1) / len(scores),
            'neutral_ratio': sum(1 for s in scores if -0.1 <= s <= 0.1) / len(scores),
            'average_confidence': np.mean(confidences),
            'total_articles': len(results)
        }


class SentimentTimeSeries:
    """Sentiment time series analysis"""
    
    def __init__(self, nlp_processor: NLPProcessor):
        self.nlp_processor = nlp_processor
        
    async def create_sentiment_series(self, 
                                    articles: List[Dict], 
                                    time_window: str = '1H') -> pd.DataFrame:
        """Create sentiment time series from articles"""
        # Process articles
        results = await self.nlp_processor.process_articles(articles)
        
        # Create DataFrame
        data = []
        for i, result in enumerate(results):
            data.append({
                'timestamp': articles[i].get('published_at', datetime.now()),
                'sentiment_score': result.sentiment_score,
                'confidence': result.confidence,
                'sentiment_label': result.sentiment_label
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to time window
        sentiment_series = df['sentiment_score'].resample(time_window).mean()
        confidence_series = df['confidence'].resample(time_window).mean()
        
        result_df = pd.DataFrame({
            'sentiment': sentiment_series,
            'confidence': confidence_series,
            'count': df['sentiment_score'].resample(time_window).count()
        })
        
        return result_df.fillna(0)
    
    def detect_sentiment_shifts(self, 
                              sentiment_series: pd.Series, 
                              threshold: float = 0.3) -> List[Dict]:
        """Detect significant sentiment shifts"""
        shifts = []
        
        # Calculate rolling statistics
        rolling_mean = sentiment_series.rolling(window=24).mean()  # 24-hour window
        rolling_std = sentiment_series.rolling(window=24).std()
        
        # Detect shifts
        for i in range(1, len(sentiment_series)):
            current_sentiment = sentiment_series.iloc[i]
            previous_mean = rolling_mean.iloc[i-1]
            previous_std = rolling_std.iloc[i-1]
            
            if previous_std > 0:
                z_score = (current_sentiment - previous_mean) / previous_std
                
                if abs(z_score) > threshold:
                    shifts.append({
                        'timestamp': sentiment_series.index[i],
                        'sentiment': current_sentiment,
                        'z_score': z_score,
                        'magnitude': 'high' if abs(z_score) > 2 else 'medium'
                    })
        
        return shifts


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        nlp = NLPProcessor()
        await nlp.initialize()
        
        # Test with sample text
        sample_text = "Bitcoin is surging to new all-time highs! The cryptocurrency market is showing incredible strength."
        result = await nlp.process_text(sample_text)
        
        print(f"Sentiment: {result.sentiment_label} ({result.sentiment_score:.3f})")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Entities: {[e['text'] for e in result.entities]}")
    
    asyncio.run(main())
