"""
Enhanced Crypto Trading Pipeline - Feature Engineering Module

This module handles advanced feature engineering for trading signals:
- NLP processing with sentence transformers and LLM embeddings
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Alpha factor generation from combined NLP + price data
- Feature store for vectorized data storage
- Sentiment analysis and entity extraction
"""

from .nlp_processor import NLPProcessor
from .technical_indicators import TechnicalIndicators, IndicatorCalculator
from .alpha_factors import AlphaFactorGenerator, FactorAnalyzer
from .feature_store import FeatureStore, FeatureManager

__all__ = [
    'NLPProcessor',
    'TechnicalIndicators',
    'IndicatorCalculator',
    'AlphaFactorGenerator',
    'FactorAnalyzer',
    'FeatureStore',
    'FeatureManager'
]
