"""
Data Validation and Quality Assurance

Ensures data integrity and quality across all data sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger
import re


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float  # 0-1
    suggestions: List[str]


class DataValidator:
    """Validates data quality and integrity"""
    
    def __init__(self):
        self.required_news_fields = ['title', 'content', 'url', 'source', 'published_at']
        self.required_price_fields = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.required_social_fields = ['id', 'content', 'author', 'platform', 'created_at']
    
    def validate_news_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate news data quality"""
        errors = []
        warnings = []
        suggestions = []
        
        if df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings, 0.0, suggestions)
        
        # Check required fields
        missing_fields = [field for field in self.required_news_fields if field not in df.columns]
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
        
        # Check for null values in critical fields
        critical_fields = ['title', 'content', 'published_at']
        for field in critical_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    warnings.append(f"Field '{field}' has {null_count} null values")
        
        # Check data types
        if 'published_at' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['published_at']):
                errors.append("Field 'published_at' is not datetime type")
        
        # Check for duplicate articles
        if 'url' in df.columns:
            duplicate_count = df['url'].duplicated().sum()
            if duplicate_count > 0:
                warnings.append(f"Found {duplicate_count} duplicate articles")
                suggestions.append("Consider removing duplicate articles")
        
        # Check content quality
        if 'content' in df.columns:
            short_content = df['content'].str.len() < 50
            if short_content.sum() > 0:
                warnings.append(f"Found {short_content.sum()} articles with very short content")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(len(errors), len(warnings), len(df))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            suggestions=suggestions
        )
    
    def validate_price_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate price data quality"""
        errors = []
        warnings = []
        suggestions = []
        
        if df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings, 0.0, suggestions)
        
        # Check required fields
        missing_fields = [field for field in self.required_price_fields if field not in df.columns]
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
        
        # Check for null values in OHLCV
        ohlcv_fields = ['open', 'high', 'low', 'close', 'volume']
        for field in ohlcv_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    errors.append(f"Field '{field}' has {null_count} null values")
        
        # Check for negative prices
        price_fields = ['open', 'high', 'low', 'close']
        for field in price_fields:
            if field in df.columns:
                negative_count = (df[field] < 0).sum()
                if negative_count > 0:
                    errors.append(f"Field '{field}' has {negative_count} negative values")
        
        # Check for negative volume
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                errors.append(f"Field 'volume' has {negative_volume} negative values")
        
        # Check OHLC relationships
        if all(field in df.columns for field in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            )
            if invalid_ohlc.sum() > 0:
                errors.append(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicate_timestamps = df['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                warnings.append(f"Found {duplicate_timestamps} duplicate timestamps")
                suggestions.append("Consider removing duplicate timestamps")
        
        # Check data continuity
        if 'timestamp' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff().dropna()
            if not time_diffs.empty:
                median_diff = time_diffs.median()
                large_gaps = time_diffs > median_diff * 3
                if large_gaps.sum() > 0:
                    warnings.append(f"Found {large_gaps.sum()} large time gaps in data")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(len(errors), len(warnings), len(df))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            suggestions=suggestions
        )
    
    def validate_social_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate social media data quality"""
        errors = []
        warnings = []
        suggestions = []
        
        if df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings, 0.0, suggestions)
        
        # Check required fields
        missing_fields = [field for field in self.required_social_fields if field not in df.columns]
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
        
        # Check for null values in critical fields
        critical_fields = ['content', 'created_at']
        for field in critical_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    warnings.append(f"Field '{field}' has {null_count} null values")
        
        # Check data types
        if 'created_at' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
                errors.append("Field 'created_at' is not datetime type")
        
        # Check for duplicate posts
        if 'id' in df.columns:
            duplicate_count = df['id'].duplicated().sum()
            if duplicate_count > 0:
                warnings.append(f"Found {duplicate_count} duplicate posts")
                suggestions.append("Consider removing duplicate posts")
        
        # Check content quality
        if 'content' in df.columns:
            short_content = df['content'].str.len() < 10
            if short_content.sum() > 0:
                warnings.append(f"Found {short_content.sum()} posts with very short content")
        
        # Check for spam patterns
        if 'content' in df.columns:
            spam_patterns = [
                r'https?://\S+',  # URLs
                r'@\w+',  # Mentions
                r'#\w+',  # Hashtags
            ]
            for pattern in spam_patterns:
                matches = df['content'].str.contains(pattern, regex=True, na=False)
                if matches.sum() > len(df) * 0.8:  # More than 80% have pattern
                    warnings.append(f"High frequency of pattern '{pattern}' - possible spam")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(len(errors), len(warnings), len(df))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            suggestions=suggestions
        )
    
    def _calculate_quality_score(self, error_count: int, warning_count: int, data_count: int) -> float:
        """Calculate overall data quality score (0-1)"""
        if data_count == 0:
            return 0.0
        
        # Base score
        score = 1.0
        
        # Deduct for errors (more severe)
        score -= error_count * 0.2
        
        # Deduct for warnings (less severe)
        score -= warning_count * 0.05
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def clean_data(self, df: pd.DataFrame, data_type: str = 'news') -> pd.DataFrame:
        """Clean data based on validation results"""
        if df.empty:
            return df
        
        # Remove duplicates
        if 'url' in df.columns:
            df = df.drop_duplicates(subset=['url'])
        elif 'id' in df.columns:
            df = df.drop_duplicates(subset=['id'])
        
        # Remove rows with null critical fields
        if data_type == 'news':
            critical_fields = ['title', 'content', 'published_at']
        elif data_type == 'price':
            critical_fields = ['open', 'high', 'low', 'close', 'volume']
        elif data_type == 'social':
            critical_fields = ['content', 'created_at']
        else:
            critical_fields = []
        
        for field in critical_fields:
            if field in df.columns:
                df = df.dropna(subset=[field])
        
        # Remove rows with invalid data
        if data_type == 'price':
            # Remove rows with negative prices
            price_fields = ['open', 'high', 'low', 'close']
            for field in price_fields:
                if field in df.columns:
                    df = df[df[field] >= 0]
            
            # Remove rows with negative volume
            if 'volume' in df.columns:
                df = df[df['volume'] >= 0]
        
        return df

