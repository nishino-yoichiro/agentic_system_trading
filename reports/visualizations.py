"""
Data Visualization and Chart Generation

Creates charts and visualizations for reports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
import os


class ChartGenerator:
    """Generate various types of charts"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        self.style = style
        plt.style.use(style)
    
    def plot_price_chart(self, price_data: pd.DataFrame, symbol: str, save_path: str = None) -> str:
        """Create price chart with technical indicators"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
            
            # Price chart
            ax1.plot(price_data.index, price_data['close'], label='Close Price', linewidth=2)
            if 'sma_20' in price_data.columns:
                ax1.plot(price_data.index, price_data['sma_20'], label='SMA 20', alpha=0.7)
            if 'sma_50' in price_data.columns:
                ax1.plot(price_data.index, price_data['sma_50'], label='SMA 50', alpha=0.7)
            
            ax1.set_title(f'{symbol} Price Chart')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume chart
            if 'volume' in price_data.columns:
                ax2.bar(price_data.index, price_data['volume'], alpha=0.7, color='orange')
                ax2.set_ylabel('Volume')
                ax2.set_xlabel('Date')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Price chart saved to: {save_path}")
            
            return save_path or "price_chart.png"
            
        except Exception as e:
            logger.error(f"Error creating price chart: {e}")
            return None
    
    def plot_sentiment_chart(self, sentiment_data: pd.DataFrame, save_path: str = None) -> str:
        """Create sentiment analysis chart"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if 'sentiment_score' in sentiment_data.columns:
                ax.plot(sentiment_data.index, sentiment_data['sentiment_score'], 
                       label='Sentiment Score', linewidth=2)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.fill_between(sentiment_data.index, sentiment_data['sentiment_score'], 
                               alpha=0.3, where=(sentiment_data['sentiment_score'] >= 0), 
                               color='green', label='Positive')
                ax.fill_between(sentiment_data.index, sentiment_data['sentiment_score'], 
                               alpha=0.3, where=(sentiment_data['sentiment_score'] < 0), 
                               color='red', label='Negative')
            
            ax.set_title('Sentiment Analysis Over Time')
            ax.set_ylabel('Sentiment Score')
            ax.set_xlabel('Date')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Sentiment chart saved to: {save_path}")
            
            return save_path or "sentiment_chart.png"
            
        except Exception as e:
            logger.error(f"Error creating sentiment chart: {e}")
            return None
    
    def plot_correlation_heatmap(self, data: pd.DataFrame, save_path: str = None) -> str:
        """Create correlation heatmap"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate correlation matrix
            corr_matrix = data.corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax, fmt='.2f')
            
            ax.set_title('Correlation Heatmap')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Correlation heatmap saved to: {save_path}")
            
            return save_path or "correlation_heatmap.png"
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return None


class DashboardCreator:
    """Create comprehensive dashboards"""
    
    def __init__(self, chart_generator: ChartGenerator):
        self.chart_generator = chart_generator
    
    def create_market_dashboard(self, data: Dict[str, pd.DataFrame], save_path: str = None) -> str:
        """Create comprehensive market dashboard"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Price chart
            if 'price' in data and not data['price'].empty:
                symbol = data['price']['symbol'].iloc[0] if 'symbol' in data['price'].columns else 'Unknown'
                self.chart_generator.plot_price_chart(data['price'], symbol)
            
            # Sentiment chart
            if 'news' in data and not data['news'].empty:
                self.chart_generator.plot_sentiment_chart(data['news'])
            
            # Volume analysis
            if 'price' in data and 'volume' in data['price'].columns:
                axes[1, 0].hist(data['price']['volume'], bins=30, alpha=0.7)
                axes[1, 0].set_title('Volume Distribution')
                axes[1, 0].set_xlabel('Volume')
                axes[1, 0].set_ylabel('Frequency')
            
            # Technical indicators
            if 'price' in data and 'rsi' in data['price'].columns:
                axes[1, 1].plot(data['price'].index, data['price']['rsi'])
                axes[1, 1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
                axes[1, 1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
                axes[1, 1].set_title('RSI Indicator')
                axes[1, 1].set_ylabel('RSI')
                axes[1, 1].legend()
            
            plt.suptitle('Market Dashboard', fontsize=16)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Market dashboard saved to: {save_path}")
            
            return save_path or "market_dashboard.png"
            
        except Exception as e:
            logger.error(f"Error creating market dashboard: {e}")
            return None
