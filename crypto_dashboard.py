#!/usr/bin/env python3
"""
Unified Crypto Dashboard
Multi-symbol dashboard with sentiment analysis and news visualization
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import webbrowser
from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import numpy as np
from typing import List
from crypto_analysis_engine import CryptoAnalysisEngine
from crypto_signal_generator import CryptoSentimentGenerator
import json
import os
from loguru import logger

class CryptoDashboard:
    """Unified multi-symbol crypto dashboard with sentiment analysis"""
    
    def __init__(self, symbols: List[str] = None, alpha: float = 0.5):
        self.app = Flask(__name__)
        self.symbols = symbols or ['BTC', 'ETH', 'ADA', 'SOL']
        self.analysis_engine = CryptoAnalysisEngine()
        self.sentiment_generator = CryptoSentimentGenerator(alpha=alpha)
        self.latest_data = {}
        self.last_update = None
        self.update_interval = 300  # 5 minutes
        self.running = False
        self.current_alpha = alpha
        
        # Setup routes
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            return self.render_dashboard()
        
        @self.app.route('/api/status')
        def api_status():
            return jsonify({
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'data_available': len(self.latest_data) > 0,
                'symbols': self.symbols,
                'next_update_in': self.get_next_update_time()
            })
        
        @self.app.route('/api/refresh')
        def api_refresh():
            self.update_analysis()
            return jsonify({'status': 'success', 'message': 'Analysis updated'})
        
        @self.app.route('/api/sentiment', methods=['POST'])
        def api_sentiment():
            data = request.get_json()
            new_alpha = data.get('alpha', 0.5)
            self.sentiment_generator.set_alpha(new_alpha)
            self.current_alpha = new_alpha
            self.update_analysis()
            logger.info(f"Updated sentiment alpha to: {new_alpha}")
            return jsonify({'status': 'success', 'alpha': new_alpha})
        
        @self.app.route('/api/news')
        def api_news():
            """Get news articles with filtering options"""
            try:
                if not self.latest_data:
                    return jsonify({'error': 'No data available'})
                
                # Get query parameters
                limit = request.args.get('limit', 50, type=int)
                offset = request.args.get('offset', 0, type=int)
                sentiment_filter = request.args.get('sentiment', None)
                time_window = request.args.get('time_window', 24, type=int)
                search_query = request.args.get('search', None)
                symbol_filter = request.args.get('symbol', None)
                
                # Get news data from all symbols
                all_articles = []
                for symbol in self.symbols:
                    symbol_data = self.latest_data.get(symbol, {})
                    news_data = symbol_data.get('news_data', {})
                    articles = news_data.get('articles', [])
                    
                    # Add symbol to each article
                    for article in articles:
                        article['symbol'] = symbol
                        all_articles.append(article)
                
                if not all_articles:
                    return jsonify({
                        'articles': [],
                        'total': 0,
                        'offset': 0,
                        'limit': limit,
                        'filters': {
                            'sentiment': sentiment_filter,
                            'time_window': time_window,
                            'search': search_query,
                            'symbol': symbol_filter
                        },
                        'message': 'No articles available'
                    })
            
                # Apply filters
                filtered_articles = all_articles
                
                # Symbol filter
                if symbol_filter:
                    filtered_articles = [a for a in filtered_articles if a.get('symbol') == symbol_filter]
                
                # Time window filter
                if time_window > 0:
                    cutoff_time = datetime.now() - timedelta(hours=time_window)
                    cutoff_time = pd.Timestamp(cutoff_time, tz='UTC')
                    filtered_articles = [
                        article for article in filtered_articles
                        if pd.to_datetime(article.get('timestamp', '')).tz_convert('UTC') >= cutoff_time
                    ]
                
                # Sentiment filter
                if sentiment_filter:
                    if sentiment_filter == 'positive':
                        filtered_articles = [a for a in filtered_articles if a.get('sentiment_score', 0) > 0.01]
                    elif sentiment_filter == 'negative':
                        filtered_articles = [a for a in filtered_articles if a.get('sentiment_score', 0) < -0.01]
                    elif sentiment_filter == 'neutral':
                        filtered_articles = [a for a in filtered_articles if abs(a.get('sentiment_score', 0)) <= 0.01]
                
                # Search filter
                if search_query:
                    search_lower = search_query.lower()
                    filtered_articles = [
                        article for article in filtered_articles
                        if search_lower in article.get('headline', '').lower() or 
                           search_lower in article.get('content', '').lower()
                    ]
                
                # Sort by timestamp (newest first)
                filtered_articles.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                
                # Apply pagination
                total_articles = len(filtered_articles)
                paginated_articles = filtered_articles[offset:offset + limit]
                
                return jsonify({
                    'articles': paginated_articles,
                    'total': total_articles,
                    'offset': offset,
                    'limit': limit,
                    'filters': {
                        'sentiment': sentiment_filter,
                        'time_window': time_window,
                        'search': search_query,
                        'symbol': symbol_filter
                    }
                })
                
            except Exception as e:
                logger.error(f"Error in api_news: {e}")
                return jsonify({'error': f'Failed to load news: {str(e)}'})
        
        @self.app.route('/api/news/sentiment-breakdown')
        def api_sentiment_breakdown():
            """Get sentiment breakdown statistics"""
            try:
                if not self.latest_data:
                    return jsonify({'error': 'No data available'})
                
                # Get news data from all symbols
                all_articles = []
                for symbol in self.symbols:
                    symbol_data = self.latest_data.get(symbol, {})
                    news_data = symbol_data.get('news_data', {})
                    articles = news_data.get('articles', [])
                    all_articles.extend(articles)
                
                if not all_articles:
                    return jsonify({
                        'total': 0,
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0
                    })
                
                # Calculate sentiment breakdown
                positive = sum(1 for a in all_articles if a.get('sentiment_score', 0) > 0.01)
                negative = sum(1 for a in all_articles if a.get('sentiment_score', 0) < -0.01)
                neutral = sum(1 for a in all_articles if abs(a.get('sentiment_score', 0)) <= 0.01)
                
                return jsonify({
                    'total': len(all_articles),
                    'positive': positive,
                    'negative': negative,
                    'neutral': neutral
                })
                
            except Exception as e:
                logger.error(f"Error in api_sentiment_breakdown: {e}")
                return jsonify({'error': f'Failed to load sentiment breakdown: {str(e)}'})
        
        @self.app.route('/api/symbol/<symbol>')
        def api_symbol(symbol):
            """Get analysis data for a specific symbol"""
            try:
                symbol = symbol.upper()
                if symbol not in self.symbols:
                    return jsonify({'error': f'Symbol {symbol} not found'}), 404
                
                symbol_data = self.latest_data.get(symbol, {})
                if not symbol_data:
                    return jsonify({'error': f'No data available for {symbol}'}), 404
                
                return jsonify({
                    'symbol': symbol,
                    'data': symbol_data,
                    'last_update': self.last_update.isoformat() if self.last_update else None
                })
                
            except Exception as e:
                logger.error(f"Error getting symbol data for {symbol}: {e}")
                return jsonify({'error': str(e)}), 500
    
    def update_analysis(self):
        """Update analysis for all symbols"""
        try:
            logger.info(f"Updating analysis for symbols: {self.symbols}")
            
            for symbol in self.symbols:
                try:
                    # Load price data
                    price_df = self.analysis_engine.load_symbol_data(symbol, days=7)
                    
                    # Calculate technical indicators
                    indicators_df = self.analysis_engine.calculate_technical_indicators(price_df)
                    
                    # Generate enhanced signals
                    signals = self.sentiment_generator.generate_enhanced_signals(
                        price_df, symbol=symbol
                    )
                    
                    # Calculate support/resistance
                    support_resistance = self.analysis_engine.get_support_resistance(price_df, symbol)
                    
                    # Calculate risk metrics
                    risk_metrics = self.analysis_engine.calculate_risk_metrics(price_df, symbol)
                    
                    # Load news data
                    news_df = self.sentiment_generator.load_news_data(symbol)
                    articles = []
                    if not news_df.empty:
                        articles = news_df.to_dict('records')
                        for article in articles:
                            if 'timestamp' in article:
                                article['timestamp'] = str(article['timestamp'])
                    
                    # Store symbol data
                    self.latest_data[symbol] = {
                        'signals': signals,
                        'support_resistance': support_resistance,
                        'risk_metrics': risk_metrics,
                        'price_data': {
                            'current_price': price_df['close'].iloc[-1],
                            'price_change': price_df['close'].pct_change().iloc[-1] * 100,
                            'volume': price_df['volume'].iloc[-1],
                            'high_24h': price_df['high'].tail(24).max(),
                            'low_24h': price_df['low'].tail(24).min()
                        },
                        'news_data': {
                            'articles': articles,
                            'total_articles': len(articles),
                            'last_updated': datetime.now().isoformat()
                        }
                    }
                    
                    logger.info(f"Updated analysis for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Failed to update analysis for {symbol}: {e}")
                    self.latest_data[symbol] = {
                        'error': str(e),
                        'signals': None,
                        'support_resistance': {'support': 0.0, 'resistance': 0.0},
                        'risk_metrics': {'volatility': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0},
                        'price_data': {'current_price': 0.0, 'price_change': 0.0, 'volume': 0.0, 'high_24h': 0.0, 'low_24h': 0.0},
                        'news_data': {'articles': [], 'total_articles': 0, 'last_updated': datetime.now().isoformat()}
                    }
            
            self.last_update = datetime.now()
            logger.info("Analysis update completed")
            
        except Exception as e:
            logger.error(f"Failed to update analysis: {e}")
    
    def get_next_update_time(self) -> int:
        """Get seconds until next update"""
        if not self.last_update:
            return 0
        elapsed = (datetime.now() - self.last_update).total_seconds()
        return max(0, self.update_interval - elapsed)
    
    def render_dashboard(self) -> str:
        """Render the main dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Dashboard - Multi-Symbol Analysis</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .header p {
            color: #7f8c8d;
            text-align: center;
            font-size: 1.1em;
        }
        
        .controls {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .control-group {
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .control-group h3 {
            margin-bottom: 15px;
            color: #2c3e50;
        }
        
        .sentiment-control {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .sentiment-control input[type="range"] {
            width: 200px;
        }
        
        .sentiment-control span {
            font-weight: bold;
            color: #e74c3c;
            min-width: 60px;
        }
        
        .refresh-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }
        
        .refresh-btn:hover {
            background: #2980b9;
        }
        
        .symbols-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }
        
        .symbol-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .symbol-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        }
        
        .symbol-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .symbol-name {
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .signal-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .signal-buy {
            background: #27ae60;
            color: white;
        }
        
        .signal-sell {
            background: #e74c3c;
            color: white;
        }
        
        .signal-hold {
            background: #f39c12;
            color: white;
        }
        
        .price-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .price-item {
            text-align: center;
        }
        
        .price-item .label {
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        
        .price-item .value {
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .price-positive {
            color: #27ae60;
        }
        
        .price-negative {
            color: #e74c3c;
        }
        
        .sentiment-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        
        .sentiment-info h4 {
            margin-bottom: 10px;
            color: #2c3e50;
        }
        
        .sentiment-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .sentiment-metric {
            text-align: center;
        }
        
        .sentiment-metric .label {
            font-size: 0.8em;
            color: #7f8c8d;
        }
        
        .sentiment-metric .value {
            font-weight: bold;
            color: #2c3e50;
        }
        
        .reasons {
            margin-top: 15px;
        }
        
        .reasons h4 {
            margin-bottom: 10px;
            color: #2c3e50;
        }
        
        .reason-list {
            list-style: none;
        }
        
        .reason-list li {
            background: #ecf0f1;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        .news-section {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            margin-top: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .news-controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .news-filter-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        
        .news-filter-group label {
            font-weight: bold;
            color: #2c3e50;
            font-size: 0.9em;
        }
        
        .news-filter-group input,
        .news-filter-group select {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        
        .news-stats {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .news-stat-item {
            background: #f8f9fa;
            padding: 10px 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .news-stat-item .label {
            font-size: 0.8em;
            color: #7f8c8d;
        }
        
        .news-stat-item .value {
            font-weight: bold;
            color: #2c3e50;
            font-size: 1.1em;
        }
        
        .news-articles {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
        }
        
        .news-article {
            border-bottom: 1px solid #eee;
            padding: 15px 0;
        }
        
        .news-article:last-child {
            border-bottom: none;
        }
        
        .article-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 10px;
        }
        
        .article-title {
            font-weight: bold;
            color: #2c3e50;
            flex: 1;
            margin-right: 15px;
        }
        
        .article-sentiment {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .article-sentiment.positive {
            background: #d5f4e6;
            color: #27ae60;
        }
        
        .article-sentiment.negative {
            background: #fadbd8;
            color: #e74c3c;
        }
        
        .article-sentiment.neutral {
            background: #f8f9fa;
            color: #7f8c8d;
        }
        
        .article-meta {
            font-size: 0.8em;
            color: #7f8c8d;
            margin-bottom: 8px;
        }
        
        .article-content {
            font-size: 0.9em;
            color: #555;
            line-height: 1.4;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
        }
        
        .status-bar {
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .status-item {
            display: inline-block;
            margin: 0 20px;
            color: #7f8c8d;
        }
        
        .status-item strong {
            color: #2c3e50;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .symbols-grid {
                grid-template-columns: 1fr;
            }
            
            .price-info {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Crypto Dashboard</h1>
            <p>Multi-Symbol Analysis with Sentiment Integration</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <h3>Sentiment Control</h3>
                <div class="sentiment-control">
                    <label>Alpha (α):</label>
                    <input type="range" id="alphaSlider" min="0" max="2" step="0.1" value="0.5">
                    <span id="alphaValue">0.5</span>
                </div>
            </div>
            
            <div class="control-group">
                <h3>Actions</h3>
                <button class="refresh-btn" onclick="refreshAnalysis()">🔄 Refresh Analysis</button>
            </div>
        </div>
        
        <div class="status-bar">
            <div class="status-item">
                <strong>Last Update:</strong> <span id="lastUpdate">Loading...</span>
            </div>
            <div class="status-item">
                <strong>Next Update:</strong> <span id="nextUpdate">--</span>
            </div>
            <div class="status-item">
                <strong>Symbols:</strong> <span id="symbolsList">Loading...</span>
            </div>
        </div>
        
        <div id="symbolsContainer" class="symbols-grid">
            <div class="loading">Loading analysis...</div>
        </div>
        
        <div class="news-section">
            <h2>📰 News Articles & Sentiment Analysis</h2>
            
            <div class="news-controls">
                <div class="news-filter-group">
                    <label>Search:</label>
                    <input type="text" id="searchInput" placeholder="Search articles...">
                </div>
                
                <div class="news-filter-group">
                    <label>Symbol:</label>
                    <select id="symbolFilter">
                        <option value="">All Symbols</option>
                    </select>
                </div>
                
                <div class="news-filter-group">
                    <label>Sentiment:</label>
                    <select id="sentimentFilter">
                        <option value="">All</option>
                        <option value="positive">Positive</option>
                        <option value="negative">Negative</option>
                        <option value="neutral">Neutral</option>
                    </select>
                </div>
                
                <div class="news-filter-group">
                    <label>Time Window:</label>
                    <select id="timeWindowFilter">
                        <option value="24">Last 24h</option>
                        <option value="72">Last 3 days</option>
                        <option value="168">Last week</option>
                        <option value="0">All time</option>
                    </select>
                </div>
                
                <div class="news-filter-group">
                    <button class="refresh-btn" onclick="loadNewsArticles()">🔄 Refresh News</button>
                </div>
            </div>
            
            <div class="news-stats" id="newsStats">
                <div class="news-stat-item">
                    <div class="label">Total Articles</div>
                    <div class="value" id="totalArticles">0</div>
                </div>
                <div class="news-stat-item">
                    <div class="label">Positive</div>
                    <div class="value" id="positiveArticles">0</div>
                </div>
                <div class="news-stat-item">
                    <div class="label">Negative</div>
                    <div class="value" id="negativeArticles">0</div>
                </div>
                <div class="news-stat-item">
                    <div class="label">Neutral</div>
                    <div class="value" id="neutralArticles">0</div>
                </div>
            </div>
            
            <div id="newsArticles" class="news-articles">
                <div class="loading">Loading news articles...</div>
            </div>
        </div>
    </div>
    
    <script>
        let currentPage = 0;
        const pageSize = 20;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            loadStatus();
            loadSymbols();
            loadNewsStats();
            loadNewsArticles();
            
            // Setup auto-refresh
            setInterval(loadStatus, 30000); // Every 30 seconds
            setInterval(loadSymbols, 300000); // Every 5 minutes
            
            // Setup sentiment control
            const alphaSlider = document.getElementById('alphaSlider');
            alphaSlider.addEventListener('input', function() {
                document.getElementById('alphaValue').textContent = this.value;
            });
            
            alphaSlider.addEventListener('change', function() {
                updateSentiment(parseFloat(this.value));
            });
            
            // Setup news filters
            document.getElementById('searchInput').addEventListener('input', debounce(applyFilters, 300));
            document.getElementById('symbolFilter').addEventListener('change', applyFilters);
            document.getElementById('sentimentFilter').addEventListener('change', applyFilters);
            document.getElementById('timeWindowFilter').addEventListener('change', applyFilters);
        });
        
        function loadStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('lastUpdate').textContent = 
                        data.last_update ? new Date(data.last_update).toLocaleString() : 'Never';
                    document.getElementById('nextUpdate').textContent = 
                        data.next_update_in ? `${Math.floor(data.next_update_in / 60)}m ${data.next_update_in % 60}s` : '--';
                    document.getElementById('symbolsList').textContent = data.symbols.join(', ');
                })
                .catch(error => console.error('Error loading status:', error));
        }
        
        function loadSymbols() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const symbols = data.symbols || [];
                    const container = document.getElementById('symbolsContainer');
                    
                    if (symbols.length === 0) {
                        container.innerHTML = '<div class="loading">No symbols available</div>';
                        return;
                    }
                    
                    // Load data for each symbol
                    Promise.all(symbols.map(symbol => 
                        fetch(`/api/symbol/${symbol}`)
                            .then(response => response.json())
                            .catch(error => ({ symbol, error: error.message }))
                    )).then(symbolData => {
                        container.innerHTML = symbolData.map(data => renderSymbolCard(data)).join('');
                    });
                })
                .catch(error => {
                    console.error('Error loading symbols:', error);
                    document.getElementById('symbolsContainer').innerHTML = 
                        '<div class="loading">Error loading symbols</div>';
                });
        }
        
        function renderSymbolCard(data) {
            if (data.error) {
                return `
                    <div class="symbol-card">
                        <div class="symbol-header">
                            <div class="symbol-name">${data.symbol}</div>
                            <div class="signal-badge signal-hold">ERROR</div>
                        </div>
                        <div class="loading">Error: ${data.error}</div>
                    </div>
                `;
            }
            
            const signals = data.data?.signals || {};
            const priceData = data.data?.price_data || {};
            const supportResistance = data.data?.support_resistance || {};
            const riskMetrics = data.data?.risk_metrics || {};
            
            const signalType = signals.enhanced_signal_type || signals.base_signal_type || 'HOLD';
            const signalClass = signalType.toLowerCase();
            
            const priceChange = priceData.price_change || 0;
            const priceChangeClass = priceChange >= 0 ? 'price-positive' : 'price-negative';
            const priceChangeSymbol = priceChange >= 0 ? '+' : '';
            
            return `
                <div class="symbol-card">
                    <div class="symbol-header">
                        <div class="symbol-name">${data.symbol}</div>
                        <div class="signal-badge signal-${signalClass}">${signalType}</div>
                    </div>
                    
                    <div class="price-info">
                        <div class="price-item">
                            <div class="label">Current Price</div>
                            <div class="value">$${(priceData.current_price || 0).toFixed(2)}</div>
                        </div>
                        <div class="price-item">
                            <div class="label">24h Change</div>
                            <div class="value ${priceChangeClass}">${priceChangeSymbol}${priceChange.toFixed(2)}%</div>
                        </div>
                        <div class="price-item">
                            <div class="label">Volume</div>
                            <div class="value">${(priceData.volume || 0).toLocaleString()}</div>
                        </div>
                        <div class="price-item">
                            <div class="label">High/Low</div>
                            <div class="value">$${(priceData.high_24h || 0).toFixed(2)} / $${(priceData.low_24h || 0).toFixed(2)}</div>
                        </div>
                    </div>
                    
                    <div class="sentiment-info">
                        <h4>Sentiment Analysis</h4>
                        <div class="sentiment-metrics">
                            <div class="sentiment-metric">
                                <div class="label">Sentiment Score</div>
                                <div class="value">${(signals.sentiment_score || 0).toFixed(3)}</div>
                            </div>
                            <div class="sentiment-metric">
                                <div class="label">Multiplier</div>
                                <div class="value">${(signals.sentiment_multiplier || 1).toFixed(3)}</div>
                            </div>
                            <div class="sentiment-metric">
                                <div class="label">Base Strength</div>
                                <div class="value">${(signals.base_signal_strength || 0).toFixed(3)}</div>
                            </div>
                            <div class="sentiment-metric">
                                <div class="label">Enhanced Strength</div>
                                <div class="value">${(signals.enhanced_signal_strength || 0).toFixed(3)}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="reasons">
                        <h4>Signal Reasoning</h4>
                        <ul class="reason-list">
                            ${(signals.reasons || []).map(reason => `<li>${reason}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            `;
        }
        
        function updateSentiment(alpha) {
            fetch('/api/sentiment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ alpha: alpha })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    loadSymbols(); // Refresh symbols with new sentiment
                }
            })
            .catch(error => console.error('Error updating sentiment:', error));
        }
        
        function refreshAnalysis() {
            fetch('/api/refresh')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        loadStatus();
                        loadSymbols();
                        loadNewsStats();
                        loadNewsArticles();
                    }
                })
                .catch(error => console.error('Error refreshing analysis:', error));
        }
        
        function loadNewsStats() {
            fetch('/api/news/sentiment-breakdown')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('totalArticles').textContent = data.total || 0;
                    document.getElementById('positiveArticles').textContent = data.positive || 0;
                    document.getElementById('negativeArticles').textContent = data.negative || 0;
                    document.getElementById('neutralArticles').textContent = data.neutral || 0;
                })
                .catch(error => console.error('Error loading news stats:', error));
        }
        
        function loadNewsArticles() {
            const search = document.getElementById('searchInput').value;
            const symbol = document.getElementById('symbolFilter').value;
            const sentiment = document.getElementById('sentimentFilter').value;
            const timeWindow = document.getElementById('timeWindowFilter').value;
            
            const params = new URLSearchParams({
                limit: pageSize,
                offset: currentPage * pageSize,
                search: search,
                symbol: symbol,
                sentiment: sentiment,
                time_window: timeWindow
            });
            
            fetch(`/api/news?${params}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('newsArticles').innerHTML = 
                            `<div class="loading">Error: ${data.error}</div>`;
                        return;
                    }
                    
                    displayNewsArticles(data.articles || []);
                })
                .catch(error => {
                    console.error('Error loading news articles:', error);
                    document.getElementById('newsArticles').innerHTML = 
                        '<div class="loading">Error loading news articles</div>';
                });
        }
        
        function displayNewsArticles(articles) {
            const container = document.getElementById('newsArticles');
            
            if (articles.length === 0) {
                container.innerHTML = '<div class="loading">No articles found</div>';
                return;
            }
            
            container.innerHTML = articles.map(article => `
                <div class="news-article">
                    <div class="article-header">
                        <div class="article-title">${article.headline || 'No title'}</div>
                        <div class="article-sentiment ${getSentimentClass(article.sentiment_score)}">
                            ${getSentimentLabel(article.sentiment_score)}
                        </div>
                    </div>
                    <div class="article-meta">
                        ${article.symbol || 'Unknown'} • ${article.source || 'Unknown'} • 
                        ${new Date(article.timestamp).toLocaleString()}
                    </div>
                    <div class="article-content">
                        ${(article.content || '').substring(0, 200)}...
                    </div>
                </div>
            `).join('');
        }
        
        function getSentimentClass(score) {
            if (score > 0.01) return 'positive';
            if (score < -0.01) return 'negative';
            return 'neutral';
        }
        
        function getSentimentLabel(score) {
            if (score > 0.01) return 'Positive';
            if (score < -0.01) return 'Negative';
            return 'Neutral';
        }
        
        function applyFilters() {
            currentPage = 0;
            loadNewsArticles();
        }
        
        function debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }
    </script>
</body>
</html>
        """
    
    def start_auto_update(self):
        """Start automatic updates"""
        if not self.running:
            self.running = True
            logger.info("Auto-update started")
            
            def update_loop():
                while self.running:
                    time.sleep(self.update_interval)
                    if self.running:
                        self.update_analysis()
            
            update_thread = threading.Thread(target=update_loop, daemon=True)
            update_thread.start()
    
    def run(self, host='localhost', port=8080, debug=False):
        """Run the dashboard"""
        try:
            logger.info("Starting Crypto Dashboard...")
            
            # Initial analysis update
            self.update_analysis()
            
            # Start auto-update
            self.start_auto_update()
            
            logger.info(f"Dashboard running at http://{host}:{port}")
            logger.info("Press Ctrl+C to stop")
            
            # Open browser
            webbrowser.open(f"http://{host}:{port}")
            
            # Run Flask app
            self.app.run(host=host, port=port, debug=debug)
            
        except KeyboardInterrupt:
            logger.info("Dashboard stopped by user")
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
        finally:
            self.running = False

if __name__ == "__main__":
    import sys
    
    # Get symbols from command line or use default
    symbols = ['BTC', 'ETH', 'ADA', 'SOL']
    if len(sys.argv) > 1:
        symbols = [s.upper().strip() for s in sys.argv[1:]]
    
    dashboard = CryptoDashboard(symbols=symbols)
    dashboard.run()
