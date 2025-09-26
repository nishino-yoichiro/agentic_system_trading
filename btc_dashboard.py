"""
BTC Dashboard - Founder/Ironman Style
Clean web dashboard that auto-refreshes and serves latest analysis
"""

import asyncio
import threading
import time
from datetime import datetime
from pathlib import Path
import webbrowser
from flask import Flask, render_template_string, jsonify
import pandas as pd
from btc_analysis_engine import BTCAnalysisEngine
from btc_professional_report import BTCProfessionalReport
import json
import os
from loguru import logger

class BTCDashboard:
    """Clean web dashboard for BTC analysis"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.analysis_engine = BTCAnalysisEngine()
        self.report_generator = BTCProfessionalReport()
        self.latest_data = None
        self.last_update = None
        self.update_interval = 300  # 5 minutes
        self.running = False
        
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
                'data_available': self.latest_data is not None,
                'next_update_in': self.get_next_update_time()
            })
        
        @self.app.route('/api/refresh')
        def api_refresh():
            """Manual refresh endpoint"""
            self.update_analysis()
            return jsonify({'status': 'refreshed', 'timestamp': datetime.now().isoformat()})
    
    def render_dashboard(self):
        """Render the main dashboard"""
        if not self.latest_data:
            return self.render_loading()
        
        # Check for data errors
        if 'error' in self.latest_data:
            return self.render_error(self.latest_data['error'])
        
        recommendation = self.latest_data.get('recommendation', {})
        charts = self.latest_data.get('charts', {})
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>₿ BTC Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://s3.tradingview.com/tv.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            line-height: 1.6;
        }}
        
        .header {{
            background: linear-gradient(135deg, #f7931a, #ff6b35);
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(247, 147, 26, 0.3);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            font-weight: 300;
            margin-bottom: 10px;
        }}
        
        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .status-bar {{
            background: #1a1a1a;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid #333;
        }}
        
        .status-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #28a745;
        }}
        
        .recommendation {{
            background: {'#1e4d2b' if recommendation.get('action') == 'BUY' else '#4d1e1e' if recommendation.get('action') == 'SELL' else '#4d4d1e'};
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            border-left: 5px solid {'#28a745' if recommendation.get('action') == 'BUY' else '#dc3545' if recommendation.get('action') == 'SELL' else '#ffc107'};
        }}
        
        .rec-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .rec-action {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .rec-confidence {{
            font-size: 1.2em;
            opacity: 0.8;
        }}
        
        .rec-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .rec-item {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
        }}
        
        .rec-item h4 {{
            margin-bottom: 5px;
            opacity: 0.8;
        }}
        
        .rec-item .value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        
        .charts {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        .chart-container {{
            background: #1a1a1a;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #333;
        }}
        
        .chart-title {{
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #f7931a;
        }}
        
        .chart {{
            width: 100%;
            height: 400px;
        }}
        
        .tradingview-widget-container {{
            width: 100%;
            height: 400px;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }}
        
        .tradingview-widget-container iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
        
        .our-analysis-chart {{
            background: #1a1a1a;
            padding: 20px;
            border-radius: 10px;
            height: 300px;
            overflow-y: auto;
        }}
        
        .analysis-header {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #333;
        }}
        
        .current-signal {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .signal-action {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .signal-confidence {{
            font-size: 1.2em;
            opacity: 0.8;
        }}
        
        .price-levels {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}
        
        .price-item {{
            text-align: center;
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }}
        
        .price-label {{
            display: block;
            font-size: 0.9em;
            opacity: 0.7;
            margin-bottom: 5px;
        }}
        
        .price-value {{
            display: block;
            font-size: 1.3em;
            font-weight: bold;
        }}
        
        .price-value.support {{
            color: #28a745;
        }}
        
        .price-value.resistance {{
            color: #dc3545;
        }}
        
        .signal-timeline-chart {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}
        
        .timeline-item {{
            background: rgba(255,255,255,0.03);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #f7931a;
        }}
        
        .timeline-time {{
            font-size: 0.9em;
            opacity: 0.7;
            margin-bottom: 8px;
        }}
        
        .timeline-content {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .volume-indicator {{
            font-size: 1.2em;
            font-weight: bold;
            padding: 8px 12px;
            border-radius: 5px;
            text-align: center;
        }}
        
        .volume-reason {{
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        .indicator-row {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        
        .indicator {{
            background: rgba(255,255,255,0.1);
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        
        .risk-metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }}
        
        .risk-item {{
            text-align: center;
            padding: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 5px;
        }}
        
        .risk-label {{
            display: block;
            font-size: 0.8em;
            opacity: 0.7;
            margin-bottom: 3px;
        }}
        
        .risk-value {{
            display: block;
            font-size: 1.1em;
            font-weight: bold;
        }}
        
        .strategy-overlay, .volume-analysis, .technical-analysis {{
            background: #1a1a1a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            border: 1px solid #333;
        }}
        
        .strategy-overlay h4, .volume-analysis h4, .technical-analysis h4 {{
            color: #f7931a;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .signal-grid, .volume-stats, .indicator-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }}
        
        .signal-item, .volume-item, .indicator-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: rgba(255,255,255,0.05);
            border-radius: 5px;
        }}
        
        .signal-label, .volume-label, .indicator-label {{
            font-weight: 500;
            opacity: 0.8;
        }}
        
        .signal-value, .volume-value, .indicator-value {{
            font-weight: bold;
            font-size: 1.1em;
        }}
        
        .action-buy {{
            color: #28a745;
        }}
        
        .action-sell {{
            color: #dc3545;
        }}
        
        .action-hold {{
            color: #ffc107;
        }}
        
        .bullish {{
            color: #28a745;
        }}
        
        .bearish {{
            color: #dc3545;
        }}
        
        .neutral {{
            color: #6c757d;
        }}
        
        .signal-timeline {{
            background: #1a1a1a;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            border: 1px solid #333;
        }}
        
        .signal-timeline h3 {{
            color: #f7931a;
            margin-bottom: 20px;
        }}
        
        .timeline-container {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        
        .timeline-item {{
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #f7931a;
        }}
        
        .timeline-time {{
            font-size: 0.9em;
            opacity: 0.7;
            margin-bottom: 5px;
        }}
        
        .timeline-action {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .timeline-reason {{
            margin-bottom: 10px;
            line-height: 1.5;
        }}
        
        .timeline-metrics {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        
        .metric {{
            background: rgba(255,255,255,0.1);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        
        .volume-increasing {{
            color: #28a745;
        }}
        
        .volume-decreasing {{
            color: #dc3545;
        }}
        
        .volume-stable {{
            color: #6c757d;
        }}
        
        .risk-2 {{
            color: #28a745;
        }}
        
        .risk-1 {{
            color: #ffc107;
        }}
        
        .reasoning {{
            background: #1a1a1a;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            border: 1px solid #333;
        }}
        
        .reasoning h3 {{
            color: #f7931a;
            margin-bottom: 15px;
        }}
        
        .reasoning ul {{
            list-style: none;
            padding: 0;
        }}
        
        .reasoning li {{
            padding: 8px 0;
            border-bottom: 1px solid #333;
        }}
        
        .reasoning li:last-child {{
            border-bottom: none;
        }}
        
        .refresh-btn {{
            background: #f7931a;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }}
        
        .refresh-btn:hover {{
            background: #e8830a;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .rec-details {{
                grid-template-columns: 1fr;
            }}
            
            .status-bar {{
                flex-direction: column;
                gap: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>₿ BTC Dashboard</h1>
        <p>Real-time Analysis & Trading Signals</p>
    </div>
    
    <div class="container">
        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot"></div>
                <span>Live Data</span>
            </div>
            <div class="status-item">
                <span>Last Update: {self.last_update.strftime('%H:%M:%S') if self.last_update else 'Never'}</span>
            </div>
            <div class="status-item">
                <button class="refresh-btn" onclick="refreshData()">Refresh Now</button>
            </div>
        </div>
        
        <div class="recommendation">
            <div class="rec-header">
                <div class="rec-action">{recommendation.get('action', 'HOLD')}</div>
                <div class="rec-confidence">{recommendation.get('confidence', 0):.1%} Confidence</div>
            </div>
            
            <div class="rec-details">
                <div class="rec-item">
                    <h4>Current Price</h4>
                    <div class="value">${recommendation.get('current_price', 0):,.2f}</div>
                </div>
                <div class="rec-item">
                    <h4>Price Target</h4>
                    <div class="value">${recommendation.get('price_target', 0):,.2f}</div>
                </div>
                <div class="rec-item">
                    <h4>Stop Loss</h4>
                    <div class="value">${recommendation.get('stop_loss', 0):,.2f}</div>
                </div>
                <div class="rec-item">
                    <h4>Risk/Reward</h4>
                    <div class="value">{recommendation.get('risk_reward_ratio', 0):.2f}</div>
                </div>
            </div>
        </div>
        
        <div class="charts">
            <div class="chart-container">
                <div class="chart-title">📈 BTC/USD - TradingView Professional Chart</div>
                <div class="tradingview-widget-container">
                    <div id="tradingview_btc" class="chart"></div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">🎯 Our Analysis & Signals</div>
                <div class="our-analysis-chart">
                    <div class="analysis-header">
                        <div class="current-signal">
                            <span class="signal-action action-{recommendation.get('action', 'HOLD').lower()}">{recommendation.get('action', 'HOLD')}</span>
                            <span class="signal-confidence">{recommendation.get('confidence', 0):.1%} Confidence</span>
                        </div>
                        <div class="price-levels">
                            <div class="price-item">
                                <span class="price-label">Current:</span>
                                <span class="price-value">${recommendation.get('current_price', 0):,.0f}</span>
                            </div>
                            <div class="price-item">
                                <span class="price-label">Support:</span>
                                <span class="price-value support">${recommendation.get('stop_loss', 0):,.0f}</span>
                            </div>
                            <div class="price-item">
                                <span class="price-label">Resistance:</span>
                                <span class="price-value resistance">${recommendation.get('price_target', 0):,.0f}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="signal-timeline-chart">
                        <div class="timeline-item">
                            <div class="timeline-time">Volume Analysis</div>
                            <div class="timeline-content">
                                <div class="volume-indicator volume-{recommendation.get('volume_trend', 'stable')}">
                                    {recommendation.get('volume_ratio', 1):.2f}x Average
                                </div>
                                <div class="volume-reason">
                                    {'Significant volume spike detected' if recommendation.get('volume_ratio', 1) > 1.5 else 'Volume within normal range'}
                                </div>
                            </div>
                        </div>
                        
                        <div class="timeline-item">
                            <div class="timeline-time">Technical Signals</div>
                            <div class="timeline-content">
                                <div class="indicator-row">
                                    <span class="indicator">RSI: {recommendation.get('rsi', 50):.1f}</span>
                                    <span class="indicator">MACD: {recommendation.get('macd', 0):.2f}</span>
                                    <span class="indicator">Change: {recommendation.get('price_change_24h', 0):.2f}%</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="timeline-item">
                            <div class="timeline-time">Risk Assessment</div>
                            <div class="timeline-content">
                                <div class="risk-metrics">
                                    <div class="risk-item">
                                        <span class="risk-label">Risk/Reward:</span>
                                        <span class="risk-value">{recommendation.get('risk_reward_ratio', 1):.2f}</span>
                                    </div>
                                    <div class="risk-item">
                                        <span class="risk-label">Stop Loss:</span>
                                        <span class="risk-value">${recommendation.get('stop_loss', 0):,.0f}</span>
                                    </div>
                                    <div class="risk-item">
                                        <span class="risk-label">Target:</span>
                                        <span class="risk-value">${recommendation.get('price_target', 0):,.0f}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">📊 Volume Analysis</div>
                <div class="tradingview-widget-container">
                    <div id="tradingview_volume" class="chart"></div>
                </div>
                <div class="volume-analysis">
                    <h4>📈 Volume Analysis</h4>
                    <div class="volume-stats">
                        <div class="volume-item">
                            <span class="volume-label">Current Volume:</span>
                            <span class="volume-value">{recommendation.get('current_volume', 'N/A')}</span>
                        </div>
                        <div class="volume-item">
                            <span class="volume-label">Volume Ratio:</span>
                            <span class="volume-value">{recommendation.get('volume_ratio', 'N/A'):.2f}x</span>
                        </div>
                        <div class="volume-item">
                            <span class="volume-label">Volume Trend:</span>
                            <span class="volume-value">{recommendation.get('volume_trend', 'N/A')}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">🔧 Technical Indicators</div>
                <div class="tradingview-widget-container">
                    <div id="tradingview_technical" class="chart"></div>
                </div>
                <div class="technical-analysis">
                    <h4>🔧 Our Technical Analysis</h4>
                    <div class="indicator-grid">
                        <div class="indicator-item">
                            <span class="indicator-label">RSI:</span>
                            <span class="indicator-value {recommendation.get('rsi_status', 'neutral')}">{recommendation.get('rsi', 'N/A'):.1f}</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-label">MACD:</span>
                            <span class="indicator-value {recommendation.get('macd_status', 'neutral')}">{recommendation.get('macd', 'N/A'):.2f}</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-label">Price Change:</span>
                            <span class="indicator-value {recommendation.get('price_change_status', 'neutral')}">{recommendation.get('price_change_24h', 'N/A'):.2f}%</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-label">Confidence:</span>
                            <span class="indicator-value">{recommendation.get('confidence', 0):.1%}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="reasoning">
            <h3>📊 Analysis Reasoning</h3>
            <ul>
                {''.join([f'<li>{reason}</li>' for reason in recommendation.get('reasoning', [])])}
            </ul>
        </div>
        
        <div class="signal-timeline">
            <h3>🎯 Signal Timeline & Reasoning</h3>
            <div class="timeline-container">
                <div class="timeline-item">
                    <div class="timeline-time">Current Signal</div>
                    <div class="timeline-action action-{recommendation.get('action', 'HOLD').lower()}">{recommendation.get('action', 'HOLD')}</div>
                    <div class="timeline-reason">
                        <strong>Why:</strong> {recommendation.get('reasoning', ['No specific reasoning available'])[0] if recommendation.get('reasoning') else 'Analysis in progress'}
                    </div>
                    <div class="timeline-metrics">
                        <span class="metric">Confidence: {recommendation.get('confidence', 0):.1%}</span>
                        <span class="metric">Volume Ratio: {recommendation.get('volume_ratio', 1):.2f}x</span>
                        <span class="metric">RSI: {recommendation.get('rsi', 50):.1f}</span>
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-time">Volume Analysis</div>
                    <div class="timeline-action volume-{recommendation.get('volume_trend', 'stable')}">{recommendation.get('volume_trend', 'stable').title()}</div>
                    <div class="timeline-reason">
                        <strong>Volume Status:</strong> Current volume is {recommendation.get('volume_ratio', 1):.2f}x the average. 
                        {'This indicates significant buying/selling pressure.' if recommendation.get('volume_ratio', 1) > 1.5 else 'Volume is within normal ranges.'}
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-time">Risk Assessment</div>
                    <div class="timeline-action risk-{recommendation.get('risk_reward_ratio', 1)}">R/R: {recommendation.get('risk_reward_ratio', 1):.2f}</div>
                    <div class="timeline-reason">
                        <strong>Risk/Reward:</strong> {'Favorable risk/reward ratio' if recommendation.get('risk_reward_ratio', 1) > 2 else 'Moderate risk/reward ratio'} 
                        with stop loss at ${recommendation.get('stop_loss', 0):,.0f} and target at ${recommendation.get('price_target', 0):,.0f}.
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function refreshData() {{
            fetch('/api/refresh')
                .then(response => response.json())
                .then(data => {{
                    if (data.status === 'refreshed') {{
                        location.reload();
                    }}
                }});
        }}
        
        // Initialize TradingView charts
        function initTradingViewCharts() {{
            // Main BTC chart
            new TradingView.widget({{
                "autosize": true,
                "symbol": "COINBASE:BTCUSD",
                "interval": "1",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#1a1a1a",
                "enable_publishing": false,
                "hide_top_toolbar": false,
                "hide_legend": false,
                "save_image": false,
                "container_id": "tradingview_btc",
                "studies": [
                    "RSI@tv-basicstudies",
                    "MACD@tv-basicstudies",
                    "Volume@tv-basicstudies"
                ]
            }});
            
            // Volume chart
            new TradingView.widget({{
                "autosize": true,
                "symbol": "COINBASE:BTCUSD",
                "interval": "1",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#1a1a1a",
                "enable_publishing": false,
                "hide_top_toolbar": true,
                "hide_legend": false,
                "save_image": false,
                "container_id": "tradingview_volume",
                "studies": [
                    "Volume@tv-basicstudies"
                ]
            }});
            
            // Technical indicators chart
            new TradingView.widget({{
                "autosize": true,
                "symbol": "COINBASE:BTCUSD",
                "interval": "1",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#1a1a1a",
                "enable_publishing": false,
                "hide_top_toolbar": true,
                "hide_legend": false,
                "save_image": false,
                "container_id": "tradingview_technical",
                "studies": [
                    "RSI@tv-basicstudies",
                    "MACD@tv-basicstudies",
                    "Bollinger Bands@tv-basicstudies",
                    "EMA@tv-basicstudies"
                ]
            }});
        }}
        
        // Initialize charts when page loads
        window.addEventListener('load', initTradingViewCharts);
        
        // Auto-refresh every 5 minutes
        setInterval(() => {{
            location.reload();
        }}, 300000);
    </script>
</body>
</html>
        """
        return html
    
    def render_loading(self):
        """Render loading page"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>₿ BTC Dashboard - Loading</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .loading {
            text-align: center;
        }
        .spinner {
            border: 4px solid #333;
            border-top: 4px solid #f7931a;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="loading">
        <div class="spinner"></div>
        <h2>₿ Loading BTC Analysis...</h2>
        <p>Generating latest market analysis</p>
    </div>
    <script>
        setTimeout(() => location.reload(), 2000);
    </script>
</body>
</html>
        """
    
    def render_error(self, error_message):
        """Render error page"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>₿ BTC Dashboard - Error</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }}
        .error {{
            text-align: center;
            background: #2d1b1b;
            padding: 40px;
            border-radius: 15px;
            border: 2px solid #dc3545;
            max-width: 500px;
        }}
        .error-icon {{
            font-size: 4em;
            color: #dc3545;
            margin-bottom: 20px;
        }}
        .error h2 {{
            color: #dc3545;
            margin-bottom: 15px;
        }}
        .error p {{
            margin-bottom: 20px;
            opacity: 0.8;
        }}
        .retry-btn {{
            background: #dc3545;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }}
        .retry-btn:hover {{
            background: #c82333;
        }}
    </style>
</head>
<body>
    <div class="error">
        <div class="error-icon">⚠️</div>
        <h2>Data Quality Issue</h2>
        <p>{error_message}</p>
        <p>The system detected that the data may not be real or sufficient for analysis.</p>
        <button class="retry-btn" onclick="location.reload()">Retry Analysis</button>
    </div>
</body>
</html>
        """
    
    def update_analysis(self):
        """Update the analysis data"""
        try:
            logger.info("Updating BTC analysis...")
            
            # Load real data once
            df = self.analysis_engine.load_real_btc_data(days=7)
            
            if df.empty:
                logger.error("No BTC data available")
                return
            
            # Verify we have real data
            logger.info(f"Loaded {len(df)} data points")
            logger.info(f"Volume range: {df['volume'].min():.2f} to {df['volume'].max():.2f}")
            logger.info(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
            
            # Validate data quality
            if df['volume'].nunique() < 10:  # Less than 10 unique volume values
                logger.warning("Volume data appears synthetic - insufficient variation")
                self.latest_data = {'error': 'Volume data appears synthetic'}
                return
            
            if df['close'].nunique() < 10:  # Less than 10 unique price values
                logger.warning("Price data appears synthetic - insufficient variation")
                self.latest_data = {'error': 'Price data appears synthetic'}
                return
            
            # Generate new analysis
            recommendation = self.analysis_engine.generate_recommendation(df)
            
            # Generate charts with real data
            price_chart = self.report_generator._create_price_chart(df, recommendation)
            volume_chart = self.report_generator._create_volume_chart(df)
            technical_chart = self.report_generator._create_technical_chart(df)
            
            # Get additional analysis data
            indicators = self.analysis_engine.calculate_technical_indicators(df)
            
            # Store latest data
            self.latest_data = {
                'recommendation': {
                    'action': recommendation.action,
                    'confidence': recommendation.confidence,
                    'current_price': recommendation.current_price,
                    'price_target': recommendation.price_target,
                    'stop_loss': recommendation.stop_loss,
                    'risk_reward_ratio': recommendation.risk_reward_ratio,
                    'reasoning': recommendation.reasoning,
                    'rsi': indicators.get('rsi', 0),
                    'macd': indicators.get('macd', 0),
                    'volume_ratio': indicators.get('volume_ratio', 0),
                    'price_change_24h': indicators.get('price_change_24h', 0),
                    'current_volume': df['volume'].iloc[-1] if not df.empty else 0,
                    'rsi_status': 'bullish' if indicators.get('rsi', 50) > 70 else 'bearish' if indicators.get('rsi', 50) < 30 else 'neutral',
                    'macd_status': 'bullish' if indicators.get('macd', 0) > 0 else 'bearish',
                    'price_change_status': 'bullish' if indicators.get('price_change_24h', 0) > 0 else 'bearish',
                    'volume_trend': 'increasing' if indicators.get('volume_ratio', 1) > 1.2 else 'decreasing' if indicators.get('volume_ratio', 1) < 0.8 else 'stable'
                },
                'charts': {
                    'price_chart': price_chart,
                    'volume_chart': volume_chart,
                    'technical_chart': technical_chart
                }
            }
            
            self.last_update = datetime.now()
            
            # Generate new report every 5 minutes
            try:
                logger.info("Generating new BTC professional report...")
                report_html = self.report_generator.generate_report()
                
                # Save the report to a fixed location for easy access
                report_path = Path('reports/btc_latest_report.html')
                report_path.parent.mkdir(exist_ok=True)
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_html)
                
                logger.info(f"Report saved to: {report_path}")
                
                # Also save with timestamp for history
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                timestamped_path = Path(f'reports/btc_report_{timestamp}.html')
                with open(timestamped_path, 'w', encoding='utf-8') as f:
                    f.write(report_html)
                
                logger.info(f"Timestamped report saved to: {timestamped_path}")
                
            except Exception as e:
                logger.error(f"Error generating report: {e}")
            
            logger.info("BTC analysis updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating analysis: {e}")
    
    def get_next_update_time(self):
        """Get seconds until next update"""
        if not self.last_update:
            return 0
        elapsed = (datetime.now() - self.last_update).total_seconds()
        return max(0, self.update_interval - elapsed)
    
    def start_auto_update(self):
        """Start auto-update thread"""
        def update_loop():
            while self.running:
                self.update_analysis()
                time.sleep(self.update_interval)
        
        self.running = True
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        logger.info("Auto-update started")
    
    def run(self, host='localhost', port=8080, auto_open=True):
        """Run the dashboard"""
        logger.info("Starting BTC Dashboard...")
        
        # Initial update
        self.update_analysis()
        
        # Start auto-update
        self.start_auto_update()
        
        # Open browser
        if auto_open:
            webbrowser.open(f'http://{host}:{port}')
        
        logger.info(f"Dashboard running at http://{host}:{port}")
        logger.info("Press Ctrl+C to stop")
        
        # Run Flask app
        self.app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    dashboard = BTCDashboard()
    dashboard.run()
