"""
Visualization utilities for the crypto trading pipeline
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Basic chart generation utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_price_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create a basic price chart"""
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name=f'{symbol} Price',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title=f'{symbol} Price Chart',
                xaxis_title='Time',
                yaxis_title='Price ($)',
                template='plotly_white'
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating price chart: {e}")
            return go.Figure()
    
    def create_volume_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create a basic volume chart"""
        try:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['volume'],
                name=f'{symbol} Volume',
                marker_color='lightblue'
            ))
            fig.update_layout(
                title=f'{symbol} Volume Chart',
                xaxis_title='Time',
                yaxis_title='Volume',
                template='plotly_white'
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating volume chart: {e}")
            return go.Figure()

class DashboardCreator:
    """Basic dashboard creation utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_summary_dashboard(self, data: Dict[str, Any]) -> str:
        """Create a basic HTML dashboard"""
        try:
            html = f"""
            <html>
            <head>
                <title>Crypto Trading Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric {{ background: #f0f0f0; padding: 10px; margin: 5px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>Crypto Trading Dashboard</h1>
                <div class="metric">
                    <h3>Data Summary</h3>
                    <p>Total symbols: {len(data.get('symbols', []))}</p>
                    <p>Last updated: {data.get('last_updated', 'Unknown')}</p>
                </div>
            </body>
            </html>
            """
            return html
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {e}")
            return "<html><body><h1>Error creating dashboard</h1></body></html>"
