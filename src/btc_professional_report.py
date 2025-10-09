"""
Professional BTC Analysis Report
Clean, data-driven analysis using real BTC data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from btc_analysis_engine import BTCAnalysisEngine, BTCRecommendation
from loguru import logger


class BTCProfessionalReport:
    """Clean, professional BTC analysis report"""
    
    def __init__(self):
        self.analysis_engine = BTCAnalysisEngine()
        self.output_dir = Path("reports/btc_analysis")
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_report(self) -> str:
        """Generate comprehensive BTC analysis report"""
        logger.info("Generating professional BTC analysis report...")
        
        # Load real data
        df = self.analysis_engine.load_symbol_data('BTC', days=30)  # Last 30 days
        
        if df.empty:
            logger.error("No BTC data available")
            return None
        
        # Generate analysis
        recommendation = self.analysis_engine.generate_recommendation(df)
        
        # Create charts
        price_chart = self._create_price_chart(df, recommendation)
        volume_chart = self._create_volume_chart(df)
        technical_chart = self._create_technical_chart(df)
        
        # Generate HTML report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"btc_analysis_{timestamp}.html"
        
        html_content = self._create_html_report(recommendation, price_chart, volume_chart, technical_chart)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {report_file}")
        return str(report_file)
    
    def _create_price_chart(self, df: pd.DataFrame, recommendation: BTCRecommendation) -> str:
        """Create price chart with key levels"""
        fig = go.Figure()
        
        # Convert timezone-aware index to naive for Plotly compatibility
        x_data = df.index.tz_localize(None) if df.index.tz else df.index
        
        # Price line
        fig.add_trace(go.Scatter(
            x=x_data,
            y=df['close'],
            name='BTC Price',
            line=dict(color='#f7931a', width=2)
        ))
        
        # Key levels
        current_price = recommendation.current_price
        support = recommendation.support_resistance.current_support
        resistance = recommendation.support_resistance.current_resistance
        
        # Support line
        fig.add_hline(
            y=support,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Support: ${support:,.0f}"
        )
        
        # Resistance line
        fig.add_hline(
            y=resistance,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Resistance: ${resistance:,.0f}"
        )
        
        # Current price line
        fig.add_hline(
            y=current_price,
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Current: ${current_price:,.0f}"
        )
        
        fig.update_layout(
            title="BTC Price Analysis with Key Levels",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            height=500,
            showlegend=True
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_volume_chart(self, df: pd.DataFrame) -> str:
        """Create volume analysis chart"""
        fig = go.Figure()
        
        # Convert timezone-aware index to naive for Plotly compatibility
        x_data = df.index.tz_localize(None) if df.index.tz else df.index
        
        # Volume bars
        fig.add_trace(go.Bar(
            x=x_data,
            y=df['volume'],
            name='Volume',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Volume moving average
        volume_sma = df['volume'].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=x_data,
            y=volume_sma,
            name='Volume SMA(20)',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Volume Analysis",
            xaxis_title="Time",
            yaxis_title="Volume",
            height=300,
            showlegend=True
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_technical_chart(self, df: pd.DataFrame) -> str:
        """Create technical indicators chart"""
        # Convert timezone-aware index to naive for Plotly compatibility
        x_data = df.index.tz_localize(None) if df.index.tz else df.index
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('RSI', 'MACD'),
            row_heights=[0.5, 0.5]
        )
        
        # RSI
        fig.add_trace(go.Scatter(
            x=x_data,
            y=rsi,
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=1, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=1)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=macd,
            name='MACD',
            line=dict(color='blue', width=2)
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=macd_signal,
            name='Signal',
            line=dict(color='red', width=2)
        ), row=2, col=1)
        
        fig.add_trace(go.Bar(
            x=x_data,
            y=macd_histogram,
            name='Histogram',
            marker_color='lightblue',
            opacity=0.7
        ), row=2, col=1)
        
        fig.update_layout(
            title="Technical Indicators",
            height=600,
            showlegend=True
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_html_report(self, recommendation: BTCRecommendation, price_chart: str, 
                          volume_chart: str, technical_chart: str) -> str:
        """Create the HTML report"""
        
        # Determine recommendation color
        if recommendation.action == "BUY":
            rec_color = "#d4edda"  # Green
            rec_text_color = "#155724"
        elif recommendation.action == "SELL":
            rec_color = "#f8d7da"  # Red
            rec_text_color = "#721c24"
        else:
            rec_color = "#fff3cd"  # Yellow
            rec_text_color = "#856404"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>BTC Professional Analysis - {datetime.now().strftime('%Y-%m-%d')}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #f7931a, #ff6b35);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .recommendation {{
            background-color: {rec_color};
            color: {rec_text_color};
            padding: 30px;
            border-left: 5px solid {rec_text_color};
            margin: 20px;
            border-radius: 5px;
        }}
        .recommendation h2 {{
            margin: 0 0 20px 0;
            font-size: 1.8em;
        }}
        .rec-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .rec-item {{
            background: rgba(255,255,255,0.7);
            padding: 15px;
            border-radius: 5px;
        }}
        .rec-item h4 {{
            margin: 0 0 10px 0;
            color: {rec_text_color};
        }}
        .rec-item .value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        .reasoning {{
            margin: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .reasoning h3 {{
            margin: 0 0 15px 0;
            color: #495057;
        }}
        .reasoning ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .reasoning li {{
            margin: 8px 0;
            line-height: 1.5;
        }}
        .chart-section {{
            margin: 20px;
            padding: 20px;
            background: white;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }}
        .chart-section h3 {{
            margin: 0 0 20px 0;
            color: #495057;
        }}
        .chart {{
            width: 100%;
            height: 500px;
        }}
        .flags {{
            margin: 20px;
            padding: 20px;
            background: #e9ecef;
            border-radius: 5px;
        }}
        .flags h3 {{
            margin: 0 0 15px 0;
            color: #495057;
        }}
        .flag-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        .flag {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }}
        .flag.bullish {{
            border-left-color: #28a745;
        }}
        .flag.bearish {{
            border-left-color: #dc3545;
        }}
        .flag h4 {{
            margin: 0 0 10px 0;
            color: #495057;
        }}
        .flag p {{
            margin: 0;
            font-size: 0.9em;
            color: #6c757d;
        }}
        .footer {{
            background: #343a40;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>₿ Bitcoin Professional Analysis</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="recommendation">
            <h2>🎯 Trading Recommendation: {recommendation.action}</h2>
            <div class="rec-details">
                <div class="rec-item">
                    <h4>Confidence</h4>
                    <div class="value">{recommendation.confidence:.1%}</div>
                </div>
                <div class="rec-item">
                    <h4>Current Price</h4>
                    <div class="value">${recommendation.current_price:,.2f}</div>
                </div>
                <div class="rec-item">
                    <h4>Price Target</h4>
                    <div class="value">${recommendation.price_target:,.2f}</div>
                </div>
                <div class="rec-item">
                    <h4>Stop Loss</h4>
                    <div class="value">${recommendation.stop_loss:,.2f}</div>
                </div>
                <div class="rec-item">
                    <h4>Risk/Reward</h4>
                    <div class="value">{recommendation.risk_reward_ratio:.2f}</div>
                </div>
                <div class="rec-item">
                    <h4>Time Horizon</h4>
                    <div class="value">{recommendation.time_horizon}</div>
                </div>
            </div>
        </div>
        
        <div class="reasoning">
            <h3>📊 Analysis Reasoning</h3>
            <ul>
                {''.join([f'<li>{reason}</li>' for reason in recommendation.reasoning])}
            </ul>
        </div>
        
        <div class="flags">
            <h3>🚩 Trading Flags</h3>
            <div class="flag-grid">
                {''.join([f'''
                <div class="flag {'bullish' if flag.bullish else 'bearish'}">
                    <h4>{flag.name.replace('_', ' ').title()}</h4>
                    <p>{flag.reasoning}</p>
                </div>
                ''' for flag in recommendation.flags])}
            </div>
        </div>
        
        <div class="chart-section">
            <h3>📈 Price Analysis</h3>
            <div class="chart">{price_chart}</div>
        </div>
        
        <div class="chart-section">
            <h3>📊 Volume Analysis</h3>
            <div class="chart">{volume_chart}</div>
        </div>
        
        <div class="chart-section">
            <h3>🔧 Technical Indicators</h3>
            <div class="chart">{technical_chart}</div>
        </div>
        
        <div class="footer">
            <p>This analysis is for educational purposes only. Trading involves substantial risk of loss.</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html


if __name__ == "__main__":
    report_generator = BTCProfessionalReport()
    report_file = report_generator.generate_report()
    print(f"Professional BTC report generated: {report_file}")
