"""
Comprehensive Report Generation

Features:
- PDF report generation with charts and analysis
- Performance metrics and risk analysis
- Trading recommendations and portfolio insights
- Market overview and sentiment analysis
- Customizable report templates
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from loguru import logger
import json
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import base64


class ReportGenerator:
    """Generate comprehensive trading reports"""
    
    def __init__(self):
        self.template_dir = Path("templates")
        self.output_dir = Path("reports")
        self.chart_dir = Path("charts")
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.chart_dir.mkdir(exist_ok=True)
        
        # Report styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom report styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        ))
    
    async def initialize(self):
        """Initialize the report generator"""
        logger.info("Report generator initialized")
    
    async def generate_daily_report(
        self,
        report_data: Dict[str, Any],
        output_dir: Optional[Path] = None
    ) -> str:
        """Generate comprehensive daily report"""
        logger.info("Generating daily report")
        
        try:
            output_dir = output_dir or self.output_dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"daily_report_{timestamp}.pdf"
            report_path = output_dir / report_filename
            
            # Create PDF document
            doc = SimpleDocTemplate(str(report_path), pagesize=A4)
            story = []
            
            # Generate report sections
            story.extend(self._create_header(report_data))
            story.extend(self._create_executive_summary(report_data))
            story.extend(self._create_market_overview(report_data))
            story.extend(self._create_trading_recommendations(report_data))
            story.extend(self._create_risk_analysis(report_data))
            story.extend(self._create_performance_metrics(report_data))
            story.extend(self._create_appendix(report_data))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Daily report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            raise
    
    def _create_header(self, data: Dict[str, Any]) -> List:
        """Create report header"""
        elements = []
        
        # Title
        title = Paragraph("Enhanced Crypto Trading Pipeline", self.styles['CustomTitle'])
        elements.append(title)
        
        # Subtitle
        subtitle = Paragraph("Daily Trading Report", self.styles['Heading2'])
        elements.append(subtitle)
        
        # Date and time
        timestamp = data.get('timestamp', datetime.now())
        date_str = timestamp.strftime("%B %d, %Y at %H:%M UTC")
        date_para = Paragraph(f"Generated on {date_str}", self.styles['CustomBody'])
        elements.append(date_para)
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _create_executive_summary(self, data: Dict[str, Any]) -> List:
        """Create executive summary section"""
        elements = []
        
        # Section header
        header = Paragraph("Executive Summary", self.styles['CustomHeading'])
        elements.append(header)
        
        # Get key metrics
        recommendations = data.get('recommendations', [])
        features = data.get('features', {})
        sentiment = features.get('sentiment', {})
        
        # Summary statistics
        total_recommendations = len(recommendations)
        buy_signals = len([r for r in recommendations if r.get('action') == 'buy'])
        sell_signals = len([r for r in recommendations if r.get('action') == 'sell'])
        
        avg_sentiment = sentiment.get('average_sentiment', 0)
        sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
        
        # Create summary text
        summary_text = f"""
        <b>Market Overview:</b> The market shows {sentiment_label.lower()} sentiment with an average score of {avg_sentiment:.2f}.
        <br/><br/>
        <b>Trading Recommendations:</b> Generated {total_recommendations} recommendations: {buy_signals} buy signals and {sell_signals} sell signals.
        <br/><br/>
        <b>Key Insights:</b> The pipeline analyzed market data and generated actionable trading signals based on technical analysis, sentiment analysis, and Monte Carlo simulations.
        """
        
        summary = Paragraph(summary_text, self.styles['CustomBody'])
        elements.append(summary)
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _create_market_overview(self, data: Dict[str, Any]) -> List:
        """Create market overview section"""
        elements = []
        
        # Section header
        header = Paragraph("Market Overview", self.styles['CustomHeading'])
        elements.append(header)
        
        # Sentiment analysis
        features = data.get('features', {})
        sentiment = features.get('sentiment', {})
        
        if sentiment:
            sentiment_text = f"""
            <b>Sentiment Analysis:</b><br/>
            • Average Sentiment: {sentiment.get('average_sentiment', 0):.2f}<br/>
            • Positive Articles: {sentiment.get('positive_ratio', 0):.1%}<br/>
            • Negative Articles: {sentiment.get('negative_ratio', 0):.1%}<br/>
            • Total Articles Analyzed: {sentiment.get('total_articles', 0)}<br/>
            • Average Confidence: {sentiment.get('average_confidence', 0):.2f}
            """
            elements.append(Paragraph(sentiment_text, self.styles['CustomBody']))
        
        # Price data summary
        price_data = data.get('price_data', {})
        if price_data:
            price_text = f"<b>Price Data:</b> Analyzed {len(price_data)} assets with historical price data."
            elements.append(Paragraph(price_text, self.styles['CustomBody']))
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _create_trading_recommendations(self, data: Dict[str, Any]) -> List:
        """Create trading recommendations section"""
        elements = []
        
        # Section header
        header = Paragraph("Trading Recommendations", self.styles['CustomHeading'])
        elements.append(header)
        
        recommendations = data.get('recommendations', [])
        
        if not recommendations:
            no_recs = Paragraph("No trading recommendations generated.", self.styles['CustomBody'])
            elements.append(no_recs)
            elements.append(Spacer(1, 20))
            return elements
        
        # Create recommendations table
        table_data = [['Symbol', 'Action', 'Confidence', 'Reasoning']]
        
        for rec in recommendations[:10]:  # Show top 10
            symbol = rec.get('symbol', 'N/A')
            action = rec.get('action', 'N/A')
            confidence = rec.get('confidence', 0)
            reasoning = rec.get('reasoning', 'N/A')[:50] + '...' if len(rec.get('reasoning', '')) > 50 else rec.get('reasoning', 'N/A')
            
            table_data.append([
                symbol,
                action.upper(),
                f"{confidence:.2f}",
                reasoning
            ])
        
        # Create table
        table = Table(table_data, colWidths=[1*inch, 0.8*inch, 0.8*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        return elements
    
    def _create_risk_analysis(self, data: Dict[str, Any]) -> List:
        """Create risk analysis section"""
        elements = []
        
        # Section header
        header = Paragraph("Risk Analysis", self.styles['CustomHeading'])
        elements.append(header)
        
        # Simulation results
        features = data.get('features', {})
        simulation = features.get('simulation', {})
        
        if simulation:
            risk_text = f"""
            <b>Portfolio Risk Metrics:</b><br/>
            • Expected Return: {simulation.get('expected_return', 0):.2%}<br/>
            • Volatility: {simulation.get('volatility', 0):.2%}<br/>
            • Sharpe Ratio: {simulation.get('sharpe_ratio', 0):.2f}<br/>
            • Maximum Drawdown: {simulation.get('max_drawdown', 0):.2%}<br/>
            • VaR (95%): {simulation.get('var_95', 0):.2%}<br/>
            • CVaR (95%): {simulation.get('cvar_95', 0):.2%}
            """
            elements.append(Paragraph(risk_text, self.styles['CustomBody']))
        else:
            no_risk = Paragraph("Risk analysis data not available.", self.styles['CustomBody'])
            elements.append(no_risk)
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _create_performance_metrics(self, data: Dict[str, Any]) -> List:
        """Create performance metrics section"""
        elements = []
        
        # Section header
        header = Paragraph("Performance Metrics", self.styles['CustomHeading'])
        elements.append(header)
        
        # Technical indicators summary
        features = data.get('features', {})
        technical = features.get('technical', {})
        
        if technical:
            # Calculate average RSI
            rsi_values = [indicators.rsi for indicators in technical.values() if hasattr(indicators, 'rsi')]
            avg_rsi = np.mean(rsi_values) if rsi_values else 50
            
            # Calculate average MACD
            macd_values = [indicators.macd for indicators in technical.values() if hasattr(indicators, 'macd')]
            avg_macd = np.mean(macd_values) if macd_values else 0
            
            perf_text = f"""
            <b>Technical Indicators Summary:</b><br/>
            • Average RSI: {avg_rsi:.1f}<br/>
            • Average MACD: {avg_macd:.4f}<br/>
            • Assets Analyzed: {len(technical)}<br/>
            • Market Regime: {technical.get('market_regime', 'Unknown')}
            """
            elements.append(Paragraph(perf_text, self.styles['CustomBody']))
        else:
            no_perf = Paragraph("Performance metrics not available.", self.styles['CustomBody'])
            elements.append(no_perf)
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _create_appendix(self, data: Dict[str, Any]) -> List:
        """Create appendix section"""
        elements = []
        
        # Section header
        header = Paragraph("Appendix", self.styles['CustomHeading'])
        elements.append(header)
        
        # Configuration info
        config = data.get('config', {})
        config_text = f"""
        <b>Pipeline Configuration:</b><br/>
        • Data Collection Period: {config.get('time_horizon_days', 30)} days<br/>
        • Simulation Runs: {config.get('simulation_runs', 10000)}<br/>
        • Confidence Threshold: {config.get('confidence_threshold', 0.6)}<br/>
        • Risk Tolerance: {config.get('risk_tolerance', 'medium')}
        """
        elements.append(Paragraph(config_text, self.styles['CustomBody']))
        
        # Disclaimer
        disclaimer = Paragraph(
            "<b>Disclaimer:</b> This report is for educational and research purposes only. "
            "Trading involves substantial risk of loss. Always do your own research and "
            "consider consulting with a financial advisor before making investment decisions.",
            self.styles['CustomBody']
        )
        elements.append(Spacer(1, 10))
        elements.append(disclaimer)
        
        return elements
    
    async def generate_summary_report(self, data: Dict[str, Any]) -> str:
        """Generate a brief summary report"""
        logger.info("Generating summary report")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"summary_report_{timestamp}.txt"
            report_path = self.output_dir / report_filename
            
            # Create summary content
            recommendations = data.get('recommendations', [])
            features = data.get('features', {})
            sentiment = features.get('sentiment', {})
            
            summary_content = f"""
Enhanced Crypto Trading Pipeline - Summary Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

MARKET SENTIMENT:
- Average Sentiment: {sentiment.get('average_sentiment', 0):.2f}
- Positive Articles: {sentiment.get('positive_ratio', 0):.1%}
- Total Articles: {sentiment.get('total_articles', 0)}

TRADING RECOMMENDATIONS:
- Total Recommendations: {len(recommendations)}
- Buy Signals: {len([r for r in recommendations if r.get('action') == 'buy'])}
- Sell Signals: {len([r for r in recommendations if r.get('action') == 'sell'])}

TOP RECOMMENDATIONS:
"""
            
            for i, rec in enumerate(recommendations[:5], 1):
                summary_content += f"{i}. {rec.get('symbol', 'N/A')} - {rec.get('action', 'N/A').upper()} (Confidence: {rec.get('confidence', 0):.2f})\n"
            
            summary_content += f"""
RISK ANALYSIS:
- Expected Return: {features.get('simulation', {}).get('expected_return', 0):.2%}
- Volatility: {features.get('simulation', {}).get('volatility', 0):.2%}
- Max Drawdown: {features.get('simulation', {}).get('max_drawdown', 0):.2%}

Disclaimer: This report is for educational purposes only. Trading involves risk.
"""
            
            # Write to file
            with open(report_path, 'w') as f:
                f.write(summary_content)
            
            logger.info(f"Summary report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        generator = ReportGenerator()
        await generator.initialize()
        
        # Example report data
        report_data = {
            'timestamp': datetime.now(),
            'recommendations': [
                {'symbol': 'BTC', 'action': 'buy', 'confidence': 0.8, 'reasoning': 'Strong technical indicators'},
                {'symbol': 'ETH', 'action': 'sell', 'confidence': 0.6, 'reasoning': 'Overbought conditions'}
            ],
            'features': {
                'sentiment': {
                    'average_sentiment': 0.3,
                    'positive_ratio': 0.7,
                    'total_articles': 50,
                    'average_confidence': 0.8
                },
                'simulation': {
                    'expected_return': 0.15,
                    'volatility': 0.25,
                    'sharpe_ratio': 0.6,
                    'max_drawdown': 0.1,
                    'var_95': -0.05,
                    'cvar_95': -0.08
                }
            },
            'config': {
                'time_horizon_days': 30,
                'simulation_runs': 10000,
                'confidence_threshold': 0.6,
                'risk_tolerance': 'medium'
            }
        }
        
        # Generate reports
        daily_report = await generator.generate_daily_report(report_data)
        summary_report = await generator.generate_summary_report(report_data)
        
        print(f"Daily report: {daily_report}")
        print(f"Summary report: {summary_report}")
    
    asyncio.run(main())
