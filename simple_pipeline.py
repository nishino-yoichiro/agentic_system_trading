#!/usr/bin/env python3
"""
Simple Crypto Trading Pipeline

This pipeline focuses on analysis and reporting using existing data.
It can optionally update data using the data_manager.py system.

Usage:
    python simple_pipeline.py                    # Run analysis on existing data
    python simple_pipeline.py --update-data     # Update data first, then analyze
    python simple_pipeline.py --days 7          # Update last 7 days, then analyze
"""

import asyncio
import logging
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from data_ingestion.incremental_collector import IncrementalDataCollector
from data_ingestion.crypto_collector import CryptoDataCollector
from data_ingestion.news_collector import NewsCollector
from feature_engineering.nlp_processor import NLPProcessor
from feature_engineering.technical_indicators_ta import IndicatorCalculator
from trading_logic.signal_generator import SignalGenerator
from simulation.portfolio_simulator import PortfolioSimulator
from simulation.correlation_engine import CorrelationEngine
from trading_logic.portfolio_optimizer import PortfolioOptimizer
from reports.report_generator import ReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/simple_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SimpleCryptoPipeline:
    """Simple pipeline that focuses on analysis using existing data"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_collector = IncrementalDataCollector(self.data_dir, {})
        self.crypto_collector = CryptoDataCollector()
        self.news_collector = NewsCollector(self.data_dir)
        self.nlp_processor = NLPProcessor()
        self.indicator_calculator = IndicatorCalculator()
        self.signal_generator = SignalGenerator()
        self.portfolio_simulator = PortfolioSimulator()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.report_generator = ReportGenerator()
        
        # Get symbols from config
        self.symbols = self.config.get('assets', {}).get('crypto', []) + self.config.get('assets', {}).get('stocks', [])
        self.crypto_symbols = self.config.get('assets', {}).get('crypto', [])
        self.stock_symbols = self.config.get('assets', {}).get('stocks', [])
        
        # Data storage
        self.price_data = {}
        self.news_data = []
        self.technical_indicators = {}
        self.sentiment_data = {}
        self.simulation_result = None
        self.recommendations = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        import yaml
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    async def initialize(self):
        """Initialize pipeline components"""
        logger.info("Initializing simple pipeline...")
        
        try:
            # Initialize NLP processor
            await self.nlp_processor.initialize()
            
            # Initialize portfolio simulator
            await self.portfolio_simulator.initialize()
            
            logger.info("Pipeline initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    async def load_existing_data(self, days_back: int = 7) -> Dict[str, Any]:
        """Load existing data from parquet files"""
        logger.info(f"Loading existing data (last {days_back} days)...")
        
        try:
            # Get consolidated data from existing files
            consolidated = await self.data_collector.get_consolidated_data(self.symbols, days_back)
            
            # Extract data
            self.price_data = consolidated.get('price_data', {})
            self.news_data = consolidated.get('news_data', [])
            
            # Log data summary
            total_points = sum(len(df) for df in self.price_data.values() if isinstance(df, pd.DataFrame))
            logger.info(f"Loaded {len(self.price_data)} symbols with {total_points:,} total data points")
            logger.info(f"Loaded {len(self.news_data)} news articles")
            
            # Log data ages
            for symbol, age_info in consolidated.get('data_ages', {}).items():
                if symbol != 'news':
                    logger.info(f"{symbol}: {age_info['data_points']} points, "
                              f"{age_info['age_hours']:.1f}h old, "
                              f"fresh: {age_info['is_fresh']}")
            
            return {
                'price_symbols': len(self.price_data),
                'news_count': len(self.news_data),
                'total_data_points': total_points,
                'data_ages': consolidated.get('data_ages', {})
            }
            
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            raise
    
    async def update_data(self, days_back: int = 7) -> Dict[str, Any]:
        """Update data using the data manager system"""
        logger.info(f"Updating data (last {days_back} days)...")
        
        try:
            # Use the data manager to update crypto data
            from data_manager import CryptoDataManager
            data_manager = CryptoDataManager()
            
            # Auto-append recent data
            crypto_results = await data_manager.auto_append_recent(days_back)
            
            # Update stock data
            stock_results = {}
            if self.stock_symbols:
                stock_results = await self.data_collector.collect_incremental_price_data(self.stock_symbols)
                logger.info(f"Updated stock data for {len(stock_results)} symbols")
            
            # Reload data after update
            data_result = await self.load_existing_data(days_back)
            
            return {
                'crypto_updates': crypto_results,
                'stock_updates': stock_results,
                'data_result': data_result
            }
            
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            raise
    
    async def collect_news(self, days_back: int = 1, force_refresh: bool = False) -> Dict[str, Any]:
        """Collect news data from multiple sources with intelligent caching"""
        logger.info("Collecting news data...")
        
        try:
            # Collect news for crypto tickers (with caching)
            news_result = await self.news_collector.collect_news(self.crypto_symbols, days_back=days_back, force_refresh=force_refresh)
            
            # Convert to format expected by pipeline
            self.news_data = []
            for article in news_result['articles']:
                self.news_data.append({
                    'timestamp': article.timestamp,
                    'ticker': article.ticker,
                    'source': article.source,
                    'headline': article.headline,
                    'url': article.url,
                    'content': article.content,
                    'sentiment_score': article.sentiment_score,
                    'sentiment_label': article.sentiment_label
                })
            
            # Extract sentiment data
            self.sentiment_data = self._extract_sentiment_data(self.news_data)
            
            logger.info(f"Collected {len(self.news_data)} news articles")
            
            return {
                'news_count': len(self.news_data),
                'sources_used': news_result['stats']['sources_used'],
                'sentiment_summary': self.sentiment_data
            }
            
        except Exception as e:
            logger.error(f"Error collecting news: {e}")
            raise
    
    async def process_news(self) -> Dict[str, Any]:
        """Process news data with NLP"""
        logger.info("Processing news data...")
        
        try:
            if not self.news_data:
                logger.info("No news data to process")
                return {'processed_news': [], 'sentiment_summary': {}}
            
            # Process news with NLP
            processed_news = []
            if self.news_data and len(self.news_data) > 0:
                news_df = pd.DataFrame(self.news_data)
                if not news_df.empty:
                    processed_news_df = await self.nlp_processor.process_news_batch(news_df)
                    # Convert DataFrame back to list of dicts
                    processed_news = processed_news_df.to_dict('records')
                    self.news_data = processed_news
            
            # Extract sentiment data
            self.sentiment_data = self._extract_sentiment_data(processed_news)
            
            logger.info(f"Processed {len(processed_news)} news articles")
            
            return {
                'processed_news': processed_news,
                'sentiment_summary': self.sentiment_data
            }
            
        except Exception as e:
            logger.error(f"Error processing news: {e}")
            raise
    
    async def calculate_technical_indicators(self) -> Dict[str, Any]:
        """Calculate technical indicators for all symbols"""
        logger.info("Calculating technical indicators...")
        
        try:
            indicators = {}
            
            for symbol, df in self.price_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    try:
                        symbol_indicators = self.indicator_calculator.calculate_all_indicators(df, symbol)
                        indicators[symbol] = symbol_indicators
                        logger.info(f"Calculated indicators for {symbol}")
                    except Exception as e:
                        logger.error(f"Error calculating indicators for {symbol}: {e}")
                        indicators[symbol] = {}
            
            self.technical_indicators = indicators
            logger.info(f"Calculated indicators for {len(indicators)} symbols")
            
            return {
                'indicators': indicators,
                'symbols_processed': len(indicators)
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            raise
    
    async def run_simulations(self) -> Dict[str, Any]:
        """Run portfolio simulations"""
        logger.info("Running portfolio simulations...")
        
        try:
            # Convert price data to simulation format
            assets_data = {}
            for symbol, df in self.price_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Calculate returns
                    returns = df['close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized
                    mean_return = returns.mean() * 252  # Annualized
                    current_price = df['close'].iloc[-1]
                    
                    assets_data[symbol] = {
                        'returns': returns,
                        'volatility': volatility,
                        'mean_return': mean_return,
                        'current_price': current_price
                    }
            
            # Run portfolio simulation
            simulation_result = await self.portfolio_simulator.run_portfolio_simulation(
                assets_data=assets_data,
                num_simulations=10000
            )
            
            self.simulation_result = simulation_result
            logger.info("Portfolio simulation completed")
            
            return simulation_result
            
        except Exception as e:
            logger.error(f"Error running simulations: {e}")
            raise
    
    async def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate trading recommendations"""
        logger.info("Generating trading recommendations...")
        
        try:
            # Generate signals
            signals = await self.signal_generator.fuse_signals(
                technical_signals=self.technical_indicators,
                sentiment_data=self.sentiment_data,
                simulation_data=self.simulation_result,
                price_data=self.price_data
            )
            
            # Optimize portfolio
            recommendations = await self.portfolio_optimizer.optimize_portfolio(
                signals=signals,
                price_data=self.price_data
            )
            
            self.recommendations = recommendations
            logger.info(f"Generated {len(recommendations)} recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise
    
    async def generate_report(self) -> str:
        """Generate comprehensive report"""
        logger.info("Generating comprehensive report...")
        
        try:
            # Prepare report data
            report_data = {
                'price_data': self.price_data,
                'news_data': self.news_data,
                'technical_indicators': self.technical_indicators,
                'sentiment_data': self.sentiment_data,
                'simulation_result': self.simulation_result,
                'recommendations': self.recommendations,
                'timestamp': datetime.now()
            }
            
            # Generate report
            report_path = await self.report_generator.generate_daily_report(report_data)
            
            logger.info(f"Report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    async def generate_btc_daily_brief(self) -> Optional[str]:
        """Generate BTC-focused daily brief"""
        try:
            from reports.btc_daily_brief import BTCDailyBriefGenerator
            
            logger.info("Generating BTC Daily Brief...")
            
            # Initialize BTC brief generator
            btc_generator = BTCDailyBriefGenerator(self.data_dir)
            
            # Generate brief
            timestamp = btc_generator.generate_daily_brief(days_back=7)
            
            if timestamp:
                brief_path = f"reports/btc_briefs/btc_daily_brief_{timestamp}.html"
                logger.info(f"BTC Daily Brief generated: {brief_path}")
                return brief_path
            else:
                logger.warning("Failed to generate BTC Daily Brief")
                return None
                
        except Exception as e:
            logger.error(f"Error generating BTC Daily Brief: {e}")
            return None
    
    def _extract_sentiment_data(self, processed_news: List[Dict]) -> Dict[str, Any]:
        """Extract sentiment data from processed news"""
        if not processed_news:
            return {}
        
        # Calculate overall sentiment
        sentiments = [article.get('sentiment_score', 0) for article in processed_news]
        sentiment_scores = [article.get('sentiment_score', 0) for article in processed_news]
        
        return {
            'overall_sentiment': np.mean(sentiments) if sentiments else 0,
            'sentiment_scores': sentiment_scores,
            'positive_count': sum(1 for s in sentiment_scores if s > 0.1),
            'negative_count': sum(1 for s in sentiment_scores if s < -0.1),
            'neutral_count': sum(1 for s in sentiment_scores if -0.1 <= s <= 0.1)
        }
    
    async def run_pipeline(self, update_data: bool = False, days_back: int = 7) -> Dict[str, Any]:
        """Run the complete pipeline"""
        start_time = datetime.now()
        logger.info("Starting simple pipeline execution...")
        
        try:
            # Initialize
            await self.initialize()
            
            # Update data if requested
            if update_data:
                update_result = await self.update_data(days_back)
                logger.info("Data update completed")
            else:
                # Just load existing data
                data_result = await self.load_existing_data(days_back)
                logger.info("Data loading completed")
            
            # Collect and process news (with caching)
            news_result = await self.collect_news(days_back, force_refresh=False)
            processed_news_result = await self.process_news()
            
            # Calculate technical indicators
            indicators_result = await self.calculate_technical_indicators()
            
            # Run simulations
            simulation_result = await self.run_simulations()
            
            # Generate recommendations
            recommendations = await self.generate_recommendations()
            
            # Generate report
            report_path = await self.generate_report()
            
            # Generate BTC Daily Brief
            btc_brief_path = await self.generate_btc_daily_brief()
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'duration': duration,
                'data_result': data_result if not update_data else update_result.get('data_result', {}),
                'news_result': news_result,
                'processed_news_result': processed_news_result,
                'indicators_result': indicators_result,
                'simulation_result': simulation_result,
                'recommendations_count': len(recommendations),
                'report_path': report_path,
                'btc_brief_path': btc_brief_path,
                'timestamp': start_time.isoformat()
            }
            
            logger.info(f"Simple pipeline completed successfully in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Simple pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': (datetime.now() - start_time).total_seconds(),
                'timestamp': start_time.isoformat()
            }

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple Crypto Trading Pipeline")
    parser.add_argument('--update-data', action='store_true', 
                       help='Update data before analysis')
    parser.add_argument('--days', type=int, default=7,
                       help='Days of data to use/update')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = SimpleCryptoPipeline()
    
    # Run pipeline
    result = await pipeline.run_pipeline(
        update_data=args.update_data,
        days_back=args.days
    )
    
    # Print results
    print("\n" + "="*60)
    print("SIMPLE PIPELINE EXECUTION SUMMARY")
    print("="*60)
    print(f"Success: {result['success']}")
    print(f"Duration: {result['duration']:.2f} seconds")
    
    if result['success']:
        data_result = result.get('data_result', {})
        news_result = result.get('news_result', {})
        print(f"Price Symbols: {data_result.get('price_symbols', 0)}")
        print(f"News Articles: {news_result.get('news_count', 0)}")
        print(f"News Sources: {', '.join(news_result.get('sources_used', []))}")
        print(f"Total Data Points: {data_result.get('total_data_points', 0)}")
        print(f"Recommendations: {result.get('recommendations_count', 0)}")
        print(f"Report: {result.get('report_path', 'N/A')}")
        print(f"BTC Daily Brief: {result.get('btc_brief_path', 'N/A')}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
