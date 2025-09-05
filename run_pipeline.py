#!/usr/bin/env python3
"""
Enhanced Crypto Trading Pipeline - Main Orchestrator

This is the main entry point for the enhanced crypto trading pipeline.
It orchestrates all components to collect data, engineer features,
run simulations, and generate trading recommendations.

Usage:
    python run_pipeline.py [--config CONFIG_FILE] [--mode MODE] [--output OUTPUT_DIR]

Modes:
    - full: Run complete pipeline (data collection + analysis + recommendations)
    - data: Only collect and store data
    - analysis: Only run analysis on existing data
    - recommendations: Only generate recommendations
    - backtest: Run backtesting on historical data
"""

import asyncio
import argparse
import yaml
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import pipeline components
from data_ingestion.news_apis import collect_crypto_news, collect_stock_news
from data_ingestion.price_apis import collect_price_data
from feature_engineering.nlp_processor import NLPProcessor
from feature_engineering.technical_indicators import IndicatorCalculator, TechnicalSignalGenerator
from simulation.portfolio_simulator import PortfolioSimulator
from trading_logic.signal_generator import SignalGenerator
from trading_logic.portfolio_optimizer import PortfolioOptimizer
from reports.report_generator import ReportGenerator


class EnhancedCryptoPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.reports_dir = Path(self.config.get('reports_dir', 'reports'))
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        (self.data_dir / 'raw').mkdir(exist_ok=True)
        (self.data_dir / 'processed').mkdir(exist_ok=True)
        
        # Initialize components
        self.nlp_processor = None
        self.indicator_calculator = IndicatorCalculator()
        self.signal_generator = TechnicalSignalGenerator()
        self.portfolio_simulator = None
        self.signal_fusion = None
        self.portfolio_optimizer = None
        self.report_generator = None
        
        # Data storage
        self.news_data = []
        self.price_data = {}
        self.features = {}
        self.signals = {}
        self.recommendations = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'data_dir': 'data',
            'reports_dir': 'reports',
            'assets': {
                'crypto': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL'],
                'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            },
            'time_horizon_days': 30,
            'simulation_runs': 10000,
            'confidence_threshold': 0.6,
            'max_positions': 10,
            'risk_tolerance': 'medium'
        }
    
    async def initialize(self):
        """Initialize all pipeline components"""
        logger.info("Initializing Enhanced Crypto Pipeline...")
        
        try:
            # Initialize NLP processor
            self.nlp_processor = NLPProcessor()
            await self.nlp_processor.initialize()
            
            # Initialize portfolio simulator
            self.portfolio_simulator = PortfolioSimulator()
            await self.portfolio_simulator.initialize()
            
            # Initialize signal fusion
            self.signal_fusion = SignalGenerator()
            await self.signal_fusion.initialize()
            
            # Initialize portfolio optimizer
            self.portfolio_optimizer = PortfolioOptimizer()
            await self.portfolio_optimizer.initialize()
            
            # Initialize report generator
            self.report_generator = ReportGenerator()
            await self.report_generator.initialize()
            
            logger.info("Pipeline initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    async def collect_data(self, hours_back: int = 24) -> Dict[str, Any]:
        """Collect data from all sources"""
        logger.info(f"Collecting data for the last {hours_back} hours...")
        
        try:
            # Load API keys
            api_keys = self._load_api_keys()
            
            # Collect news data
            logger.info("Collecting news data...")
            crypto_news = []
            stock_news = []
            
            if api_keys.get('newsapi'):
                crypto_news = await collect_crypto_news(
                    api_keys['newsapi'], 
                    hours_back=hours_back,
                    max_articles=1000
                )
                stock_news = await collect_stock_news(
                    api_keys['newsapi'],
                    hours_back=hours_back,
                    max_articles=1000
                )
            
            self.news_data = crypto_news + stock_news
            logger.info(f"Collected {len(self.news_data)} news articles")
            
            # Collect price data
            logger.info("Collecting price data...")
            all_symbols = (self.config['assets']['crypto'] + 
                          self.config['assets']['stocks'])
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Get 30 days of historical data
            
            self.price_data = await collect_price_data(
                all_symbols,
                start_date,
                end_date,
                polygon_key=api_keys.get('polygon'),
                binance_key=api_keys.get('binance'),
                coingecko_key=api_keys.get('coingecko')
            )
            
            logger.info(f"Collected price data for {len(self.price_data)} symbols")
            
            # Save raw data
            await self._save_raw_data()
            
            return {
                'news_count': len(self.news_data),
                'price_symbols': len(self.price_data),
                'collection_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            raise
    
    async def engineer_features(self) -> Dict[str, Any]:
        """Engineer features from collected data"""
        logger.info("Engineering features...")
        
        try:
            # Process news with NLP
            if self.news_data and self.nlp_processor:
                logger.info("Processing news with NLP...")
                news_results = await self.nlp_processor.process_articles(self.news_data)
                
                # Calculate sentiment metrics
                sentiment_metrics = self.nlp_processor.calculate_sentiment_metrics(news_results)
                self.features['sentiment'] = sentiment_metrics
                
                logger.info(f"Processed {len(news_results)} articles with NLP")
            
            # Calculate technical indicators
            logger.info("Calculating technical indicators...")
            technical_features = {}
            
            for symbol, price_df in self.price_data.items():
                if len(price_df) > 20:  # Need minimum data for indicators
                    indicators = self.indicator_calculator.calculate_all_indicators(price_df)
                    technical_features[symbol] = indicators
                    
                    # Generate technical signals
                    signals = self.signal_generator.generate_signals(price_df)
                    self.signals[symbol] = signals
            
            self.features['technical'] = technical_features
            logger.info(f"Calculated technical indicators for {len(technical_features)} symbols")
            
            # Save processed features
            await self._save_processed_features()
            
            return {
                'sentiment_features': len(self.features.get('sentiment', {})),
                'technical_symbols': len(technical_features),
                'signals_generated': len(self.signals)
            }
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            raise
    
    async def run_simulations(self) -> Dict[str, Any]:
        """Run Monte Carlo simulations"""
        logger.info("Running Monte Carlo simulations...")
        
        try:
            if not self.portfolio_simulator:
                raise RuntimeError("Portfolio simulator not initialized")
            
            # Prepare simulation data
            simulation_data = {}
            for symbol, price_df in self.price_data.items():
                if len(price_df) > 20:
                    # Calculate returns
                    returns = price_df['close'].pct_change().dropna()
                    
                    simulation_data[symbol] = {
                        'returns': returns,
                        'current_price': price_df['close'].iloc[-1],
                        'volatility': returns.std() * np.sqrt(252),  # Annualized
                        'mean_return': returns.mean() * 252  # Annualized
                    }
            
            # Run portfolio simulations
            simulation_results = await self.portfolio_simulator.run_portfolio_simulation(
                simulation_data,
                time_horizon=self.config.get('time_horizon_days', 30),
                num_simulations=self.config.get('simulation_runs', 10000),
                risk_tolerance=self.config.get('risk_tolerance', 'medium')
            )
            
            self.features['simulation'] = simulation_results
            logger.info("Monte Carlo simulations completed")
            
            return {
                'simulation_completed': True,
                'portfolio_scenarios': len(simulation_results.get('scenarios', [])),
                'expected_return': simulation_results.get('expected_return', 0),
                'volatility': simulation_results.get('volatility', 0)
            }
            
        except Exception as e:
            logger.error(f"Error running simulations: {e}")
            raise
    
    async def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate trading recommendations"""
        logger.info("Generating trading recommendations...")
        
        try:
            if not self.signal_fusion or not self.portfolio_optimizer:
                raise RuntimeError("Signal fusion or portfolio optimizer not initialized")
            
            # Fuse all signals
            fused_signals = await self.signal_fusion.fuse_signals(
                self.signals,
                self.features.get('sentiment', {}),
                self.features.get('simulation', {})
            )
            
            # Generate portfolio recommendations
            portfolio_recs = await self.portfolio_optimizer.optimize_portfolio(
                fused_signals,
                self.price_data,
                max_positions=self.config.get('max_positions', 10),
                confidence_threshold=self.config.get('confidence_threshold', 0.6)
            )
            
            self.recommendations = portfolio_recs
            logger.info(f"Generated {len(self.recommendations)} recommendations")
            
            return self.recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise
    
    async def generate_report(self) -> str:
        """Generate comprehensive report"""
        logger.info("Generating comprehensive report...")
        
        try:
            if not self.report_generator:
                raise RuntimeError("Report generator not initialized")
            
            # Prepare report data
            report_data = {
                'timestamp': datetime.now(),
                'news_data': self.news_data,
                'price_data': self.price_data,
                'features': self.features,
                'signals': self.signals,
                'recommendations': self.recommendations,
                'config': self.config
            }
            
            # Generate report
            report_path = await self.report_generator.generate_daily_report(
                report_data,
                output_dir=self.reports_dir
            )
            
            logger.info(f"Report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    async def run_full_pipeline(self, hours_back: int = 24) -> Dict[str, Any]:
        """Run the complete pipeline"""
        logger.info("Starting full pipeline execution...")
        
        start_time = datetime.now()
        
        try:
            # Initialize
            await self.initialize()
            
            # Collect data
            data_results = await self.collect_data(hours_back)
            
            # Engineer features
            feature_results = await self.engineer_features()
            
            # Run simulations
            simulation_results = await self.run_simulations()
            
            # Generate recommendations
            recommendations = await self.generate_recommendations()
            
            # Generate report
            report_path = await self.generate_report()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results = {
                'success': True,
                'duration_seconds': duration,
                'data_collection': data_results,
                'feature_engineering': feature_results,
                'simulation': simulation_results,
                'recommendations_count': len(recommendations),
                'report_path': report_path,
                'timestamp': end_time
            }
            
            logger.info(f"Pipeline completed successfully in {duration:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment or config"""
        api_keys = {}
        
        # Try to load from local config file
        local_config_path = "config/api_keys_local.yaml"
        if os.path.exists(local_config_path):
            try:
                with open(local_config_path, 'r') as f:
                    local_config = yaml.safe_load(f)
                    api_keys.update(local_config)
            except Exception as e:
                logger.warning(f"Error loading local API keys: {e}")
        
        # Load from environment variables
        env_keys = {
            'newsapi': os.getenv('NEWSAPI_KEY'),
            'polygon': os.getenv('POLYGON_API_KEY'),
            'binance': os.getenv('BINANCE_API_KEY'),
            'coingecko': os.getenv('COINGECKO_API_KEY'),
            'alpaca_key': os.getenv('ALPACA_API_KEY'),
            'alpaca_secret': os.getenv('ALPACA_SECRET_KEY')
        }
        
        for key, value in env_keys.items():
            if value:
                api_keys[key] = value
        
        return api_keys
    
    async def _save_raw_data(self):
        """Save raw collected data"""
        try:
            # Save news data
            if self.news_data:
                news_df = pd.DataFrame([
                    {
                        'id': article.id,
                        'title': article.title,
                        'content': article.content,
                        'url': article.url,
                        'source': article.source,
                        'published_at': article.published_at,
                        'category': article.category
                    }
                    for article in self.news_data
                ])
                news_df.to_parquet(self.data_dir / 'raw' / 'news.parquet', index=False)
            
            # Save price data
            for symbol, price_df in self.price_data.items():
                price_df.to_parquet(self.data_dir / 'raw' / f'prices_{symbol}.parquet')
            
            logger.info("Raw data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving raw data: {e}")
    
    async def _save_processed_features(self):
        """Save processed features"""
        try:
            # Save features as JSON (simplified)
            import json
            
            features_json = {}
            for key, value in self.features.items():
                if isinstance(value, dict):
                    features_json[key] = value
                else:
                    features_json[key] = str(value)
            
            with open(self.data_dir / 'processed' / 'features.json', 'w') as f:
                json.dump(features_json, f, indent=2, default=str)
            
            logger.info("Processed features saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving processed features: {e}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced Crypto Trading Pipeline')
    parser.add_argument('--config', default='config/pipeline_config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--mode', default='full', 
                       choices=['full', 'data', 'analysis', 'recommendations', 'backtest'],
                       help='Pipeline mode')
    parser.add_argument('--hours-back', type=int, default=24,
                       help='Hours of data to collect')
    parser.add_argument('--output', default='reports',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add("logs/pipeline.log", level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize pipeline
    pipeline = EnhancedCryptoPipeline(args.config)
    
    try:
        if args.mode == 'full':
            results = await pipeline.run_full_pipeline(args.hours_back)
            print(f"\n{'='*60}")
            print("PIPELINE EXECUTION SUMMARY")
            print(f"{'='*60}")
            print(f"Success: {results['success']}")
            print(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
            print(f"News Articles: {results.get('data_collection', {}).get('news_count', 0)}")
            print(f"Price Symbols: {results.get('data_collection', {}).get('price_symbols', 0)}")
            print(f"Recommendations: {results.get('recommendations_count', 0)}")
            print(f"Report: {results.get('report_path', 'N/A')}")
            print(f"{'='*60}")
            
        elif args.mode == 'data':
            await pipeline.initialize()
            results = await pipeline.collect_data(args.hours_back)
            print(f"Data collection completed: {results}")
            
        elif args.mode == 'analysis':
            await pipeline.initialize()
            # Load existing data and run analysis
            print("Analysis mode - would load existing data and run analysis")
            
        elif args.mode == 'recommendations':
            await pipeline.initialize()
            # Load existing data and generate recommendations
            print("Recommendations mode - would load existing data and generate recommendations")
            
        elif args.mode == 'backtest':
            print("Backtest mode - would run backtesting on historical data")
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
