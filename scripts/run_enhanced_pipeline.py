"""
Enhanced Crypto Trading Pipeline with Incremental Data Collection
Uses bulk historical data + real-time incremental updates
"""

import asyncio
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_ingestion.incremental_collector import IncrementalDataCollector, DataType, RefreshStrategy
from data_ingestion.continuous_collector import ContinuousDataCollector
from data_ingestion.crypto_collector import CryptoDataCollector
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
        logging.FileHandler('logs/enhanced_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EnhancedCryptoPipeline:
    """Enhanced pipeline with incremental data collection"""
    
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
        (self.data_dir / 'incremental').mkdir(exist_ok=True)
        
        # Load API keys
        self.api_keys = self._load_api_keys()
        
        # Initialize incremental collector
        self.data_collector = IncrementalDataCollector(self.data_dir, self.api_keys)
        
        # Initialize crypto collector (Coinbase Advanced)
        self.crypto_collector = CryptoDataCollector(self.api_keys)
        
        # Initialize components
        self.nlp_processor = None
        self.indicator_calculator = IndicatorCalculator()
        self.signal_generator = SignalGenerator()
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
        
        # Symbols to track
        self.symbols = self.config.get('symbols', [
            'BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI',
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'BAC', 'XOM'
        ])
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from local file"""
        try:
            import yaml
            api_keys_file = Path("config/api_keys_local.yaml")
            if api_keys_file.exists():
                with open(api_keys_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning("API keys file not found, using empty keys")
                return {}
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            return {}
    
    async def initialize(self):
        """Initialize pipeline components"""
        logger.info("Initializing enhanced pipeline...")
        
        try:
            # Initialize NLP processor
            self.nlp_processor = NLPProcessor()
            await self.nlp_processor.initialize()
            
            # Initialize portfolio simulator
            self.portfolio_simulator = PortfolioSimulator()
            await self.portfolio_simulator.initialize()
            
            # Initialize signal fusion (this is a dataclass, not instantiated)
            self.signal_fusion = None
            
            # Initialize portfolio optimizer
            self.portfolio_optimizer = PortfolioOptimizer()
            
            # Initialize report generator
            self.report_generator = ReportGenerator()
            
            logger.info("Pipeline initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    async def setup_historical_data(self, days_back: int = 365) -> Dict[str, Any]:
        """Setup initial historical data (one-time bulk collection)"""
        logger.info(f"Setting up historical data for {days_back} days...")
        
        try:
            # Check if historical data already exists
            metadata_file = self.data_dir / 'collection_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                last_bulk = metadata.get('last_bulk_collection', {})
                if last_bulk and last_bulk.get('days_back', 0) >= days_back:
                    logger.info("Historical data already exists, skipping bulk collection")
                    return {'status': 'skipped', 'reason': 'data_exists'}
            
            # Collect bulk historical data
            result = await self.data_collector.collect_bulk_historical_data(self.symbols, days_back)
            
            logger.info(f"Historical data setup completed: {result['api_calls_made']} API calls made")
            return result
            
        except Exception as e:
            logger.error(f"Error setting up historical data: {e}")
            raise
    
    async def collect_crypto_data(self) -> Dict[str, Any]:
        """Collect 30 days of minute-level crypto data using Coinbase Advanced API"""
        logger.info("Collecting 30 days of minute-level crypto data from Coinbase Advanced...")
        
        try:
            # Get crypto symbols (first 10 symbols that are crypto)
            crypto_symbols = [s for s in self.symbols if s in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']]
            
            if not crypto_symbols:
                logger.warning("No crypto symbols found in symbol list")
                return {'crypto_data': {}, 'crypto_count': 0}
            
            # Get configuration
            days_back = self.config.get('data_collection', {}).get('price_lookback_days', 30)
            granularity = self.config.get('data_collection', {}).get('crypto_granularity', 60)
            
            logger.info(f"Collecting {days_back} days of {granularity}s granularity data for {len(crypto_symbols)} crypto symbols")
            
            # Collect crypto data with minute-level granularity
            crypto_data = await self.crypto_collector.collect_crypto_data(
                symbols=crypto_symbols, 
                days_back=days_back
            )
            
            # Save crypto data
            await self.crypto_collector.save_crypto_data(crypto_data, self.data_dir)
            
            # Get summary
            summary = await self.crypto_collector.get_crypto_summary(crypto_symbols)
            
            # Calculate total data points collected
            total_data_points = sum(len(df) for df in crypto_data.values())
            
            logger.info(f"Collected {total_data_points} total minute-level data points for {len(crypto_data)} symbols")
            
            return {
                'crypto_data': crypto_data,
                'crypto_count': len(crypto_data),
                'crypto_summary': summary,
                'total_data_points': total_data_points,
                'granularity_seconds': granularity,
                'days_collected': days_back,
                'collection_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error collecting crypto data: {e}")
            raise

    async def collect_incremental_data(self) -> Dict[str, Any]:
        """Collect incremental data (real-time updates)"""
        logger.info("Collecting incremental data...")
        
        try:
            # Collect crypto data first (using Coinbase)
            crypto_result = await self.collect_crypto_data()
            
            # Collect incremental price data for stocks
            stock_symbols = [s for s in self.symbols if s not in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']]
            price_result = await self.data_collector.collect_incremental_price_data(stock_symbols)
            
            # Collect news data (daily refresh)
            news_result = await self.data_collector.collect_news_data(24)
            
            return {
                'crypto_updates': crypto_result,
                'price_updates': price_result,
                'news_updates': news_result,
                'collection_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error collecting incremental data: {e}")
            raise
    
    async def get_consolidated_data(self) -> Dict[str, Any]:
        """Get consolidated data for analysis"""
        logger.info("Getting consolidated data...")
        
        try:
            # Get days_back from config
            days_back = self.config.get('data_collection', {}).get('price_lookback_days', 7)
            
            # Collect fresh crypto data first
            logger.info("Collecting fresh crypto data...")
            crypto_result = await self.collect_crypto_data()
            
            # Collect stock data
            logger.info("Collecting stock data...")
            stock_symbols = [s for s in self.symbols if s not in ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']]
            if stock_symbols:
                stock_result = await self.data_collector.collect_incremental_price_data(stock_symbols)
                logger.info(f"Collected stock data for {len(stock_symbols)} symbols")
            
            # Get consolidated data (includes both crypto and stock data)
            consolidated = await self.data_collector.get_consolidated_data(self.symbols, days_back)
            
            # Convert to pipeline format
            self.news_data = consolidated.get('news_data', [])
            self.price_data = consolidated.get('price_data', {})
            
            # Log data ages
            for symbol, age_info in consolidated.get('data_ages', {}).items():
                if symbol != 'news':
                    logger.info(f"{symbol}: {age_info['data_points']} points, "
                              f"{age_info['age_hours']:.1f}h old, "
                              f"fresh: {age_info['is_fresh']}")
            
            return {
                'news_count': len(self.news_data),
                'price_symbols': len(self.price_data),
                'total_data_points': consolidated.get('total_data_points', 0),
                'data_ages': consolidated.get('data_ages', {}),
                'collection_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting consolidated data: {e}")
            raise
    
    async def engineer_features(self) -> Dict[str, Any]:
        """Engineer features from consolidated data"""
        logger.info("Engineering features...")
        
        try:
            # Process news with NLP
            if self.news_data and self.nlp_processor:
                logger.info(f"Processing {len(self.news_data)} news articles with NLP...")
                processed_news = await self.nlp_processor.process_news_batch(pd.DataFrame(self.news_data))
                self.news_data = processed_news
                logger.info(f"Processed {len(processed_news)} articles with NLP")
            
            # Calculate technical indicators (if enabled)
            ta_enabled = self.config.get('feature_engineering', {}).get('technical_indicators', {}).get('enabled', True)
            
            if ta_enabled:
                logger.info("Calculating technical indicators...")
                features = {}
                for symbol, df in self.price_data.items():
                    if not df.empty:
                        # For crypto data with single point, skip technical indicators
                        if len(df) < 20:
                            logger.info(f"Skipping technical indicators for {symbol}: insufficient data ({len(df)} points)")
                            # Store basic price info for crypto
                            features[symbol] = {
                                'current_price': df['close'].iloc[-1] if 'close' in df.columns else None,
                                'data_points': len(df),
                                'is_crypto': symbol in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']
                            }
                        else:
                            features[symbol] = self.indicator_calculator.calculate_all_indicators(df, symbol)
                            logger.info(f"Calculated indicators for {symbol}: technical indicators complete")
            else:
                logger.info("Technical analysis disabled - skipping indicator calculation for faster execution")
                features = {}
                for symbol, df in self.price_data.items():
                    if not df.empty:
                        features[symbol] = {
                            'current_price': df['close'].iloc[-1] if 'close' in df.columns else None,
                            'data_points': len(df),
                            'is_crypto': symbol in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']
                        }
            
            self.features = features
            
            # Save processed features
            features_file = self.data_dir / 'processed' / 'features.json'
            with open(features_file, 'w') as f:
                json.dump({k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in features.items()}, f, indent=2, default=str)
            
            logger.info("Feature engineering completed")
            return {'features_calculated': len(features)}
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            raise
    
    async def run_simulations(self) -> Dict[str, Any]:
        """Run Monte Carlo simulations"""
        logger.info("Running Monte Carlo simulations...")
        
        try:
            # Prepare data for simulation
            price_data = {}
            for symbol, df in self.price_data.items():
                if not df.empty and 'close' in df.columns:
                    price_data[symbol] = df['close'].values
            
            if not price_data:
                logger.warning("No price data available for simulation")
                return {'simulation_runs': 0}
            
            # Run portfolio simulation
            simulation_runs = self.config.get('simulation', {}).get('runs', 10000)
            logger.info(f"Running portfolio simulation with {simulation_runs} scenarios")
            
            # Calculate correlations (need returns data, not price data)
            correlation_engine = CorrelationEngine()
            returns_data = {}
            for symbol, prices in price_data.items():
                if len(prices) > 1:
                    returns = pd.Series(prices).pct_change().dropna()
                    returns_data[symbol] = returns
            
            correlations = await correlation_engine.calculate_correlation_matrix(returns_data)
            
            # Convert price data to the format expected by portfolio simulator
            assets_data = {}
            for symbol, prices in price_data.items():
                if len(prices) > 1:
                    # Calculate returns from price data
                    returns = pd.Series(prices).pct_change().dropna()
                    
                    # Calculate basic statistics
                    mean_return = returns.mean() * 252  # Annualized
                    volatility = returns.std() * np.sqrt(252)  # Annualized
                    current_price = prices[-1]
                    
                    assets_data[symbol] = {
                        'returns': returns,
                        'volatility': volatility,
                        'mean_return': mean_return,
                        'current_price': current_price
                    }
            
            # Run simulation
            simulation_result = await self.portfolio_simulator.run_portfolio_simulation(
                assets_data=assets_data,
                num_simulations=simulation_runs
            )
            
            logger.info("Monte Carlo simulations completed")
            return simulation_result
            
        except Exception as e:
            logger.error(f"Error running simulations: {e}")
            raise
    
    async def generate_recommendations(self, simulation_result: Any = None) -> List[Dict[str, Any]]:
        """Generate trading recommendations"""
        logger.info("Generating trading recommendations...")
        
        try:
            # Generate technical signals (using features as technical signals)
            technical_signals = {}
            for symbol, features in self.features.items():
                if isinstance(features, dict) and 'rsi' in features:
                    technical_signals[symbol] = features
            
            # Fuse signals using the signal generator
            fused_signals = await self.signal_generator.fuse_signals(
                technical_signals=technical_signals,
                sentiment_data={},  # No sentiment data for now
                simulation_data=simulation_result.__dict__ if hasattr(simulation_result, '__dict__') else {},
                price_data=self.price_data
            )
            
            # Optimize portfolio
            if self.portfolio_optimizer:
                recommendations = await self.portfolio_optimizer.optimize_portfolio(fused_signals, self.price_data)
            else:
                recommendations = []
            
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
            if not self.report_generator:
                self.report_generator = ReportGenerator()
            
            # Generate report
            report_data = {
                'price_data': self.price_data,
                'news_data': self.news_data,
                'features': self.features,
                'recommendations': self.recommendations
            }
            report_path = await self.report_generator.generate_daily_report(
                report_data=report_data,
                output_dir=self.reports_dir
            )
            
            logger.info(f"Report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    async def run_enhanced_pipeline(self, setup_historical: bool = False, days_back: int = 365) -> Dict[str, Any]:
        """Run the enhanced pipeline with incremental data collection"""
        logger.info("Starting enhanced pipeline execution...")
        
        start_time = datetime.now()
        
        try:
            # Initialize
            await self.initialize()
            
            # Setup historical data if requested
            if setup_historical:
                await self.setup_historical_data(days_back)
            
            # Get consolidated data
            data_result = await self.get_consolidated_data()
            
            # Engineer features
            features_result = await self.engineer_features()
            
            # Run simulations
            simulation_result = await self.run_simulations()
            
            # Generate recommendations
            recommendations = await self.generate_recommendations(simulation_result)
            
            # Generate report
            report_path = await self.generate_report()
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'duration': duration,
                'data_result': data_result,
                'features_result': features_result,
                'simulation_result': simulation_result,
                'recommendations_count': len(recommendations),
                'report_path': report_path,
                'timestamp': start_time.isoformat()
            }
            
            logger.info(f"Enhanced pipeline completed successfully in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': (datetime.now() - start_time).total_seconds(),
                'timestamp': start_time.isoformat()
            }
    
    async def start_continuous_collection(self, interval_seconds: int = 20):
        """Start continuous data collection in background"""
        logger.info(f"Starting continuous collection every {interval_seconds} seconds...")
        
        collector = ContinuousDataCollector(self.data_dir, self.api_keys, self.symbols)
        await collector.start_continuous_collection(interval_seconds)

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced Crypto Trading Pipeline")
    parser.add_argument('--setup-historical', action='store_true', 
                       help='Setup historical data (one-time bulk collection)')
    parser.add_argument('--days-back', type=int, default=365,
                       help='Days of historical data to collect')
    parser.add_argument('--continuous', action='store_true',
                       help='Start continuous data collection')
    parser.add_argument('--interval', type=int, default=20,
                       help='Continuous collection interval in seconds')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Target specific ticker (e.g., BTC, ETH). If not specified, uses all configured symbols')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = EnhancedCryptoPipeline()
    
    # Override symbols if ticker is specified
    if args.ticker:
        pipeline.symbols = [args.ticker.upper()]
        logger.info(f"Targeting specific ticker: {args.ticker.upper()}")
    
    if args.continuous:
        # Start continuous collection
        await pipeline.start_continuous_collection(args.interval)
    else:
        # Run enhanced pipeline
        result = await pipeline.run_enhanced_pipeline(
            setup_historical=args.setup_historical,
            days_back=args.days_back
        )
        
        # Print results
        print("\n" + "="*60)
        print("ENHANCED PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Success: {result['success']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        
        if result['success']:
            data_result = result.get('data_result', {})
            print(f"News Articles: {data_result.get('news_count', 0)}")
            print(f"Price Symbols: {data_result.get('price_symbols', 0)}")
            print(f"Total Data Points: {data_result.get('total_data_points', 0)}")
            print(f"Recommendations: {result.get('recommendations_count', 0)}")
            print(f"Report: {result.get('report_path', 'N/A')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        print("="*60)

if __name__ == "__main__":
    asyncio.run(main())

