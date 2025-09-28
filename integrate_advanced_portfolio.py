#!/usr/bin/env python3
"""
Integrate Advanced Portfolio System with Existing Crypto Pipeline
This script integrates the new advanced portfolio management system
with the existing crypto trading pipeline
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from advanced_portfolio_system import AdvancedPortfolioSystem
from crypto_analysis_engine import CryptoAnalysisEngine
from crypto_signal_generator import CryptoSentimentGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoPortfolioIntegration:
    """Integrate advanced portfolio system with crypto pipeline"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 target_volatility: float = 0.10,
                 max_strategy_weight: float = 0.25,
                 correlation_threshold: float = 0.5,
                 fractional_kelly: float = 0.25):
        
        # Initialize advanced portfolio system
        self.portfolio_system = AdvancedPortfolioSystem(
            initial_capital=initial_capital,
            target_volatility=target_volatility,
            max_strategy_weight=max_strategy_weight,
            correlation_threshold=correlation_threshold,
            fractional_kelly=fractional_kelly
        )
        
        # Initialize crypto components
        self.crypto_engine = CryptoAnalysisEngine()
        self.sentiment_generator = CryptoSentimentGenerator(alpha=0.5)
        
        # Available crypto symbols
        self.crypto_symbols = ['BTC', 'ETH', 'ADA', 'SOL', 'AVAX', 'DOT', 'LINK', 'MATIC', 'UNI']
        
        # Strategy mapping
        self.strategy_mapping = {
            'BTC': 'ES_Sweep_Reclaim',      # Bitcoin as futures-like
            'ETH': 'NQ_Breakout',           # Ethereum as tech breakout
            'ADA': 'SPY_Mean_Reversion',    # Cardano as mean reversion
            'SOL': 'EURUSD_Carry',          # Solana as carry trade
            'AVAX': 'Options_IV_Crush'      # Avalanche as volatility play
        }
        
    def load_crypto_data(self, symbols: list, days: int = 30) -> dict:
        """Load crypto data for specified symbols"""
        
        logger.info(f"Loading crypto data for {symbols} over {days} days")
        
        market_data = {}
        
        for symbol in symbols:
            try:
                # Load data using crypto analysis engine
                df = self.crypto_engine.load_symbol_data(symbol, days=days)
                
                if df.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Ensure we have required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Missing required columns for {symbol}")
                    continue
                
                # Map to strategy name
                strategy_name = self.strategy_mapping.get(symbol, f'{symbol}_Generic')
                
                market_data[strategy_name] = df
                logger.info(f"Loaded {len(df)} data points for {symbol} -> {strategy_name}")
                
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                continue
        
        return market_data
    
    def generate_crypto_signals(self, market_data: dict) -> dict:
        """Generate crypto-specific signals using existing pipeline"""
        
        logger.info("Generating crypto signals using existing pipeline")
        
        signals = {}
        
        for strategy_name, data in market_data.items():
            try:
                # Map back to crypto symbol
                crypto_symbol = None
                for symbol, mapped_strategy in self.strategy_mapping.items():
                    if mapped_strategy == strategy_name:
                        crypto_symbol = symbol
                        break
                
                if not crypto_symbol:
                    logger.warning(f"No crypto symbol mapping for {strategy_name}")
                    continue
                
                # Generate signals using existing crypto pipeline
                if 'BTC' in strategy_name or 'ETH' in strategy_name:
                    # Use sentiment-enhanced signals for major cryptos
                    signal_data = self.sentiment_generator.generate_enhanced_signals(
                        data, symbol=crypto_symbol
                    )
                else:
                    # Use base signals for other cryptos
                    signal_data = self.crypto_engine.generate_signals(data, crypto_symbol)
                
                # Convert to signal format expected by portfolio system
                signal_strength = signal_data.get('enhanced_signal_strength', 
                                                signal_data.get('signal_strength', 0))
                signal_type = signal_data.get('enhanced_signal_type', 
                                            signal_data.get('signal_type', 'HOLD'))
                
                # Convert signal type to numeric
                if signal_type == 'BUY':
                    signal_value = 1
                elif signal_type == 'SELL':
                    signal_value = -1
                else:
                    signal_value = 0
                
                # Create signal object
                from strategy_framework import Signal
                signal = Signal(
                    timestamp=data.index[-1],
                    signal=signal_value,
                    strength=abs(signal_strength),
                    price=data['close'].iloc[-1],
                    reason=f"Crypto signal: {signal_type}",
                    strategy_name=strategy_name
                )
                
                signals[strategy_name] = [signal]
                logger.info(f"Generated signal for {strategy_name}: {signal_type} (strength: {signal_strength:.3f})")
                
            except Exception as e:
                logger.error(f"Error generating signal for {strategy_name}: {e}")
                continue
        
        return signals
    
    def run_crypto_portfolio_simulation(self, 
                                      symbols: list, 
                                      start_date: datetime, 
                                      end_date: datetime) -> dict:
        """Run crypto portfolio simulation using advanced portfolio system"""
        
        logger.info(f"Running crypto portfolio simulation for {symbols}")
        
        # Load crypto data
        market_data = self.load_crypto_data(symbols)
        
        if not market_data:
            logger.error("No market data available for simulation")
            return {}
        
        # Generate crypto signals
        signals = self.generate_crypto_signals(market_data)
        
        # Run portfolio simulation
        results = self.portfolio_system.run_live_trading_simulation(
            market_data=market_data,
            start_date=start_date,
            end_date=end_date
        )
        
        # Add crypto-specific analysis
        results['crypto_analysis'] = self._analyze_crypto_performance(results, symbols)
        
        return results
    
    def _analyze_crypto_performance(self, results: dict, symbols: list) -> dict:
        """Analyze crypto-specific performance metrics"""
        
        crypto_analysis = {
            'symbol_performance': {},
            'correlation_analysis': {},
            'sentiment_impact': {}
        }
        
        # Analyze individual symbol performance
        for symbol in symbols:
            strategy_name = self.strategy_mapping.get(symbol)
            if strategy_name and strategy_name in results.get('strategy_performance', {}):
                perf = results['strategy_performance'][strategy_name]
                crypto_analysis['symbol_performance'][symbol] = {
                    'total_return': perf.get('total_return', 0),
                    'volatility': perf.get('volatility', 0),
                    'sharpe': perf.get('sharpe', 0),
                    'max_drawdown': perf.get('max_drawdown', 0),
                    'hit_rate': perf.get('hit_rate', 0)
                }
        
        # Analyze correlation between crypto assets
        if self.portfolio_system.portfolio_manager.correlation_matrix is not None:
            corr_matrix = self.portfolio_system.portfolio_manager.correlation_matrix
            strategy_names = list(self.portfolio_system.portfolio_manager.strategies.keys())
            
            crypto_analysis['correlation_analysis'] = {
                'matrix': corr_matrix.tolist(),
                'strategy_names': strategy_names,
                'avg_correlation': float(np.mean(np.abs(corr_matrix)))
            }
        
        return crypto_analysis
    
    def generate_crypto_report(self, results: dict, output_dir: str = "crypto_portfolio_reports"):
        """Generate crypto-specific portfolio report"""
        
        logger.info(f"Generating crypto portfolio report in {output_dir}")
        
        # Generate standard portfolio report
        self.portfolio_system.generate_comprehensive_report(results, output_dir)
        
        # Add crypto-specific analysis
        if 'crypto_analysis' in results:
            self._plot_crypto_performance(results['crypto_analysis'], output_dir)
            self._save_crypto_analysis(results['crypto_analysis'], output_dir)
        
        logger.info("Crypto portfolio report generated successfully")
    
    def _plot_crypto_performance(self, crypto_analysis: dict, output_dir: str):
        """Plot crypto-specific performance charts"""
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Plot symbol performance
        if 'symbol_performance' in crypto_analysis:
            symbol_perf = crypto_analysis['symbol_performance']
            
            if symbol_perf:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Extract metrics
                symbols = list(symbol_perf.keys())
                returns = [symbol_perf[s]['total_return'] for s in symbols]
                volatilities = [symbol_perf[s]['volatility'] for s in symbols]
                sharpes = [symbol_perf[s]['sharpe'] for s in symbols]
                drawdowns = [symbol_perf[s]['max_drawdown'] for s in symbols]
                
                # Plot returns
                axes[0,0].bar(symbols, returns, color='skyblue')
                axes[0,0].set_title('Crypto Returns by Symbol')
                axes[0,0].set_ylabel('Total Return')
                
                # Plot volatilities
                axes[0,1].bar(symbols, volatilities, color='lightcoral')
                axes[0,1].set_title('Crypto Volatilities by Symbol')
                axes[0,1].set_ylabel('Volatility')
                
                # Plot Sharpe ratios
                axes[1,0].bar(symbols, sharpes, color='lightgreen')
                axes[1,0].set_title('Crypto Sharpe Ratios by Symbol')
                axes[1,0].set_ylabel('Sharpe Ratio')
                
                # Plot drawdowns
                axes[1,1].bar(symbols, drawdowns, color='gold')
                axes[1,1].set_title('Crypto Max Drawdowns by Symbol')
                axes[1,1].set_ylabel('Max Drawdown')
                
                plt.tight_layout()
                plt.savefig(output_path / 'crypto_performance.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Plot correlation matrix
        if 'correlation_analysis' in crypto_analysis:
            corr_data = crypto_analysis['correlation_analysis']
            if 'matrix' in corr_data and 'strategy_names' in corr_data:
                corr_matrix = np.array(corr_data['matrix'])
                strategy_names = corr_data['strategy_names']
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, 
                           annot=True, 
                           cmap='RdBu_r', 
                           center=0,
                           square=True,
                           xticklabels=strategy_names,
                           yticklabels=strategy_names)
                plt.title('Crypto Strategy Correlation Matrix')
                plt.tight_layout()
                plt.savefig(output_path / 'crypto_correlation.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _save_crypto_analysis(self, crypto_analysis: dict, output_dir: str):
        """Save crypto analysis to JSON"""
        
        import json
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / 'crypto_analysis.json', 'w') as f:
            json.dump(crypto_analysis, f, indent=2, default=str)

def main():
    """Main function to run crypto portfolio integration"""
    
    logger.info("Starting Crypto Portfolio Integration Demo")
    
    # Configuration
    initial_capital = 100000
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    crypto_symbols = ['BTC', 'ETH', 'ADA', 'SOL', 'AVAX']
    
    # Create crypto portfolio integration
    crypto_portfolio = CryptoPortfolioIntegration(
        initial_capital=initial_capital,
        target_volatility=0.10,
        max_strategy_weight=0.25,
        correlation_threshold=0.5,
        fractional_kelly=0.25
    )
    
    # Run simulation
    logger.info("Running crypto portfolio simulation...")
    results = crypto_portfolio.run_crypto_portfolio_simulation(
        symbols=crypto_symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    # Generate report
    logger.info("Generating crypto portfolio report...")
    crypto_portfolio.generate_crypto_report(results, "crypto_portfolio_reports")
    
    # Print summary
    print("\n" + "="*60)
    print("CRYPTO PORTFOLIO INTEGRATION RESULTS")
    print("="*60)
    
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        print(f"\nPortfolio Performance:")
        print(f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2f}")
        print(f"  Final Capital: ${metrics.get('final_capital', 0):,.2f}")
    
    if 'crypto_analysis' in results:
        crypto_analysis = results['crypto_analysis']
        if 'symbol_performance' in crypto_analysis:
            print(f"\nCrypto Symbol Performance:")
            for symbol, perf in crypto_analysis['symbol_performance'].items():
                print(f"  {symbol}: {perf.get('total_return', 0):.2f} return, {perf.get('sharpe', 0):.2f} Sharpe")
    
    print(f"\nReport generated in 'crypto_portfolio_reports' directory")
    print("="*60)
    
    logger.info("Crypto portfolio integration completed successfully!")

if __name__ == "__main__":
    main()
