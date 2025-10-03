#!/usr/bin/env python3
"""
Run Advanced Portfolio Management System
Example script demonstrating the complete system
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from advanced_portfolio_system import AdvancedPortfolioSystem

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(start_date: datetime, end_date: datetime, symbols: list) -> dict:
    """Generate sample market data for testing"""
    
    logger.info("Generating sample market data...")
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    market_data = {}
    
    for symbol in symbols:
        # Generate realistic price data
        np.random.seed(hash(symbol) % 2**32)  # Different seed for each symbol
        
        # Generate price series with trend and volatility
        n_days = len(date_range)
        
        # Base price
        base_price = 100 if 'SPY' in symbol else 50
        
        # Generate returns with some autocorrelation
        returns = np.random.normal(0.0005, 0.02, n_days)  # 0.05% daily return, 2% volatility
        
        # Add some trend
        trend = np.linspace(0, 0.3, n_days)  # 30% total trend over period
        returns += trend / n_days
        
        # Add some volatility clustering
        for i in range(1, n_days):
            if abs(returns[i-1]) > 0.03:  # High volatility day
                returns[i] *= 1.5  # Increase next day volatility
        
        # Calculate prices
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLCV data
        data = pd.DataFrame(index=date_range)
        data['open'] = prices
        data['high'] = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        data['low'] = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        data['close'] = prices
        data['volume'] = np.random.lognormal(10, 0.5, n_days).astype(int)
        
        # Ensure high >= low
        data['high'] = np.maximum(data['high'], data['low'])
        data['low'] = np.minimum(data['high'], data['low'])
        
        market_data[symbol] = data
        
        logger.info(f"Generated {len(data)} days of data for {symbol}")
    
    return market_data

def main():
    """Main function to run the advanced portfolio system"""
    
    logger.info("Starting Advanced Portfolio Management System Demo")
    
    # Configuration
    initial_capital = 100000
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Symbols for different strategies (matching strategy framework names)
    symbols = [
        'Sweep_Reclaim',      # For sweep/reclaim strategy
        'Breakout_Continuation',      # For breakout strategy
        'Mean_Reversion',             # For mean reversion strategy
        'FX_Carry_Trend',          # For FX carry strategy
        'Options_IV_Crush'              # For options IV strategy
    ]
    
    # Generate sample data
    market_data = generate_sample_data(start_date, end_date, symbols)
    
    # Create advanced portfolio system
    system = AdvancedPortfolioSystem(
        initial_capital=initial_capital,
        target_volatility=0.10,
        max_strategy_weight=0.25,
        correlation_threshold=0.5,
        fractional_kelly=0.25
    )
    
    logger.info("Running live trading simulation...")
    
    # Run simulation
    results = system.run_live_trading_simulation(
        market_data=market_data,
        start_date=start_date,
        end_date=end_date
    )
    
    # Generate comprehensive report
    logger.info("Generating comprehensive report...")
    system.generate_comprehensive_report(results, "portfolio_reports")
    
    # Print summary
    print("\n" + "="*60)
    print("ADVANCED PORTFOLIO MANAGEMENT SYSTEM RESULTS")
    print("="*60)
    
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        print(f"\nPortfolio Performance:")
        print(f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"  Annualized Return: {metrics.get('annualized_return', 0):.2f}")
        print(f"  Volatility: {metrics.get('volatility', 0):.2f}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2f}")
        print(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2f}")
        print(f"  Final Capital: ${metrics.get('final_capital', 0):,.2f}")
    
    if 'strategy_performance' in results:
        print(f"\nStrategy Performance:")
        for strategy, perf in results['strategy_performance'].items():
            print(f"  {strategy}:")
            print(f"    Return: {perf.get('total_return', 0):.2f}")
            print(f"    Volatility: {perf.get('volatility', 0):.2f}")
            print(f"    Sharpe: {perf.get('sharpe', 0):.2f}")
            print(f"    Max DD: {perf.get('max_drawdown', 0):.2f}")
            print(f"    Hit Rate: {perf.get('hit_rate', 0):.2f}")
    
    print(f"\nReport generated in 'portfolio_reports' directory")
    print("="*60)
    
    logger.info("Demo completed successfully!")

if __name__ == "__main__":
    main()
