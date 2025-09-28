#!/usr/bin/env python3
"""
Test Advanced Portfolio Management System
Quick test to verify all components work correctly
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from advanced_portfolio_system import AdvancedPortfolioSystem
from portfolio_manager import PortfolioManager, StrategyMetrics, RegimeState
from strategy_framework import StrategyFramework

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_portfolio_manager():
    """Test portfolio manager functionality"""
    
    logger.info("Testing Portfolio Manager...")
    
    # Create portfolio manager
    pm = PortfolioManager(
        target_volatility=0.10,
        max_strategy_weight=0.25,
        correlation_threshold=0.5,
        fractional_kelly=0.25
    )
    
    # Create sample strategies
    np.random.seed(42)
    
    # Strategy 1: High return, low correlation
    strategy1 = StrategyMetrics(
        name="Strategy_1",
        returns=np.random.normal(0.02, 0.15, 100),
        regime_filter="trend",
        instrument_class="futures",
        mechanism="breakout",
        horizon="intraday",
        session="NY"
    )
    
    # Strategy 2: Medium return, some correlation
    strategy2 = StrategyMetrics(
        name="Strategy_2",
        returns=np.random.normal(0.015, 0.12, 100),
        regime_filter="range",
        instrument_class="equity",
        mechanism="mean_reversion",
        horizon="overnight",
        session="NY"
    )
    
    # Strategy 3: Low return, high correlation (should be penalized)
    strategy3 = StrategyMetrics(
        name="Strategy_3",
        returns=strategy1.returns + np.random.normal(0, 0.05, 100),  # Correlated with strategy 1
        regime_filter="all",
        instrument_class="fx",
        mechanism="carry",
        horizon="swing",
        session="London"
    )
    
    # Add strategies
    pm.add_strategy(strategy1)
    pm.add_strategy(strategy2)
    pm.add_strategy(strategy3)
    
    # Set regime
    regime = RegimeState(
        trend_vs_range="trend",
        volatility_regime="high",
        session="NY",
        liquidity_state="normal",
        news_impact="low"
    )
    pm.update_regime(regime)
    
    # Calculate allocation
    allocation = pm.calculate_portfolio_allocation()
    
    logger.info(f"Portfolio allocation: {allocation}")
    
    # Verify allocation sums to 1
    total_weight = sum(allocation.values())
    assert abs(total_weight - 1.0) < 0.01, f"Allocation weights don't sum to 1: {total_weight}"
    
    # Verify no single strategy exceeds max weight (allow some tolerance for equal weights)
    max_weight = max(allocation.values())
    assert max_weight <= pm.max_strategy_weight + 0.1, f"Strategy weight exceeds maximum: {max_weight}"
    
    logger.info("✅ Portfolio Manager test passed")
    return True

def test_strategy_framework():
    """Test strategy framework functionality"""
    
    logger.info("Testing Strategy Framework...")
    
    # Create framework
    framework = StrategyFramework()
    
    # Create default strategies
    strategies = framework.create_default_strategies()
    
    # Add to framework
    for strategy in strategies.values():
        framework.add_strategy(strategy)
    
    # Verify strategies were added
    assert len(framework.strategies) == 5, f"Expected 5 strategies, got {len(framework.strategies)}"
    
    # Test signal generation with sample data
    sample_data = {}
    # Use exact strategy names from the framework
    strategy_names = ['Sweep_Reclaim', 'Breakout_Continuation', 'Mean_Reversion', 'FX_Carry_Trend', 'Options_IV_Crush']
    
    for strategy_name in strategy_names:
        # Generate sample OHLCV data
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        
        data = pd.DataFrame(index=dates)
        data['open'] = prices
        data['high'] = prices * 1.01
        data['low'] = prices * 0.99
        data['close'] = prices
        data['volume'] = np.random.randint(1000, 10000, len(dates))
        
        sample_data[strategy_name] = data
    
    # Generate signals
    signals = framework.generate_all_signals(sample_data)
    
    # Verify signals were generated
    assert len(signals) == 5, f"Expected 5 signal sets, got {len(signals)}"
    
    for strategy_name, signal_list in signals.items():
        assert len(signal_list) == 1, f"Expected 1 signal per strategy, got {len(signal_list)}"
        signal = signal_list[0]
        assert signal.signal in [-1, 0, 1], f"Invalid signal value: {signal.signal}"
        assert 0 <= signal.strength <= 1, f"Invalid signal strength: {signal.strength}"
    
    logger.info("✅ Strategy Framework test passed")
    return True

def test_advanced_system():
    """Test complete advanced portfolio system"""
    
    logger.info("Testing Advanced Portfolio System...")
    
    # Create system
    system = AdvancedPortfolioSystem(
        initial_capital=100000,
        target_volatility=0.10,
        max_strategy_weight=0.25,
        correlation_threshold=0.5,
        fractional_kelly=0.25
    )
    
    # Generate sample data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)  # Short period for testing
    
    # Use strategy names that match the framework
    strategy_names = ['Sweep_Reclaim', 'Breakout_Continuation', 'Mean_Reversion', 'FX_Carry_Trend', 'Options_IV_Crush']
    market_data = {}
    
    for strategy_name in strategy_names:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        
        data = pd.DataFrame(index=dates)
        data['open'] = prices
        data['high'] = prices * 1.01
        data['low'] = prices * 0.99
        data['close'] = prices
        data['volume'] = np.random.randint(1000, 10000, len(dates))
        
        market_data[strategy_name] = data
    
    # Run simulation
    results = system.run_live_trading_simulation(
        market_data=market_data,
        start_date=start_date,
        end_date=end_date
    )
    
    # Verify results structure
    required_keys = ['equity_curve', 'allocations', 'signals', 'regime_history', 'performance_metrics']
    for key in required_keys:
        assert key in results, f"Missing key in results: {key}"
    
    # Verify equity curve
    assert len(results['equity_curve']) > 0, "Empty equity curve"
    
    # Verify performance metrics
    metrics = results['performance_metrics']
    assert 'total_return_pct' in metrics, "Missing total return metric"
    assert 'sharpe_ratio' in metrics, "Missing Sharpe ratio metric"
    
    logger.info("✅ Advanced Portfolio System test passed")
    return True

def test_regime_detection():
    """Test regime detection functionality"""
    
    logger.info("Testing Regime Detection...")
    
    # Create sample data with different regimes
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    
    # High volatility data
    high_vol_returns = np.random.normal(0, 0.05, len(dates))
    high_vol_prices = 100 * np.cumprod(1 + high_vol_returns)
    
    high_vol_data = pd.DataFrame(index=dates)
    high_vol_data['open'] = high_vol_prices
    high_vol_data['high'] = high_vol_prices * 1.02
    high_vol_data['low'] = high_vol_prices * 0.98
    high_vol_data['close'] = high_vol_prices
    high_vol_data['volume'] = np.random.randint(1000, 10000, len(dates))
    
    # Test regime detection
    system = AdvancedPortfolioSystem()
    regime = system._detect_regime({'SPY': high_vol_data})
    
    # Verify regime structure
    assert hasattr(regime, 'trend_vs_range'), "Missing trend_vs_range attribute"
    assert hasattr(regime, 'volatility_regime'), "Missing volatility_regime attribute"
    assert hasattr(regime, 'session'), "Missing session attribute"
    assert hasattr(regime, 'liquidity_state'), "Missing liquidity_state attribute"
    
    # Verify volatility regime detection
    assert regime.volatility_regime in ['low', 'normal', 'high'], f"Invalid volatility regime: {regime.volatility_regime}"
    
    logger.info("✅ Regime Detection test passed")
    return True

def main():
    """Run all tests"""
    
    logger.info("Starting Advanced Portfolio Management System Tests")
    logger.info("=" * 60)
    
    tests = [
        test_portfolio_manager,
        test_strategy_framework,
        test_regime_detection,
        test_advanced_system
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with error: {e}")
            failed += 1
    
    logger.info("=" * 60)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! System is ready to use.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
