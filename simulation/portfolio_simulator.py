"""
Advanced Portfolio Simulator with Monte Carlo Methods

Features:
- Monte Carlo simulations with correlation modeling
- Portfolio optimization using modern portfolio theory
- Risk metrics calculation (VaR, CVaR, drawdown)
- Market regime detection and adaptation
- Scenario analysis and stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from loguru import logger
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SimulationResult:
    """Result of portfolio simulation"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    probability_of_profit: float
    scenarios: List[Dict[str, Any]]
    optimal_weights: Dict[str, float]
    risk_metrics: Dict[str, float]
    regime_analysis: Dict[str, Any]


class PortfolioSimulator:
    """Advanced portfolio simulator with Monte Carlo methods"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.correlation_engine = None
        self.regime_detector = None
        
    async def initialize(self):
        """Initialize the simulator"""
        from .correlation_engine import CorrelationEngine
        from .regime_detector import RegimeDetector
        
        self.correlation_engine = CorrelationEngine()
        self.regime_detector = RegimeDetector()
        
        logger.info("Portfolio simulator initialized")
    
    async def run_portfolio_simulation(
        self,
        assets_data: Dict[str, Dict[str, Any]],
        time_horizon: int = 30,
        num_simulations: int = 10000,
        risk_tolerance: str = 'medium',
        rebalance_frequency: int = 7
    ) -> SimulationResult:
        """
        Run comprehensive portfolio simulation
        
        Args:
            assets_data: Dictionary with asset data (returns, volatility, etc.)
            time_horizon: Simulation horizon in days
            num_simulations: Number of Monte Carlo scenarios
            risk_tolerance: Risk tolerance level ('low', 'medium', 'high')
            rebalance_frequency: Rebalancing frequency in days
        """
        logger.info(f"Running portfolio simulation with {num_simulations} scenarios")
        
        try:
            # Prepare asset data
            symbols = list(assets_data.keys())
            returns_data = {symbol: assets_data[symbol]['returns'] for symbol in symbols}
            
            # Calculate correlation matrix
            correlation_matrix = await self.correlation_engine.calculate_correlation_matrix(returns_data)
            
            # Detect market regime
            regime = await self.regime_detector.detect_regime(returns_data)
            
            # Adjust parameters based on regime
            adjusted_params = self._adjust_for_regime(assets_data, regime)
            
            # Run Monte Carlo simulation
            scenarios = self._run_monte_carlo_simulation(
                adjusted_params,
                correlation_matrix,
                time_horizon,
                num_simulations,
                rebalance_frequency
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(scenarios)
            
            # Optimize portfolio weights
            optimal_weights = self._optimize_portfolio_weights(
                adjusted_params,
                correlation_matrix,
                risk_tolerance
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(scenarios)
            
            # Regime analysis
            regime_analysis = await self.regime_detector.analyze_regime_impact(
                scenarios, regime
            )
            
            result = SimulationResult(
                expected_return=performance_metrics['expected_return'],
                volatility=performance_metrics['volatility'],
                sharpe_ratio=performance_metrics['sharpe_ratio'],
                sortino_ratio=performance_metrics['sortino_ratio'],
                max_drawdown=risk_metrics['max_drawdown'],
                var_95=risk_metrics['var_95'],
                var_99=risk_metrics['var_99'],
                cvar_95=risk_metrics['cvar_95'],
                cvar_99=risk_metrics['cvar_99'],
                probability_of_profit=performance_metrics['probability_of_profit'],
                scenarios=scenarios,
                optimal_weights=optimal_weights,
                risk_metrics=risk_metrics,
                regime_analysis=regime_analysis
            )
            
            logger.info("Portfolio simulation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio simulation: {e}")
            raise
    
    def _adjust_for_regime(self, assets_data: Dict, regime: str) -> Dict:
        """Adjust asset parameters based on market regime"""
        adjusted_data = assets_data.copy()
        
        regime_multipliers = {
            'bull_market': {'volatility': 0.8, 'drift': 1.2},
            'bear_market': {'volatility': 1.3, 'drift': 0.7},
            'high_volatility': {'volatility': 1.5, 'drift': 0.9},
            'low_volatility': {'volatility': 0.7, 'drift': 1.1},
            'sideways': {'volatility': 1.0, 'drift': 1.0}
        }
        
        multiplier = regime_multipliers.get(regime, {'volatility': 1.0, 'drift': 1.0})
        
        for symbol, data in adjusted_data.items():
            data['volatility'] *= multiplier['volatility']
            data['mean_return'] *= multiplier['drift']
        
        return adjusted_data
    
    def _run_monte_carlo_simulation(
        self,
        assets_data: Dict,
        correlation_matrix: np.ndarray,
        time_horizon: int,
        num_simulations: int,
        rebalance_frequency: int
    ) -> List[Dict[str, Any]]:
        """Run Monte Carlo simulation"""
        symbols = list(assets_data.keys())
        n_assets = len(symbols)
        
        # Prepare parameters
        means = np.array([assets_data[symbol]['mean_return'] for symbol in symbols])
        volatilities = np.array([assets_data[symbol]['volatility'] for symbol in symbols])
        current_prices = np.array([assets_data[symbol]['current_price'] for symbol in symbols])
        
        # Convert to daily parameters
        daily_means = means / 252
        daily_volatilities = volatilities / np.sqrt(252)
        
        # Generate correlated random shocks
        scenarios = []
        
        for sim in range(num_simulations):
            # Generate random shocks for each day
            portfolio_values = []
            asset_prices = current_prices.copy()
            
            for day in range(time_horizon):
                # Generate correlated random shocks
                random_shocks = self._generate_correlated_shocks(
                    correlation_matrix, n_assets
                )
                
                # Update asset prices
                daily_returns = daily_means + daily_volatilities * random_shocks
                asset_prices *= (1 + daily_returns)
                
                # Calculate portfolio value (equal weight for now)
                portfolio_value = np.sum(asset_prices) / n_assets
                portfolio_values.append(portfolio_value)
            
            # Store scenario
            scenarios.append({
                'simulation_id': sim,
                'final_value': portfolio_values[-1],
                'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
                'daily_values': portfolio_values,
                'asset_prices': asset_prices.tolist(),
                'max_drawdown': self._calculate_max_drawdown(portfolio_values)
            })
        
        return scenarios
    
    def _generate_correlated_shocks(self, correlation_matrix: np.ndarray, n_assets: int) -> np.ndarray:
        """Generate correlated random shocks using Cholesky decomposition"""
        try:
            # Cholesky decomposition
            L = np.linalg.cholesky(correlation_matrix)
            
            # Generate independent random shocks
            independent_shocks = np.random.normal(0, 1, n_assets)
            
            # Transform to correlated shocks
            correlated_shocks = L @ independent_shocks
            
            return correlated_shocks
        except np.linalg.LinAlgError:
            # If correlation matrix is not positive definite, use independent shocks
            logger.warning("Correlation matrix not positive definite, using independent shocks")
            return np.random.normal(0, 1, n_assets)
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_risk_metrics(self, scenarios: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        returns = [scenario['total_return'] for scenario in scenarios]
        returns_array = np.array(returns)
        
        # VaR and CVaR
        var_95 = np.percentile(returns_array, 5)
        var_99 = np.percentile(returns_array, 1)
        
        cvar_95 = np.mean(returns_array[returns_array <= var_95])
        cvar_99 = np.mean(returns_array[returns_array <= var_99])
        
        # Maximum drawdown
        max_drawdown = max(scenario['max_drawdown'] for scenario in scenarios)
        
        # Tail risk metrics
        tail_ratio = abs(cvar_95) / abs(var_95) if var_95 != 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown': max_drawdown,
            'tail_ratio': tail_ratio,
            'skewness': self._calculate_skewness(returns_array),
            'kurtosis': self._calculate_kurtosis(returns_array)
        }
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0
        return np.mean(((returns - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0
        return np.mean(((returns - mean) / std) ** 4) - 3
    
    def _calculate_performance_metrics(self, scenarios: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics"""
        returns = [scenario['total_return'] for scenario in scenarios]
        returns_array = np.array(returns)
        
        expected_return = np.mean(returns_array)
        volatility = np.std(returns_array)
        
        # Sharpe ratio
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_array[returns_array < 0]
        downside_volatility = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = (expected_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Probability of profit
        probability_of_profit = np.mean(returns_array > 0)
        
        # Calmar ratio
        max_drawdown = max(scenario['max_drawdown'] for scenario in scenarios)
        calmar_ratio = expected_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'probability_of_profit': probability_of_profit
        }
    
    def _optimize_portfolio_weights(
        self,
        assets_data: Dict,
        correlation_matrix: np.ndarray,
        risk_tolerance: str
    ) -> Dict[str, float]:
        """Optimize portfolio weights using modern portfolio theory"""
        symbols = list(assets_data.keys())
        n_assets = len(symbols)
        
        # Prepare expected returns and covariance matrix
        expected_returns = np.array([assets_data[symbol]['mean_return'] for symbol in symbols])
        volatilities = np.array([assets_data[symbol]['volatility'] for symbol in symbols])
        
        # Create covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Risk tolerance parameters
        risk_params = {
            'low': 0.1,
            'medium': 0.5,
            'high': 1.0
        }
        risk_aversion = risk_params.get(risk_tolerance, 0.5)
        
        # Objective function (mean-variance optimization)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -(portfolio_return - risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        
        # Bounds (no short selling)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            optimal_weights = result.x
        except Exception as e:
            logger.warning(f"Optimization failed: {e}, using equal weights")
            optimal_weights = x0
        
        # Return as dictionary
        return {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}
    
    async def run_stress_test(
        self,
        assets_data: Dict,
        stress_scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run stress testing scenarios"""
        logger.info("Running stress test scenarios")
        
        stress_results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            # Adjust asset data for stress scenario
            stressed_data = self._apply_stress_scenario(assets_data, scenario_params)
            
            # Run simulation with stressed data
            result = await self.run_portfolio_simulation(
                stressed_data,
                time_horizon=30,
                num_simulations=5000
            )
            
            stress_results[scenario_name] = {
                'expected_return': result.expected_return,
                'volatility': result.volatility,
                'var_95': result.var_95,
                'max_drawdown': result.max_drawdown
            }
        
        return stress_results
    
    def _apply_stress_scenario(self, assets_data: Dict, scenario_params: Dict) -> Dict:
        """Apply stress scenario to asset data"""
        stressed_data = assets_data.copy()
        
        for symbol, data in stressed_data.items():
            # Apply volatility shock
            if 'volatility_shock' in scenario_params:
                data['volatility'] *= scenario_params['volatility_shock']
            
            # Apply return shock
            if 'return_shock' in scenario_params:
                data['mean_return'] += scenario_params['return_shock']
            
            # Apply correlation shock
            if 'correlation_shock' in scenario_params:
                # This would require more complex correlation matrix adjustment
                pass
        
        return stressed_data


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create sample asset data
        assets_data = {
            'BTC': {
                'returns': pd.Series(np.random.normal(0.001, 0.05, 100)),
                'volatility': 0.8,
                'mean_return': 0.2,
                'current_price': 50000
            },
            'ETH': {
                'returns': pd.Series(np.random.normal(0.001, 0.06, 100)),
                'volatility': 0.9,
                'mean_return': 0.25,
                'current_price': 3000
            },
            'AAPL': {
                'returns': pd.Series(np.random.normal(0.0005, 0.02, 100)),
                'volatility': 0.3,
                'mean_return': 0.1,
                'current_price': 150
            }
        }
        
        # Initialize simulator
        simulator = PortfolioSimulator()
        await simulator.initialize()
        
        # Run simulation
        result = await simulator.run_portfolio_simulation(assets_data)
        
        print(f"Expected Return: {result.expected_return:.2%}")
        print(f"Volatility: {result.volatility:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"VaR 95%: {result.var_95:.2%}")
    
    asyncio.run(main())

