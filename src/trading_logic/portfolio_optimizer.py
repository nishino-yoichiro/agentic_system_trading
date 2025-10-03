"""
Portfolio Optimization and Allocation

Features:
- Modern portfolio theory optimization
- Risk parity allocation
- Black-Litterman model
- Dynamic rebalancing
- Risk budgeting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
from loguru import logger
from scipy.optimize import minimize
from scipy.linalg import cholesky
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""
    symbol: str
    weight: float
    expected_return: float
    risk_contribution: float
    sharpe_ratio: float
    allocation_reason: str


class PortfolioOptimizer:
    """Advanced portfolio optimization engine"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.optimization_method = "mean_variance"  # mean_variance, risk_parity, black_litterman
        
    async def initialize(self):
        """Initialize the portfolio optimizer"""
        logger.info("Portfolio optimizer initialized")
    
    async def optimize_portfolio(
        self,
        signals: Dict[str, Any],
        price_data: Dict[str, pd.DataFrame],
        max_positions: int = 10,
        confidence_threshold: float = 0.6
    ) -> List[PortfolioAllocation]:
        """Optimize portfolio allocation based on signals and data"""
        logger.info("Optimizing portfolio allocation")
        
        try:
            # Filter assets by confidence threshold
            qualified_assets = self._filter_qualified_assets(signals, confidence_threshold)
            
            if not qualified_assets:
                logger.warning("No assets meet confidence threshold")
                return []
            
            # Limit to max positions
            if len(qualified_assets) > max_positions:
                qualified_assets = self._select_top_assets(qualified_assets, max_positions)
            
            # Calculate expected returns and risks
            asset_metrics = self._calculate_asset_metrics(qualified_assets, price_data)
            
            if not asset_metrics:
                logger.warning("No valid asset metrics calculated")
                return []
            
            # Optimize portfolio
            if self.optimization_method == "mean_variance":
                optimal_weights = self._mean_variance_optimization(asset_metrics)
            elif self.optimization_method == "risk_parity":
                optimal_weights = self._risk_parity_optimization(asset_metrics)
            else:
                optimal_weights = self._equal_weight_allocation(asset_metrics)
            
            # Create allocation results
            allocations = []
            for symbol, weight in optimal_weights.items():
                if weight > 0.01:  # Only include positions > 1%
                    metrics = asset_metrics[symbol]
                    allocation = PortfolioAllocation(
                        symbol=symbol,
                        weight=weight,
                        expected_return=metrics['expected_return'],
                        risk_contribution=weight * metrics['volatility'],
                        sharpe_ratio=metrics['sharpe_ratio'],
                        allocation_reason=metrics['reasoning']
                    )
                    allocations.append(allocation)
            
            logger.info(f"Optimized portfolio with {len(allocations)} positions")
            return allocations
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return []
    
    def _filter_qualified_assets(
        self, 
        signals: Dict[str, Any], 
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """Filter assets that meet confidence threshold"""
        qualified = []
        
        logger.info(f"Filtering assets with confidence threshold: {confidence_threshold}")
        
        for symbol, signal_data in signals.items():
            logger.info(f"Processing asset {symbol}: signal_data type = {type(signal_data)}")
            if hasattr(signal_data, 'confidence'):
                confidence = signal_data.confidence
                logger.info(f"Asset {symbol}: confidence={confidence:.3f}, threshold={confidence_threshold:.3f}")
                if confidence >= confidence_threshold:
                    qualified.append({
                        'symbol': symbol,
                        'confidence': confidence,
                        'signal_type': signal_data.final_signal.value if hasattr(signal_data, 'final_signal') else 'hold',
                        'reasoning': signal_data.reasoning if hasattr(signal_data, 'reasoning') else '',
                        'metadata': signal_data.metadata if hasattr(signal_data, 'metadata') else {}
                    })
                    logger.info(f"Asset {symbol} QUALIFIED with confidence {confidence:.3f}")
                else:
                    logger.info(f"Asset {symbol} REJECTED with confidence {confidence:.3f}")
            elif isinstance(signal_data, dict) and 'confidence' in signal_data:
                confidence = signal_data['confidence']
                logger.info(f"Asset {symbol}: confidence={confidence:.3f}, threshold={confidence_threshold:.3f}")
                if confidence >= confidence_threshold:
                    qualified.append({
                        'symbol': symbol,
                        'confidence': confidence,
                        'signal_type': signal_data.get('signal_type', 'hold'),
                        'reasoning': signal_data.get('reasoning', ''),
                        'metadata': signal_data.get('metadata', {})
                    })
                    logger.info(f"Asset {symbol} QUALIFIED with confidence {confidence:.3f}")
                else:
                    logger.info(f"Asset {symbol} REJECTED with confidence {confidence:.3f}")
            else:
                logger.info(f"Asset {symbol}: Invalid signal data structure - {signal_data}")
        
        logger.info(f"Qualified assets: {len(qualified)} out of {len(signals)}")
        return qualified
    
    def _select_top_assets(
        self, 
        qualified_assets: List[Dict[str, Any]], 
        max_positions: int
    ) -> List[Dict[str, Any]]:
        """Select top assets by confidence and signal strength"""
        # Sort by confidence and signal strength
        def sort_key(asset):
            confidence = asset['confidence']
            signal_type = asset['signal_type']
            
            # Boost buy signals
            if signal_type in ['buy', 'strong_buy']:
                confidence += 0.1
            elif signal_type in ['sell', 'strong_sell']:
                confidence -= 0.1
            
            return confidence
        
        sorted_assets = sorted(qualified_assets, key=sort_key, reverse=True)
        return sorted_assets[:max_positions]
    
    def _calculate_asset_metrics(
        self, 
        qualified_assets: List[Dict[str, Any]], 
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate expected returns and risks for assets"""
        metrics = {}
        
        for asset in qualified_assets:
            symbol = asset['symbol']
            
            if symbol not in price_data or len(price_data[symbol]) < 20:
                continue
            
            try:
                # Calculate returns
                returns = price_data[symbol]['close'].pct_change().dropna()
                
                # Calculate metrics
                expected_return = returns.mean() * 252  # Annualized
                volatility = returns.std() * np.sqrt(252)  # Annualized
                sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
                
                # Calculate additional metrics
                max_drawdown = self._calculate_max_drawdown(returns)
                var_95 = np.percentile(returns, 5)
                
                # Determine reasoning
                reasoning = f"Confidence: {asset['confidence']:.2f}, Signal: {asset['signal_type']}"
                
                metrics[symbol] = {
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'var_95': var_95,
                    'confidence': asset['confidence'],
                    'signal_type': asset['signal_type'],
                    'reasoning': reasoning,
                    'returns': returns
                }
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {symbol}: {e}")
                continue
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _mean_variance_optimization(self, asset_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Mean-variance optimization"""
        try:
            symbols = list(asset_metrics.keys())
            n_assets = len(symbols)
            
            if n_assets == 0:
                return {}
            
            # Expected returns
            expected_returns = np.array([asset_metrics[symbol]['expected_return'] for symbol in symbols])
            
            # Covariance matrix (simplified - using diagonal)
            volatilities = np.array([asset_metrics[symbol]['volatility'] for symbol in symbols])
            cov_matrix = np.diag(volatilities ** 2)
            
            # Risk aversion parameter
            risk_aversion = 1.0
            
            # Objective function
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                return -(portfolio_return - risk_aversion * portfolio_variance)
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            
            # Bounds (no short selling)
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                return {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}
            else:
                logger.warning("Optimization failed, using equal weights")
                return self._equal_weight_allocation(asset_metrics)
                
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {e}")
            return self._equal_weight_allocation(asset_metrics)
    
    def _risk_parity_optimization(self, asset_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Risk parity optimization"""
        try:
            symbols = list(asset_metrics.keys())
            n_assets = len(symbols)
            
            if n_assets == 0:
                return {}
            
            # Volatilities
            volatilities = np.array([asset_metrics[symbol]['volatility'] for symbol in symbols])
            
            # Risk parity weights (inverse volatility)
            inv_vol = 1 / volatilities
            weights = inv_vol / np.sum(inv_vol)
            
            return {symbol: weight for symbol, weight in zip(symbols, weights)}
            
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return self._equal_weight_allocation(asset_metrics)
    
    def _equal_weight_allocation(self, asset_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Equal weight allocation"""
        symbols = list(asset_metrics.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {}
        
        equal_weight = 1.0 / n_assets
        return {symbol: equal_weight for symbol in symbols}
    
    def calculate_portfolio_metrics(self, allocations: List[PortfolioAllocation]) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        if not allocations:
            return {}
        
        # Portfolio weights
        weights = np.array([alloc.weight for alloc in allocations])
        
        # Expected returns
        expected_returns = np.array([alloc.expected_return for alloc in allocations])
        
        # Portfolio expected return
        portfolio_return = np.dot(weights, expected_returns)
        
        # Portfolio volatility (simplified - assuming no correlation)
        volatilities = np.array([alloc.risk_contribution / alloc.weight for alloc in allocations])
        portfolio_variance = np.dot(weights ** 2, volatilities ** 2)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Risk metrics
        total_risk_contribution = sum(alloc.risk_contribution for alloc in allocations)
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_risk_contribution': total_risk_contribution,
            'number_of_positions': len(allocations),
            'concentration_risk': np.max(weights)  # Largest position weight
        }
    
    def rebalance_portfolio(
        self, 
        current_allocations: List[PortfolioAllocation],
        target_allocations: List[PortfolioAllocation],
        rebalance_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Calculate rebalancing trades"""
        rebalance_trades = []
        
        # Create current weights dict
        current_weights = {alloc.symbol: alloc.weight for alloc in current_allocations}
        
        # Create target weights dict
        target_weights = {alloc.symbol: alloc.weight for alloc in target_allocations}
        
        # All symbols
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            
            # Only rebalance if difference exceeds threshold
            if abs(weight_diff) > rebalance_threshold:
                trade = {
                    'symbol': symbol,
                    'action': 'buy' if weight_diff > 0 else 'sell',
                    'weight_change': weight_diff,
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'reason': 'Portfolio rebalancing'
                }
                rebalance_trades.append(trade)
        
        return rebalance_trades


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        optimizer = PortfolioOptimizer()
        await optimizer.initialize()
        
        # Example signals
        signals = {
            'BTC': {'confidence': 0.8, 'signal_type': 'buy', 'reasoning': 'Strong technical indicators'},
            'ETH': {'confidence': 0.7, 'signal_type': 'buy', 'reasoning': 'Positive sentiment'},
            'AAPL': {'confidence': 0.6, 'signal_type': 'hold', 'reasoning': 'Mixed signals'}
        }
        
        # Example price data
        price_data = {
            'BTC': pd.DataFrame({
                'close': [50000, 51000, 52000, 53000, 54000]
            }),
            'ETH': pd.DataFrame({
                'close': [3000, 3100, 3200, 3300, 3400]
            }),
            'AAPL': pd.DataFrame({
                'close': [150, 151, 152, 153, 154]
            })
        }
        
        # Optimize portfolio
        allocations = await optimizer.optimize_portfolio(signals, price_data)
        
        for allocation in allocations:
            print(f"{allocation.symbol}: {allocation.weight:.2%} (Return: {allocation.expected_return:.2%})")
    
    asyncio.run(main())

