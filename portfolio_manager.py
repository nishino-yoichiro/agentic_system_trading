#!/usr/bin/env python3
"""
Advanced Portfolio Management System
Implements low-correlation strategy basket with Kelly sizing, regime filtering, and risk controls
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class StrategyMetrics:
    """Metrics for a single trading strategy"""
    name: str
    returns: np.ndarray
    mean_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    avg_r_multiple: float = 0.0
    kelly_fraction: float = 0.0
    regime_filter: str = "all"
    instrument_class: str = "equity"
    mechanism: str = "trend"
    horizon: str = "intraday"
    session: str = "NY"

@dataclass
class RegimeState:
    """Current market regime state"""
    trend_vs_range: str = "range"  # "trend" or "range"
    volatility_regime: str = "normal"  # "low", "normal", "high"
    session: str = "NY"  # "Asia", "London", "NY"
    liquidity_state: str = "normal"  # "low", "normal", "high"
    news_impact: str = "low"  # "low", "medium", "high"

class PortfolioManager:
    """Advanced portfolio management with Kelly sizing and correlation control"""
    
    def __init__(self, 
                 target_volatility: float = 0.10,
                 max_strategy_weight: float = 0.25,
                 correlation_threshold: float = 0.5,
                 fractional_kelly: float = 0.25,
                 lookback_periods: int = 90):
        
        self.target_volatility = target_volatility
        self.max_strategy_weight = max_strategy_weight
        self.correlation_threshold = correlation_threshold
        self.fractional_kelly = fractional_kelly
        self.lookback_periods = lookback_periods
        
        # Strategy tracking
        self.strategies: Dict[str, StrategyMetrics] = {}
        self.current_regime = RegimeState()
        
        # Risk controls
        self.max_portfolio_drawdown = 0.15
        self.correlation_surge_threshold = 0.6
        self.kill_switch_active = False
        
        # Performance tracking
        self.portfolio_returns = []
        self.correlation_matrix = None
        self.allocation_history = []
        
    def add_strategy(self, strategy: StrategyMetrics):
        """Add a strategy to the portfolio"""
        self.strategies[strategy.name] = strategy
        logger.info(f"Added strategy: {strategy.name}")
        
    def update_regime(self, regime: RegimeState):
        """Update current market regime"""
        self.current_regime = regime
        logger.debug(f"Regime updated: {regime}")
        
    def calculate_strategy_metrics(self, strategy: StrategyMetrics) -> StrategyMetrics:
        """Calculate comprehensive metrics for a strategy"""
        returns = strategy.returns
        
        if len(returns) == 0:
            return strategy
            
        # Basic metrics
        strategy.mean_return = np.mean(returns)
        strategy.volatility = np.std(returns)
        strategy.sharpe_ratio = strategy.mean_return / strategy.volatility if strategy.volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        strategy.max_drawdown = np.min(drawdowns)
        
        # Hit rate and R-multiple
        positive_returns = returns[returns > 0]
        strategy.hit_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        strategy.avg_r_multiple = np.mean(returns) if len(returns) > 0 else 0
        
        # Kelly fraction
        if strategy.volatility > 0:
            strategy.kelly_fraction = max(0, strategy.mean_return / (strategy.volatility ** 2))
        
        return strategy
    
    def vol_target_strategies(self) -> Dict[str, StrategyMetrics]:
        """Volatility target all strategies to the same level"""
        vol_targeted = {}
        
        for name, strategy in self.strategies.items():
            if len(strategy.returns) == 0:
                continue
                
            # Calculate recent volatility using EWMA
            returns = strategy.returns
            if len(returns) < 10:
                continue
                
            # EWMA volatility (lambda = 0.94)
            squared_returns = returns ** 2
            ewma_vol = self._calculate_ewma_volatility(squared_returns, lambda_param=0.94)
            
            if ewma_vol > 0:
                # Scale returns to target volatility
                scaling_factor = self.target_volatility / ewma_vol
                adjusted_returns = returns * scaling_factor
                
                # Create new strategy with adjusted returns
                adjusted_strategy = StrategyMetrics(
                    name=name,
                    returns=adjusted_returns,
                    regime_filter=strategy.regime_filter,
                    instrument_class=strategy.instrument_class,
                    mechanism=strategy.mechanism,
                    horizon=strategy.horizon,
                    session=strategy.session
                )
                
                # Recalculate metrics with adjusted returns
                adjusted_strategy = self.calculate_strategy_metrics(adjusted_strategy)
                vol_targeted[name] = adjusted_strategy
        
        return vol_targeted
    
    def _calculate_ewma_volatility(self, squared_returns: np.ndarray, lambda_param: float = 0.94) -> float:
        """Calculate EWMA volatility"""
        if len(squared_returns) == 0:
            return 0.0
            
        # Initialize with sample variance
        initial_var = np.var(squared_returns)
        
        # EWMA calculation
        ewma_var = initial_var
        for ret in squared_returns:
            ewma_var = lambda_param * ewma_var + (1 - lambda_param) * ret
            
        return np.sqrt(ewma_var)
    
    def calculate_correlation_matrix(self, strategies: Dict[str, StrategyMetrics]) -> np.ndarray:
        """Calculate correlation matrix between strategies"""
        if len(strategies) < 2:
            return np.array([[1.0]])
            
        # Get returns for all strategies
        returns_matrix = []
        strategy_names = []
        
        for name, strategy in strategies.items():
            if len(strategy.returns) > 0:
                returns_matrix.append(strategy.returns)
                strategy_names.append(name)
        
        if len(returns_matrix) < 2:
            return np.array([[1.0]])
            
        # Ensure all return series have the same length
        min_length = min(len(returns) for returns in returns_matrix)
        aligned_returns = [returns[-min_length:] for returns in returns_matrix]
        
        # Calculate correlation matrix
        returns_array = np.array(aligned_returns)
        correlation_matrix = np.corrcoef(returns_array)
        
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def calculate_kelly_weights(self, strategies: Dict[str, StrategyMetrics]) -> Dict[str, float]:
        """Calculate Kelly-optimal weights for strategies"""
        if len(strategies) < 2:
            return {name: 1.0 for name in strategies.keys()}
            
        # Prepare data for Kelly calculation
        returns_list = []
        strategy_names = []
        
        for name, strategy in strategies.items():
            if len(strategy.returns) > 0:
                returns_list.append(strategy.returns)
                strategy_names.append(name)
        
        if len(returns_list) < 2:
            return {name: 1.0 for name in strategies.keys()}
            
        # Align returns
        min_length = min(len(returns) for returns in returns_list)
        aligned_returns = [returns[-min_length:] for returns in returns_list]
        returns_array = np.array(aligned_returns)
        
        # Calculate mean returns and covariance matrix
        mean_returns = np.array([np.mean(returns) for returns in aligned_returns])
        cov_matrix = np.cov(returns_array)
        
        # Add small regularization to avoid singular matrix
        regularization = 1e-6
        cov_matrix += regularization * np.eye(cov_matrix.shape[0])
        
        try:
            # Calculate Kelly weights: f* = Σ^(-1) * μ
            inv_cov = np.linalg.inv(cov_matrix)
            kelly_raw = inv_cov @ mean_returns
            
            # Normalize to sum to 1
            kelly_weights = kelly_raw / np.sum(np.abs(kelly_raw))
            
            # Apply fractional Kelly
            kelly_weights *= self.fractional_kelly
            
            # Create weight dictionary
            weight_dict = {}
            for i, name in enumerate(strategy_names):
                weight_dict[name] = max(0, kelly_weights[i])  # No shorting
                
            return weight_dict
            
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, using equal weights")
            return {name: 1.0/len(strategy_names) for name in strategy_names}
    
    def apply_regime_filters(self, weights: Dict[str, float], strategies: Dict[str, StrategyMetrics]) -> Dict[str, float]:
        """Apply regime-based filters to strategy weights"""
        filtered_weights = weights.copy()
        
        for name, strategy in strategies.items():
            if name not in filtered_weights:
                continue
                
            # Check regime compatibility
            if not self._is_strategy_compatible_with_regime(strategy, self.current_regime):
                filtered_weights[name] = 0.0
                logger.debug(f"Filtered out {name} due to regime incompatibility")
        
        return filtered_weights
    
    def _is_strategy_compatible_with_regime(self, strategy: StrategyMetrics, regime: RegimeState) -> bool:
        """Check if strategy is compatible with current regime"""
        
        # Trend vs Range filter
        if strategy.mechanism == "breakout" and regime.trend_vs_range == "range":
            return False
        if strategy.mechanism == "mean_reversion" and regime.trend_vs_range == "trend":
            return False
            
        # Volatility filter
        if strategy.mechanism == "breakout" and regime.volatility_regime == "low":
            return False
        if strategy.mechanism == "mean_reversion" and regime.volatility_regime == "high":
            return False
            
        # Session filter
        if strategy.session != "all" and strategy.session != regime.session:
            return False
            
        # Liquidity filter
        if strategy.mechanism == "sweep_reclaim" and regime.liquidity_state == "low":
            return False
            
        return True
    
    def apply_correlation_penalty(self, weights: Dict[str, float], strategies: Dict[str, StrategyMetrics]) -> Dict[str, float]:
        """Apply correlation penalty to reduce crowding"""
        if self.correlation_matrix is None:
            return weights
            
        penalized_weights = weights.copy()
        
        for i, (name, strategy) in enumerate(strategies.items()):
            if name not in penalized_weights:
                continue
                
            # Calculate average correlation to other strategies
            if i < len(self.correlation_matrix):
                correlations = np.abs(self.correlation_matrix[i, :])
                # Remove self-correlation (1.0)
                other_correlations = np.delete(correlations, i)
                avg_correlation = np.mean(other_correlations) if len(other_correlations) > 0 else 0
                
                # Apply penalty
                penalty_factor = 1 / (1 + avg_correlation)
                penalized_weights[name] *= penalty_factor
                
                logger.debug(f"Applied correlation penalty to {name}: {penalty_factor:.3f}")
        
        return penalized_weights
    
    def apply_risk_controls(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply risk controls and caps"""
        controlled_weights = weights.copy()
        
        # Apply maximum strategy weight cap
        for name in controlled_weights:
            controlled_weights[name] = min(controlled_weights[name], self.max_strategy_weight)
        
        # Check for correlation surge
        if self.correlation_matrix is not None:
            avg_correlation = np.mean(np.abs(self.correlation_matrix))
            if avg_correlation > self.correlation_surge_threshold:
                logger.warning(f"Correlation surge detected: {avg_correlation:.3f}")
                # Halve all weights but don't zero them out
                for name in controlled_weights:
                    controlled_weights[name] *= 0.5
        
        # Normalize weights to sum to 1
        total_weight = sum(controlled_weights.values())
        if total_weight > 0:
            for name in controlled_weights:
                controlled_weights[name] /= total_weight
        else:
            # If all weights are zero, use equal weights
            logger.warning("All weights are zero, using equal weights")
            for name in controlled_weights:
                controlled_weights[name] = 1.0 / len(controlled_weights)
        
        return controlled_weights
    
    def calculate_portfolio_allocation(self) -> Dict[str, float]:
        """Calculate optimal portfolio allocation using all methods"""
        if not self.strategies:
            return {}
            
        logger.info("Calculating portfolio allocation...")
        
        # Step 1: Volatility target all strategies
        vol_targeted = self.vol_target_strategies()
        if not vol_targeted:
            logger.warning("No strategies available after volatility targeting")
            return {}
        
        # Step 2: Calculate correlation matrix
        self.calculate_correlation_matrix(vol_targeted)
        
        # Step 3: Calculate Kelly weights
        kelly_weights = self.calculate_kelly_weights(vol_targeted)
        
        # Step 4: Apply regime filters
        regime_filtered = self.apply_regime_filters(kelly_weights, vol_targeted)
        
        # Step 5: Apply correlation penalty
        correlation_adjusted = self.apply_correlation_penalty(regime_filtered, vol_targeted)
        
        # Step 6: Apply risk controls
        final_weights = self.apply_risk_controls(correlation_adjusted)
        
        # Log allocation
        logger.info("Final portfolio allocation:")
        for name, weight in final_weights.items():
            logger.info(f"  {name}: {weight:.3f}")
        
        # Store allocation history
        self.allocation_history.append({
            'timestamp': datetime.now(),
            'weights': final_weights.copy(),
            'regime': self.current_regime
        })
        
        return final_weights
    
    def calculate_portfolio_metrics(self, weights: Dict[str, float], strategies: Dict[str, StrategyMetrics]) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        if not weights or not strategies:
            return {}
            
        # Calculate weighted portfolio returns
        portfolio_returns = None
        
        for name, weight in weights.items():
            if name in strategies and len(strategies[name].returns) > 0:
                if portfolio_returns is None:
                    portfolio_returns = weight * strategies[name].returns
                else:
                    # Align lengths
                    min_length = min(len(portfolio_returns), len(strategies[name].returns))
                    portfolio_returns = portfolio_returns[:min_length] + weight * strategies[name].returns[:min_length]
        
        if portfolio_returns is None:
            return {}
            
        # Calculate portfolio metrics
        portfolio_mean = np.mean(portfolio_returns)
        portfolio_vol = np.std(portfolio_returns)
        portfolio_sharpe = portfolio_mean / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calculate portfolio drawdown
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calculate diversification ratio
        individual_vols = [np.std(strategy.returns) for strategy in strategies.values() if len(strategy.returns) > 0]
        avg_individual_vol = np.mean(individual_vols) if individual_vols else 0
        diversification_ratio = avg_individual_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        return {
            'portfolio_mean_return': portfolio_mean,
            'portfolio_volatility': portfolio_vol,
            'portfolio_sharpe': portfolio_sharpe,
            'portfolio_max_drawdown': max_drawdown,
            'diversification_ratio': diversification_ratio,
            'num_strategies': len(weights),
            'effective_strategies': sum(1 for w in weights.values() if w > 0.01)
        }
    
    def check_kill_switches(self) -> bool:
        """Check if any kill switches should be activated"""
        if self.kill_switch_active:
            return True
            
        # Check portfolio drawdown
        if self.portfolio_returns:
            recent_returns = self.portfolio_returns[-30:]  # Last 30 periods
            if len(recent_returns) > 0:
                cumulative = np.cumprod(1 + recent_returns)
                current_dd = (cumulative[-1] - np.max(cumulative)) / np.max(cumulative)
                if current_dd < -self.max_portfolio_drawdown:
                    logger.warning(f"Kill switch activated: drawdown {current_dd:.3f}")
                    self.kill_switch_active = True
                    return True
        
        return False
    
    def update_portfolio_returns(self, returns: float):
        """Update portfolio returns for monitoring"""
        self.portfolio_returns.append(returns)
        
        # Keep only recent history
        if len(self.portfolio_returns) > 1000:
            self.portfolio_returns = self.portfolio_returns[-1000:]
    
    def get_correlation_heatmap_data(self) -> Dict[str, Any]:
        """Get data for correlation heatmap visualization"""
        if self.correlation_matrix is None:
            return {}
            
        strategy_names = list(self.strategies.keys())
        
        return {
            'correlation_matrix': self.correlation_matrix.tolist(),
            'strategy_names': strategy_names,
            'avg_correlation': float(np.mean(np.abs(self.correlation_matrix)))
        }
    
    def save_allocation_history(self, filepath: str):
        """Save allocation history to file"""
        history_data = []
        for entry in self.allocation_history:
            history_data.append({
                'timestamp': entry['timestamp'].isoformat(),
                'weights': entry['weights'],
                'regime': {
                    'trend_vs_range': entry['regime'].trend_vs_range,
                    'volatility_regime': entry['regime'].volatility_regime,
                    'session': entry['regime'].session,
                    'liquidity_state': entry['regime'].liquidity_state,
                    'news_impact': entry['regime'].news_impact
                }
            })
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Allocation history saved to {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create portfolio manager
    pm = PortfolioManager(
        target_volatility=0.10,
        max_strategy_weight=0.25,
        correlation_threshold=0.5,
        fractional_kelly=0.25
    )
    
    # Create sample strategies
    np.random.seed(42)
    
    # Strategy 1: Trend following (ES futures)
    trend_returns = np.random.normal(0.02, 0.15, 100)
    trend_strategy = StrategyMetrics(
        name="ES_Trend_Following",
        returns=trend_returns,
        regime_filter="trend",
        instrument_class="futures",
        mechanism="breakout",
        horizon="intraday",
        session="NY"
    )
    
    # Strategy 2: Mean reversion (SPY)
    mean_rev_returns = np.random.normal(0.015, 0.12, 100)
    mean_rev_strategy = StrategyMetrics(
        name="SPY_Mean_Reversion",
        returns=mean_rev_returns,
        regime_filter="range",
        instrument_class="equity",
        mechanism="mean_reversion",
        horizon="overnight",
        session="NY"
    )
    
    # Strategy 3: FX Carry (EUR/USD)
    fx_returns = np.random.normal(0.01, 0.08, 100)
    fx_strategy = StrategyMetrics(
        name="EURUSD_Carry",
        returns=fx_returns,
        regime_filter="all",
        instrument_class="fx",
        mechanism="carry",
        horizon="swing",
        session="London"
    )
    
    # Add strategies
    pm.add_strategy(trend_strategy)
    pm.add_strategy(mean_rev_strategy)
    pm.add_strategy(fx_strategy)
    
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
    print("Portfolio Allocation:", allocation)
    
    # Calculate portfolio metrics
    vol_targeted = pm.vol_target_strategies()
    metrics = pm.calculate_portfolio_metrics(allocation, vol_targeted)
    print("Portfolio Metrics:", metrics)
