"""
Risk Metrics Calculation

Calculate various risk metrics for portfolio analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger
from scipy import stats


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    volatility: float
    beta: float


class RiskCalculator:
    """Calculate various risk metrics"""
    
    def __init__(self):
        pass
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if returns.empty:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if returns.empty:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if prices.empty:
            return 0.0
        
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if returns.empty:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if downside_returns.empty or downside_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
    
    def calculate_calmar_ratio(self, returns: pd.Series, prices: pd.Series) -> float:
        """Calculate Calmar ratio"""
        if returns.empty or prices.empty:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_dd = abs(self.calculate_max_drawdown(prices))
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / max_dd
    
    def calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        if returns.empty:
            return 0.0
        
        return returns.std() * np.sqrt(252)
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market"""
        if asset_returns.empty or market_returns.empty:
            return 0.0
        
        # Align the series
        aligned_data = pd.concat([asset_returns, market_returns], axis=1, join='inner')
        if aligned_data.empty:
            return 0.0
        
        asset_col, market_col = aligned_data.columns
        
        covariance = aligned_data[asset_col].cov(aligned_data[market_col])
        market_variance = aligned_data[market_col].var()
        
        if market_variance == 0:
            return 0.0
        
        return covariance / market_variance
    
    def calculate_all_metrics(self, returns: pd.Series, prices: pd.Series, 
                            market_returns: pd.Series = None, risk_free_rate: float = 0.02) -> RiskMetrics:
        """Calculate all risk metrics"""
        try:
            # Value at Risk
            var_95 = self.calculate_var(returns, 0.95)
            var_99 = self.calculate_var(returns, 0.99)
            
            # Conditional Value at Risk
            cvar_95 = self.calculate_cvar(returns, 0.95)
            cvar_99 = self.calculate_cvar(returns, 0.99)
            
            # Drawdown
            max_drawdown = self.calculate_max_drawdown(prices)
            
            # Risk-adjusted returns
            sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate)
            sortino_ratio = self.calculate_sortino_ratio(returns, risk_free_rate)
            calmar_ratio = self.calculate_calmar_ratio(returns, prices)
            
            # Volatility
            volatility = self.calculate_volatility(returns)
            
            # Beta
            beta = 0.0
            if market_returns is not None:
                beta = self.calculate_beta(returns, market_returns)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                volatility=volatility,
                beta=beta
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def calculate_portfolio_risk(self, weights: pd.Series, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        try:
            # Portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Portfolio volatility
            portfolio_vol = portfolio_returns.std() * np.sqrt(252)
            
            # Portfolio VaR
            portfolio_var = self.calculate_var(portfolio_returns, 0.95)
            
            # Portfolio Sharpe ratio
            portfolio_sharpe = self.calculate_sharpe_ratio(portfolio_returns)
            
            return {
                'volatility': portfolio_vol,
                'var_95': portfolio_var,
                'sharpe_ratio': portfolio_sharpe
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {'volatility': 0, 'var_95': 0, 'sharpe_ratio': 0}
