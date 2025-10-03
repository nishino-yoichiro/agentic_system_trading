"""
Correlation Engine for Dynamic Correlation Modeling

Features:
- Dynamic correlation calculation
- Rolling correlation windows
- Correlation regime detection
- Cross-asset correlation analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from loguru import logger
from scipy.stats import pearsonr
from sklearn.covariance import LedoitWolf


class CorrelationEngine:
    """Advanced correlation modeling engine"""
    
    def __init__(self, window_size: int = 252):
        self.window_size = window_size
        self.correlation_cache = {}
        self.min_window_size = 10  # Minimum data points for correlation
        
    async def calculate_correlation_matrix(self, returns_data: Dict[str, pd.Series]) -> np.ndarray:
        """Calculate dynamic correlation matrix"""
        symbols = list(returns_data.keys())
        n_assets = len(symbols)
        
        if n_assets < 2:
            return np.eye(n_assets)
        
        # Align data
        aligned_data = self._align_returns_data(returns_data)
        
        if aligned_data is None or len(aligned_data) < self.min_window_size:
            logger.warning("Insufficient data for correlation calculation, using identity matrix")
            return np.eye(n_assets)
        
        # Adjust window size based on available data
        actual_window_size = min(len(aligned_data), self.window_size)
        if actual_window_size < self.window_size:
            logger.info(f"Using {actual_window_size} data points for correlation (requested: {self.window_size})")
        
        # Calculate correlation matrix
        correlation_matrix = aligned_data.corr().values
        
        # Ensure positive definiteness
        correlation_matrix = self._ensure_positive_definite(correlation_matrix)
        
        return correlation_matrix
    
    def _align_returns_data(self, returns_data: Dict[str, pd.Series]) -> Optional[pd.DataFrame]:
        """Align returns data to common time index"""
        try:
            # Find common time index
            common_index = None
            for symbol, returns in returns_data.items():
                if common_index is None:
                    common_index = returns.index
                else:
                    common_index = common_index.intersection(returns.index)
            
            if len(common_index) < self.min_window_size:
                return None
            
            # Create aligned DataFrame
            aligned_data = pd.DataFrame(index=common_index)
            for symbol, returns in returns_data.items():
                aligned_data[symbol] = returns.loc[common_index]
            
            return aligned_data
            
        except Exception as e:
            logger.error(f"Error aligning returns data: {e}")
            return None
    
    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure correlation matrix is positive definite"""
        try:
            # Check if matrix is positive definite
            eigenvals = np.linalg.eigvals(matrix)
            if np.all(eigenvals > 0):
                return matrix
            
            # Use Ledoit-Wolf shrinkage if not positive definite
            lw = LedoitWolf()
            shrunk_matrix = lw.fit(matrix).covariance_
            
            # Convert back to correlation matrix
            std_devs = np.sqrt(np.diag(shrunk_matrix))
            correlation_matrix = shrunk_matrix / np.outer(std_devs, std_devs)
            
            return correlation_matrix
            
        except Exception as e:
            logger.warning(f"Error ensuring positive definiteness: {e}, using identity matrix")
            return np.eye(matrix.shape[0])
    
    def calculate_rolling_correlation(
        self, 
        returns_data: Dict[str, pd.Series], 
        window: int = 60
    ) -> pd.DataFrame:
        """Calculate rolling correlation between assets"""
        aligned_data = self._align_returns_data(returns_data)
        if aligned_data is None:
            return pd.DataFrame()
        
        return aligned_data.rolling(window=window).corr()
    
    def detect_correlation_regime(self, correlation_matrix: np.ndarray) -> str:
        """Detect correlation regime based on matrix characteristics"""
        try:
            # Calculate average correlation
            n = correlation_matrix.shape[0]
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            avg_correlation = np.mean(correlation_matrix[mask])
            
            # Calculate correlation dispersion
            correlation_std = np.std(correlation_matrix[mask])
            
            # Determine regime
            if avg_correlation > 0.7:
                return "high_correlation"
            elif avg_correlation > 0.4:
                return "medium_correlation"
            elif avg_correlation > 0.1:
                return "low_correlation"
            else:
                return "negative_correlation"
                
        except Exception as e:
            logger.error(f"Error detecting correlation regime: {e}")
            return "unknown"
    
    def calculate_correlation_stability(self, returns_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate correlation stability metrics"""
        try:
            aligned_data = self._align_returns_data(returns_data)
            if aligned_data is None:
                return {}
            
            symbols = list(returns_data.keys())
            stability_metrics = {}
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j:
                        # Calculate rolling correlation
                        rolling_corr = aligned_data[symbol1].rolling(60).corr(aligned_data[symbol2])
                        
                        # Calculate stability metrics
                        stability_metrics[f"{symbol1}_{symbol2}"] = {
                            'mean_correlation': rolling_corr.mean(),
                            'correlation_std': rolling_corr.std(),
                            'correlation_range': rolling_corr.max() - rolling_corr.min(),
                            'stability_score': 1 - (rolling_corr.std() / (rolling_corr.max() - rolling_corr.min() + 1e-8))
                        }
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"Error calculating correlation stability: {e}")
            return {}

