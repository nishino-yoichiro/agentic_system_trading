"""
Market Regime Detection and Analysis

Features:
- Bull/bear/sideways market detection
- Volatility regime classification
- Regime transition analysis
- Regime-specific parameter adjustment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """Advanced market regime detection"""
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
        self.regime_history = []
        
    async def detect_regime(self, returns_data: Dict[str, pd.Series]) -> str:
        """Detect current market regime"""
        try:
            # Align data
            aligned_data = self._align_returns_data(returns_data)
            if aligned_data is None or len(aligned_data) < 60:
                return "unknown"
            
            # Calculate regime indicators
            indicators = self._calculate_regime_indicators(aligned_data)
            
            # Classify regime
            regime = self._classify_regime(indicators)
            
            # Store regime history
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': regime,
                'indicators': indicators
            })
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return "unknown"
    
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
            
            if len(common_index) < 60:
                return None
            
            # Create aligned DataFrame
            aligned_data = pd.DataFrame(index=common_index)
            for symbol, returns in returns_data.items():
                aligned_data[symbol] = returns.loc[common_index]
            
            return aligned_data
            
        except Exception as e:
            logger.error(f"Error aligning returns data: {e}")
            return None
    
    def _calculate_regime_indicators(self, aligned_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate regime indicators"""
        try:
            # Calculate portfolio returns (equal weight)
            portfolio_returns = aligned_data.mean(axis=1)
            
            # Recent performance indicators
            recent_returns = portfolio_returns.tail(20)
            long_term_returns = portfolio_returns.tail(60)
            
            # Volatility indicators
            recent_volatility = recent_returns.std() * np.sqrt(252)
            long_term_volatility = long_term_returns.std() * np.sqrt(252)
            
            # Trend indicators
            recent_trend = recent_returns.mean() * 252
            long_term_trend = long_term_returns.mean() * 252
            
            # Momentum indicators
            momentum_1m = recent_returns.sum()
            momentum_3m = long_term_returns.sum()
            
            # Volatility regime
            volatility_ratio = recent_volatility / long_term_volatility if long_term_volatility > 0 else 1
            
            # Drawdown analysis
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Market breadth (how many assets are positive)
            recent_positive_ratio = (aligned_data.tail(20) > 0).mean().mean()
            
            return {
                'recent_trend': recent_trend,
                'long_term_trend': long_term_trend,
                'recent_volatility': recent_volatility,
                'long_term_volatility': long_term_volatility,
                'volatility_ratio': volatility_ratio,
                'momentum_1m': momentum_1m,
                'momentum_3m': momentum_3m,
                'max_drawdown': max_drawdown,
                'positive_ratio': recent_positive_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating regime indicators: {e}")
            return {}
    
    def _classify_regime(self, indicators: Dict[str, float]) -> str:
        """Classify market regime based on indicators"""
        try:
            recent_trend = indicators.get('recent_trend', 0)
            long_term_trend = indicators.get('long_term_trend', 0)
            volatility_ratio = indicators.get('volatility_ratio', 1)
            max_drawdown = indicators.get('max_drawdown', 0)
            positive_ratio = indicators.get('positive_ratio', 0.5)
            
            # High volatility regime
            if volatility_ratio > 1.5:
                return "high_volatility"
            
            # Low volatility regime
            if volatility_ratio < 0.7:
                return "low_volatility"
            
            # Bear market (negative trends and high drawdown)
            if (recent_trend < -0.1 or long_term_trend < -0.05) and max_drawdown < -0.2:
                return "bear_market"
            
            # Bull market (positive trends and low drawdown)
            if recent_trend > 0.1 and long_term_trend > 0.05 and max_drawdown > -0.1:
                return "bull_market"
            
            # Sideways market (neutral trends)
            if abs(recent_trend) < 0.05 and abs(long_term_trend) < 0.05:
                return "sideways"
            
            # Transitional regimes
            if recent_trend > 0.05 and positive_ratio > 0.6:
                return "bull_transition"
            elif recent_trend < -0.05 and positive_ratio < 0.4:
                return "bear_transition"
            
            return "sideways"
            
        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return "unknown"
    
    async def analyze_regime_impact(
        self, 
        scenarios: List[Dict], 
        current_regime: str
    ) -> Dict[str, Any]:
        """Analyze impact of current regime on scenarios"""
        try:
            if not scenarios:
                return {}
            
            # Extract scenario returns
            returns = [scenario['total_return'] for scenario in scenarios]
            
            # Calculate regime-specific metrics
            regime_metrics = {
                'regime': current_regime,
                'scenario_count': len(scenarios),
                'expected_return': np.mean(returns),
                'volatility': np.std(returns),
                'positive_scenarios': sum(1 for r in returns if r > 0),
                'negative_scenarios': sum(1 for r in returns if r < 0),
                'probability_of_profit': sum(1 for r in returns if r > 0) / len(returns)
            }
            
            # Regime-specific adjustments
            if current_regime == "bull_market":
                regime_metrics['risk_adjustment'] = 0.8  # Lower risk in bull market
                regime_metrics['return_adjustment'] = 1.2  # Higher expected returns
            elif current_regime == "bear_market":
                regime_metrics['risk_adjustment'] = 1.3  # Higher risk in bear market
                regime_metrics['return_adjustment'] = 0.7  # Lower expected returns
            elif current_regime == "high_volatility":
                regime_metrics['risk_adjustment'] = 1.5  # Much higher risk
                regime_metrics['return_adjustment'] = 1.0  # Neutral returns
            else:
                regime_metrics['risk_adjustment'] = 1.0
                regime_metrics['return_adjustment'] = 1.0
            
            return regime_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing regime impact: {e}")
            return {}
    
    def get_regime_history(self, days_back: int = 30) -> List[Dict]:
        """Get regime history for analysis"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        return [regime for regime in self.regime_history if regime['timestamp'] >= cutoff_date]
    
    def calculate_regime_transitions(self) -> Dict[str, int]:
        """Calculate regime transition frequencies"""
        if len(self.regime_history) < 2:
            return {}
        
        transitions = {}
        for i in range(1, len(self.regime_history)):
            from_regime = self.regime_history[i-1]['regime']
            to_regime = self.regime_history[i]['regime']
            transition_key = f"{from_regime}_to_{to_regime}"
            transitions[transition_key] = transitions.get(transition_key, 0) + 1
        
        return transitions

