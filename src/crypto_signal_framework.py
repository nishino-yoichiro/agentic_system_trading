"""
Crypto Trading Signal Framework
==============================

A comprehensive framework for generating orthogonal crypto trading signals
across multiple dimensions: instrument, mechanism, horizon, and session/regime.

Author: Quantitative Strategy Designer
Date: 2025-09-28
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0

class RegimeType(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    ASIA_SESSION = "asia"
    LONDON_SESSION = "london"
    NY_SESSION = "ny"

@dataclass
class Signal:
    """Trading signal with metadata"""
    signal_type: SignalType
    confidence: float  # 0-1
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    timestamp: pd.Timestamp = None
    strategy_name: str = ""
    risk_size: float = 0.0  # Position size as fraction of capital

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    symbol: str
    mechanism: str
    horizon: str
    session: str
    vol_target: float = 0.10  # 10% annualized volatility target
    max_weight: float = 0.25  # Maximum 25% allocation
    min_confidence: float = 0.6  # Minimum confidence to trade
    regime_filters: List[RegimeType] = None

class CryptoSignalFramework:
    """
    Main framework for crypto trading signals with orthogonal strategies
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategies = {}
        self.signal_history = []
        self.portfolio_weights = {}
        self.correlation_matrix = None
        self.regime_state = {}
        
        # Risk management parameters
        self.max_correlation = 0.5
        self.correlation_surge_threshold = 0.6
        self.max_drawdown_threshold = 0.20
        self.vol_target = 0.10
        
        # Execution controls
        self.max_concurrent_trades = 3
        self.min_bar_separation = 5
        self.active_trades = {}
        
    def add_strategy(self, config: StrategyConfig, signal_function: Callable):
        """Add a trading strategy to the framework"""
        self.strategies[config.name] = {
            'config': config,
            'function': signal_function,
            'returns': [],
            'sharpe': 0.0,
            'max_dd': 0.0,
            'hit_rate': 0.0
        }
        logger.info(f"Added strategy: {config.name} for {config.symbol}")
    
    def detect_regime(self, data: pd.DataFrame, symbol: str) -> Dict[RegimeType, bool]:
        """Detect current market regime"""
        if len(data) < 50:
            return {regime: False for regime in RegimeType}
        
        # Calculate regime indicators (optimized for large datasets)
        returns = data['close'].pct_change().dropna()
        
        # Use only recent data for regime detection to avoid hanging
        recent_data = data.tail(100)  # Only use last 100 data points for regime detection
        recent_returns = recent_data['close'].pct_change().dropna()
        
        if len(recent_returns) < 20:
            volatility = 0.0
            vol_percentile = 0.5
        else:
            volatility = recent_returns.rolling(20).std().iloc[-1]
            vol_percentile = (recent_returns.rolling(20).std().rank(pct=True)).iloc[-1]
        
        # Trend detection using recent data only
        high = recent_data['high']
        low = recent_data['low']
        close = recent_data['close']
        
        # Simple trend detection (optimized)
        if len(close) >= 20:
            sma_20 = close.rolling(20).mean().iloc[-1]
            price_vs_sma20 = close.iloc[-1] / sma_20
        else:
            price_vs_sma20 = 1.0
            
        if len(close) >= 50:
            sma_50 = close.rolling(50).mean().iloc[-1]
            price_vs_sma50 = close.iloc[-1] / sma_50
        else:
            price_vs_sma50 = 1.0
        
        # Session detection using data timestamp (not current time)
        data_timestamp = data.index[-1]
        if hasattr(data_timestamp, 'tz'):
            if data_timestamp.tz is None:
                ny_time = data_timestamp.tz_localize('UTC').tz_convert('America/New_York')
            else:
                ny_time = data_timestamp.tz_convert('America/New_York')
        else:
            ny_time = data_timestamp
        
        hour = ny_time.hour
        minute = ny_time.minute
        current_minutes = hour * 60 + minute
        
        # Session times (ET)
        asia_start = 21 * 60  # 9 PM ET (Asia open)
        asia_end = 5 * 60     # 5 AM ET (Asia close)
        london_start = 3 * 60  # 3 AM ET (London open)
        london_end = 12 * 60   # 12 PM ET (London close)
        ny_start = 9 * 60 + 30  # 9:30 AM ET (NY open)
        ny_end = 16 * 60        # 4 PM ET (NY close)
        
        # Check if in session (handle overnight sessions)
        in_asia = (current_minutes >= asia_start) or (current_minutes < asia_end)
        in_london = london_start <= current_minutes <= london_end
        in_ny = ny_start <= current_minutes <= ny_end
        
        regime = {
            RegimeType.TREND_UP: price_vs_sma20 > 1.02 and price_vs_sma50 > 1.01,
            RegimeType.TREND_DOWN: price_vs_sma20 < 0.98 and price_vs_sma50 < 0.99,
            RegimeType.RANGE: not (price_vs_sma20 > 1.02 or price_vs_sma20 < 0.98),
            RegimeType.HIGH_VOL: vol_percentile > 0.7,
            RegimeType.LOW_VOL: vol_percentile < 0.3,
            RegimeType.ASIA_SESSION: in_asia,
            RegimeType.LONDON_SESSION: in_london,
            RegimeType.NY_SESSION: in_ny
        }
        
        self.regime_state[symbol] = regime
        return regime
    
    def calculate_volatility_targeting(self, returns: pd.Series, target_vol: float = 0.10) -> float:
        """Calculate volatility targeting multiplier"""
        if len(returns) < 20:
            return 1.0
        
        realized_vol = returns.std() * np.sqrt(252)  # Annualized
        if realized_vol == 0:
            return 1.0
        
        vol_multiplier = target_vol / realized_vol
        return np.clip(vol_multiplier, 0.1, 10.0)  # Cap between 0.1x and 10x
    
    def calculate_correlation_matrix(self) -> np.ndarray:
        """Calculate correlation matrix between strategies"""
        if len(self.strategies) < 2:
            return np.eye(len(self.strategies))
        
        strategy_returns = []
        strategy_names = []
        
        for name, strategy in self.strategies.items():
            if len(strategy['returns']) > 10:  # Need minimum data
                strategy_returns.append(strategy['returns'])
                strategy_names.append(name)
        
        if len(strategy_returns) < 2:
            return np.eye(len(self.strategies))
        
        # Pad shorter series with zeros
        max_length = max(len(returns) for returns in strategy_returns)
        padded_returns = []
        
        for returns in strategy_returns:
            if len(returns) < max_length:
                padded = np.zeros(max_length)
                padded[-len(returns):] = returns
                padded_returns.append(padded)
            else:
                padded_returns.append(returns[-max_length:])
        
        correlation_matrix = np.corrcoef(padded_returns)
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def calculate_allocation_weights(self) -> Dict[str, float]:
        """Calculate optimal allocation weights using Sharpe tilt and correlation penalty"""
        if not self.strategies:
            return {}
        
        # Calculate correlation matrix
        self.calculate_correlation_matrix()
        
        # Start with equal weights
        weights = {name: 1.0 / len(self.strategies) for name in self.strategies.keys()}
        
        # Apply Sharpe tilt
        for name, strategy in self.strategies.items():
            if strategy['sharpe'] > 0:
                weights[name] *= max(strategy['sharpe'], 0)
            else:
                weights[name] *= 0.1  # Minimal weight for negative Sharpe
        
        # Apply correlation penalty
        if self.correlation_matrix is not None:
            strategy_names = list(self.strategies.keys())
            for i, name in enumerate(strategy_names):
                if i < len(self.correlation_matrix):
                    avg_correlation = np.mean(np.abs(self.correlation_matrix[i]))
                    if avg_correlation > 0:
                        weights[name] /= (1 + avg_correlation)
        
        # Check for correlation surge
        if self.correlation_matrix is not None:
            avg_correlation = np.mean(np.abs(self.correlation_matrix))
            if avg_correlation > self.correlation_surge_threshold:
                logger.warning(f"Correlation surge detected: {avg_correlation:.3f}")
                for name in weights:
                    weights[name] *= 0.5  # Halve all weights
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            for name in weights:
                weights[name] /= total_weight
        
        # Apply maximum weight cap
        for name in weights:
            weights[name] = min(weights[name], self.strategies[name]['config'].max_weight)
        
        # Renormalize after capping
        total_weight = sum(weights.values())
        if total_weight > 0:
            for name in weights:
                weights[name] /= total_weight
        
        self.portfolio_weights = weights
        return weights
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate signals for all strategies by iterating through each timestamp"""
        all_signals = []
        
        # Calculate allocation weights
        weights = self.calculate_allocation_weights()
        
        # Only log once per call
        if not hasattr(self, '_logged_strategies'):
            logger.info(f"Generating signals for {len(self.strategies)} strategies")
            self._logged_strategies = True
        
        for name, strategy in self.strategies.items():
            config = strategy['config']
            symbol = config.symbol
            
            if symbol not in data:
                continue
            
            symbol_data = data[symbol]
            if len(symbol_data) < 50:  # Need minimum data
                continue
            
            # Iterate through each timestamp to generate signals
            for i in range(50, len(symbol_data)):  # Start from index 50 to have enough history
                # Get data up to current timestamp
                current_data = symbol_data.iloc[:i+1].copy()
                
                # Detect regime for current data
                regime = self.detect_regime(current_data, symbol)
                
                # Check regime filters
                if config.regime_filters:
                    regime_ok = any(regime[filter_regime] for filter_regime in config.regime_filters)
                    if not regime_ok:
                        continue
                
                # Generate signal for current timestamp
                try:
                    signal = strategy['function'](current_data, regime)
                    if signal and signal.confidence >= config.min_confidence:
                        # Apply volatility targeting
                        returns = current_data['close'].pct_change().dropna()
                        vol_multiplier = self.calculate_volatility_targeting(returns, config.vol_target)
                        
                        # Set risk size based on allocation weight
                        signal.risk_size = weights.get(name, 0.0) * vol_multiplier
                        signal.strategy_name = name
                        signal.timestamp = current_data.index[-1]
                        
                        all_signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error generating signal for {name} at timestamp {current_data.index[-1]}: {e}")
                    continue
        
        # Apply execution controls (temporarily disabled for testing)
        filtered_signals = all_signals  # self.apply_execution_controls(all_signals)
        
        # Update signal history
        self.signal_history.extend(filtered_signals)
        
        return filtered_signals
    
    def apply_execution_controls(self, signals: List[Signal]) -> List[Signal]:
        """Apply execution controls and overlap management"""
        if not signals:
            return signals
        
        # Sort by confidence (highest first)
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered_signals = []
        symbol_trades = {}
        
        for signal in signals:
            symbol = signal.strategy_name.split('_')[0] if '_' in signal.strategy_name else 'BTC'
            
            # Check concurrency limits
            if len(self.active_trades) >= self.max_concurrent_trades:
                continue
            
            # Check symbol-specific limits
            if symbol in symbol_trades:
                if symbol_trades[symbol] >= 1:  # Max 1 trade per symbol
                    continue
            
            # Check minimum bar separation (simplified)
            recent_signals = [s for s in self.signal_history[-10:] 
                            if s.strategy_name == signal.strategy_name]
            if len(recent_signals) > 0:
                continue  # Skip if recent signal exists
            
            filtered_signals.append(signal)
            symbol_trades[symbol] = symbol_trades.get(symbol, 0) + 1
            self.active_trades[signal.strategy_name] = signal
        
        return filtered_signals
    
    def update_strategy_performance(self, strategy_name: str, returns: List[float]):
        """Update strategy performance metrics"""
        if strategy_name not in self.strategies:
            return
        
        strategy = self.strategies[strategy_name]
        strategy['returns'].extend(returns)
        
        if len(strategy['returns']) > 10:
            returns_array = np.array(strategy['returns'])
            
            # Calculate metrics
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            if std_return > 0:
                strategy['sharpe'] = mean_return / std_return * np.sqrt(252)
            
            # Calculate max drawdown
            cumulative = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            strategy['max_dd'] = np.min(drawdown)
            
            # Calculate hit rate
            strategy['hit_rate'] = np.mean(returns_array > 0)
    
    def check_health_switches(self) -> bool:
        """Check if any kill switches should be triggered"""
        # Check correlation surge
        if self.correlation_matrix is not None:
            avg_correlation = np.mean(np.abs(self.correlation_matrix))
            if avg_correlation > self.correlation_surge_threshold:
                logger.warning("Correlation surge detected - halving all weights")
                for name in self.portfolio_weights:
                    self.portfolio_weights[name] *= 0.5
                return True
        
        # Check individual strategy drawdowns
        for name, strategy in self.strategies.items():
            if strategy['max_dd'] < -self.max_drawdown_threshold:
                logger.warning(f"Strategy {name} hit max drawdown threshold")
                self.portfolio_weights[name] = 0.0
        
        return False
