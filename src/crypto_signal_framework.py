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
from datetime import datetime
from collections import deque
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Strategy runtime categories for efficient data access"""
    CONSTANT_TIME = 1      # Only needs current row (e.g., time-based)
    FIXED_LOOKBACK = 2     # Needs fixed number of past bars (e.g., ATR, MA)
    FULL_CONTEXT = 3       # Needs full historical context (e.g., complex patterns)

@dataclass
class StrategyMetadata:
    """Strategy data requirements and configuration"""
    lookback: int = 0                    # Number of past bars needed
    fields_required: List[str] = None   # Subset of columns needed
    strategy_type: StrategyType = StrategyType.CONSTANT_TIME
    batch_mode: bool = False            # True if needs full context
    min_confidence: float = 0.0         # Minimum confidence threshold
    vol_target: float = 0.10            # Volatility target
    
    def __post_init__(self):
        if self.fields_required is None:
            self.fields_required = ["close", "timestamp"]

class BaseStrategy:
    """Base class for all trading strategies with metadata interface"""
    
    def __init__(self, name: str, metadata: StrategyMetadata):
        self.name = name
        self.metadata = metadata
        
    def generate_signal(self, current_row: pd.Series, history: Optional[pd.DataFrame] = None) -> Optional['Signal']:
        """
        Generate trading signal based on current data and optional history
        
        Args:
            current_row: Current market data row
            history: Historical data (only provided if strategy needs it)
            
        Returns:
            Signal object or None
        """
        raise NotImplementedError("Subclasses must implement generate_signal")
    
    def get_data_requirements(self) -> StrategyMetadata:
        """Return strategy's data requirements"""
        return self.metadata

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
    ASIA_SESSION = "asia_session"
    LONDON_SESSION = "london_session"
    NY_SESSION = "ny_session"

@dataclass
class Signal:
    """Trading signal with metadata"""
    signal_type: SignalType
    confidence: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    risk_size: float = 0.0
    strategy_name: str = ""
    timestamp: Optional[datetime] = None

@dataclass
class StrategyConfig:
    """Configuration for trading strategies"""
    name: str
    symbol: str
    mechanism: str
    horizon: str
    session: str
    vol_target: float = 0.10  # 10% annualized volatility target
    max_weight: float = 0.25  # Maximum 25% allocation
    min_confidence: float = 0.6  # Minimum confidence to trade
    regime_filters: List[RegimeType] = None

class SlidingWindowBuffer:
    """Memory-efficient sliding window buffer for strategy data"""
    
    def __init__(self, max_lookback: int):
        self.max_lookback = max_lookback
        self.buffer = deque(maxlen=max_lookback + 1)
        self.timestamps = deque(maxlen=max_lookback + 1)
        
    def add(self, row: pd.Series):
        """Add new data point to buffer"""
        self.buffer.append(row)
        self.timestamps.append(row.name)  # Store timestamp
        
    def get_history(self, lookback: int) -> Optional[pd.DataFrame]:
        """Get historical data for specified lookback period"""
        if len(self.buffer) < lookback:
            return None
            
        # Get last 'lookback' rows
        recent_data = list(self.buffer)[-lookback:]
        recent_timestamps = list(self.timestamps)[-lookback:]
        
        # Convert to DataFrame
        return pd.DataFrame(recent_data, index=recent_timestamps)
    
    def get_current(self) -> Optional[pd.Series]:
        """Get current data point"""
        return self.buffer[-1] if self.buffer else None
    
    def is_ready(self, lookback: int) -> bool:
        """Check if buffer has enough data for lookback"""
        return len(self.buffer) >= lookback

class SignalFramework:
    """Professional-grade signal framework with strategy metadata and sliding windows"""
    
    def __init__(self, max_lookback: int = 100):
        self.max_lookback = max_lookback
        self.strategies = {}
        self.buffers = {}  # symbol -> SlidingWindowBuffer
        self.signal_history = []
        self.portfolio_weights = {}
        self.regime_state = {}
        self.correlation_matrix = None
        
        # Risk management parameters
        self.max_drawdown_threshold = -0.15
        self.correlation_surge_threshold = 0.7
        
    def add_strategy(self, strategy: BaseStrategy):
        """Add a strategy with its metadata"""
        self.strategies[strategy.name] = strategy
        logger.info(f"Added strategy: {strategy.name} (type: {strategy.metadata.strategy_type.name})")
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], progress_bar=None, strategies: List[str] = None, live_mode: bool = False) -> List['Signal']:
        """
        Generate signals using strategy metadata and sliding window buffers
        
        This is the core optimization - O(n) complexity instead of O(n²)
        """
        logger.debug(f"generate_signals called with data keys: {list(data.keys())}")
        logger.debug(f"generate_signals called with strategies: {strategies}")
        
        all_signals = []
        
        # Initialize buffers for each symbol
        for symbol, symbol_data in data.items():
            self.buffers[symbol] = SlidingWindowBuffer(self.max_lookback)
        
        # Calculate allocation weights
        weights = self.calculate_allocation_weights()
        
        # Filter strategies if specified
        strategies_to_process = strategies if strategies else list(self.strategies.keys())
        logger.debug(f"Strategies to process: {strategies_to_process}")
        
        for name, strategy in self.strategies.items():
            # Only process requested strategies
            if name not in strategies_to_process:
                continue
            metadata = strategy.get_data_requirements()
            
            # Find matching symbol data
            symbol_data = None
            symbol = None
            for sym, df in data.items():
                if len(df) >= metadata.lookback + 1:  # Need at least lookback + current
                    symbol_data = df
                    symbol = sym
                    break
            
            if symbol_data is None:
                logger.warning(f"No suitable data found for strategy {name}")
                continue
            
            buffer = self.buffers[symbol]
            
            # Only log processing info at debug level to reduce noise
            logger.debug(f"Processing {name} with {metadata.strategy_type.name} access pattern")
            
            # Process data based on strategy type
            if metadata.strategy_type.name == "CONSTANT_TIME":
                # Only process relevant timestamps for constant-time strategies
                relevant_timestamps = self._get_relevant_timestamps(symbol_data, name, live_mode)
                
                if progress_bar:
                    progress_bar.set_description(f"Processing {name} (constant-time)")
                    progress_bar.update(len(relevant_timestamps))
                
                for timestamp in relevant_timestamps:
                    try:
                        # For live mode, use the most recent data row but with current timestamp
                        if live_mode and timestamp not in symbol_data.index:
                            current_row = symbol_data.iloc[-1].copy()
                            current_row.name = timestamp  # Set the timestamp for the strategy
                        else:
                            current_row = symbol_data.loc[timestamp]
                        
                        buffer.add(current_row)
                        
                        # Detect regime using recent data
                        regime_data = symbol_data.tail(50)
                        regime = self.detect_regime(regime_data, symbol)
                        
                        # Generate signal
                        signal = strategy.generate_signal(current_row, None)
                        if signal and signal.confidence >= metadata.min_confidence:
                            signal = self._apply_risk_management(signal, metadata, weights.get(name, 0.0), regime_data)
                            signal.strategy_name = name
                            signal.timestamp = timestamp
                            all_signals.append(signal)
                            
                    except Exception as e:
                        logger.error(f"Error generating signal for {name} at {timestamp}: {e}")
                        continue
                        
            elif metadata.strategy_type.name == "FIXED_LOOKBACK":
                # Use sliding window for fixed lookback strategies
                iteration_range = range(metadata.lookback, len(symbol_data))
                
                if progress_bar:
                    progress_bar.set_description(f"Processing {name} (fixed lookback: {metadata.lookback})")
                    for i in iteration_range:
                        if i % 100 == 0 and progress_bar:
                            progress_bar.set_postfix({"Strategy": name, "Signals": len(all_signals)})
                            progress_bar.update(100)
                        
                        try:
                            current_row = symbol_data.iloc[i]
                            buffer.add(current_row)
                            
                            if buffer.is_ready(metadata.lookback):
                                history = buffer.get_history(metadata.lookback)
                                regime = self.detect_regime(history, symbol)
                                
                                signal = strategy.generate_signal(current_row, history)
                                if signal and signal.confidence >= metadata.min_confidence:
                                    signal = self._apply_risk_management(signal, metadata, weights.get(name, 0.0), history)
                                    signal.strategy_name = name
                                    signal.timestamp = current_row.name
                                    all_signals.append(signal)
                                    
                        except Exception as e:
                            logger.error(f"Error generating signal for {name} at {current_row.name}: {e}")
                            continue
                
                # Update progress bar for remaining iterations
                remaining = len(iteration_range) % 100
                if remaining > 0 and progress_bar:
                    progress_bar.update(remaining)
                    
            elif metadata.strategy_type.name == "FULL_CONTEXT":
                # Full context strategies (legacy support)
                iteration_range = range(50, len(symbol_data))
                
                if progress_bar:
                    progress_bar.set_description(f"Processing {name} (full context)")
                    for i in iteration_range:
                        if i % 100 == 0 and progress_bar:
                            progress_bar.set_postfix({"Strategy": name, "Signals": len(all_signals)})
                            progress_bar.update(100)
                        
                        try:
                            current_row = symbol_data.iloc[i]
                            buffer.add(current_row)
                            
                            # Provide full context (but avoid copying)
                            history = symbol_data.iloc[:i+1]
                            regime = self.detect_regime(history, symbol)
                            
                            signal = strategy.generate_signal(current_row, history)
                            if signal and signal.confidence >= metadata.min_confidence:
                                signal = self._apply_risk_management(signal, metadata, weights.get(name, 0.0), history)
                                signal.strategy_name = name
                                signal.timestamp = current_row.name
                                all_signals.append(signal)
                                
                        except Exception as e:
                            logger.error(f"Error generating signal for {name} at {current_row.name}: {e}")
                            continue
                
                # Update progress bar for remaining iterations
                remaining = len(iteration_range) % 100
                if remaining > 0 and progress_bar:
                    progress_bar.update(remaining)
        
        # Update signal history
        self.signal_history.extend(all_signals)
        
        # Debug logging for troubleshooting
        logger.debug(f"Signal generation completed: {len(all_signals)} signals")
        if len(all_signals) > 0:
            logger.info(f"Generated {len(all_signals)} signals using optimized framework")
        else:
            logger.debug(f"Generated {len(all_signals)} signals using optimized framework")
        return all_signals
    
    def _apply_risk_management(self, signal: 'Signal', metadata: StrategyMetadata, 
                              weight: float, data: pd.DataFrame) -> 'Signal':
        """Apply volatility targeting and risk management"""
        try:
            # Apply volatility targeting
            returns = data['close'].pct_change().dropna()
            vol_multiplier = self.calculate_volatility_targeting(returns, metadata.vol_target)
            
            # Set risk size based on allocation weight
            signal.risk_size = weight * vol_multiplier
            
        except Exception as e:
            logger.warning(f"Error applying risk management: {e}")
            signal.risk_size = weight
            
        return signal
    
    def calculate_allocation_weights(self) -> Dict[str, float]:
        """Calculate portfolio allocation weights for strategies"""
        if not self.strategies:
            return {}
        
        # Simple equal weight allocation for now
        weight_per_strategy = 1.0 / len(self.strategies)
        return {name: weight_per_strategy for name in self.strategies.keys()}
    
    def detect_regime(self, data: pd.DataFrame, symbol: str) -> Dict['RegimeType', bool]:
        """Detect current market regime"""
        if len(data) < 50:
            return {regime: False for regime in RegimeType}
        
        # Calculate regime indicators (optimized for large datasets)
        returns = data['close'].pct_change().dropna()
        
        # Use only recent data for regime detection to avoid hanging
        recent_returns = returns.tail(50)
        
        # Volatility regime
        vol_percentile = recent_returns.rolling(window=20).std().rank(pct=True).iloc[-1]
        
        # Trend regime
        sma20 = data['close'].rolling(window=20).mean().iloc[-1]
        sma50 = data['close'].rolling(window=50).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        price_vs_sma20 = current_price / sma20 if sma20 > 0 else 1.0
        price_vs_sma50 = current_price / sma50 if sma50 > 0 else 1.0
        
        # Session detection
        current_time = data.index[-1]
        ny_time = current_time.tz_convert('US/Eastern') if hasattr(current_time, 'tz_convert') else current_time
        
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
    
    def _get_relevant_timestamps(self, symbol_data: pd.DataFrame, strategy_name: str, live_mode: bool = False) -> List[datetime]:
        """Get timestamps where signals might occur for simple strategies"""
        if strategy_name == 'btc_ny_session':
            if live_mode:
                # Live mode: only check if current time is NY session
                if len(symbol_data) > 0:
                    latest_timestamp = symbol_data.index[-1]
                    # Check if it's NY session time
                    ny_open_hour_utc = 14
                    ny_open_minute_utc = 30
                    ny_close_hour_utc = 21
                    ny_close_minute_utc = 0
                    
                    if ((latest_timestamp.hour == ny_open_hour_utc and latest_timestamp.minute == ny_open_minute_utc) or
                        (latest_timestamp.hour == ny_close_hour_utc and latest_timestamp.minute == ny_close_minute_utc)):
                        return [latest_timestamp]
                return []
            else:
                # Backtest mode: check all timestamps for NY session times
                relevant_times = []
                for timestamp in symbol_data.index:
                    # NY market hours in UTC (9:30 AM - 4:00 PM EST = 14:30 - 21:00 UTC)
                    ny_open_hour_utc = 14
                    ny_open_minute_utc = 30
                    ny_close_hour_utc = 21
                    ny_close_minute_utc = 0
                    
                    # Check if it's NY open or close time (exact time)
                    if ((timestamp.hour == ny_open_hour_utc and timestamp.minute == ny_open_minute_utc) or
                        (timestamp.hour == ny_close_hour_utc and timestamp.minute == ny_close_minute_utc)):
                        relevant_times.append(timestamp)
                
                return relevant_times
        elif strategy_name == 'test_every_minute':
            if live_mode:
                # Live mode: use current time, not historical data
                from datetime import datetime, timezone
                current_time = datetime.now(timezone.utc)
                current_minute = current_time.replace(second=0, microsecond=0)
                return [current_minute]
            else:
                # Backtest mode: check every minute
                return symbol_data.index.tolist()
        else:
            if live_mode:
                # Live mode: Use current time for all strategies
                # Let each strategy handle its own timing logic
                from datetime import datetime, timezone
                current_time = datetime.now(timezone.utc)
                return [current_time]
            else:
                # Backtest mode: check every hour to reduce processing
                return symbol_data.index[::60].tolist()  # Every 60 minutes

# Legacy compatibility - create alias for existing code
CryptoSignalFramework = SignalFramework