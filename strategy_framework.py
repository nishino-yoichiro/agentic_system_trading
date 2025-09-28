#!/usr/bin/env python3
"""
Strategy Implementation Framework
Implements the 5-edge basket from the portfolio management system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import talib

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Trading signal with metadata"""
    timestamp: datetime
    signal: int  # 1 = buy, -1 = sell, 0 = hold
    strength: float  # Signal strength (0-1)
    price: float
    reason: str
    strategy_name: str

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.signals = []
        self.positions = {}
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on market data"""
        pass
    
    def update_position(self, symbol: str, signal: Signal):
        """Update position based on signal"""
        if signal.signal == 1:  # Buy
            self.positions[symbol] = 1
        elif signal.signal == -1:  # Sell
            self.positions[symbol] = 0
        # Hold keeps current position
    
    def get_current_position(self, symbol: str) -> int:
        """Get current position for symbol"""
        return self.positions.get(symbol, 0)

class SweepReclaimStrategy(BaseStrategy):
    """Sweep/Reclaim strategy for index futures at NY open"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Sweep_Reclaim", config)
        self.sweep_threshold = config.get('sweep_threshold', 0.002)  # 0.2%
        self.reclaim_threshold = config.get('reclaim_threshold', 0.001)  # 0.1%
        self.lookback_periods = config.get('lookback_periods', 20)
        
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate sweep/reclaim signal"""
        
        if len(data) < self.lookback_periods:
            return Signal(
                timestamp=data.index[-1],
                signal=0,
                strength=0.0,
                price=data['close'].iloc[-1],
                reason="Insufficient data",
                strategy_name=self.name
            )
        
        # Get recent data
        recent = data.tail(self.lookback_periods)
        current_price = recent['close'].iloc[-1]
        high = recent['high'].max()
        low = recent['low'].min()
        
        # Check for sweep (price breaks recent high/low)
        sweep_up = current_price > high * (1 + self.sweep_threshold)
        sweep_down = current_price < low * (1 - self.sweep_threshold)
        
        # Check for reclaim (price comes back within range)
        reclaim_up = sweep_up and current_price < high * (1 - self.reclaim_threshold)
        reclaim_down = sweep_down and current_price > low * (1 + self.reclaim_threshold)
        
        if reclaim_up:
            signal = 1  # Buy on reclaim after sweep up
            strength = min(abs(current_price - high) / high, 1.0)
            reason = f"Reclaim after sweep up: {current_price:.2f} vs {high:.2f}"
        elif reclaim_down:
            signal = -1  # Sell on reclaim after sweep down
            strength = min(abs(current_price - low) / low, 1.0)
            reason = f"Reclaim after sweep down: {current_price:.2f} vs {low:.2f}"
        else:
            signal = 0
            strength = 0.0
            reason = "No sweep/reclaim pattern detected"
        
        return Signal(
            timestamp=data.index[-1],
            signal=signal,
            strength=strength,
            price=current_price,
            reason=reason,
            strategy_name=self.name
        )

class BreakoutContinuationStrategy(BaseStrategy):
    """Breakout continuation strategy for NQ during high-vol regime"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Breakout_Continuation", config)
        self.breakout_threshold = config.get('breakout_threshold', 0.01)  # 1%
        self.volume_threshold = config.get('volume_threshold', 1.5)  # 1.5x average
        self.lookback_periods = config.get('lookback_periods', 20)
        
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate breakout continuation signal"""
        
        if len(data) < self.lookback_periods:
            return Signal(
                timestamp=data.index[-1],
                signal=0,
                strength=0.0,
                price=data['close'].iloc[-1],
                reason="Insufficient data",
                strategy_name=self.name
            )
        
        # Get recent data
        recent = data.tail(self.lookback_periods)
        current_price = recent['close'].iloc[-1]
        current_volume = recent['volume'].iloc[-1]
        avg_volume = recent['volume'].mean()
        
        # Calculate breakout levels
        resistance = recent['high'].max()
        support = recent['low'].min()
        range_size = resistance - support
        
        # Check for breakout
        breakout_up = current_price > resistance + (range_size * self.breakout_threshold)
        breakout_down = current_price < support - (range_size * self.breakout_threshold)
        
        # Check volume confirmation
        volume_confirmed = current_volume > avg_volume * self.volume_threshold
        
        if breakout_up and volume_confirmed:
            signal = 1  # Buy on breakout up
            strength = min((current_price - resistance) / resistance, 1.0)
            reason = f"Breakout up with volume: {current_price:.2f} > {resistance:.2f}"
        elif breakout_down and volume_confirmed:
            signal = -1  # Sell on breakout down
            strength = min((support - current_price) / support, 1.0)
            reason = f"Breakout down with volume: {current_price:.2f} < {support:.2f}"
        else:
            signal = 0
            strength = 0.0
            reason = "No breakout pattern or insufficient volume"
        
        return Signal(
            timestamp=data.index[-1],
            signal=signal,
            strength=strength,
            price=current_price,
            reason=reason,
            strategy_name=self.name
        )

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy for SPY overnight gap fade"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Mean_Reversion", config)
        self.gap_threshold = config.get('gap_threshold', 0.005)  # 0.5%
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.lookback_periods = config.get('lookback_periods', 14)
        
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate mean reversion signal"""
        
        if len(data) < self.lookback_periods:
            return Signal(
                timestamp=data.index[-1],
                signal=0,
                strength=0.0,
                price=data['close'].iloc[-1],
                reason="Insufficient data",
                strategy_name=self.name
            )
        
        # Get recent data
        recent = data.tail(self.lookback_periods)
        current_price = recent['close'].iloc[-1]
        
        # Calculate gap (overnight move)
        if len(data) >= 2:
            prev_close = data['close'].iloc[-2]
            gap = (current_price - prev_close) / prev_close
        else:
            gap = 0
        
        # Calculate RSI
        rsi = talib.RSI(recent['close'].values, timeperiod=14)
        current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
        
        # Check for mean reversion setup
        gap_up = gap > self.gap_threshold
        gap_down = gap < -self.gap_threshold
        rsi_oversold = current_rsi < self.rsi_oversold
        rsi_overbought = current_rsi > self.rsi_overbought
        
        if gap_up and rsi_overbought:
            signal = -1  # Sell on gap up with overbought RSI
            strength = min(abs(gap), 1.0)
            reason = f"Gap up fade: gap={gap:.3f}, RSI={current_rsi:.1f}"
        elif gap_down and rsi_oversold:
            signal = 1  # Buy on gap down with oversold RSI
            strength = min(abs(gap), 1.0)
            reason = f"Gap down fade: gap={gap:.3f}, RSI={current_rsi:.1f}"
        else:
            signal = 0
            strength = 0.0
            reason = f"No mean reversion setup: gap={gap:.3f}, RSI={current_rsi:.1f}"
        
        return Signal(
            timestamp=data.index[-1],
            signal=signal,
            strength=strength,
            price=current_price,
            reason=reason,
            strategy_name=self.name
        )

class FXCarryTrendStrategy(BaseStrategy):
    """FX Carry/Trend strategy with MA filter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("FX_Carry_Trend", config)
        self.fast_ma = config.get('fast_ma', 10)
        self.slow_ma = config.get('slow_ma', 20)
        self.carry_threshold = config.get('carry_threshold', 0.001)  # 0.1%
        self.lookback_periods = config.get('lookback_periods', 20)
        
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate FX carry/trend signal"""
        
        if len(data) < self.lookback_periods:
            return Signal(
                timestamp=data.index[-1],
                signal=0,
                strength=0.0,
                price=data['close'].iloc[-1],
                reason="Insufficient data",
                strategy_name=self.name
            )
        
        # Get recent data
        recent = data.tail(self.lookback_periods)
        current_price = recent['close'].iloc[-1]
        
        # Calculate moving averages
        fast_ma = recent['close'].rolling(window=self.fast_ma).mean()
        slow_ma = recent['close'].rolling(window=self.slow_ma).mean()
        
        current_fast_ma = fast_ma.iloc[-1]
        current_slow_ma = slow_ma.iloc[-1]
        
        # Check for trend
        trend_up = current_fast_ma > current_slow_ma
        trend_down = current_fast_ma < current_slow_ma
        
        # Calculate carry (simplified as price momentum)
        if len(recent) >= 5:
            carry = (current_price - recent['close'].iloc[-5]) / recent['close'].iloc[-5]
        else:
            carry = 0
        
        # Generate signal
        if trend_up and carry > self.carry_threshold:
            signal = 1  # Buy on uptrend with positive carry
            strength = min(abs(carry), 1.0)
            reason = f"Uptrend with carry: MA={current_fast_ma:.4f}>{current_slow_ma:.4f}, carry={carry:.3f}"
        elif trend_down and carry < -self.carry_threshold:
            signal = -1  # Sell on downtrend with negative carry
            strength = min(abs(carry), 1.0)
            reason = f"Downtrend with carry: MA={current_fast_ma:.4f}<{current_slow_ma:.4f}, carry={carry:.3f}"
        else:
            signal = 0
            strength = 0.0
            reason = f"No trend/carry setup: MA={current_fast_ma:.4f} vs {current_slow_ma:.4f}, carry={carry:.3f}"
        
        return Signal(
            timestamp=data.index[-1],
            signal=signal,
            strength=strength,
            price=current_price,
            reason=reason,
            strategy_name=self.name
        )

class OptionsIVCrushStrategy(BaseStrategy):
    """Options IV Crush strategy for event-driven trades"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Options_IV_Crush", config)
        self.iv_threshold = config.get('iv_threshold', 0.3)  # 30% IV
        self.iv_crush_threshold = config.get('iv_crush_threshold', 0.05)  # 5% IV crush
        self.lookback_periods = config.get('lookback_periods', 20)
        
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate options IV crush signal"""
        
        if len(data) < self.lookback_periods:
            return Signal(
                timestamp=data.index[-1],
                signal=0,
                strength=0.0,
                price=data['close'].iloc[-1],
                reason="Insufficient data",
                strategy_name=self.name
            )
        
        # Get recent data
        recent = data.tail(self.lookback_periods)
        current_price = recent['close'].iloc[-1]
        
        # Calculate implied volatility (simplified as price volatility)
        returns = recent['close'].pct_change().dropna()
        current_iv = returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate IV crush (recent IV vs historical average)
        avg_iv = returns.rolling(window=10).std().mean() * np.sqrt(252)
        iv_crush = (current_iv - avg_iv) / avg_iv if avg_iv > 0 else 0
        
        # Check for IV crush setup
        high_iv = current_iv > self.iv_threshold
        iv_crushing = iv_crush < -self.iv_crush_threshold
        
        if high_iv and not iv_crushing:
            signal = 1  # Buy straddle when IV is high but not crushing yet
            strength = min(current_iv / self.iv_threshold, 1.0)
            reason = f"High IV setup: IV={current_iv:.3f}, crush={iv_crush:.3f}"
        elif iv_crushing:
            signal = -1  # Sell when IV is crushing
            strength = min(abs(iv_crush), 1.0)
            reason = f"IV crush: IV={current_iv:.3f}, crush={iv_crush:.3f}"
        else:
            signal = 0
            strength = 0.0
            reason = f"No IV setup: IV={current_iv:.3f}, crush={iv_crush:.3f}"
        
        return Signal(
            timestamp=data.index[-1],
            signal=signal,
            strength=strength,
            price=current_price,
            reason=reason,
            strategy_name=self.name
        )

class StrategyFramework:
    """Framework for managing multiple strategies"""
    
    def __init__(self):
        self.strategies = {}
        self.signal_history = []
        
    def add_strategy(self, strategy: BaseStrategy):
        """Add a strategy to the framework"""
        self.strategies[strategy.name] = strategy
        logger.info(f"Added strategy: {strategy.name}")
        
    def generate_all_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[Signal]]:
        """Generate signals for all strategies"""
        
        all_signals = {}
        
        for strategy_name, strategy in self.strategies.items():
            # Try to find matching data by strategy name or symbol
            matching_data = None
            for data_name, df in data.items():
                # More flexible matching
                if (strategy_name in data_name or 
                    data_name in strategy_name or
                    any(part in data_name for part in strategy_name.split('_')) or
                    any(part in strategy_name for part in data_name.split('_'))):
                    matching_data = df
                    break
            
            if matching_data is not None:
                signal = strategy.generate_signal(matching_data)
                all_signals[strategy_name] = [signal]
                
                # Store in history
                self.signal_history.append(signal)
                
                logger.debug(f"Generated signal for {strategy_name}: {signal.signal} ({signal.reason})")
            else:
                logger.warning(f"No matching data found for strategy {strategy_name}")
        
        return all_signals
    
    def get_strategy_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get strategy configurations for portfolio manager"""
        
        configs = {}
        
        for strategy_name, strategy in self.strategies.items():
            configs[strategy_name] = {
                'regime_filter': strategy.config.get('regime_filter', 'all'),
                'instrument_class': strategy.config.get('instrument_class', 'equity'),
                'mechanism': strategy.config.get('mechanism', 'trend'),
                'horizon': strategy.config.get('horizon', 'intraday'),
                'session': strategy.config.get('session', 'NY')
            }
        
        return configs
    
    def create_default_strategies(self) -> Dict[str, BaseStrategy]:
        """Create the default 5-edge strategy basket"""
        
        strategies = {}
        
        # 1. Sweep/Reclaim (intraday) on index futures (NY open)
        sweep_config = {
            'regime_filter': 'all',
            'instrument_class': 'futures',
            'mechanism': 'sweep_reclaim',
            'horizon': 'intraday',
            'session': 'NY',
            'sweep_threshold': 0.002,
            'reclaim_threshold': 0.001,
            'lookback_periods': 20
        }
        strategies['Sweep_Reclaim'] = SweepReclaimStrategy(sweep_config)
        
        # 2. Breakout Continuation (intraday) on NQ during high-vol regime
        breakout_config = {
            'regime_filter': 'high_vol',
            'instrument_class': 'futures',
            'mechanism': 'breakout',
            'horizon': 'intraday',
            'session': 'NY',
            'breakout_threshold': 0.01,
            'volume_threshold': 1.5,
            'lookback_periods': 20
        }
        strategies['Breakout_Continuation'] = BreakoutContinuationStrategy(breakout_config)
        
        # 3. Mean Reversion (overnight) on SPY in low-vol regimes
        mean_rev_config = {
            'regime_filter': 'low_vol',
            'instrument_class': 'equity',
            'mechanism': 'mean_reversion',
            'horizon': 'overnight',
            'session': 'NY',
            'gap_threshold': 0.005,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'lookback_periods': 14
        }
        strategies['Mean_Reversion'] = MeanReversionStrategy(mean_rev_config)
        
        # 4. FX Carry/Trend (swing) with simple MA filter
        fx_config = {
            'regime_filter': 'all',
            'instrument_class': 'fx',
            'mechanism': 'carry',
            'horizon': 'swing',
            'session': 'London',
            'fast_ma': 10,
            'slow_ma': 20,
            'carry_threshold': 0.001,
            'lookback_periods': 20
        }
        strategies['FX_Carry_Trend'] = FXCarryTrendStrategy(fx_config)
        
        # 5. Options IV Crush (event-driven)
        options_config = {
            'regime_filter': 'event',
            'instrument_class': 'options',
            'mechanism': 'volatility',
            'horizon': 'event',
            'session': 'all',
            'iv_threshold': 0.3,
            'iv_crush_threshold': 0.05,
            'lookback_periods': 20
        }
        strategies['Options_IV_Crush'] = OptionsIVCrushStrategy(options_config)
        
        return strategies

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create strategy framework
    framework = StrategyFramework()
    
    # Create default strategies
    strategies = framework.create_default_strategies()
    
    # Add strategies to framework
    for strategy in strategies.values():
        framework.add_strategy(strategy)
    
    print("Strategy framework created with 5-edge basket:")
    for name in strategies.keys():
        print(f"  - {name}")
    
    print("\nStrategy configurations:")
    configs = framework.get_strategy_configs()
    for name, config in configs.items():
        print(f"  {name}: {config}")
