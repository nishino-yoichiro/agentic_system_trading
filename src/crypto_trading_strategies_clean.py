"""
Crypto Trading Strategies
========================

Orthogonal trading strategies for crypto markets across multiple dimensions:
- Instrument: BTC, ETH, major alts
- Mechanism: liquidity sweep, breakout, mean reversion, funding arbitrage
- Horizon: intraday, swing, positional
- Session/Regime: Asia, London, NY sessions with volatility filters

Author: Quantitative Strategy Designer
Date: 2025-09-28
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from crypto_signal_framework import Signal, SignalType, RegimeType, StrategyConfig
import logging

logger = logging.getLogger(__name__)

class CryptoTradingStrategies:
    """Collection of orthogonal crypto trading strategies"""
    
    def __init__(self):
        self.strategies = {}
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all trading strategies"""
        
        # 1. BTC NY Session Buy/Sell
        self.strategies['btc_ny_session'] = {
            'config': StrategyConfig(
                name='btc_ny_session',
                symbol='BTC',
                mechanism='session_trade',
                horizon='intraday',
                session='ny',
                regime_filters=None  # Remove regime filters for testing
            ),
            'function': self.btc_ny_session_trade
        }
        
        # 2. Liquidity Sweep Reversal Strategy
        self.strategies['liquidity_sweep_reversal'] = {
            'config': StrategyConfig(
                name='liquidity_sweep_reversal',
                symbol='BTC',
                mechanism='liquidity_sweep',
                horizon='intraday',
                session='all',
                regime_filters=[RegimeType.HIGH_VOL],
                min_confidence=0.7
            ),
            'function': self.liquidity_sweep_reversal
        }
        
        # 3. Volume Weighted Trend Continuation Strategy
        self.strategies['volume_weighted_trend_continuation'] = {
            'config': StrategyConfig(
                name='volume_weighted_trend_continuation',
                symbol='BTC',
                mechanism='trend_continuation',
                horizon='intraday',
                session='all',
                regime_filters=[RegimeType.TREND_UP, RegimeType.TREND_DOWN],
                min_confidence=0.6
            ),
            'function': self.volume_weighted_trend_continuation
        }
        
        # 4. Volatility Expansion Breakout Strategy
        self.strategies['volatility_expansion_breakout'] = {
            'config': StrategyConfig(
                name='volatility_expansion_breakout',
                symbol='BTC',
                mechanism='volatility_expansion',
                horizon='intraday',
                session='all',
                regime_filters=[RegimeType.LOW_VOL, RegimeType.HIGH_VOL],
                min_confidence=0.7
            ),
            'function': self.volatility_expansion_breakout
        }
    
    def liquidity_sweep_reversal(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """
        Liquidity Sweep Reversal Strategy
        
        Detects liquidity hunts (stop-loss sweeps) beyond recent swing points with volume confirmation,
        then trades the reversal back inside the range.
        
        Edge: Exploits market maker behavior of hunting stops beyond obvious levels, then reversing.
        Non-collinear with trend/mean-reversion as it's purely structure-based.
        """
        if len(data) < 25:
            return None
        
        # Parameters
        swing_lookback = 20
        volume_multiplier = 1.5
        body_size_multiplier = 0.8
        stop_buffer = 0.005
        
        # Calculate swing levels
        recent_swing_high = data['high'].rolling(swing_lookback).max().iloc[-1]
        recent_swing_low = data['low'].rolling(swing_lookback).min().iloc[-1]
        
        # Current bar data
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_close = data['close'].iloc[-1]
        current_open = data['open'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Calculate indicators
        volume_mean = data['volume'].rolling(20).mean().iloc[-1]
        body_size_mean = abs(data['close'] - data['open']).rolling(20).mean().iloc[-1]
        current_body_size = abs(current_close - current_open)
        
        # Check for wick above swing high (short signal)
        wick_above_swing = (current_high > recent_swing_high and 
                           current_close < recent_swing_high)
        
        # Check for wick below swing low (long signal)
        wick_below_swing = (current_low < recent_swing_low and 
                           current_close > recent_swing_low)
        
        # Volume confirmation
        volume_spike = current_volume > volume_mean * volume_multiplier
        
        # Body size confirmation
        body_confirmation = current_body_size > body_size_mean * body_size_multiplier
        
        # Reversal confirmation
        reversal_up = current_close > current_open
        reversal_down = current_close < current_open
        
        signal_type = SignalType.FLAT
        confidence = 0.0
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        # Short signal: wick above swing high + reversal down
        if (wick_above_swing and volume_spike and body_confirmation and 
            reversal_down and current_close < recent_swing_high):
            signal_type = SignalType.SHORT
            confidence = 0.75
            stop_loss = recent_swing_high * (1 + stop_buffer)
            take_profit = (recent_swing_high + recent_swing_low) / 2
            reason = f"Liquidity sweep reversal SHORT - wick above swing high {recent_swing_high:.2f}"
            
        # Long signal: wick below swing low + reversal up
        elif (wick_below_swing and volume_spike and body_confirmation and 
              reversal_up and current_close > recent_swing_low):
            signal_type = SignalType.LONG
            confidence = 0.75
            stop_loss = recent_swing_low * (1 - stop_buffer)
            take_profit = (recent_swing_high + recent_swing_low) / 2
            reason = f"Liquidity sweep reversal LONG - wick below swing low {recent_swing_low:.2f}"
        
        if signal_type != SignalType.FLAT:
            return Signal(
                signal_type=signal_type,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason
            )
        
        return None
    
    def volume_weighted_trend_continuation(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """
        Volume Weighted Trend Continuation Strategy
        
        Trades trend continuation only when volume confirms real participation,
        using linear regression slope for trend direction and volume-weighted confirmation.
        
        Edge: Combines trend following with volume confirmation to avoid false breakouts.
        Non-collinear with mean reversion as it's trend-continuation based.
        """
        if len(data) < 25:
            return None
        
        # Parameters
        trend_lookback = 20
        volume_lookback = 20
        slope_threshold = 0.0001
        volume_multiplier = 1.2
        
        # Calculate trend slope using linear regression
        def linear_regression_slope(series, window):
            if len(series) < window:
                return 0
            x = np.arange(window)
            y = series.iloc[-window:].values
            if len(y) != window or np.isnan(y).any():
                return 0
            try:
                slope = np.polyfit(x, y, 1)[0]
                return slope
            except:
                return 0
        
        trend_slope = linear_regression_slope(data['close'], trend_lookback)
        
        # Volume analysis
        volume_mean = data['volume'].rolling(volume_lookback).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        volume_confirmation = current_volume > volume_mean * volume_multiplier
        
        # Price levels for pullback detection
        close_mean_5 = data['close'].rolling(5).mean().iloc[-1]
        close_mean_20 = data['close'].rolling(20).mean().iloc[-1]
        current_close = data['close'].iloc[-1]
        prev_close = data['close'].iloc[-2] if len(data) > 1 else current_close
        
        # Momentum confirmation
        momentum_confirmation = current_close > prev_close
        
        # Volume acceleration
        prev_volume = data['volume'].iloc[-2] if len(data) > 1 else current_volume
        volume_acceleration = current_volume > prev_volume * 1.1
        
        signal_type = SignalType.FLAT
        confidence = 0.0
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        # Long signal: uptrend + volume confirmation + pullback entry
        if (trend_slope > slope_threshold and 
            volume_confirmation and 
            current_close < close_mean_5 and 
            current_close > close_mean_20 and
            momentum_confirmation and
            volume_acceleration):
            
            signal_type = SignalType.LONG
            confidence = 0.70
            stop_loss = entry_price * 0.98  # 2% stop loss
            take_profit = entry_price * 1.03  # 3% take profit
            reason = f"Volume-weighted trend continuation LONG - slope: {trend_slope:.6f}, vol: {current_volume/volume_mean:.2f}x"
            
        # Short signal: downtrend + volume confirmation + pullback entry
        elif (trend_slope < -slope_threshold and 
              volume_confirmation and 
              current_close > close_mean_5 and 
              current_close < close_mean_20 and
              not momentum_confirmation and
              volume_acceleration):
            
            signal_type = SignalType.SHORT
            confidence = 0.70
            stop_loss = entry_price * 1.02  # 2% stop loss
            take_profit = entry_price * 0.97  # 3% take profit
            reason = f"Volume-weighted trend continuation SHORT - slope: {trend_slope:.6f}, vol: {current_volume/volume_mean:.2f}x"
        
        if signal_type != SignalType.FLAT:
            return Signal(
                signal_type=signal_type,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason
            )
        
        return None
    
    def volatility_expansion_breakout(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """
        Volatility Expansion Breakout Strategy
        
        Detects prolonged volatility compression, waits for expansion breakout with volume confirmation,
        then trades the follow-through momentum.
        
        Edge: Exploits the natural cycle of compression and expansion in crypto markets.
        Non-collinear with other strategies as it's purely volatility-based.
        """
        if len(data) < 50:
            return None
        
        # Parameters
        atr_period = 14
        compression_percentile = 30
        body_multiplier = 1.5
        volume_multiplier = 1.3
        trailing_atr_multiplier = 0.5
        
        # Calculate ATR
        def calculate_atr(data, period):
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            return atr
        
        atr_14 = calculate_atr(data, atr_period)
        atr_5 = calculate_atr(data, 5)
        
        # Current values
        current_atr_14 = atr_14.iloc[-1]
        current_atr_5 = atr_5.iloc[-1]
        current_close = data['close'].iloc[-1]
        current_open = data['open'].iloc[-1]
        current_high = data['high'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Calculate compression
        atr_percentile_30 = atr_14.rolling(50).quantile(0.3).iloc[-1]
        compression = current_atr_14 < atr_percentile_30
        
        # Check compression duration (simplified - look at last 10 bars)
        compression_duration = (atr_14.iloc[-10:] < atr_percentile_30).sum() >= 10
        
        # Breakout detection
        body_size = abs(current_close - current_open)
        body_size_mean = abs(data['close'] - data['open']).rolling(20).mean().iloc[-1]
        breakout_candle = body_size > body_size_mean * body_multiplier
        
        # Volume confirmation
        volume_mean = data['volume'].rolling(20).mean().iloc[-1]
        volume_confirmation = current_volume > volume_mean * volume_multiplier
        
        # ATR acceleration
        atr_acceleration = current_atr_5 > current_atr_14 * 1.2
        
        # Breakout direction
        recent_high_10 = data['high'].rolling(10).max().iloc[-1]
        breakout_direction = current_close > current_open and current_close > recent_high_10
        
        signal_type = SignalType.FLAT
        confidence = 0.0
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        # Long signal: compression + breakout + volume + acceleration
        if (compression and compression_duration and 
            breakout_candle and volume_confirmation and 
            atr_acceleration and breakout_direction):
            
            signal_type = SignalType.LONG
            confidence = 0.80
            # Trailing stop will be calculated dynamically
            stop_loss = current_high - current_atr_14 * trailing_atr_multiplier
            take_profit = entry_price * 1.05  # 5% take profit
            reason = f"Volatility expansion breakout LONG - ATR: {current_atr_14:.4f}, vol: {current_volume/volume_mean:.2f}x"
        
        if signal_type != SignalType.FLAT:
            return Signal(
                signal_type=signal_type,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason
            )
        
        return None
    
    def btc_ny_session_trade(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """
        BTC NY Session Buy/Sell Strategy
        
        Principle: Simple buy at NY open, sell at NY close
        - Buy when NY session starts (9:30 AM ET)
        - Sell when NY session ends (4:00 PM ET)
        - Target: Capture intraday NY session moves
        """
        if len(data) < 1:
            return None
        
        # Get current time in NY timezone
        current_time = data.index[-1]
        if hasattr(current_time, 'tz'):
            if current_time.tz is None:
                ny_time = current_time.tz_localize('UTC').tz_convert('America/New_York')
            else:
                ny_time = current_time.tz_convert('America/New_York')
        else:
            ny_time = current_time
        
        hour = ny_time.hour
        minute = ny_time.minute
        
        # NY session times (ET)
        ny_open_time = 9 * 60 + 30  # 9:30 AM in minutes
        ny_close_time = 16 * 60      # 4:00 PM in minutes
        current_minutes = hour * 60 + minute
        
        # Check if we're in NY session
        in_ny_session = ny_open_time <= current_minutes <= ny_close_time
        
        # Simple: Generate signals for any NY session time
        # Buy in first half of session, sell in second half
        session_midpoint = (ny_open_time + ny_close_time) // 2  # ~12:45 PM
        
        is_morning = ny_open_time <= current_minutes <= session_midpoint
        is_afternoon = session_midpoint < current_minutes <= ny_close_time
        
        current_close = data['close'].iloc[-1]
        
        confidence = 0.0
        signal_type = SignalType.FLAT
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        # Simple NY session strategy: Buy at 9:30 AM ET, Sell at 4:30 PM ET
        # Buy at exactly 9:30 AM ET
        if hour == 9 and minute == 30:
            signal_type = SignalType.LONG
            confidence = 0.80
            stop_loss = entry_price * 0.95  # 5% stop loss
            take_profit = entry_price * 1.05  # 5% take profit
            reason = f"BTC NY open buy - {ny_time.strftime('%H:%M')} ET"
            
        # Sell at exactly 4:30 PM ET
        elif hour == 16 and minute == 30:
            signal_type = SignalType.SHORT
            confidence = 0.80
            stop_loss = entry_price * 1.05  # 5% stop loss
            take_profit = entry_price * 0.95  # 5% take profit
            reason = f"BTC NY close sell - {ny_time.strftime('%H:%M')} ET"
        
        if signal_type != SignalType.FLAT:
            return Signal(
                signal_type=signal_type,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason
            )
        
        return None
    
    def get_strategy_configs(self) -> List[StrategyConfig]:
        """Get all strategy configurations"""
        return [strategy['config'] for strategy in self.strategies.values()]
    
    def get_all_strategies(self) -> Dict:
        """Get all available strategies"""
        return self.strategies