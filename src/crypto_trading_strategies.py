#!/usr/bin/env python3
"""
Crypto Trading Strategies
=========================

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
from typing import Dict, List, Tuple, Optional, Any
try:
    from .crypto_signal_framework import Signal, SignalType, RegimeType, StrategyConfig
except ImportError:
    # Fallback for when running as script
    from crypto_signal_framework import Signal, SignalType, RegimeType, StrategyConfig
import logging

logger = logging.getLogger(__name__)

class CryptoTradingStrategies:
    """Collection of orthogonal crypto trading strategies"""
    
    def __init__(self):
        self.strategies = {}
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize trading strategies"""
        
        # 1. BTC NY Session Strategy
        self.strategies['btc_ny_session'] = {
            'config': StrategyConfig(
                name='btc_ny_session',
                symbol='BTC',
                mechanism='session_trade',
                horizon='intraday',
                session='ny',
                regime_filters=None
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
                mechanism='volatility_breakout',
                horizon='intraday',
                session='all',
                regime_filters=[RegimeType.LOW_VOL],
                min_confidence=0.8
            ),
            'function': self.volatility_expansion_breakout
        }
        
        # 5. Daily AVWAP Z-Score Reversion Strategy
        self.strategies['daily_avwap_zscore_reversion'] = {
            'config': StrategyConfig(
                name='daily_avwap_zscore_reversion',
                symbol='BTC',
                mechanism='mean_reversion',
                horizon='intraday',
                session='all',
                regime_filters=[RegimeType.HIGH_VOL],
                min_confidence=0.7
            ),
            'function': self.daily_avwap_zscore_reversion
        }
        
        # 6. Opening Range Break & Retest Strategy
        self.strategies['opening_range_break_retest'] = {
            'config': StrategyConfig(
                name='opening_range_break_retest',
                symbol='BTC',
                mechanism='breakout_retest',
                horizon='intraday',
                session='all',
                regime_filters=[RegimeType.TREND_UP, RegimeType.TREND_DOWN],
                min_confidence=0.75
            ),
            'function': self.opening_range_break_retest
        }
        
        # 7. Keltner Exhaustion Fade Strategy
        self.strategies['keltner_exhaustion_fade'] = {
            'config': StrategyConfig(
                name='keltner_exhaustion_fade',
                symbol='BTC',
                mechanism='exhaustion_fade',
                horizon='intraday',
                session='all',
                regime_filters=[RegimeType.HIGH_VOL],
                min_confidence=0.7
            ),
            'function': self.keltner_exhaustion_fade
        }
        
        # 8. Fakeout Reversion Strategy
        self.strategies['fakeout_reversion'] = {
            'config': StrategyConfig(
                name='fakeout_reversion',
                symbol='BTC',
                mechanism='fakeout_reversion',
                horizon='intraday',
                session='all',
                regime_filters=[RegimeType.LOW_VOL],
                min_confidence=0.75
            ),
            'function': self.fakeout_reversion
        }
    
    def get_strategy_configs(self) -> List[StrategyConfig]:
        """Get all strategy configurations"""
        return [strategy['config'] for strategy in self.strategies.values()]
    
    def get_strategy_function(self, strategy_name: str) -> Optional[callable]:
        """Get strategy function by name"""
        if strategy_name in self.strategies:
            return self.strategies[strategy_name]['function']
        return None
    
    def btc_ny_session_trade(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """BTC NY Session Strategy - Buy at NY open, sell at NY close"""
        if len(data) < 2:
            return None
        
        current_time = data.index[-1]
        current_close = data['close'].iloc[-1]
        
        # NY market hours (9:30 AM - 4:30 PM ET)
        ny_open_hour = 9
        ny_open_minute = 30
        ny_close_hour = 16
        ny_close_minute = 30
        
        # Convert to UTC (assuming data is in UTC)
        ny_open_time = current_time.replace(hour=ny_open_hour, minute=ny_open_minute, second=0, microsecond=0)
        ny_close_time = current_time.replace(hour=ny_close_hour, minute=ny_close_minute, second=0, microsecond=0)
        
        signal_type = SignalType.FLAT
        confidence = 0.0
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        # Check if we're in NY session
        if ny_open_time <= current_time <= ny_close_time:
            # Buy signal at NY open
            if current_time.hour == ny_open_hour and current_time.minute == ny_open_minute:
                signal_type = SignalType.LONG
                confidence = 0.80
                stop_loss = entry_price * 0.95  # 5% stop loss
                take_profit = entry_price * 1.05  # 5% take profit
                reason = "NY session open - Buy signal"
            
            # Sell signal at NY close
            elif current_time.hour == ny_close_hour and current_time.minute == ny_close_minute:
                signal_type = SignalType.SHORT
                confidence = 0.80
                stop_loss = entry_price * 1.05  # 5% stop loss
                take_profit = entry_price * 0.95  # 5% take profit
                reason = "NY session close - Sell signal"
        
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
    
    def liquidity_sweep_reversal(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """Liquidity Sweep Reversal Strategy"""
        if len(data) < 20:
            return None
        
        # Parameters
        swing_lookback = 20
        volume_multiplier = 1.5
        body_size_multiplier = 0.8
        stop_buffer = 0.005
        
        # Calculate swing highs and lows
        swing_highs = data['high'].rolling(swing_lookback).max()
        swing_lows = data['low'].rolling(swing_lookback).min()
        
        # Current bar data
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_close = data['close'].iloc[-1]
        current_open = data['open'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Volume analysis
        volume_mean = data['volume'].rolling(20).mean().iloc[-1]
        volume_confirmation = current_volume > volume_mean * volume_multiplier
        
        # Body size analysis
        body_size = abs(current_close - current_open)
        avg_body_size = abs(data['close'] - data['open']).rolling(20).mean().iloc[-1]
        body_confirmation = body_size > avg_body_size * body_size_multiplier
        
        signal_type = SignalType.FLAT
        confidence = 0.0
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        # Long signal: wick below swing low with volume confirmation
        if (current_low < swing_lows.iloc[-1] * (1 - stop_buffer) and 
            volume_confirmation and body_confirmation):
            
            signal_type = SignalType.LONG
            confidence = 0.75
            stop_loss = swing_lows.iloc[-1] * (1 - stop_buffer * 2)
            take_profit = swing_highs.iloc[-1] * (1 + stop_buffer)
            reason = f"Liquidity sweep reversal LONG - Low: {current_low:.2f}, Swing: {swing_lows.iloc[-1]:.2f}"
        
        # Short signal: wick above swing high with volume confirmation
        elif (current_high > swing_highs.iloc[-1] * (1 + stop_buffer) and 
              volume_confirmation and body_confirmation):
            
            signal_type = SignalType.SHORT
            confidence = 0.75
            stop_loss = swing_highs.iloc[-1] * (1 + stop_buffer * 2)
            take_profit = swing_lows.iloc[-1] * (1 - stop_buffer)
            reason = f"Liquidity sweep reversal SHORT - High: {current_high:.2f}, Swing: {swing_highs.iloc[-1]:.2f}"
        
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
        """Volume Weighted Trend Continuation Strategy"""
        if len(data) < 20:
            return None
        
        # Parameters
        trend_lookback = 20
        volume_lookback = 20
        slope_threshold = 0.0001
        volume_multiplier = 1.2
        
        # Calculate trend slope using linear regression
        prices = data['close'].tail(trend_lookback).values
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        trend_slope = slope / prices[-1]  # Normalize by current price
        
        # Volume analysis
        current_volume = data['volume'].iloc[-1]
        volume_mean = data['volume'].rolling(volume_lookback).mean().iloc[-1]
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
        """Volatility Expansion Breakout Strategy"""
        if len(data) < 20:
            return None
        
        # Parameters
        atr_period = 14
        compression_percentile = 30
        body_multiplier = 1.5
        volume_multiplier = 1.3
        trailing_atr_multiplier = 0.5
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(atr_period).mean()
        
        # Current bar data
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_close = data['close'].iloc[-1]
        current_open = data['open'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Body size analysis
        body_size = abs(current_close - current_open)
        avg_body_size = abs(data['close'] - data['open']).rolling(20).mean().iloc[-1]
        body_confirmation = body_size > avg_body_size * body_multiplier
        
        # Volume analysis
        volume_mean = data['volume'].rolling(20).mean().iloc[-1]
        volume_confirmation = current_volume > volume_mean * volume_multiplier
        
        # Volatility compression detection
        recent_atr = atr.tail(20)
        compression_threshold = recent_atr.quantile(compression_percentile / 100)
        is_compressed = recent_atr.iloc[-1] < compression_threshold
        
        signal_type = SignalType.FLAT
        confidence = 0.0
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        # Long signal: compression + breakout + volume confirmation
        if (is_compressed and body_confirmation and volume_confirmation and 
            current_close > current_open):
            
            signal_type = SignalType.LONG
            confidence = 0.80
            stop_loss = entry_price - (atr.iloc[-1] * trailing_atr_multiplier)
            take_profit = entry_price + (atr.iloc[-1] * 2)
            reason = f"Volatility expansion breakout LONG - ATR: {atr.iloc[-1]:.2f}, Vol: {current_volume/volume_mean:.2f}x"
        
        # Short signal: compression + breakdown + volume confirmation
        elif (is_compressed and body_confirmation and volume_confirmation and 
              current_close < current_open):
            
            signal_type = SignalType.SHORT
            confidence = 0.80
            stop_loss = entry_price + (atr.iloc[-1] * trailing_atr_multiplier)
            take_profit = entry_price - (atr.iloc[-1] * 2)
            reason = f"Volatility expansion breakout SHORT - ATR: {atr.iloc[-1]:.2f}, Vol: {current_volume/volume_mean:.2f}x"
        
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
    
    def daily_avwap_zscore_reversion(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """Daily AVWAP Z-Score Reversion Strategy"""
        if len(data) < 90:
            return None
        
        # Parameters
        z_threshold = 1.6
        lookback = 90
        volume_multiplier = 1.7
        atr_period = 14
        
        # Calculate daily AVWAP
        daily_data = data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(daily_data) < lookback:
            return None
        
        # Calculate AVWAP for each day
        daily_avwap = (daily_data['high'] + daily_data['low'] + daily_data['close']) / 3
        
        # Calculate Z-score
        avwap_mean = daily_avwap.rolling(lookback).mean().iloc[-1]
        avwap_std = daily_avwap.rolling(lookback).std().iloc[-1]
        current_price = data['close'].iloc[-1]
        z_score = (current_price - avwap_mean) / avwap_std if avwap_std > 0 else 0
        
        # Volume analysis
        current_volume = data['volume'].iloc[-1]
        volume_mean = data['volume'].rolling(20).mean().iloc[-1]
        volume_confirmation = current_volume > volume_mean * volume_multiplier
        
        # ATR for stop loss
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(atr_period).mean().iloc[-1]
        
        signal_type = SignalType.FLAT
        confidence = 0.0
        entry_price = current_price
        stop_loss = None
        take_profit = None
        reason = ""
        
        # Long signal: oversold snapback to AVWAP
        if (z_score < -z_threshold and volume_confirmation and 
            current_price < avwap_mean):
            
            signal_type = SignalType.LONG
            confidence = 0.70
            stop_loss = entry_price - (atr * 1.5)
            take_profit = avwap_mean
            reason = f"AVWAP Z-score reversion LONG - Z: {z_score:.2f}, AVWAP: {avwap_mean:.2f}"
        
        # Short signal: overbought snapback to AVWAP
        elif (z_score > z_threshold and volume_confirmation and 
              current_price > avwap_mean):
            
            signal_type = SignalType.SHORT
            confidence = 0.70
            stop_loss = entry_price + (atr * 1.5)
            take_profit = avwap_mean
            reason = f"AVWAP Z-score reversion SHORT - Z: {z_score:.2f}, AVWAP: {avwap_mean:.2f}"
        
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
    
    def opening_range_break_retest(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """Opening Range Break & Retest Strategy"""
        if len(data) < 30:
            return None
        
        # Parameters
        or_minutes = 30
        delta = 0.15
        volume_multiplier = 1.2
        retest_epsilon = 0.05
        atr_period = 14
        
        # Get opening range (first 30 minutes of day)
        daily_data = data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(daily_data) < 1:
            return None
        
        # Calculate opening range high and low
        current_day_start = data.index[-1].replace(hour=0, minute=0, second=0, microsecond=0)
        opening_range_data = data[data.index >= current_day_start].head(or_minutes)
        
        if len(opening_range_data) < or_minutes:
            return None
        
        or_high = opening_range_data['high'].max()
        or_low = opening_range_data['low'].min()
        or_range = or_high - or_low
        
        # Current bar data
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_close = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Volume analysis
        volume_mean = data['volume'].rolling(20).mean().iloc[-1]
        volume_confirmation = current_volume > volume_mean * volume_multiplier
        
        # ATR for stop loss
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(atr_period).mean().iloc[-1]
        
        signal_type = SignalType.FLAT
        confidence = 0.0
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        # Long signal: ORH retest after breakout
        if (current_close > or_high * (1 + delta) and 
            current_low <= or_high * (1 + retest_epsilon) and 
            volume_confirmation):
            
            signal_type = SignalType.LONG
            confidence = 0.75
            stop_loss = or_high * (1 - retest_epsilon)
            take_profit = entry_price + (atr * 2)
            reason = f"ORH retest LONG - ORH: {or_high:.2f}, Retest: {current_low:.2f}"
        
        # Short signal: ORL retest after breakdown
        elif (current_close < or_low * (1 - delta) and 
              current_high >= or_low * (1 - retest_epsilon) and 
              volume_confirmation):
            
            signal_type = SignalType.SHORT
            confidence = 0.75
            stop_loss = or_low * (1 + retest_epsilon)
            take_profit = entry_price - (atr * 2)
            reason = f"ORL retest SHORT - ORL: {or_low:.2f}, Retest: {current_high:.2f}"
        
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
    
    def keltner_exhaustion_fade(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """Keltner Exhaustion Fade Strategy"""
        if len(data) < 20:
            return None
        
        # Parameters
        ema_period = 20
        atr_period = 20
        multiplier = 2.2
        volume_multiplier = 2.0
        body_multiplier = 0.8
        
        # Calculate EMA
        ema = data['close'].ewm(span=ema_period).mean()
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(atr_period).mean()
        
        # Calculate Keltner bands
        upper_band = ema + (atr * multiplier)
        lower_band = ema - (atr * multiplier)
        
        # Current bar data
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_close = data['close'].iloc[-1]
        current_open = data['open'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Body size analysis
        body_size = abs(current_close - current_open)
        avg_body_size = abs(data['close'] - data['open']).rolling(20).mean().iloc[-1]
        body_confirmation = body_size > avg_body_size * body_multiplier
        
        # Volume analysis
        volume_mean = data['volume'].rolling(20).mean().iloc[-1]
        volume_confirmation = current_volume > volume_mean * volume_multiplier
        
        signal_type = SignalType.FLAT
        confidence = 0.0
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        # Long signal: lower band exhaustion fade
        if (current_low < lower_band.iloc[-1] and 
            current_close > lower_band.iloc[-1] and 
            body_confirmation and volume_confirmation):
            
            signal_type = SignalType.LONG
            confidence = 0.70
            stop_loss = lower_band.iloc[-1] * 0.995
            take_profit = ema.iloc[-1]
            reason = f"Keltner exhaustion fade LONG - Lower: {lower_band.iloc[-1]:.2f}, Close: {current_close:.2f}"
        
        # Short signal: upper band exhaustion fade
        elif (current_high > upper_band.iloc[-1] and 
              current_close < upper_band.iloc[-1] and 
              body_confirmation and volume_confirmation):
            
            signal_type = SignalType.SHORT
            confidence = 0.70
            stop_loss = upper_band.iloc[-1] * 1.005
            take_profit = ema.iloc[-1]
            reason = f"Keltner exhaustion fade SHORT - Upper: {upper_band.iloc[-1]:.2f}, Close: {current_close:.2f}"
        
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
    
    def fakeout_reversion(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """Fakeout Reversion Strategy"""
        if len(data) < 7 * 24 * 60:  # Need at least 7 days of minute data
            return None
        
        # Parameters
        compression_days = 7
        vol_z_threshold = 1.5
        vol_reversion_threshold = 1.0
        atr_compression_threshold = 0.3
        range_compression_threshold = 0.02
        similarity_threshold = 0.7
        
        # Calculate daily data
        daily_data = data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(daily_data) < compression_days:
            return None
        
        # Calculate compression metrics
        recent_highs = daily_data['high'].tail(compression_days)
        recent_lows = daily_data['low'].tail(compression_days)
        recent_range = (recent_highs.max() - recent_lows.min()) / recent_lows.min()
        
        # ATR compression
        high_low = daily_data['high'] - daily_data['low']
        high_close = np.abs(daily_data['high'] - daily_data['close'].shift())
        low_close = np.abs(daily_data['low'] - daily_data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean()
        atr_compression = atr.tail(compression_days).std() / atr.tail(compression_days).mean()
        
        # Volume analysis
        recent_volume = daily_data['volume'].tail(compression_days)
        vol_mean = recent_volume.mean()
        vol_std = recent_volume.std()
        vol_z_score = (recent_volume.iloc[-1] - vol_mean) / vol_std if vol_std > 0 else 0
        
        # Current bar data
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_close = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Check for compression and fakeout
        is_compressed = (recent_range < range_compression_threshold and 
                        atr_compression < atr_compression_threshold)
        
        # Fakeout detection
        fakeout_up = (current_high > recent_highs.max() and 
                     current_close < recent_highs.max() * 0.99)
        fakeout_down = (current_low < recent_lows.min() and 
                       current_close > recent_lows.min() * 1.01)
        
        signal_type = SignalType.FLAT
        confidence = 0.0
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        # Long signal: fakeout down + reversion up
        if (is_compressed and fakeout_down and 
            vol_z_score > vol_z_threshold):
            
            signal_type = SignalType.LONG
            confidence = 0.75
            stop_loss = recent_lows.min() * 0.98
            take_profit = recent_highs.max() * 1.02
            reason = f"Fakeout reversion LONG - Low: {current_low:.2f}, Range: {recent_range:.3f}"
        
        # Short signal: fakeout up + reversion down
        elif (is_compressed and fakeout_up and 
              vol_z_score > vol_z_threshold):
            
            signal_type = SignalType.SHORT
            confidence = 0.75
            stop_loss = recent_highs.max() * 1.02
            take_profit = recent_lows.min() * 0.98
            reason = f"Fakeout reversion SHORT - High: {current_high:.2f}, Range: {recent_range:.3f}"
        
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

