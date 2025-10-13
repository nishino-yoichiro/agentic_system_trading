#!/usr/bin/env python3
"""
Professional Crypto Trading Strategies
======================================

Converted legacy strategies to use the new metadata-driven framework.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
import logging
import talib

from crypto_signal_framework import BaseStrategy, StrategyMetadata, StrategyType, Signal, SignalType, RegimeType

logger = logging.getLogger(__name__)

def create_signal(signal_type: SignalType, confidence: float, entry_price: float, 
                  timestamp, reason: str, strategy_name: str, 
                  stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Signal:
    """Helper function to create Signal objects with proper timestamp handling"""
    # Convert timestamp to datetime if needed
    if hasattr(timestamp, 'to_pydatetime'):
        timestamp = timestamp.to_pydatetime()
    elif hasattr(timestamp, 'timestamp'):
        timestamp = datetime.fromtimestamp(timestamp.timestamp())
    elif isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    return Signal(
        signal_type=signal_type,
        confidence=confidence,
        entry_price=entry_price,
        timestamp=timestamp,
        reason=reason,
        strategy_name=strategy_name,
        stop_loss=stop_loss,
        take_profit=take_profit
    )

class LiquiditySweepReversalStrategy(BaseStrategy):
    """
    Liquidity Sweep Reversal Strategy
    
    Identifies liquidity sweeps (stop hunts) and trades the reversal.
    Works best in high volatility regimes.
    """
    
    def __init__(self):
        metadata = StrategyMetadata(
            lookback=20,  # Need recent high/low levels
            fields_required=["high", "low", "close", "volume"],
            strategy_type=StrategyType.FIXED_LOOKBACK,
            batch_mode=False,
            min_confidence=0.7,
            vol_target=0.15
        )
        
        super().__init__("liquidity_sweep_reversal", metadata)
        
        self.sweep_threshold = 0.002  # 0.2% sweep
        self.reversal_threshold = 0.001  # 0.1% reversal
        self.volume_threshold = 1.5  # 1.5x average volume
    
    def generate_signal(self, current_row: pd.Series, history: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        """Generate liquidity sweep reversal signal"""
        
        if history is None or len(history) < 20:
            logger.debug(f"LiquiditySweepReversal: Insufficient history: {len(history) if history is not None else 0}")
            return None
        
        current_price = current_row['close']
        current_volume = current_row['volume']
        
        # Calculate recent levels
        recent_high = history['high'].max()
        recent_low = history['low'].min()
        avg_volume = history['volume'].mean()
        
        # Check for liquidity sweep
        sweep_up = current_price > recent_high * (1 + self.sweep_threshold)
        sweep_down = current_price < recent_low * (1 - self.sweep_threshold)
        
        # Check volume confirmation
        volume_confirmed = current_volume > avg_volume * self.volume_threshold
        
        # Check for reversal
        reversal_up = sweep_up and current_price < recent_high * (1 - self.reversal_threshold)
        reversal_down = sweep_down and current_price > recent_low * (1 + self.reversal_threshold)
        
        # Debug logging
        logger.debug(f"LiquiditySweepReversal: price={current_price:.2f}, recent_high={recent_high:.2f}, recent_low={recent_low:.2f}")
        logger.debug(f"LiquiditySweepReversal: sweep_up={sweep_up}, sweep_down={sweep_down}, volume_confirmed={volume_confirmed}")
        logger.debug(f"LiquiditySweepReversal: reversal_up={reversal_up}, reversal_down={reversal_down}")
        
        if reversal_up and volume_confirmed:
            return create_signal(
                signal_type=SignalType.LONG,
                confidence=0.8,
                entry_price=current_price,
                timestamp=current_row.name,
                reason=f"Liquidity sweep reversal up: {current_price:.2f} vs {recent_high:.2f}",
                strategy_name=self.name
            )
        elif reversal_down and volume_confirmed:
            return create_signal(
                signal_type=SignalType.SHORT,
                confidence=0.8,
                entry_price=current_price,
                timestamp=current_row.name,
                reason=f"Liquidity sweep reversal down: {current_price:.2f} vs {recent_low:.2f}",
                strategy_name=self.name
            )
        
        return None

class VolumeWeightedTrendContinuationStrategy(BaseStrategy):
    """
    Volume Weighted Trend Continuation Strategy
    
    Identifies strong trends with volume confirmation and trades continuation.
    """
    
    def __init__(self):
        metadata = StrategyMetadata(
            lookback=20,
            fields_required=["high", "low", "close", "volume"],
            strategy_type=StrategyType.FIXED_LOOKBACK,
            batch_mode=False,
            min_confidence=0.6,
            vol_target=0.12
        )
        
        super().__init__("volume_weighted_trend_continuation", metadata)
        
        self.trend_periods = 10
        self.volume_threshold = 1.3
        self.breakout_threshold = 0.005  # 0.5%
    
    def generate_signal(self, current_row: pd.Series, history: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        """Generate volume weighted trend continuation signal"""
        
        if history is None or len(history) < 20:
            return None
        
        current_price = current_row['close']
        current_volume = current_row['volume']
        
        # Calculate trend
        recent_closes = history['close'].tail(self.trend_periods)
        trend_slope = np.polyfit(range(len(recent_closes)), recent_closes.values, 1)[0]
        
        # Calculate volume
        avg_volume = history['volume'].mean()
        volume_confirmed = current_volume > avg_volume * self.volume_threshold
        
        # Calculate breakout levels
        recent_high = history['high'].max()
        recent_low = history['low'].min()
        
        # Check for breakout with trend confirmation
        breakout_up = current_price > recent_high * (1 + self.breakout_threshold)
        breakout_down = current_price < recent_low * (1 - self.breakout_threshold)
        
        if breakout_up and trend_slope > 0 and volume_confirmed:
            return Signal(
                timestamp=current_row.name,
                signal_type=SignalType.LONG,
                entry_price=current_price,
                confidence=0.7,
                reason=f"Volume weighted trend continuation up: slope={trend_slope:.4f}"
            )
        elif breakout_down and trend_slope < 0 and volume_confirmed:
            return Signal(
                timestamp=current_row.name,
                signal_type=SignalType.SHORT,
                entry_price=current_price,
                confidence=0.7,
                reason=f"Volume weighted trend continuation down: slope={trend_slope:.4f}"
            )
        
        return None

class VolatilityExpansionBreakoutStrategy(BaseStrategy):
    """
    Volatility Expansion Breakout Strategy
    
    Identifies low volatility periods followed by expansion breakouts.
    """
    
    def __init__(self):
        metadata = StrategyMetadata(
            lookback=20,
            fields_required=["high", "low", "close"],
            strategy_type=StrategyType.FIXED_LOOKBACK,
            batch_mode=False,
            min_confidence=0.8,
            vol_target=0.08
        )
        
        super().__init__("volatility_expansion_breakout", metadata)
        
        self.vol_periods = 10
        self.low_vol_threshold = 0.5  # 50% of average volatility
        self.breakout_threshold = 0.008  # 0.8%
    
    def generate_signal(self, current_row: pd.Series, history: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        """Generate volatility expansion breakout signal"""
        
        if history is None or len(history) < 20:
            return None
        
        current_price = current_row['close']
        
        # Calculate volatility
        returns = history['close'].pct_change().dropna()
        current_vol = returns.tail(self.vol_periods).std()
        avg_vol = returns.std()
        
        # Check for low volatility period
        low_vol_period = current_vol < avg_vol * self.low_vol_threshold
        
        # Calculate breakout levels
        recent_high = history['high'].max()
        recent_low = history['low'].min()
        
        # Check for breakout
        breakout_up = current_price > recent_high * (1 + self.breakout_threshold)
        breakout_down = current_price < recent_low * (1 - self.breakout_threshold)
        
        if breakout_up and low_vol_period:
            return Signal(
                timestamp=current_row.name,
                signal_type=SignalType.LONG,
                entry_price=current_price,
                confidence=0.8,
                reason=f"Volatility expansion breakout up: vol={current_vol:.4f}"
            )
        elif breakout_down and low_vol_period:
            return Signal(
                timestamp=current_row.name,
                signal_type=SignalType.SHORT,
                entry_price=current_price,
                confidence=0.8,
                reason=f"Volatility expansion breakout down: vol={current_vol:.4f}"
            )
        
        return None

class DailyAVWAPZScoreReversionStrategy(BaseStrategy):
    """
    Daily AVWAP Z-Score Reversion Strategy
    
    Uses Average Volume Weighted Average Price (AVWAP) z-scores for mean reversion.
    """
    
    def __init__(self):
        metadata = StrategyMetadata(
            lookback=20,
            fields_required=["high", "low", "close", "volume"],
            strategy_type=StrategyType.FIXED_LOOKBACK,
            batch_mode=False,
            min_confidence=0.7,
            vol_target=0.12
        )
        
        super().__init__("daily_avwap_zscore_reversion", metadata)
        
        self.zscore_threshold = 2.0  # 2 standard deviations
        self.lookback_days = 5
    
    def generate_signal(self, current_row: pd.Series, history: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        """Generate AVWAP z-score reversion signal"""
        
        if history is None or len(history) < 20:
            return None
        
        current_price = current_row['close']
        
        # Calculate AVWAP
        volume_price = history['close'] * history['volume']
        avwap = volume_price.sum() / history['volume'].sum()
        
        # Calculate z-score
        recent_prices = history['close'].tail(self.lookback_days)
        price_std = recent_prices.std()
        zscore = (current_price - avwap) / price_std if price_std > 0 else 0
        
        # Check for extreme z-scores
        if zscore > self.zscore_threshold:
            return Signal(
                timestamp=current_row.name,
                signal_type=SignalType.SHORT,
                entry_price=current_price,
                confidence=0.7,
                reason=f"AVWAP z-score reversion short: zscore={zscore:.2f}"
            )
        elif zscore < -self.zscore_threshold:
            return Signal(
                timestamp=current_row.name,
                signal_type=SignalType.LONG,
                entry_price=current_price,
                confidence=0.7,
                reason=f"AVWAP z-score reversion long: zscore={zscore:.2f}"
            )
        
        return None

class OpeningRangeBreakRetestStrategy(BaseStrategy):
    """
    Opening Range Break & Retest Strategy
    
    Identifies opening range breaks followed by retests for continuation trades.
    """
    
    def __init__(self):
        metadata = StrategyMetadata(
            lookback=20,
            fields_required=["high", "low", "close"],
            strategy_type=StrategyType.FIXED_LOOKBACK,
            batch_mode=False,
            min_confidence=0.75,
            vol_target=0.10
        )
        
        super().__init__("opening_range_break_retest", metadata)
        
        self.opening_range_periods = 4  # First 4 periods of day
        self.breakout_threshold = 0.003  # 0.3%
        self.retest_threshold = 0.001  # 0.1%
    
    def generate_signal(self, current_row: pd.Series, history: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        """Generate opening range break & retest signal"""
        
        if history is None or len(history) < 20:
            return None
        
        current_price = current_row['close']
        
        # Calculate opening range (simplified - first few periods)
        opening_data = history.head(self.opening_range_periods)
        opening_high = opening_data['high'].max()
        opening_low = opening_data['low'].min()
        
        # Check for breakout
        breakout_up = current_price > opening_high * (1 + self.breakout_threshold)
        breakout_down = current_price < opening_low * (1 - self.breakout_threshold)
        
        # Check for retest
        retest_up = breakout_up and current_price < opening_high * (1 + self.retest_threshold)
        retest_down = breakout_down and current_price > opening_low * (1 - self.retest_threshold)
        
        if retest_up:
            return Signal(
                timestamp=current_row.name,
                signal_type=SignalType.LONG,
                entry_price=current_price,
                confidence=0.75,
                reason=f"Opening range break & retest up: {current_price:.2f}"
            )
        elif retest_down:
            return Signal(
                timestamp=current_row.name,
                signal_type=SignalType.SHORT,
                entry_price=current_price,
                confidence=0.75,
                reason=f"Opening range break & retest down: {current_price:.2f}"
            )
        
        return None

class KeltnerExhaustionFadeStrategy(BaseStrategy):
    """
    Keltner Exhaustion Fade Strategy
    
    Uses Keltner Channels to identify exhaustion moves for fade trades.
    """
    
    def __init__(self):
        metadata = StrategyMetadata(
            lookback=20,
            fields_required=["high", "low", "close"],
            strategy_type=StrategyType.FIXED_LOOKBACK,
            batch_mode=False,
            min_confidence=0.7,
            vol_target=0.15
        )
        
        super().__init__("keltner_exhaustion_fade", metadata)
        
        self.keltner_period = 20
        self.keltner_multiplier = 2.0
        self.exhaustion_threshold = 0.8  # 80% of channel width
    
    def generate_signal(self, current_row: pd.Series, history: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        """Generate Keltner exhaustion fade signal"""
        
        if history is None or len(history) < 20:
            return None
        
        current_price = current_row['close']
        
        # Calculate Keltner Channels
        recent_data = history.tail(self.keltner_period)
        ema = recent_data['close'].ewm(span=self.keltner_period).mean().iloc[-1]
        atr = recent_data['high'].subtract(recent_data['low']).mean()
        
        upper_channel = ema + (atr * self.keltner_multiplier)
        lower_channel = ema - (atr * self.keltner_multiplier)
        
        # Check for exhaustion moves
        exhaustion_up = current_price > upper_channel * (1 + self.exhaustion_threshold * atr / ema)
        exhaustion_down = current_price < lower_channel * (1 - self.exhaustion_threshold * atr / ema)
        
        if exhaustion_up:
            return Signal(
                timestamp=current_row.name,
                signal_type=SignalType.SHORT,
                entry_price=current_price,
                confidence=0.7,
                reason=f"Keltner exhaustion fade short: {current_price:.2f} vs {upper_channel:.2f}"
            )
        elif exhaustion_down:
            return Signal(
                timestamp=current_row.name,
                signal_type=SignalType.LONG,
                entry_price=current_price,
                confidence=0.7,
                reason=f"Keltner exhaustion fade long: {current_price:.2f} vs {lower_channel:.2f}"
            )
        
        return None

class FakeoutReversionStrategy(BaseStrategy):
    """
    Fakeout Reversion Strategy
    
    Identifies false breakouts and trades the reversion.
    """
    
    def __init__(self):
        metadata = StrategyMetadata(
            lookback=20,
            fields_required=["high", "low", "close", "volume"],
            strategy_type=StrategyType.FIXED_LOOKBACK,
            batch_mode=False,
            min_confidence=0.75,
            vol_target=0.08
        )
        
        super().__init__("fakeout_reversion", metadata)
        
        self.fakeout_threshold = 0.005  # 0.5%
        self.reversion_threshold = 0.002  # 0.2%
        self.volume_threshold = 0.8  # Below average volume
    
    def generate_signal(self, current_row: pd.Series, history: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        """Generate fakeout reversion signal"""
        
        if history is None or len(history) < 20:
            return None
        
        current_price = current_row['close']
        current_volume = current_row['volume']
        
        # Calculate levels
        recent_high = history['high'].max()
        recent_low = history['low'].min()
        avg_volume = history['volume'].mean()
        
        # Check for fakeout (breakout with low volume)
        fakeout_up = (current_price > recent_high * (1 + self.fakeout_threshold) and 
                     current_volume < avg_volume * self.volume_threshold)
        fakeout_down = (current_price < recent_low * (1 - self.fakeout_threshold) and 
                       current_volume < avg_volume * self.volume_threshold)
        
        # Check for reversion
        reversion_up = fakeout_up and current_price < recent_high * (1 + self.reversion_threshold)
        reversion_down = fakeout_down and current_price > recent_low * (1 - self.reversion_threshold)
        
        if reversion_up:
            return Signal(
                timestamp=current_row.name,
                signal_type=SignalType.SHORT,
                entry_price=current_price,
                confidence=0.75,
                reason=f"Fakeout reversion short: {current_price:.2f}"
            )
        elif reversion_down:
            return Signal(
                timestamp=current_row.name,
                signal_type=SignalType.LONG,
                entry_price=current_price,
                confidence=0.75,
                reason=f"Fakeout reversion long: {current_price:.2f}"
            )
        
        return None

# Strategy factory function
def create_strategy(strategy_name: str) -> Optional[BaseStrategy]:
    """Create strategy instance by name"""
    
    strategies = {
        'liquidity_sweep_reversal': LiquiditySweepReversalStrategy,
        'volume_weighted_trend_continuation': VolumeWeightedTrendContinuationStrategy,
        'volatility_expansion_breakout': VolatilityExpansionBreakoutStrategy,
        'daily_avwap_zscore_reversion': DailyAVWAPZScoreReversionStrategy,
        'opening_range_break_retest': OpeningRangeBreakRetestStrategy,
        'keltner_exhaustion_fade': KeltnerExhaustionFadeStrategy,
        'fakeout_reversion': FakeoutReversionStrategy
    }
    
    if strategy_name in strategies:
        return strategies[strategy_name]()
    
    return None
