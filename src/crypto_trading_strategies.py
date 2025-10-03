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
        
        # 1. BTC Asia Session Liquidity Sweep
        self.strategies['btc_asia_sweep'] = {
            'config': StrategyConfig(
                name='btc_asia_sweep',
                symbol='BTC',
                mechanism='liquidity_sweep',
                horizon='intraday',
                session='asia',
                regime_filters=[RegimeType.ASIA_SESSION, RegimeType.HIGH_VOL]
            ),
            'function': self.btc_asia_liquidity_sweep
        }
        
        # 2. ETH Breakout Continuation
        self.strategies['eth_breakout_continuation'] = {
            'config': StrategyConfig(
                name='eth_breakout_continuation',
                symbol='ETH',
                mechanism='breakout_continuation',
                horizon='swing',
                session='ny',
                regime_filters=[RegimeType.TREND_UP, RegimeType.HIGH_VOL]
            ),
            'function': self.eth_breakout_continuation
        }
        
        # 3. BTC Mean Reversion (Overnight)
        self.strategies['btc_mean_reversion'] = {
            'config': StrategyConfig(
                name='btc_mean_reversion',
                symbol='BTC',
                mechanism='mean_reversion',
                horizon='overnight',
                session='ny',
                regime_filters=[RegimeType.RANGE, RegimeType.LOW_VOL]
            ),
            'function': self.btc_mean_reversion
        }
        
        # 4. ETH Funding Rate Arbitrage
        self.strategies['eth_funding_arb'] = {
            'config': StrategyConfig(
                name='eth_funding_arb',
                symbol='ETH',
                mechanism='funding_arbitrage',
                horizon='intraday',
                session='london',
                regime_filters=[RegimeType.HIGH_VOL]
            ),
            'function': self.eth_funding_arbitrage
        }
        
        # 5. BTC Volatility Compression
        self.strategies['btc_vol_compression'] = {
            'config': StrategyConfig(
                name='btc_vol_compression',
                symbol='BTC',
                mechanism='volatility_compression',
                horizon='positional',
                session='london',
                regime_filters=[RegimeType.LOW_VOL, RegimeType.RANGE]
            ),
            'function': self.btc_volatility_compression
        }
        
        # 6. ETH Cross-Exchange Basis Trade
        self.strategies['eth_basis_trade'] = {
            'config': StrategyConfig(
                name='eth_basis_trade',
                symbol='ETH',
                mechanism='basis_trade',
                horizon='swing',
                session='asia',
                regime_filters=[RegimeType.HIGH_VOL]
            ),
            'function': self.eth_cross_exchange_basis
        }
        
        # 7. BTC NY Open London Low Sweep
        self.strategies['btc_ny_open_london_sweep'] = {
            'config': StrategyConfig(
                name='btc_ny_open_london_sweep',
                symbol='BTC',
                mechanism='liquidity_sweep',
                horizon='intraday',
                session='ny',
                regime_filters=[RegimeType.NY_SESSION, RegimeType.HIGH_VOL]
            ),
            'function': self.btc_ny_open_london_sweep
        }
        
        # 8. BTC NY Session Buy/Sell
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
    
    def btc_asia_liquidity_sweep(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """
        BTC Asia Session Liquidity Sweep Strategy
        
        Principle: Liquidity sweeps during Asia session create predictable reversions
        - Look for wicks that exceed recent high/low by 0.5-1%
        - Enter on reclaim of previous level
        - Target: 0.3-0.8% move with 1:2 R/R
        """
        if len(data) < 50:
            return None
        
        # Calculate liquidity levels
        recent_high = data['high'].rolling(20).max()
        recent_low = data['low'].rolling(20).min()
        
        # Current price levels
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_close = data['close'].iloc[-1]
        
        # Check for liquidity sweep conditions
        high_sweep = current_high > recent_high.iloc[-2] * 1.005  # 0.5% above recent high
        low_sweep = current_low < recent_low.iloc[-2] * 0.995   # 0.5% below recent low
        
        # Check for reclaim
        high_reclaim = current_close > recent_high.iloc[-2]
        low_reclaim = current_close < recent_low.iloc[-2]
        
        # Volume confirmation
        volume_surge = data['volume'].iloc[-1] > data['volume'].rolling(20).mean().iloc[-1] * 1.5
        
        confidence = 0.0
        signal_type = SignalType.FLAT
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        if high_sweep and high_reclaim and volume_surge:
            # Long signal after high sweep and reclaim
            signal_type = SignalType.LONG
            confidence = 0.75
            stop_loss = recent_high.iloc[-2] * 0.998  # Just below reclaimed level
            take_profit = entry_price * 1.006  # 0.6% target
            reason = "BTC Asia high sweep reclaim with volume"
            
        elif low_sweep and low_reclaim and volume_surge:
            # Short signal after low sweep and reclaim
            signal_type = SignalType.SHORT
            confidence = 0.75
            stop_loss = recent_low.iloc[-2] * 1.002  # Just above reclaimed level
            take_profit = entry_price * 0.994  # 0.6% target
            reason = "BTC Asia low sweep reclaim with volume"
        
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
    
    def eth_breakout_continuation(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """
        ETH Breakout Continuation Strategy
        
        Principle: Strong breakouts with volume tend to continue
        - Look for breakouts above resistance with 2x average volume
        - Enter on pullback to breakout level
        - Target: 2-4% continuation move
        """
        if len(data) < 50:
            return None
        
        # Calculate resistance levels
        resistance = data['high'].rolling(20).max()
        support = data['low'].rolling(20).min()
        
        current_close = data['close'].iloc[-1]
        current_high = data['high'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        
        # Check for breakout
        breakout_level = resistance.iloc[-2]
        breakout_confirmed = current_high > breakout_level * 1.002  # 0.2% above resistance
        volume_confirmation = current_volume > avg_volume * 2.0
        
        # Check for pullback entry
        pullback_entry = current_close <= breakout_level * 1.001  # Slight pullback
        
        # Trend confirmation
        sma_20 = data['close'].rolling(20).mean()
        trend_up = current_close > sma_20.iloc[-1]
        
        confidence = 0.0
        signal_type = SignalType.FLAT
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        if breakout_confirmed and volume_confirmation and pullback_entry and trend_up:
            signal_type = SignalType.LONG
            confidence = 0.80
            stop_loss = breakout_level * 0.995  # Below breakout level
            take_profit = entry_price * 1.025  # 2.5% target
            reason = "ETH breakout continuation with volume"
        
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
    
    def btc_mean_reversion(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """
        BTC Mean Reversion Strategy (Overnight)
        
        Principle: Extreme moves tend to revert to mean in ranging markets
        - Look for 2+ standard deviation moves from 20-period mean
        - Enter on reversion with RSI confirmation
        - Target: Return to mean with 1:1.5 R/R
        """
        if len(data) < 50:
            return None
        
        # Calculate mean reversion indicators
        sma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        
        # RSI for overbought/oversold
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_close = data['close'].iloc[-1]
        current_sma = sma_20.iloc[-1]
        current_std = std_20.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Mean reversion conditions
        upper_band = current_sma + 2 * current_std
        lower_band = current_sma - 2 * current_std
        
        overbought = current_close > upper_band and current_rsi > 70
        oversold = current_close < lower_band and current_rsi < 30
        
        # Volume confirmation (lower volume on extremes)
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        volume_confirmation = current_volume < avg_volume * 0.8
        
        confidence = 0.0
        signal_type = SignalType.FLAT
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        if overbought and volume_confirmation:
            signal_type = SignalType.SHORT
            confidence = 0.70
            stop_loss = upper_band * 1.002  # Above upper band
            take_profit = current_sma  # Target mean
            reason = "BTC mean reversion short - overbought"
            
        elif oversold and volume_confirmation:
            signal_type = SignalType.LONG
            confidence = 0.70
            stop_loss = lower_band * 0.998  # Below lower band
            take_profit = current_sma  # Target mean
            reason = "BTC mean reversion long - oversold"
        
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
    
    def eth_funding_arbitrage(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """
        ETH Funding Rate Arbitrage Strategy
        
        Principle: Extreme funding rates create arbitrage opportunities
        - Look for funding rate extremes (>0.01% or <-0.01%)
        - Enter opposite position to funding direction
        - Target: Funding rate normalization
        """
        if len(data) < 50:
            return None
        
        # Simulate funding rate (in production, would use real funding data)
        # For demo, use price momentum as proxy
        price_change = data['close'].pct_change(8).iloc[-1]  # 8-period change
        funding_rate = price_change * 0.1  # Simulate funding rate
        
        current_close = data['close'].iloc[-1]
        
        # Funding rate thresholds
        high_funding = funding_rate > 0.01  # 0.01% funding
        low_funding = funding_rate < -0.01  # -0.01% funding
        
        # Volume confirmation
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        volume_confirmation = current_volume > avg_volume * 1.2
        
        confidence = 0.0
        signal_type = SignalType.FLAT
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        if high_funding and volume_confirmation:
            # High funding rate - short the perp
            signal_type = SignalType.SHORT
            confidence = 0.65
            stop_loss = entry_price * 1.01  # 1% stop
            take_profit = entry_price * 0.995  # 0.5% target
            reason = f"ETH funding arbitrage short - rate: {funding_rate:.4f}"
            
        elif low_funding and volume_confirmation:
            # Low funding rate - long the perp
            signal_type = SignalType.LONG
            confidence = 0.65
            stop_loss = entry_price * 0.99  # 1% stop
            take_profit = entry_price * 1.005  # 0.5% target
            reason = f"ETH funding arbitrage long - rate: {funding_rate:.4f}"
        
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
    
    def btc_volatility_compression(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """
        BTC Volatility Compression Strategy
        
        Principle: Low volatility periods often precede high volatility moves
        - Look for volatility compression (Bollinger Band squeeze)
        - Enter on volatility expansion breakout
        - Target: 1-3% move in breakout direction
        """
        if len(data) < 50:
            return None
        
        # Calculate Bollinger Bands
        sma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        
        # Volatility compression indicator
        band_width = (upper_band - lower_band) / sma_20
        avg_band_width = band_width.rolling(20).mean()
        
        current_close = data['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_width = band_width.iloc[-1]
        avg_width = avg_band_width.iloc[-1]
        
        # Compression conditions
        compression = current_width < avg_width * 0.7  # 30% below average width
        breakout_up = current_close > current_upper
        breakout_down = current_close < current_lower
        
        # Volume confirmation
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        volume_confirmation = current_volume > avg_volume * 1.5
        
        confidence = 0.0
        signal_type = SignalType.FLAT
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        if compression and breakout_up and volume_confirmation:
            signal_type = SignalType.LONG
            confidence = 0.75
            stop_loss = sma_20.iloc[-1]  # Below middle band
            take_profit = entry_price * 1.02  # 2% target
            reason = "BTC volatility compression breakout up"
            
        elif compression and breakout_down and volume_confirmation:
            signal_type = SignalType.SHORT
            confidence = 0.75
            stop_loss = sma_20.iloc[-1]  # Above middle band
            take_profit = entry_price * 0.98  # 2% target
            reason = "BTC volatility compression breakout down"
        
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
    
    def eth_cross_exchange_basis(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """
        ETH Cross-Exchange Basis Trade Strategy
        
        Principle: Price differences between exchanges create arbitrage opportunities
        - Look for significant basis (spot vs perp) differences
        - Enter when basis exceeds normal range
        - Target: Basis convergence
        """
        if len(data) < 50:
            return None
        
        # Simulate basis calculation (in production, would use real exchange data)
        # For demo, use price momentum and volatility as proxy
        price_momentum = data['close'].pct_change(5).iloc[-1]
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Simulate basis as function of momentum and volatility
        simulated_basis = price_momentum * volatility * 100  # Basis in basis points
        
        current_close = data['close'].iloc[-1]
        
        # Basis thresholds
        high_basis = simulated_basis > 5  # 5 basis points
        low_basis = simulated_basis < -5  # -5 basis points
        
        # Volume confirmation
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        volume_confirmation = current_volume > avg_volume * 1.3
        
        confidence = 0.0
        signal_type = SignalType.FLAT
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        if high_basis and volume_confirmation:
            # High basis - short perp, long spot (simplified as short)
            signal_type = SignalType.SHORT
            confidence = 0.60
            stop_loss = entry_price * 1.008  # 0.8% stop
            take_profit = entry_price * 0.996  # 0.4% target
            reason = f"ETH basis trade short - basis: {simulated_basis:.2f}bp"
            
        elif low_basis and volume_confirmation:
            # Low basis - long perp, short spot (simplified as long)
            signal_type = SignalType.LONG
            confidence = 0.60
            stop_loss = entry_price * 0.992  # 0.8% stop
            take_profit = entry_price * 1.004  # 0.4% target
            reason = f"ETH basis trade long - basis: {simulated_basis:.2f}bp"
        
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
    
    def btc_ny_open_london_sweep(self, data: pd.DataFrame, regime: Dict[RegimeType, bool]) -> Optional[Signal]:
        """
        BTC NY Open London Low Sweep Strategy
        
        Principle: NY open often sweeps London session lows with volume surge
        - Look for price sweep below London session low (last 8 hours)
        - Enter on reclaim above London low with volume confirmation
        - Target: 0.5-1% move with tight stop below London low
        """
        if len(data) < 50:
            return None
        
        # Calculate London session low (last 8 hours = 32 periods for 15min data)
        london_low = data['low'].rolling(32).min()
        
        # Current levels
        current_close = data['close'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_high = data['high'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # London session levels
        london_low_level = london_low.iloc[-2]  # Previous period's London low
        london_high_level = data['high'].rolling(32).max().iloc[-2]
        
        # Check for London low sweep
        london_sweep = current_low < london_low_level * 0.998  # 0.2% below London low
        
        # Check for reclaim above London low
        london_reclaim = current_close > london_low_level
        
        # Volume confirmation - NY open typically has volume surge
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        volume_surge = current_volume > avg_volume * 1.8  # 80% above average
        
        # Additional confirmation - price should be recovering
        recovery = current_close > current_low * 1.002  # 0.2% above the low
        
        confidence = 0.0
        signal_type = SignalType.FLAT
        entry_price = current_close
        stop_loss = None
        take_profit = None
        reason = ""
        
        if london_sweep and london_reclaim and volume_surge and recovery:
            signal_type = SignalType.LONG
            confidence = 0.80
            stop_loss = london_low_level * 0.995  # 0.5% below London low
            take_profit = entry_price * 1.008  # 0.8% target
            reason = f"BTC NY open London sweep reclaim - swept {london_low_level:.2f}, reclaimed at {current_close:.2f}"
        
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
        if len(data) < 50:
            return None
        
        
        # Get current timestamp (assuming data has datetime index)
        current_time = data.index[-1]
        
        # Convert to NY timezone for session detection
        if hasattr(current_time, 'tz'):
            if current_time.tz is None:
                ny_time = current_time.tz_localize('UTC').tz_convert('America/New_York')
            else:
                ny_time = current_time.tz_convert('America/New_York')
        else:
            ny_time = current_time
        
        # Extract hour and minute
        hour = ny_time.hour
        minute = ny_time.minute
        
        # NY session times (9:30 AM - 4:00 PM ET)
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
        
        # Generate signals only at specific times (once per day)
        # Buy at NY open (9:30 AM) - only at exact open time
        if in_ny_session and current_minutes == ny_open_time:
            signal_type = SignalType.LONG
            confidence = 0.70
            stop_loss = entry_price * 0.98  # 2% stop loss
            take_profit = entry_price * 1.03  # 3% take profit
            reason = f"BTC NY open buy - {ny_time.strftime('%H:%M')} ET"
            
        # Sell at NY close (4:00 PM) - only at exact close time
        elif in_ny_session and current_minutes == ny_close_time:
            signal_type = SignalType.SHORT
            confidence = 0.70
            stop_loss = entry_price * 1.02  # 2% stop loss
            take_profit = entry_price * 0.97  # 3% take profit
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
    
    def get_all_strategies(self) -> Dict:
        """Get all available strategies"""
        return self.strategies
    
    def get_strategy_configs(self) -> List[StrategyConfig]:
        """Get all strategy configurations"""
        return [strategy['config'] for strategy in self.strategies.values()]
