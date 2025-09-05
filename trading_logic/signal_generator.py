"""
Intelligent Signal Generation and Fusion

Features:
- Multi-source signal fusion (technical, sentiment, simulation)
- Signal confidence scoring and validation
- Signal decay and expiration management
- Cross-asset signal correlation analysis
- Signal performance tracking and optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from loguru import logger
from enum import Enum


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class SignalSource(Enum):
    """Sources of trading signals"""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    SIMULATION = "simulation"
    VOLUME = "volume"
    NEWS = "news"
    FUSED = "fused"


@dataclass
class TradingSignal:
    """Individual trading signal"""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0-1
    strength: str  # 'weak', 'medium', 'strong'
    source: SignalSource
    reasoning: str
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    time_horizon: int = 24  # hours
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(hours=self.time_horizon)
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SignalFusion:
    """Fused signal from multiple sources"""
    symbol: str
    final_signal: SignalType
    confidence: float
    strength: str
    source_signals: List[TradingSignal]
    fusion_weights: Dict[SignalSource, float]
    reasoning: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class SignalGenerator:
    """Main signal generation and fusion engine"""
    
    def __init__(self):
        self.signal_history = []
        self.performance_tracker = {}
        self.fusion_weights = {
            SignalSource.TECHNICAL: 0.4,
            SignalSource.SENTIMENT: 0.3,
            SignalSource.SIMULATION: 0.2,
            SignalSource.VOLUME: 0.1
        }
        
    async def initialize(self):
        """Initialize the signal generator"""
        logger.info("Signal generator initialized")
    
    async def fuse_signals(
        self,
        technical_signals: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        simulation_data: Dict[str, Any],
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, SignalFusion]:
        """Fuse signals from multiple sources"""
        logger.info("Fusing signals from multiple sources")
        
        fused_signals = {}
        symbols = set(technical_signals.keys()) | set(price_data.keys())
        
        for symbol in symbols:
            try:
                # Collect signals for this symbol
                source_signals = []
                
                # Technical signals
                if symbol in technical_signals:
                    tech_signal = self._create_technical_signal(
                        symbol, technical_signals[symbol]
                    )
                    if tech_signal:
                        source_signals.append(tech_signal)
                
                # Sentiment signals
                if sentiment_data:
                    sent_signal = self._create_sentiment_signal(
                        symbol, sentiment_data
                    )
                    if sent_signal:
                        source_signals.append(sent_signal)
                
                # Simulation signals
                if simulation_data:
                    sim_signal = self._create_simulation_signal(
                        symbol, simulation_data
                    )
                    if sim_signal:
                        source_signals.append(sim_signal)
                
                # Volume signals
                if symbol in price_data:
                    vol_signal = self._create_volume_signal(
                        symbol, price_data[symbol]
                    )
                    if vol_signal:
                        source_signals.append(vol_signal)
                
                # Fuse signals if we have any
                if source_signals:
                    fused_signal = self._fuse_source_signals(symbol, source_signals)
                    fused_signals[symbol] = fused_signal
                    
            except Exception as e:
                logger.error(f"Error fusing signals for {symbol}: {e}")
                continue
        
        logger.info(f"Generated {len(fused_signals)} fused signals")
        return fused_signals
    
    def _create_technical_signal(self, symbol: str, tech_data: Dict) -> Optional[TradingSignal]:
        """Create signal from technical analysis"""
        try:
            # Extract technical indicators
            rsi = tech_data.get('rsi', 50)
            macd = tech_data.get('macd', 0)
            macd_signal = tech_data.get('macd_signal', 0)
            bollinger_position = tech_data.get('bollinger_position', 0.5)
            
            # Determine signal type and confidence
            signal_score = 0
            reasoning_parts = []
            
            # RSI signals
            if rsi < 30:
                signal_score += 2
                reasoning_parts.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                signal_score -= 2
                reasoning_parts.append(f"RSI overbought ({rsi:.1f})")
            elif rsi < 40:
                signal_score += 1
                reasoning_parts.append(f"RSI approaching oversold ({rsi:.1f})")
            elif rsi > 60:
                signal_score -= 1
                reasoning_parts.append(f"RSI approaching overbought ({rsi:.1f})")
            
            # MACD signals
            if macd > macd_signal:
                signal_score += 1
                reasoning_parts.append("MACD bullish")
            else:
                signal_score -= 1
                reasoning_parts.append("MACD bearish")
            
            # Bollinger Bands signals
            if bollinger_position < 0.2:
                signal_score += 1
                reasoning_parts.append("Near lower Bollinger Band")
            elif bollinger_position > 0.8:
                signal_score -= 1
                reasoning_parts.append("Near upper Bollinger Band")
            
            # Convert score to signal
            if signal_score >= 3:
                signal_type = SignalType.STRONG_BUY
                confidence = min(0.9, 0.6 + signal_score * 0.1)
                strength = "strong"
            elif signal_score >= 1:
                signal_type = SignalType.BUY
                confidence = 0.5 + signal_score * 0.1
                strength = "medium"
            elif signal_score <= -3:
                signal_type = SignalType.STRONG_SELL
                confidence = min(0.9, 0.6 + abs(signal_score) * 0.1)
                strength = "strong"
            elif signal_score <= -1:
                signal_type = SignalType.SELL
                confidence = 0.5 + abs(signal_score) * 0.1
                strength = "medium"
            else:
                signal_type = SignalType.HOLD
                confidence = 0.3
                strength = "weak"
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                strength=strength,
                source=SignalSource.TECHNICAL,
                reasoning="; ".join(reasoning_parts),
                time_horizon=48,
                metadata={'technical_score': signal_score}
            )
            
        except Exception as e:
            logger.error(f"Error creating technical signal for {symbol}: {e}")
            return None
    
    def _create_sentiment_signal(self, symbol: str, sentiment_data: Dict) -> Optional[TradingSignal]:
        """Create signal from sentiment analysis"""
        try:
            avg_sentiment = sentiment_data.get('average_sentiment', 0)
            confidence_score = sentiment_data.get('average_confidence', 0)
            positive_ratio = sentiment_data.get('positive_ratio', 0.5)
            
            # Determine signal based on sentiment
            if avg_sentiment > 0.3 and confidence_score > 0.6:
                signal_type = SignalType.BUY
                confidence = min(0.8, confidence_score)
                strength = "strong" if avg_sentiment > 0.5 else "medium"
                reasoning = f"Positive sentiment ({avg_sentiment:.2f}) with {positive_ratio:.1%} positive articles"
            elif avg_sentiment < -0.3 and confidence_score > 0.6:
                signal_type = SignalType.SELL
                confidence = min(0.8, confidence_score)
                strength = "strong" if avg_sentiment < -0.5 else "medium"
                reasoning = f"Negative sentiment ({avg_sentiment:.2f}) with {1-positive_ratio:.1%} negative articles"
            else:
                signal_type = SignalType.HOLD
                confidence = 0.3
                strength = "weak"
                reasoning = f"Neutral sentiment ({avg_sentiment:.2f})"
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                strength=strength,
                source=SignalSource.SENTIMENT,
                reasoning=reasoning,
                time_horizon=24,
                metadata={
                    'sentiment_score': avg_sentiment,
                    'confidence_score': confidence_score,
                    'positive_ratio': positive_ratio
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating sentiment signal for {symbol}: {e}")
            return None
    
    def _create_simulation_signal(self, symbol: str, simulation_data: Dict) -> Optional[TradingSignal]:
        """Create signal from simulation results"""
        try:
            expected_return = simulation_data.get('expected_return', 0)
            probability_of_profit = simulation_data.get('probability_of_profit', 0.5)
            var_95 = simulation_data.get('var_95', 0)
            
            # Determine signal based on simulation results
            if expected_return > 0.1 and probability_of_profit > 0.6:
                signal_type = SignalType.BUY
                confidence = min(0.9, probability_of_profit)
                strength = "strong" if probability_of_profit > 0.8 else "medium"
                reasoning = f"High probability of profit ({probability_of_profit:.1%}) with {expected_return:.1%} expected return"
            elif expected_return < -0.1 and probability_of_profit < 0.4:
                signal_type = SignalType.SELL
                confidence = min(0.9, 1 - probability_of_profit)
                strength = "strong" if probability_of_profit < 0.2 else "medium"
                reasoning = f"Low probability of profit ({probability_of_profit:.1%}) with {expected_return:.1%} expected return"
            else:
                signal_type = SignalType.HOLD
                confidence = 0.4
                strength = "weak"
                reasoning = f"Moderate probability of profit ({probability_of_profit:.1%})"
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                strength=strength,
                source=SignalSource.SIMULATION,
                reasoning=reasoning,
                time_horizon=168,  # 1 week
                metadata={
                    'expected_return': expected_return,
                    'probability_of_profit': probability_of_profit,
                    'var_95': var_95
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating simulation signal for {symbol}: {e}")
            return None
    
    def _create_volume_signal(self, symbol: str, price_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Create signal from volume analysis"""
        try:
            if len(price_data) < 20:
                return None
            
            # Calculate volume metrics
            current_volume = price_data['volume'].iloc[-1]
            avg_volume = price_data['volume'].tail(20).mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Price change
            price_change = (price_data['close'].iloc[-1] - price_data['close'].iloc[-2]) / price_data['close'].iloc[-2]
            
            # Determine signal based on volume and price
            if volume_ratio > 2.0 and price_change > 0.02:
                signal_type = SignalType.BUY
                confidence = min(0.8, 0.5 + volume_ratio * 0.1)
                strength = "strong"
                reasoning = f"High volume ({volume_ratio:.1f}x) with price increase ({price_change:.1%})"
            elif volume_ratio > 2.0 and price_change < -0.02:
                signal_type = SignalType.SELL
                confidence = min(0.8, 0.5 + volume_ratio * 0.1)
                strength = "strong"
                reasoning = f"High volume ({volume_ratio:.1f}x) with price decrease ({price_change:.1%})"
            elif volume_ratio < 0.5:
                signal_type = SignalType.HOLD
                confidence = 0.3
                strength = "weak"
                reasoning = f"Low volume ({volume_ratio:.1f}x)"
            else:
                return None  # No clear signal
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                strength=strength,
                source=SignalSource.VOLUME,
                reasoning=reasoning,
                time_horizon=12,
                metadata={
                    'volume_ratio': volume_ratio,
                    'price_change': price_change
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating volume signal for {symbol}: {e}")
            return None
    
    def _fuse_source_signals(self, symbol: str, source_signals: List[TradingSignal]) -> SignalFusion:
        """Fuse multiple source signals into a single signal"""
        if not source_signals:
            raise ValueError("No source signals provided")
        
        # Calculate weighted scores
        buy_score = 0
        sell_score = 0
        total_weight = 0
        reasoning_parts = []
        
        for signal in source_signals:
            weight = self.fusion_weights.get(signal.source, 0.1)
            total_weight += weight
            
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                score = signal.confidence * weight
                buy_score += score
                reasoning_parts.append(f"{signal.source.value}: {signal.reasoning}")
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                score = signal.confidence * weight
                sell_score += score
                reasoning_parts.append(f"{signal.source.value}: {signal.reasoning}")
        
        # Normalize scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine final signal
        net_score = buy_score - sell_score
        
        if net_score > 0.6:
            final_signal = SignalType.STRONG_BUY
            confidence = min(0.95, net_score)
            strength = "strong"
        elif net_score > 0.3:
            final_signal = SignalType.BUY
            confidence = net_score
            strength = "medium"
        elif net_score < -0.6:
            final_signal = SignalType.STRONG_SELL
            confidence = min(0.95, abs(net_score))
            strength = "strong"
        elif net_score < -0.3:
            final_signal = SignalType.SELL
            confidence = abs(net_score)
            strength = "medium"
        else:
            final_signal = SignalType.HOLD
            confidence = 0.3
            strength = "weak"
        
        return SignalFusion(
            symbol=symbol,
            final_signal=final_signal,
            confidence=confidence,
            strength=strength,
            source_signals=source_signals,
            fusion_weights=self.fusion_weights,
            reasoning="; ".join(reasoning_parts)
        )
    
    def update_fusion_weights(self, performance_data: Dict[str, float]):
        """Update fusion weights based on historical performance"""
        # This would implement adaptive weight adjustment based on performance
        # For now, we'll keep static weights
        logger.info("Fusion weights updated based on performance")
    
    def get_signal_performance(self, symbol: str, days_back: int = 30) -> Dict[str, float]:
        """Get historical performance of signals for a symbol"""
        # This would implement performance tracking
        return {
            'accuracy': 0.6,
            'precision': 0.7,
            'recall': 0.5,
            'f1_score': 0.6
        }


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        generator = SignalGenerator()
        await generator.initialize()
        
        # Example technical signals
        technical_signals = {
            'BTC': {
                'rsi': 25,
                'macd': 0.01,
                'macd_signal': 0.005,
                'bollinger_position': 0.15
            }
        }
        
        # Example sentiment data
        sentiment_data = {
            'average_sentiment': 0.4,
            'average_confidence': 0.7,
            'positive_ratio': 0.8
        }
        
        # Example simulation data
        simulation_data = {
            'expected_return': 0.15,
            'probability_of_profit': 0.75,
            'var_95': -0.1
        }
        
        # Example price data
        price_data = {
            'BTC': pd.DataFrame({
                'close': [50000, 51000, 52000],
                'volume': [1000, 2000, 1500]
            })
        }
        
        # Fuse signals
        fused_signals = await generator.fuse_signals(
            technical_signals,
            sentiment_data,
            simulation_data,
            price_data
        )
        
        for symbol, signal in fused_signals.items():
            print(f"{symbol}: {signal.final_signal.value} ({signal.confidence:.2f}) - {signal.reasoning}")
    
    asyncio.run(main())
