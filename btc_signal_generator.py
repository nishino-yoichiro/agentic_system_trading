"""
Minimal BTC Signal Generator
Generates actual trading signals for BTC using available data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger


class BTCSignalGenerator:
    """Minimal signal generator focused on BTC"""
    
    def __init__(self):
        self.symbol = "BTC"
    
    def generate_btc_signals(self, price_df: pd.DataFrame, indicators_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate BTC trading signals from price and indicator data"""
        
        if price_df.empty:
            return self._create_hold_signal("No price data available")
        
        try:
            # Basic price analysis
            current_price = price_df['close'].iloc[-1]
            price_24h_ago = price_df['close'].iloc[-1440] if len(price_df) > 1440 else price_df['close'].iloc[0]
            price_change_24h = (current_price - price_24h_ago) / price_24h_ago * 100
            
            # Volume analysis
            current_volume = price_df['volume'].iloc[-1]
            avg_volume_7d = price_df['volume'].tail(10080).mean() if len(price_df) > 10080 else price_df['volume'].mean()
            volume_ratio = current_volume / avg_volume_7d if avg_volume_7d > 0 else 1
            
            # Technical indicators (if available)
            rsi = 50  # Default neutral
            macd = 0
            macd_signal = 0
            
            if indicators_df is not None and not indicators_df.empty:
                if 'rsi' in indicators_df.columns:
                    rsi = indicators_df['rsi'].iloc[-1]
                if 'macd' in indicators_df.columns:
                    macd = indicators_df['macd'].iloc[-1]
                if 'macd_signal' in indicators_df.columns:
                    macd_signal = indicators_df['macd_signal'].iloc[-1]
            
            # Calculate signal score
            signal_score = 0
            reasoning_parts = []
            
            # Price momentum (40% weight)
            if price_change_24h > 5:
                signal_score += 3
                reasoning_parts.append(f"Strong 24h gain: +{price_change_24h:.1f}%")
            elif price_change_24h > 2:
                signal_score += 1
                reasoning_parts.append(f"Moderate 24h gain: +{price_change_24h:.1f}%")
            elif price_change_24h < -5:
                signal_score -= 3
                reasoning_parts.append(f"Significant 24h decline: {price_change_24h:.1f}%")
            elif price_change_24h < -2:
                signal_score -= 1
                reasoning_parts.append(f"Moderate 24h decline: {price_change_24h:.1f}%")
            
            # RSI signals (30% weight)
            if rsi < 30:
                signal_score += 2
                reasoning_parts.append(f"RSI oversold: {rsi:.1f}")
            elif rsi < 40:
                signal_score += 1
                reasoning_parts.append(f"RSI approaching oversold: {rsi:.1f}")
            elif rsi > 70:
                signal_score -= 2
                reasoning_parts.append(f"RSI overbought: {rsi:.1f}")
            elif rsi > 60:
                signal_score -= 1
                reasoning_parts.append(f"RSI approaching overbought: {rsi:.1f}")
            
            # MACD signals (20% weight)
            if macd > macd_signal:
                signal_score += 1
                reasoning_parts.append("MACD bullish crossover")
            elif macd < macd_signal:
                signal_score -= 1
                reasoning_parts.append("MACD bearish crossover")
            
            # Volume confirmation (10% weight)
            if volume_ratio > 2.0:
                signal_score += 1
                reasoning_parts.append(f"High volume: {volume_ratio:.1f}x average")
            elif volume_ratio < 0.5:
                signal_score -= 1
                reasoning_parts.append(f"Low volume: {volume_ratio:.1f}x average")
            
            # Convert score to signal
            if signal_score >= 4:
                action = "STRONG_BUY"
                confidence = min(0.95, 0.7 + signal_score * 0.05)
                strength = "strong"
            elif signal_score >= 2:
                action = "BUY"
                confidence = min(0.9, 0.5 + signal_score * 0.1)
                strength = "medium"
            elif signal_score <= -4:
                action = "STRONG_SELL"
                confidence = min(0.95, 0.7 + abs(signal_score) * 0.05)
                strength = "strong"
            elif signal_score <= -2:
                action = "SELL"
                confidence = min(0.9, 0.5 + abs(signal_score) * 0.1)
                strength = "medium"
            else:
                action = "HOLD"
                confidence = 0.4
                strength = "weak"
                reasoning_parts.append("Mixed signals - no clear direction")
            
            # Calculate price targets
            if action in ["BUY", "STRONG_BUY"]:
                price_target = current_price * (1 + abs(signal_score) * 0.02)  # 2% per signal point
                stop_loss = current_price * (1 - 0.05)  # 5% stop loss
            elif action in ["SELL", "STRONG_SELL"]:
                price_target = current_price * (1 - abs(signal_score) * 0.02)
                stop_loss = current_price * (1 + 0.05)
            else:
                price_target = current_price
                stop_loss = current_price * 0.95
            
            return {
                'symbol': self.symbol,
                'action': action,
                'confidence': confidence,
                'strength': strength,
                'current_price': current_price,
                'price_change_24h': price_change_24h,
                'price_target': price_target,
                'stop_loss': stop_loss,
                'reasoning': "; ".join(reasoning_parts),
                'signals': {
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'volume_ratio': volume_ratio,
                    'signal_score': signal_score
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating BTC signals: {e}")
            return self._create_hold_signal(f"Error: {str(e)}")
    
    def _create_hold_signal(self, reason: str) -> Dict[str, Any]:
        """Create a default HOLD signal"""
        return {
            'symbol': self.symbol,
            'action': 'HOLD',
            'confidence': 0.3,
            'strength': 'weak',
            'current_price': 0,
            'price_change_24h': 0,
            'price_target': 0,
            'stop_loss': 0,
            'reasoning': reason,
            'signals': {},
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test the signal generator
    generator = BTCSignalGenerator()
    
    # Create test data
    test_data = pd.DataFrame({
        'close': [50000, 51000, 52000, 53000, 54000],
        'volume': [1000, 2000, 1500, 3000, 2500]
    }, index=pd.date_range('2024-01-01', periods=5, freq='1H'))
    
    # Test indicators
    test_indicators = pd.DataFrame({
        'rsi': [45, 50, 55, 60, 65],
        'macd': [0.01, 0.02, 0.03, 0.04, 0.05],
        'macd_signal': [0.005, 0.015, 0.025, 0.035, 0.045]
    }, index=pd.date_range('2024-01-01', periods=5, freq='1H'))
    
    signals = generator.generate_btc_signals(test_data, test_indicators)
    print("BTC Signal Test:")
    print(f"Action: {signals['action']}")
    print(f"Confidence: {signals['confidence']:.2f}")
    print(f"Reasoning: {signals['reasoning']}")
