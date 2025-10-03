#!/usr/bin/env python3
"""
Unified Crypto Analysis Engine
Multi-symbol analysis engine that works for all crypto assets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
from loguru import logger

from feature_engineering.technical_indicators import TechnicalIndicators
from feature_engineering.alpha_factors import AlphaFactorGenerator
from feature_engineering.nlp_processor import NLPProcessor

class CryptoAnalysisEngine:
    """Unified analysis engine for all crypto assets"""
    
    def __init__(self):
        self.technical_indicators = None  # Will be initialized when needed
        self.alpha_factors = AlphaFactorGenerator()
        self.nlp_processor = NLPProcessor()
        
    def load_symbol_data(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Load historical data for any symbol"""
        try:
            symbol = symbol.upper()
            
            # Try to load from parquet files first
            file_path = f"data/crypto_db/{symbol}_historical.parquet"
            if Path(file_path).exists():
                df = pd.read_parquet(file_path)
                df.index = pd.to_datetime(df.index)
                
                # Filter to last N days
                cutoff_date = datetime.now() - timedelta(days=days)
                cutoff_date = pd.Timestamp(cutoff_date, tz='UTC')
                df = df[df.index >= cutoff_date]
                
                if df.empty:
                    raise ValueError(f"No data available for {symbol} in the last {days} days")
                
                logger.info(f"Loaded {len(df)} data points for {symbol} from {file_path}")
                return df
            
            # Fallback: try to load from other sources
            raise FileNotFoundError(f"No data file found for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            raise
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for any symbol"""
        try:
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Initialize technical indicators calculator if needed
            if self.technical_indicators is None:
                from feature_engineering.technical_indicators import IndicatorCalculator
                self.technical_indicators = IndicatorCalculator()
            
            # Calculate technical indicators
            indicators_data = self.technical_indicators.calculate_all_indicators(df)
            
            # Convert indicators to DataFrame
            indicators_dict = {}
            for field_name, field_value in indicators_data.__dict__.items():
                if not field_name.startswith('_'):
                    indicators_dict[field_name] = field_value
            
            indicators_df = pd.DataFrame([indicators_dict], index=df.index[-1:])
            
            # Calculate alpha factors
            alpha_factors = self.alpha_factors.generate_momentum_factors(df)
            alpha_dict = {}
            for name, factor in alpha_factors.items():
                alpha_dict[name] = factor.values.iloc[-1] if len(factor.values) > 0 else 0.0
            
            alpha_df = pd.DataFrame([alpha_dict], index=df.index[-1:])
            
            # Combine all indicators
            result_df = pd.concat([df, indicators_df, alpha_df], axis=1)
            
            # Remove any duplicate columns
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]
            
            logger.info(f"Calculated {len(indicators_dict)} technical indicators and {len(alpha_dict)} alpha factors")
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {e}")
            raise
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """Generate trading signals for any symbol"""
        try:
            if df.empty:
                raise ValueError("No data available for signal generation")
            
            # Get latest data point
            latest = df.iloc[-1]
            current_price = latest['close']
            current_volume = latest['volume']
            
            # Calculate signal strength based on technical indicators
            signal_strength = 0.0
            signal_reasons = []
            
            # RSI signals
            if 'rsi' in df.columns:
                rsi = latest['rsi']
                if rsi < 30:
                    signal_strength += 0.3
                    signal_reasons.append(f"RSI oversold ({rsi:.1f})")
                elif rsi > 70:
                    signal_strength -= 0.3
                    signal_reasons.append(f"RSI overbought ({rsi:.1f})")
            
            # MACD signals
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd = latest['macd']
                macd_signal = latest['macd_signal']
                if macd > macd_signal:
                    signal_strength += 0.2
                    signal_reasons.append("MACD bullish crossover")
                else:
                    signal_strength -= 0.2
                    signal_reasons.append("MACD bearish crossover")
            
            # Bollinger Bands
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                bb_upper = latest['bb_upper']
                bb_lower = latest['bb_lower']
                if current_price <= bb_lower:
                    signal_strength += 0.25
                    signal_reasons.append("Price at lower Bollinger Band")
                elif current_price >= bb_upper:
                    signal_strength -= 0.25
                    signal_reasons.append("Price at upper Bollinger Band")
            
            # Moving Average signals
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                sma_20 = latest['sma_20']
                sma_50 = latest['sma_50']
                if sma_20 > sma_50:
                    signal_strength += 0.15
                    signal_reasons.append("SMA 20 > SMA 50 (bullish trend)")
                else:
                    signal_strength -= 0.15
                    signal_reasons.append("SMA 20 < SMA 50 (bearish trend)")
            
            # Volume analysis
            if 'volume_sma' in df.columns:
                volume_ratio = current_volume / latest['volume_sma']
                if volume_ratio > 1.5:
                    signal_strength += 0.1
                    signal_reasons.append(f"High volume ({volume_ratio:.1f}x average)")
                elif volume_ratio < 0.5:
                    signal_strength -= 0.1
                    signal_reasons.append(f"Low volume ({volume_ratio:.1f}x average)")
            
            # Determine signal type
            if signal_strength > 0.3:
                signal_type = "BUY"
            elif signal_strength < -0.3:
                signal_type = "SELL"
            else:
                signal_type = "HOLD"
            
            # Calculate confidence
            confidence = min(abs(signal_strength), 1.0)
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'current_price': current_price,
                'current_volume': current_volume,
                'reasons': signal_reasons,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate signals for {symbol}: {e}")
            return {
                'symbol': symbol,
                'signal_type': "ERROR",
                'signal_strength': 0.0,
                'confidence': 0.0,
                'current_price': 0.0,
                'current_volume': 0.0,
                'reasons': [f"Error: {str(e)}"],
                'timestamp': datetime.now().isoformat()
            }
    
    def get_support_resistance(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        try:
            if df.empty:
                return {'support': 0.0, 'resistance': 0.0}
            
            # Use recent high/low for support/resistance
            recent_data = df.tail(20)  # Last 20 periods
            
            support = recent_data['low'].min()
            resistance = recent_data['high'].max()
            
            return {
                'support': float(support),
                'resistance': float(resistance)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate support/resistance for {symbol}: {e}")
            return {'support': 0.0, 'resistance': 0.0}
    
    def calculate_risk_metrics(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate risk metrics for any symbol"""
        try:
            if df.empty or len(df) < 2:
                return {'volatility': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0}
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(365 * 24 * 60)  # Assuming minute data
            
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Sharpe ratio (assuming risk-free rate = 0)
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0.0
            
            return {
                'volatility': float(volatility),
                'max_drawdown': float(max_drawdown),
                'sharpe_ratio': float(sharpe_ratio)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate risk metrics for {symbol}: {e}")
            return {'volatility': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0}
