"""
Advanced Technical Indicators for Financial Analysis (Windows Compatible)

Features:
- Comprehensive technical indicators using ta library (Windows compatible)
- Volume-based indicators
- Volatility measures
- Trend analysis
- Support/resistance levels
- Custom indicator combinations
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import ta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TechnicalIndicators:
    """Container for all technical indicators"""
    # Price-based indicators
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    ema_50: float
    
    # Momentum indicators
    rsi: float
    rsi_14: float
    rsi_21: float
    macd: float
    macd_signal: float
    macd_histogram: float
    stochastic_k: float
    stochastic_d: float
    williams_r: float
    cci: float
    roc: float
    
    # Volatility indicators
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    bollinger_width: float
    bollinger_position: float
    atr: float
    keltner_upper: float
    keltner_middle: float
    keltner_lower: float
    
    # Volume indicators
    volume_sma: float
    volume_ratio: float
    obv: float
    ad_line: float
    mfi: float
    vwap: float
    
    # Trend indicators
    adx: float
    adx_plus: float
    adx_minus: float
    parabolic_sar: float
    ichimoku_tenkan: float
    ichimoku_kijun: float
    ichimoku_senkou_a: float
    ichimoku_senkou_b: float
    ichimoku_chikou: float
    
    # Support/Resistance
    support_level: float
    resistance_level: float
    pivot_point: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float
    
    # Custom indicators
    price_momentum: float
    volume_momentum: float
    volatility_regime: str
    trend_strength: float
    market_regime: str


class IndicatorCalculator:
    """Calculate technical indicators from OHLCV data using ta library with caching"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.min_periods = 20  # Minimum periods for reliable indicators
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl_hours = 6  # Cache indicators for 6 hours
        
    def _get_cache_key(self, symbol: str, data_hash: str) -> str:
        """Generate cache key for indicators"""
        return f"{symbol}_indicators_{data_hash}.json"
    
    def _check_cached_indicators(self, symbol: str, data_hash: str) -> Optional[TechnicalIndicators]:
        """Check if indicators are cached and fresh"""
        try:
            cache_file = self.cache_dir / self._get_cache_key(symbol, data_hash)
            if not cache_file.exists():
                return None
            
            # Check file age
            import time
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > self.cache_ttl_hours * 3600:
                logger.debug(f"Cache expired for {symbol} indicators ({file_age/3600:.1f} hours old)")
                return None
            
            # Load cached indicators
            import json
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Convert back to TechnicalIndicators object
            return TechnicalIndicators(**cached_data)
            
        except Exception as e:
            logger.debug(f"Error checking cache for {symbol}: {e}")
            return None
    
    def _cache_indicators(self, symbol: str, data_hash: str, indicators: TechnicalIndicators):
        """Cache calculated indicators"""
        try:
            cache_file = self.cache_dir / self._get_cache_key(symbol, data_hash)
            import json
            with open(cache_file, 'w') as f:
                json.dump(indicators.__dict__, f, indent=2)
            logger.debug(f"Cached indicators for {symbol}")
        except Exception as e:
            logger.debug(f"Error caching indicators for {symbol}: {e}")
    
    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Generate hash for data to detect changes"""
        import hashlib
        # Use a more stable hash based on data characteristics rather than exact values
        if len(df) == 0:
            return "empty"
        
        # Use shape, date range, and basic statistics for stability
        shape_data = f"{df.shape}".encode()
        date_range = f"{df.index.min()}_{df.index.max()}".encode() if hasattr(df.index, 'min') else b""
        price_stats = f"{df['close'].mean():.2f}_{df['close'].std():.2f}".encode() if 'close' in df.columns else b""
        
        return hashlib.md5(shape_data + date_range + price_stats).hexdigest()[:8]
        
    def calculate_all_indicators(self, df: pd.DataFrame, symbol: str = None) -> TechnicalIndicators:
        """Calculate all technical indicators using ta library with caching"""
        if len(df) < self.min_periods:
            logger.warning(f"Insufficient data: {len(df)} periods (minimum: {self.min_periods})")
            return self._create_empty_indicators()
        
        # Check cache if symbol provided
        if symbol:
            data_hash = self._get_data_hash(df)
            logger.debug(f"Data hash for {symbol}: {data_hash}")
            cached_indicators = self._check_cached_indicators(symbol, data_hash)
            if cached_indicators is not None:
                logger.info(f"[CACHED] Using cached indicators for {symbol}")
                return cached_indicators
            else:
                logger.debug(f"No cached indicators found for {symbol} with hash {data_hash}")
        
        try:
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Need: {required_cols}")
            
            # Calculate indicators using ta library
            indicators = TechnicalIndicators(
                # Moving averages
                sma_20=self._safe_calculate(lambda: ta.trend.sma_indicator(df['close'], window=20).iloc[-1]),
                sma_50=self._safe_calculate(lambda: ta.trend.sma_indicator(df['close'], window=50).iloc[-1]),
                sma_200=self._safe_calculate(lambda: ta.trend.sma_indicator(df['close'], window=200).iloc[-1]),
                ema_12=self._safe_calculate(lambda: ta.trend.ema_indicator(df['close'], window=12).iloc[-1]),
                ema_26=self._safe_calculate(lambda: ta.trend.ema_indicator(df['close'], window=26).iloc[-1]),
                ema_50=self._safe_calculate(lambda: ta.trend.ema_indicator(df['close'], window=50).iloc[-1]),
                
                # Momentum indicators
                rsi=self._safe_calculate(lambda: ta.momentum.rsi(df['close'], window=14).iloc[-1]),
                rsi_14=self._safe_calculate(lambda: ta.momentum.rsi(df['close'], window=14).iloc[-1]),
                rsi_21=self._safe_calculate(lambda: ta.momentum.rsi(df['close'], window=21).iloc[-1]),
                macd=self._safe_calculate(lambda: ta.trend.macd(df['close']).iloc[-1]),
                macd_signal=self._safe_calculate(lambda: ta.trend.macd_signal(df['close']).iloc[-1]),
                macd_histogram=self._safe_calculate(lambda: ta.trend.macd_diff(df['close']).iloc[-1]),
                stochastic_k=self._safe_calculate(lambda: ta.momentum.stoch(df['high'], df['low'], df['close']).iloc[-1]),
                stochastic_d=self._safe_calculate(lambda: ta.momentum.stoch_signal(df['high'], df['low'], df['close']).iloc[-1]),
                williams_r=self._safe_calculate(lambda: ta.momentum.williams_r(df['high'], df['low'], df['close']).iloc[-1]),
                cci=self._safe_calculate(lambda: ta.trend.cci(df['high'], df['low'], df['close']).iloc[-1]),
                roc=self._safe_calculate(lambda: ta.momentum.roc(df['close'], window=10).iloc[-1]),
                
                # Bollinger Bands
                bollinger_upper=self._safe_calculate(lambda: ta.volatility.bollinger_hband(df['close']).iloc[-1]),
                bollinger_middle=self._safe_calculate(lambda: ta.volatility.bollinger_mavg(df['close']).iloc[-1]),
                bollinger_lower=self._safe_calculate(lambda: ta.volatility.bollinger_lband(df['close']).iloc[-1]),
                bollinger_width=self._calculate_bollinger_width(df),
                bollinger_position=self._calculate_bollinger_position(df),
                
                # Volatility
                atr=self._safe_calculate(lambda: ta.volatility.average_true_range(df['high'], df['low'], df['close']).iloc[-1]),
                keltner_upper=self._calculate_keltner_upper(df),
                keltner_middle=self._calculate_keltner_middle(df),
                keltner_lower=self._calculate_keltner_lower(df),
                
                # Volume indicators
                volume_sma=self._calculate_volume_sma(df),
                volume_ratio=self._calculate_volume_ratio(df),
                obv=self._safe_calculate(lambda: ta.volume.on_balance_volume(df['close'], df['volume']).iloc[-1]),
                ad_line=self._safe_calculate(lambda: ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume']).iloc[-1]),
                mfi=self._safe_calculate(lambda: ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume']).iloc[-1]),
                vwap=self._calculate_vwap(df),
                
                # Trend indicators
                adx=self._safe_calculate(lambda: ta.trend.adx(df['high'], df['low'], df['close']).iloc[-1]),
                adx_plus=self._safe_calculate(lambda: ta.trend.adx_pos(df['high'], df['low'], df['close']).iloc[-1]),
                adx_minus=self._safe_calculate(lambda: ta.trend.adx_neg(df['high'], df['low'], df['close']).iloc[-1]),
                parabolic_sar=self._safe_calculate(lambda: ta.trend.psar_up(df['high'], df['low'], df['close']).iloc[-1]),
                ichimoku_tenkan=self._calculate_ichimoku_tenkan(df),
                ichimoku_kijun=self._calculate_ichimoku_kijun(df),
                ichimoku_senkou_a=self._calculate_ichimoku_senkou_a(df),
                ichimoku_senkou_b=self._calculate_ichimoku_senkou_b(df),
                ichimoku_chikou=self._calculate_ichimoku_chikou(df),
                
                # Support/Resistance
                support_level=self._calculate_support_level(df),
                resistance_level=self._calculate_resistance_level(df),
                pivot_point=self._calculate_pivot_point(df),
                r1=self._calculate_r1(df),
                r2=self._calculate_r2(df),
                r3=self._calculate_r3(df),
                s1=self._calculate_s1(df),
                s2=self._calculate_s2(df),
                s3=self._calculate_s3(df),
                
                # Custom indicators
                price_momentum=self._calculate_price_momentum(df),
                volume_momentum=self._calculate_volume_momentum(df),
                volatility_regime=self._determine_volatility_regime(df),
                trend_strength=self._calculate_trend_strength(df),
                market_regime=self._determine_market_regime(df)
            )
            
            # Cache the calculated indicators if symbol provided
            if symbol:
                self._cache_indicators(symbol, data_hash, indicators)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return self._create_empty_indicators()
    
    def _safe_calculate(self, func, default=0.0):
        """Safely calculate indicator with error handling"""
        try:
            result = func()
            return float(result) if not np.isnan(result) and not np.isinf(result) else default
        except Exception as e:
            logger.debug(f"Error calculating indicator: {e}")
            return default
    
    def _calculate_bollinger_width(self, df: pd.DataFrame) -> float:
        """Calculate Bollinger Band width"""
        try:
            upper = ta.volatility.bollinger_hband(df['close'])
            middle = ta.volatility.bollinger_mavg(df['close'])
            lower = ta.volatility.bollinger_lband(df['close'])
            width = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1] if middle.iloc[-1] != 0 else 0
            return float(width) if not np.isnan(width) else 0.0
        except:
            return 0.0
    
    def _calculate_bollinger_position(self, df: pd.DataFrame) -> float:
        """Calculate position within Bollinger Bands (0-1)"""
        try:
            upper = ta.volatility.bollinger_hband(df['close'])
            lower = ta.volatility.bollinger_lband(df['close'])
            current_price = df['close'].iloc[-1]
            position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) if upper.iloc[-1] != lower.iloc[-1] else 0.5
            return float(position) if not np.isnan(position) else 0.5
        except:
            return 0.5
    
    def _calculate_keltner_upper(self, df: pd.DataFrame) -> float:
        """Calculate Keltner Channel upper band"""
        try:
            ema = ta.trend.ema_indicator(df['close'], window=20)
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            return float(ema.iloc[-1] + 2 * atr.iloc[-1]) if not np.isnan(ema.iloc[-1]) and not np.isnan(atr.iloc[-1]) else 0.0
        except:
            return 0.0
    
    def _calculate_keltner_middle(self, df: pd.DataFrame) -> float:
        """Calculate Keltner Channel middle band"""
        try:
            ema = ta.trend.ema_indicator(df['close'], window=20)
            return float(ema.iloc[-1]) if not np.isnan(ema.iloc[-1]) else 0.0
        except:
            return 0.0
    
    def _calculate_keltner_lower(self, df: pd.DataFrame) -> float:
        """Calculate Keltner Channel lower band"""
        try:
            ema = ta.trend.ema_indicator(df['close'], window=20)
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            return float(ema.iloc[-1] - 2 * atr.iloc[-1]) if not np.isnan(ema.iloc[-1]) and not np.isnan(atr.iloc[-1]) else 0.0
        except:
            return 0.0
    
    def _calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """Calculate volume ratio (current vs average)"""
        try:
            if len(df) < 20:
                return 1.0
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(20).mean()
            return float(current_volume / avg_volume) if avg_volume > 0 else 1.0
        except:
            return 1.0
    
    def _calculate_volume_sma(self, df: pd.DataFrame) -> float:
        """Calculate Volume Simple Moving Average"""
        try:
            if len(df) < 20:
                return float(df['volume'].mean()) if len(df) > 0 else 0.0
            return float(df['volume'].tail(20).mean())
        except:
            return 0.0
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        try:
            if 'vwap' in df.columns:
                return float(df['vwap'].iloc[-1])
            
            # Calculate VWAP if not provided
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            return float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else 0.0
        except:
            return 0.0
    
    def _calculate_ichimoku_tenkan(self, df: pd.DataFrame) -> float:
        """Calculate Ichimoku Tenkan-sen (9-period)"""
        try:
            if len(df) < 9:
                return 0.0
            highest_high = df['high'].tail(9).max()
            lowest_low = df['low'].tail(9).min()
            return float((highest_high + lowest_low) / 2)
        except:
            return 0.0
    
    def _calculate_ichimoku_kijun(self, df: pd.DataFrame) -> float:
        """Calculate Ichimoku Kijun-sen (26-period)"""
        try:
            if len(df) < 26:
                return 0.0
            highest_high = df['high'].tail(26).max()
            lowest_low = df['low'].tail(26).min()
            return float((highest_high + lowest_low) / 2)
        except:
            return 0.0
    
    def _calculate_ichimoku_senkou_a(self, df: pd.DataFrame) -> float:
        """Calculate Ichimoku Senkou Span A"""
        try:
            tenkan = self._calculate_ichimoku_tenkan(df)
            kijun = self._calculate_ichimoku_kijun(df)
            return float((tenkan + kijun) / 2)
        except:
            return 0.0
    
    def _calculate_ichimoku_senkou_b(self, df: pd.DataFrame) -> float:
        """Calculate Ichimoku Senkou Span B (52-period)"""
        try:
            if len(df) < 52:
                return 0.0
            highest_high = df['high'].tail(52).max()
            lowest_low = df['low'].tail(52).min()
            return float((highest_high + lowest_low) / 2)
        except:
            return 0.0
    
    def _calculate_ichimoku_chikou(self, df: pd.DataFrame) -> float:
        """Calculate Ichimoku Chikou Span (26 periods ago)"""
        try:
            if len(df) < 26:
                return 0.0
            return float(df['close'].iloc[-26])
        except:
            return 0.0
    
    def _calculate_support_level(self, df: pd.DataFrame) -> float:
        """Calculate support level (lowest low in recent period)"""
        try:
            recent_lows = df['low'].tail(20)
            return float(recent_lows.min())
        except:
            return 0.0
    
    def _calculate_resistance_level(self, df: pd.DataFrame) -> float:
        """Calculate resistance level (highest high in recent period)"""
        try:
            recent_highs = df['high'].tail(20)
            return float(recent_highs.max())
        except:
            return 0.0
    
    def _calculate_pivot_point(self, df: pd.DataFrame) -> float:
        """Calculate pivot point"""
        try:
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            close = df['close'].iloc[-1]
            return float((high + low + close) / 3)
        except:
            return 0.0
    
    def _calculate_r1(self, df: pd.DataFrame) -> float:
        """Calculate R1 resistance level"""
        try:
            pivot = self._calculate_pivot_point(df)
            low = df['low'].iloc[-1]
            return float(2 * pivot - low)
        except:
            return 0.0
    
    def _calculate_r2(self, df: pd.DataFrame) -> float:
        """Calculate R2 resistance level"""
        try:
            pivot = self._calculate_pivot_point(df)
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            return float(pivot + (high - low))
        except:
            return 0.0
    
    def _calculate_r3(self, df: pd.DataFrame) -> float:
        """Calculate R3 resistance level"""
        try:
            pivot = self._calculate_pivot_point(df)
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            return float(high + 2 * (pivot - low))
        except:
            return 0.0
    
    def _calculate_s1(self, df: pd.DataFrame) -> float:
        """Calculate S1 support level"""
        try:
            pivot = self._calculate_pivot_point(df)
            high = df['high'].iloc[-1]
            return float(2 * pivot - high)
        except:
            return 0.0
    
    def _calculate_s2(self, df: pd.DataFrame) -> float:
        """Calculate S2 support level"""
        try:
            pivot = self._calculate_pivot_point(df)
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            return float(pivot - (high - low))
        except:
            return 0.0
    
    def _calculate_s3(self, df: pd.DataFrame) -> float:
        """Calculate S3 support level"""
        try:
            pivot = self._calculate_pivot_point(df)
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            return float(low - 2 * (high - pivot))
        except:
            return 0.0
    
    def _calculate_price_momentum(self, df: pd.DataFrame) -> float:
        """Calculate price momentum (rate of change)"""
        try:
            if len(df) < 10:
                return 0.0
            return float((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10])
        except:
            return 0.0
    
    def _calculate_volume_momentum(self, df: pd.DataFrame) -> float:
        """Calculate volume momentum"""
        try:
            if len(df) < 10:
                return 0.0
            return float((df['volume'].iloc[-1] - df['volume'].iloc[-10]) / df['volume'].iloc[-10]) if df['volume'].iloc[-10] > 0 else 0.0
        except:
            return 0.0
    
    def _determine_volatility_regime(self, df: pd.DataFrame) -> str:
        """Determine volatility regime"""
        try:
            if len(df) < 20:
                return "unknown"
            
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            if volatility > 0.4:
                return "high"
            elif volatility > 0.2:
                return "medium"
            else:
                return "low"
        except:
            return "unknown"
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength (0-1)"""
        try:
            if len(df) < 20:
                return 0.5
            
            # Linear regression slope
            x = np.arange(len(df['close'].tail(20)))
            y = df['close'].tail(20).values
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize to 0-1 range
            normalized_slope = (slope / df['close'].iloc[-1]) * 100
            trend_strength = min(max(normalized_slope + 0.5, 0), 1)
            
            return float(trend_strength)
        except:
            return 0.5
    
    def _determine_market_regime(self, df: pd.DataFrame) -> str:
        """Determine overall market regime"""
        try:
            close = df['close'].values
            volume = df['volume'].values
            
            if len(close) < 20:
                return "unknown"
            
            # Price trend
            price_momentum = self._calculate_price_momentum(df)
            
            # Volume trend
            volume_momentum = self._calculate_volume_momentum(df)
            
            # Volatility
            volatility_regime = self._determine_volatility_regime(df)
            
            # Determine regime
            if price_momentum > 0.05 and volume_momentum > 0.1:
                return "bull_high_volume"
            elif price_momentum > 0.05:
                return "bull_low_volume"
            elif price_momentum < -0.05 and volume_momentum > 0.1:
                return "bear_high_volume"
            elif price_momentum < -0.05:
                return "bear_low_volume"
            elif volatility_regime == "high":
                return "sideways_volatile"
            else:
                return "sideways_quiet"
                
        except:
            return "unknown"
    
    def _create_empty_indicators(self) -> TechnicalIndicators:
        """Create empty indicators with default values"""
        return TechnicalIndicators(
            sma_20=0.0, sma_50=0.0, sma_200=0.0,
            ema_12=0.0, ema_26=0.0, ema_50=0.0,
            rsi=50.0, rsi_14=50.0, rsi_21=50.0,
            macd=0.0, macd_signal=0.0, macd_histogram=0.0,
            stochastic_k=50.0, stochastic_d=50.0, williams_r=-50.0,
            cci=0.0, roc=0.0,
            bollinger_upper=0.0, bollinger_middle=0.0, bollinger_lower=0.0,
            bollinger_width=0.0, bollinger_position=0.5,
            atr=0.0, keltner_upper=0.0, keltner_middle=0.0, keltner_lower=0.0,
            volume_sma=0.0, volume_ratio=1.0, obv=0.0, ad_line=0.0, mfi=50.0, vwap=0.0,
            adx=0.0, adx_plus=0.0, adx_minus=0.0, parabolic_sar=0.0,
            ichimoku_tenkan=0.0, ichimoku_kijun=0.0, ichimoku_senkou_a=0.0,
            ichimoku_senkou_b=0.0, ichimoku_chikou=0.0,
            support_level=0.0, resistance_level=0.0, pivot_point=0.0,
            r1=0.0, r2=0.0, r3=0.0, s1=0.0, s2=0.0, s3=0.0,
            price_momentum=0.0, volume_momentum=0.0, volatility_regime="unknown",
            trend_strength=0.5, market_regime="unknown"
        )


class TechnicalSignalGenerator:
    """Generate trading signals from technical indicators"""
    
    def __init__(self):
        self.calculator = IndicatorCalculator()
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive trading signals"""
        indicators = self.calculator.calculate_all_indicators(df)
        
        signals = {
            'momentum_signals': self._generate_momentum_signals(indicators),
            'trend_signals': self._generate_trend_signals(indicators),
            'volatility_signals': self._generate_volatility_signals(indicators),
            'volume_signals': self._generate_volume_signals(indicators),
            'support_resistance_signals': self._generate_support_resistance_signals(indicators, df),
            'overall_signal': self._generate_overall_signal(indicators)
        }
        
        return signals
    
    def _generate_momentum_signals(self, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """Generate momentum-based signals"""
        signals = {}
        
        # RSI signals
        if indicators.rsi < 30:
            signals['rsi'] = {'signal': 'oversold', 'strength': 'strong'}
        elif indicators.rsi > 70:
            signals['rsi'] = {'signal': 'overbought', 'strength': 'strong'}
        elif indicators.rsi < 40:
            signals['rsi'] = {'signal': 'oversold', 'strength': 'weak'}
        elif indicators.rsi > 60:
            signals['rsi'] = {'signal': 'overbought', 'strength': 'weak'}
        else:
            signals['rsi'] = {'signal': 'neutral', 'strength': 'none'}
        
        # MACD signals
        if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
            signals['macd'] = {'signal': 'bullish', 'strength': 'strong' if indicators.macd_histogram > 0.01 else 'weak'}
        elif indicators.macd < indicators.macd_signal and indicators.macd_histogram < 0:
            signals['macd'] = {'signal': 'bearish', 'strength': 'strong' if indicators.macd_histogram < -0.01 else 'weak'}
        else:
            signals['macd'] = {'signal': 'neutral', 'strength': 'none'}
        
        # Stochastic signals
        if indicators.stochastic_k < 20 and indicators.stochastic_d < 20:
            signals['stochastic'] = {'signal': 'oversold', 'strength': 'strong'}
        elif indicators.stochastic_k > 80 and indicators.stochastic_d > 80:
            signals['stochastic'] = {'signal': 'overbought', 'strength': 'strong'}
        else:
            signals['stochastic'] = {'signal': 'neutral', 'strength': 'none'}
        
        return signals
    
    def _generate_trend_signals(self, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """Generate trend-based signals"""
        signals = {}
        
        # Moving average signals (simplified - would need current price)
        if indicators.trend_strength > 0.6:
            signals['trend'] = {'signal': 'uptrend', 'strength': 'strong'}
        elif indicators.trend_strength < 0.4:
            signals['trend'] = {'signal': 'downtrend', 'strength': 'strong'}
        else:
            signals['trend'] = {'signal': 'sideways', 'strength': 'none'}
        
        # ADX trend strength
        if indicators.adx > 25:
            signals['adx'] = {'signal': 'strong_trend', 'strength': 'strong' if indicators.adx > 40 else 'medium'}
        elif indicators.adx > 20:
            signals['adx'] = {'signal': 'weak_trend', 'strength': 'weak'}
        else:
            signals['adx'] = {'signal': 'no_trend', 'strength': 'none'}
        
        return signals
    
    def _generate_volatility_signals(self, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """Generate volatility-based signals"""
        signals = {}
        
        # Bollinger Bands signals
        if indicators.bollinger_position < 0.1:
            signals['bollinger'] = {'signal': 'oversold', 'strength': 'strong'}
        elif indicators.bollinger_position > 0.9:
            signals['bollinger'] = {'signal': 'overbought', 'strength': 'strong'}
        elif indicators.bollinger_position < 0.2:
            signals['bollinger'] = {'signal': 'oversold', 'strength': 'weak'}
        elif indicators.bollinger_position > 0.8:
            signals['bollinger'] = {'signal': 'overbought', 'strength': 'weak'}
        else:
            signals['bollinger'] = {'signal': 'neutral', 'strength': 'none'}
        
        # Volatility regime
        signals['volatility_regime'] = {'signal': indicators.volatility_regime, 'strength': 'none'}
        
        return signals
    
    def _generate_volume_signals(self, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """Generate volume-based signals"""
        signals = {}
        
        # Volume ratio signals
        if indicators.volume_ratio > 2.0:
            signals['volume'] = {'signal': 'high_volume', 'strength': 'strong'}
        elif indicators.volume_ratio > 1.5:
            signals['volume'] = {'signal': 'high_volume', 'strength': 'weak'}
        elif indicators.volume_ratio < 0.5:
            signals['volume'] = {'signal': 'low_volume', 'strength': 'strong'}
        else:
            signals['volume'] = {'signal': 'normal_volume', 'strength': 'none'}
        
        # MFI signals
        if indicators.mfi < 20:
            signals['mfi'] = {'signal': 'oversold', 'strength': 'strong'}
        elif indicators.mfi > 80:
            signals['mfi'] = {'signal': 'overbought', 'strength': 'strong'}
        else:
            signals['mfi'] = {'signal': 'neutral', 'strength': 'none'}
        
        return signals
    
    def _generate_support_resistance_signals(self, indicators: TechnicalIndicators, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate support/resistance signals"""
        signals = {}
        current_price = df['close'].iloc[-1]
        
        # Support/resistance levels
        if current_price <= indicators.support_level * 1.02:  # Within 2% of support
            signals['support'] = {'signal': 'near_support', 'strength': 'strong'}
        elif current_price >= indicators.resistance_level * 0.98:  # Within 2% of resistance
            signals['resistance'] = {'signal': 'near_resistance', 'strength': 'strong'}
        else:
            signals['support'] = {'signal': 'neutral', 'strength': 'none'}
            signals['resistance'] = {'signal': 'neutral', 'strength': 'none'}
        
        return signals
    
    def _generate_overall_signal(self, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """Generate overall trading signal"""
        # This is a simplified version - in practice, you'd use more sophisticated logic
        score = 0
        
        # RSI contribution
        if indicators.rsi < 30:
            score += 2
        elif indicators.rsi > 70:
            score -= 2
        elif indicators.rsi < 40:
            score += 1
        elif indicators.rsi > 60:
            score -= 1
        
        # MACD contribution
        if indicators.macd > indicators.macd_signal:
            score += 1
        else:
            score -= 1
        
        # Trend contribution
        if indicators.trend_strength > 0.6:
            score += 1
        elif indicators.trend_strength < 0.4:
            score -= 1
        
        # Determine signal
        if score >= 3:
            return {'signal': 'strong_buy', 'confidence': min(score / 5, 1.0)}
        elif score >= 1:
            return {'signal': 'buy', 'confidence': score / 5}
        elif score <= -3:
            return {'signal': 'strong_sell', 'confidence': min(abs(score) / 5, 1.0)}
        elif score <= -1:
            return {'signal': 'sell', 'confidence': abs(score) / 5}
        else:
            return {'signal': 'hold', 'confidence': 0.5}


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Calculate indicators
    calculator = IndicatorCalculator()
    indicators = calculator.calculate_all_indicators(df)
    
    print(f"RSI: {indicators.rsi:.2f}")
    print(f"MACD: {indicators.macd:.4f}")
    print(f"Bollinger Position: {indicators.bollinger_position:.2f}")
    print(f"Market Regime: {indicators.market_regime}")
    
    # Generate signals
    signal_generator = TechnicalSignalGenerator()
    signals = signal_generator.generate_signals(df)
    
    print(f"\nOverall Signal: {signals['overall_signal']}")

